"""Build one batch of frames of an ATMO v5 run.

Port of AtmoStepWorker + atmo_step.py (panahon-nwp v1.2). One flow run handles a batch
(one forecast day by default) instead of one queue job per frame.

For every 15-minute frame in --step-from..--step-to:
  1. download that frame's chunk objects (about 30 MB)
  2. build the 10 bands and a UGRD/VGRD GRIB2
  3. gdal_translate the bands into a cloud-optimised TIFF   -> <run_dir>/out/tiff/
  4. grib2json the wind GRIB2 into the wind-particle .bin   -> <run_dir>/out/bin/

A frame whose TIFF (and .bin) already exists is skipped, so a retried batch only redoes
what is missing. Outputs are written under a temporary name and renamed.

The wind .bin needs grib2json (https://github.com/weacast/weacast-grib2json, the tool the
legacy app uses), found at /opt/flow/grib2json/bin/grib2json or wherever GRIB2JSON_BIN
points; Java 8 comes with the PyNWP env. On a machine without it the batch still
succeeds and reports wind_bin = 0, and the flow skips the .bin delivery.

Environment: ATMO_S3_* (the node's ATMO_S3 connection, see atmo_common); optional
overrides GDAL_TRANSLATE_BIN, GRIB2JSON_BIN, PROJ_LIB.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import atmo_common as ac

# Where deploy/envs/pynwp.yml installs the weacast-grib2json checkout.
GRIB2JSON_DEFAULT = "/opt/flow/grib2json/bin/grib2json"

REQUIRED = ["mean_sea_level_pressure", "2m_temperature", "2m_dewpoint_temperature",
            "10m_u_component_of_wind", "10m_v_component_of_wind", "total_cloud_cover", "reflectivity"]


def relative_humidity(tmp_k, dpt_k):
    """Magnus formula, same as the legacy ecmwfRH.py."""
    import numpy as np

    t = tmp_k - 273.15
    d = dpt_k - 273.15
    rh = 100.0 * np.exp((17.625 * d) / (243.04 + d)) / np.exp((17.625 * t) / (243.04 + t))
    return np.clip(rh, 0.0, 100.0)


def rain_total(run_directory, step, meta):
    """Rain since the run started (mm), computed for every step by rain_total.py."""
    import numpy as np

    path = ac.total_path(run_directory, step)
    if not os.path.exists(path):
        raise RuntimeError("running rain total for step %d is missing (%s)" % (step, path))
    total = np.load(path).astype(np.float32)
    if total.shape != (meta["ny"], meta["nx"]):
        raise RuntimeError("running rain total for step %d has the wrong shape" % step)
    return total


def write_raster(out_prefix, bands, meta):
    """10 Float32 bands as a raw file plus a VRT that gdal_translate turns into a COG."""
    import numpy as np

    ny, nx = meta["ny"], meta["nx"]
    bil = out_prefix + ".bil"
    with open(bil, "wb") as f:
        for arr in bands:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    north = max(meta["lat_first"], meta["lat_last"])
    west = meta["lon_first"]
    dx, dy = meta["dlon"], meta["dlat"]
    geo = f"{west - dx / 2:.10f}, {dx:.12f}, 0, {north + dy / 2:.10f}, 0, {-dy:.12f}"
    plane = nx * ny * 4
    xml = [f'<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">',
           "  <SRS>EPSG:4326</SRS>",
           f"  <GeoTransform>{geo}</GeoTransform>"]
    for k, (name, desc, unit) in enumerate(ac.BANDS):
        xml += [
            f'  <VRTRasterBand dataType="Float32" band="{k + 1}" subClass="VRTRawRasterBand">',
            f"    <Description>{desc}</Description>",
            "    <NoDataValue>nan</NoDataValue>",
            f"    <UnitType>{unit}</UnitType>",
            f'    <Metadata><MDI key="parameter">{name}</MDI></Metadata>',
            f'    <SourceFilename relativeToVRT="1">{os.path.basename(bil)}</SourceFilename>',
            f"    <ImageOffset>{k * plane}</ImageOffset>",
            "    <PixelOffset>4</PixelOffset>",
            f"    <LineOffset>{nx * 4}</LineOffset>",
            "    <ByteOrder>LSB</ByteOrder>",
            "  </VRTRasterBand>",
        ]
    xml.append("</VRTDataset>")
    with open(out_prefix + ".vrt", "w") as f:
        f.write("\n".join(xml) + "\n")


def write_wind_grib(path, ugrd, vgrd, step, meta):
    """Two-message GRIB2 (UGRD, VGRD) on the run's own grid, built from the ATMO templates."""
    import numpy as np
    from eccodes import (codes_grib_new_from_file, codes_release, codes_set,
                         codes_set_values, codes_write)

    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ATMO_template")
    init = datetime.fromtimestamp(meta["t0_epoch"], tz=timezone.utc)
    minutes = step * meta["step_seconds"] // 60
    north = max(meta["lat_first"], meta["lat_last"])
    south = min(meta["lat_first"], meta["lat_last"])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "wb") as fout:
        for template, values in (("atmo.ugrd.template.gr2", ugrd), ("atmo.vgrd.template.gr2", vgrd)):
            with open(os.path.join(template_dir, template), "rb") as fin:
                gid = codes_grib_new_from_file(fin)
            if gid is None:
                raise RuntimeError("empty GRIB template " + template)
            codes_set(gid, "Ni", meta["nx"])
            codes_set(gid, "Nj", meta["ny"])
            codes_set(gid, "latitudeOfFirstGridPointInDegrees", north)
            codes_set(gid, "longitudeOfFirstGridPointInDegrees", meta["lon_first"])
            codes_set(gid, "latitudeOfLastGridPointInDegrees", south)
            codes_set(gid, "longitudeOfLastGridPointInDegrees", meta["lon_last"])
            codes_set(gid, "iDirectionIncrementInDegrees", meta["dlon"])
            codes_set(gid, "jDirectionIncrementInDegrees", meta["dlat"])
            codes_set(gid, "dataDate", int(init.strftime("%Y%m%d")))
            codes_set(gid, "dataTime", int(init.strftime("%H%M")))
            codes_set(gid, "indicatorOfUnitOfTimeRange", 0)  # minutes
            codes_set(gid, "forecastTime", int(minutes))
            flat = np.asarray(values, dtype=np.float64).ravel()
            if np.isnan(flat).any():
                codes_set(gid, "bitmapPresent", 1)
                codes_set(gid, "missingValue", 9999.0)
                flat = np.where(np.isnan(flat), 9999.0, flat)
            codes_set_values(gid, flat)
            codes_write(gid, fout)
            codes_release(gid)
    os.replace(path + ".tmp", path)


def build_bands(run_directory, step, step_dir, meta):
    """Returns (bands in TIFF order, ugrd, vgrd), all north-up float32."""
    import numpy as np

    absent = [name for name in REQUIRED if not os.path.exists(os.path.join(step_dir, name))]
    if absent:
        raise RuntimeError("missing input chunks for step %d: %s" % (step, ", ".join(absent)))

    def surface(name):
        return ac.field_2d(os.path.join(step_dir, name), meta)

    rain_1h = surface("precipitation")            # mm in the past hour; absent for the first hour
    tmp = surface("2m_temperature")
    dpt = surface("2m_dewpoint_temperature")
    ugrd = surface("10m_u_component_of_wind")
    vgrd = surface("10m_v_component_of_wind")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns stay NaN
        refc = np.nanmax(ac.field_3d(os.path.join(step_dir, "reflectivity"), meta), axis=0)

    values = {
        "PRATE": rain_1h / 3600.0,
        "PRES": surface("mean_sea_level_pressure"),
        "RH": relative_humidity(tmp, dpt),
        "TMP": tmp,
        "UGRD": ugrd,
        "VGRD": vgrd,
        "TCDC": surface("total_cloud_cover"),
        "REFC": refc,
        "DPT": dpt,
        "TPRATE": rain_total(run_directory, step, meta),
    }
    bands = [ac.north_up(values[name], meta).astype(np.float32) for name, _, _ in ac.BANDS]
    return bands, ac.north_up(ugrd, meta), ac.north_up(vgrd, meta)


def child_env():
    env = dict(os.environ)
    # conda's PROJ data sits beside the interpreter; the legacy app set this explicitly.
    env.setdefault("PROJ_LIB", os.path.join(sys.prefix, "share", "proj"))
    # The grib2json launcher runs $JAVA_HOME/bin/java. The PyNWP env carries its own
    # Java 8 (bin/java under the env prefix), so a worker needs no system Java.
    if not env.get("JAVA_HOME") and os.path.exists(os.path.join(sys.prefix, "bin", "java")):
        env["JAVA_HOME"] = sys.prefix
    return env


def to_cog(vrt, tiff_out):
    """Cloud-optimised TIFF, written under a temporary name first. Same options as legacy."""
    partial = tiff_out + ".part"
    if os.path.exists(partial):
        os.remove(partial)
    cmd = [ac.tool("gdal_translate", "GDAL_TRANSLATE_BIN"), "-q", "-of", "COG", "-ot", "Float32",
           "-co", "COMPRESS=ZSTD", "-co", "LEVEL=9", "-co", "PREDICTOR=3", "-co", "BLOCKSIZE=256",
           "-co", "NUM_THREADS=ALL_CPUS", "-co", "BIGTIFF=IF_SAFER", vrt, partial]
    done = subprocess.run(cmd, env=child_env(), capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(partial):
        raise RuntimeError("gdal_translate failed: " + done.stderr.strip()[-500:])
    os.replace(partial, tiff_out)


def to_wind_bin(grib2json, wind_grib, bin_out):
    """The wind-particle .bin: grib2json's UGRD+VGRD records as gzipped JSON, which is what
    the legacy ParameterWorker stores (gzencode(json_encode($json), 9))."""
    cmd = [grib2json, "-d", "-n", "--fp", "wind", "--fc", "2", "--fs", "103", wind_grib]
    done = subprocess.run(cmd, env=child_env(), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("grib2json failed: " + done.stderr.strip()[-500:])
    try:
        records = json.loads(done.stdout)
    except ValueError:
        records = None
    if not records:
        raise RuntimeError("grib2json found no wind records in " + wind_grib)
    partial = bin_out + ".part"
    with gzip.open(partial, "wb", compresslevel=9) as f:
        f.write(json.dumps(records, separators=(",", ":")).encode())
    os.replace(partial, bin_out)


def die_with_parent():
    """Pool worker initializer: take SIGTERM the moment the batch process dies.

    When a batch outlives its node timeout the engine kills the batch process, and only
    that one. Its pool workers used to live on as orphans, downloading and writing the
    same frames as the retry that follows. Linux can tie a child's life to its parent's.
    """
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        if os.getppid() == 1:           # the parent died before the call took effect
            os._exit(1)
    except Exception:                   # not Linux: nothing to tie
        pass


def process_step(job):
    """One frame. Runs in a worker process; returns the frame record."""
    run_directory, zarr, model, step, grib2json = (
        job["run_dir"], job["zarr"], job["model"], job["step"], job["grib2json"])
    meta = ac.load_meta(run_directory)
    init = ac.init_time(job["run_id"])
    tiff_out = os.path.join(run_directory, "out", "tiff", ac.tiff_name(model, init, step))
    bin_out = os.path.join(run_directory, "out", "bin", ac.bin_name(model, init, step))
    record = {
        "step": step,
        "minutes": step * ac.STEP_MINUTES,
        "forecast_time": ac.manila_naive(init + timedelta(minutes=step * ac.STEP_MINUTES)),
        "tiff": tiff_out,
        "bin": bin_out if grib2json else None,
        "header_parameters": "|".join(desc for _, desc, _ in ac.BANDS),
        "skipped": False,
    }
    if os.path.exists(tiff_out) and (not grib2json or os.path.exists(bin_out)):
        record["skipped"] = True
        return record

    step_dir = os.path.join(run_directory, "step", ac.step_name(step))
    wind_grib = os.path.join(step_dir, "wind.uv.gr2")
    os.makedirs(step_dir, exist_ok=True)
    try:
        # 1. Only this frame's objects are downloaded.
        copies = {zarr + "%s/c/%d/%s" % (variable, step, key): os.path.join(step_dir, variable)
                  for variable, key in ac.FRAME_VARIABLES.items()}
        ac.s5cmd_fetch(copies)
        # precipitation was fetched for the running total; absent there means absent on S3.
        chunk = ac.chunk_path(run_directory, step)
        if os.path.exists(chunk):
            shutil.copyfile(chunk, os.path.join(step_dir, "precipitation"))

        # 2. Bands and the wind GRIB2.
        bands, ugrd, vgrd = build_bands(run_directory, step, step_dir, meta)
        write_raster(os.path.join(step_dir, "frame"), bands, meta)

        # 3. Cloud-optimised TIFF.
        os.makedirs(os.path.dirname(tiff_out), exist_ok=True)
        to_cog(os.path.join(step_dir, "frame.vrt"), tiff_out)

        # 4. Wind-particle .bin.
        if grib2json:
            write_wind_grib(wind_grib, ugrd, vgrd, step, meta)
            os.makedirs(os.path.dirname(bin_out), exist_ok=True)
            to_wind_bin(grib2json, wind_grib, bin_out)
    except SystemExit as exc:      # atmo_common.fail() inside a worker process
        raise RuntimeError("step %d: exited with %s" % (step, exc.code))
    except Exception as exc:
        raise RuntimeError("step %d: %s" % (step, exc))
    finally:
        shutil.rmtree(step_dir, ignore_errors=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="atmo")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--zarr", required=True)
    p.add_argument("--forecast-id", type=int, required=True)
    p.add_argument("--batch-no", type=int, required=True)
    p.add_argument("--step-from", type=int, required=True)
    p.add_argument("--step-to", type=int, required=True)
    p.add_argument("--workers", type=int, default=4, help="frames built at the same time")
    args = p.parse_args()

    if args.step_to < args.step_from:
        ac.fail("frames: --step-to is before --step-from", 2)
    run_directory = ac.run_dir(args.work_dir, args.run_id)
    if not os.path.exists(os.path.join(run_directory, "meta.json")):
        ac.fail("frames: %s has no meta.json - the atmo-run flow has not prepared this run" % run_directory)
    init = ac.init_time(args.run_id)
    grib2json = ac.env("GRIB2JSON_BIN") or (GRIB2JSON_DEFAULT if os.path.exists(GRIB2JSON_DEFAULT) else None)
    if not grib2json:
        ac.log("frames: no grib2json at %s (or GRIB2JSON_BIN) - wind .bin files are not produced"
               % GRIB2JSON_DEFAULT)

    steps = list(range(args.step_from, args.step_to + 1))
    jobs = [{"run_dir": run_directory, "zarr": args.zarr, "model": args.model, "run_id": args.run_id,
             "step": step, "grib2json": grib2json} for step in steps]

    records, errors = [], []
    with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=die_with_parent) as pool:
        futures = [pool.submit(process_step, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
            ac.progress(100 * done / len(jobs))
    if errors:
        # Finished frames stay on disk; the retry skips them.
        ac.fail("frames: %d of %d frames failed, first: %s" % (len(errors), len(jobs), errors[0]))

    records.sort(key=lambda r: r["step"])
    frames_json = os.path.join(run_directory, "batch", "%02d.frames.json" % args.batch_no)
    ac.write_json_atomic(frames_json, {
        "model": args.model, "run_id": args.run_id, "forecast_id": args.forecast_id,
        "batch_no": args.batch_no,
        "tiff_folder": ac.tiff_folder(args.model, init),
        "bin_folder": ac.bin_folder(args.model, init),
        "frames": records,
    })
    ac.emit({
        "model": args.model,
        "run_id": args.run_id,
        "forecast_id": args.forecast_id,
        "batch_no": args.batch_no,
        "frame_count": len(records),
        "skipped_count": sum(1 for r in records if r["skipped"]),
        "wind_bin": 1 if grib2json else 0,
        "tiff_folder": ac.tiff_folder(args.model, init),
        "bin_folder": ac.bin_folder(args.model, init),
        "tiffs": [r["tiff"] for r in records],
        "bins": [r["bin"] for r in records if r["bin"]],
        "frames_json": frames_json,
    })


if __name__ == "__main__":
    main()
