"""Point data cache of an ATMO run, in the Redis layout pagasa-api reads.

Port of AtmoPointCacheWorker + atmo_point_cache.py (panahon-nwp v1.2).

usage:
  point_cache.py build  --work-dir D --run-id R --forecast-id N --steps 769 [--model atmo] [--tiff-dir T]
  point_cache.py delete --forecast-id N
  point_cache.py exists --forecast-id N        (exit 0 when the run is fully cached)

Layout (the ECMWF cache is the original)
  key    "{forecast_id}.{n}"   n = y * nx + x, y counted from the SOUTH edge, x from the west
  value  uint16 P, uint16 H (little-endian), then P*H float32, parameter-major
  key    "{forecast_id}.meta"  JSON describing grid, parameters and time steps; written last,
                               so its presence means the build finished

The build reads the run's finished TIFFs from the staging run folder. Each one is turned
into a raw Float32 file with gdal_translate (one temporary file at a time), so no GDAL
Python bindings are needed. A rebuild is safe: it removes the meta key first.

Memory: the whole cube (steps x parameters x points, float32) is held at once, exactly
like the legacy build. 265 steps x 10 parameters on a 576 x 960 grid is about 5.9 GB.

Environment: REDIS_HOST, REDIS_PORT, REDIS_DATABASE, REDIS_USERNAME, REDIS_PASSWORD (the
node's REDIS connection); optional overrides GDAL_TRANSLATE_BIN, PROJ_LIB.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

import atmo_common as ac

BATCH = 2000


def connect():
    import redis

    return redis.Redis(
        host=ac.env("REDIS_HOST", "127.0.0.1"),
        port=int(ac.env("REDIS_PORT", "6379")),
        db=int(ac.env("REDIS_DATABASE") or ac.env("REDIS_DB", "0")),
        username=ac.env("REDIS_USERNAME"),
        password=ac.env("REDIS_PASSWORD"),
    )


def read_header(hdr_path):
    """Grid of an ENVI raster: size, and the centre of its north-west cell."""
    with open(hdr_path) as f:
        text = f.read()

    def value(name):
        return text.split(name, 1)[1].split("=", 1)[1].splitlines()[0].strip()

    nx, ny = int(value("samples")), int(value("lines"))
    # map info = {projection, tie x, tie y, west edge, north edge, dx, dy, ...}
    info = [part.strip() for part in value("map info").strip("{}").split(",")]
    west_edge, north_edge, dx, dy = (float(v) for v in info[3:7])
    north = north_edge - dy / 2
    return {"nx": nx, "ny": ny, "dlon": dx, "dlat": dy,
            "west": west_edge + dx / 2, "north": north, "south": north - (ny - 1) * dy}


def read_tiff(path, raw_path):
    """All bands of one TIFF as float32 (bands, ny, nx), rows north to south as stored,
    plus the grid it is on."""
    import numpy as np

    env = dict(os.environ)
    env.setdefault("PROJ_LIB", os.path.join(sys.prefix, "share", "proj"))
    cmd = [ac.tool("gdal_translate", "GDAL_TRANSLATE_BIN"), "-q", "-of", "ENVI", "-ot", "Float32",
           "-co", "INTERLEAVE=BSQ", path, raw_path]
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(raw_path):
        ac.fail("point_cache: gdal_translate failed for %s: %s" % (path, done.stderr.strip()[-500:]))
    stem = os.path.splitext(raw_path)[0]
    grid = read_header(stem + ".hdr")
    nx, ny = grid["nx"], grid["ny"]
    data = np.fromfile(raw_path, dtype="<f4")
    for leftover in (raw_path, raw_path + ".aux.xml", stem + ".hdr"):
        if os.path.exists(leftover):
            os.remove(leftover)
    if data.size % (nx * ny) != 0:
        ac.fail("point_cache: %s is not a %dx%d raster" % (path, nx, ny))
    return data.reshape(-1, ny, nx), grid


def build(args):
    import numpy as np

    init = ac.init_time(args.run_id)
    run_directory = ac.run_dir(args.work_dir, args.run_id)
    minutes = ac.cache_minutes(args.steps, args.quarter_hour_span_hours)
    frames = []
    # Staging by default; --tiff-dir points a rebuild at the delivered copies instead
    # (the staging run folder is gone once finish_run has closed the run).
    tiff_dir = args.tiff_dir or os.path.join(run_directory, "out", "tiff")
    for minute in minutes:
        path = os.path.join(tiff_dir, ac.tiff_name(args.model, init, minute // ac.STEP_MINUTES))
        if not os.path.exists(path):
            ac.fail("point_cache: forecast %d has no TIFF for minute %d (%s)" % (args.forecast_id, minute, path))
        frames.append({"minutes": minute, "path": path})

    names = [name for name, _ in ac.CACHE_PARAMETERS]
    bands = [band for _, band in ac.CACHE_PARAMETERS]
    n_params, n_steps = len(bands), len(frames)
    if n_params > 65535 or n_steps > 65535:
        ac.fail("point_cache: too many parameters or steps for the uint16 header")

    started = time.time()
    cube = grid = None
    tmp_dir = os.path.join(run_directory, "cache")
    os.makedirs(tmp_dir, exist_ok=True)
    raw_path = os.path.join(tmp_dir, "frame.bil")
    for h, frame in enumerate(frames):
        tiff, frame_grid = read_tiff(frame["path"], raw_path)
        if grid is None:
            grid = frame_grid
            nx, ny = grid["nx"], grid["ny"]
            n_points = nx * ny
            cube = np.empty((n_steps, n_params, n_points), dtype=np.float32)
        elif (frame_grid["nx"], frame_grid["ny"]) != (nx, ny):
            ac.fail("point_cache: %s is on a different grid" % frame["path"])
        if tiff.shape[0] < max(bands):
            ac.fail("point_cache: %s has %d bands, need %d" % (frame["path"], tiff.shape[0], max(bands)))
        # TIFF rows run north to south; the cache counts rows from the south.
        picked = tiff[[b - 1 for b in bands], ::-1, :].reshape(n_params, n_points)
        # JSON cannot carry NaN, so "no data" (rain in the first hour) is stored as 0.
        cube[h] = np.nan_to_num(picked, nan=0.0, posinf=0.0, neginf=0.0)
        if h % 10 == 0:
            ac.progress(50 * h / n_steps)
    read_seconds = time.time() - started

    r = connect()
    r.delete("%d.meta" % args.forecast_id)        # a rebuild is incomplete until meta is rewritten
    header = struct.pack("<HH", n_params, n_steps)
    stride = n_params * n_steps * 4
    pipe = r.pipeline(transaction=False)
    for start in range(0, n_points, BATCH):
        stop = min(start + BATCH, n_points)
        # [steps, params, points] -> [points, params, steps], one contiguous record per point
        block = np.ascontiguousarray(cube[:, :, start:stop].transpose(2, 1, 0)).tobytes()
        for i in range(stop - start):
            pipe.setex("%d.%d" % (args.forecast_id, start + i), args.ttl,
                       header + block[i * stride:(i + 1) * stride])
        pipe.execute()
        if (start // BATCH) % 25 == 0:
            ac.progress(50 + 50 * stop / n_points)

    meta = {
        "forecast_id": args.forecast_id,
        "model": args.model,
        "initialized_at_utc": init.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "nx": nx, "ny": ny, "count": n_points,
        "south": round(grid["south"], 6), "west": round(grid["west"], 6),
        "dlat": grid["dlat"], "dlon": grid["dlon"],
        "index": "n = y * nx + x, y from the south edge, x from the west edge",
        "parameters": names,
        "step_minutes": [f["minutes"] for f in frames],
    }
    r.setex("%d.meta" % args.forecast_id, args.ttl, json.dumps(meta))
    ac.emit({
        "model": args.model, "run_id": args.run_id, "forecast_id": args.forecast_id,
        "keys": n_points, "parameters": n_params, "steps": n_steps,
        "bytes_per_key": 4 + stride, "read_seconds": round(read_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
    })


def delete(forecast_id):
    """Remove a run's keys. Returns how many were removed."""
    r = connect()
    removed = 0
    raw = r.get("%d.meta" % forecast_id)
    if raw:
        count = int(json.loads(raw)["count"])
        for start in range(0, count, BATCH):
            removed += r.unlink(*["%d.%d" % (forecast_id, n) for n in range(start, min(start + BATCH, count))])
    else:
        # No meta (an interrupted build, or keys from before meta existed): find them by pattern.
        keys = []
        for key in r.scan_iter(match="%d.*" % forecast_id, count=10000):
            keys.append(key)
            if len(keys) >= BATCH:
                removed += r.unlink(*keys)
                keys = []
        if keys:
            removed += r.unlink(*keys)
    removed += r.unlink("%d.meta" % forecast_id)
    return removed


def exists(forecast_id):
    return bool(connect().exists("%d.meta" % forecast_id))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("action", choices=["build", "delete", "exists"])
    p.add_argument("--model", default="atmo")
    p.add_argument("--work-dir")
    p.add_argument("--run-id")
    p.add_argument("--forecast-id", type=int, required=True)
    p.add_argument("--steps", type=int, default=769)
    p.add_argument("--tiff-dir", default="", help="read the TIFFs from here instead of the staging run folder")
    p.add_argument("--quarter-hour-span-hours", type=int, default=24,
                   help="every 15-minute frame for this many hours, then one frame per hour")
    p.add_argument("--ttl", type=int, default=86400, help="seconds a key lives")
    args = p.parse_args()

    if args.action == "build":
        if not args.work_dir or not args.run_id:
            ac.fail("point_cache build needs --work-dir and --run-id", 2)
        build(args)
    elif args.action == "delete":
        ac.emit({"forecast_id": args.forecast_id, "removed": delete(args.forecast_id)})
    else:
        sys.exit(0 if exists(args.forecast_id) else 1)


if __name__ == "__main__":
    main()
