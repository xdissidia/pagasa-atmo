"""Find the next ATMO v5 run to process and claim it.

Port of `app:atmo-run` + AtmoV5::prepareRun (panahon-nwp v1.2). First node of the
atmo-v2-run flow; ticks every 15 minutes and halts unless a complete run that has not
been claimed yet is on S3.

When it claims a run it
  1. downloads the grid / time axis and writes <run_dir>/meta.json
  2. creates (or re-claims) the run's model_forecasts row in the ETL database
  3. emits the run and its batches of frames for the rest of the flow

Environment: ATMO_S3_* and NWP_DB_* (the node's ATMO_S3 and NWP_DB connections).
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import atmo_common as ac


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="atmo-v2", help="nwp_models.model_name to publish as")
    p.add_argument("--work-dir", required=True, help="staging folder that holds one folder per run")
    p.add_argument("--steps", type=int, default=769, help="15-minute frames to process, starting at 0")
    p.add_argument("--batch-size", type=int, default=96, help="frames per batch (96 = one forecast day)")
    p.add_argument("--run", default="", help="a specific run folder, e.g. v5-20261113_180000")
    p.add_argument("--stalled-minutes", type=int, default=120,
                   help="re-claim a started run that recorded no frame for this long")
    return p.parse_args()


def halt(reason):
    ac.log("find_run: " + reason)
    ac.emit({"halt": True, "reason": reason})


def find_run(explicit):
    """Returns (run_id, zarr) or (None, reason)."""
    if explicit:
        ac.init_time(explicit)  # validates the name
        zarr = ac.zarr_path(explicit)
        return (explicit, zarr) if zarr else (None, explicit + " is not on S3 or is still being uploaded")
    # Newest first; a run still uploading has no root zarr.json yet, so fall back to the one before.
    for run_id in ac.recent_run_ids():
        zarr = ac.zarr_path(run_id)
        if zarr:
            return run_id, zarr
        ac.log("find_run: %s is still being uploaded" % run_id)
    return None, "no complete ATMO v5 run found under " + ac.s3_root()


def prepare(run_directory, zarr):
    """Download the grid and time axis of a run and describe them in meta.json."""
    import numpy as np

    meta_path = os.path.join(run_directory, "meta.json")
    if os.path.exists(meta_path):
        return ac.load_meta(run_directory)

    ac.s5cmd_fetch({
        zarr + "zarr.json": os.path.join(run_directory, "zarr.json"),
        zarr + "latitude/c/0": os.path.join(run_directory, "latitude.c0"),
        zarr + "longitude/c/0": os.path.join(run_directory, "longitude.c0"),
        zarr + "time/c/0": os.path.join(run_directory, "time.c0"),
        zarr + "level/c/0": os.path.join(run_directory, "level.c0"),
    })
    lat = ac.read_chunk(os.path.join(run_directory, "latitude.c0"), "<f8")
    lon = ac.read_chunk(os.path.join(run_directory, "longitude.c0"), "<f8")
    time = ac.read_chunk(os.path.join(run_directory, "time.c0"), "<i8")
    level = ac.read_chunk(os.path.join(run_directory, "level.c0"), "<i4")
    if lat is None or lon is None or time is None:
        ac.fail("find_run: latitude/longitude/time chunk missing in " + run_directory)

    spacing = np.diff(time) // 1_000_000_000
    if len(set(spacing.tolist())) != 1:
        ac.fail("find_run: time axis is not evenly spaced")
    if lon[-1] < lon[0]:
        ac.fail("find_run: longitude is not west to east")

    meta = {
        "ny": int(lat.size),
        "nx": int(lon.size),
        "nlev": int(level.size) if level is not None else 0,
        "nt": int(time.size),
        "lat_first": float(lat[0]),
        "lat_last": float(lat[-1]),
        "lon_first": float(lon[0]),
        "lon_last": float(lon[-1]),
        "lat_ascending": bool(lat[-1] > lat[0]),
        "dlat": float(abs(lat[-1] - lat[0]) / (lat.size - 1)),
        "dlon": float(abs(lon[-1] - lon[0]) / (lon.size - 1)),
        "t0_epoch": int(time[0] // 1_000_000_000),
        "step_seconds": int(spacing[0]),
    }
    ac.write_json_atomic(meta_path, meta)
    return meta


def ensure_model(cursor, model, steps):
    """The nwp_models row atmo-v2 publishes under; created on first use.
    Locked FOR UPDATE: it is the mutex that keeps two ticks from claiming one run."""
    cursor.execute("SELECT id FROM nwp_models WHERE model_name = %s FOR UPDATE", (model,))
    row = cursor.fetchone()
    if row:
        return row["id"]
    parameters = {name: {} for name in ac.MODEL_PARAMETERS}
    model_id = ac.insert_row(cursor, "nwp_models", {
        "model_name": model,
        "forecast_time_step_type": "minutes",
        "data_type": "zarr",
        "root_directory": ac.s3_root(),
        "file_count": steps,
        "parameter_count": 1,
        "total_parameter_count": steps,
        "parameters": json.dumps(parameters),
        "created_at": ac.utc_now(),
        "updated_at": ac.utc_now(),
    })
    ac.log("find_run: created nwp_models row %d for %s" % (model_id, model))
    return model_id


def claim(connection, args, run_id, zarr, init):
    """Returns (forecast_id, None) when the run is ours to process, (None, reason) otherwise."""
    with connection.cursor() as cursor:
        model_id = ensure_model(cursor, args.model, args.steps)
        cursor.execute(
            "SELECT id, is_completed, file_downloaded_count, updated_at FROM model_forecasts "
            "WHERE nwp_model_id = %s AND initialized_time = %s ORDER BY id LIMIT 1",
            (model_id, init.strftime("%Y-%m-%d %H:%M:%S")),
        )
        existing = cursor.fetchone()
        now = ac.utc_now()

        if existing:
            if existing["is_completed"]:
                connection.rollback()
                return None, run_id + " is already processed"
            updated = existing["updated_at"]
            fresh = updated and updated > datetime.utcnow() - timedelta(minutes=args.stalled_minutes)
            if fresh:
                connection.rollback()
                return None, "%s is in progress (%d of %d frames)" % (
                    run_id, existing["file_downloaded_count"] or 0, args.steps)
            # Stalled (a failed first attempt, a lost worker): take it again under the
            # same id, so everything already recorded for it stays valid.
            cursor.execute("UPDATE model_forecasts SET directory = %s, file_count = %s, parameter_count = %s, "
                           "updated_at = %s WHERE id = %s", (zarr, args.steps, args.steps, now, existing["id"]))
            forecast_id = existing["id"]
            ac.log("find_run: re-claimed stalled forecast %d for %s" % (forecast_id, run_id))
        else:
            forecast_id = ac.insert_row(cursor, "model_forecasts", {
                "nwp_model_id": model_id,
                "initialized_time": init.strftime("%Y-%m-%d %H:%M:%S"),
                "directory": zarr,
                "file_count": args.steps,
                "file_downloaded_count": 0,
                "parameter_count": args.steps,
                "completed_parameter_count": 0,
                "is_completed": 0,
                "is_cached": 0,
                "created_at": now,
                "updated_at": now,
            })
        ac.model_message(cursor, model_id, "AtmoV2 (%s) : START, %d frames" % (run_id, args.steps))
    connection.commit()
    return forecast_id, None


def main():
    args = parse_args()
    run_id, zarr = find_run(args.run)
    if not run_id:
        return halt(zarr)

    init = ac.init_time(run_id)
    run_directory = ac.run_dir(args.work_dir, run_id)

    connection = ac.db_connect()
    try:
        # Cheap look before any download: most ticks end here.
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT f.is_completed FROM model_forecasts f JOIN nwp_models m ON m.id = f.nwp_model_id "
                "WHERE m.model_name = %s AND f.initialized_time = %s AND f.is_completed = 1 LIMIT 1",
                (args.model, init.strftime("%Y-%m-%d %H:%M:%S")),
            )
            done = cursor.fetchone()
        connection.rollback()
        if done:
            return halt(run_id + " is already processed")

        meta = prepare(run_directory, zarr)
        if meta["step_seconds"] != ac.STEP_MINUTES * 60:
            ac.fail("find_run: %s time step is %d s, expected %d s"
                    % (run_id, meta["step_seconds"], ac.STEP_MINUTES * 60))
        if meta["nt"] < args.steps:
            ac.fail("find_run: %s has only %d timesteps, %d requested" % (run_id, meta["nt"], args.steps))

        forecast_id, reason = claim(connection, args, run_id, zarr, init)
    finally:
        connection.close()
    if forecast_id is None:
        return halt(reason)

    run_at = init.strftime("%Y-%m-%d %H:%M:%S")
    rows = [{
        "model": args.model, "run_id": run_id, "run_at": run_at, "forecast_id": forecast_id,
        "zarr": zarr, "batch_no": batch_no, "step_from": first, "step_to": last,
    } for batch_no, first, last in ac.batches(args.steps, args.batch_size)]

    manifest = {
        "model": args.model,
        "run_id": run_id,
        "run_at": run_at,
        "zarr": zarr,
        "forecast_id": forecast_id,
        "steps": args.steps,
        "run_dir": run_directory,
        "run_json": os.path.join(run_directory, "run.json"),
        "grid": {"nx": meta["nx"], "ny": meta["ny"]},
        "batch_count": len(rows),
        "batches": rows,
    }
    # rain_total re-emits this, so the batch list reaches the sink behind it.
    ac.write_json_atomic(manifest["run_json"], manifest)
    ac.emit(manifest)


if __name__ == "__main__":
    main()
