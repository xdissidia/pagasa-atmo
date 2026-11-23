"""Close an ATMO run: mark it complete and cached, drop its working files, prune old runs.

Port of AtmoStepWorker::finish + AtmoV5::prune / deleteRun (panahon-nwp v1.2). Last node of
the atmo-v2-cache flow, so it only runs after the point cache was built.

Pruning keeps the newest --keep-runs completed runs of this model. For every older run it
removes the database rows and the Redis keys. The delivered TIFF / .bin folders are removed
too when the public tree is reachable as a local path (NWP_PUBLIC_ROOT); when it is not
(an SFTP or S3 disk), they are listed in the manifest as `stale_folders` for whatever owns
that storage to expire.

Environment: NWP_DB_*, REDIS_* (see point_cache), NWP_PUBLIC_ROOT (optional).
"""
import argparse
import os
import shutil

import atmo_common as ac
import point_cache

# Child tables of a forecast, deleted before the forecast row itself.
CHILD_TABLES = ["model_forecast_tiff_data", "model_forecast_file_data", "model_forecast_tiffs",
                "model_forecast_data", "model_forecast_file_variables", "model_forecast_files"]


def delete_run(cursor, forecast, model, work_dir, public_root):
    """Everything a pruned run left behind. Returns the public folders it could not remove."""
    forecast_id = forecast["id"]
    init = forecast["initialized_time"]
    folders = [ac.tiff_folder(model, init), ac.bin_folder(model, init)]
    stale = []
    for folder in folders:
        if public_root and os.path.isdir(public_root):
            shutil.rmtree(os.path.join(public_root, folder), ignore_errors=True)
        else:
            stale.append(folder)

    run_id = "v5-" + init.strftime("%Y%m%d_%H%M%S")
    shutil.rmtree(ac.run_dir(work_dir, run_id), ignore_errors=True)

    try:
        point_cache.delete(forecast_id)
    except Exception as exc:
        # Keys expire on their own; a Redis outage must not block the cleanup.
        ac.log("finish_run: could not purge point cache of forecast %d: %s" % (forecast_id, exc))

    for table in CHILD_TABLES:
        try:
            cursor.execute("DELETE FROM `%s` WHERE model_forecast_id = %%s" % table, (forecast_id,))
        except Exception as exc:   # a table this deployment does not have
            ac.log("finish_run: skipped %s: %s" % (table, exc))
    cursor.execute("DELETE FROM model_forecasts WHERE id = %s", (forecast_id,))
    ac.log("finish_run: removed run %s (forecast %d)" % (init, forecast_id))
    return stale


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="atmo-v2")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--forecast-id", type=int, required=True)
    p.add_argument("--keep-runs", type=int, default=2, help="completed runs kept on disk and in the database")
    args = p.parse_args()

    public_root = (ac.env("NWP_PUBLIC_ROOT") or "").rstrip("/")
    connection = ac.db_connect()
    removed, stale = [], []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, nwp_model_id, file_count FROM model_forecasts WHERE id = %s FOR UPDATE",
                           (args.forecast_id,))
            forecast = cursor.fetchone()
            if not forecast:
                ac.fail("finish_run: forecast %d is gone" % args.forecast_id)
            cursor.execute("SELECT COUNT(*) AS n FROM model_forecast_tiffs WHERE model_forecast_id = %s",
                           (args.forecast_id,))
            frames = cursor.fetchone()["n"]
            if frames < forecast["file_count"]:
                ac.fail("finish_run: forecast %d has %d of %d frames recorded"
                        % (args.forecast_id, frames, forecast["file_count"]))
            if not point_cache.exists(args.forecast_id):
                ac.fail("finish_run: forecast %d is not in the point cache" % args.forecast_id)

            cursor.execute("UPDATE model_forecasts SET is_completed = 1, is_cached = 1, updated_at = %s WHERE id = %s",
                           (ac.utc_now(), args.forecast_id))
            ac.model_message(cursor, forecast["nwp_model_id"],
                             "AtmoV2 (%s) : COMPLETED, %d frames, cached" % (args.run_id, frames))

            # Newest first. Runs newer than the last one we keep stay, finished or not.
            cursor.execute("SELECT id, initialized_time, is_completed FROM model_forecasts "
                           "WHERE nwp_model_id = %s ORDER BY initialized_time DESC", (forecast["nwp_model_id"],))
            kept = 0
            for row in cursor.fetchall():
                if kept < max(1, args.keep_runs):
                    kept += 1 if (row["is_completed"] or row["id"] == args.forecast_id) else 0
                    continue
                stale += delete_run(cursor, row, args.model, args.work_dir, public_root)
                removed.append(str(row["initialized_time"]))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    # The TIFFs were delivered and cached; the staging copy has no reader left.
    shutil.rmtree(ac.run_dir(args.work_dir, args.run_id), ignore_errors=True)

    ac.emit({
        "model": args.model, "run_id": args.run_id, "forecast_id": args.forecast_id,
        "frames": frames, "pruned_runs": removed, "stale_folders": stale,
        "finished_at_utc": ac.utc_now(),
    })


if __name__ == "__main__":
    main()
