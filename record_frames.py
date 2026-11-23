"""Record a delivered batch of frames in the ETL database.

Runs after the batch's TIFFs (and wind .bin files) were delivered to the public disk.
Writes what the legacy AtmoStepWorker / ParameterWorker write, so consumers of
panahon_models_etl see atmo-v2 exactly like atmo:

  model_forecast_tiffs            one row per frame
  model_forecast_files            ) one row each per wind .bin,
  model_forecast_file_variables   ) variable UVGRD
  model_forecast_data             )
  model_forecasts                 file_downloaded_count, updated_at

Those tables carry no unique indexes, so every write here is find-then-insert inside one
transaction that first locks the run's model_forecasts row: a redelivered job waits for
the first copy and then finds its rows already there.

Environment: NWP_DB_* (the node's NWP_DB connection) and NWP_PUBLIC_ROOT (root of the
node's NWP_PUBLIC delivery disk). file_path is stored as that root + --public-prefix +
the file, the absolute path on the serving host, like legacy.

The public URL base comes from --public-url, else NWP_PUBLIC_URL (a disk that declares
one) + the prefix, else it is read off the newest file_url another model already has in
the database, so atmo-v2 links are shaped exactly like the ones consumers use today.
"""
import argparse
import json
import os

import atmo_common as ac


def find_or_insert(cursor, table, where, values):
    """Returns (id, created). `where` identifies the row; `values` are the extra columns."""
    clause = " AND ".join("`%s` = %%s" % column for column in where)
    cursor.execute("SELECT id FROM `%s` WHERE %s ORDER BY id LIMIT 1" % (table, clause), list(where.values()))
    row = cursor.fetchone()
    if row:
        return row["id"], False
    now = ac.utc_now()
    return ac.insert_row(cursor, table, {**where, **values, "created_at": now, "updated_at": now}), True


def public_url_base(cursor, args):
    """URL of the NWP tree, without a trailing slash."""
    if args.public_url:
        return args.public_url.rstrip("/")
    declared = (ac.env("NWP_PUBLIC_URL") or "").rstrip("/")
    if declared:
        prefix = args.public_prefix.strip("/")
        return declared + ("/" + prefix if prefix else "")
    # Legacy URLs are {base}/{MODEL}/TIFF/{run}/{file}: four segments after the base.
    cursor.execute("SELECT file_url FROM model_forecast_tiffs WHERE file_url LIKE %s ORDER BY id DESC LIMIT 1",
                   ("%/TIFF/%/%",))
    row = cursor.fetchone()
    if row and row["file_url"].count("/") > 6:
        return row["file_url"].rsplit("/", 4)[0]
    ac.fail("record_frames: cannot tell the public URL of the NWP tree - no other model has a "
            "file_url to copy it from; pass --public-url")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frames-json", required=True, help="batch file written by frames.py")
    p.add_argument("--public-prefix", default="NWP", help="folder of the NWP tree inside the delivery disk")
    p.add_argument("--public-url", default="", help="URL of the NWP tree; default: derived (see the docstring)")
    args = p.parse_args()

    with open(args.frames_json) as f:
        batch = json.load(f)
    forecast_id = int(batch["forecast_id"])
    public_root = ac.public_root(args.public_prefix)

    connection = ac.db_connect()
    created = {"tiffs": 0, "winds": 0}
    try:
        with connection.cursor() as cursor:
            public_url = public_url_base(cursor, args)

            def located(folder, path):
                rel = folder + "/" + os.path.basename(path)
                return public_url + "/" + rel, (public_root + "/" + rel) if public_root else rel

            cursor.execute("SELECT id, nwp_model_id FROM model_forecasts WHERE id = %s FOR UPDATE", (forecast_id,))
            forecast = cursor.fetchone()
            if not forecast:
                ac.fail("record_frames: forecast %d is gone (pruned while this batch waited?)" % forecast_id)
            model_id = forecast["nwp_model_id"]
            now = ac.utc_now()

            for frame in batch["frames"]:
                url, path = located(batch["tiff_folder"], frame["tiff"])
                tiff_id, new = find_or_insert(
                    cursor, "model_forecast_tiffs",
                    {"model_forecast_id": forecast_id, "header_forecast_time": frame["minutes"]},
                    {"nwp_model_id": model_id, "parameters": "|".join(ac.MODEL_PARAMETERS),
                     "header_parameters": frame["header_parameters"],
                     "forecast_time": frame["forecast_time"], "file_url": url, "file_path": path})
                if new:
                    created["tiffs"] += 1
                else:
                    cursor.execute("UPDATE model_forecast_tiffs SET file_url = %s, file_path = %s, updated_at = %s "
                                   "WHERE id = %s", (url, path, now, tiff_id))

                if not frame.get("bin"):
                    continue
                url, path = located(batch["bin_folder"], frame["bin"])
                file_id, _ = find_or_insert(
                    cursor, "model_forecast_files",
                    {"model_forecast_id": forecast_id, "file": path},
                    {"file_datetime": now, "read_at": now})
                variable_id, _ = find_or_insert(
                    cursor, "model_forecast_file_variables",
                    {"model_forecast_file_id": file_id, "variable": "UVGRD"},
                    {"nwp_model_id": model_id, "model_forecast_id": forecast_id,
                     "processed_at": now, "completed_at": now})
                data_id, new = find_or_insert(
                    cursor, "model_forecast_data",
                    {"model_forecast_id": forecast_id, "variable": "UVGRD", "forecast_time": frame["forecast_time"]},
                    {"nwp_model_id": model_id, "model_forecast_file_variable_id": variable_id, "json_path": url})
                if new:
                    created["winds"] += 1
                else:
                    cursor.execute("UPDATE model_forecast_data SET json_path = %s, updated_at = %s WHERE id = %s",
                                   (url, now, data_id))

            cursor.execute("SELECT COUNT(*) AS n FROM model_forecast_tiffs WHERE model_forecast_id = %s", (forecast_id,))
            total = cursor.fetchone()["n"]
            # updated_at is the run's heartbeat: find_run re-claims a run that stops moving.
            cursor.execute("UPDATE model_forecasts SET file_downloaded_count = %s, updated_at = %s WHERE id = %s",
                           (total, now, forecast_id))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    ac.emit({
        "model": batch["model"], "run_id": batch["run_id"], "forecast_id": forecast_id,
        "batch_no": batch["batch_no"], "frames_recorded": len(batch["frames"]),
        "tiffs_created": created["tiffs"], "winds_created": created["winds"],
        "frames_total": total, "public_url": public_url, "recorded_at_utc": ac.utc_now(),
    })


if __name__ == "__main__":
    main()
