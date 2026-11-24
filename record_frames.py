"""Record a delivered batch of frames in the ETL database.

Runs after the batch's TIFFs (and wind .bin files) were delivered to the public disk.
Writes what the legacy AtmoStepWorker / ParameterWorker write, so consumers of
panahon_models_etl see the flow-built atmo exactly like legacy:

  model_forecast_tiffs            one row per frame
  model_forecast_files            ) one row each per wind .bin,
  model_forecast_file_variables   ) variable UVGRD
  model_forecast_data             )
  model_forecasts                 file_downloaded_count, updated_at

Those tables carry no unique indexes, so every write here is find-then-insert inside one
transaction that first locks the run's model_forecasts row: a redelivered job waits for
the first copy and then finds its rows already there.

They also carry no index on model_forecast_id and hold hundreds of thousands of rows, so
a lookup per frame is a full table scan per frame - minutes per batch, with the lock
held, while the run's other batches time out waiting for it. What the run already has
is therefore read ONCE per table and matched in memory, and the session is told to wait
for the lock rather than give up after MariaDB's default 50 seconds.

Environment: NWP_DB_* (the node's NWP_DB connection) and NWP_PUBLIC_ROOT (root of the
node's NWP_PUBLIC delivery disk). file_path is stored as that root + --public-prefix +
the file, the absolute path on the serving host, like legacy.

The public URL base comes from --public-url, else NWP_PUBLIC_URL (a disk that declares
one) + the prefix, else it is read off the newest file_url another model already has in
the database, so these links are shaped exactly like the ones consumers use today.
"""
import argparse
import json
import os

import atmo_common as ac


# Seconds to wait for the run's lock. All batches of a run finish close together and
# record one after another; must stay under the node timeout (300 s).
LOCK_WAIT_SECONDS = 240


def existing(cursor, table, forecast_id, key_columns):
    """{key: id} of the rows this run already has in a table - one scan, not one per frame."""
    cursor.execute("SELECT id, %s FROM `%s` WHERE model_forecast_id = %%s ORDER BY id DESC"
                   % (", ".join("`%s`" % c for c in key_columns), table), (forecast_id,))
    # Newest first, so the dict ends up holding the OLDEST id of any duplicate.
    return {tuple(str(row[c]) for c in key_columns): row["id"] for row in cursor.fetchall()}


def insert(cursor, table, values):
    now = ac.utc_now()
    return ac.insert_row(cursor, table, {**values, "created_at": now, "updated_at": now})


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

            # End the read snapshot the URL lookup opened BEFORE waiting for the lock. The
            # ETL server runs MariaDB 11.8 with innodb_snapshot_isolation=ON: a locking read
            # of a row that another batch changed after this transaction's snapshot fails
            # with error 1020 "Record has changed since last read" - and batches finishing
            # together all update the forecast row. With no snapshot, the locking read is
            # the transaction's first read and simply sees the latest row.
            connection.rollback()
            cursor.execute("SET SESSION innodb_lock_wait_timeout = %s", (LOCK_WAIT_SECONDS,))
            cursor.execute("SELECT id, nwp_model_id FROM model_forecasts WHERE id = %s FOR UPDATE", (forecast_id,))
            forecast = cursor.fetchone()
            if not forecast:
                ac.fail("record_frames: forecast %d is gone (pruned while this batch waited?)" % forecast_id)
            model_id = forecast["nwp_model_id"]
            now = ac.utc_now()

            has_wind = any(frame.get("bin") for frame in batch["frames"])
            tiffs = existing(cursor, "model_forecast_tiffs", forecast_id, ["header_forecast_time"])
            files = existing(cursor, "model_forecast_files", forecast_id, ["file"]) if has_wind else {}
            variables = existing(cursor, "model_forecast_file_variables", forecast_id,
                                 ["model_forecast_file_id", "variable"]) if has_wind else {}
            winds = existing(cursor, "model_forecast_data", forecast_id,
                             ["variable", "forecast_time"]) if has_wind else {}

            for frame in batch["frames"]:
                url, path = located(batch["tiff_folder"], frame["tiff"])
                key = (str(frame["minutes"]),)
                if key in tiffs:
                    cursor.execute("UPDATE model_forecast_tiffs SET file_url = %s, file_path = %s, updated_at = %s "
                                   "WHERE id = %s", (url, path, now, tiffs[key]))
                else:
                    tiffs[key] = insert(cursor, "model_forecast_tiffs", {
                        "model_forecast_id": forecast_id, "header_forecast_time": frame["minutes"],
                        "nwp_model_id": model_id, "parameters": "|".join(ac.MODEL_PARAMETERS),
                        "header_parameters": frame["header_parameters"],
                        "forecast_time": frame["forecast_time"], "file_url": url, "file_path": path})
                    created["tiffs"] += 1

                if not frame.get("bin"):
                    continue
                url, path = located(batch["bin_folder"], frame["bin"])
                file_id = files.get((path,))
                if file_id is None:
                    file_id = files[(path,)] = insert(cursor, "model_forecast_files", {
                        "model_forecast_id": forecast_id, "file": path, "file_datetime": now, "read_at": now})
                variable_id = variables.get((str(file_id), "UVGRD"))
                if variable_id is None:
                    variable_id = variables[(str(file_id), "UVGRD")] = insert(
                        cursor, "model_forecast_file_variables", {
                            "model_forecast_file_id": file_id, "variable": "UVGRD", "nwp_model_id": model_id,
                            "model_forecast_id": forecast_id, "processed_at": now, "completed_at": now})
                key = ("UVGRD", frame["forecast_time"])
                if key in winds:
                    cursor.execute("UPDATE model_forecast_data SET json_path = %s, updated_at = %s WHERE id = %s",
                                   (url, now, winds[key]))
                else:
                    winds[key] = insert(cursor, "model_forecast_data", {
                        "model_forecast_id": forecast_id, "variable": "UVGRD", "forecast_time": frame["forecast_time"],
                        "nwp_model_id": model_id, "model_forecast_file_variable_id": variable_id, "json_path": url})
                    created["winds"] += 1

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
