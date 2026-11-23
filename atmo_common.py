"""Shared helpers for the atmo-v2 flow scripts.

Ported from panahon-nwp (branch v1.2) storage/script/atmo/atmo_common.py and
app/Traits/S5cmd.php / app/Support/AtmoV5.php, with the flow node contract added.

Node contract (see BBC.md in the flow app):
  - exit 0 = success, anything else fails the node
  - the LAST stdout line is the node's JSON manifest ("emit")
  - "PROGRESS <n>" lines on stdout report percent done
  - {"halt": true} in the manifest ends the run gracefully
  - manifests carry file PATHS and small scalars, never bulk data

The ATMO v5 store is zarr v3 with one object per timestep and a Blosc(zstd) codec, so a
chunk object can be decoded on its own without xarray or zarr.

Credentials arrive through the environment and nothing here reads a config file. The
flow node names stored connections in its `connections` map (ENV PREFIX => connection)
and the engine exports each as {PREFIX}_{KEY} variables at execution time:

  ATMO_S3     S3 connection      ATMO_S3_KEY, _SECRET, _REGION, _BUCKET, _ROOT
  NWP_DB      database           NWP_DB_HOST, _PORT, _DATABASE, _USERNAME, _PASSWORD
  REDIS       Redis store        REDIS_HOST, _PORT, _DATABASE, _USERNAME, _PASSWORD
  NWP_PUBLIC  delivery disk      NWP_PUBLIC_ROOT (and _URL when the disk has one)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

RUN_PATTERN = re.compile(r"^v5-(\d{8}_\d{6})$")

STEP_MINUTES = 15

# TIFF band order, fixed. The first six match the older ATMO TIFFs.
# (name, description, unit)
BANDS = [
    ("PRATE", "PRATE:surface", "kg m-2 s-1"),
    ("PRES", "PRES:mean sea level", "Pa"),
    ("RH", "RH:2 m above ground", "%"),
    ("TMP", "TMP:2 m above ground", "K"),
    ("UGRD", "UGRD:10 m above ground", "m s-1"),
    ("VGRD", "VGRD:10 m above ground", "m s-1"),
    ("TCDC", "TCDC:entire atmosphere", "%"),
    ("REFC", "REFC:entire atmosphere", "dBZ"),
    ("DPT", "DPT:2 m above ground", "K"),
    ("TPRATE", "TPRATE:surface", "kg m-2"),
]

# model_forecast_tiffs.parameters, same list the legacy model config carries.
MODEL_PARAMETERS = ["PRATE", "PRES", "RH", "TMP", "UVGRD", "TCDC", "REFC", "DPT", "TPRATE"]

# Point cache parameter order => 1-based TIFF band. The first eight match the ECMWF cache.
CACHE_PARAMETERS = [
    ("TPRATE", 10), ("PRES", 2), ("VGRD", 6), ("UGRD", 5), ("DPT", 9),
    ("TMP", 4), ("PRATE", 1), ("RH", 3), ("TCDC", 7), ("REFC", 8),
]

# Arrays downloaded for every frame: zarr array name => chunk key after "c/<step>/".
# precipitation is handled apart (fetched once per run for the running total).
FRAME_VARIABLES = {
    "mean_sea_level_pressure": "0/0",
    "2m_temperature": "0/0",
    "2m_dewpoint_temperature": "0/0",
    "10m_u_component_of_wind": "0/0",
    "10m_v_component_of_wind": "0/0",
    "total_cloud_cover": "0/0",
    "reflectivity": "0/0/0",
}


# --------------------------------------------------------------------------- contract

def emit(manifest):
    """Print the node manifest as the last stdout line."""
    sys.stdout.flush()
    print(json.dumps(manifest, separators=(",", ":")), flush=True)


def progress(percent):
    print("PROGRESS %d" % max(0, min(100, int(percent))), flush=True)


def log(message):
    """Diagnostics go to stderr so they can never be mistaken for the manifest."""
    print(message, file=sys.stderr, flush=True)


def fail(message, code=1):
    log(message)
    sys.exit(code)


def env(name, default=None, required=False):
    value = os.environ.get(name)
    if value is None or value == "":
        if required:
            fail("missing environment variable %s" % name, 2)
        return default
    return value


# --------------------------------------------------------------------------- naming

def init_time(run_id):
    """Start of a run (UTC, naive) from its folder name, e.g. v5-20261113_180000."""
    match = RUN_PATTERN.match(run_id)
    if not match:
        fail('"%s" is not a run name like v5-20261113_180000' % run_id, 2)
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def model_prefix(model):
    """atmo-v2 -> ATMO-V2: folder and file-name prefix on the public disk."""
    return model.upper()


def step_name(step):
    return "%04d" % step


# The legacy app formats these with PHP's "Ymd_HiT". Its TIFF names shift a UTC-labelled
# time by +8 h, and its .bin names use a real Asia/Manila time, whose abbreviation is PST.
# Both quirks are kept so atmo-v2 paths are shaped exactly like atmo's.
def tiff_stamp(moment_utc):
    return (moment_utc + timedelta(hours=8)).strftime("%Y%m%d_%H%M") + "UTC"


def bin_stamp(moment_utc):
    return (moment_utc + timedelta(hours=8)).strftime("%Y%m%d_%H%M") + "PST"


def tiff_folder(model, init):
    return "%s/TIFF/%s" % (model_prefix(model), tiff_stamp(init))


def bin_folder(model, init):
    return "%s/BIN/%s" % (model_prefix(model), bin_stamp(init))


def tiff_name(model, init, step):
    valid = init + timedelta(minutes=step * STEP_MINUTES)
    return "%s_%s.%s.tiff" % (model_prefix(model), tiff_stamp(valid), step_name(step))


def bin_name(model, init, step):
    valid = init + timedelta(minutes=step * STEP_MINUTES)
    return "%s_UVGRD_%s.%s.bin" % (model_prefix(model), bin_stamp(valid), step_name(step))


def manila_naive(moment_utc):
    """What the legacy app stores in forecast_time columns: Manila local time, no zone."""
    return (moment_utc + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def batches(steps, size):
    """Split frames 0..steps-1 into consecutive batches. Returns [(batch_no, first, last)]."""
    if steps < 1 or size < 1:
        raise ValueError("steps and size must be positive")
    out = []
    for batch_no, first in enumerate(range(0, steps, size)):
        out.append((batch_no, first, min(first + size, steps) - 1))
    return out


def cache_minutes(frames, quarter_hour_span_hours=24, step=STEP_MINUTES):
    """Minutes since run start of every time step kept in the point cache: each
    15-minute frame for the first hours, then hourly, limited to the frames the run has.
    Port of AtmoV5::cacheMinutes."""
    span = quarter_hour_span_hours * 60
    last = (frames - 1) * step
    minutes = list(range(0, min(span, last) + 1, step))
    minutes += list(range(span + 60, last + 1, 60))
    return minutes


# --------------------------------------------------------------------------- run dir

def run_dir(work_dir, run_id):
    return os.path.join(work_dir, run_id)


def load_meta(run_directory):
    with open(os.path.join(run_directory, "meta.json")) as f:
        return json.load(f)


def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def chunk_path(run_directory, step):
    """The precipitation chunk of a step, as downloaded for the running total."""
    return os.path.join(run_directory, "rain", "in", "precip_" + step_name(step))


def total_path(run_directory, step):
    """The running rain total of a step, written by rain_total.py."""
    return os.path.join(run_directory, "rain", "tprate_s%s.npy" % step_name(step))


# --------------------------------------------------------------------------- chunks

def read_chunk(path, dtype="<f4"):
    """Decode one downloaded chunk object. Returns None when the file is absent
    (zarr does not store chunks that are entirely fill value)."""
    import numpy as np
    from numcodecs import blosc

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(blosc.decompress(raw), dtype=dtype)


def field_2d(path, meta):
    """2-D surface field as float32 (ny, nx), NaN-filled when the chunk is absent."""
    import numpy as np

    ny, nx = meta["ny"], meta["nx"]
    data = read_chunk(path)
    if data is None:
        return np.full((ny, nx), np.nan, dtype=np.float32)
    return data.reshape(ny, nx).astype(np.float32, copy=False)


def field_3d(path, meta):
    """Pressure-level field as float32 (nlev, ny, nx)."""
    import numpy as np

    ny, nx, nlev = meta["ny"], meta["nx"], meta["nlev"]
    data = read_chunk(path)
    if data is None:
        return np.full((nlev, ny, nx), np.nan, dtype=np.float32)
    return data.reshape(nlev, ny, nx).astype(np.float32, copy=False)


def north_up(arr, meta):
    """The store keeps latitude south to north; rasters and GRIB want north first."""
    return arr[::-1] if meta["lat_ascending"] else arr


# --------------------------------------------------------------------------- s5cmd

def tool(name, env_name):
    """A binary from the environment override, else next to this interpreter (the conda
    env's bin), else whatever PATH has."""
    override = env(env_name)
    if override:
        return override
    beside = os.path.join(os.path.dirname(sys.executable), name)
    return beside if os.path.exists(beside) else name


def s3_root():
    """s3://bucket/prefix/ holding one folder per run.

    From an S3 connection: its bucket plus its root (the key prefix, e.g. forecasts/v5).
    Without a bucket, ATMO_S3_ROOT may carry a full s3:// URL instead.
    """
    bucket = env("ATMO_S3_BUCKET")
    if bucket:
        prefix = (env("ATMO_S3_ROOT") or "").strip("/")
        return "s3://%s/%s" % (bucket, prefix + "/" if prefix else "")
    root = env("ATMO_S3_ROOT", "s3://atmo-philippines-exchange/forecasts/v5/")
    return root if root.endswith("/") else root + "/"


def _s5cmd(args):
    # The bucket is Requester Pays: requests without --request-payer are refused.
    cmd = [tool("s5cmd", "S5CMD_BIN"), "--request-payer", "requester"]
    profile = env("ATMO_S3_PROFILE")
    if profile:
        cmd += ["--profile", profile]
    child_env = dict(os.environ)
    child_env["AWS_REGION"] = env("ATMO_S3_REGION", "us-east-1")
    if not profile:
        # ATMO_S3_KEY / _SECRET are what an S3 connection exports.
        key = env("ATMO_S3_KEY") or env("ATMO_S3_ACCESS_KEY_ID")
        secret = env("ATMO_S3_SECRET") or env("ATMO_S3_SECRET_ACCESS_KEY")
        if not key or not secret:
            fail("no S3 credentials: the node's ATMO_S3 connection has no key/secret", 2)
        child_env["AWS_ACCESS_KEY_ID"] = key
        child_env["AWS_SECRET_ACCESS_KEY"] = secret
    return subprocess.run(cmd + args, env=child_env, capture_output=True, text=True)


def s5cmd_dirs(prefix):
    """Folder names (without the trailing slash) directly under an S3 prefix."""
    done = _s5cmd(["ls", prefix])
    if done.returncode != 0:
        if "no object found" in done.stderr:
            return []  # an empty or absent prefix is not an error
        fail("s5cmd ls failed for %s: %s" % (prefix, done.stderr.strip()[-500:]))
    dirs = []
    for line in done.stdout.splitlines():
        match = re.match(r"^\s*DIR\s+(.+?)/\s*$", line)
        if match:
            dirs.append(match.group(1))
    return dirs


def s5cmd_exists(key):
    return _s5cmd(["ls", key]).returncode == 0


def s5cmd_fetch(copies, optional=()):
    """Download many objects in one s5cmd call.

    copies    {s3 key: local path}
    optional  local paths that are allowed to be absent afterwards
    Returns the local paths that were not downloaded (always a subset of optional).
    """
    if not copies:
        return []
    lines = []
    for key, local in copies.items():
        os.makedirs(os.path.dirname(local), exist_ok=True)
        lines.append('cp "%s" "%s"' % (key, local))
    with tempfile.NamedTemporaryFile("w", prefix="s5cmd", suffix=".txt", delete=False) as batch:
        batch.write("\n".join(lines) + "\n")
    try:
        done = _s5cmd(["run", batch.name])
    finally:
        os.unlink(batch.name)
    # s5cmd exits non-zero when any one copy fails, so judge by what is on disk.
    absent = [local for local in copies.values()
              if not os.path.exists(local) or os.path.getsize(local) == 0]
    unexpected = sorted(set(absent) - set(optional))
    if unexpected:
        fail("s5cmd could not download %d object(s), first: %s - %s"
             % (len(unexpected), unexpected[0], done.stderr.strip()[:500]))
    return absent


def recent_run_ids(limit=3):
    """Newest run folders under the root, newest first. Other folders are ignored."""
    runs = sorted((d for d in s5cmd_dirs(s3_root()) if RUN_PATTERN.match(d)), reverse=True)
    return runs[:limit]


def zarr_path(run_id):
    """S3 path of the run's zarr store (with trailing slash), or None while the run is
    still being uploaded. The root zarr.json is the last object ATMO writes."""
    prefix = s3_root() + run_id + "/"
    stores = [d for d in s5cmd_dirs(prefix) if d.endswith(".zarr")]
    if len(stores) != 1:
        return None
    zarr = prefix + stores[0] + "/"
    return zarr if s5cmd_exists(zarr + "zarr.json") else None


# --------------------------------------------------------------------------- database

def db_connect():
    """Connection to the legacy ETL database (panahon_models_etl)."""
    import pymysql

    return pymysql.connect(
        host=env("NWP_DB_HOST", "127.0.0.1"),
        port=int(env("NWP_DB_PORT", "3306")),
        user=env("NWP_DB_USERNAME", required=True),
        password=env("NWP_DB_PASSWORD", ""),
        database=env("NWP_DB_DATABASE", required=True),
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def public_root(prefix):
    """Absolute path of the NWP tree on the host that serves it: the delivery disk's
    root plus the folder inside it (e.g. .../PANaHON + NWP). '' when unknown."""
    root = (env("NWP_PUBLIC_ROOT") or "").rstrip("/")
    if not root:
        return ""
    return root + ("/" + prefix.strip("/") if prefix.strip("/") else "")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def table_columns(cursor, table):
    cursor.execute("SHOW COLUMNS FROM `%s`" % table)
    return {row["Field"] for row in cursor.fetchall()}


def insert_row(cursor, table, values):
    """INSERT the given columns, silently leaving out any the table does not have
    (the legacy schema differs slightly between deployments)."""
    known = table_columns(cursor, table)
    kept = {k: v for k, v in values.items() if k in known}
    columns = ", ".join("`%s`" % k for k in kept)
    marks = ", ".join(["%s"] * len(kept))
    cursor.execute("INSERT INTO `%s` (%s) VALUES (%s)" % (table, columns, marks), list(kept.values()))
    return cursor.lastrowid


def model_message(cursor, model_id, message):
    """Same operator trail the legacy app writes with NwpModel::pMsg()."""
    try:
        insert_row(cursor, "messages", {
            "messageable_type": "App\\Models\\NwpModel", "messageable_id": model_id,
            "type": "PROCESS", "message": message[:255],
            "created_at": utc_now(), "updated_at": utc_now(),
        })
    except Exception as exc:  # a missing messages table must never fail a run
        log("could not write message: %s" % exc)
