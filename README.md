# pagasa-atmo

The `atmo` script group of the Flow orchestration engine: ATMO v5 forecast runs
(zarr on S3) into 10-band COG TIFFs, wind-particle `.bin` files and the Redis
point cache, published as model `atmo-v2`. Ported from panahon-nwp, branch v1.2.

Clone it as the folder **`atmo`** inside the scripts tree (`FLOWS_SCRIPTS_ROOT`);
flow nodes reference the scripts as `atmo/<name>`:

    git clone <this repo> /mnt/data/home/ict/flow-scripts/atmo

| Script | Does |
|---|---|
| `find_run` | finds and claims the newest complete run, emits its batches |
| `rain_total` | running rain total every frame's TPRATE band uses |
| `frames` | builds one batch of frames: COG TIFF + wind `.bin` |
| `record_frames` | records a delivered batch in the ETL database |
| `point_cache` | builds / deletes / checks the Redis point cache |
| `finish_run` | marks the run complete, cleans up, prunes old runs |

The flows that use them are seeded by `AtmoV2FlowSeeder` in the Flow app; its
docblock lists the worker `.env` values, and `deploy/envs/pynwp.yml` there is the
conda environment (`PyNWP`, runtime `python@3.12-nwp`).

The copy on the docker host is the live production tree: a change is live for
every active flow immediately.

## Tests

    python -m unittest test_atmo -v        # pure logic, no S3 / database / Redis
    php artisan flows:scripts --validate   # on a worker: manifest and entrypoints
