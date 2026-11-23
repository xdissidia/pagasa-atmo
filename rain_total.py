"""Build the running rain total (TPRATE) for every frame of an ATMO v5 run.

Port of AtmoRainTotalWorker + atmo_rain_total.py (panahon-nwp v1.2). Runs once per run,
before any frame: every frame's TPRATE band comes from the totals written here.

The store gives p[i] = rain (mm) in the hour ending at step i, every 15 minutes, for
i >= 4. With c[i] the rain since the run started:

    p[i] = c[i] - c[i-4]      so      c[i] = p[i] + c[i-4]

That recurrence is exact. It splits the steps into four chains (i mod 4). The chain
through step 0 starts at c[0] = 0 and is fully known. The other three start at c[1],
c[2], c[3], the rain in the first 15 / 30 / 45 minutes, which the store does not give.

Those three numbers are pinned down by the fact that a running total never decreases:
c[i] >= c[i-1] for every step. Each is set to the smallest value that keeps the whole
series non-decreasing. That is the true value wherever at least one 15-minute period of
the matching chain was dry at that grid point, which over a multi-day run is almost
everywhere; otherwise it is a lower bound, off by less than the first hour's rain.

Reads   <run_dir>/rain/in/precip_<step:04d>     one chunk object per step 4..last (downloaded here)
Writes  <run_dir>/rain/tprate_s<step:04d>.npy   float32, native row order, mm, steps 0..last
"""
import argparse
import json
import os

import atmo_common as ac


def chain_offsets(read_step, last_step, shape):
    """c[1], c[2], c[3]: the smallest starting totals that keep the series non-decreasing.

    read_step(i) returns p[i] as float64 (>= 0, no NaN). Chain r holds steps 4k + r.
    With S_r(k) the sum of p over chain r up to cycle k, c[4k + r] = c[r] + S_r(k), and
    c[4k + r] >= c[4k + r - 1] for every k gives c[r] >= c[r-1] + max_k (S_{r-1}(k) - S_r(k)).
    """
    import numpy as np

    sums = [np.zeros(shape, dtype=np.float64) for _ in range(4)]
    gaps = [np.zeros(shape, dtype=np.float64) for _ in range(4)]  # gaps[r] = max_k (S_{r-1} - S_r), k=0 gives 0
    for step in range(4, last_step + 1):
        r = step % 4
        sums[r] += read_step(step)
        if r:
            np.maximum(gaps[r], sums[r - 1] - sums[r], out=gaps[r])
    offsets = [np.zeros(shape, dtype=np.float64)]
    for r in (1, 2, 3):
        offsets.append(offsets[r - 1] + gaps[r])
    return offsets


def running_totals(read_step, last_step, shape):
    """Yield (step, c[step]) for step 0..last_step, as float64."""
    import numpy as np

    offsets = chain_offsets(read_step, last_step, shape)
    chain = [offset.copy() for offset in offsets]   # c at the latest step of each chain
    previous = np.zeros(shape, dtype=np.float64)
    for step in range(0, last_step + 1):
        r = step % 4
        if step >= 4:
            chain[r] += read_step(step)
        # Guard against rounding: never below the step before.
        total = np.maximum(chain[r], previous)
        previous = total
        yield step, total


def build(run_directory, last_step):
    import numpy as np

    meta = ac.load_meta(run_directory)
    shape = (meta["ny"], meta["nx"])
    missing = set()

    def read_step(step):
        path = ac.chunk_path(run_directory, step)
        if not os.path.exists(path):
            missing.add(step)   # a chunk that is entirely "no data" is not stored
        hourly = np.nan_to_num(ac.field_2d(path, meta), nan=0.0).astype(np.float64)
        return np.maximum(hourly, 0.0)

    peak = 0.0
    for step, total in running_totals(read_step, last_step, shape):
        out = ac.total_path(run_directory, step)
        np.save(out + ".tmp.npy", total.astype(np.float32))
        os.replace(out + ".tmp.npy", out)
        peak = float(total.max())
        if step % 50 == 0:
            ac.progress(50 + 50 * step / max(1, last_step))
    return sorted(missing), peak


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-json", required=True, help="run.json written by find_run")
    args = p.parse_args()

    with open(args.run_json) as f:
        run = json.load(f)
    run_directory, zarr, last = run["run_dir"], run["zarr"], run["steps"] - 1

    skipped = os.path.exists(ac.total_path(run_directory, last))
    missing, peak = [], None
    if not skipped:
        # precipitation[i] is the rain in the hour ending at step i, so every step's
        # chunk is needed, not just the whole hours. The chunks stay in rain/in and the
        # frame batches reuse them, so nothing is downloaded twice.
        copies = {zarr + "precipitation/c/%d/0/0" % step: ac.chunk_path(run_directory, step)
                  for step in range(4, last + 1)}
        os.makedirs(os.path.join(run_directory, "rain", "in"), exist_ok=True)
        # A chunk that is entirely "no data" is not stored, so any of these may be absent.
        ac.s5cmd_fetch(copies, optional=list(copies.values()))
        ac.progress(50)
        missing, peak = build(run_directory, last)
        if missing:
            ac.log("rain_total: no precipitation chunk for steps %s (counted as 0)"
                   % ",".join(map(str, missing)))

    run["rain_total"] = {"skipped": skipped, "missing_chunks": len(missing), "largest_mm": peak}
    ac.emit(run)


if __name__ == "__main__":
    main()
