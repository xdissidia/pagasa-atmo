"""Unit tests for the pure logic of the atmo scripts.

    {FLOWS_ENV_ROOT}/PyNWP/bin/python -m unittest test_atmo -v      (from this folder)

No S3, database or Redis needed. The rain-total tests need numpy and are skipped
without it, so the file also runs under a bare system python.
"""
import contextlib
import io
import json
import types
import unittest
from datetime import datetime, timedelta
from unittest import mock

import atmo_common as ac
import find_run

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class Batches(unittest.TestCase):
    def test_production_run_is_nine_batches(self):
        out = ac.batches(769, 96)
        self.assertEqual(len(out), 9)
        self.assertEqual(out[0], (0, 0, 95))
        self.assertEqual(out[7], (7, 672, 767))
        self.assertEqual(out[8], (8, 768, 768))        # the last batch is one frame

    def test_every_step_lands_in_exactly_one_batch(self):
        for steps, size in ((769, 96), (10, 4), (96, 96), (1, 96), (97, 96)):
            covered = [s for _, first, last in ac.batches(steps, size) for s in range(first, last + 1)]
            self.assertEqual(covered, list(range(steps)), (steps, size))

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            ac.batches(0, 96)
        with self.assertRaises(ValueError):
            ac.batches(10, 0)


class CacheMinutes(unittest.TestCase):
    def test_eight_day_run_keeps_265_steps(self):
        # 15-minute frames for 24 h (97), then hourly up to 192 h (168). Same as legacy.
        minutes = ac.cache_minutes(769)
        self.assertEqual(len(minutes), 265)
        self.assertEqual(minutes[:3], [0, 15, 30])
        self.assertEqual(minutes[96:99], [1440, 1500, 1560])
        self.assertEqual(minutes[-1], 768 * 15)

    def test_short_run_is_limited_to_its_frames(self):
        self.assertEqual(ac.cache_minutes(10), [m * 15 for m in range(10)])
        self.assertEqual(ac.cache_minutes(1), [0])

    def test_every_cached_minute_is_a_real_frame(self):
        for minute in ac.cache_minutes(769):
            self.assertEqual(minute % ac.STEP_MINUTES, 0)
            self.assertLess(minute // ac.STEP_MINUTES, 769)


class Naming(unittest.TestCase):
    INIT = datetime(2026, 11, 13, 18, 0, 0)

    def test_run_name_parses_to_utc_start(self):
        self.assertEqual(ac.init_time("v5-20261113_180000"), self.INIT)
        self.assertIsNone(ac.RUN_PATTERN.match("v5_20260817_00"))   # the stray folder on S3

    def test_paths_are_shaped_like_the_legacy_model(self):
        # Legacy atmo: ATMO/TIFF/20261114_0200UTC/ATMO_20261114_0215UTC.0001.tiff
        self.assertEqual(ac.tiff_folder("atmo", self.INIT), "ATMO/TIFF/20261114_0200UTC")
        self.assertEqual(ac.tiff_name("atmo", self.INIT, 1), "ATMO_20261114_0215UTC.0001.tiff")
        # Legacy atmo: ATMO/BIN/20261114_0200PST/ATMO_UVGRD_20261114_0215PST.0001.bin
        self.assertEqual(ac.bin_folder("atmo", self.INIT), "ATMO/BIN/20261114_0200PST")
        self.assertEqual(ac.bin_name("atmo", self.INIT, 1), "ATMO_UVGRD_20261114_0215PST.0001.bin")
        self.assertEqual(ac.tiff_name("atmo", self.INIT, 768), "ATMO_20261122_0200UTC.0768.tiff")

    def test_forecast_time_is_manila_local(self):
        self.assertEqual(ac.manila_naive(self.INIT), "2026-11-14 02:00:00")

    def test_cache_parameters_point_at_real_bands(self):
        names = [name for name, _, _ in ac.BANDS]
        for name, band in ac.CACHE_PARAMETERS:
            self.assertEqual(names[band - 1], name)
        self.assertEqual(len(ac.CACHE_PARAMETERS), len(ac.BANDS))


@unittest.skipIf(np is None, "numpy not installed")
class RainTotal(unittest.TestCase):
    """p[i] = c[i] - c[i-4]: rebuild the running total c from the hourly amounts p."""

    def totals(self, quarter_hour_rain):
        import rain_total

        truth = np.cumsum(np.asarray(quarter_hour_rain, dtype=np.float64))
        truth = np.concatenate([[0.0], truth])                 # c[0] = 0
        last = len(truth) - 1

        def read_step(step):                                   # what the store gives, for step >= 4
            return np.full((1, 1), truth[step] - truth[step - 4])

        got = [total[0, 0] for _, total in rain_total.running_totals(read_step, last, (1, 1))]
        return np.array(got), truth

    def test_exact_when_each_chain_has_a_dry_period(self):
        # Dry start, a shower, dry again: every chain sees a dry 15 minutes.
        got, truth = self.totals([0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0])
        np.testing.assert_allclose(got, truth)

    def test_never_decreases(self):
        rng = np.random.default_rng(7)
        got, _ = self.totals(rng.uniform(0, 3, 60) * (rng.uniform(0, 1, 60) > 0.5))
        self.assertTrue(np.all(np.diff(got) >= -1e-9))

    def test_is_a_lower_bound_when_it_never_stops_raining(self):
        got, truth = self.totals([1.0] * 24)
        self.assertTrue(np.all(got <= truth + 1e-9))
        # ...off by less than the first hour's rain.
        self.assertLess(float(np.max(truth - got)), 4.0)

    def test_all_dry_is_all_zero(self):
        got, _ = self.totals([0.0] * 12)
        np.testing.assert_array_equal(got, np.zeros(13))


class CheckOnly(unittest.TestCase):
    """find_run --check-only: the atmo-run precheck. S3 and the database are stubbed."""

    RUN = "v5-20261124_000000"

    def check(self, row, found=(RUN, "s3://bucket/v5-20261124_000000.zarr")):
        cursor = mock.MagicMock()
        cursor.fetchone.return_value = row
        connection = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        args = types.SimpleNamespace(run="", model="atmo", steps=769, stalled_minutes=120)
        out = io.StringIO()
        with mock.patch.object(find_run, "find_run", return_value=found), \
                mock.patch.object(ac, "db_connect", return_value=connection), \
                contextlib.redirect_stdout(out):
            find_run.check(args)
        connection.commit.assert_not_called()
        return json.loads(out.getvalue().strip().splitlines()[-1])

    def test_new_run_is_admitted(self):
        self.assertEqual(self.check(None), {"admit": True, "reason": self.RUN + " is new", "run_id": self.RUN})

    def test_processed_run_is_not(self):
        got = self.check({"is_completed": 1, "file_downloaded_count": 769, "updated_at": datetime.utcnow()})
        self.assertFalse(got["admit"])
        self.assertIn("already processed", got["reason"])

    def test_run_in_progress_is_not(self):
        got = self.check({"is_completed": 0, "file_downloaded_count": 96, "updated_at": datetime.utcnow()})
        self.assertFalse(got["admit"])
        self.assertIn("in progress (96 of 769", got["reason"])

    def test_stalled_run_is_admitted_for_a_reclaim(self):
        stale = datetime.utcnow() - timedelta(minutes=300)
        got = self.check({"is_completed": 0, "file_downloaded_count": 96, "updated_at": stale})
        self.assertTrue(got["admit"])

    def test_no_complete_run_on_s3_is_not(self):
        got = self.check(None, found=(None, "no complete ATMO v5 run found"))
        self.assertEqual(got, {"admit": False, "reason": "no complete ATMO v5 run found"})


if __name__ == "__main__":
    unittest.main()
