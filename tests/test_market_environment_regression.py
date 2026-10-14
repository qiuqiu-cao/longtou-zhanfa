"""Synthetic timing and missing-input contracts; no historical market data."""

import unittest
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from market_environment_regression import features, forecast, verify_bindings


CHANNELS = (
    "sh_a_return_pct",
    "sz_a_return_pct",
    "limit_count",
    "down_limit_count",
    "failed_rate",
    "max_board",
    "relay_promotion_rate",
    "yesterday_premium",
)
ARMS = {"LEVEL", "RIDGE_LEVEL", "RIDGE_LEVEL_DELTA"}


def synthetic_observations():
    # The explicit session list is the authoritative calendar for these tests.
    sessions = pd.bdate_range("2024-01-02", periods=180).strftime("%Y-%m-%d").tolist()
    t = np.arange(len(sessions), dtype=float)
    obs = pd.DataFrame(
        {
            "sh_a_return_pct": 0.7 * np.sin(t / 4.0),
            "sz_a_return_pct": 1.1 * np.cos(t / 7.0),
            "limit_count": 20.0 + np.floor(8.0 * (1.0 + np.sin(t / 9.0))),
            "down_limit_count": np.floor(4.0 * (1.0 + np.cos(t / 6.0))),
            "failed_rate": 0.25 + 0.12 * np.sin(t / 11.0),
            "max_board": 2.0 + np.floor(2.0 * (1.0 + np.sin(t / 13.0))),
            "relay_promotion_rate": 0.4 + 0.2 * np.cos(t / 8.0),
            "yesterday_premium": 0.02 * np.sin(t / 5.0) + 0.003 * np.cos(t / 17.0),
        },
        index=pd.Index(sessions, name="date"),
    )
    return sessions, obs


def ordered(rows):
    return rows.sort_values(["signal_date", "arm", "metric"]).reset_index(drop=True)


class MarketEnvironmentRegressionIntegrity(unittest.TestCase):
    def setUp(self):
        self.sessions, self.obs = synthetic_observations()

    def assert_complete_forecast(self, rows, position):
        self.assertEqual(len(rows), len(ARMS) * len(CHANNELS))
        self.assertEqual(set(rows.arm), ARMS)
        self.assertEqual(set(rows.metric), set(CHANNELS))
        self.assertFalse(rows.duplicated(["signal_date", "arm", "metric"]).any())
        self.assertTrue(rows.signal_date.eq(self.sessions[position]).all())
        self.assertTrue(rows.target_date.eq(self.sessions[position + 1]).all())

    def test_future_observations_cannot_change_any_signal_day_forecast(self):
        position = 160
        original = forecast(self.obs.copy(), self.sessions, [position])
        self.assert_complete_forecast(original, position)
        self.assertTrue(np.isfinite(original.prediction).all())

        changed = self.obs.copy()
        changed.loc[self.sessions[position + 1] :, :] = 1_000_000.0
        altered = forecast(changed, self.sessions, [position])
        pd.testing.assert_frame_equal(ordered(original), ordered(altered))

    def test_delta_does_not_jump_over_an_unobserved_official_session(self):
        missing_position = 95
        gapped = self.obs.copy()
        gapped.loc[self.sessions[missing_position], :] = np.nan
        prepared = features(gapped)
        levels = prepared["LEVEL"]
        deltas = prepared["LEVEL_DELTA"]
        delta_columns = [name + "_delta" for name in CHANNELS]

        self.assertTrue(levels.index.equals(gapped.index))
        self.assertTrue(deltas.index.equals(gapped.index))
        self.assertTrue(levels.loc[self.sessions[missing_position]].isna().all())
        self.assertTrue(
            deltas.loc[self.sessions[missing_position], delta_columns].isna().all()
        )
        self.assertTrue(
            deltas.loc[self.sessions[missing_position + 1], delta_columns].isna().all()
        )
        self.assertTrue(
            np.isfinite(
                deltas.loc[self.sessions[missing_position + 2], delta_columns].to_numpy()
            ).all()
        )

    def test_training_uses_only_last_120_closed_official_target_dates(self):
        position = len(self.sessions) - 2
        original = forecast(self.obs.copy(), self.sessions, [position])
        self.assert_complete_forecast(original, position)
        self.assertTrue(original.train_n.eq(120).all())
        self.assertTrue(
            original.max_training_target_date.eq(self.sessions[position]).all()
        )

        changed = self.obs.copy()
        # Both regions lie outside the permitted training window. The last
        # observation is the forecast's future target, despite existing offline.
        changed.iloc[:35, :] = -1_000_000.0
        changed.iloc[-1, :] = 1_000_000.0
        altered = forecast(changed, self.sessions, [position])
        pd.testing.assert_frame_equal(ordered(original), ordered(altered))

    def test_missing_labels_are_not_imputed_and_current_unknown_is_retained(self):
        position = 160
        target_metric = "yesterday_premium"
        missing_label_positions = [45, 92, 134, 158]
        incomplete = self.obs.copy()
        incomplete.loc[
            [self.sessions[p] for p in missing_label_positions], target_metric
        ] = np.nan
        rows = forecast(incomplete, self.sessions, [position])
        self.assert_complete_forecast(rows, position)
        affected = rows[rows.metric.eq(target_metric)]
        self.assertEqual(set(affected.arm), ARMS)
        self.assertTrue(affected.train_n.eq(116).all())
        self.assertTrue(np.isfinite(affected.prediction).all())
        self.assertTrue(rows.loc[rows.metric.ne(target_metric), "train_n"].eq(120).all())

        unknown_current = self.obs.copy()
        unknown_current.loc[self.sessions[position], :] = np.nan
        unknown_rows = forecast(unknown_current, self.sessions, [position])
        self.assert_complete_forecast(unknown_rows, position)
        challengers = unknown_rows[unknown_rows.arm.ne("LEVEL")]
        baseline = unknown_rows[unknown_rows.arm.eq("LEVEL")]
        self.assertTrue(challengers.prediction.isna().all())
        self.assertTrue(
            challengers.status.astype(str)
            .str.contains("UNKNOWN|MISSING|UNAVAILABLE", case=False, regex=True)
            .all()
        )
        self.assertTrue(np.isfinite(baseline.prediction).all())


class SourceMutationContract(unittest.TestCase):
    def test_changed_protected_definition_is_rejected_before_model_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'upstream_definition.py'
            path.write_text('fixed original definition', encoding='utf-8')
            original = hashlib.sha256(path.read_bytes()).hexdigest()
            protocol = {'input_hashes': {}, 'protected_sha256': {path.name: original}}
            verify_bindings(protocol, root)
            path.write_text('changed definition with same file name', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protected source hash mismatch'):
                verify_bindings(protocol, root)


if __name__ == "__main__":
    unittest.main()
