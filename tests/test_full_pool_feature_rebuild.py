import unittest
import pandas as pd

from full_pool_feature_rebuild import MARKET_COLUMNS, build


def fixture():
    observations = pd.DataFrame([
        {"code": "1", "signal_date": "2025-01-02", "episode_id": "e1", "boards": 1, "start_known": True},
        {"code": "600001", "signal_date": "2025-01-02", "episode_id": "e2", "boards": 2, "start_known": True},
        {"code": "600002", "signal_date": "2025-01-02", "episode_id": "e3", "boards": None, "start_known": False},
    ])
    daily = pd.DataFrame([{"date": "2025-01-02", "code": code, "open": 10, "high": 11,
                           "low": 10, "close": 11, "pre_close": 10, "amount": 100,
                           "turnover": .1, "float_cap": 1000, "st": False, "new_listing": False,
                           "status_verified": True, "source": "daily", "universe_complete": False}
                          for code in ("1", "600001", "600002")])
    context = {column: .1 for column in MARKET_COLUMNS}
    context.update(limit_count=2, consecutive_count=1, failed_rate=.2, seal_rate=.8,
                   yesterday_premium=.01, previous_limit_tail_loss=-.02,
                   promotion_overall=.3, promotion_mid=.4, space_feedback=1,
                   middle_density=.5, ladder_gap_count=0)
    market = pd.DataFrame([{"date": "2025-01-02", "source_timestamp": "2025-01-02 15:00:00",
                            "data_status": "verified", "source": "market",
                            "emotion_state": "WARMING", **context}])
    return observations, daily, market


class FullPoolFeatureRebuildTests(unittest.TestCase):
    def test_first_and_relay_share_one_schema(self):
        frame, report = build(*fixture())
        self.assertEqual(len(frame), 3)
        self.assertTrue(report["first_and_relay_rows_in_same_schema"])
        self.assertEqual(report["board_group_counts"]["FIRST_BOARD"], 1)
        self.assertEqual(report["board_group_counts"]["CONSECUTIVE_BOARD"], 1)

    def test_left_censored_height_is_not_filled(self):
        frame, report = build(*fixture())
        row = frame[frame.code.eq("600002")].iloc[0]
        self.assertEqual(row.feature_status, "left_censored_identity")
        self.assertTrue(pd.isna(row.boards_for_comparison))
        self.assertEqual(report["left_censored_rows"], 1)

    def test_cross_section_features_use_complete_observed_date(self):
        frame, _ = build(*fixture())
        first = frame[frame.code.eq("000001")].iloc[0]
        relay = frame[frame.code.eq("600001")].iloc[0]
        self.assertEqual(first.candidate_pool_count, 3)
        self.assertEqual(first.unknown_height_pool_count, 1)
        self.assertEqual(first.height_gap_to_known_top, 1)
        self.assertEqual(relay.sole_known_top_proxy, 1)

    def test_missing_daily_bar_is_not_zero_filled(self):
        observations, daily, market = fixture()
        daily = daily[daily.code.ne("600001")]
        frame, _ = build(observations, daily, market)
        row = frame[frame.code.eq("600001")].iloc[0]
        self.assertEqual(row.feature_status, "daily_bar_or_security_status_unavailable")
        self.assertTrue(pd.isna(row.open_gap_pct))

    def test_duplicate_candidate_is_rejected(self):
        observations, daily, market = fixture()
        observations = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build(observations, daily, market)

    def test_output_remains_diagnostic(self):
        _, report = build(*fixture())
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")
        self.assertFalse(report["model_training_authorized"])

    def test_expanded_emotion_evidence_is_preserved(self):
        frame, report = build(*fixture())
        self.assertIn("market_failed_tail_loss", frame)
        self.assertIn("market_state_retreat", frame)
        self.assertEqual(frame.iloc[0].market_emotion_state, "WARMING")
        self.assertGreaterEqual(report["emotion_evidence_columns"], 40)

    def test_valid_one_hot_state_replaces_corrupted_display_label(self):
        observations, daily, market = fixture()
        for column in ("state_start", "state_warming", "state_acceleration", "state_consensus",
                       "state_divergence", "state_retreat", "state_repair"):
            market[column] = 0.0
        market["state_warming"] = 1.0
        market["emotion_state"] = "\ufffd\ufffd\ufffd"
        frame, report = build(observations, daily, market)
        self.assertEqual(frame.iloc[0].market_emotion_state, "\u5347\u6e29")
        self.assertEqual(report["emotion_state_one_hot_reconstructed_rows"], 1)
        self.assertEqual(report["emotion_state_unresolved_rows"], 0)

    def test_rebuilt_date_schema_is_normalized(self):
        observations, daily, market = fixture()
        observations = observations.rename(columns={"signal_date": "date"}).drop(columns=["episode_id", "start_known"])
        frame, report = build(observations, daily, market)
        self.assertEqual(len(frame), 3)
        self.assertTrue(report["first_and_relay_rows_in_same_schema"])

    def test_historical_cutoff_excludes_later_observations(self):
        observations, daily, market = fixture()
        later = observations.iloc[[0]].copy()
        later["signal_date"] = "2025-01-03"
        later_daily = daily.iloc[[0]].copy()
        later_daily["date"] = "2025-01-03"
        frame, _ = build(pd.concat([observations, later], ignore_index=True),
                         pd.concat([daily, later_daily], ignore_index=True), market,
                         end_date="2025-01-02")
        self.assertEqual(len(frame), 3)
