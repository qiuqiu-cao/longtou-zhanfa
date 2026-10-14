"""Synthetic ranker/clock fixtures; no market or forward performance evidence."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import theme_competition_matched_challengers as m

DATES = pd.bdate_range("2026-10-08", periods=61).strftime("%Y-%m-%d").tolist()
VALIDATION = DATES[60]
FIT_TIME = VALIDATION+"T09:00:00+08:00"


def fixture_day(date, index=61, labeled=False):
    rows = []
    for j in range(4):
        boards = 2 if j < 2 else 3
        events = [{"type": "THEME_LADDER_PRESENT_AT_CLOSE"}]
        if j == 0:
            events.append({"type": "CAPACITY_LEADER_BY_CLOSE_AMOUNT", "rank": 1})
        rows.append({"date": date, "code": f"60000{j+1}", "boards": boards,
            "signal_index": index, "baseline_raw_score": [.3, .6, .5, .7][j],
            "actual_one_word_daily": 0, "available_at": date+"T15:10:00+08:00",
            "eligible_for_leader_factor_training": True, "full_market_complete": True,
            "theme_factor_schema_status": "V150_COMPLETE", "same_theme_peer_count": 3,
            "same_theme_board_peer_count": 1, "higher_theme_peer_count": 2 if boards == 2 else 0,
            "events": json.dumps(events), "target_date": None, "label": j%2 if labeled else None,
            "label_valid": labeled, "label_available_at": None,
            "original_production_call": "UNCHANGED_FIXTURE"})
    return pd.DataFrame(rows)


def training_rows():
    parts = []
    for i, date in enumerate(DATES[:60]):
        rows = fixture_day(date, i+1, labeled=i < 59)
        rows["target_date"] = DATES[i+1]
        if i < 59:
            rows["label_available_at"] = DATES[i+1]+"T15:20:00+08:00"
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


class MatchedThemeChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = training_rows()
        with patch.object(m, "now", return_value=FIT_TIME):
            cls.fitted = m.MatchedThemeRankers().fit(cls.rows, VALIDATION)

    def test_matched_training_boundary_and_weights_contract(self):
        a = self.fitted.training_audit
        self.assertEqual(a["declared_rows"], 240)
        self.assertEqual(a["boundary_purged_rows"], 4)
        self.assertEqual(a["eligible_rows"], 236)
        self.assertEqual(a["training_dates"], 59)
        self.assertLess(a["max_label_date"], VALIDATION)
        self.assertTrue(a["same_training_indices_for_both_models"])
        self.assertFalse(a["probability_calibrated"])
        for model in self.fitted.models:
            self.assertEqual(model[-1].C, .25)

    def test_future_metadata_never_becomes_a_feature(self):
        rows = fixture_day(VALIDATION)
        changed = rows.copy()
        changed["label"] = [1, 1, 1, 1]
        changed["target_day_close"] = [10000]*4
        pd.testing.assert_frame_equal(m.features(rows), m.features(changed))
        with patch.object(m, "now", return_value=VALIDATION+"T15:20:00+08:00"):
            with self.assertRaisesRegex(ValueError, "future outcomes"):
                self.fitted.predict(changed)

    def test_unknown_theme_keeps_null_and_whole_day_fallback(self):
        rows = fixture_day(VALIDATION)
        rows.loc[0, "eligible_for_leader_factor_training"] = False
        x = m.features(rows)
        self.assertTrue(x.loc[0, m.THEME_FIELDS[1:]].isna().all())
        with patch.object(m, "now", return_value=VALIDATION+"T15:20:00+08:00"):
            result = self.fitted.predict(rows)
        self.assertEqual(len(result), 4)
        self.assertTrue(result.research_route.eq("DAY_FALLBACK_BASELINE_THEME_UNAVAILABLE").all())
        selected = [result.loc[result[k+"_selected"], "code"].tolist() for k in m.MODEL_IDS]
        self.assertEqual(selected[0], selected[1])
        self.assertEqual(selected[0], selected[2])
        self.assertTrue(result.original_production_call.eq("UNCHANGED_FIXTURE").all())

    def test_active_models_select_once_without_production_binary_changes(self):
        rows = fixture_day(VALIDATION)
        with patch.object(m, "now", return_value=VALIDATION+"T15:20:00+08:00"):
            result = self.fitted.predict(rows)
        for k in m.MODEL_IDS:
            self.assertEqual(result[k+"_selected"].sum(), 1)
        self.assertTrue(result.research_route.eq("MATCHED_RESEARCH_RANKERS").all())
        self.assertFalse(result.production_model_modified.any())
        self.assertTrue(result.original_production_call.eq("UNCHANGED_FIXTURE").all())

    def test_unknown_domain_or_baseline_keeps_all_rows_without_guessing(self):
        for field in ["actual_one_word_daily", "baseline_raw_score"]:
            with self.subTest(field=field):
                rows = fixture_day(VALIDATION)
                rows[field] = rows[field].astype(float)
                rows.loc[0, field] = np.nan
                with patch.object(m, "now", return_value=VALIDATION+"T15:20:00+08:00"):
                    result = self.fitted.predict(rows)
                self.assertEqual(len(result), 4)
                self.assertFalse(result[[k+"_selected" for k in m.MODEL_IDS]].any().any())

    def test_real_clock_and_completion_cutoff_enforced(self):
        rows = fixture_day(VALIDATION)
        with patch.object(m, "now", return_value=VALIDATION+"T17:00:00+08:00"):
            with self.assertRaisesRegex(ValueError, "actual signal-day clock"):
                self.fitted.predict(rows)
        with patch.object(m, "now", side_effect=[VALIDATION+"T16:29:59+08:00", VALIDATION+"T16:30:01+08:00"]):
            with self.assertRaisesRegex(ValueError, "complete shadow prediction"):
                self.fitted.predict(rows)

    def test_training_cannot_use_holdout_future_or_premature_labels(self):
        changed = self.rows.copy()
        changed.loc[0, "signal_index"] = 81
        with patch.object(m, "now", return_value=FIT_TIME):
            with self.assertRaisesRegex(ValueError, "validation/test"):
                m.MatchedThemeRankers().fit(changed, VALIDATION)
        for time in [VALIDATION+"T10:00:00+08:00", DATES[1]+"T14:59:00+08:00"]:
            changed = self.rows.copy()
            changed.loc[0, "label_available_at"] = time
            with patch.object(m, "now", return_value=FIT_TIME):
                with self.assertRaisesRegex(ValueError, "actually available"):
                    m.MatchedThemeRankers().fit(changed, VALIDATION)

    def test_no_refit_or_sliding_insufficient_training_window(self):
        fitted = copy.deepcopy(self.fitted)
        with self.assertRaisesRegex(ValueError, "fixed after"):
            fitted.fit(self.rows, VALIDATION)
        with patch.object(m, "now", return_value=FIT_TIME):
            with self.assertRaisesRegex(ValueError, "do not roll"):
                m.MatchedThemeRankers().fit(self.rows.iloc[:80].copy(), VALIDATION)

    def test_training_completion_crossing_boundary_refuses_models(self):
        ranker = m.MatchedThemeRankers()
        with patch.object(m, "now", side_effect=[FIT_TIME, VALIDATION+"T09:15:01+08:00"]):
            with self.assertRaisesRegex(ValueError, "complete model formation"):
                ranker.fit(self.rows, VALIDATION)
        self.assertIsNone(ranker.models)

    def test_full_market_gate_and_invalid_counts_fail_closed(self):
        rows = fixture_day(VALIDATION)
        rows.loc[0, "full_market_complete"] = False
        self.assertTrue(m.features(rows).loc[0, m.THEME_FIELDS[1:]].isna().all())
        rows = fixture_day(VALIDATION)
        rows.loc[0, "higher_theme_peer_count"] = 99
        with self.assertRaisesRegex(ValueError, "invalid rival counts"):
            m.features(rows)

    def test_registered_slot_block_intervals_are_deterministic(self):
        values = [1, np.nan, 0, 1, np.nan, 0, 1, 1]
        first = m.circular_block_interval(values, .975)
        self.assertEqual(first, m.circular_block_interval(values, .975))
        self.assertLessEqual(first[0], first[1])
        self.assertIsNone(m.circular_block_interval([np.nan]*60))


if __name__ == "__main__":
    unittest.main()
