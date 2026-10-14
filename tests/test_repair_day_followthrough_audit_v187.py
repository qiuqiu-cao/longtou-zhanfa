import unittest

import pandas as pd

from repair_day_followthrough_audit_v187 import build, prepare


class RepairDayFollowthroughAuditV187Tests(unittest.TestCase):
    def frame(self):
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=240)
        for index, date in enumerate(dates):
            repair_day = index % 2 == 0
            broad = (index // 2) % 2 == 0
            if repair_day:
                state = "修复"
            else:
                state = "升温" if broad else "退潮"
            rows.append({
                "date": date.strftime("%Y-%m-%d"), "data_status": "verified",
                "emotion_state": state,
                "previous_limit_median_return": .02 if broad else -.01,
                "promotion_mid": .40 if broad else .10,
                "failed_rate": .15 if broad else .45,
                "down_limit_count": 2 if broad else 10,
                "limit_count": 50 if broad else 20,
                "max_board": 5 if broad else 2,
            })
        return pd.DataFrame(rows)

    def test_signal_day_quality_uses_same_day_and_outcome_uses_next_day(self):
        prepared = prepare(self.frame())
        first = prepared.iloc[0]
        self.assertEqual(first.repair_quality_class, "BROAD_REPAIR")
        self.assertEqual(first.next_state, "升温")
        self.assertTrue(first.next_advance)
        self.assertFalse(first.next_relapse)

    def test_time_split_and_recent_repair_context_are_diagnostic_only(self):
        report, _ = build(self.frame(), bootstrap_draws=200)
        self.assertEqual(report["split"]["development_sessions"], 84)
        self.assertEqual(report["split"]["validation_sessions"], 36)
        self.assertTrue(report["data_contract"][
            "next_session_fields_used_only_as_labels"])
        self.assertFalse(report["governance"][
            "calibrated_next_relapse_probability_allowed"])
        self.assertFalse(report["governance"]["repair_score_entry_authorized"])

    def test_broad_and_narrow_repairs_are_not_collapsed(self):
        report, _ = build(self.frame(), bootstrap_draws=200)
        validation = report["summaries"]["validation"]
        self.assertLess(validation["BROAD_REPAIR"]["relapse_rate"],
                        validation["NARROW_REPAIR"]["relapse_rate"])
        self.assertGreater(validation["BROAD_REPAIR"]["advance_rate"],
                           validation["NARROW_REPAIR"]["advance_rate"])

    def test_missing_same_day_quality_is_not_filled_as_zero(self):
        frame = self.frame()
        frame.loc[0, "promotion_mid"] = None
        prepared = prepare(frame)
        self.assertNotIn(frame.loc[0, "date"],
                         prepared.date.dt.strftime("%Y-%m-%d").tolist())


if __name__ == "__main__":
    unittest.main()
