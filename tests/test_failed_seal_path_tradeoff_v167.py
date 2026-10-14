import unittest

import pandas as pd

from failed_seal_path_tradeoff_v167 import audit, classify_path


class FailedSealPathTradeoffV167Tests(unittest.TestCase):
    def inputs(self):
        features, outcomes, keys = [], [], []
        for day in range(20):
            date = (pd.Timestamp("2026-01-01") + pd.offsets.BDay(day)).strftime("%Y-%m-%d")
            for index, (failed, span) in enumerate(((0, 0), (1, 10), (3, 50))):
                code = f"{day:03d}{index:03d}"
                group_good = index == 1
                features.append({
                    "signal_date": date, "code": code,
                    "failed_seal_count": failed, "last_minus_first_minute": span,
                })
                outcomes.append({
                    "signal_date": date, "code": code,
                    "promotion_label": int(group_good),
                    "fill_status": "FILLED_OPEN_PROXY",
                    "net_return_B_plus_2": .03 if group_good else -.03,
                    "big_loss_B_plus_1": 0 if group_good else 1,
                })
                keys.append({"signal_date": date, "code": code})
        return pd.DataFrame(features), pd.DataFrame(outcomes), pd.DataFrame(keys)

    def test_path_classification(self):
        self.assertEqual(classify_path(0, 0), "LOCKED_NO_REOPEN")
        self.assertEqual(classify_path(1, 30), "QUICK_REPAIR")
        self.assertEqual(classify_path(1, 31), "PROLONGED_CONTEST")
        self.assertEqual(classify_path(None, 3), "UNAVAILABLE")

    def test_small_sample_cannot_pass_even_with_good_direction(self):
        _, report = audit(*self.inputs(), bootstrap_repetitions=100)
        self.assertGreater(
            report["quick_repair_minus_prolonged_contest"]["promotion_rate"], 0)
        self.assertFalse(
            report["governance"]["failed_seal_path_positive_ranking_factor_allowed"])
        self.assertFalse(
            report["governance"]["raw_failed_seal_count_monotonic_rule_allowed"])

    def test_duplicate_keys_are_rejected(self):
        features, outcomes, keys = self.inputs()
        keys = pd.concat([keys, keys.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            audit(features, outcomes, keys, bootstrap_repetitions=10)

    def test_missing_b_bar_is_not_counted_as_execution_failure(self):
        features, outcomes, keys = self.inputs()
        mask = outcomes.signal_date.eq(outcomes.signal_date.max())
        outcomes.loc[mask, "fill_status"] = "MISSING_B_BAR"
        outcomes.loc[mask, ["net_return_B_plus_2", "big_loss_B_plus_1"]] = None
        _, report = audit(features, outcomes, keys, bootstrap_repetitions=20)
        self.assertEqual(report["missing_execution_evidence_rows"], int(mask.sum()))
        self.assertEqual(report["complete_execution_sessions"], 19)
        self.assertFalse(report["governance"]["missing_bar_is_unfilled_trade"])


if __name__ == "__main__":
    unittest.main()
