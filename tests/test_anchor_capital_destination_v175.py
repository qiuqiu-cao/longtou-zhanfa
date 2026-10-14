import unittest

import pandas as pd

from anchor_capital_destination_v175 import build


class AnchorCapitalDestinationV175Tests(unittest.TestCase):
    def features(self):
        rows = []
        layouts = {
            "2024-01-02": [("600001", 3), ("600002", 2), ("000001", 1)],
            "2024-01-03": [("600003", 3), ("600004", 2), ("000002", 1)],
            "2024-01-04": [("600005", 3), ("600006", 2), ("000003", 1)],
            "2024-01-05": [("600007", 3), ("600008", 2), ("000004", 1)],
        }
        for date, values in layouts.items():
            for code, boards in values:
                rows.append({"signal_date": date, "code": code, "boards": boards,
                             "start_known": True,
                             "feature_status": "historical_reconstruction_not_forward_archive"})
        return pd.DataFrame(rows)

    def outcomes(self):
        promotion = {
            # Anchor fails, lower relay promotes.
            "600001": 0, "600002": 1, "000001": 0,
            # Anchor fails, first board promotes.
            "600003": 0, "600004": 0, "000002": 1,
            # Anchor advances; lower outcomes do not redefine the destination.
            "600005": 1, "600006": 0, "000003": 1,
            # Anchor fails and no prior inventory member promotes.
            "600007": 0, "600008": 0, "000004": 0,
        }
        rows = []
        for row in self.features().itertuples():
            rows.append({"signal_date": row.signal_date, "code": row.code,
                         "promotion_LABEL_ONLY": promotion[row.code]})
        return pd.DataFrame(rows)

    def market(self):
        return pd.DataFrame([
            {"date": f"2024-01-0{day}", "data_status": "verified",
             "state_divergence": int(day in {4, 6}),
             "state_retreat": 0, "failed_rate": day / 100,
             "previous_limit_median_return": (day - 4) / 100}
            for day in range(2, 7)
        ])

    def test_destination_lanes_are_mutually_exclusive_and_anchor_is_separate(self):
        rows, report = build(self.features(), self.outcomes(), self.market(),
                             "2024-01-10")
        labels = rows.set_index("signal_date").capital_destination_label_only.to_dict()
        self.assertEqual(labels["2024-01-02"],
                         "ANCHOR_FAILED_LOWER_RELAY_RATE_DOMINANT")
        self.assertEqual(labels["2024-01-03"],
                         "ANCHOR_FAILED_FIRST_BOARD_RATE_DOMINANT")
        self.assertEqual(labels["2024-01-04"], "ANCHOR_ADVANCED")
        self.assertEqual(labels["2024-01-05"], "ANCHOR_FAILED_NO_INTERNAL_PROMOTION")
        self.assertEqual(report["anchor_failure_sessions"], 3)
        self.assertEqual(report["anchor_advance_sessions"], 1)

    def test_realized_destination_is_never_a_prior_close_feature_or_entry(self):
        _, report = build(self.features(), self.outcomes(), self.market(),
                          "2024-01-10")
        governance = report["governance"]
        self.assertFalse(governance["anchor_failure_alone_implies_retreat"])
        self.assertFalse(governance[
            "same_session_destination_as_signal_date_feature_allowed"])
        self.assertFalse(governance["automatic_high_to_low_entry_allowed"])
        self.assertFalse(governance["stock_reranking_allowed"])

    def test_incomplete_promotion_session_is_excluded_not_filled_with_zero(self):
        outcomes = self.outcomes()
        outcomes.loc[outcomes.code.eq("600004"), "promotion_LABEL_ONLY"] = None
        rows, _ = build(self.features(), outcomes, self.market(), "2024-01-10")
        self.assertNotIn("2024-01-03", set(rows.signal_date))

    def test_duplicate_point_in_time_key_is_rejected(self):
        features = pd.concat([self.features(), self.features().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            build(features, self.outcomes(), self.market(), "2024-01-10")


if __name__ == "__main__":
    unittest.main()
