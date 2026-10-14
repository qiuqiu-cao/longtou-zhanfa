import unittest

import pandas as pd

from breadth_quality_emotion_v178 import build


class BreadthQualityEmotionV178Tests(unittest.TestCase):
    def frame(self):
        rows = []
        for index, values in enumerate([
            (10, 2, 2, .20, .20, .01, 1, 0, 0),
            (15, 3, 3, .30, .30, .02, 2, 0, 0),
            (20, 4, 4, .20, .40, .03, 3, 1, 0),
            (23, 3, 3, .35, .10, -.01, 4, 0, 1),
            (30, 6, 5, .25, .50, .04, 1, 0, 0),
            (35, 7, 6, .15, .60, .05, 0, 0, 0),
        ]):
            limit_count, relay, height, failed, promo, median, down, div, retreat = values
            rows.append({
                "date": f"2025-01-{index + 2:02d}", "data_status": "verified",
                "limit_count": limit_count, "consecutive_count": relay,
                "max_board": height, "failed_rate": failed,
                "promotion_mid": promo,
                "previous_limit_median_return": median,
                "down_limit_count": down, "state_divergence": div,
                "state_retreat": retreat,
            })
        return pd.DataFrame(rows)

    def test_classifies_failed_rate_direction_not_breadth_alone(self):
        rows, report = build(self.frame(), "2025-12-31")
        states = dict(zip(rows.signal_date, rows.breadth_quality))
        self.assertEqual(states["2025-01-03"], "FRAGILE_EXPANSION")
        self.assertEqual(states["2025-01-04"], "CLEAN_EXPANSION")
        self.assertFalse(report["governance"][
            "limit_count_growth_alone_sufficient_for_repair"])
        self.assertIn("chronological_split_bootstrap_comparisons", report)

    def test_targets_are_from_immediately_next_session(self):
        rows, _ = build(self.frame(), "2025-12-31")
        row = rows.loc[rows.signal_date.eq("2025-01-03")].iloc[0]
        self.assertEqual(row.next_session, "2025-01-04")
        self.assertEqual(row.next_adverse_state, 1)
        self.assertAlmostEqual(row.next_failed_rate, .20)
        self.assertAlmostEqual(row.next_previous_limit_median_return, .03)

    def test_requires_verified_previous_and_next_sessions(self):
        frame = self.frame()
        frame.loc[2, "data_status"] = "sparse"
        rows, _ = build(frame, "2025-12-31")
        self.assertNotIn("2025-01-03", set(rows.signal_date))
        self.assertNotIn("2025-01-04", set(rows.signal_date))
        self.assertNotIn("2025-01-05", set(rows.signal_date))

    def test_rejects_duplicate_dates_and_never_authorizes_entry(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "valid and unique"):
            build(pd.concat([frame, frame.iloc[[0]]]), "2025-12-31")
        _, report = build(frame, "2025-12-31")
        self.assertFalse(report["governance"]["direct_stock_reranking_allowed"])
        self.assertFalse(report["governance"]["entry_authorized"])


if __name__ == "__main__":
    unittest.main()
