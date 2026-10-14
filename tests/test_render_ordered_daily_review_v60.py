import unittest

from render_ordered_daily_review_v60 import render


class OrderedReviewTests(unittest.TestCase):
    def test_required_sections_are_in_fixed_order(self):
        report = {
            "signal_date": "2026-09-08",
            "market": {"emotion_regime": "升温", "seal_rate": .67, "failed_rate": .33,
                       "down_limit_count": 0,
                       "previous_limit_feedback": {"mean_return": .03, "median_return": .02,
                                                   "promotion_count": 2, "observation_count": 10,
                                                   "promotion_rate": .2},
                       "failed_limit_entry_feedback": {"median_return": -.03,
                                                       "tail_10_return": -.06}},
            "pool_summary": {"closing_limit_mainboard": 2, "failed_board_mainboard": 1},
            "theme_summary": {"verified_candidate_count": 0, "closing_candidate_count": 2},
            "announcement_summary": {"active_hard_events": 0, "active_hard_candidate_codes": 0},
            "first_board_candidates": [], "relay_candidates": [],
            "roles": {"emotion_anchors": []},
        }
        text = render(report)
        labels = ["1. 数据质量", "2. 市场情绪", "3. 涨停梯队", "4. 题材竞争",
                  "5. 首板候选", "6. 连板候选", "7. 三个角色", "8. 综合首选",
                  "9. 次日执行条件", "10. 空仓条件"]
        positions = [text.index(label) for label in labels]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("DIAGNOSTIC_ONLY", text)


if __name__ == "__main__":
    unittest.main()
