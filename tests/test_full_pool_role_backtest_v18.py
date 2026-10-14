import unittest

import pandas as pd

from full_pool_role_backtest_v18 import (
    _metrics, _metrics_by_group, _path_metrics, _path_metrics_by_emotion,
    _path_metrics_by_group, _select_top,
)


class FullPoolRoleBacktestTests(unittest.TestCase):
    def test_roles_are_scored_against_their_declared_target(self):
        rows = pd.DataFrame([
            {"signal_date": "2025-01-02", "code": "000002", "boards": 2,
             "board_group": "CONSECUTIVE_BOARD", "score": .9, "target": 1},
            {"signal_date": "2025-01-02", "code": "000001", "boards": 1,
             "board_group": "FIRST_BOARD", "score": .8, "target": 0},
        ])
        selected = _select_top(rows, "score")
        result = _metrics(selected, "score", "target")
        self.assertEqual(result["selected_sessions"], 1)
        self.assertEqual(result["known_target"], 1)
        self.assertEqual(result["positive"], 1)
        self.assertEqual(len(result["positive_rate_wilson_95"]), 2)

    def test_path_metrics_preserve_unknowns_and_horizons(self):
        rows = pd.DataFrame([
            {"b1_net_close_mark_LABEL_ONLY": .02,
             "b2_net_close_mark_LABEL_ONLY": .01,
             "b3_net_close_mark_LABEL_ONLY": -.08},
            {"b1_net_close_mark_LABEL_ONLY": None,
             "b2_net_close_mark_LABEL_ONLY": .03,
             "b3_net_close_mark_LABEL_ONLY": .04},
        ])
        result = _path_metrics(rows)
        self.assertEqual(result["b1"]["known"], 1)
        self.assertEqual(result["b3"]["large_loss_rate"], 0.5)
        self.assertEqual(len(result["b3"]["positive_rate_wilson_95"]), 2)
        self.assertEqual(len(result["b3"]["large_loss_rate_wilson_95"]), 2)
        self.assertEqual(result["b2_b3_both_positive"]["known"], 2)
        self.assertEqual(result["b2_b3_both_positive"]["rate"], 0.5)

    def test_path_metrics_can_be_grouped_by_emotion_without_filling_missing(self):
        rows = pd.DataFrame([
            {"market_emotion_state": "升温", "b1_net_close_mark_LABEL_ONLY": .01},
            {"market_emotion_state": "退潮", "b1_net_close_mark_LABEL_ONLY": None},
        ])
        result = _path_metrics_by_emotion(rows)
        self.assertEqual(result["升温"]["b1"]["known"], 1)
        self.assertEqual(result["退潮"]["b1"]["known"], 0)

    def test_board_group_metrics_keep_first_and_consecutive_separate(self):
        rows = pd.DataFrame([
            {"board_group": "FIRST_BOARD", "score": .9, "target": 1,
             "b1_net_close_mark_LABEL_ONLY": .02},
            {"board_group": "CONSECUTIVE_BOARD", "score": .8, "target": 0,
             "b1_net_close_mark_LABEL_ONLY": -.03},
        ])
        self.assertEqual(_metrics_by_group(rows, "score", "target", "board_group")
                         ["FIRST_BOARD"]["positive"], 1)
        self.assertEqual(_path_metrics_by_group(rows, "board_group")
                         ["CONSECUTIVE_BOARD"]["b1"]["positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
