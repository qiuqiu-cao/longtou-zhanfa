import unittest

import pandas as pd

from full_pool_proxy_walkforward_v17 import _detail_columns, _paired_proxy_delta, _window


class FullPoolProxyWalkForwardTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([
            {"signal_date": "2024-01-02", "exit_date_LABEL_ONLY": "2024-01-04",
             "episode_id": "old", "opportunity_proxy_label": 0, "board_group": "FIRST_BOARD"},
            {"signal_date": "2024-01-03", "exit_date_LABEL_ONLY": "2024-01-05",
             "episode_id": "old2", "opportunity_proxy_label": 1, "board_group": "CONSECUTIVE_BOARD"},
            {"signal_date": "2024-02-01", "exit_date_LABEL_ONLY": "2024-02-05",
             "episode_id": "new1", "opportunity_proxy_label": 1, "board_group": "FIRST_BOARD"},
            {"signal_date": "2024-02-02", "exit_date_LABEL_ONLY": "2024-02-06",
             "episode_id": "new2", "opportunity_proxy_label": 0, "board_group": "CONSECUTIVE_BOARD"},
        ])

    def test_window_purges_validation_episode_and_requires_both_roles(self):
        train, validation, error = _window(self.rows(), "2024-02-01", "2024-03-01")
        self.assertIsNone(error)
        self.assertEqual(set(train.episode_id), {"old", "old2"})
        self.assertEqual(len(validation), 2)

    def test_detail_columns_include_forward_path_when_available(self):
        columns = _detail_columns(pd.DataFrame(columns=[
            "code", "signal_date", "opportunity_proxy_label",
            "next_session_one_price_LABEL_ONLY", "next_session_touched_limit_LABEL_ONLY",
            "b2_close_mark_net_return_proxy", "b1_net_close_mark_LABEL_ONLY",
            "b2_net_close_mark_LABEL_ONLY", "b3_net_close_mark_LABEL_ONLY",
        ]))
        self.assertIn("b1_net_close_mark_LABEL_ONLY", columns)
        self.assertIn("b3_net_close_mark_LABEL_ONLY", columns)

    def test_paired_delta_uses_same_sessions(self):
        model = pd.DataFrame({"signal_date": ["a", "b", "c", "d", "e"],
                              "opportunity_proxy_label": [1, 1, 1, 0, 0]})
        baseline = pd.DataFrame({"signal_date": ["a", "b", "c", "d", "e"],
                                 "opportunity_proxy_label": [0, 0, 1, 0, 0]})
        result = _paired_proxy_delta(model, baseline, repetitions=100, seed=1)
        self.assertEqual(result["paired_sessions"], 5)
        self.assertAlmostEqual(result["difference"], .4)


if __name__ == "__main__":
    unittest.main()
