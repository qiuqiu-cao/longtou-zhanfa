import unittest

import numpy as np
import pandas as pd

from first_board_profit_ranker_v196 import (
    EMOTION_FEATURES, build_dataset, daily_top1_metrics, _eligible_profit_train,
)


def feature_row(date, code):
    row = {"signal_date": date, "code": code, "name": code}
    row.update({name: 1.0 for name in EMOTION_FEATURES})
    return row


class FirstBoardProfitRankerV196Test(unittest.TestCase):
    def test_queue_is_zero_but_missing_and_right_censor_are_unavailable(self):
        features = pd.DataFrame([
            feature_row("2026-01-05", "A"), feature_row("2026-01-05", "B"),
            feature_row("2026-01-05", "C"),
        ])
        outcomes = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "A",
             "fill_status": "UNVERIFIABLE_LIMIT_QUEUE", "exit_date_B_plus_1": None,
             "exit_date_B_plus_2": None, "net_return_B_plus_1": None,
             "net_return_B_plus_2": None, "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-05", "code": "B",
             "fill_status": "MISSING_B_BAR", "exit_date_B_plus_1": None,
             "exit_date_B_plus_2": None, "net_return_B_plus_1": None,
             "net_return_B_plus_2": None, "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-05", "code": "C",
             "fill_status": "FILLED_OPEN_PROXY", "exit_date_B_plus_1": "2026-01-07",
             "exit_date_B_plus_2": None, "net_return_B_plus_1": .02,
             "net_return_B_plus_2": None, "big_loss_B_plus_1": 0},
        ])
        rows = build_dataset(features, outcomes).set_index("code")
        self.assertEqual(rows.loc["A", "executable_positive_B_plus_2"], 0)
        self.assertTrue(np.isnan(rows.loc["B", "executable_positive_B_plus_2"]))
        self.assertTrue(np.isnan(rows.loc["C", "executable_positive_B_plus_2"]))

    def test_top1_queue_is_not_replaced_by_profitable_second_choice(self):
        rows = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "A", "score": .9,
             "profit_target_available": 1, "executable_positive_B_plus_2": 0,
             "fill_status": "UNVERIFIABLE_LIMIT_QUEUE", "net_return_B_plus_1": None,
             "net_return_B_plus_2": None, "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-05", "code": "B", "score": .8,
             "profit_target_available": 1, "executable_positive_B_plus_2": 1,
             "fill_status": "FILLED_OPEN_PROXY", "net_return_B_plus_1": .1,
             "net_return_B_plus_2": .1, "big_loss_B_plus_1": 0},
        ])
        selected, metrics = daily_top1_metrics(rows, "score")
        self.assertEqual(selected.iloc[0].code, "A")
        self.assertEqual(metrics["executable_positive_B_plus_2_count"], 0)
        self.assertEqual(metrics["filled_trades"], 0)

    def test_top1_missing_target_remains_unavailable_not_backfilled(self):
        rows = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "A", "score": .9,
             "profit_target_available": 0, "executable_positive_B_plus_2": np.nan,
             "fill_status": "MISSING_B_BAR", "net_return_B_plus_1": None,
             "net_return_B_plus_2": None, "big_loss_B_plus_1": None},
            {"signal_date": "2026-01-05", "code": "B", "score": .8,
             "profit_target_available": 1, "executable_positive_B_plus_2": 1,
             "fill_status": "FILLED_OPEN_PROXY", "net_return_B_plus_1": .1,
             "net_return_B_plus_2": .1, "big_loss_B_plus_1": 0},
        ])
        selected, metrics = daily_top1_metrics(rows, "score")
        self.assertEqual(selected.iloc[0].code, "A")
        self.assertEqual(metrics["target_unavailable_sessions"], 1)
        self.assertEqual(metrics["executable_positive_B_plus_2_count"], 0)

    def test_training_uses_only_outcomes_known_by_score_date(self):
        rows = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "A", "profit_target_available": 1,
             "exit_date_B_plus_2": "2026-01-08"},
            {"signal_date": "2026-01-06", "code": "B", "profit_target_available": 1,
             "exit_date_B_plus_2": "2026-01-09"},
        ])
        train = _eligible_profit_train(rows, "2026-01-08")
        self.assertEqual(train.code.tolist(), ["A"])


if __name__ == "__main__":
    unittest.main()
