import unittest

import pandas as pd

from first_board_big_loss_veto_v199 import (
    eligible_risk_train, fixed_veto_metrics, risk_metrics,
)


class FirstBoardBigLossVetoV199Test(unittest.TestCase):
    def test_training_uses_only_labels_known_by_score_close(self):
        rows = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "A",
             "big_loss_target_available": 1, "big_loss_target": 1,
             "exit_date_B_plus_1": "2026-01-07"},
            {"signal_date": "2026-01-06", "code": "B",
             "big_loss_target_available": 1, "big_loss_target": 0,
             "exit_date_B_plus_1": "2026-01-08"},
        ])
        train = eligible_risk_train(rows, "2026-01-07")
        self.assertEqual(train.code.tolist(), ["A"])

    def test_veto_abstains_to_cash_without_second_choice(self):
        frozen = pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "A", "risk": .8,
             "fill_status": "FILLED_OPEN_PROXY", "big_loss_B_plus_1": 1,
             "net_return_B_plus_2": -.2},
            {"signal_date": "2026-01-06", "code": "B", "risk": .1,
             "fill_status": "FILLED_OPEN_PROXY", "big_loss_B_plus_1": 0,
             "net_return_B_plus_2": .1},
        ])
        result = fixed_veto_metrics(frozen, "risk", thresholds=(.2,))[0]
        self.assertEqual(result["frozen_decisions"], 2)
        self.assertEqual(result["retained_without_replacement"], 1)
        self.assertEqual(result["abstained_to_cash"], 1)
        self.assertEqual(result["B_plus_1_big_loss_rate"], 0)

    def test_risk_metrics_separate_discrimination_and_calibration(self):
        rows = pd.DataFrame([
            {"big_loss_target_available": 1, "big_loss_target": 0, "risk": .1},
            {"big_loss_target_available": 1, "big_loss_target": 0, "risk": .2},
            {"big_loss_target_available": 1, "big_loss_target": 1, "risk": .9},
        ])
        result = risk_metrics(rows, "risk")
        self.assertEqual(result["known_filled_rows"], 3)
        self.assertEqual(result["auc"], 1)
        self.assertAlmostEqual(result["observed_big_loss_rate"], 1 / 3)
        self.assertIn("brier", result)


if __name__ == "__main__":
    unittest.main()
