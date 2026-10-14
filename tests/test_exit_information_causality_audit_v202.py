import unittest

import pandas as pd

from exit_information_causality_audit_v202 import audit, summarize


class ExitInformationCausalityAuditV202Test(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            {"signal_date": "2026-01-05", "code": "000001", "name": "A",
             "B_plus_1_exit_status": "B_PLUS_1_OPEN_EXIT_PROXY",
             "B_plus_1_open_exit_net_return": -.10,
             "existing_B_plus_1_close_net_return": .05},
            {"signal_date": "2026-01-05", "code": "000002", "name": "B",
             "B_plus_1_exit_status": "B_PLUS_1_OPEN_EXIT_PROXY",
             "B_plus_1_open_exit_net_return": .05,
             "existing_B_plus_1_close_net_return": -.10},
            {"signal_date": "2026-01-05", "code": "000003", "name": "C",
             "B_plus_1_exit_status": "ONE_PRICE_DOWN_LIMIT_UNSELLABLE",
             "B_plus_1_open_exit_net_return": None,
             "existing_B_plus_1_close_net_return": -.12},
        ])

    def test_hindsight_timing_oracle_is_separated_from_fixed_rules(self):
        result = summarize(self.frame())
        self.assertAlmostEqual(
            result["fixed_precommitted_B_plus_1_open_proxy"]["mean_net_return"],
            -.025)
        self.assertAlmostEqual(
            result["fixed_B_plus_1_close_mark"]["mean_net_return"], -.025)
        self.assertAlmostEqual(
            result["hindsight_choose_better_open_or_close"]["mean_net_return"],
            .05)
        self.assertAlmostEqual(
            result["hindsight_timing_choice_mean_optimism_vs_better_fixed_rule"],
            .075)

    def test_same_open_conditional_exit_is_forbidden(self):
        top1 = pd.DataFrame([{
            "signal_date": "2026-01-05", "code": "000001",
            "ranker": "profit_emotion_probability"}])
        report = audit(self.frame(), top1, {
            "technical_complete_sequences": 0,
            "full_complete_sequences": 0,
        })
        self.assertFalse(report["exit_information_contract"][
            "official_open_can_condition_same_open_exit"])
        self.assertFalse(report["governance"][
            "hindsight_open_close_timing_oracle_allowed"])
        self.assertFalse(report["intraday_exit_readiness"][
            "post_open_exit_model_ready"])


if __name__ == "__main__":
    unittest.main()
