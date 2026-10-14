import unittest

from full_pool_emotion_value_audit import audit


class FullPoolEmotionValueAuditTests(unittest.TestCase):
    def reports(self, active_deltas=(.1, .1, .1)):
        rank = {
            "status": "EXPLORATORY_ONLY", "target": "proxy",
            "groups": {
                "identity": {"all_candidates": {"candidate_auc": .60, "candidate_brier": .10},
                             "daily_top1": {"proxy_success_rate": .30}},
                "identity_plus_emotion": {"all_candidates": {"candidate_auc": .62, "candidate_brier": .09},
                                          "daily_top1": {"proxy_success_rate": .35}},
            },
        }
        windows = {}
        for name, delta in zip(("calibration", "validation", "reused_research_evaluation"), active_deltas):
            windows[name] = {
                "unconditional_top1": {"proxy_precision": .3},
                "emotion_active_top1": {"proxy_precision": .3 + delta, "selections": 30},
            }
        return rank, {"target": "proxy", "windows": windows}

    def test_accepts_only_stable_additive_and_active_uplift(self):
        result = audit(*self.reports())
        self.assertEqual(result["status"], "EMOTION_POSITIVE_FACTOR_ACCEPTED")
        self.assertTrue(result["positive_buy_score_authorized"])

    def test_one_negative_time_window_rejects_positive_bonus(self):
        result = audit(*self.reports(active_deltas=(.1, -.01, .1)))
        self.assertEqual(result["status"], "EMOTION_POSITIVE_FACTOR_REJECTED")
        self.assertTrue(result["risk_context_use_only"])

    def test_rank_degradation_rejects_even_if_active_filter_helps(self):
        rank, selective = self.reports()
        rank["groups"]["identity_plus_emotion"]["all_candidates"]["candidate_auc"] = .59
        result = audit(rank, selective)
        self.assertFalse(result["positive_buy_score_authorized"])

    def test_target_mismatch_is_rejected(self):
        rank, selective = self.reports()
        selective["target"] = "other"
        with self.assertRaises(ValueError):
            audit(rank, selective)


if __name__ == "__main__":
    unittest.main()
