import copy
import unittest

from group_rank_governance_v153 import BASELINE, CHALLENGER, evaluate


def summary(rate, difference=None, ci=None):
    item = {
        "proxy_success_rate": rate,
        "path_proxy": {"b2": {"known": 100, "positive_rate": .5,
                                "large_loss_rate": .1, "mean_net_close_mark": .01}},
    }
    if difference is not None:
        item["paired_vs_group_baseline"] = {
            "paired_sessions": 120, "difference": difference, "bootstrap95": ci,
        }
    return item


def report(first=(.05, [.01, .09]), relay=(.04, [.005, .08])):
    return {
        "status": "EXPLORATORY_ONLY",
        "target_coverage": {"explicit_touch_field_used": True,
                            "no_touch_rows_retained_as_negative": True},
        "aggregate_selected_by_board_group": {
            CHALLENGER: {
                "FIRST_BOARD": summary(.20, *first),
                "CONSECUTIVE_BOARD": summary(.25, *relay),
            },
            BASELINE: {
                "FIRST_BOARD": summary(.15),
                "CONSECUTIVE_BOARD": summary(.21),
            },
        },
    }


class GroupRankGovernanceTests(unittest.TestCase):
    def test_both_groups_need_strict_positive_paired_interval(self):
        result = evaluate(report())
        self.assertTrue(result["first_board_probability_improvement_claim_allowed"])
        self.assertTrue(result["relay_probability_improvement_claim_allowed"])
        self.assertFalse(result["deployment_allowed"])

    def test_crossing_zero_fails_only_affected_group(self):
        result = evaluate(report(first=(.03, [-.01, .07])))
        self.assertFalse(result["first_board_probability_improvement_claim_allowed"])
        self.assertTrue(result["relay_probability_improvement_claim_allowed"])
        self.assertEqual(result["status"], "GROUP_RANK_UPLIFT_NOT_PROVEN")

    def test_old_leaky_target_report_is_rejected(self):
        item = report()
        item["target_coverage"]["no_touch_rows_retained_as_negative"] = False
        with self.assertRaises(ValueError):
            evaluate(item)


if __name__ == "__main__":
    unittest.main()
