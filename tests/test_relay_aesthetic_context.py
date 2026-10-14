import copy
import unittest

import relay_aesthetic_context as a


def path(grid=True, word=False, drop=0, recovery=0, drawdown=0):
    return {"interval_minutes": 5, "grid_complete": grid,
            "actual_one_word_observed_OHLC": word,
            "features": {"sampled_close_drop_after_limit": drop,
                         "sampled_later_limit_close_after_drop": recovery,
                         "sampled_after_touch_drawdown_pct": drawdown},
            "provider_reported_seal": {"bomb_count": "1", "first_last_clock_gap_minutes": "24.5"}}


def row(value=None, peers=None):
    result = {"candidate_id": "2026-09-30:002242", "signal_date": "2026-09-30", "code": "002242",
              "name": "fixture", "boards": 3, "actual_one_word_daily": 0,
              "features": {}, "source_groups": {}}
    if value:
        result["features"]["sampled_grid_complete"] = float(value["grid_complete"])
        result["source_groups"]["sampled_path"] = {"value": value}
    if peers:
        result["features"]["sampled_current_peer_expected"] = 1.
        result["source_groups"]["sampled_peers"] = {"value": peers}
    return result


def comparison(complete=True, minimum=1, final=2):
    return {"peer_codes": ["600241"], "missing_peer_codes": [] if complete else ["600241"],
            "scope_complete": True, "complete_price_comparison": complete,
            "features": {"relative_min_pp": minimum, "relative_last_pp": final, "positive_bar_share": 1}}


class AestheticFactsTests(unittest.TestCase):
    def test_intrabar_drop_is_not_missed_pressure_free_board(self):
        result = a.describe(row(path(drawdown=-2.64)))
        self.assertEqual(result["price_path_state"], "INTRABAR_PRICE_PRESSURE_WITHOUT_SAMPLED_CLOSE_BREAK")
        self.assertEqual(result["price_path_facts"]["first_last_reported_gap_minutes"], 24.5)
        self.assertIsNone(result["price_path_facts"]["actual_reseal_verified"])

    def test_sampled_price_recovery_does_not_certify_absorption(self):
        result = a.describe(row(path(drop=1, recovery=1, drawdown=-5)))
        self.assertEqual(result["price_path_state"], "OBSERVED_PRICE_DROP_AND_LATER_RECOVERY_NOT_SEAL_ABSORPTION")
        self.assertFalse(result["price_path_facts"]["absorption_verified"])
        self.assertIsNone(result["aesthetic_score_or_probability"])

    def test_one_word_leadership_is_not_tested_turnover_or_entry(self):
        result = a.describe(row(path(word=True)))
        self.assertEqual(result["price_path_state"], "ONE_WORD_OBSERVED_REOPEN_PRESSURE_NOT_TESTED")
        self.assertEqual(result["trade_entry"], "NOT_CREATED_BY_DESCRIPTOR")

    def test_absence_in_incomplete_sampling_stays_unknown(self):
        result = a.describe(row(path(grid=False, drop=None, recovery=None, drawdown=None)))
        self.assertEqual(result["price_path_state"], "UNKNOWN_INCOMPLETE_SAMPLED_GRID")

    def test_unqualified_raw_group_cannot_supply_price_interpretation(self):
        sample = row(path(drop=1, recovery=1))
        sample["features"]["sampled_grid_complete"] = None
        self.assertEqual(a.describe(sample)["price_path_state"], "UNKNOWN_NO_QUALIFIED_INTRADAY_PRICE_PATH")

    def test_current_survivors_do_not_replace_prior_rivals(self):
        peers = {"current": comparison(), "prior": comparison(False)}
        result = a.describe(row(path(), peers))
        self.assertTrue(result["current_survivor_comparison"]["complete_comparison"])
        self.assertEqual(result["prior_rival_comparison"]["state"], "UNKNOWN")
        self.assertIn("COMPLETE_ORIGINAL_RIVAL_COMPARISON_UNAVAILABLE", result["missing_evidence"])

    def test_complete_prior_price_lead_is_not_active_displacement(self):
        result = a.describe(row(path(word=True), {"current": comparison(False), "prior": comparison()}))
        prior = result["prior_rival_comparison"]
        self.assertEqual(prior["state"], "RELATIVE_PRICE_AHEAD_AT_EVERY_OBSERVED_POINT")
        self.assertFalse(prior["active_displacement_or_core_identity_confirmed"])

    def test_relative_mixed_final_advantage_is_not_persistent_lead(self):
        result = a.describe(row(path(), {"current": comparison(minimum=-2, final=1), "prior": comparison(False)}))
        self.assertEqual(result["current_survivor_comparison"]["state"], "RELATIVE_PRICE_RECOVERED_OR_MIXED_WITH_FINAL_ADVANTAGE")

    def test_tiny_positive_rounding_difference_is_not_dominance(self):
        result = a.describe(row(path(), {"current": comparison(minimum=.01, final=.01), "prior": comparison(False)}))
        self.assertTrue(result["current_survivor_comparison"]["small_positive_gap_may_reflect_limit_price_rounding_not_competitive_dominance"])
        self.assertFalse(result["automatic_leader_identity"])


if __name__ == "__main__": unittest.main()
