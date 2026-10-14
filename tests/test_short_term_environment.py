import copy
import unittest

from short_term_environment import METRICS, analyze


def fixture(minute, value):
    return {"scope_id": "synthetic-mainboard", "metrics": {
        key: {"value": value, "unit": unit, "sample_count": 20,
              "definition_id": key + "-fixed-v1", "cohort_id": "fixed-test-cohort",
              "observed_at": f"2026-09-04T09:{minute}:00+08:00",
              "available_at": f"2026-09-04T09:{minute}:00+08:00",
              "source": "SYNTHETIC_NOT_MARKET", "data_status": "verified"}
        for key, (_, _, unit) in METRICS.items()}}


class EnvironmentTests(unittest.TestCase):
    asof = "2026-09-04T09:35:00+08:00"

    def test_deterioration_is_not_buy_rejection_or_probability(self):
        result = analyze(fixture("35", .1), self.asof, fixture("30", .2))
        self.assertEqual(result["hypothesis"], "BROAD_DETERIORATION_HYPOTHESIS")
        self.assertIsNone(result["premium_probability"])
        self.assertNotIn("decision", result)

    def test_missing_not_zero(self):
        self.assertEqual(analyze(None, self.asof)["status"], "UNAVAILABLE")
        current = fixture("35", .1)
        current["metrics"].pop("relay_premium")
        self.assertEqual(analyze(current, self.asof, fixture("30", .2))["hypothesis"], "INSUFFICIENT_EVIDENCE")

    def test_delayed_inferred_stale_and_changed_cohort_do_not_confirm(self):
        for update in ({"data_status": "delayed"}, {"data_status": "inferred"},
                       {"cohort_id": "different"}, {"sample_count": 19},
                       {"observed_at": "2026-09-04T09:29:00+08:00"}):
            current, previous = fixture("35", .2), fixture("25", .1)
            current["metrics"]["relay_premium"].update(update)
            self.assertEqual(analyze(current, self.asof, previous)["hypothesis"], "INSUFFICIENT_EVIDENCE")

    def test_future_nonfinite_bad_unit_rejected(self):
        for update in ({"available_at": "2026-09-04T09:36:00+08:00"},
                       {"value": float("nan")}, {"unit": "yuan"}, {"sample_count": 0}):
            current = fixture("35", .2)
            current["metrics"]["relay_premium"].update(update)
            with self.assertRaises(ValueError):
                analyze(current, self.asof)

    def test_outcomes_do_not_change_analysis(self):
        current = fixture("35", .2)
        changed = copy.deepcopy(current)
        changed["next_day_return"] = -10
        self.assertEqual(analyze(current, self.asof), analyze(changed, self.asof))
