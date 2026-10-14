import unittest

import pandas as pd

from repair_layer_transmission_audit_v188 import build, prepare


class RepairLayerTransmissionAuditV188Tests(unittest.TestCase):
    def frame(self):
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=300)
        for index, date in enumerate(dates):
            repair = index % 2 == 0
            cycle = (index // 2) % 4
            synchronized = cycle == 0
            if repair:
                flags = dict(state_repair=1, state_divergence=0,
                             state_retreat=0, state_start=0,
                             state_warming=0, state_acceleration=0,
                             state_consensus=0)
            elif synchronized:
                flags = dict(state_repair=0, state_divergence=0,
                             state_retreat=0, state_start=0,
                             state_warming=1, state_acceleration=0,
                             state_consensus=0)
            else:
                flags = dict(state_repair=0, state_divergence=0,
                             state_retreat=1, state_start=0,
                             state_warming=0, state_acceleration=0,
                             state_consensus=0)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "data_status": "verified", **flags,
                "promotion_1to2": .40 if synchronized else .10,
                "promotion_mid": .60 if synchronized else .20,
                "space_feedback": 1 if synchronized else 0,
                "previous_limit_median_return": .02 if synchronized else -.01,
                "failed_rate": .15 if synchronized else .45,
                "down_limit_count": 2 if synchronized else 10,
            })
        return pd.DataFrame(rows)

    def test_same_day_layers_and_next_day_target_are_separate(self):
        prepared = prepare(self.frame())
        first = prepared.iloc[0]
        self.assertEqual(first["date"], pd.Timestamp("2024-01-02"))
        self.assertEqual(first["next_date"], pd.Timestamp("2024-01-03"))
        self.assertFalse(first["next_relapse"])

    def test_thresholds_are_frozen_from_development_only(self):
        base, _ = build(self.frame(), bootstrap_draws=100)
        changed = self.frame()
        repair_indices = changed.index[changed.state_repair.eq(1)]
        changed.loc[repair_indices[110:], "promotion_mid"] = .99
        second, _ = build(changed, bootstrap_draws=100)
        self.assertEqual(base["frozen_development_thresholds"],
                         second["frozen_development_thresholds"])

    def test_governance_remains_diagnostic(self):
        report, observations = build(self.frame(), bootstrap_draws=100)
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")
        self.assertGreaterEqual(len(observations), 100)
        self.assertIsInstance(report[
            "synchronized_repair_lower_relapse_stable_across_time"], bool)
        self.assertIsInstance(report[
            "middle_confirmation_lower_relapse_stable_across_time"], bool)
        governance = report["governance"]
        self.assertTrue(governance[
            "historical_repair_layer_scenario_context_allowed"])
        self.assertFalse(governance[
            "calibrated_layer_followthrough_probability_allowed"])
        self.assertFalse(governance["repair_layer_entry_authorized"])

    def test_missing_same_day_layer_is_excluded_not_zero_filled(self):
        frame = self.frame()
        frame.loc[0, "promotion_mid"] = None
        prepared = prepare(frame)
        self.assertNotIn(pd.Timestamp("2024-01-02"), set(prepared.date))


if __name__ == "__main__":
    unittest.main()
