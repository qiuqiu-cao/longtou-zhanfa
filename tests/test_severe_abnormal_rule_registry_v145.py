import unittest

from severe_abnormal_rule_registry_v145 import (
    FORMULA,
    exchange_for_mainboard_code,
    rule_for,
    verified_rule_for_code,
)


class SevereAbnormalRuleRegistryTests(unittest.TestCase):
    def test_exchange_and_benchmark_mapping(self):
        self.assertEqual(exchange_for_mainboard_code("600000"), "SSE")
        self.assertEqual(exchange_for_mainboard_code("002001"), "SZSE")
        self.assertEqual(
            verified_rule_for_code("603001", "2026-09-17")["benchmark_code"],
            "sh000002",
        )
        self.assertEqual(
            verified_rule_for_code("000001", "2026-09-17")["benchmark_code"],
            "sz399107",
        )

    def test_non_mainboard_codes_fail_closed(self):
        for code in ("300001", "688001", "920001", "abc"):
            with self.assertRaises(ValueError):
                exchange_for_mainboard_code(code)

    def test_version_boundary_is_explicit(self):
        self.assertEqual(rule_for("SSE", "2026-07-05").rule_id, "SSE_MAIN_20230217")
        self.assertEqual(rule_for("SSE", "2026-07-06").rule_id, "SSE_MAIN_20260706")
        self.assertEqual(rule_for("SZSE", "2026-07-06").rule_id, "SZSE_MAIN_20260706")

    def test_unsupported_history_is_not_guessed(self):
        with self.assertRaises(ValueError):
            rule_for("SSE", "2023-02-16")

    def test_rule_contract_contains_official_mechanics(self):
        rule = rule_for("SZSE", "2026-09-17")
        self.assertEqual(rule.positive_10d_threshold_pct, 100)
        self.assertEqual(rule.positive_30d_threshold_pct, 200)
        self.assertEqual(rule.same_direction_abnormal_count_10d, 4)
        self.assertEqual(rule.formula, FORMULA)
        self.assertIn("exchange_publication", rule.reset_rule)
        self.assertEqual(rule.no_limit_session_rule, "exclude_sessions_without_price_limit")
        self.assertEqual(rule.verification_status, "verified")


if __name__ == "__main__":
    unittest.main()
