import copy
import unittest

import pandas as pd

from build_specific_theme_roles_v150 import VERSION, build, validate_evidence


ASOF = "2026-09-17T20:00:00+08:00"


def evidence(**updates):
    group = {
        "theme_group_id": "pcb_material",
        "theme": "高速PCB基材",
        "theme_granularity": "specific_subtheme",
        "specificity_basis": "同一直接产品链和同一催化",
        "codes": ["600001", "000001"],
        "driver_type": "industry",
        "driver_summary": "上游材料价格与订单变化",
        "driver_source_url": "https://example.com/driver",
        "driver_first_available_at": "2026-09-17T14:00:00+08:00",
        "driver_claim_status": "verified_fact",
        "expectation_status": "partially_traded",
        "expectation_basis": "题材首日扩散但尚未一致加速",
        "expectation_assessed_at": "2026-09-17T15:10:00+08:00",
        "membership_source_url": "https://example.com/member",
        "membership_published_at": "2026-09-17T13:00:00+08:00",
        "source_tier": "A_PRIMARY",
    }
    group.update(updates)
    return {"schema_version": VERSION, "signal_date": "2026-09-17", "groups": [group]}


def pool():
    return pd.DataFrame([
        {"signal_date": "2026-09-17", "code": "600001", "name": "甲", "boards": 2,
         "pool_assignment": "CLOSING_LIMIT_UP", "amount": 20, "first_seal_time": 93000},
        {"signal_date": "2026-09-17", "code": "000001", "name": "乙", "boards": 1,
         "pool_assignment": "CLOSING_LIMIT_UP", "amount": 10, "first_seal_time": 93500},
    ])


class SpecificThemeRolesTests(unittest.TestCase):
    def test_verified_specific_group_builds_roles_and_positive_factor(self):
        result = build(pool(), evidence(), ASOF)
        self.assertEqual(len(result), 2)
        leader = result[result.code.eq("600001")].iloc[0]
        self.assertEqual(leader.theme_role, "THEME_HEIGHT_LEADER")
        self.assertTrue(leader.positive_catalyst_factor_eligible)
        self.assertEqual(leader.theme_granularity, "specific_subtheme")
        self.assertEqual(leader.theme_status, "verified_specific_point_in_time_schema")

    def test_broad_group_is_rejected_not_silently_downgraded(self):
        with self.assertRaises(ValueError):
            validate_evidence(evidence(theme_granularity="broad_concept"), ASOF)

    def test_future_catalyst_membership_or_assessment_rejected(self):
        for field in ("driver_first_available_at", "membership_published_at", "expectation_assessed_at"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    validate_evidence(evidence(**{field: "2026-09-18T09:00:00+08:00"}), ASOF)

    def test_narrative_rumor_pure_emotion_cannot_be_positive_factor(self):
        variants = [
            {"driver_claim_status": "market_narrative"},
            {"driver_claim_status": "rumor"},
            {"driver_type": "pure_emotion", "driver_claim_status": "market_narrative"},
            {"source_tier": "C_MARKET_REVIEW"},
            {"expectation_status": "fully_traded"},
        ]
        for update in variants:
            with self.subTest(update=update):
                result = build(pool(), evidence(**update), ASOF)
                self.assertFalse(result.positive_catalyst_factor_eligible.any())
                if (update.get("driver_claim_status") in {"market_narrative", "rumor"}
                        or update.get("source_tier") == "C_MARKET_REVIEW"):
                    self.assertFalse(
                        result.theme_status.eq("verified_specific_point_in_time_schema").any())

    def test_unknown_expectation_is_not_a_positive_factor(self):
        result = build(pool(), evidence(expectation_status="unknown"), ASOF)
        self.assertFalse(result.positive_catalyst_factor_eligible.any())
        self.assertEqual(result.iloc[0].theme_status,
                         "verified_specific_point_in_time_schema")

    def test_pure_emotion_cannot_be_mislabeled_verified_fact(self):
        with self.assertRaises(ValueError):
            validate_evidence(evidence(driver_type="pure_emotion", driver_claim_status="verified_fact"), ASOF)

    def test_primary_membership_cannot_be_duplicated_across_groups(self):
        item = evidence()
        second = copy.deepcopy(item["groups"][0])
        second["theme_group_id"] = "other"
        item["groups"].append(second)
        with self.assertRaises(ValueError):
            validate_evidence(item, ASOF)


if __name__ == "__main__":
    unittest.main()
