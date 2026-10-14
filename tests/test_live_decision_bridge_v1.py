import unittest

import pandas as pd

import live_decision_bridge_v1 as bridge


class LiveDecisionBridgeV1Tests(unittest.TestCase):
    @staticmethod
    def review(status="CASH", primary=None):
        return {
            "status": status, "signal_date": "2026-09-04",
            "market": {"emotion_regime": "修复"},
            "blocking_reasons": [] if status != "CASH" else ["NO_VALIDATED_THRESHOLD"],
            "first_board_candidates": [{
                "code": "600001", "name": "首板", "boards": 1, "board_group": "FIRST_BOARD",
                "eligibility_status": "VERIFIED_ELIGIBLE", "active_hard_risks": [],
            }],
            "relay_candidates": [{
                "code": "000002", "name": "连板", "boards": 2, "board_group": "CONSECUTIVE_BOARD",
                "eligibility_status": "VERIFIED_ELIGIBLE", "active_hard_risks": [],
            }],
            "roles": {"current_best_buy_opportunity": primary},
            "selection": {"conditional_primary": primary},
        }

    @staticmethod
    def quote_manifest(status="CONDITIONAL_PUBLIC_PRICE_FRESH"):
        return {
            "signal_date": "2026-09-04", "trade_date": "2026-09-07", "asof": "09:35",
            "status": status, "direct_entry_authorized": False,
            "market_breadth": {"status": "INFERRED_FROM_FRESH_PUBLIC_QUOTES", "decline_ratio": .40},
        }

    @staticmethod
    def quotes(at_limit=False):
        return pd.DataFrame([
            {"code": "600001", "quote_data_status": "VERIFIED_PUBLIC_QUOTE_FRESH",
             "quote_age_seconds": 4, "auction_gap_proxy": .03, "last": 10.3,
             "at_limit_price_proxy": at_limit, "public_level1_buyability": "UNAVAILABLE_NO_QUEUE" if at_limit else "PRICE_NOT_LOCKED_LIMIT_PROXY",
             "public_feed_decision": "WATCH_AT_LIMIT_BUYABILITY_UNKNOWN" if at_limit else "WATCH_PUBLIC_PRICE_ONLY"},
            {"code": "000002", "quote_data_status": "VERIFIED_PUBLIC_QUOTE_FRESH",
             "quote_age_seconds": 5, "auction_gap_proxy": .02, "last": 11.2,
             "at_limit_price_proxy": False, "public_level1_buyability": "PRICE_NOT_LOCKED_LIMIT_PROXY",
             "public_feed_decision": "WATCH_PUBLIC_PRICE_ONLY"},
        ])

    @staticmethod
    def minute_manifest():
        return {
            "signal_date": "2026-09-04", "trade_date": "2026-09-07", "asof": "09:35",
            "status": "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH", "direct_entry_authorized": False,
        }

    @staticmethod
    def measurements(risk=False):
        return pd.DataFrame([
            {"code": "600001", "minute_data_status": "VERIFIED_PUBLIC_MINUTE_BAR_FRESH",
             "close_to_session_vwap_proxy": .01, "vwap_slope_5m_proxy": .001,
             "failed_seal_twice_proxy": risk, "below_vwap_expanding_proxy": False,
             "reseal_minute_proxy": 4},
            {"code": "000002", "minute_data_status": "VERIFIED_PUBLIC_MINUTE_BAR_FRESH",
             "close_to_session_vwap_proxy": .02, "vwap_slope_5m_proxy": .002,
             "failed_seal_twice_proxy": False, "below_vwap_expanding_proxy": False,
             "reseal_minute_proxy": 3},
        ])

    def test_close_review_cash_cannot_be_upgraded_by_live_prices(self):
        result = bridge.build(self.review(), self.quote_manifest(), self.quotes())
        self.assertEqual(result["status"], "CASH")
        self.assertFalse(result["direct_entry_authorized"])
        self.assertIn("NO_FROZEN_CONDITIONAL_PRIMARY_FROM_CLOSE_REVIEW", result["blocking_reasons"])

    def test_fresh_public_price_keeps_conditional_primary_on_watch(self):
        primary = {"code": "600001", "name": "首板"}
        result = bridge.build(self.review("CONDITIONAL_PRIMARY", primary), self.quote_manifest(), self.quotes())
        self.assertEqual(result["status"], "WATCH_PUBLIC_PRICE_ONLY")
        self.assertEqual(result["primary_live_evidence"]["code"], "600001")
        self.assertFalse(result["direct_entry_authorized"])

    def test_public_minute_risk_can_cancel_but_never_enter(self):
        primary = {"code": "600001", "name": "首板"}
        result = bridge.build(
            self.review("CONDITIONAL_PRIMARY", primary), self.quote_manifest(), self.quotes(),
            self.minute_manifest(), self.measurements(risk=True),
        )
        self.assertEqual(result["status"], "CANCEL")
        self.assertIn("PRIMARY_PUBLIC_MINUTE_RISK_PROXY", result["blocking_reasons"])
        self.assertFalse(result["direct_entry_authorized"])

    def test_at_limit_price_requires_separate_buyability_evidence(self):
        primary = {"code": "600001", "name": "首板"}
        result = bridge.build(
            self.review("CONDITIONAL_PRIMARY", primary), self.quote_manifest(), self.quotes(at_limit=True),
        )
        self.assertEqual(result["status"], "WATCH_AT_LIMIT_BUYABILITY_UNKNOWN")
        self.assertIn("LEVEL2_QUEUE_AND_FILL_UNAVAILABLE", result["blocking_reasons"])

    def test_session_mismatch_is_rejected(self):
        manifest = self.quote_manifest()
        manifest["signal_date"] = "2026-09-03"
        with self.assertRaisesRegex(ValueError, "signal_date differ"):
            bridge.build(self.review(), manifest, self.quotes())

    def test_1000_below_vwap_is_logged_but_does_not_change_live_decision(self):
        primary = {"code": "600001", "name": "首板"}
        quote_manifest = self.quote_manifest(); quote_manifest["asof"] = "10:00"
        minute_manifest = self.minute_manifest(); minute_manifest["asof"] = "10:00"
        measurements = self.measurements(); measurements.loc[measurements.code.eq("600001"),
                                                               "close_to_session_vwap_proxy"] = -.001
        result = bridge.build(self.review("CONDITIONAL_PRIMARY", primary), quote_manifest,
                              self.quotes(), minute_manifest, measurements)
        self.assertEqual(result["status"], "WATCH_PUBLIC_EVIDENCE_ONLY")
        self.assertEqual(result["primary_live_evidence"]["absorption_shadow_gate_v139"],
                         "SHADOW_VETO_BELOW_VWAP")
        self.assertEqual(result["absorption_shadow_gate_live_effect"], "NONE_RESEARCH_ONLY")


if __name__ == "__main__":
    unittest.main()
