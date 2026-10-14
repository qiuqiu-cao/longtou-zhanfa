import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from next_session_expectation_contract_v271 import (
    validate_plan, evaluate_stage, band_result, register, load_registered, verify_fill, settle,
)
from leader_identity_evidence_v273 import judge


def plan():
    return {
        "signal_date": "2026-09-21", "target_date": "2026-09-22",
        "calendar": {"status": "verified", "source": "TEST_FIXTURE", "sessions": ["2026-09-21", "2026-09-22", "2026-09-23"]},
        "evidence_cutoff": "2026-09-21T18:00:00+08:00", "model_version": "TEST_ONLY",
        "input_hash": "a"*64, "denominator_status": "verified_complete",
        "market": {"primary": "RISK_ON", "probability": None, "opposing_evidence": ["TEST_ONLY"], "invalidation": ["TEST_ONLY"]},
        "primary_candidate": "600001", "highest_accept_price": 10,
        "candidates": [{"code": "600001", "boards": 2, "identity": "LEADER_SEED", "promotion_call": "PROMOTE",
                        "identity_evidence": ["TEST_ONLY"], "identity_invalidation": ["TEST_ONLY"]}],
        "bands": [{"stage": s, "metric": "failed_rate", "unit": "ratio", "scope": "MAINBOARD",
                   "direction": "LOWER_BETTER", "low": .2, "high": .3, "basis": "TEST_ONLY", "hard_veto": True}
                  for s in ["AUCTION", "OPEN_0935", "OPEN_1000"]],
        "evidence": [{"observed_at": "2026-09-21T15:00:00+08:00", "available_at": "2026-09-21T17:00:00+08:00",
                      "source": "TEST_ONLY", "status": "verified"}],
    }


def observation(value=.25, stage="09:25:00", **updates):
    row = {"metric": "failed_rate", "scope": "MAINBOARD", "unit": "ratio", "value": value,
           "status": "verified", "source": "TEST_ONLY", "observed_at": f"2026-09-22T{stage}+08:00",
           "available_at": f"2026-09-22T{stage}+08:00"}
    row.update(updates)
    return row


class TemporalContractTests(unittest.TestCase):
    def test_eod_pending_and_late_replay(self):
        first = validate_plan(plan(), "2026-09-21T18:01:00+08:00")
        self.assertEqual(first["forecast_status"], "FORWARD_REGISTERED")
        self.assertEqual(first["execution_state"], "PENDING_OPEN_CONFIRMATION")
        self.assertEqual(validate_plan(plan(), "2026-09-22T18:49:00+08:00")["forecast_status"], "HISTORICAL_REPLAY")

    def test_wrong_successor_rejected(self):
        p = plan(); p["target_date"] = "2026-09-23"
        with self.assertRaises(ValueError): validate_plan(p, "2026-09-21T18:01:00+08:00")

    def test_unknown_calendar_and_probability_rejected(self):
        p = plan(); p["calendar"]["status"] = "inferred"
        with self.assertRaises(ValueError): validate_plan(p, "2026-09-21T18:01:00+08:00")
        p = plan(); p["market"]["probability"] = .8
        with self.assertRaises(ValueError): validate_plan(p, "2026-09-21T18:01:00+08:00")

    def test_future_evidence_rejected(self):
        p = plan(); p["evidence"][0]["available_at"] = "2026-09-22T09:25:00+08:00"
        with self.assertRaises(ValueError): validate_plan(p, "2026-09-21T18:01:00+08:00")

    def test_lower_bad_metric_is_exceeds_and_boundary_meets(self):
        b = plan()["bands"][0]
        self.assertEqual(band_result(b, observation(.1)), "EXCEEDS")
        self.assertEqual(band_result(b, observation(.3)), "MEETS")
        self.assertEqual(band_result(b, observation(.4)), "MISSES")

    def test_missing_not_negative_and_deterministic(self):
        a = evaluate_stage(plan(), "AUCTION", [], "2026-09-22T09:26:00+08:00")
        b = evaluate_stage(plan(), "AUCTION", [], "2026-09-22T09:26:00+08:00")
        self.assertEqual(a, b)
        self.assertEqual(a["expectation_state"], "UNAVAILABLE")
        self.assertEqual(a["execution_state"], "WATCH_DATA_MISSING")

    def test_conflicting_observations_do_not_force_positive(self):
        p = plan(); p["bands"][0]["hard_veto"] = False
        extra = deepcopy(p["bands"][0]); extra["metric"] = "breadth"; extra["direction"] = "HIGHER_BETTER"
        p["bands"].append(extra)
        r = evaluate_stage(p, "AUCTION", [observation(.1), observation(.1, metric="breadth")], "2026-09-22T09:26:00+08:00")
        self.assertEqual(r["expectation_state"], "CONFLICTING")
        self.assertEqual(r["execution_state"], "WATCH_CONFLICTING")

    def test_post_stage_information_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_stage(plan(), "AUCTION", [observation(stage="09:35:00")], "2026-09-22T09:36:00+08:00")

    def test_hard_veto_latched(self):
        first = evaluate_stage(plan(), "AUCTION", [observation(.5)], "2026-09-22T09:26:00+08:00")
        second = evaluate_stage(plan(), "OPEN_0935", [observation(.1, stage="09:35:00")], "2026-09-22T09:36:00+08:00", previous=first)
        self.assertEqual(second["execution_state"], "CANCELLED")
        self.assertFalse(second["automatic_replacement_allowed"])

    def test_no_same_open_hindsight_fill(self):
        r = evaluate_stage(plan(), "OPEN_0935", [observation(stage="09:35:00")], "2026-09-22T09:36:00+08:00")
        f = {"code": "600001", "price": 9.9, "status": "broker_confirmed", "source": "TEST_ONLY", "filled_at": "2026-09-22T09:30:00+08:00"}
        with self.assertRaises(ValueError): verify_fill(r, f)
        f["filled_at"] = "2026-09-22T09:36:01+08:00"
        self.assertTrue(verify_fill(r, f))
        f["price"] = 10.1
        with self.assertRaises(ValueError): verify_fill(r, f)

    def test_immutable_hash_detects_edits(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = register(plan(), d, "2026-09-21T18:01:00+08:00")
            load_registered(p)
            value = json.loads(Path(p).read_text(encoding="utf-8")); value["plan"]["market"]["primary"] = "RISK_OFF"
            Path(p).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError): load_registered(p)

    def test_forward_primary_cannot_be_cherry_picked_by_refreezing(self):
        with tempfile.TemporaryDirectory() as d:
            register(plan(), d, "2026-09-21T18:01:00+08:00")
            p = plan(); p["market"]["primary"] = "RISK_OFF"
            with self.assertRaises(ValueError): register(p, d, "2026-09-21T19:01:00+08:00")

    def test_settlement_keeps_failures_and_rejects_wrong_day(self):
        with tempfile.TemporaryDirectory() as d:
            _, registered = register(plan(), d, "2026-09-21T18:01:00+08:00")
            source = {"status": "verified", "source": "TEST_ONLY", "observed_at": "2026-09-22T15:00:00+08:00", "available_at": "2026-09-22T16:00:00+08:00"}
            outcomes = {"date": "2026-09-22", "market": {**source, "primary": "RISK_OFF"},
                        "limit_pool": {**source, "scope": "MAINBOARD_NON_ST_CLOSING_LIMIT_UP", "complete": True, "rows": []}}
            r = settle(registered, outcomes, "2026-09-22T18:00:00+08:00")
            self.assertFalse(r["market_primary_correct"])
            self.assertFalse(r["primary_candidate_promoted"])
            self.assertEqual(len(r["candidate_results"]), 1)
            outcomes["date"] = "2026-09-23"
            with self.assertRaises(ValueError): settle(registered, outcomes, "2026-09-23T18:00:00+08:00")

    def test_highest_alone_never_confirms_cycle_leader(self):
        r = judge(5, True, [], "2026-09-21T18:01:00+08:00")
        self.assertEqual(r["identity"], "SPACE_ANCHOR")
        self.assertIsNone(r["next_board_probability"])
        self.assertFalse(r["trade_authorized"])

    def test_rival_failure_needs_complete_denominator(self):
        keys = ["theme_verified", "sector_support", "first_board_turnover_verified", "same_height_active_lead",
                "genuine_divergence", "absorption_verified", "reseal_verified", "all_rivals_failed"]
        evidence = [{"fact": k, "value": True, "status": "verified", "source": "TEST_ONLY",
                    "observed_at": "2026-09-21T15:00:00+08:00", "available_at": "2026-09-21T17:00:00+08:00"} for k in keys]
        r = judge(3, True, evidence, "2026-09-21T18:01:00+08:00")
        self.assertEqual(r["identity"], "TURNOVER_CORE")
        self.assertEqual(r["competition_status"], "COMPETING_OR_UNKNOWN")
        evidence.append({**evidence[0], "fact": "competition_denominator_complete"})
        self.assertEqual(judge(3, True, evidence, "2026-09-21T18:01:00+08:00")["identity"], "POSITION_WINNER")


if __name__ == "__main__":
    unittest.main()
