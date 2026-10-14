from copy import deepcopy
import tempfile
import unittest

from forecast_review_bridge_v277 import audit, attach, review_digest
from next_session_expectation_contract_v271 import register, digest
from test_next_session_expectation_v271 import plan


def review():
    return {"signal_date": "2026-09-21", "asof": "2026-09-21T17:00:00+08:00",
            "status": "CASH", "relay_candidates": [{"code": "600001", "boards": 2}],
            "pool_summary": {"relay": 1}, "decision_scope": "TEST_ONLY"}


def frozen(r=None, recorded="2026-09-21T18:01:00+08:00", update=None):
    r = review() if r is None else r
    p = plan(); p["review_hash"] = review_digest(r)
    if update:
        update(p)
    with tempfile.TemporaryDirectory() as directory:
        _, wrapped = register(p, directory, recorded)
    return wrapped


class BindingTests(unittest.TestCase):
    def test_bound_forecast_keeps_cash_and_pending(self):
        r = review(); attach(r, frozen(r))
        self.assertEqual(r["status"], "CASH")
        self.assertEqual(r["next_session_expectation"]["status"], "BOUND")
        self.assertTrue(r["next_session_expectation"]["forward_accuracy_eligible"])
        self.assertFalse(r["next_session_expectation"]["trade_authorized"])
        self.assertEqual(r["next_session_expectation"]["execution_state"], "PENDING_OPEN_CONFIRMATION")
        self.assertEqual(audit(r, frozen())["status"], "BOUND")

    def test_missing_plan_not_counted_as_prediction(self):
        self.assertFalse(audit(review())["forward_accuracy_eligible"])

    def test_exact_review_and_denominator_checked(self):
        r = review(); w = frozen(r)
        r["relay_candidates"].append({"code": "600002", "boards": 3})
        r["pool_summary"]["relay"] = 2
        a = audit(r, w)
        self.assertIn("RELAY_DENOMINATOR_MISMATCH", a["issues"])
        self.assertIn("EXACT_REVIEW_HASH_MISMATCH", a["issues"])
        self.assertFalse(a["forward_accuracy_eligible"])

    def test_hindsight_review_cannot_bind(self):
        r = review(); r["asof"] = "2026-09-22T15:00:00+08:00"
        a = audit(r, frozen(r))
        self.assertIn("REVIEW_NOT_AVAILABLE_AT_EVIDENCE_CUTOFF", a["issues"])

    def test_replay_and_tamper_rejected(self):
        r = review(); w = frozen(r, recorded="2026-09-22T18:01:00+08:00")
        self.assertFalse(audit(r, w)["forward_accuracy_eligible"])
        w["forecast_status"] = "FORWARD_REGISTERED"
        w["sha256"] = digest({k:v for k,v in w.items() if k != "sha256"})
        self.assertIn("REGISTRATION_STATUS_MISMATCH", audit(r, w)["issues"])
        w["plan"]["market"]["primary"] = "RISK_OFF"
        with self.assertRaises(ValueError): audit(r, w)


if __name__ == "__main__":
    unittest.main()
