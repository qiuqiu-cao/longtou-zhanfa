import unittest
import json
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from intraday_forward_v10 import append_event, derive_outcome, verify, validate_plan_time


class IntradayForwardV10Tests(unittest.TestCase):
    def test_plan_must_be_frozen_before_trade_day_0915(self):
        now = datetime(2026, 9, 2, 9, 15, tzinfo=ZoneInfo("Asia/Shanghai"))
        with self.assertRaisesRegex(RuntimeError, "before trade-day 09:15"):
            validate_plan_time("2026-09-01", "2026-09-02", now)

    def test_plan_is_allowed_after_signal_close(self):
        now = datetime(2026, 9, 1, 15, 6, tzinfo=ZoneInfo("Asia/Shanghai"))
        validate_plan_time("2026-09-01", "2026-09-02", now)

    def test_one_word_down_is_large_loss_but_net_return_unresolved(self):
        row = pd.Series({
            "trade_close": 10.0, "trade_limit_up_price": 10.0,
            "exit_open": 9.0, "exit_high": 9.0, "exit_low": 9.0,
            "exit_close": 9.0, "exit_limit_down_price": 9.0,
            "buy_cost_bps": 1.0, "sell_cost_bps": 6.0, "slippage_bps": 3.0,
        })
        result = derive_outcome(row, True)
        self.assertTrue(result["sell_blocked"])
        self.assertTrue(result["large_loss"])
        self.assertIsNone(result["net_return"])
        self.assertLess(result["mark_to_market_return"], -0.07)

    def test_unconfirmed_shadow_does_not_create_proxy_fill(self):
        row = pd.Series({
            "trade_close": 10.0, "trade_limit_up_price": 10.0,
            "exit_open": 11.0, "exit_high": 11.0, "exit_low": 10.0,
            "exit_close": 11.0, "exit_limit_down_price": 9.0,
            "buy_cost_bps": 1.0, "sell_cost_bps": 6.0, "slippage_bps": 3.0,
        })
        result = derive_outcome(row, False)
        self.assertFalse(result["executed_proxy"])
        self.assertIsNone(result["large_loss"])

    def test_hash_chain_detects_plan_event_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            basis = {
                "version": "v10-intraday-forward-1", "model_lock_id": "9738e2c165075798",
                "signal_date": "2026-09-01", "trade_date": "2026-09-02",
                "emotion_lock_id": "emotion", "emotion_event_hash": "event",
                "candidate_code": None, "candidate_name": None, "candidate_boards": None,
                "joint_score": None, "input_sha256": "input",
            }
            from intraday_forward_v10 import hash_value
            session_id = hash_value(basis)[:20]
            session = output / session_id
            session.mkdir()
            plan = {
                **basis, "session_id": session_id,
                "planned_at": "2026-09-01T15:06:00+08:00",
                "decision": "NO_SHADOW_CANDIDATE",
            }
            plan_path = session / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            score_path = session / "frozen_scores.csv"
            pd.DataFrame({"code": [], "shadow_observation": []}).to_csv(score_path, index=False)
            import intraday_forward_v10 as module
            append_event({
                "event_type": "PLAN", "session_id": session_id,
                "model_lock_id": "9738e2c165075798", "signal_date": "2026-09-01",
                "trade_date": "2026-09-02", "planned_at": plan["planned_at"],
                "candidate_code": None, "candidate_boards": None,
                "plan_file": str(plan_path), "plan_sha256": module.sha256_file(plan_path),
                "score_file": str(score_path), "score_sha256": module.sha256_file(score_path),
            }, output)
            self.assertEqual(verify(output)["status"], "PASS")
            ledger = output / "intraday_forward_ledger.jsonl"
            text = ledger.read_text(encoding="utf-8").replace('"candidate_code": null', '"candidate_code": "600001"')
            ledger.write_text(text, encoding="utf-8")
            self.assertEqual(verify(output)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
