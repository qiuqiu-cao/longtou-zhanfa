import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import emotion_forward_v4 as forward


class EmotionForwardV4Tests(unittest.TestCase):
    def _copy_lock(self, output: Path) -> None:
        source = Path(forward.DEFAULT_OUTPUT) / forward.LOCK_FILE
        (output / forward.LOCK_FILE).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    def _forecast_payload(self, signal_date: str = "2026-09-01") -> dict:
        return {
            "event_type": "FORECAST",
            "lock_id": forward.load_lock(forward.DEFAULT_OUTPUT)["lock_id"],
            "signal_date": signal_date,
            "recorded_at": signal_date + "T15:10:00+08:00",
            "emotion_state": "升温",
            "active_state": True,
            "forecasts": {
                "adverse_state_next": {"probability": 0.4, "causal_base": 0.45, "comparison_baseline": 0.3},
                "relay_damage_next": {"probability": 0.25, "causal_base": 0.32},
            },
        }

    def test_current_challenger_lock_verifies(self):
        self.assertEqual(forward.verify_challenger_lock(forward.DEFAULT_OUTPUT)["status"], "PASS")

    def test_development_end_cannot_be_registered_as_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._copy_lock(output)
            historical_input = output / "market_through_development_end.csv"
            market = pd.read_csv(forward.DEFAULT_INPUT)
            market[pd.to_datetime(market["date"]) <= pd.Timestamp("2026-08-31")].to_csv(
                historical_input, index=False
            )
            with self.assertRaisesRegex(RuntimeError, "not later than locked development end"):
                forward.build_forecast_payload(
                    historical_input,
                    output,
                    now=datetime(2026, 8, 31, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                )

    def test_hash_chain_and_duplicate_forecast_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._copy_lock(output)
            forward.append_event(output, self._forecast_payload())
            self.assertEqual(forward.verify_ledger(output)["status"], "PASS")
            forward.append_event(output, self._forecast_payload())
            verification = forward.verify_ledger(output)
            self.assertEqual(verification["status"], "FAIL")
            self.assertTrue(any("duplicate forecast" in item for item in verification["errors"]))

    def test_settlement_must_bind_prior_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._copy_lock(output)
            forward.append_event(output, {
                "event_type": "SETTLEMENT",
                "lock_id": forward.load_lock(forward.DEFAULT_OUTPUT)["lock_id"],
                "signal_date": "2026-09-01", "labels": {},
            })
            verification = forward.verify_ledger(output)
            self.assertEqual(verification["status"], "FAIL")
            self.assertTrue(any("settlement without forecast" in item for item in verification["errors"]))

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._copy_lock(output)
            forward.append_event(output, self._forecast_payload())
            path = output / forward.LEDGER_FILE
            event = json.loads(path.read_text(encoding="utf-8").strip())
            event["forecasts"]["adverse_state_next"]["probability"] = 0.01
            path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(forward.verify_ledger(output)["status"], "FAIL")

    def test_damage_label_preserves_unknown(self):
        self.assertIsNone(forward.damage_label(pd.Series({})))
        self.assertEqual(forward.damage_label(pd.Series({
            "loss_effect_health": 0.2, "promotion_mid": None, "down_limit_count": None,
        })), 1.0)
        self.assertEqual(forward.damage_label(pd.Series({
            "loss_effect_health": 0.7, "promotion_mid": 0.4, "down_limit_count": 1,
        })), 0.0)

    def test_next_trading_day_settlement_updates_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self._copy_lock(output)
            forward.append_event(output, self._forecast_payload())
            rows = []
            for date, state, loss, stamp in [
                ("2026-09-01", "升温", 0.6, "2026-09-01 15:00:00"),
                ("2026-09-02", "大分歧", 0.2, "2026-09-02 15:00:00"),
            ]:
                row = {
                    "date": date, "emotion_state": state,
                    "emotion_adverse_markov_probability": 0.3,
                    "loss_effect_health": loss, "promotion_mid": 0.1,
                    "down_limit_count": 8, "source": "test",
                    "source_timestamp": stamp, "data_status": "verified",
                    "emotion_data_completeness": 1.0,
                }
                for feature in forward.challenger.FEATURES:
                    row.setdefault(feature, 0.5)
                rows.append(row)
            input_path = output / "market.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            settled = forward.settle_forecasts(
                input_path, output,
                now=datetime(2026, 9, 2, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
            )
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["labels"]["adverse_state_next"], 1.0)
            self.assertEqual(settled[0]["labels"]["relay_damage_next"], 1.0)
            self.assertEqual(forward.verify_ledger(output)["status"], "PASS")
            monitor = forward.forward_monitor(output)
            self.assertEqual(monitor["resolved_forecasts"], 1)
            self.assertEqual(monitor["status"], "INSUFFICIENT_FORWARD_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
