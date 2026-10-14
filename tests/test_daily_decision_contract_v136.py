import json
import tempfile
import unittest
from pathlib import Path

from daily_decision_contract_v136 import audit, run
from tests.test_daily_decision_contract import DailyDecisionContractTests
from tests.test_emotion_selection_contract import fixture


def closing_report():
    report = DailyDecisionContractTests().report()
    report["stage"] = "CLOSING_PLAN"
    outlook = fixture()["next_session_outlook"]
    outlook["target_session"] = "2026-09-08"
    outlook["selection_links"] = []
    for item in outlook["evidence"]:
        item["observed_at"] = report["asof"]
        item["available_at"] = report["asof"]
    report["next_session_outlook"] = outlook
    return report


class IntegratedContractTest(unittest.TestCase):
    def test_valid_cash_report_no_entry(self):
        result = audit(closing_report())
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["entry_permission"], "NO_ENTER")

    def test_missing_outlook_blocks_old_valid_report(self):
        report = closing_report()
        del report["next_session_outlook"]
        self.assertEqual(audit(report)["status"], "INVALID_REPORT")

    def test_malformed_outlook_fails_closed(self):
        report = closing_report()
        report["next_session_outlook"] = None
        self.assertEqual(audit(report)["status"], "INVALID_REPORT")

    def test_output_is_immutable_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"source.json", Path(directory)/"audit.json"
            source.write_text(json.dumps(closing_report()), encoding="utf-8")
            result = run(source, target)
            self.assertEqual(len(result["source_hashes"]), 4)
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                run(source, target)
            self.assertEqual(target.read_bytes(), original)
