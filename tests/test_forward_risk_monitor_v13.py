import unittest

import forward_risk_monitor_v13 as monitor


class ForwardRiskMonitorV13Tests(unittest.TestCase):
    def _events(self, n: int, losses: int, promotions: int):
        events = []
        for index in range(n):
            session = f"s{index}"
            events.append({"event_type": "PLAN", "session_id": session, "candidate_boards": 3})
            events.append({
                "event_type": "SETTLEMENT", "session_id": session, "executed_proxy": True,
                "large_loss": index < losses, "promotion_target": index < promotions,
                "sell_blocked": False,
            })
        return events

    def test_missing_labels_are_not_counted_as_safe(self):
        events = self._events(10, 0, 8)
        for event in events:
            if event.get("event_type") == "SETTLEMENT":
                event["large_loss"] = None
        report = monitor.summarize(events, {"status": "PASS"})
        self.assertEqual(report["overall"]["resolved_large_loss"], 0)
        self.assertEqual(report["status"], "DATA_COLLECTION_REQUIRED")

    def test_sixty_clean_resolved_labels_pass_component_gate(self):
        report = monitor.summarize(self._events(60, 0, 45), {"status": "PASS"})
        self.assertTrue(report["point_gate_passed"])
        self.assertTrue(report["confidence_gate_passed"])
        self.assertEqual(report["status"], "RISK_COMPONENT_FORWARD_GATE_PASSED")

    def test_point_rate_above_ten_percent_fails(self):
        report = monitor.summarize(self._events(60, 7, 45), {"status": "PASS"})
        self.assertFalse(report["point_gate_passed"])
        self.assertEqual(report["status"], "FORWARD_RISK_GATE_FAILED")

    def test_bad_ledger_never_counts_events(self):
        report = monitor.summarize(self._events(60, 0, 60), {"status": "FAIL"})
        self.assertEqual(report["status"], "LEDGER_VERIFICATION_FAILED")


if __name__ == "__main__":
    unittest.main()
