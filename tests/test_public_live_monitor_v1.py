import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import public_live_monitor_v1 as monitor


class PublicLiveMonitorV1Tests(unittest.TestCase):
    def test_discover_review_uses_newest_signal_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "results" / "daily_review_old" / "review.json"
            newer = root / "results" / "daily_review_new" / "review.json"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_text('{"signal_date":"2026-09-03"}', encoding="utf-8")
            newer.write_text('{"signal_date":"2026-09-04"}', encoding="utf-8")
            self.assertEqual(monitor.discover_review(root), newer)

    def test_prior_session_binding_rejects_stale_candidate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.csv"
            review = root / "review.json"
            calendar = root / "market.csv"
            pd.DataFrame([{
                "signal_date": "2026-09-03", "code": "600001", "name": "旧候选", "boards": 1,
            }]).to_csv(candidate, index=False)
            review.write_text('{"signal_date":"2026-09-03"}', encoding="utf-8")
            pd.DataFrame({"date": ["2026-09-03", "2026-09-04"]}).to_csv(calendar, index=False)
            with self.assertRaisesRegex(RuntimeError, "stale close inputs"):
                monitor.validate_prior_session_binding(candidate, review, "2026-09-07", calendar)

    def test_weekend_aborts_without_calling_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.csv"
            candidate.write_text("placeholder", encoding="utf-8")
            calls = []

            def runner(*args):
                calls.append(args)
                return {}

            result = monitor.run_monitor(
                candidate, "2026-09-06", root / "out", None,
                clock=lambda: "2026-09-06T09:10:00+08:00", capture_runner=runner,
            )
            self.assertEqual(result["status"], "CAPTURE_SEQUENCE_INCOMPLETE")
            self.assertEqual(calls, [])
            self.assertFalse(result["direct_entry_authorized"])

    def test_late_start_skips_old_points_and_captures_ten_oclock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.csv"
            candidate.write_text("placeholder", encoding="utf-8")
            review = root / "review.json"
            review.write_text('{"signal_date":"2026-09-04"}', encoding="utf-8")
            calls = []
            minute_calls = []
            decision_calls = []

            def runner(candidate_path, trade_date, point, output, universe):
                calls.append((trade_date, point, universe))
                return {
                    "status": "CONDITIONAL_PUBLIC_PRICE_FRESH", "capture_id": "capture-1",
                    "manifest_path": "capture.json", "direct_entry_authorized": False,
                }

            def minute_runner(candidate_path, trade_date, point, output):
                minute_calls.append((trade_date, point))
                return {
                    "status": "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH",
                    "capture_id": "minute-1", "manifest_path": "minute.json",
                    "direct_entry_authorized": False,
                }

            def decision_runner(review_path, quote_manifest, output, minute_manifest):
                decision_calls.append((review_path, quote_manifest, output, minute_manifest))
                output.mkdir(parents=True)
                return {
                    "status": "CASH", "decision_path": str(output / "decision.json"),
                    "report_path": str(output / "report.md"),
                }

            result = monitor.run_monitor(
                candidate, "2026-09-07", root / "out", root / "universe.csv",
                review_path=review, clock=lambda: "2026-09-07T10:00:05+08:00",
                capture_runner=runner, minute_runner=minute_runner,
                decision_runner=decision_runner,
            )
            self.assertEqual([point for _, point, _ in calls], ["10:00"])
            self.assertEqual(minute_calls, [("2026-09-07", "10:00")])
            self.assertEqual(result["skipped_captures"], 5)
            self.assertEqual(result["completed_captures"], 1)
            self.assertEqual(result["completed_decisions"], 1)
            self.assertEqual(result["expected_decisions"], 3)
            self.assertEqual(len(decision_calls), 1)
            events = [json.loads(line) for line in Path(result["ledger_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 6)
            self.assertEqual(events[0]["previous_event_hash"], "GENESIS")
            self.assertEqual(events[-1]["previous_event_hash"], events[-2]["event_hash"])
            self.assertFalse(events[-1]["direct_entry_authorized"])
            self.assertEqual(events[-1]["minute_status"], "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH")
            self.assertEqual(events[-1]["decision_status"], "CASH")

    def test_ten_oclock_can_append_shadow_registry_without_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidate = root/"candidate.csv"; candidate.write_text("x", encoding="utf-8")
            review = root/"review.json"; review.write_text('{"signal_date":"2026-09-04"}', encoding="utf-8")
            def quote(*args): return {"status":"CONDITIONAL_PUBLIC_PRICE_FRESH", "manifest_path":"q.json"}
            def minute(*args): return {"status":"CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH", "manifest_path":"m.json"}
            def decision(review_path, quote_path, output, minute_path):
                output.mkdir(parents=True); path=output/"decision.json"; path.write_text("{}",encoding="utf-8")
                return {"status":"CASH", "decision_path":str(path), "report_path":str(output/"r.md")}
            calls=[]
            def shadow(registry_root, decision_path):
                calls.append((registry_root,decision_path)); return {"event_hash":"a"*64,"snapshot":"s.json"}
            result=monitor.run_monitor(candidate,"2026-09-07",root/"out",None,review_path=review,
                clock=lambda:"2026-09-07T10:00:05+08:00",capture_runner=quote,minute_runner=minute,
                decision_runner=decision,absorption_registry_root=root/"registry",
                absorption_snapshot_runner=shadow)
            self.assertEqual(len(calls),1)
            self.assertEqual(result["absorption_forward_snapshots"],1)
            self.assertEqual(result["absorption_forward_failures"],0)
            self.assertFalse(result["direct_entry_authorized"])

    def test_decision_failure_is_fail_closed_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.csv"
            candidate.write_text("placeholder", encoding="utf-8")
            review = root / "review.json"
            review.write_text('{"signal_date":"2026-09-04"}', encoding="utf-8")

            def quote_runner(*args):
                return {"status": "CONDITIONAL_PUBLIC_PRICE_FRESH", "manifest_path": "quote.json"}

            def minute_runner(*args):
                return {"status": "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH", "manifest_path": "minute.json"}

            def decision_runner(*args):
                raise ValueError("pool mismatch")

            result = monitor.run_monitor(
                candidate, "2026-09-07", root / "out", None, review_path=review,
                clock=lambda: "2026-09-07T10:00:05+08:00", capture_runner=quote_runner,
                minute_runner=minute_runner, decision_runner=decision_runner,
            )
            self.assertEqual(result["failed_decisions"], 1)
            self.assertEqual(result["status"], "CAPTURE_SEQUENCE_INCOMPLETE")
            events = [json.loads(line) for line in Path(result["ledger_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["status"], "CASH_CAPTURE_FAILED")
            self.assertEqual(events[-1]["decision_error"], "pool mismatch")

    def test_full_sequence_builds_three_bound_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.csv"
            candidate.write_text("placeholder", encoding="utf-8")
            review = root / "review.json"
            review.write_text('{"signal_date":"2026-09-04"}', encoding="utf-8")
            current = [monitor.local_timestamp("2026-09-07T09:19:59+08:00")]
            decisions = []

            def clock():
                return current[0]

            def sleeper(seconds):
                current[0] += pd.Timedelta(seconds=seconds)

            def quote_runner(candidate_path, trade_date, point, output, universe):
                path = root / f"quote-{point.replace(':', '')}.json"
                path.write_text('{"kind":"quote"}', encoding="utf-8")
                return {
                    "status": "CONDITIONAL_PUBLIC_PRICE_FRESH", "capture_id": f"q-{point}",
                    "manifest_path": str(path),
                }

            def minute_runner(candidate_path, trade_date, point, output):
                path = root / f"minute-{point.replace(':', '')}.json"
                path.write_text('{"kind":"minute"}', encoding="utf-8")
                return {
                    "status": "CONDITIONAL_PUBLIC_MINUTE_EVIDENCE_FRESH", "capture_id": f"m-{point}",
                    "manifest_path": str(path),
                }

            def decision_runner(review_path, quote_manifest, output, minute_manifest):
                decisions.append((quote_manifest.name, minute_manifest.name if minute_manifest else None))
                output.mkdir(parents=True)
                (output / "decision.json").write_text('{"status":"WATCH"}', encoding="utf-8")
                (output / "report.md").write_text("# WATCH", encoding="utf-8")
                return {
                    "status": "WATCH_PUBLIC_EVIDENCE_ONLY",
                    "decision_path": str(output / "decision.json"),
                    "report_path": str(output / "report.md"),
                }

            result = monitor.run_monitor(
                candidate, "2026-09-07", root / "out", None, review_path=review,
                clock=clock, sleeper=sleeper, capture_runner=quote_runner,
                minute_runner=minute_runner, decision_runner=decision_runner,
            )
            self.assertEqual(result["completed_captures"], 6)
            self.assertEqual(result["completed_decisions"], 3)
            self.assertEqual(result["failed_decisions"], 0)
            self.assertEqual(result["status"], "CAPTURE_AND_DECISION_SEQUENCE_COMPLETE_PUBLIC_ONLY")
            self.assertEqual(decisions, [
                ("quote-0925.json", None),
                ("quote-0935.json", "minute-0935.json"),
                ("quote-1000.json", "minute-1000.json"),
            ])
            verification = monitor.verify_run(Path(result["manifest_path"]))
            self.assertEqual(verification["status"], "PASS")
            # Later edits to mutable source inputs cannot invalidate the
            # immutable copies bound inside an already completed run.
            candidate.write_text("later mutation", encoding="utf-8")
            review.write_text('{"signal_date":"2099-01-01"}', encoding="utf-8")
            self.assertEqual(monitor.verify_run(Path(result["manifest_path"]))["status"], "PASS")
            decision_path = root / "out" / "2026-09-07" / result["run_id"] / "decisions" / "1000" / "decision.json"
            decision_path.write_text('{"status":"TAMPERED"}', encoding="utf-8")
            self.assertEqual(monitor.verify_run(Path(result["manifest_path"]))["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
