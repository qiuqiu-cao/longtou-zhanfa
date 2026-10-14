import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import announcement_forward_v14 as forward


TZ = ZoneInfo("Asia/Shanghai")


class AnnouncementForwardV14Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan_path = self.root / "plan.json"
        self.manifest_path = self.root / "manifest.json"
        self.announcement_path = self.root / "announcements.csv"
        self.candidate_path = self.root / "candidates.csv"
        self.output = self.root / "out"
        self.plan = {
            "session_id": "session1", "signal_date": "2026-09-01", "trade_date": "2026-09-02",
            "planned_at": "2026-09-01T20:00:00+08:00", "candidate_code": "003005",
            "candidate_name": "竞业达", "candidate_boards": 3,
        }
        self.plan_path.write_text(json.dumps(self.plan, ensure_ascii=False), encoding="utf-8")
        self.candidates = pd.DataFrame([
            {"date": "2026-09-01", "code": "003005", "name": "竞业达", "boards": 3},
            {"date": "2026-09-01", "code": "600001", "name": "高标", "boards": 6},
        ])
        self.candidates.to_csv(self.candidate_path, index=False, encoding="utf-8-sig")

    def tearDown(self):
        self.temp.cleanup()

    def write_announcements(self, rows, collected_at="2026-09-01T22:00:00+08:00"):
        frame = pd.DataFrame(rows)
        frame.to_csv(self.announcement_path, index=False, encoding="utf-8-sig")
        manifest = {
            "signal_date": "2026-09-01", "collected_at": collected_at, "candidate_count": 2,
            "announcements": {"status": "verified_index_date_only", "rows": len(frame),
                              "candidate_codes": 2, "failed_codes": 0, "errors": {}},
            "files": {"announcements": {"path": str(self.announcement_path),
                                            "sha256": forward.sha256_file(self.announcement_path)}},
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def row(code="003005", risk_type="风险提示_异常波动", hard=1,
            title="股票交易异常波动公告", published="2026-09-01 23:59:59"):
        return {
            "code": code, "name": "竞业达" if code == "003005" else "高标",
            "published_at": published, "risk_type": risk_type, "title": title,
            "url": "https://example.test/notice", "risk_hard": hard,
            "published_at_status": "date_only_conservative_end_of_day",
            "source": "test index", "source_timestamp": "2026-09-01T22:00:00+08:00",
            "data_status": "verified_index_date_only",
        }

    def bind_now(self):
        return forward.bind(
            self.plan_path, self.manifest_path, self.announcement_path, self.candidate_path,
            self.output, datetime(2026, 9, 2, 8, 30, tzinfo=TZ), require_registered_v10=False,
        )

    def test_abnormal_notice_rejects_candidate(self):
        self.write_announcements([self.row(), self.row(code="600001", hard=0, risk_type="其他公告")])
        binding = self.bind_now()
        self.assertEqual(binding["decision"], "HARD_RISK_REJECT")
        self.assertEqual(binding["candidate_hard_risk_count"], 1)

    def test_completed_reduction_alone_does_not_reject(self):
        self.write_announcements([
            self.row(risk_type="historical_corporate_action_closed", hard=0,
                     title="减持计划完成的公告"),
            self.row(code="600001", hard=0, risk_type="其他公告"),
        ])
        binding = self.bind_now()
        self.assertEqual(binding["decision"], "NO_HARD_RISK_FOUND")

    def test_date_only_same_day_notice_is_not_known_at_same_day_close(self):
        self.write_announcements([self.row(), self.row(code="600001", hard=0, risk_type="其他公告")])
        announcements = pd.read_csv(self.announcement_path, dtype={"code": "string"})
        at_close = forward.effective_rows(
            announcements, forward.local_timestamp("2026-09-01T15:00:00+08:00")
        )
        self.assertTrue(at_close.empty)
        preopen = forward.effective_rows(
            announcements, forward.local_timestamp("2026-09-02T08:30:00+08:00")
        )
        self.assertEqual(len(preopen), 2)

    def test_late_bind_is_rejected(self):
        self.write_announcements([self.row(), self.row(code="600001", hard=0, risk_type="其他公告")])
        with self.assertRaisesRegex(ValueError, "before trade-date 09:15"):
            forward.bind(
                self.plan_path, self.manifest_path, self.announcement_path, self.candidate_path,
                self.output, datetime(2026, 9, 2, 9, 15, tzinfo=TZ), require_registered_v10=False,
            )

    def test_posthoc_collection_cannot_be_used_as_forward_evidence(self):
        self.write_announcements(
            [self.row(), self.row(code="600001", hard=0, risk_type="其他公告")],
            collected_at="2026-09-02T10:00:00+08:00",
        )
        binding = self.bind_now()
        self.assertEqual(binding["decision"], "ANNOUNCEMENT_DATA_UNAVAILABLE")
        self.assertFalse(binding["coverage_complete"])

    def test_tampering_is_detected(self):
        self.write_announcements([self.row(), self.row(code="600001", hard=0, risk_type="其他公告")])
        self.bind_now()
        self.assertEqual(forward.verify(self.output)["status"], "PASS")
        frozen = self.output / "session1" / "frozen_announcements.csv"
        frozen.write_text(frozen.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8-sig")
        self.assertEqual(forward.verify(self.output)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
