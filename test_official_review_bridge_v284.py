from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import official_review_bridge_v284 as bridge
from next_session_expectation_contract_v271 import stamp


ROOT = Path(__file__).resolve().parent
UPDATE = ROOT / "results/v283/manual_official_20261002_v1/manual_official_update.json"
ARCHIVE = ROOT / "data/v283/holiday_official_20261002_v1"
REGISTERED = ROOT / "data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json"
REVIEW = ROOT / "results/v280/holiday_forward_20261002_v1/review.json"


class OfficialBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.update, cls.frozen = bridge.authenticate(UPDATE, ARCHIVE, REGISTERED)
        cls.original = bridge.read(REVIEW)

    def review(self):
        result = deepcopy(self.original)
        result.pop(bridge.forecast.KEY)
        result["asof"] = "2026-10-02T20:00:00+08:00"
        return result

    def attach(self, review):
        with patch.object(bridge, "now", return_value=stamp("2026-10-02T21:00:00+08:00")):
            return bridge.attach(review, self.update, self.frozen)

    def authenticate_changed(self, changed):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "update.json"
            path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            return bridge.authenticate(path, ARCHIVE, REGISTERED)

    def test_real_annotation_source_chain_covers_all_eleven(self):
        self.assertEqual(self.update["candidate_denominator"], 11)
        self.assertEqual(self.update["fulltext_reviewed_candidates"], 9)
        self.assertEqual(self.update["registration_hash"], self.frozen["sha256"])

    def test_payload_fact_edit_is_rejected_even_when_source_hashes_are_unchanged(self):
        changed = deepcopy(self.update)
        changed["candidate_updates"][3]["facts"][0] = "TEST_ONLY fabricated cooperation"
        with self.assertRaisesRegex(ValueError, "payload differs"):
            self.authenticate_changed(changed)

    def test_probability_permission_and_candidate_omission_edits_rejected(self):
        for key, value in (("probability_adjustment", .9), ("trade_authorized", True)):
            changed = deepcopy(self.update)
            changed["candidate_updates"][0][key] = value
            with self.assertRaisesRegex(ValueError, "payload differs"):
                self.authenticate_changed(changed)
        changed = deepcopy(self.update)
        changed["candidate_updates"].pop()
        with self.assertRaisesRegex(ValueError, "payload differs"):
            self.authenticate_changed(changed)

    def test_source_hash_change_rejected(self):
        changed = deepcopy(self.update)
        changed["source_hashes"][str(REGISTERED)] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.authenticate_changed(changed)

    def test_appendix_cannot_backfill_original_knowledge_time(self):
        review = self.review()
        review["asof"] = self.original["asof"]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.attach(review)

    def test_wrong_date_subset_duplicate_and_wrong_board_rejected(self):
        bad = []
        review = self.review(); review["signal_date"] = "2026-09-29"; bad.append(review)
        review = self.review(); review["relay_candidates"].pop(); bad.append(review)
        review = self.review(); review["relay_candidates"].append(deepcopy(review["relay_candidates"][0])); bad.append(review)
        review = self.review(); review["relay_candidates"][0]["boards"] += 1; bad.append(review)
        for review in bad:
            with self.assertRaises(ValueError): self.attach(review)

    def test_frozen_binding_cannot_silently_survive_changed_review(self):
        with self.assertRaisesRegex(ValueError, "before forecast binding"):
            self.attach(deepcopy(self.original))

    def test_future_review_time_rejected(self):
        review = self.review(); review["asof"] = "2099-01-01T00:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "future"):
            self.attach(review)

    def test_valid_attachment_preserves_all_calls_and_trade_state(self):
        review = self.review(); original = deepcopy(review)
        self.attach(review)
        self.assertEqual(review["relay_candidates"], original["relay_candidates"])
        self.assertEqual(review["status"], original["status"])
        self.assertFalse(review[bridge.KEY]["trade_authorized"])
        bridge.forecast.attach(review)
        self.assertEqual(review[bridge.forecast.KEY]["status"], "NOT_REGISTERED")
        self.assertFalse(review[bridge.forecast.KEY]["forward_accuracy_eligible"])
        text = bridge.markdown(review[bridge.KEY])
        self.assertIn("九阳股份与华为已建立所传战略合作", text)
        self.assertIn("13.14万元", text)
        self.assertIn("未证实清洁", text)

    def test_cli_new_appendix_preserves_original_files_and_rejects_overwrite(self):
        paths = (REVIEW, REGISTERED, UPDATE)
        before = {p: bridge.sha(p) for p in paths}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "appendix"
            with patch.object(bridge, "now", return_value=stamp("2026-10-02T21:00:00+08:00")):
                result = bridge.run(REVIEW, UPDATE, ARCHIVE, REGISTERED,
                                    "2026-10-02T20:00:00+08:00", output)
            self.assertFalse(result["original_calls_rewritten"])
            self.assertTrue((output / "review.md").exists())
            with self.assertRaises(FileExistsError):
                bridge.run(REVIEW, UPDATE, ARCHIVE, REGISTERED,
                           "2026-10-02T20:00:00+08:00", output)
        self.assertEqual(before, {p: bridge.sha(p) for p in paths})

    def test_daily_review_run_displays_authenticated_facts_without_trade_upgrade(self):
        import pandas as pd
        import current_daily_review as daily
        from tests.test_current_daily_review import CurrentDailyReviewTests
        fixtures = CurrentDailyReviewTests()
        template = fixtures.snapshot().iloc[1].to_dict()
        rows = [{**template, "signal_date": "2026-09-30", "code": r["code"],
                 "name": "TEST_ONLY", "boards": r["boards"]}
                for r in self.frozen["plan"]["candidates"]]
        manifest = fixtures.manifest()
        manifest.update(signal_date="2026-09-30", mainboard_closing_limit_rows=11,
                        mainboard_failed_board_rows=0, verified_eligible_closing_rows=0,
                        security_unavailable_or_ineligible_closing_rows=11,
                        mainboard_board_level_counts={str(b):sum(r["boards"]==b for r in rows)
                                                     for b in {r["boards"] for r in rows}})
        market = fixtures.market(); market["date"] = "2026-09-30"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "snapshot.csv", index=False)
            market.to_csv(root / "market.csv", index=False)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "policy.json").write_text('{"operational_threshold": null}', encoding="utf-8")
            with patch.object(bridge, "now", return_value=stamp("2026-10-02T21:00:00+08:00")):
                report = daily.run(root / "snapshot.csv", root / "manifest.json",
                    root / "market.csv", root / "policy.json", root / "output",
                    "2026-10-02T20:00:00+08:00", manual_official_update_path=UPDATE,
                    official_archive_path=ARCHIVE, official_registered_path=REGISTERED)
            self.assertEqual(report["status"], "CASH")
            self.assertEqual(report[KEY := bridge.KEY]["candidate_denominator"], 11)
            self.assertFalse(report[KEY]["trade_authorized"])
            text = (root / "output/review.md").read_text(encoding="utf-8")
            self.assertIn("九阳股份与华为已建立所传战略合作", text)
            self.assertIn("NOT_REGISTERED", text)


if __name__ == "__main__":
    unittest.main()
