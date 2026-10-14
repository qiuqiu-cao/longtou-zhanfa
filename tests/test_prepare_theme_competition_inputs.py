"""Synthetic fixtures only; these tests do not create market/forward evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import prepare_theme_competition_inputs as adapter
import build_specific_theme_roles_v150 as v150
import theme_competition_archive_v151 as v151

DATE = "2026-10-08"
CUTOFF = adapter.stamp(DATE+"T16:30:00+08:00")
CLOCK = adapter.stamp(DATE+"T16:00:00+08:00")


def pool():
    return {"date": DATE, "available_at": DATE+"T15:05:00+08:00",
            "capture_complete_within_provider_response": True, "rows": [
        {"code": "600001", "name": "fixture_A", "boards": 2, "amount": 100, "seal_first": 93000},
        {"code": "002001", "name": "fixture_B", "boards": 2, "amount": 90, "seal_first": 93500},
        {"code": "000001", "name": "fixture_C", "boards": 3, "amount": 150, "seal_first": 94000},
        {"code": "600002", "name": "fixture_unknown", "boards": 1, "amount": 80, "seal_first": 100000},
    ]}


def evidence():
    return {"schema_version": v150.VERSION, "signal_date": DATE, "groups": [{
        "theme_group_id": "fixture_group", "theme": "fixture_specific_theme",
        "theme_granularity": "specific_subtheme", "specificity_basis": "synthetic shared product",
        "codes": ["600001", "002001", "000001"], "driver_type": "industry",
        "driver_summary": "synthetic catalyst", "driver_source_url": "https://example.com/driver",
        "driver_first_available_at": DATE+"T14:00:00+08:00", "driver_claim_status": "verified_fact",
        "expectation_status": "unknown", "expectation_basis": "synthetic assessment unknown",
        "expectation_assessed_at": DATE+"T15:10:00+08:00",
        "membership_source_url": "https://example.com/membership",
        "membership_published_at": DATE+"T13:00:00+08:00", "source_tier": "A_PRIMARY",
    }]}


def fixture_sources(root):
    item = evidence()
    ep = root/"evidence.json"
    ep.write_text(json.dumps(item), encoding="utf-8")
    refs = {}
    for kind, text in [("membership", "fixture_specific_theme 600001 002001 000001"),
                       ("driver", "synthetic catalyst"), ("assessment", "synthetic assessment unknown")]:
        p = root/(kind+".txt")
        p.write_text(text, encoding="utf-8")
        ref = {"raw_path": str(p.resolve()), "raw_sha256": adapter.sha(p),
               "captured_at": DATE+"T15:15:00+08:00", "excerpt": text}
        if kind != "assessment":
            ref.update(url=item["groups"][0]["membership_source_url" if kind == "membership" else "driver_source_url"],
                       published_at=item["groups"][0]["membership_published_at" if kind == "membership" else "driver_first_available_at"],
                       publication_time_basis="REVIEWED_EXACT_TIMESTAMP")
        refs[kind] = ref
    bp = root/"bindings.json"
    binding = {"signal_date": DATE, "evidence_sha256": adapter.sha(ep),
               "recorded_at": DATE+"T15:25:00+08:00", "reviewed_at": DATE+"T15:20:00+08:00",
               "review_status": "REVIEWED_SOURCE_TEXT", "reviewer": "synthetic_test_fixture",
               "review_notes": "synthetic; no real securities or performance claims",
               "groups": {"fixture_group": refs}}
    bp.write_text(json.dumps(binding), encoding="utf-8")
    return ep, bp, binding


class ThemeInputPreparationTests(unittest.TestCase):
    def test_unknown_roster_and_theme_nulls_preserved(self):
        p, pm, roles, tm, intake, comp, cm, factors = adapter.frames(pool(), DATE, CLOCK.isoformat(), CUTOFF)
        self.assertEqual(len(factors), 4)
        self.assertEqual(set(factors.code), {r["code"] for r in pool()["rows"]})
        self.assertTrue(factors.same_theme_board_peer_count.isna().all())
        self.assertTrue(factors.theme_membership_known.isna().all())
        self.assertFalse(factors.eligible_for_leader_factor_training.any())
        self.assertTrue(roles.theme_status.eq("UNAVAILABLE").all())
        self.assertFalse(intake.provider_categories_verify_active_theme.any())
        self.assertEqual(cm["verified_theme_candidate_count"], 0)

    def test_late_preparation_cannot_reconstruct_past_cutoff_features(self):
        *_, comp, cm, factors = adapter.frames(pool(), DATE, DATE+"T17:00:00+08:00", CUTOFF)
        self.assertTrue(comp.empty)
        self.assertEqual(cm["status"], "NOT_AVAILABLE_BY_FIXED_CUTOFF")
        self.assertTrue(factors.same_theme_peer_count.isna().all())

    def test_reviewed_primary_roles_use_original_peer_rules_and_preserve_unknown(self):
        values = adapter.frames(pool(), DATE, CLOCK.isoformat(), CUTOFF, evidence=evidence())
        factors = values[-1].set_index("code")
        self.assertEqual(factors.loc["600001", "same_theme_board_peer_count"], 1)
        self.assertEqual(factors.loc["600001", "higher_theme_peer_count"], 1)
        self.assertTrue(pd.isna(factors.loc["600002", "same_theme_peer_count"]))
        self.assertTrue(values[5].data_status.eq("UNVERIFIED_PROVIDER_CLOSING_STRUCTURE").all())
        self.assertFalse(values[2].positive_catalyst_factor_eligible.fillna(False).any())

    def test_unknown_security_status_cannot_pass_v151(self):
        p, pm, roles, tm, intake, comp, cm, factors = adapter.frames(pool(), DATE, CLOCK.isoformat(), CUTOFF)
        with self.assertRaisesRegex(ValueError, "ineligible or unverified"):
            v151.build(p, pm, comp, cm, CUTOFF.isoformat())

    def test_context_candidate_mismatch_rejected(self):
        context = {"scope": {"price_signal_date": DATE, "codes": [r["code"] for r in pool()["rows"]]},
                   "candidates": [{"code": c} for c in ["600001", "002001", "000001", "600003"]]}
        with self.assertRaisesRegex(ValueError, "entire unique"):
            adapter.frames(pool(), DATE, CLOCK.isoformat(), CUTOFF, context=context)

    def test_actual_source_bindings_validate_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as td:
            ep, bp, b = fixture_sources(Path(td))
            checked, hashes = adapter.validate_theme_sources(ep, bp, DATE, CUTOFF, CLOCK)
            self.assertEqual(checked, evidence())
            self.assertEqual(len(hashes), 5)
            Path(b["groups"]["fixture_group"]["driver"]["raw_path"]).write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                adapter.validate_theme_sources(ep, bp, DATE, CUTOFF, CLOCK)

    def test_bad_review_source_clocks_date_and_membership_fail_closed(self):
        variants = [
            ("empty review", lambda b: b.update(review_status="UNREVIEWED")),
            ("late review", lambda b: b.update(reviewed_at=DATE+"T17:00:00+08:00")),
            ("wrong date", lambda b: b.update(signal_date="2026-09-30")),
            ("capture after review", lambda b: b["groups"]["fixture_group"]["driver"].update(captured_at=DATE+"T15:30:00+08:00")),
            ("no publication basis", lambda b: b["groups"]["fixture_group"]["membership"].update(publication_time_basis="UNKNOWN")),
            ("missing named member", lambda b: b["groups"]["fixture_group"]["membership"].update(excerpt="fixture_specific_theme 600001")),
            ("naive clock", lambda b: b.update(reviewed_at=DATE+"T15:20:00")),
        ]
        for label, change in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                ep, bp, b = fixture_sources(Path(td))
                change(b)
                bp.write_text(json.dumps(b), encoding="utf-8")
                with self.assertRaises(ValueError):
                    adapter.validate_theme_sources(ep, bp, DATE, CUTOFF, CLOCK)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/"output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                adapter.prepare("unused", DATE, "unused", output)

    def test_complete_derived_archive_crossing_cutoff_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ["manifest.json", "normalized.json"]:
                (root/name).write_text("{}", encoding="utf-8")
            protocol = root/"protocol.json"
            protocol.write_text(json.dumps({"registered_at": "2026-10-03T03:08:27+08:00",
                "prospective_information_mode": "B_FIXED_EVENING",
                "prospective_knowledge_cutoff_local": "16:30:00 Asia/Shanghai"}), encoding="utf-8")
            manifest = {"recorded_at": DATE+"T15:05:00+08:00", "calendar": {"status": "verified", "source": "https://example.com/synthetic_calendar", "sessions": [DATE, "2026-10-09"]}}
            with patch.object(adapter.outcome_v293, "reconstruct", return_value=(manifest, {(DATE, "limit_up"): pool()}, {}, [])), \
                 patch.object(adapter, "validate_theme_sources", return_value=(evidence(), {})), \
                 patch.object(adapter, "now", side_effect=[DATE+"T16:29:59+08:00", DATE+"T16:29:59+08:00", DATE+"T16:30:01+08:00"]):
                with self.assertRaisesRegex(ValueError, "complete derived archive"):
                    adapter.prepare(root, DATE, protocol, root/"output", theme_evidence="fixture", theme_bindings="fixture")
            self.assertFalse((root/"output/manifest.json").exists())
            self.assertFalse(json.loads((root/"output/failure.json").read_text())["prediction_use_allowed"])


if __name__ == "__main__":
    unittest.main()
