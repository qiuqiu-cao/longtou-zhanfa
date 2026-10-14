"""Synthetic final-boundary checks; never invoke a real V298 freeze."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import model_upgrade_freeze as freeze


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze.shadow, "now_utc", lambda: datetime(2030, 1, 2, 9, 5, tzinfo=timezone.utc))
    calendar = {"status": "verified", "source": "SYNTHETIC_OFFICIAL_ONLY",
                "scope": ["2030-01-01", "2030-01-04"], "sessions": ["2030-01-02", "2030-01-03"]}
    calendar_path = tmp_path / "calendar.json"
    calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    active = tmp_path / "active_pointer.json"
    active.write_text('{"synthetic":true}', encoding="utf-8")
    binding = {"model_path": str(tmp_path / "model.joblib"), "model_sha256": "b" * 64}
    record = {"module": "relay", "signal_date": "2030-01-02", "target_date": "2030-01-03",
              "knowledge_cutoff": "2030-01-02T16:50:00+08:00", "registered_at": "2030-01-02T16:55:00+08:00",
              "protocol_path": str(protocol), "original_codes": ["000001", "000002"], "forward": True,
              "model_bindings": {"C1": binding}, "arms": {"C1": {"status": "KNOWN", "candidates": [
                  {"code": "000001", "boards": 2, "actual_one_word_daily": None, "raw_score": .8, "predicted_positive": True},
                  {"code": "000002", "boards": 3, "actual_one_word_daily": 0, "raw_score": .4, "predicted_positive": False}]}}}
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(record), encoding="utf-8")
    pointer = {"active_arm": "C1", "effective_signal": "2030-01-02",
               "promoted_at": "2030-01-01T16:00:00+08:00", "model": binding, "suspended": False}
    proof = {"status": "VERIFIED_ACTIVE", "registration_path": str(registration),
             "registration_sha256": freeze.shadow.file_sha256(registration), "active_path": str(active),
             "active_sha256": freeze.shadow.file_sha256(active), "protocol_path": str(protocol),
             "protocol_sha256": freeze.shadow.file_sha256(protocol), "receipt_path": str(tmp_path / "accepted.json"),
             "receipt_sha256": "c" * 64, "verified_at": "2030-01-02T17:05:00+08:00"}
    rows = [{"code": "000001", "boards": 2, "actual_one_word_daily": None, "probability": None},
            {"code": "000002", "boards": 3, "actual_one_word_daily": False, "probability": None}]
    pending = {"signal_date": record["signal_date"], "target_date": record["target_date"], "calendar": calendar,
               "knowledge_cutoff": "2030-01-02T17:00:00+08:00", "input_hash": "a" * 64, "model_version": "SYNTH",
               "market": {"direction": "CONTINUE", "probability": None}, "candidates": rows, "primary_candidate": None,
               "daily_research": {"draft_status": freeze.apply.PENDING, "candidates": [
                   {"code": row["code"], "events": {"promotion": {"call": None, "probability": None},
                                                   "break": {"call": None, "probability": None}}} for row in rows]}}
    plan = freeze.apply._adapt_verified(pending, record, pointer, proof)
    plan["daily_research"]["draft_status"] = "AUTHOR_REVIEWED"
    plan_path = tmp_path / "reviewed.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    sources = {str(Path(path).resolve()): freeze.shadow.file_sha256(path)
               for path in (freeze.__file__, freeze.ENTRY, freeze.shadow.__file__, freeze.apply.__file__)}
    guard = {"formed_at": "2030-01-01T16:00:00+08:00", "protocol_sha256": proof["protocol_sha256"], "sources": sources}
    guard_path = tmp_path / freeze.GUARD_NAME
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    stopped, delegated = [], []
    monkeypatch.setattr(freeze.shadow, "_binding_failure", lambda path, module, error: stopped.append((path, module, str(error))))
    monkeypatch.setattr(freeze.shadow, "verify_active_for_registration", lambda *_: (record, pointer, deepcopy(proof)))
    def run(command, **kwargs):
        assert (tmp_path / "execution.json").exists(), "reserve unique writable receipt before registration"
        delegated.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"path":"SYNTHETIC_REGISTERED"}', stderr="")
    monkeypatch.setattr(freeze.subprocess, "run", run)
    return plan, plan_path, calendar_path, tmp_path / "execution.json", record, pointer, proof, guard_path, stopped, delegated


def test_reviewed_unknown_ohlc_keeps_scores_and_delegates_original_argv(bundle):
    plan, path, calendar, output, _, _, _, _, stopped, delegated = bundle
    before = path.read_bytes()
    result = freeze.freeze_plan(path, calendar, output, output.parent / "registry")
    assert result["status"] == "ORIGINAL_V298_FREEZE_COMPLETE"
    assert path.read_bytes() == before and output.exists() and not stopped
    assert plan["candidates"][0]["model_raw_score"] == .8
    assert plan["nonword_selection_status"] == "UNKNOWN" and plan["non_one_word_primary_candidate"] is None
    command, options = delegated[0]
    assert command[1] == str(freeze.ENTRY) and command[2] == "freeze"
    assert "--calendar" not in command and "--output" not in command
    assert options["shell"] is False


def test_unreviewed_draft_refuses_without_suspending_module(bundle):
    plan, path, calendar, output, _, _, _, _, stopped, delegated = bundle
    plan["daily_research"]["draft_status"] = freeze.apply.PENDING
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(freeze.FreezeGuardError, match="actual author review"):
        freeze.freeze_plan(path, calendar, output)
    assert not stopped and not delegated and not output.exists()


def test_guard_dependency_hash_change_stops_before_freeze(bundle):
    _, path, calendar, output, _, _, _, guard_path, stopped, delegated = bundle
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    guard["sources"][str(freeze.ENTRY.resolve())] = "0" * 64
    guard_path.write_text(json.dumps(guard), encoding="utf-8")
    with pytest.raises(freeze.FreezeGuardError, match="source hash conflict"):
        freeze.freeze_plan(path, calendar, output)
    assert len(stopped) == 1 and stopped[0][1] == "relay" and not delegated and not output.exists()


def test_active_proof_change_after_author_review_stops_before_freeze(bundle, monkeypatch):
    _, path, calendar, output, record, pointer, proof, _, stopped, delegated = bundle
    changed = deepcopy(proof)
    changed["active_sha256"] = "0" * 64
    monkeypatch.setattr(freeze.shadow, "verify_active_for_registration", lambda *_: (record, pointer, changed))
    with pytest.raises(freeze.FreezeGuardError, match="active_sha256"):
        freeze.freeze_plan(path, calendar, output)
    assert len(stopped) == 1 and not delegated and not output.exists()


def test_score_tampering_after_author_review_stops_before_freeze(bundle):
    plan, path, calendar, output, _, _, _, _, stopped, delegated = bundle
    plan["candidates"][0]["model_raw_score"] = float("nan")
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(freeze.FreezeGuardError, match="score"):
        freeze.freeze_plan(path, calendar, output)
    assert len(stopped) == 1 and not delegated and not output.exists()


def test_negative_manual_primary_never_becomes_trade_primary(bundle):
    plan, path, calendar, output, _, _, _, _, stopped, delegated = bundle
    plan["primary_candidate"] = "000002"
    plan["highest_accept_price"] = 12.0
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(freeze.FreezeGuardError, match="execution primary requires"):
        freeze.freeze_plan(path, calendar, output)
    assert len(stopped) == 1 and not delegated and not output.exists()


def test_deleted_application_binding_cannot_delegate_as_legacy_plan(bundle):
    plan, path, calendar, output, _, _, _, _, stopped, delegated = bundle
    plan.pop("model_upgrade_application")
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(freeze.FreezeGuardError, match="require original model_upgrade_application"):
        freeze.freeze_plan(path, calendar, output)
    assert not delegated and not output.exists()
