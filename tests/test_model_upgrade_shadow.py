"""Synthetic fixtures only: no real data, model fit, forward run or deployment."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import joblib
import numpy as np
import pytest

import model_upgrade_shadow as shadow


class FixedScores:
    def __init__(self, module):
        self.classes_ = np.array(["CONTINUE", "IMPROVE", "WORSEN"] if module == "market" else ["0", "1"])

    def predict_proba(self, frame):
        scores = [.2, .7, .1] if len(self.classes_) == 3 else [.3, .7]
        return np.tile(scores, (len(frame), 1))

    def fit(self, *args, **kwargs):
        raise AssertionError("shadow must never fit")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    # This is an explicit fictional official-calendar fixture, never production.
    clock = [shadow.stamp("2026-10-08T08:00:00+00:00")]
    monkeypatch.setattr(shadow, "now_utc", lambda: clock[0])
    source = tmp_path / "synthetic_source.json"
    source.write_text("synthetic raw response", encoding="utf-8")
    source_hashes = {str(source): shadow.file_sha256(source)}
    calendar = write(tmp_path / "calendar.json", {
        "status": "verified", "source": ["synthetic://fixture/calendar"],
        "scope": ["2026-10-08", "2026-10-12"], "sessions": ["2026-10-08", "2026-10-09", "2026-10-12"]})
    protocol = {"version": "model-upgrade-fixed-1", "recorded_at": "2026-10-07T10:00:00+00:00",
                "freeze_cutoff": "2026-09-01", "calendar_path": str(calendar),
                "calendar_sha256": shadow.file_sha256(calendar), "modules": {},
                "forward_sessions": 120, "min_market_pairs": 108, "min_class_support": 20,
                "min_relay_pairs": 60, "min_coverage": .9, "min_improvement": .03,
                "bootstrap": {"draws": 2000, "seed": 20261007, "block_sessions": 5,
                              "tails": [.00625, .99375]}, "preferred_arm": "C1",
                "rollback": {"sessions": 60, "deterioration": .05}}
    provenance = write(tmp_path / "provenance_BEFORE_FITS.json", {
        "provenance": {"source_hashes": source_hashes, "dependency_hashes": {}}})
    protocol.update(provenance_path=str(provenance), provenance_sha256=shadow.file_sha256(provenance))
    for module in ("market", "relay"):
        protocol["modules"][module] = {
            "arms": {a: a for a in ("BASE", "C1", "C2")},
            "model_manifests": {a: f"models/{module}/{a}/manifest.json" for a in ("BASE", "C1", "C2")}}
    proto_path = write(tmp_path / "protocol.json", protocol)
    code_bindings = {str(Path(shadow.__file__).with_name(name).resolve()): shadow.file_sha256(Path(shadow.__file__).with_name(name))
                     for name in ("model_upgrade_shadow.py", "model_upgrade_metrics.py", "model_upgrade_inputs.py")}
    write(tmp_path / "implementation_BEFORE_FITS.json", {
        "protocol_sha256": shadow.file_sha256(proto_path), "source_hashes": source_hashes,
        "sources": code_bindings})
    for module in protocol["modules"]:
        for arm in ("BASE", "C1", "C2"):
            folder = tmp_path / "models" / module / arm
            folder.mkdir(parents=True)
            model_path = folder / "model.joblib"
            model = FixedScores(module)
            joblib.dump(model, model_path)
            write(folder / "manifest.json", {
                "module": module, "arm": arm, "fields": ["x"], "model_path": str(model_path),
                "sha256": shadow.file_sha256(model_path), "classes": model.classes_.tolist(),
                "training_max_label_date": "2026-08-31", "formed_at": "2026-10-07T11:00:00+00:00",
                "frozen_at": "2026-10-07T11:05:00+00:00",
                "cutoff": "2026-09-01", "sources": source_hashes,
                "runtime_versions": {p: shadow.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "joblib")},
                "training_majority": "CONTINUE" if module == "market" else "0",
                "protocol_sha256": shadow.file_sha256(proto_path)})
    pack = {"signal_date": "2026-10-08", "available_at": "2026-10-08T07:30:00+00:00",
            "calendar": {}, "market": {"features": {"x": 2.}, "quality": 1.,
                                        "status": "verified", "persistence": "CONTINUE"},
            "relay": [{"code": "000001", "boards": 3, "name": "synthetic A", "actual_one_word_daily": 1,
                       "features": {"x": 2.}},
                      {"code": "000002", "boards": 2, "name": "synthetic B", "actual_one_word_daily": 0,
                       "features": {"x": 1.}}], "sources": source_hashes, "scope": "synthetic all"}
    outcome_source = tmp_path / "outcome_response.json"
    outcome_source.write_text("synthetic independent response", encoding="utf-8")
    outcome = {"market_target": "IMPROVE", "candidates": {"000001": True, "000002": None},
               "sources": {str(outcome_source): shadow.file_sha256(outcome_source)}}
    calls = {"outcomes": 0}

    def build(record, archive):
        calls["outcomes"] += 1
        return deepcopy(outcome)

    monkeypatch.setitem(sys.modules, "model_upgrade_inputs", SimpleNamespace(
        load_feature_archive=lambda path, cutoff=None: deepcopy(pack), build_outcomes=build))
    return SimpleNamespace(path=proto_path, protocol=protocol, calendar=calendar, clock=clock,
                           pack=pack, outcome=outcome, source=source, calls=calls,
                           root=tmp_path / "shadow", feature_path=tmp_path / "feature_archive")


def register(env, module="market", name="first.json"):
    output = env.root / module / name
    record = shadow.predict(env.path, module, env.feature_path, env.path.parent / "models", output)
    return output, record


def test_true_shanghai_date_and_closed_time():
    days = ["2026-10-08", "2026-10-09"]
    shadow.require_current_closed_signal("2026-10-08", days, "2026-10-08T07:00:00+00:00")
    with pytest.raises(ValueError, match="not closed"):
        shadow.require_current_closed_signal("2026-10-08", days, "2026-10-08T06:59:59+00:00")
    with pytest.raises(ValueError, match="true current"):
        shadow.require_current_closed_signal("2026-10-08", days, "2026-10-09T08:00:00+00:00")
    with pytest.raises(ValueError, match="timezone"):
        shadow.stamp("2026-10-08T16:00:00")


def test_official_grid_holiday_gap_and_partial_prefix():
    days = ["2026-09-30", "2026-10-08", "2026-10-09"]
    assert shadow.next_official_session(days, "2026-09-30") == "2026-10-08"
    assert shadow.official_window_grid(days, "2026-10-08") == days[1:]
    with pytest.raises(ValueError, match="extension"):
        shadow.official_window_grid(days, "2026-10-08", require_full=True)
    with pytest.raises(ValueError, match="future"):
        shadow.validate_cutoff("2026-10-08", "2026-10-08T09:00:00Z", "2026-10-08T08:00:00Z")


def test_registration_raw_scores_without_labels_and_no_fit(env):
    path, record = register(env)
    assert "outcomes" not in record
    assert record["target_date"] == "2026-10-09"
    assert record["arms"]["BASE"]["raw_scores"]["IMPROVE"] == .7
    assert record["original_codes"] == ["000001", "000002"]
    assert record["score_calibrated"] is False
    assert record["production_model_changed"] is False
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already frozen"):
        register(env, name="second.json")
    assert path.read_bytes() == before
    assert not (env.root / "market" / "second.json").exists()


def test_all_unknown_features_remain_unknown(env):
    env.pack["market"]["features"]["x"] = None
    _, record = register(env)
    assert all(row["status"] == "UNKNOWN" and row["prediction"] is None for row in record["arms"].values())


def test_unknown_oneword_keeps_complete_original_roster(env):
    env.pack["relay"][1]["actual_one_word_daily"] = None
    _, record = register(env, "relay")
    assert record["naive"]["highest_nonword"] is None
    for arm in record["arms"].values():
        assert arm["status"] == "KNOWN" and arm["primary_eligible"] is None
        assert [r["code"] for r in arm["candidates"]] == ["000001", "000002"]
        assert all(r["raw_score"] == .7 for r in arm["candidates"])
    assert not (env.root / "relay" / "window.json").exists()


def test_nonword_baseline_and_raw_positive_score(env):
    _, record = register(env, "relay")
    assert record["naive"]["highest_nonword"] == "000002"
    assert all(row["raw_score"] == .7 and row["predicted_positive"] is True
               for row in record["arms"]["BASE"]["candidates"])


def test_output_cannot_touch_production(env):
    protected = env.path.parent / "production.json"
    protected.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="shadow directory"):
        shadow.predict(env.path, "market", env.feature_path, env.path.parent / "models", protected)
    assert protected.read_text(encoding="utf-8") == "unchanged"


def test_settlement_before_close_rejected_before_adapter(env):
    path, _ = register(env)
    with pytest.raises(ValueError, match="not closed"):
        shadow.settle(path, env.path.parent / "new_outcome_archive")
    assert env.calls["outcomes"] == 0
    assert not (env.root / "settlements.jsonl").exists()


def test_settlement_unknown_and_append_revision_preserve_original(env):
    path, _ = register(env)
    original = path.read_bytes()
    env.clock[0] = shadow.stamp("2026-10-09T08:00:00Z")
    first = shadow.settle(path, env.path.parent / "new_outcome_archive")
    assert first["record"]["outcomes"]["candidates"]["000002"] is None
    assert first["record"]["candidate_group_complete"] is False
    same = shadow.settle(path, env.path.parent / "new_outcome_archive")
    assert same["entry_hash"] == first["entry_hash"]
    env.outcome["candidates"]["000002"] = False
    second = shadow.settle(path, env.path.parent / "new_outcome_archive_2")
    assert second["revision"] == 2
    assert second["record"]["candidate_group_complete"] is True
    assert path.read_bytes() == original
    assert len(shadow._read_chain(env.root / "settlements.jsonl")) == 2


def test_hash_conflict_stops_challenge_and_keeps_original(env):
    path, _ = register(env)
    original = path.read_bytes()
    env.source.write_text("changed raw response", encoding="utf-8")
    env.clock[0] = shadow.stamp("2026-10-09T08:00:00Z")
    with pytest.raises(ValueError, match="hash binding conflict"):
        shadow.settle(path, env.path.parent / "new_outcome_archive")
    pointer = shadow._pointer(env.root)["modules"]["market"]
    assert pointer["active_arm"] == "BASE" and pointer["suspended"] is True
    assert path.read_bytes() == original
    assert not (env.root / "settlements.jsonl").exists()


def test_rollback_suspends_challengers_in_future_new_signal(env):
    register(env)
    shadow.rollback(env.path, "market", "synthetic fixed node decline")
    env.clock[0] = shadow.stamp("2026-10-09T08:00:00Z")
    env.pack["signal_date"] = "2026-10-09"
    env.pack["available_at"] = "2026-10-09T07:30:00Z"
    _, record = register(env, name="next.json")
    assert record["arms"]["BASE"]["status"] == "KNOWN"
    assert all(record["arms"][a]["status"] == "UNKNOWN" for a in ("C1", "C2"))
    window = json.loads((env.root / "market" / "window.json").read_text(encoding="utf-8"))
    assert window["first_signal_date"] == "2026-10-08"


def test_calendar_extension_retains_original_and_rejects_rewrite(env):
    before = env.calendar.read_bytes()
    extension = write(env.path.parent / "extension.json", {
        "status": "verified", "source": ["synthetic://later/official"],
        "scope": ["2026-10-08", "2026-10-13"],
        "sessions": ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]})
    shadow.bind_calendar(env.path, extension)
    assert shadow._protocol(env.path)[2]["sessions"][-1] == "2026-10-13"
    assert env.calendar.read_bytes() == before
    bad = write(env.path.parent / "bad_extension.json", {
        "status": "verified", "source": ["synthetic://bad"], "scope": ["2026-10-08", "2026-10-14"],
        "sessions": ["2026-10-08", "2026-10-12", "2026-10-13", "2026-10-14"]})
    with pytest.raises(ValueError, match="original official prefix"):
        shadow.bind_calendar(env.path, bad)


def test_fixed_acceptance_cannot_mature_with_three_calendar_dates(env, monkeypatch):
    # Isolate the data gate under fictional explicit authorization. Production
    # authorization itself is tested separately and remains held by default.
    monkeypatch.setattr(shadow.production_policy, "require_allowed", lambda *_: None)
    register(env)
    env.clock[0] = shadow.stamp("2026-10-12T08:00:00Z")
    receipt = shadow.review_node(env.path, "market", 120, env.root / "reviews" / "first.json")
    assert receipt["result"]["status"] == "NOT_MATURE"
    assert receipt["result"]["upgrade_eligible"] is False
    with pytest.raises(ValueError, match="gate not passed"):
        shadow.promote(env.path, "market", env.root / "reviews" / "first.json")
    assert not (env.root / "active_pointer.json").exists()


def test_forged_receipt_does_not_bypass_live_acceptance(env, monkeypatch):
    # Fictional unit-only authorization ensures the live evidence gate is reached.
    monkeypatch.setattr(shadow.production_policy, "require_allowed", lambda *_: None)
    register(env)
    env.clock[0] = shadow.stamp("2026-10-12T08:00:00Z")
    receipt = shadow.review_node(env.path, "market")
    receipt["result"] = {"upgrade_eligible": True, "selected_arm": "C1", "gates": {"C1": {"passed": True}}}
    forged = write(env.root / "reviews" / "forged.json", receipt)
    with pytest.raises(ValueError, match="gate not passed"):
        shadow.promote(env.path, "market", forged)
    assert not (env.root / "active_pointer.json").exists()


def test_pointer_tampering_is_rejected_by_append_chain(env):
    shadow.rollback(env.path, "market", "synthetic")
    pointer = json.loads((env.root / "active_pointer.json").read_text(encoding="utf-8"))
    pointer["modules"]["market"]["active_arm"] = "C2"
    write(env.root / "active_pointer.json", pointer)
    with pytest.raises(ValueError, match="binding conflict"):
        shadow._pointer(env.root)


@pytest.mark.parametrize("change,pattern", [
    ({"formed_at": "2026-10-08T09:00:00Z"}, "formation"),
    ({"training_max_label_date": "2026-09-01"}, "training labels"),
    ({"fields": ["next_return"]}, "future truth"),
    ({"classes": ["0", "1"]}, "class definition"),
])
def test_frozen_knowledge_and_target_contract_rejects(env, change, pattern):
    path = env.path.parent / "models" / "market" / "BASE" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    write(path, {**manifest, **change})
    with pytest.raises(ValueError, match=pattern):
        register(env)
    assert not (env.root / "registrations.jsonl").exists()


def test_archive_cannot_form_after_actual_cutoff(env):
    env.pack["available_at"] = "2026-10-08T08:00:01Z"
    with pytest.raises(ValueError, match="formed after"):
        register(env)


def test_first_window_cannot_shift_to_drop_initial_days(env):
    register(env)
    window_path = env.root / "market" / "window.json"
    window = json.loads(window_path.read_text(encoding="utf-8"))
    window["first_signal_date"] = "2026-10-09"
    write(window_path, window)
    with pytest.raises(ValueError, match="window hash conflict"):
        shadow.review_node(env.path, "market", 120)
    assert shadow._pointer(env.root)["modules"]["market"]["suspended"] is True


def test_receipt_retains_original_audited_prefix_after_later_append(env):
    register(env)
    receipt = shadow.review_node(env.path, "market")
    shadow._append_chain(env.root / "registrations.jsonl", {
        "event": "SYNTHETIC_APPEND_CHECK_ONLY", "module": "unused", "signal_date": "2026-10-09"})
    shadow._receipt_binding_check(receipt)
    ledger = env.root / "registrations.jsonl"
    text = ledger.read_text(encoding="utf-8")
    ledger.write_text(text.replace("SYNTHETIC_APPEND_CHECK_ONLY", "ALTERED"), encoding="utf-8")
    with pytest.raises(ValueError, match="chain binding conflict"):
        shadow._receipt_binding_check(receipt)


def test_base_proof_never_claims_active_upgrade(env):
    path, record = register(env)
    verified_record, pointer, proof = shadow.verify_active_for_registration(path, env.root / "active_pointer.json")
    assert verified_record == record
    assert pointer["active_arm"] == "BASE"
    assert proof["status"] == "UNCHANGED_BASE"


def test_all_unknown_first_day_waits_then_locks_first_usable_signal(env, monkeypatch):
    env.pack["market"]["features"]["x"] = None
    register(env)
    original_pack = deepcopy(env.pack)
    monkeypatch.setattr(sys.modules["model_upgrade_inputs"], "load_feature_archive",
                        lambda path, cutoff=None: deepcopy(original_pack if cutoff and cutoff.startswith("2026-10-08") else env.pack))
    assert not (env.root / "market" / "window.json").exists()
    env.clock[0] = shadow.stamp("2026-10-09T08:00:00Z")
    env.pack.update(signal_date="2026-10-09", available_at="2026-10-09T07:30:00Z")
    env.pack["market"]["features"]["x"] = 2.
    register(env, name="first_usable.json")
    window = json.loads((env.root / "market" / "window.json").read_text(encoding="utf-8"))
    assert window["first_signal_date"] == "2026-10-09"
    reviewed = shadow._review_inputs(env.path, "market")[4]
    assert [r["signal_date"] for r in reviewed] == ["2026-10-09"]


def test_partial_market_cannot_start_fixed_window(env):
    env.pack["market"]["status"] = "partial"
    _, record = register(env)
    assert record["arms"]["BASE"]["status"] == "KNOWN"
    assert not (env.root / "market" / "window.json").exists()


@pytest.mark.parametrize("frozen", ["2026-10-07T10:30:00Z", "2026-10-08T08:00:01Z"])
def test_models_must_be_actually_frozen_before_prediction(env, frozen):
    manifest_path = env.path.parent / "models" / "market" / "C2" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write(manifest_path, {**manifest, "frozen_at": frozen})
    with pytest.raises(ValueError, match="freeze outside"):
        register(env)


def test_runtime_change_rejects_prediction(env, monkeypatch):
    actual_version = shadow.metadata.version
    monkeypatch.setattr(shadow.metadata, "version", lambda p: "changed" if p == "joblib" else actual_version(p))
    with pytest.raises(ValueError, match="runtime version binding conflict"):
        register(env)
    assert shadow._pointer(env.root)["modules"]["market"]["suspended"] is True


def test_cli_review_reports_acceptance_without_replacement_authorization(monkeypatch, capsys):
    calls = []
    receipt = {"result": {"upgrade_eligible": True, "selected_arm": "C1", "gates": {"C1": {"passed": True}}}}
    monkeypatch.setattr(shadow, "review_node", lambda *a: receipt)
    monkeypatch.setattr(shadow, "promote", lambda *a: calls.append(a) or {"synthetic_dispatch": True})
    shadow.main(["review", "--protocol", "synthetic", "--module", "market", "--node", "120", "--output", "receipt"])
    assert not calls
    result = json.loads(capsys.readouterr().out)
    assert result["review"] == receipt
    assert result["promotion"]["status"] == "BLOCKED_USER_PRODUCTION_HOLD"
    assert result["promotion"]["selected_arm"] == "BASE"
    assert "NO_EXPLICIT_NEW_REPLACEMENT_AUTHORIZATION" in result["promotion"]["reason"]


def test_manifest_training_majority_dictionary_preserves_naive_baseline(env):
    for arm in ("BASE", "C1", "C2"):
        path = env.path.parent / "models" / "market" / arm / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        write(path, {**manifest, "training_majority": {"class": "CONTINUE", "counts": {"CONTINUE": 20}}})
    _, record = register(env)
    assert record["naive"]["majority"] == "CONTINUE"


def test_fixed_sixty_nodes_are_nonoverlapping_and_not_repeated(env, monkeypatch):
    from datetime import date, timedelta
    import model_upgrade_metrics
    sessions = [(date(2026, 10, 8) + timedelta(days=i)).isoformat() for i in range(123)]
    env.clock[0] = shadow.stamp(sessions[-1] + "T08:00:00Z")
    pointer = {"modules": {"market": {"active_arm": "C1", "effective_signal": sessions[0], "suspended": False}}}
    shadow._save_pointer(env.root, pointer, "SYNTHETIC_FIXED_NODE_FIXTURE")
    rows = [{"signal_date": d} for d in sessions[:-1]]
    derived = {"first_signal_date": sessions[0], "closed_through_date": sessions[-1]}
    calendar = {"sessions": sessions}
    monkeypatch.setattr(shadow, "_review_inputs", lambda *a: (env.path, deepcopy(derived), calendar, env.root, rows, {}))
    calls = []
    monkeypatch.setattr(model_upgrade_metrics, "rollback_check", lambda module, selected, protocol, days:
                        calls.append([r["signal_date"] for r in selected]) or {"rollback": False, "selected_arm": "C1"})
    first = shadow.review_node(env.path, "market", 60)
    second = shadow.review_node(env.path, "market", 60)
    repeated = shadow.review_node(env.path, "market", 60)
    assert first["node_index"] == 1 and second["node_index"] == 2
    assert calls == [sessions[:60], sessions[60:120]]
    assert repeated["result"]["status"] == "ALREADY_CHECKED"
    assert len(shadow._read_chain(env.root / "market" / "rollback_nodes.jsonl")) == 2


def test_protocol_immutable_provenance_fallback_uses_bound_implementation(env):
    protocol = json.loads(env.path.read_text(encoding="utf-8"))
    provenance_hash = protocol.pop("provenance_sha256")
    protocol.pop("provenance_path")
    write(env.path, protocol)
    snapshot_path = env.path.parent / "implementation_BEFORE_FITS.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot.update(protocol_sha256=shadow.file_sha256(env.path), provenance_sha256=provenance_hash)
    write(snapshot_path, snapshot)
    assert shadow._protocol(env.path)[1]["version"] == "model-upgrade-fixed-1"
    provenance = env.path.parent / "provenance_BEFORE_FITS.json"
    provenance.write_text("changed synthetic provenance", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance receipt hash conflict"):
        shadow._protocol(env.path)


def test_python_runtime_change_is_rejected(env):
    manifest_path = env.path.parent / "models" / "market" / "BASE" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write(manifest_path, {**manifest, "python_version": "synthetic different Python"})
    with pytest.raises(ValueError, match="Python runtime version binding conflict"):
        register(env)


def synthetic_correction(env):
    names = ["model_upgrade_research.py", "model_upgrade_history.py", "model_upgrade_inputs.py",
             "model_upgrade_shadow.py", "model_upgrade_metrics.py", "model_upgrade_apply.py"]
    allowed = {"model_upgrade_research.py", "model_upgrade_inputs.py", "model_upgrade_shadow.py"}
    originals, snapshots, corrected = {}, {}, {}
    preserve = env.path.parent / "used_sources_preserved_after_fit"
    preserve.mkdir()
    for name in names:
        current = Path(shadow.__file__).with_name(name).resolve()
        saved = preserve / name
        # Fake prior code lives only in this temporary fixture. Real files stay untouched.
        saved.write_bytes(("synthetic previously used " + name).encode() if name in allowed else current.read_bytes())
        originals[str(current)] = shadow.file_sha256(saved)
        snapshots[str(current)] = {"path": str(saved), "sha256": originals[str(current)]}
        corrected[str(current)] = shadow.file_sha256(current)
    implementation_path = env.path.parent / "implementation_BEFORE_FITS.json"
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    write(implementation_path, {**implementation, "sources": originals})
    correction = {"protocol_sha256": shadow.file_sha256(env.path),
                  "original_implementation_sha256": shadow.file_sha256(implementation_path),
                  "formed_at": env.clock[0].isoformat(), "source_snapshots": snapshots,
                  "corrected_sources": corrected, "numeric_refits": 0,
                  "feature_formula_changed": False, "parameter_changed": False}
    return write(env.path.parent / "implementation_correction_AFTER_FITS.json", correction)


def test_post_fit_correction_overlay_preserves_original_receipt(env):
    correction_path = synthetic_correction(env)
    original_path = env.path.parent / "implementation_BEFORE_FITS.json"
    before = original_path.read_bytes()
    assert shadow._protocol(env.path)[1]["version"] == "model-upgrade-fixed-1"
    assert original_path.read_bytes() == before
    correction_path.rename(correction_path.with_suffix(".preserved"))
    with pytest.raises(ValueError, match="source hash binding conflict"):
        shadow._protocol(env.path)


def test_post_fit_correction_rejects_protected_change_or_escaped_snapshot(env):
    correction_path = synthetic_correction(env)
    original = json.loads(correction_path.read_text(encoding="utf-8"))
    tampered = deepcopy(original)
    metrics = str(Path(shadow.__file__).with_name("model_upgrade_metrics.py").resolve())
    tampered["corrected_sources"][metrics] = "0" * 64
    write(correction_path, tampered)
    with pytest.raises(ValueError, match="cannot alter history, metrics or apply"):
        shadow._protocol(env.path)
    tampered = deepcopy(original)
    source = str(Path(shadow.__file__).resolve())
    saved = Path(tampered["source_snapshots"][source]["path"])
    outside = env.path.parent / "outside_saved_source.py"
    outside.write_bytes(saved.read_bytes())
    tampered["source_snapshots"][source]["path"] = str(outside)
    write(correction_path, tampered)
    with pytest.raises(ValueError, match="outside preservation directory"):
        shadow._protocol(env.path)


def test_pre_forward_correction_chain_rejects_tamper_and_late_window_change(env):
    from datetime import timedelta
    after_path = synthetic_correction(env)
    after = json.loads(after_path.read_text(encoding="utf-8"))
    implementation_path = env.path.parent / "implementation_BEFORE_FITS.json"
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    metrics = str(Path(shadow.__file__).with_name("model_upgrade_metrics.py").resolve())
    prior_metrics = Path(after["source_snapshots"][metrics]["path"])
    prior_metrics.write_bytes(b"synthetic original metrics before forward-gate correction")
    initial_hash = shadow.file_sha256(prior_metrics)
    implementation["sources"][metrics] = initial_hash
    write(implementation_path, implementation)
    after["original_implementation_sha256"] = shadow.file_sha256(implementation_path)
    after["source_snapshots"][metrics]["sha256"] = initial_hash
    after["corrected_sources"][metrics] = initial_hash
    after["formed_at"] = (env.clock[0] - timedelta(seconds=2)).isoformat()
    write(after_path, after)
    before = deepcopy(after)
    before.update(prior_correction_sha256=shadow.file_sha256(after_path), formed_at=env.clock[0].isoformat())
    before["corrected_sources"][metrics] = shadow.file_sha256(metrics)
    before_path = write(env.path.parent / "implementation_correction_BEFORE_FORWARD.json", before)
    assert shadow._protocol(env.path)[1]["version"] == "model-upgrade-fixed-1"
    tampered = {**before, "prior_correction_sha256": "0" * 64}
    write(before_path, tampered)
    with pytest.raises(ValueError, match="prior correction hash binding conflict"):
        shadow._protocol(env.path)
    shadow._append_chain(env.root / "registrations.jsonl", {"event": "REGISTRATION", "module": "market"})
    env.clock[0] += timedelta(seconds=2)
    late = {**before, "formed_at": (env.clock[0] - timedelta(seconds=1)).isoformat()}
    write(before_path, late)
    with pytest.raises(ValueError, match="already started real registration window"):
        shadow._protocol(env.path)
