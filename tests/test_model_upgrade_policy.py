"""Synthetic human-authorization controls; no fitting or production mutation."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import model_upgrade_policy as policy
import model_upgrade_shadow as shadow


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def held(tmp_path, monkeypatch):
    clock = datetime(2030, 1, 2, 9, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(shadow, "now_utc", lambda: clock)
    path = _write(tmp_path / "protocol.json", {"version": "model-upgrade-fixed-1", "modules": {"market": {}, "relay": {}}})
    hold = {"version": policy.HOLD_VERSION, "formed_at": "2029-12-31T14:00:00Z",
            "protocol_path": str(path), "protocol_sha256": policy._sha(path), "status": "HOLD_ORIGINAL_BASE",
            "user_instruction": "SYNTHETIC human says retain BASE", "modules": ["market", "relay"],
            "allow_promote": False, "allow_apply": False, "allow_upgraded_freeze": False,
            "allow_shadow_prediction": True, "allow_shadow_settlement": True, "allow_fixed_review": True,
            "numerical_refits": 0, "feature_or_parameter_changed": False}
    _write(tmp_path / policy.HOLD_NAME, hold)
    return path, hold, clock


def test_passed_120_review_remains_research_without_auto_promotion(held, monkeypatch, capsys):
    path, _, _ = held
    receipt = {"result": {"upgrade_eligible": True, "selected_arm": "C1", "gates": {"C1": {"passed": True}}}}
    monkeypatch.setattr(shadow, "review_node", lambda *_: deepcopy(receipt))
    def forbidden(*_):
        raise AssertionError("hold must not promote after research pass")
    monkeypatch.setattr(shadow, "promote", forbidden)
    shadow.main(["review", "--protocol", str(path), "--module", "market", "--node", "120", "--output", "synthetic_receipt"])
    result = json.loads(capsys.readouterr().out)
    assert result["review"] == receipt
    assert result["promotion"]["status"] == "BLOCKED_USER_PRODUCTION_HOLD"
    assert result["promotion"]["selected_arm"] == "BASE"


def test_direct_promotion_cannot_release_hold_or_stop_research(held, monkeypatch):
    path, _, _ = held
    before = (path.parent / policy.HOLD_NAME).read_bytes()
    monkeypatch.setattr(shadow, "_protocol", lambda *_: (_ for _ in ()).throw(AssertionError("must refuse before promotion loads models")))
    with pytest.raises(policy.ProductionHoldError, match="USER_PRODUCTION_HOLD"):
        shadow.promote(path, "relay", "synthetic_accepted_receipt")
    assert (path.parent / policy.HOLD_NAME).read_bytes() == before
    assert not (path.parent / "shadow" / "active_pointer.json").exists()


def test_apply_hold_does_not_call_failure_rollback_and_base_stays_base(held, monkeypatch):
    path, _, _ = held
    root = path.parent / "shadow"
    root.mkdir()
    active = _write(root / "active_pointer.json", {})
    record = {"module": "market", "protocol_path": str(path)}
    registration = _write(root / "original_registration.json", record)
    monkeypatch.setattr(shadow, "_registered", lambda *_: (record, path, {}, {}, root))
    pointer = {"modules": {"market": {"active_arm": "C1"}}}
    monkeypatch.setattr(shadow, "_pointer", lambda *_: pointer)
    stopped = []
    monkeypatch.setattr(shadow, "_binding_failure", lambda *_: stopped.append(True))
    with pytest.raises(policy.ProductionHoldError, match="apply is forbidden"):
        shadow.verify_active_for_registration(registration, active)
    assert not stopped
    pointer["modules"]["market"] = {"active_arm": "BASE"}
    _, _, proof = shadow.verify_active_for_registration(registration, active)
    assert proof["status"] == "UNCHANGED_BASE" and not stopped


def test_deleting_hold_cannot_bypass_existing_policy_receipt(held):
    path, _, clock = held
    _write(path.parent / policy.LAYER_NAME, {"hold_path": str(path.parent / policy.HOLD_NAME)})
    (path.parent / policy.HOLD_NAME).unlink()
    with pytest.raises(FileNotFoundError):
        policy.require_allowed(path, "market", "promote", clock)


def test_new_protocol_without_explicit_replacement_authorization_defaults_base(tmp_path):
    with pytest.raises(policy.ProductionHoldError, match="NO_EXPLICIT_NEW_REPLACEMENT_AUTHORIZATION"):
        policy.require_allowed(tmp_path / "new_protocol.json", "market", "promote")


def _full_receipt_chain(path, hold):
    calendar = _write(path.parent / "calendar.json", {"status": "verified", "source": "SYNTHETIC_OFFICIAL",
        "scope": ["2030-01-01", "2030-01-03"], "sessions": ["2030-01-01", "2030-01-02", "2030-01-03"]})
    provenance = _write(path.parent / "provenance_BEFORE_FITS.json", {"provenance": {"source_hashes": {}, "dependency_hashes": {}}})
    protocol = {"version": "model-upgrade-fixed-1", "recorded_at": "2029-12-30T12:00:00Z",
                "calendar_path": str(calendar), "calendar_sha256": policy._sha(calendar),
                "provenance_path": str(provenance), "provenance_sha256": policy._sha(provenance),
                "modules": {"market": {}, "relay": {}}}
    _write(path, protocol)
    hold.update(protocol_sha256=policy._sha(path))
    _write(path.parent / policy.HOLD_NAME, hold)
    names = ("model_upgrade_research.py", "model_upgrade_history.py", "model_upgrade_inputs.py",
             "model_upgrade_shadow.py", "model_upgrade_metrics.py", "model_upgrade_apply.py")
    sources = {str(Path(shadow.__file__).with_name(name).resolve()): policy._sha(Path(shadow.__file__).with_name(name)) for name in names}
    implementation = _write(path.parent / "implementation_BEFORE_FITS.json", {"protocol_sha256": policy._sha(path), "sources": sources})
    snapshots = {}
    for source, digest in sources.items():
        saved = path.parent / "used_sources_preserved_after_fit" / Path(source).name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(Path(source).read_bytes())
        snapshots[source] = {"path": str(saved), "sha256": digest}
    common = {"protocol_sha256": policy._sha(path), "original_implementation_sha256": policy._sha(implementation),
              "numeric_refits": 0, "feature_formula_changed": False, "parameter_changed": False,
              "source_snapshots": snapshots, "corrected_sources": sources}
    after = _write(path.parent / "implementation_correction_AFTER_FITS.json", {**common, "formed_at": "2029-12-31T10:00:00Z"})
    before = _write(path.parent / "implementation_correction_BEFORE_FORWARD.json", {**common, "formed_at": "2029-12-31T12:00:00Z",
                                                                                  "prior_correction_sha256": policy._sha(after)})
    preimage = path.parent / "authorization_change_preimage" / "model_upgrade_shadow.py"
    preimage.parent.mkdir()
    preimage.write_bytes(Path(shadow.__file__).read_bytes())
    source_policy = str(Path(policy.__file__).resolve())
    layer = {"version": policy.LAYER_VERSION, "formed_at": "2029-12-31T15:00:00Z",
             "protocol_sha256": policy._sha(path), "parent_path": str(before), "parent_sha256": policy._sha(before),
             "hold_path": str(path.parent / policy.HOLD_NAME), "hold_sha256": policy._sha(path.parent / policy.HOLD_NAME),
             "preimage_path": str(preimage), "preimage_sha256": policy._sha(preimage),
             "sources": {**sources, source_policy: policy._sha(source_policy)},
             "numerical_refits": 0, "feature_or_parameter_changed": False}
    return _write(path.parent / policy.LAYER_NAME, layer), layer


def test_bound_hold_layer_preserves_prior_chain_and_all_current_sources(held):
    path, hold, _ = held
    _, _ = _full_receipt_chain(path, hold)
    before = (path.parent / "implementation_correction_BEFORE_FORWARD.json").read_bytes()
    _, protocol, calendar, _ = shadow._protocol(path)
    assert protocol["modules"] == {"market": {}, "relay": {}}
    assert calendar["sessions"][-1] == "2030-01-03"
    assert (path.parent / "implementation_correction_BEFORE_FORWARD.json").read_bytes() == before
    assert policy.load_hold(path, shadow.now_utc())["allow_shadow_prediction"] is True


@pytest.mark.parametrize("conflict", ["parent", "preimage", "other_source", "refit", "delete_hold"])
def test_policy_receipt_conflict_never_weakens_prior_bindings(held, conflict):
    path, hold, _ = held
    layer_path, layer = _full_receipt_chain(path, hold)
    if conflict == "parent":
        layer["parent_sha256"] = "0" * 64
    elif conflict == "preimage":
        layer["preimage_sha256"] = "0" * 64
    elif conflict == "other_source":
        source = str(Path(shadow.__file__).with_name("model_upgrade_metrics.py").resolve())
        layer["sources"][source] = "0" * 64
    elif conflict == "refit":
        layer["numerical_refits"] = 1
    else:
        (path.parent / policy.HOLD_NAME).unlink()
    _write(layer_path, layer)
    with pytest.raises((ValueError, FileNotFoundError)):
        shadow._protocol(path)


def _authorize(path, hold):
    hold_layer_path, hold_layer = _full_receipt_chain(path, hold)
    original_policy = path.parent / "authorization_evidence_gated_20261008_preimage" / "model_upgrade_policy.py"
    original_policy.parent.mkdir()
    original_policy.write_bytes(Path(policy.__file__).read_bytes())
    authorization = {"version": policy.AUTHORIZATION_VERSION, "formed_at": "2030-01-02T09:00:00Z",
        "protocol_path": str(path), "protocol_sha256": policy._sha(path),
        "prior_hold_path": str(path.parent / policy.HOLD_NAME), "prior_hold_sha256": policy._sha(path.parent / policy.HOLD_NAME),
        "status": "EVIDENCE_GATED_REPLACEMENT_AUTHORIZED", "source_kind": "HUMAN_USER_MESSAGE_CURRENT_TASK",
        "user_instruction": policy.AUTHORIZED_USER_INSTRUCTION,
        "user_instruction_sha256": policy.hashlib.sha256(policy.AUTHORIZED_USER_INSTRUCTION.encode("utf-8")).hexdigest(),
        "modules": ["market", "relay"], "allow_promote": True, "allow_apply": True, "allow_upgraded_freeze": True,
        "existing_fixed_evidence_gates_required": True, "allow_immediate_ungated_replacement": False,
        "numerical_refits": 0, "feature_or_parameter_changed": False}
    auth_path = _write(path.parent / policy.AUTHORIZATION_NAME, authorization)
    overlay = {"version": policy.AUTH_LAYER_VERSION, "formed_at": "2030-01-02T09:01:00Z",
        "protocol_sha256": policy._sha(path), "parent_path": str(hold_layer_path), "parent_sha256": policy._sha(hold_layer_path),
        "hold_path": str(path.parent / policy.HOLD_NAME), "hold_sha256": policy._sha(path.parent / policy.HOLD_NAME),
        "authorization_path": str(auth_path), "authorization_sha256": policy._sha(auth_path),
        "preimage_path": str(original_policy), "preimage_sha256": policy._sha(original_policy),
        "sources": hold_layer["sources"], "numerical_refits": 0, "feature_or_parameter_changed": False}
    overlay_path = _write(path.parent / policy.AUTH_LAYER_NAME, overlay)
    return auth_path, authorization, overlay_path, overlay


def test_new_explicit_authorization_releases_only_permission_preserving_full_chain(held):
    path, hold, clock = held
    _authorize(path, hold)
    original_hold = (path.parent / policy.HOLD_NAME).read_bytes()
    old_layer = (path.parent / policy.LAYER_NAME).read_bytes()
    proof = policy.require_allowed(path, "relay", "promote", clock)
    assert proof["status"] == "AUTHORIZED_SUBJECT_TO_UNCHANGED_EVIDENCE_GATES"
    assert proof["authorization"]["existing_fixed_evidence_gates_required"] is True
    shadow._protocol(path)
    assert (path.parent / policy.HOLD_NAME).read_bytes() == original_hold
    assert (path.parent / policy.LAYER_NAME).read_bytes() == old_layer
    assert not (path.parent / "shadow" / "active_pointer.json").exists()


@pytest.mark.parametrize("conflict", ["future", "fake_source", "fake_instruction", "hold_hash", "protocol_hash", "ungated", "no_overlay"])
def test_new_authorization_conflicts_do_not_release_hold(held, conflict):
    path, hold, clock = held
    auth_path, authorization, overlay_path, _ = _authorize(path, hold)
    if conflict == "future":
        authorization["formed_at"] = "2030-01-03T09:00:00Z"
    elif conflict == "fake_source":
        authorization["source_kind"] = "MODEL_AUTHOR_REVIEW"
    elif conflict == "fake_instruction":
        authorization["user_instruction"] = "Research looks promising"
    elif conflict == "hold_hash":
        authorization["prior_hold_sha256"] = "0" * 64
    elif conflict == "protocol_hash":
        authorization["protocol_sha256"] = "0" * 64
    elif conflict == "ungated":
        authorization["allow_immediate_ungated_replacement"] = True
    else:
        overlay_path.unlink()
    _write(auth_path, authorization)
    with pytest.raises((ValueError, FileNotFoundError)):
        policy.require_allowed(path, "market", "apply", clock)


@pytest.mark.parametrize("conflict", ["parent", "preimage", "other_source", "future", "refit", "authorization_hash", "delete_authorization"])
def test_new_overlay_preserves_original_implementation_bindings(held, conflict):
    path, hold, clock = held
    auth_path, _, overlay_path, overlay = _authorize(path, hold)
    if conflict == "parent":
        overlay["parent_sha256"] = "0" * 64
    elif conflict == "preimage":
        overlay["preimage_sha256"] = "0" * 64
    elif conflict == "other_source":
        source = str(Path(shadow.__file__).with_name("model_upgrade_metrics.py").resolve())
        overlay["sources"][source] = "0" * 64
    elif conflict == "future":
        overlay["formed_at"] = "2030-01-03T09:00:00Z"
    elif conflict == "refit":
        overlay["numerical_refits"] = 1
    elif conflict == "authorization_hash":
        overlay["authorization_sha256"] = "0" * 64
    else:
        auth_path.unlink()
    _write(overlay_path, overlay)
    with pytest.raises((ValueError, FileNotFoundError)):
        policy.require_allowed(path, "relay", "promote", clock)


def test_authorization_cannot_bypass_existing_mature_evidence_gate(held, monkeypatch):
    path, hold, _ = held
    _authorize(path, hold)
    root = path.parent / "shadow"
    root.mkdir()
    receipt = _write(root / "review120.json", {"node": 120, "module": "relay", "protocol_sha256": policy._sha(path),
        "result": {"upgrade_eligible": True, "selected_arm": "C1", "gates": {"C1": {"passed": True}}}})
    monkeypatch.setattr(shadow, "_protocol", lambda *_: (path, {}, {}, root))
    monkeypatch.setattr(shadow, "_receipt_binding_check", lambda *_: None)
    monkeypatch.setattr(shadow, "review_node", lambda *_: {"result": {"upgrade_eligible": False}})
    with pytest.raises(ValueError, match="promotion gate not passed"):
        shadow.promote(path, "relay", receipt)
    assert not (root / "active_pointer.json").exists()
