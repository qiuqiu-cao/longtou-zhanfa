import json
from pathlib import Path

import pytest

import model_upgrade_compat as compat
import model_upgrade_inputs as native_inputs
import model_upgrade_shadow as native_shadow
from model_upgrade_research import sha, read

ROOT = Path(__file__).resolve().parents[1]
COMPAT_ROOT = ROOT/"research/model_upgrade_validation/shadow_compat_20261009T093237Z"
POLICY = COMPAT_ROOT/"compatibility_BEFORE_USE.json"
FIXED = ROOT/"research/model_upgrade_validation/fixed_20261007T112017Z"
PROTOCOL = FIXED/"protocol_BEFORE_EFFECTS.json"
ARCHIVE = ROOT/"data/v293/outcome_link_20261009T0920Z"


@pytest.fixture(scope="module")
def isolated():
    return compat.runtime(POLICY)


def test_only_approved_extension_is_recognized_with_original_core_code_unchanged(isolated):
    inputs, shadow = isolated
    policy, proof = compat.verify_policy(POLICY)
    assert len(policy["original_numeric_code_hashes"]) == 8
    assert inputs._derive.__code__ is native_inputs._derive.__code__
    assert shadow._protocol.__code__ is native_shadow._protocol.__code__
    assert shadow._registered.__code__ is native_shadow._registered.__code__
    assert inputs._derive.__globals__ is not native_inputs._derive.__globals__
    assert shadow._registered.__globals__ is not native_shadow._registered.__globals__
    assert str(POLICY.resolve()) in proof
    path, protocol, _, _ = shadow._protocol(PROTOCOL)
    assert path == PROTOCOL
    assert protocol["forward_sessions"] == 120
    assert protocol["preferred_arm"] == "C1"


def test_original_global_guards_remain_active_and_unchanged(isolated):
    inputs, shadow = isolated
    assert native_inputs.verify_protocol is not inputs.verify_protocol
    assert native_shadow._source_check is not shadow._source_check
    policy = read(POLICY)
    with pytest.raises(ValueError, match="binding conflict"):
        native_shadow._source_check({policy["exception_path"]: policy["original_sha256"]})
    with pytest.raises(ValueError, match="another protocol"):
        inputs.verify_protocol(ROOT/"other_protocol.json")
    with pytest.raises(ValueError, match="binding conflict"):
        shadow._source_check({str(ROOT/"model_upgrade_metrics.py"): "0"*64})
    with pytest.raises(ValueError, match="binding conflict"):
        shadow._source_check({policy["exception_path"]: "0"*64})


@pytest.mark.parametrize("binding", ["exception_path", "preserved_original_path", "numeric_core"])
def test_changed_current_runner_preserved_original_or_numeric_code_is_refused(monkeypatch, binding):
    policy = read(POLICY)
    victim = (next(iter(policy["original_numeric_code_hashes"])) if binding == "numeric_core"
              else policy[binding])
    original_hash = native_shadow.file_sha256
    monkeypatch.setattr(native_shadow, "file_sha256",
                        lambda p: "0"*64 if Path(p).resolve() == Path(victim).resolve() else original_hash(p))
    with pytest.raises(ValueError, match="binding conflict"):
        compat.verify_policy(POLICY)


@pytest.mark.parametrize("change,value,reason", [
    ("numerical_refits", 1, "scope"),
    ("formed_at", "2099-01-01T00:00:00+00:00", "scope"),
    ("current_sha256", "0"*64, "policy changed"),
])
def test_unapproved_scope_time_or_policy_changes_are_refused(tmp_path, change, value, reason):
    policy = read(POLICY); policy[change] = value
    path = tmp_path/"compatibility_BEFORE_USE.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    (tmp_path/"implementation_BEFORE_USE.json").write_bytes((COMPAT_ROOT/"implementation_BEFORE_USE.json").read_bytes())
    with pytest.raises(ValueError, match=reason):
        compat.verify_policy(path)


def test_original_registration_reconstruction_keeps_forecast_scores_and_labels_separate(isolated):
    inputs, shadow = isolated
    registration = FIXED/"shadow/relay/2026-10-08/registration.json"
    record, _, _, _, _ = shadow._registered(registration)
    assert record == read(registration)
    pack = inputs.load_feature_archive(record["feature_archive"], record["knowledge_cutoff"])
    assert native_shadow.hashlib.sha256(native_shadow.canonical(pack)).hexdigest() == record["feature_projection_sha256"]
    assert len(record["original_codes"]) == len(pack["relay"]) == 10
    assert record.get("target_matured") is not True


def test_append_only_settlement_is_source_rebuilt_and_idempotent(isolated, monkeypatch):
    _, shadow = isolated
    ledger = FIXED/"shadow/settlements.jsonl"
    before = ledger.read_bytes()
    events = shadow._read_chain(ledger)
    assert len(events) == 2
    assert {e["module"] for e in events} == {"relay", "market"}
    monkeypatch.setattr(shadow, "_append_chain", lambda *args, **kwargs: pytest.fail("repeat settlement must not append"))
    for module in ("relay", "market"):
        registration = FIXED/f"shadow/{module}/2026-10-08/registration.json"
        result = shadow._settle_impl(registration, ARCHIVE)
        prior = next(e for e in events if e["module"] == module)
        assert result == prior
        assert result["record"]["candidate_group_complete"] is True
        assert result["record"]["outcomes"]["market_target"] is None
        assert len(result["record"]["outcomes"]["candidates"]) == 10
    assert ledger.read_bytes() == before


def test_model_manifest_tamper_cannot_be_exempted_by_compatibility(isolated, monkeypatch):
    _, shadow = isolated
    manifest = FIXED/"models/relay/BASE/manifest.json"
    native_hash = shadow.file_sha256
    monkeypatch.setattr(shadow, "file_sha256",
                        lambda p: "0"*64 if Path(p).resolve() == manifest else native_hash(p))
    with pytest.raises(ValueError, match="hash|binding"):
        shadow._registered(FIXED/"shadow/relay/2026-10-08/registration.json")
