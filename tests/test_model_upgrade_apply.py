"""Synthetic pending plans and verification proofs; no real model deployment."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

import model_upgrade_apply as apply


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(apply.shadow, "now_utc", lambda: datetime(2030, 1, 2, 9, 5, tzinfo=timezone.utc))
    candidates = [{"code": "000001", "name": "SYNTH_A", "boards": 2, "actual_one_word_daily": False,
                   "promotion_call": "FAIL", "identity": "CHALLENGER", "probability": None},
                  {"code": "000002", "name": "SYNTH_B", "boards": 3, "actual_one_word_daily": True,
                   "promotion_call": "PROMOTE", "identity": "SPACE_ANCHOR", "probability": None}]
    extension = [{"code": c["code"], "boards": c["boards"], "theme": {"status": "UNKNOWN"},
                  "events": {"promotion": {"call": c["promotion_call"] == "PROMOTE", "probability": None},
                             "break": {"call": c["promotion_call"] != "PROMOTE", "probability": None},
                             "reseal": {"call": None, "probability": None}}} for c in candidates]
    plan = {"signal_date": "2030-01-02", "target_date": "2030-01-03",
            "knowledge_cutoff": "2030-01-02T17:00:00+08:00", "input_hash": "a"*64,
            "model_version": "SYNTH_BASE", "market": {"primary": "RISK_OFF", "direction": "WORSEN", "probability": None},
            "candidates": candidates, "primary_candidate": None,
            "identity_primary_candidate": "000002", "non_one_word_primary_candidate": None,
            "daily_research": {"draft_status": apply.PENDING, "candidates": extension,
                               "hypotheses": [{"id": "SYNTHETIC_UNCHANGED"}]}}
    binding = {"model_path": str(tmp_path/'model.joblib'), "model_sha256": "b"*64,
               "path": str(tmp_path/'manifest.json'), "sha256": "c"*64}
    record = {"module": "relay", "signal_date": plan["signal_date"], "target_date": plan["target_date"],
              "protocol_path": str(tmp_path/'protocol.json'),
              "forward": True, "knowledge_cutoff": "2030-01-02T16:50:00+08:00",
              "registered_at": "2030-01-02T16:55:00+08:00", "original_codes": ["000001", "000002"],
              "model_bindings": {"C1": binding}, "arms": {"C1": {"status": "KNOWN", "candidates": [
                  {"code": "000001", "boards": 2, "actual_one_word_daily": 0, "raw_score": .4, "predicted_positive": False},
                  {"code": "000002", "boards": 3, "actual_one_word_daily": 1, "raw_score": .8, "predicted_positive": True}]}}}
    pointer = {"active_arm": "C1", "effective_signal": "2030-01-02", "promoted_at": "2030-01-01T16:00:00+08:00",
               "suspended": False, "model": binding}
    proof = {"status": "VERIFIED_ACTIVE", "verified_at": "2030-01-02T17:05:00+08:00",
             "registration_path": str(tmp_path/'registration.json'), "registration_sha256": "d"*64,
             "active_path": str(tmp_path/'active_pointer.json'), "active_sha256": "e"*64}
    source = tmp_path/'draft.json'
    source.write_text(json.dumps(plan), encoding='utf8')
    return plan, record, pointer, proof, source, tmp_path/'new.json'


def test_negative_nonword_stays_negative_and_all_candidates_remain(bundle):
    plan, record, pointer, proof, _, _ = bundle
    original = deepcopy(plan)
    output = apply._adapt_verified(plan, record, pointer, proof)
    assert plan == original
    assert output['candidates'][0]['promotion_call'] == 'FAIL'
    assert output['non_one_word_primary_candidate'] is None
    assert output['model_nonword_rank_first'] == '000001'
    assert output['identity_primary_candidate'] == '000002'
    assert output['primary_candidate'] is None
    assert output['candidates'][0]['probability'] is None
    assert output['candidates'][0]['model_promotion_score'] == 40.0
    assert output['candidates'][1]['model_rank'] == 1
    assert output['daily_research']['draft_status'] == apply.PENDING
    assert output['daily_research']['hypotheses'] == original['daily_research']['hypotheses']
    assert output['daily_research']['candidates'][0]['events']['reseal'] == original['daily_research']['candidates'][0]['events']['reseal']
    assert {c['code'] for c in output['candidates']} == {'000001', '000002'}


def test_positive_nonword_and_v280_code_tie_rule(bundle):
    plan, record, pointer, proof, _, _ = bundle
    for candidate in record['arms']['C1']['candidates']:
        candidate['raw_score'] = .8
        candidate['predicted_positive'] = True
    output = apply._adapt_verified(plan, record, pointer, proof)
    assert output['identity_primary_candidate'] == '000001'
    assert output['non_one_word_primary_candidate'] == '000001'
    assert output['candidates'][0]['model_rank'] == 1


def test_market_application_changes_only_direction_module(bundle):
    plan, record, pointer, proof, _, _ = bundle
    original_candidates = deepcopy(plan['candidates'])
    record['module'] = 'market'
    record['arms']['C1'] = {'status': 'KNOWN', 'prediction': 'IMPROVE', 'raw_scores': {'IMPROVE': .8, 'CONTINUE': .1, 'WORSEN': .1}}
    output = apply._adapt_verified(plan, record, pointer, proof)
    assert output['market']['direction'] == 'IMPROVE'
    assert output['market']['primary'] == 'RISK_OFF'
    assert output['market']['probability'] is None
    assert output['candidates'] == original_candidates
    assert output['non_one_word_primary_candidate'] is None


@pytest.mark.parametrize('change', ['reviewed', 'wrapped', 'frozen', 'cutoff', 'target', 'codes', 'unknown', 'not_effective', 'one_word', 'binding', 'score_threshold', 'probability'])
def test_unsafe_or_conflicting_draft_refused(bundle, change):
    plan, record, pointer, proof, _, _ = bundle
    if change == 'reviewed': plan['daily_research']['draft_status'] = 'AUTHOR_REVIEWED'
    elif change == 'wrapped': plan = {'plan': plan, 'sha256': 'x'}
    elif change == 'frozen': plan['recorded_at'] = '2030-01-02T17:00:00+08:00'
    elif change == 'cutoff': plan['knowledge_cutoff'] = '2030-01-02T16:54:00+08:00'
    elif change == 'target': plan['target_date'] = '2030-01-04'
    elif change == 'codes': record['original_codes'] = ['000001']
    elif change == 'unknown': record['arms']['C1']['status'] = 'UNKNOWN'
    elif change == 'not_effective': pointer['effective_signal'] = '2030-01-03'
    elif change == 'one_word': record['arms']['C1']['candidates'][0]['actual_one_word_daily'] = 'inferred'
    elif change == 'binding': pointer['model'] = {'model_sha256': 'x'}
    elif change == 'score_threshold': record['arms']['C1']['candidates'][0]['predicted_positive'] = True
    else: plan['candidates'][0]['probability'] = .4
    with pytest.raises(apply.UpgradeApplyError, match='STOP_NEW_MODULE'):
        apply._adapt_verified(plan, record, pointer, proof)


def test_apply_cli_path_uses_full_verifier_and_unique_output(bundle, monkeypatch):
    plan, record, pointer, proof, source, output = bundle
    calls = []
    def verify(registration, active):
        calls.append((registration, active))
        return record, pointer, proof
    monkeypatch.setattr(apply.shadow, 'verify_active_for_registration', verify)
    before = source.read_bytes()
    result = apply.apply_plan(source, 'registration', 'active', output)
    assert calls == [('registration', 'active')]
    assert result['status'] == 'NEW_DRAFT_REQUIRES_AUTHOR_REVIEW'
    assert output.exists() and source.read_bytes() == before
    with pytest.raises(FileExistsError):
        apply.apply_plan(source, 'registration', 'active', output)


def test_base_is_unchanged_and_does_not_make_new_model_plan(bundle, monkeypatch):
    _, record, pointer, proof, source, output = bundle
    pointer['active_arm'] = 'BASE'
    proof['status'] = 'UNCHANGED_BASE'
    monkeypatch.setattr(apply.shadow, 'verify_active_for_registration', lambda *_: (record, pointer, proof))
    before = source.read_bytes()
    result = apply.apply_plan(source, 'registration', 'active', output)
    assert result['status'] == 'UNCHANGED_BASE'
    assert result['output'] is None and not output.exists()
    assert source.read_bytes() == before


def test_source_acceptance_conflict_stops_without_output(bundle, monkeypatch):
    _, record, _, proof, source, output = bundle
    registration = proof['registration_path']
    with open(registration, 'w', encoding='utf8') as stream:
        json.dump(record, stream)
    def invalid(*_): raise ValueError('acceptance source hash conflict')
    monkeypatch.setattr(apply.shadow, '_verify_active_impl', invalid)
    suspended = []
    monkeypatch.setattr(apply.shadow, '_binding_failure', lambda path, module, error: suspended.append((path, module, str(error))))
    before = source.read_bytes()
    with pytest.raises(apply.UpgradeApplyError, match='source-bound verification'):
        apply.apply_plan(source, registration, 'active', output)
    assert suspended == [(record['protocol_path'], 'relay', 'acceptance source hash conflict')]
    assert not output.exists() and source.read_bytes() == before


def test_existing_execution_primary_is_not_swapped_for_backup(bundle):
    plan, record, pointer, proof, _, _ = bundle
    plan['primary_candidate'] = '000001'
    plan['highest_accept_price'] = 12.0
    output = apply._adapt_verified(plan, record, pointer, proof)
    assert output['primary_candidate'] is None
    assert output['highest_accept_price'] is None
    assert output['execution_primary_review']['previous_primary'] == '000001'
    assert output['execution_primary_review']['automatic_backup'] is False
    assert plan['primary_candidate'] == '000001'


def test_unknown_true_ohlc_keeps_all_scores_but_blocks_whole_nonword_selection(bundle):
    plan, record, pointer, proof, _, _ = bundle
    plan['primary_candidate'] = '000001'
    for candidate in record['arms']['C1']['candidates']:
        candidate['raw_score'] = .8
        candidate['predicted_positive'] = True
    record['arms']['C1']['candidates'][1]['actual_one_word_daily'] = None
    output = apply._adapt_verified(plan, record, pointer, proof)
    assert output['candidates'][0]['promotion_call'] == 'PROMOTE'
    assert output['candidates'][1]['actual_one_word_daily'] is None
    assert [c['model_raw_score'] for c in output['candidates']] == [.8, .8]
    assert output['identity_primary_candidate'] == '000001'
    assert output['model_rank_first_candidate'] == '000001'
    assert output['non_one_word_primary_candidate'] is None
    assert output['model_nonword_rank_first'] is None
    assert output['nonword_selection_status'] == 'UNKNOWN'
    assert output['nonword_selection_unknown_codes'] == ['000002']
    assert output['primary_candidate'] is None
    assert output['execution_primary_review']['reason'] == 'UNKNOWN_TRUE_OHLC_GROUP'
    assert {c['code'] for c in output['candidates']} == set(record['original_codes'])


def test_projection_source_conflict_suspends_only_corresponding_module(bundle, monkeypatch):
    _, record, pointer, proof, source, output = bundle
    pointer['model'] = {'model_sha256': 'wrong'}
    monkeypatch.setattr(apply.shadow, 'verify_active_for_registration', lambda *_: (record, pointer, proof))
    suspended = []
    monkeypatch.setattr(apply.shadow, '_binding_failure', lambda path, module, error: suspended.append((path, module, str(error))))
    with pytest.raises(apply.UpgradeApplySourceConflict, match='bindings differ'):
        apply.apply_plan(source, 'registration', 'active', output)
    assert len(suspended) == 1
    assert suspended[0][:2] == (record['protocol_path'], 'relay')
    assert not output.exists()


def test_same_module_cannot_overwrite_previous_application(bundle):
    plan, record, pointer, proof, _, _ = bundle
    plan['model_upgrade_application'] = {'relay': {'old_binding': 'retain'}}
    with pytest.raises(apply.UpgradeApplyError, match='previous module'):
        apply._adapt_verified(plan, record, pointer, proof)
