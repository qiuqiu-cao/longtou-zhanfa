import hashlib
import json
from pathlib import Path
import pytest
import relay_local_recovery as r


def make(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    fresh=tmp_path/'fresh.xml';fresh.write_text('<testsuites><testsuite tests="3" failures="0" errors="0"/></testsuites>')
    missing=tmp_path/'old_tests.xml';original='a'*64
    path=tmp_path/'recovery.json'
    data=dict(type='EXPLICIT_DELETED_PYTEST_RECORD_RECOVERY',fresh_tests_path=str(fresh),fresh_tests_sha256=r.sha(fresh),fresh_tests_passed=3,
        historical_test_records_unrecoverable=[dict(original_path=str(missing),original_expected_sha256=original)])
    path.write_text(json.dumps(data));return path,missing,original,data


def test_missing_historical_test_is_explicit_not_original_recovery(tmp_path,monkeypatch):
    path,missing,expected,_=make(tmp_path,monkeypatch)
    result=r.verify_sources({str(missing):expected},path)
    assert result['historical_test_records_unavailable']==1
    assert not result['original_test_bytes_recovered'] and result['model_data_code_hash_exceptions']==0


def test_model_or_other_missing_file_cannot_be_skipped(tmp_path,monkeypatch):
    path,missing,expected,_=make(tmp_path,monkeypatch)
    for name in ['model.joblib','source.py','other_tests.xml']:
        with pytest.raises(ValueError,match='unapproved'):r.verify_sources({str(tmp_path/name):expected},path)


def test_existing_file_wrong_hash_is_never_accepted_as_deleted_test(tmp_path,monkeypatch):
    path,missing,expected,_=make(tmp_path,monkeypatch);missing.write_text('different')
    with pytest.raises(ValueError,match='hash changed'):r.verify_sources({str(missing):expected},path)


def test_wrong_old_hash_and_new_test_hash_are_rejected(tmp_path,monkeypatch):
    path,missing,expected,data=make(tmp_path,monkeypatch)
    with pytest.raises(ValueError):r.verify_sources({str(missing):'b'*64},path)
    Path(data['fresh_tests_path']).write_text('changed')
    with pytest.raises(ValueError,match='fresh test record changed'):r.verify_sources({},path)


def test_registry_cannot_authorize_runtime_code_or_escape_project(tmp_path,monkeypatch):
    path,missing,expected,data=make(tmp_path,monkeypatch)
    for name in [tmp_path/'source.py',tmp_path.parent/'elsewhere_tests.xml']:
        data['historical_test_records_unrecoverable'][0]['original_path']=str(name)
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError,match='only'):r.verify_sources({},path)


def test_failed_fresh_tests_cannot_replace_audit_evidence(tmp_path,monkeypatch):
    path,missing,expected,data=make(tmp_path,monkeypatch)
    fresh=Path(data['fresh_tests_path']);fresh.write_text('<testsuites><testsuite tests="3" failures="1" errors="0"/></testsuites>')
    data['fresh_tests_sha256']=r.sha(fresh);path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='must pass'):r.verify_sources({},path)


def test_exact_original_model_hash_remains_strict(tmp_path,monkeypatch):
    path,_,_,_=make(tmp_path,monkeypatch);model=tmp_path/'model.joblib';model.write_bytes(b'original')
    result=r.verify_sources({str(model):r.sha(model)},path)
    assert result['exact_original_sources_verified']==1 and result['historical_test_records_unavailable']==0


def test_explicit_integration_pytest_xml_is_a_test_record(tmp_path,monkeypatch):
    path,missing,expected,data=make(tmp_path,monkeypatch)
    missing=tmp_path/'all_integration_frozen.xml'
    data['historical_test_records_unrecoverable'][0]['original_path']=str(missing)
    path.write_text(json.dumps(data))
    assert r.verify_sources({str(missing):expected},path)['historical_test_records_unavailable']==1
