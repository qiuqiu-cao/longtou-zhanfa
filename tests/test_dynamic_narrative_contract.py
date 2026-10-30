import importlib.util
from pathlib import Path
import pytest
import hashlib

PATH = Path(__file__).resolve().parents[1] / 'research/first_board_dynamic_narrative_20261030/evidence_contract.py'
spec = importlib.util.spec_from_file_location('dynamic_evidence_contract', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def source(pub, obtained, path=None):
    return {'url':'https://example.org/post', 'source_description':'test', 'published_at':pub,
            'obtained_at':obtained, 'saved_text_path':str(path) if path else '',
            'saved_text_sha256':hashlib.sha256('股票A同时被作者归入机器人和名称叙事。'.encode()).hexdigest()}

def claim(cid, theme):
    return {'claim_id':cid,'code':'000001','theme_id':theme,'theme_type':'event',
            'role':'unknown','phase':'unknown','relation':'unknown','evidence_text':'作者归类',
            'counterevidence':'未证实实际响应','reviewer':'researcher','source_id':'s',
            'reviewed_at':'2026-10-16T14:00:00+08:00',
            'source_span':'股票A同时被作者归入机器人和名称叙事。'}

def test_time_gates_are_distinct_and_late_source_does_not_backfill():
    cutoff='2026-10-16T15:00:00+08:00'
    late=m.source_state(source('2026-10-15T12:00:00+08:00','2026-10-30T19:00:00+08:00'),cutoff)
    assert late['published_by_cutoff'] and not late['acquired_by_cutoff'] and not late['actual_archived_eligible']
    future=m.source_state(source('2026-10-20T20:10:00+08:00','2026-10-30T19:00:00+08:00'),cutoff)
    assert not future['published_by_cutoff'] and not future['actual_archived_eligible']
    unknown=m.source_state(source(None,'2026-10-30T19:00:00+08:00'),cutoff)
    assert unknown['published_by_cutoff'] is None and not unknown['actual_archived_eligible']

def test_preserves_multiple_themes_and_refuses_fake_source_span(tmp_path):
    path=tmp_path/'source.txt'; path.write_text('股票A同时被作者归入机器人和名称叙事。',encoding='utf-8')
    b={'cutoff':'2026-10-16T15:00:00+08:00','formed_at':'2026-10-16T14:01:00+08:00','sources':{'s':source('2026-10-15T12:00:00+08:00','2026-10-15T12:01:00+08:00',path)},
       'claims':[claim('1','robot'),claim('2','name')]}
    result=m.check_claims(b)
    assert result['multi_memberships']['000001']==['robot','name']
    assert not result['numeric_features_generated'] and not result['ranking_changed']
    b['claims'][1]['source_span']='不存在的证据'
    with pytest.raises(ValueError): m.check_claims(b)

def test_does_not_accept_missing_counterevidence_or_timezone(tmp_path):
    c=claim('1','x'); c['counterevidence']=''
    path=tmp_path/'source.txt'; path.write_text('股票A同时被作者归入机器人和名称叙事。',encoding='utf-8')
    b={'cutoff':'2026-10-16T15:00:00+08:00','formed_at':'2026-10-16T14:01:00+08:00','sources':{'s':source('2026-10-15T12:00:00+08:00','2026-10-15T12:01:00+08:00',path)},'claims':[c]}
    with pytest.raises(ValueError): m.check_claims(b)
    with pytest.raises(ValueError): m.instant('2026-10-16T15:00:00')

def test_rejects_changed_source_and_backdated_review(tmp_path):
    path=tmp_path/'source.txt'; path.write_text('股票A同时被作者归入机器人和名称叙事。',encoding='utf-8')
    b={'cutoff':'2026-10-16T15:00:00+08:00','formed_at':'2026-10-16T14:01:00+08:00',
       'sources':{'s':source('2026-10-15T12:00:00+08:00','2026-10-15T12:01:00+08:00',path)},'claims':[claim('1','x')]}
    b['claims'][0]['reviewed_at']='2026-10-15T12:00:00+08:00'
    with pytest.raises(ValueError): m.check_claims(b)

def test_source_present_but_late_review_cannot_be_backfilled(tmp_path):
    path=tmp_path/'source.txt'; path.write_text('股票A同时被作者归入机器人和名称叙事。',encoding='utf-8')
    c=claim('1','x'); c['reviewed_at']='2026-10-16T16:00:00+08:00'
    b={'cutoff':'2026-10-16T15:00:00+08:00','formed_at':'2026-10-16T16:01:00+08:00',
       'sources':{'s':source('2026-10-15T12:00:00+08:00','2026-10-15T12:01:00+08:00',path)},'claims':[c]}
    result=m.check_claims(b)['claims'][0]
    assert result['source_eligibility']['actual_archived_eligible']
    assert not result['claim_actually_available_by_cutoff']
    b['claims'][0]['reviewed_at']='2026-10-16T14:00:00+08:00'
    path.write_text('已修改的资料',encoding='utf-8')
    with pytest.raises(ValueError): m.check_claims(b)
