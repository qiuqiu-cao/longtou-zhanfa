"""Regressions for immutable evidence, exact scores and non-event identities."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/first_board_dynamic_narrative_20261030'


def module(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


cases = module('build_case_evidence')
timeline = module('build_timeline')


def page(day, rows):
    return {'date': day, 'available_at': '2026-10-30T12:00:00+00:00', 'rows': rows}


def test_absent_event_retains_identity_but_not_an_invented_state():
    rows = [{'name': 'A', 'code': '000001', 'kind': 'limit_up', 'boards': 1,
             'raw_row_sha256': 'observed'}]
    pages = {'2026-10-09': page('2026-10-09', []),
             '2026-10-12': page('2026-10-12', rows)}
    result = timeline.timeline_rows(pages, ['A'])
    assert len(result) == 2 and [r['code'] for r in result] == ['000001', '000001']
    assert result[0]['event_page_state'] == 'NOT_IN_RETURNED_EVENT_LIST_UNKNOWN'
    assert result[0]['reported_boards'] is None and result[0]['row_sha256'] == ''
    assert all(not r['production_ranking_eligible'] for r in result)


@pytest.mark.parametrize('issue', ['wrong_date', 'ambiguous_code', 'duplicate_event', 'absent_identity'])
def test_timeline_rejects_unsafe_source_binding(issue):
    row = {'name': 'A', 'code': '000001', 'kind': 'limit_up'}
    obj = page('2026-10-09', [row])
    if issue == 'wrong_date': obj['date'] = '2026-10-12'
    if issue == 'ambiguous_code': obj['rows'].append({**row, 'code': '000002'})
    if issue == 'duplicate_event': obj['rows'].append(copy.deepcopy(row))
    if issue == 'absent_identity': obj['rows'] = []
    with pytest.raises(ValueError):
        timeline.timeline_rows({'2026-10-09': obj}, ['A'])


def test_existing_case_artifact_is_never_partially_overwritten(tmp_path):
    p = tmp_path / 'CASE_OUTCOMES_DIAGNOSTIC_ONLY.csv'
    p.write_bytes(b'original evidence')
    with pytest.raises(FileExistsError): cases.build(tmp_path)
    assert p.read_bytes() == b'original evidence'
    assert list(tmp_path.iterdir()) == [p]


def test_rebuild_preserves_scores_pool_labels_and_row_qualifications(tmp_path):
    cases.build(tmp_path)
    read = lambda p: pd.read_csv(p, dtype={'code': str}, float_precision='round_trip')
    result = read(tmp_path / 'CASE_POOL_NO_LABELS.csv')
    original = read(BASE / 'CASE_POOL_NO_LABELS.csv')
    source = read(cases.SRC / 'v3/FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv')
    keys = ['signal_date', 'code']
    assert result[keys + ['rank']].equals(original[keys + ['rank']])
    joined = result.merge(source[keys + ['s1_score', 's1_model_status']], on=keys, validate='one_to_one')
    assert joined.s1_score_x.equals(joined.s1_score_y)
    assert joined.s1_model_status_x.equals(joined.s1_model_status_y)
    assert len(result) == 92 and result.amount.isna().sum() == 11
    assert not result.production_ranking_eligible.any()
    pd.testing.assert_frame_equal(read(tmp_path / 'CASE_OUTCOMES_DIAGNOSTIC_ONLY.csv'),
                                  read(BASE / 'CASE_OUTCOMES_DIAGNOSTIC_ONLY.csv'))
    audit = json.loads((tmp_path / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    assert isinstance(audit['outcomes'], list)
    for name, expected in audit['output_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
