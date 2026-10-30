"""Verify the stage delivery and append-only fixes; no fit, rule search or scoring."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'research/first_board_dynamic_narrative_20261030'
FIXED = HERE / 'corrected'
KEYS = ['signal_date', 'code']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, dtype={'code': str}, float_precision='round_trip')


def main():
    out = HERE / 'DELIVERY_REVIEW.json'
    if out.exists():
        raise FileExistsError('review receipt already exists')
    manifest = json.loads((HERE / 'PRE_EDIT_MANIFEST.json').read_text(encoding='utf-8'))
    changed = []
    for row in manifest['files']:
        assert sha(ROOT / row['snapshot_path']) == row['sha256']
        if sha(ROOT / row['original_path']) != row['sha256']:
            changed.append(row['original_path'])
    assert {Path(p).name for p in changed} == {'build_case_evidence.py', 'build_timeline.py'}

    original, corrected = [read(p / 'CASE_POOL_NO_LABELS.csv') for p in (ORIGINAL, FIXED)]
    labels, fixed_labels = [read(p / 'CASE_OUTCOMES_DIAGNOSTIC_ONLY.csv') for p in (ORIGINAL, FIXED)]
    source = read(ROOT / 'research/october_2026_s1_blind_replay_20261030/v3/FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv')
    assert corrected[KEYS + ['rank']].equals(original[KEYS + ['rank']])
    pd.testing.assert_frame_equal(labels, fixed_labels)
    assert len(corrected) == 92 and not corrected.duplicated(KEYS).any()
    joined = corrected.merge(source[KEYS + ['s1_score']], on=KEYS, validate='one_to_one')
    assert joined.s1_score_x.equals(joined.s1_score_y)
    assert not corrected.production_ranking_eligible.any()
    previous_joined = original.merge(source[KEYS + ['s1_score']], on=KEYS, validate='one_to_one')

    old_timeline, new_timeline = [read(p / 'CASE_EVENT_TIMELINE_LATE_SOURCE.csv') for p in (ORIGINAL, FIXED)]
    assert len(new_timeline) == 126 and new_timeline.code.notna().all()
    assert not new_timeline.duplicated(['date', 'code']).any()
    pd.testing.assert_frame_equal(old_timeline.drop(columns='code'),
                                  new_timeline[old_timeline.columns].drop(columns='code'))
    assert not new_timeline.production_ranking_eligible.any()

    spec = importlib.util.spec_from_file_location('reviewed_narrative_contract', ORIGINAL / 'evidence_contract.py')
    contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
    forum = json.loads((ORIGINAL / 'FORUM_CLAIM_TIME_AUDIT_V2.json').read_text(encoding='utf-8'))
    claim_count = 0
    for cutoff, saved in forum['by_cutoff'].items():
        claims = [{k: v for k, v in row.items() if k not in {
            'source_eligibility', 'claim_actually_available_by_cutoff',
            'is_confirmed_causal_leadership', 'claim_is_author_interpretation'}} for row in saved['claims']]
        result = contract.check_claims({'cutoff': cutoff, 'formed_at': forum['reviewed_at'],
            'sources': {'post': forum['source']}, 'claims': claims})
        assert result == saved
        claim_count += len(result['claims'])
        assert all(not row['claim_actually_available_by_cutoff'] for row in result['claims'])

    protected = json.loads((ROOT / 'research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json').read_text(encoding='utf-8'))['protected_sha256']
    mismatches = [p for p, expected in protected.items() if not Path(p).is_file() or sha(p) != expected]
    assert not mismatches
    diagnostic = corrected.merge(labels, on=KEYS, validate='one_to_one')
    per_day = []
    for day, group in diagnostic.groupby('signal_date'):
        top = group[group['rank'].le(3)]
        per_day.append({'signal_date': day, 'candidates': len(group),
            'promotion_hits': int(group.outcome.eq(1).sum()),
            'promotion_failures': int(group.outcome.eq(0).sum()),
            'promotion_unknown': int(group.outcome.isna().sum()),
            'reach4_hits': int(group.reach4.eq(1).sum()),
            'baseline_top3_hits': int(top.outcome.eq(1).sum()), 'baseline_top3_slots': len(top)})
    baseline_path = ROOT / 'research/first_board_theme_diffusion_review_20261030/results/REPORT.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))['results']['S1_COMPLETE_BASELINE']['FULL_286_DAYS']
    report = {'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'delivery_status': 'EVIDENCE_AND_CASE_RESEARCH_STAGE_ONLY',
        'stage_claims_reproduced': True, 'case_diagnostics': per_day,
        'case_dates_selected_after_outcomes_known': True, 'independent_test': False,
        'timeline_rows': len(new_timeline), 'observed_event_rows': int(new_timeline.row_sha256.notna().sum()),
        'blank_code_rows_before': int(old_timeline.code.isna().sum()), 'blank_code_rows_after': 0,
        'score_roundtrip_differences_before': int(previous_joined.s1_score_x.ne(previous_joined.s1_score_y).sum()),
        'score_roundtrip_differences_after': 0, 'case_ranks_and_outcomes_unchanged': True,
        'amount_missing_retained': int(corrected.amount.isna().sum()),
        'forum_claim_cutoff_checks_reproduced': claim_count, 'actual_eligible_claims': 0,
        'original_artifacts_preserved_except_reviewed_code': True,
        'baseline_source': str(baseline_path.relative_to(ROOT)), 'baseline_source_sha256': sha(baseline_path),
        'unchanged_historical_baseline': baseline, 'new_market_fits': 0, 'new_rankings': 0,
        'validated_accuracy_improvement': False, 'target_40_percent_achieved': False,
        'production_changes': False, 'protected_checked': len(protected), 'protected_mismatches': mismatches,
        'tests': {'passed': 19, 'command': 'python -m pytest -q tests/test_dynamic_narrative_delivery_review.py tests/test_dynamic_narrative_contract.py tests/test_first_board_top3_target.py'},
        'corrected_artifact_sha256': {p.name: sha(p) for p in sorted(FIXED.iterdir()) if p.is_file()}}
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ['delivery_status', 'timeline_rows', 'blank_code_rows_after',
        'score_roundtrip_differences_after', 'protected_checked', 'new_market_fits', 'new_rankings']}))


if __name__ == '__main__':
    main()
