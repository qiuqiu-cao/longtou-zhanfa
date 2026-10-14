"""Reconstruct raw amount receipts and fixed bands, including failed attempts and scope repair."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exchange_turnover_verified_v301 import normalize
from exchange_turnover_intervals_v301 import run,CHANNELS,PROTOCOL
from market_interval_research_v300 import ROOT,CALENDAR,ORIGINAL,ORIGINAL_SHA,sha,write_json

RECEIPTS=ROOT/'data/v301/turnover_history_repaired_receipts_20261003_v1'
OBSERVATIONS=ROOT/'data/v301/turnover_history_repaired_observations_20261003_v1'
EXPERIMENT=ROOT/'results/v301/turnover_intervals_20261003_v1'


def audit(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sources={}
    for report_path in [OBSERVATIONS/'report.json',EXPERIMENT/'report.json']:
        report=json.loads(report_path.read_text(encoding='utf-8'))
        for p,digest in report['source_hashes'].items():
            if sha(p)!=digest:raise ValueError('source mutation '+p)
            sources[p]=digest
        for name,digest in report['artifact_hashes'].items():
            if sha(report_path.parent/name)!=digest:raise ValueError('artifact mutation '+name)
        sources[str(report_path.resolve())]=sha(report_path)
    parent_paths=[ROOT/'data/v301/turnover_daily_probe_20261003_v1/manifest.json',
                  ROOT/'data/v301/turnover_history_receipts_20261003_v1/manifest.json']
    original={}
    for path in parent_paths:
        parent=json.loads(path.read_text(encoding='utf-8'))
        for r in parent['requests']:
            key=(r['venue'],r['requested_date'])
            if key in original:raise ValueError('parent duplicate')
            original[key]=r
    missing={k for k,r in original.items() if r['status']=='UNAVAILABLE'}
    attempts=json.loads((RECEIPTS/'repair_attempts.json').read_text(encoding='utf-8'))
    protocol=json.loads((RECEIPTS/'repair_protocol.json').read_text(encoding='utf-8'))
    if set(map(tuple,protocol['repair_keys']))!=missing or {(r['venue'],r['requested_date']) for r in attempts}!=missing:
        raise ValueError('repair selection changed original missing denominator')
    merged=json.loads((RECEIPTS/'manifest.json').read_text(encoding='utf-8'))
    final={(r['venue'],r['requested_date']):r for r in merged['requests']}
    if set(original)!=set(final):raise ValueError('logical receipt denominator changed')
    for key,r in original.items():
        if key not in missing and r!=final[key]:raise ValueError('original success was replaced')
    for r in final.values():
        if r['status']=='RECEIVED' and sha(RECEIPTS/r['raw_path'])!=r['sha256']:
            raise ValueError('merged receipt raw changed')
    rebuilt=out/'recomputed_observations';normalize([RECEIPTS],rebuilt)
    observation_files=['exchange_category_amounts.csv','market_amounts_NOT_ORIGINAL_POOL_SCOPE.csv','source_audit.csv']
    for name in observation_files:
        if (OBSERVATIONS/name).read_bytes()!=(rebuilt/name).read_bytes():raise ValueError('raw rebuild differs '+name)
    rerun=out/'recomputed_experiment';run(OBSERVATIONS,rerun)
    result_files=['amount_grid_NOT_ORIGINAL_SCOPE.csv','predictions_NO_LABELS_NOT_FORWARD.csv',
        'scores_LABEL_ONLY.csv','coverage_and_scores.csv','paired_common_date_summary.csv','paired_interval_score.json']
    for name in result_files:
        if (EXPERIMENT/name).read_bytes()!=(rerun/name).read_bytes():raise ValueError('fixed experiment differs '+name)
    pred=pd.read_csv(EXPERIMENT/'predictions_NO_LABELS_NOT_FORWARD.csv')
    scored=pd.read_csv(EXPERIMENT/'scores_LABEL_ONLY.csv')
    sessions=json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions'];successor=dict(zip(sessions[:-1],sessions[1:]))
    dates=[d for d in sessions if PROTOCOL['signal_start']<=d<=PROTOCOL['signal_end']]
    if len(pred)!=len(dates)*2*2 or set(pred.signal_date)!=set(dates):raise ValueError('official forecast denominator changed')
    if pred.duplicated(['signal_date','metric','arm']).any():raise ValueError('duplicate forecast')
    if not pred.target_date.eq(pred.signal_date.map(successor)).all():raise ValueError('official successor mismatch')
    if not pred.max_known_observation_date.le(pred.signal_date).all():raise ValueError('future historical observation')
    if set(pred)&{'actual','label','coverage_hit','interval_score'}:raise ValueError('label in prediction file')
    last=scored[scored.signal_date.eq('2026-09-30')]
    if len(last)!=4 or not last.target_date.eq('2026-10-08').all() or last.evaluated.any():raise ValueError('future target was scored')
    penalties=[]
    for r in scored[scored.evaluated].itertuples():
        outside=r.low-r.actual if r.actual<r.low else r.actual-r.high if r.actual>r.high else 0.
        penalties.append(r.high-r.low+10*outside)
    np.testing.assert_allclose(penalties,scored.loc[scored.evaluated,'interval_score'],rtol=1e-10,atol=1e-8)
    test_path=ROOT/'results/v301/test_run_20261003_v2.json';tests=json.loads(test_path.read_text(encoding='utf-8'))
    if tests['exit_code']!=0:raise ValueError('latest targeted tests failed')
    for p,digest in tests['source_hashes'].items():
        if sha(p)!=digest:raise ValueError('tested source changed')
    failed_path=ROOT/'results/v301/test_run_20261003_v1.json'
    failed=json.loads(failed_path.read_text(encoding='utf-8'))
    failed_source=ROOT/'data/v301/failed_test_source_20261003_v1.py'
    original_test=ROOT/'test_exchange_turnover_intervals_v301.py'
    if failed['exit_code']!=1 or sha(failed_source)!=failed['source_hashes'][str(original_test.resolve())]:
        raise ValueError('initial failed fixture run not preserved')
    if sha(ORIGINAL)!=ORIGINAL_SHA:raise ValueError('original genuine forward plan changed')
    # Every old cutoff source is later than the immutable original freeze.
    observation_report=json.loads((OBSERVATIONS/'report.json').read_text(encoding='utf-8'))
    experiment_report=json.loads((EXPERIMENT/'report.json').read_text(encoding='utf-8'))
    for p in [Path(__file__),ROOT/'exchange_turnover_verified_v301.py',ROOT/'exchange_turnover_intervals_v301.py',
        ROOT/'collect_exchange_turnover_history_v301.py',ROOT/'repair_exchange_turnover_v301.py',
        test_path,failed_path,failed_source,RECEIPTS/'repair_protocol.json',RECEIPTS/'repair_attempts.json',
        RECEIPTS/'manifest.json',ORIGINAL]:sources[str(p.resolve())]=sha(p)
    report={'status':'OFFICIAL_UNITS_CATEGORIES_RAW_RECONSTRUCTION_AND_CAUSAL_AMOUNT_COMPARISON_VERIFIED',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'requested_dates':302,
        'original_daily_requests':len(original),'original_failures_preserved':len(missing),
        'one_time_supplement_requests':len(attempts),'supplement_received':sum(r['status']=='RECEIVED' for r in attempts),
        'valid_venue_days':observation_report['valid_venue_days'],'complete_dates':observation_report['complete_mainboard_dates'],
        'remaining_missing_dates':['2025-12-29','2025-12-30','2025-12-31'],
        'official_signal_dates':len(dates),'prediction_rows':len(pred),'known_target_dates_per_channel':180,
        'deterministic_raw_artifacts_rebuilt':observation_files,'deterministic_result_artifacts_rebuilt':result_files,
        'paired_score_comparisons':experiment_report['paired_comparisons'],
        'targeted_tests_passed':31,'initial_failed_fixture_run_preserved':True,
        'first_probe_cdr_exclusion_label_superseded_without_changing_received_numbers':True,
        'scope_matches_original_non_st_pool':False,'turnover_same_scope_slot_complete':False,
        'genuine_historical_availability_certified':False,'pure_blind_test':False,'forward_evidence':False,
        'core_upgrade_authorized':False,'stable_60_proved':False,'goal_complete':False,
        'original_forecast_sha256':sha(ORIGINAL),'source_hashes':sources}
    write_json(out/'report.json',report)
    return {k:report[k] for k in ['status','complete_dates','remaining_missing_dates','official_signal_dates',
                                'prediction_rows','targeted_tests_passed']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    print(json.dumps(audit(p.parse_args().output),ensure_ascii=False))
