"""Raw-source reconstruction and paired-date acceptance of V300 diagnostic research."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from market_interval_research_v300 import (
    run, sha, write_json, ROOT, CALENDAR, ORIGINAL, ORIGINAL_SHA, METRICS, PROTOCOL,
)


def paired_metrics(scored, sessions):
    rows=[]
    for period,(start,end) in PROTOCOL['periods'].items():
        dates=[d for d in sessions if start<=d<=end]
        part=scored[scored.signal_date.isin(dates)]
        for metric in METRICS:
            channel=part[part.metric.eq(metric) & part.evaluated]
            sets=[set(channel.loc[channel.arm.eq(arm),'signal_date']) for arm in PROTOCOL['arms']]
            common=sorted(sets[0]&sets[1])
            for arm in PROTOCOL['arms']:
                g=channel[channel.arm.eq(arm) & channel.signal_date.isin(common)]
                rows.append({'period':period,'metric':metric,'arm':arm,'unit':METRICS[metric][1],
                    'official_signal_dates':len(dates),'common_dates':len(common),
                    'coverage':float(g.coverage_hit.mean()), 'mean_width':float(g.width.mean()),
                    'mean_interval_score':float(g.interval_score.mean()),'median_mae':float(g.median_abs_error.mean())})
    return pd.DataFrame(rows)


def audit(experiment, output):
    experiment=Path(experiment); out=Path(output); out.mkdir(parents=True,exist_ok=False)
    report=json.loads((experiment/'report.json').read_text(encoding='utf-8'))
    for path,digest in report['source_hashes'].items():
        if sha(path)!=digest: raise ValueError('input/source mutation: '+path)
    for name,digest in report['artifact_hashes'].items():
        if sha(experiment/name)!=digest: raise ValueError('research artifact mutation: '+name)
    reconstruction=out/'recomputed_experiment'
    run(reconstruction)
    deterministic=['observations_DIAGNOSTIC.csv','observation_status.csv','predictions_NO_LABELS_NOT_FORWARD.csv',
        'scores_LABEL_ONLY.csv','metric_summary.csv','paired_interval_score.json','requested_ten_slots.json']
    for name in deterministic:
        if (experiment/name).read_bytes()!=(reconstruction/name).read_bytes():
            raise ValueError('full recomputation differs: '+name)
    sessions=json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions']
    successor=dict(zip(sessions[:-1],sessions[1:]))
    signals=[d for d in sessions if PROTOCOL['signal_start']<=d<=PROTOCOL['signal_end']]
    pred=pd.read_csv(experiment/'predictions_NO_LABELS_NOT_FORWARD.csv')
    scored=pd.read_csv(experiment/'scores_LABEL_ONLY.csv')
    if len(pred)!=len(signals)*len(METRICS)*2 or set(pred.signal_date)!=set(signals):
        raise ValueError('official signal denominator changed')
    if pred.duplicated(['signal_date','metric','arm']).any(): raise ValueError('duplicate forecast')
    if not pred.target_date.eq(pred.signal_date.map(successor)).all(): raise ValueError('successor mismatch')
    if not pred.max_known_observation_date.le(pred.signal_date).all(): raise ValueError('future training observation')
    if set(pred)&{'actual','next_return','target','label','coverage_hit','interval_score'}:
        raise ValueError('label leaked into archived predictions')
    last=scored[scored.signal_date.eq('2026-09-30')]
    if len(last)!=16 or not last.target_date.eq('2026-10-08').all() or last.evaluated.any():
        raise ValueError('unsettled original target not retained')
    # Independently recompute each evaluable penalty from scalar comparisons.
    independent=[]
    for row in scored[scored.evaluated].itertuples():
        penalty=(row.low-row.actual) if row.actual<row.low else (row.actual-row.high) if row.actual>row.high else 0.
        independent.append((row.high-row.low)+10*penalty)
    np.testing.assert_allclose(independent,scored.loc[scored.evaluated,'interval_score'],rtol=1e-10,atol=1e-10)
    paired=paired_metrics(scored,sessions)
    paired.to_csv(out/'paired_common_date_summary.csv',index=False)
    summary=paired[paired.period.eq('ALL')].pivot(index='metric',columns='arm',values='mean_interval_score')
    point_improved=(summary.PERSISTENCE_DELTA-summary.LEVEL)<0
    comparisons=report['paired_comparisons']
    max_late=next(r for r in comparisons if r['period']=='MAR_SEP' and r['metric']=='max_board')
    if sha(ORIGINAL)!=ORIGINAL_SHA: raise ValueError('original forecast was modified')
    audit_report={
        'status':'RAW_INDEX_RECONSTRUCTION_OFFICIAL_DATES_FIXED_INTERVALS_AND_PAIRED_SCORES_VERIFIED',
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'official_signal_dates':len(signals),'prediction_rows':len(pred),
        'retained_last_signal_target':'2026-10-08','last_signal_evaluated_rows':int(last.evaluated.sum()),
        'deterministic_artifacts_recomputed':deterministic,
        'source_files_checked':len(report['source_hashes']),
        'requested_metrics':10,'diagnostic_categories':7,'unavailable_categories':3,'numeric_channels':8,
        'common_date_mean_score_delta_better_channels':summary.index[point_improved].tolist(),
        'max_board_mar_sep_adjusted_difference_interval':max_late['adjusted_interval'],
        'max_board_mar_sep_excludes_zero':bool(max_late['adjusted_interval'][1]<0),
        'all_date_scores_are_not_true_forward_or_pure_blind':True,
        'core_upgrade_authorized':False,'stable_60_proved':False,'goal_complete':False,
        'original_forecast_sha256':sha(ORIGINAL),
        'source_hashes':{str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'market_interval_research_v300.py',
            ROOT/'test_market_interval_research_v300.py',experiment/'report.json',experiment/'protocol.json',
            ROOT/'results/v300/test_run_20261003_v1.json',ORIGINAL]},
        'artifact_hashes':{'paired_common_date_summary.csv':sha(out/'paired_common_date_summary.csv')},
    }
    write_json(out/'report.json',audit_report)
    return {k:audit_report[k] for k in ['status','official_signal_dates','prediction_rows',
        'common_date_mean_score_delta_better_channels','max_board_mar_sep_excludes_zero']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit(a.experiment,a.output),ensure_ascii=False))
