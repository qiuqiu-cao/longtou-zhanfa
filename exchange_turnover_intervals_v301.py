"""Fixed inspected-history bands for explicitly broader official amount scopes."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from market_interval_research_v300 import sha,write_json,CALENDAR,ORIGINAL,ORIGINAL_SHA,ROOT,interval_score,bootstrap

CHANNELS=['mainboard_a_amount_yi','shsz_stock_category_amount_yi']
PROTOCOL={'version':'v301-official-turnover-bands-1','signal_start':'2026-01-01','signal_end':'2026-09-30',
    'arms':['LEVEL','PERSISTENCE_DELTA'],'lookback_official_sessions':120,'minimum_known_observations':60,
    'quantiles':[.1,.5,.9],'quantile_method':'linear','alpha':.2,'nominal_coverage':.8,
    'unit':'亿元','bounds':'nonnegative, continuous; no rounding of interval boundaries',
    'target_join':'exact official successor, never next available observation',
    'periods':{'ALL':['2026-01-01','2026-09-30'],'JAN_FEB':['2026-01-01','2026-02-28'],
               'MAR_SEP':['2026-03-01','2026-09-30']},
    'bootstrap':{'draws':5000,'seed':301,'block_official_sessions':5,'family_channels':2,'tails':[.025/2,1-.025/2]},
    'scope':{'mainboard_a_amount_yi':'SHSZ_MAINBOARD_A_INCLUDING_ST',
             'shsz_stock_category_amount_yi':'SHSZ_MAINBOARD_A_PLUS_STAR_BOARD_PLUS_CHINEXT_A_INCLUDING_ST_CDR_SCOPE_UNRESOLVED_EXCLUDING_B_BSE'},
    'protocol_stable_60_target_redefinition':False,'threshold_selection_on_test':False,
    'scope_matches_original_non_st_pool':False,'genuine_historical_availability_certified':False,
    'pure_blind_test':False,'forward_evidence':False,'core_upgrade_authorized':False,'goal_complete':False}


def predict(observations,sessions,protocol=PROTOCOL):
    if sessions!=sorted(set(sessions)) or observations.index.tolist()!=sessions:
        raise ValueError('full official session grid required')
    rows=[]
    for pos,day in enumerate(sessions):
        if not protocol['signal_start']<=day<=protocol['signal_end']:continue
        start=max(0,pos+1-protocol['lookback_official_sessions'])
        for key in CHANNELS:
            current=observations[key].iloc[pos];history=observations[key].iloc[:pos+1]
            for arm in protocol['arms']:
                train=(history if arm=='LEVEL' else history.diff()).iloc[start:].dropna()
                state='KNOWN_DIAGNOSTIC_INTERVAL'
                if arm=='PERSISTENCE_DELTA' and not np.isfinite(current):state='UNKNOWN_CURRENT_LEVEL'
                elif len(train)<protocol['minimum_known_observations']:state='INSUFFICIENT_HISTORY'
                values=[np.nan]*3
                if state=='KNOWN_DIAGNOSTIC_INTERVAL':
                    values=np.quantile(train,protocol['quantiles'],method=protocol['quantile_method'])
                    if arm=='PERSISTENCE_DELTA':values+=current
                    values=np.maximum(values,0)
                rows.append({'signal_date':day,'target_date':sessions[pos+1] if pos+1<len(sessions) else None,
                    'metric':key,'scope':protocol['scope'][key],'arm':arm,'unit':'亿元','status':state,
                    'low':values[0],'median':values[1],'high':values[2],
                    'training_n':len(train),'window_start':sessions[start],
                    'max_known_observation_date':train.index.max() if len(train) else None,
                    'training_digest':hashlib.sha256(train.to_csv().encode()).hexdigest(),
                    'genuine_historical_availability_certified':False,'forward_evidence':False})
    return pd.DataFrame(rows)


def score(pred,observations):
    frame=pred.copy()
    frame['actual']=[observations.at[r.target_date,r.metric] if r.target_date in observations.index else np.nan
                     for r in frame.itertuples()]
    frame['evaluated']=frame.status.eq('KNOWN_DIAGNOSTIC_INTERVAL') & frame.actual.notna()
    frame['width']=frame.high-frame.low
    frame['coverage_hit']=np.nan;frame['interval_score']=np.nan;frame['median_abs_error']=np.nan
    mask=frame.evaluated
    frame.loc[mask,'coverage_hit']=frame.loc[mask,'actual'].between(frame.loc[mask,'low'],frame.loc[mask,'high']).astype(float)
    frame.loc[mask,'interval_score']=interval_score(frame.loc[mask,'actual'],frame.loc[mask,'low'],frame.loc[mask,'high'])
    frame.loc[mask,'median_abs_error']=abs(frame.loc[mask,'actual']-frame.loc[mask,'median'])
    return frame


def summaries(scored,sessions):
    all_rows,paired_rows,comparisons=[],[],[]
    for period,(start,end) in PROTOCOL['periods'].items():
        dates=[d for d in sessions if start<=d<=end]
        for key in CHANNELS:
            channel=scored[scored.signal_date.isin(dates) & scored.metric.eq(key)]
            complete=channel[channel.evaluated]
            common=set.intersection(*[set(complete.loc[complete.arm.eq(a),'signal_date']) for a in PROTOCOL['arms']])
            for arm in PROTOCOL['arms']:
                part=channel[channel.arm.eq(arm)];valid=part[part.evaluated]
                all_rows.append({'period':period,'metric':key,'arm':arm,'unit':'亿元','scope':PROTOCOL['scope'][key],
                    'official_signal_dates':len(dates),'retained_rows':len(part),
                    'intervals_issued':int(part.status.eq('KNOWN_DIAGNOSTIC_INTERVAL').sum()),
                    'known_targets':int(part.actual.notna().sum()),'evaluated_dates':len(valid),
                    'missing_target_dates':part.loc[part.actual.isna(),'signal_date'].tolist(),
                    'coverage':float(valid.coverage_hit.mean()) if len(valid) else None,
                    'mean_width':float(valid.width.mean()) if len(valid) else None,
                    'mean_interval_score':float(valid.interval_score.mean()) if len(valid) else None,
                    'median_forecast_mae':float(valid.median_abs_error.mean()) if len(valid) else None})
                pair=valid[valid.signal_date.isin(common)]
                paired_rows.append({'period':period,'metric':key,'arm':arm,'unit':'亿元',
                    'official_signal_dates':len(dates),'common_dates':len(common),
                    'coverage':float(pair.coverage_hit.mean()) if len(pair) else None,
                    'mean_width':float(pair.width.mean()) if len(pair) else None,
                    'mean_interval_score':float(pair.interval_score.mean()) if len(pair) else None,
                    'median_forecast_mae':float(pair.median_abs_error.mean()) if len(pair) else None})
            pivot=complete.pivot(index='signal_date',columns='arm',values='interval_score')
            comparison=bootstrap(pivot.PERSISTENCE_DELTA-pivot.LEVEL,dates,PROTOCOL['bootstrap']) if set(PROTOCOL['arms']).issubset(pivot) else {'status':'NO_PAIRS'}
            comparisons.append({'period':period,'metric':key,**comparison})
    return all_rows,paired_rows,comparisons


def run(archive,output):
    archive=Path(archive);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((archive/'report.json').read_text(encoding='utf-8'))
    for path,digest in report['source_hashes'].items():
        if sha(path)!=digest:raise ValueError('official source altered: '+path)
    for path,digest in report['artifact_hashes'].items():
        if sha(archive/path)!=digest:raise ValueError('amount observations altered: '+path)
    if sha(ORIGINAL)!=ORIGINAL_SHA:raise ValueError('original forward forecast changed')
    sources={str(p.resolve()):sha(p) for p in [Path(__file__),archive/'report.json',
        archive/'market_amounts_NOT_ORIGINAL_POOL_SCOPE.csv',CALENDAR,ORIGINAL,ROOT/'market_interval_research_v300.py']}
    write_json(out/'protocol.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'parameters':PROTOCOL,'source_hashes':sources})
    sessions=json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions']
    frame=pd.read_csv(archive/'market_amounts_NOT_ORIGINAL_POOL_SCOPE.csv')
    if frame.date.duplicated().any() or not set(frame.date).issubset(sessions):raise ValueError('amount dates invalid')
    obs=frame.set_index('date')[CHANNELS].reindex(sessions)
    if ((obs<0)&obs.notna()).any().any():raise ValueError('negative amount')
    obs.to_csv(out/'amount_grid_NOT_ORIGINAL_SCOPE.csv',index_label='date')
    pred=predict(obs,sessions);pred.to_csv(out/'predictions_NO_LABELS_NOT_FORWARD.csv',index=False)
    write_json(out/'prediction_receipt.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'predictions_sha256':sha(out/'predictions_NO_LABELS_NOT_FORWARD.csv'),
        'protocol_sha256':sha(out/'protocol.json'),'forward_evidence':False})
    scored=score(pred,obs);scored.to_csv(out/'scores_LABEL_ONLY.csv',index=False)
    all_rows,paired,comparisons=summaries(scored,sessions)
    pd.DataFrame(all_rows).to_csv(out/'coverage_and_scores.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_common_date_summary.csv',index=False)
    write_json(out/'paired_interval_score.json',comparisons)
    if sha(ORIGINAL)!=ORIGINAL_SHA:raise ValueError('original forecast changed during research')
    result={'version':PROTOCOL['version'],'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'OFFICIAL_BROADER_SCOPE_AMOUNT_INTERVAL_RESEARCH_ONLY','protocol':PROTOCOL,
        'source_dates':len(frame),'channels':2,'prediction_rows':len(pred),'metrics':all_rows,
        'paired_metrics':paired,'paired_comparisons':comparisons,'original_forecast_sha256':sha(ORIGINAL),
        'same_scope_original_turnover_slot_complete':False,'forward_evidence':False,'pure_blind_test':False,
        'core_upgrade_authorized':False,'stable_60_proved':False,'goal_complete':False,
        'source_hashes':sources,'artifact_hashes':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/'report.json',result)
    return {k:result[k] for k in ['status','source_dates','channels','prediction_rows']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.archive,a.output),ensure_ascii=False))
