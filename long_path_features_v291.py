"""Causal-content EOD features from verified 30-minute intervals; viewed history."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from long_interval_layers_v290 import load
import archive_long_interval_v290 as archive_engine
from closing_board_sanity_v290 import classify
from closing_label_audit_v290 import label_status
from next_session_forecast_lab_v271 import ROOT,FORBIDDEN
from relay_promotion_holdout_v256 import IDENTITY,EMOTION,PRICE_VOLUME

ARCHIVE=ROOT/'data/v290/full_long_interval_20261002_v1'
AUDIT=ROOT/'results/v290/long_interval_audit_20261002_v1/report.json'
RECOVERY=ROOT/'data/v290/long_interval_recovery_20261002_v1'
CONTROL=ROOT/'results/v279/daily_path_paired_20261002_v2/CONTROL_predictions_LABEL_ONLY.csv'
CONTROL_REPORT=CONTROL.parent/'report.json'
LABEL_AUDIT=ROOT/'results/v290/closing_label_audit_20261002_v1/report.json'
BASE_FIELDS=list(dict.fromkeys(IDENTITY+EMOTION+PRICE_VOLUME))+[
    key+'_same_height_percentile' for key in ['price','signal_amount','turnover','turnover_vs_prev']]+['signal_one_word','log_signal_amount']
PRICE_FIELDS=['m30_first_close_to_day_close','m30_midday_close_to_day_close','m30_afternoon_low_to_day_close',
    'm30_last_two_low_to_day_close','m30_afternoon_close_gain','m30_close_path_variation','m30_down_close_step_share',
    'm30_close_drawdown','m30_final_cap_price_bar_run_fraction','m30_cap_price_bar_share','m30_first_cap_high_slot']
VOLUME_FIELDS=['m30_close_vs_full_day_vwap','m30_afternoon_volume_share','m30_last_two_volume_share']
ALL_FIELDS=BASE_FIELDS+PRICE_FIELDS+VOLUME_FIELDS


def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def path_features(bars,check):
    result={key:np.nan for key in PRICE_FIELDS+VOLUME_FIELDS}
    rows=bars[bars.code.eq(check['code'])&bars.minute.str[:10].eq(check['date'])].sort_values('minute')
    if not(check['price_reconciled'] and check['grid_complete'] and check['original_pool_close_agrees']):return result
    if len(rows)!=8 or rows.minute.str[11:16].tolist()!=archive_engine.TIMES:raise ValueError('verified thirty-minute grid changed')
    close=rows.close.to_numpy(dtype=float);low=rows.low.to_numpy(dtype=float);cap=float(close[-1])
    changes=close[1:]/close[:-1]-1
    result.update(m30_first_close_to_day_close=float(close[0]/cap-1),m30_midday_close_to_day_close=float(close[3]/cap-1),
        m30_afternoon_low_to_day_close=float(low[4:].min()/cap-1),m30_last_two_low_to_day_close=float(low[-2:].min()/cap-1),
        m30_afternoon_close_gain=float(close[-1]/close[3]-1),m30_close_path_variation=float(np.abs(changes).sum()),
        m30_down_close_step_share=float((changes<0).mean()),m30_close_drawdown=float(close.min()/cap-1))
    if check['final_cap_price_bar_run'] is not None:
        result.update(m30_final_cap_price_bar_run_fraction=check['final_cap_price_bar_run']/8.,
            m30_cap_price_bar_share=check['cap_price_bar_share'],
            m30_first_cap_high_slot=(np.flatnonzero(rows.high.to_numpy()>=cap-.005001)[0]+1)/8.)
    if check['full_day_vwap'] is not None:
        volume=rows.volume_shares.to_numpy(dtype=float);total=volume.sum()
        result.update(m30_close_vs_full_day_vwap=float(cap/check['full_day_vwap']-1),
            m30_afternoon_volume_share=float(volume[4:].sum()/total),m30_last_two_volume_share=float(volume[-2:].sum()/total))
    return result


def prepare(output):
    target=Path(output)
    if target.exists():raise FileExistsError('immutable feature preparation exists')
    old=json.loads(CONTROL_REPORT.read_text(encoding='utf-8'))
    if old['fields']['CONTROL']!=BASE_FIELDS:raise ValueError('existing control feature definition changed')
    for name,wanted in old['source_hashes'].items():
        path=ROOT/name
        if archive_engine.sha(path)!=wanted:raise ValueError('existing control source changed')
    label_proof=json.loads(LABEL_AUDIT.read_text(encoding='utf-8'))
    for name,wanted in label_proof['source_hashes'].items():
        if archive_engine.sha(name)!=wanted:raise ValueError('label audit source changed')
    scope,checks,bars,daily,inputs=load(ARCHIVE,AUDIT,RECOVERY,with_bars=True)
    base=pd.read_csv(CONTROL,dtype={'code':str},usecols=list(dict.fromkeys(['date','code',*BASE_FIELDS])))
    frame=scope[['date','code','daily_true_one_word']].merge(base,on=['date','code'],how='left',validate='one_to_one')
    if frame.boards.isna().any():raise ValueError('original full candidate features missing')
    # Remove every nonsignal day before building model fields.
    signal_bars=bars.assign(date=bars.minute.str[:10]).merge(scope[['date','code']],on=['date','code'],validate='many_to_one')
    groups={key:part for key,part in signal_bars.groupby(['date','code'])}
    paths=pd.DataFrame([dict(date=c['date'],code=c['code'],**path_features(groups[(c['date'],c['code'])],c)) for c in checks])
    frame=frame.merge(paths,on=['date','code'],validate='one_to_one')
    if len(frame)!=len(scope) or frame.duplicated(['date','code']).any() or set(ALL_FIELDS)&FORBIDDEN:raise ValueError('feature scope or label leakage')
    target.mkdir(parents=True,exist_ok=False)
    feature_path=target/'features_NO_NEXT_SESSION_LABELS.csv';frame.to_csv(feature_path,index=False,encoding='utf-8-sig')
    # Join outcomes only after the allowlisted feature file is persisted.
    labels=pd.read_csv(CONTROL,dtype={'code':str},usecols=['date','code','target_date','target','label_valid'])
    labels=scope[['date','code']].merge(labels,on=['date','code'],how='left',validate='one_to_one')
    if labels.label_valid.isna().any():raise ValueError('original label cohort missing')
    labels=labels.rename(columns={'target':'target_original'});labels['target_pressure_zero']=labels.target_original
    labels['pressure_reason']='UNCHANGED_ORIGINAL'
    indexed={(c['date'],c['code']):c for c in checks};bar_groups={code:part for code,part in bars.groupby('code')}
    day_groups={symbol:part for symbol,part in daily.groupby('symbol')};contradictions=[]
    for row in labels[labels.label_valid&labels.target_original.eq(1)].itertuples():
        signal=classify(indexed[(row.date,row.code)]);next_status='UNVERIFIED_SOURCE'
        refs=day_groups[archive_engine.minute.live.sina_symbol(row.code)]
        match=refs[refs.date.eq(row.target_date)]
        if len(match)==1:
            ref=match.iloc[0];word=float(np.isclose(ref.high,ref.low,atol=.001,rtol=0))
            next_check=archive_engine.check_signal(bar_groups[row.code],refs,row.code,row.target_date,float(ref.close),word)
            next_status=classify(next_check)['status']
        status=label_status(1.,True,signal['status'],next_status)
        if status in ['POSITIVE_CONTINUATION_SIGNAL_CONTRADICTION','POSITIVE_NEXT_CLOSE_CONTRADICTION']:
            contradictions.append((row.date,row.code,status))
            labels.loc[row.Index,['target_pressure_zero','pressure_reason']]=[0.,status]
    expected={(r['date'],r['code'],r['audit_status']) for r in label_proof['positive_contradictions']}
    if set(contradictions)!=expected:raise ValueError('label contradiction set differs from independently reconstructed prices')
    labels_path=target/'labels_ORIGINAL_AND_PRESSURE_ONLY.csv';labels.to_csv(labels_path,index=False,encoding='utf-8-sig')
    legacy=pd.read_csv(CONTROL,dtype={'code':str},usecols=['date','code','probability','predicted'])
    legacy=scope[['date','code']].merge(legacy,on=['date','code'],validate='one_to_one')
    legacy_path=target/'legacy_predictions_NO_LABELS.csv';legacy.to_csv(legacy_path,index=False,encoding='utf-8-sig')
    record=dict(version='v291-long-path-feature-preparation-1',recorded_at=archive_engine.timestamp(),status='RETROSPECTIVE_FEATURES_AND_SEPARATE_LABEL_PRESSURE',
        candidate_rows=len(frame),candidate_dates=int(frame.date.nunique()),base_fields=BASE_FIELDS,price_fields=PRICE_FIELDS,volume_fields=VOLUME_FIELDS,
        coverage={key:float(frame[key].notna().mean()) for key in PRICE_FIELDS+VOLUME_FIELDS},pressure_rows=len(contradictions),
        pressure_meaning='Sensitivity assumption: original six contradictory positive labels treated as zero in a separate view; not certified complete truth',
        original_labels_rewritten=False,original_forecast_rewritten=False,goal_verified=False,trade_authorized=False,
        source_hashes={str(Path(p).resolve()):archive_engine.sha(p) for p in [*inputs,CONTROL,CONTROL_REPORT,LABEL_AUDIT,Path(__file__),
            ROOT/'closing_board_sanity_v290.py',ROOT/'closing_label_audit_v290.py',ROOT/'archive_long_interval_v290.py']},
        artifact_hashes={p.name:archive_engine.sha(p) for p in [feature_path,labels_path,legacy_path]})
    dump(target/'report.json',record)
    return {key:record[key] for key in ['status','candidate_rows','candidate_dates','coverage','pressure_rows']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(prepare(parser.parse_args().output),ensure_ascii=True))
