"""Fixed monthly paired test of daily-path clues against the same control pool.

Six V274 model/window choices per arm, chosen on prior 63 known-label dates.
Both arms select on true OHLC non-one-word Top1, retain all candidates, and
never tune on a forecast month's outcomes. Reused history remains diagnostic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from next_session_forecast_lab_v271 import ROOT,DATA,prepare_relay,prediction,top1,rate,sha
from rolling_forecast_challenger_v274 import KINDS,WINDOWS,VALIDATION_DATES,START,END,known_before,trailing,fit
from relay_daily_path_features_v279 import NEW_FIELDS
from relay_promotion_holdout_v256 import _paired

VERSION="v279-daily-path-paired-monthly-1"
CLUES=ROOT/"data/v279/daily_path_clues_20261002_v1"


def selected_nonword(pred):
    return top1(pred[pred.daily_true_one_word.eq(0)])


def frozen(frame,fields,cutoff):
    known=known_before(frame,cutoff)
    dates=sorted(known.date.unique());validation_start=dates[-VALIDATION_DATES]
    train=known[(known.date<validation_start)&(known.target_date<validation_start)]
    val=known[known.date>=validation_start]
    choices=[]
    for kind in KINDS:
        for window in WINDOWS:
            subset=trailing(train,window)
            mdl=fit(subset,fields,kind,"relay")
            p=selected_nonword(prediction(mdl,val,fields,"relay"))
            choices.append({"kind":kind,"window":window,"validation_accuracy":float(p.target.mean()),"validation_dates":len(p)})
    chosen=sorted(choices,key=lambda r:(-r['validation_accuracy'],KINDS.index(r['kind']),-(r['window'] or 100000)))[0]
    refit=trailing(known,chosen['window'])
    if refit.target_date.max()>=cutoff or val.target_date.max()>=cutoff:
        raise ValueError("future realization used in selection/refit")
    return fit(refit,fields,chosen['kind'],"relay"),{"cutoff":cutoff,"chosen":chosen,"validation_candidates":choices,
        "max_refit_label_date":refit.target_date.max(),"max_validation_label_date":val.target_date.max(),
        "validation_start":validation_start,"refit_dates":int(refit.date.nunique()),
        "test_labels_used":False,"confidence_status":"UNCALIBRATED_AFTER_REFIT"}


def summarize(pred):
    complete=pred.groupby("date").label_valid.all()
    evaluated=pred[pred.date.isin(complete[complete].index)&pred.target.notna()]
    masks={"identity":pd.Series(True,index=evaluated.index),"legacy_non_one_word_proxy":evaluated.one_word_proxy.eq(0),
           "ohlc_non_one_word":evaluated.daily_true_one_word.eq(0),
           "two_board_ohlc_nonword":evaluated.daily_true_one_word.eq(0)&evaluated.boards.eq(2),
           "three_plus_ohlc_nonword":evaluated.daily_true_one_word.eq(0)&evaluated.boards.ge(3)}
    result={"dates_total":int(pred.date.nunique()),"complete_label_dates":int(evaluated.date.nunique())}
    for key,mask in masks.items():
        pick=top1(evaluated[mask]).copy();pick['correct']=pick.target.astype(int)
        result[key]=rate(pick)
    return result


def eligible_nonword(pred):
    complete=pred.groupby('date').label_valid.all()
    picked=selected_nonword(pred[pred.date.isin(complete[complete].index)&pred.target.notna()]).copy()
    return picked[["date","target"]].rename(columns={"target":"promoted_next_day"})


def run(output):
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    sessions=pd.read_csv(DATA/'market_emotion.csv').date.tolist()
    base,fields=prepare_relay(DATA/'leader_candidates.csv',sessions)
    hp=pd.read_csv(CLUES/'historical_relay_paths_LABEL_ONLY.csv',dtype={'code':str},low_memory=False)
    base=base.merge(hp[['date','code',*NEW_FIELDS,'episode_daily_complete']],on=['date','code'],how='left',validate='one_to_one')
    september=pd.read_csv(CLUES/'september_relay_paths_LABEL_ONLY.csv',dtype={'code':str},low_memory=False)
    report={'version':VERSION,'status':'DIAGNOSTIC_ONLY','goal_verified':False,'arms':{},'paired':{},'selection_logs':{},
        'specification':{'arms':['CONTROL','DAILY_CLUES'],'model_kinds':list(KINDS),'windows':list(WINDOWS),
          'validation_dates':VALIDATION_DATES,'selection_target':'true OHLC non-one-word positive Top1 promotion',
          'update':'monthly','test_target':'exact next-session closing next-board promotion'},
        'fields':{'CONTROL':fields,'DAILY_CLUES':fields+NEW_FIELDS},'trade_authorized':False,
        'limitations':['Reused historical and September labels; no pristine blind or forward evidence',
          'Daily clues do not establish real theme, sector causality, absorption, minute reseal or fill',
          'Historical observed calendar/universe still incomplete',
          'September previous3 amount mostly unavailable; training-to-deployment drift exposed',
          'Both arms reselect on true non-one-word pool; control differs from earlier legacy-proxy validation'],
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [DATA/'leader_candidates.csv',DATA/'market_emotion.csv',
          CLUES/'historical_relay_paths_LABEL_ONLY.csv',CLUES/'september_relay_paths_LABEL_ONLY.csv',CLUES/'report.json',Path(__file__),
          ROOT/'relay_daily_path_features_v279.py',ROOT/'rolling_forecast_challenger_v274.py',
          ROOT/'next_session_forecast_lab_v271.py',ROOT/'relay_promotion_holdout_v256.py']}}
    all_outputs={}
    for arm,used in report['fields'].items():
        outputs=[];logs=[]
        for month in pd.period_range(START[:7],END[:7],freq='M'):
            cutoff=str(month.start_time.date());end=min(str(month.end_time.date()),END)
            test=base[base.date.between(cutoff,end)]
            if test.empty:continue
            mdl,log=frozen(base,used,cutoff)
            p=prediction(mdl,test,used,'relay');p['model_freeze']=cutoff;p['arm']=arm
            p['fold']='2025H2' if cutoff<'2026-01-01' else '2026YTD'
            outputs.append(p);logs.append(log)
        mdl,log=frozen(base,used,'2026-09-01')
        p=prediction(mdl,september,used,'relay');p['model_freeze']='2026-09-01';p['arm']=arm;p['fold']='SEPTEMBER'
        outputs.append(p);logs.append(log)
        pred=pd.concat(outputs,ignore_index=True);all_outputs[arm]=pred
        report['arms'][arm]={f:summarize(g) for f,g in pred.groupby('fold')}
        report['arms'][arm]['combined_pre_september']=summarize(pred[pred.fold.ne('SEPTEMBER')])
        report['selection_logs'][arm]=logs
        pred.to_csv(target/(arm+'_predictions_LABEL_ONLY.csv'),index=False,encoding='utf-8-sig')
        (target/(arm+'_checkpoint.json')).write_text(json.dumps(
            {'status':'ARM_FINISHED_PAIRING_PENDING','summary':report['arms'][arm],
             'selection_logs':logs,'prediction_sha256':sha(target/(arm+'_predictions_LABEL_ONLY.csv')),
             'source_hashes':report['source_hashes']},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'arm_finished':arm,'nonword':{k:v['ohlc_non_one_word'] for k,v in report['arms'][arm].items()}},ensure_ascii=True),flush=True)
    for fold in ['2025H2','2026YTD','SEPTEMBER']:
        control=eligible_nonword(all_outputs['CONTROL'][all_outputs['CONTROL'].fold.eq(fold)])
        challenger=eligible_nonword(all_outputs['DAILY_CLUES'][all_outputs['DAILY_CLUES'].fold.eq(fold)])
        report['paired'][fold]=_paired(challenger,control,seed=279)
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return {'status':report['status'],'paired':report['paired'],'goal_verified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.output),ensure_ascii=True))
