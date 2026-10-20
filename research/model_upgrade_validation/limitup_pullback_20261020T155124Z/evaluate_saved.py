"""Append-only evaluation repair: identical missing target dates compare equal.

No fitting or score recomputation. All 15 models and scores predate this file.
"""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('study_frozen',RUN/'study.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def evaluate():
    e=s.read('implementation_BEFORE_FITS.json');s.verify(e['bindings'])
    files=sorted(RUN.glob('*_predictions_BEFORE_VALIDATION.csv'))
    assert len(files)==15 and len(list(RUN.glob('*_fit_COMPLETED.json')))==15
    frozen={str(p):s.sha(p) for p in files+sorted(RUN.glob('*_PULLBACK_PERSONALITY.joblib'))}
    s.write('evaluation_repair_BEFORE_READ.json',dict(at=s.now(),reason='18 paired target_date values are both NaN and label_valid=false; equality permits both absent without changing unknown labels or scores',source_code=s.sha(RUN/'study.py'),repair_code=s.sha(__file__),frozen_scores_and_models=frozen,new_fits=0))
    b=s.joblib.load(RUN/'prepared_FEATURES_BEFORE_FITS.joblib')
    pred=pd.concat([pd.read_csv(p,dtype={'code':str},low_memory=False) for p in files],ignore_index=True)
    q=b['saved_roll'].rename(columns={'signal_date':'date','raw_score':'roll_score'})[['date','code','target_date','truth','roll_score']]
    pred=pred.merge(q,on=['date','code'],validate='one_to_one',suffixes=('','_saved'))
    equal=pred.target_date.eq(pred.target_date_saved)|(pred.target_date.isna()&pred.target_date_saved.isna())
    assert len(pred)==3707 and equal.all()
    v=pred.label_valid.fillna(False).astype(bool)&pred.target.notna()
    assert np.array_equal(pred.loc[v,'target'].astype(bool).to_numpy(),pred.loc[v,'truth'].astype(bool).to_numpy())
    assert pred.loc[~v,'truth'].isna().all()
    pred=pred.merge(b['saved_base'][['date','code','raw_score']].rename(columns={'raw_score':'base_score_saved'}),on=['date','code'],validate='one_to_one')
    pred.to_csv(RUN/'full_paired_predictions.csv',index=False,encoding='utf-8-sig')
    report=dict(formed_at=s.now(),fits=15,baseline_fits=0,production_refits=0,full_n=len(pred),folds={})
    for fold in s.FOLDS:
        f=pred[pred.date.between(fold['start'],fold['end'])]
        groups=dict(all=f,actual_nonword=f[f.daily_one_word.eq(0)],actual_oneword=f[f.daily_one_word.eq(1)],identity_unknown=f[f.daily_one_word.isna()],cap_pullback=f[f.pullback_has_anchor.eq(1)],no_pullback=f[f.pullback_has_anchor.eq(0)],pullback_unknown=f[f.pullback_has_anchor.isna()])
        result={}
        for name,g in groups.items():
            result[name]={arm:dict(metrics=s.metrics(g,col),top={str(k):s.top(g,col,k) for k in (1,2,3)}) for arm,col in [('ROLL252','roll_score'),('PULLBACK_PERSONALITY','new_score'),('FROZEN_BASE','base_score_saved')]}
        result['nonword_F1_delta95']=s.bootstrap(groups['actual_nonword'],'new_score')
        report['folds'][fold['id']]=result
    sep=report['folds']['2026Sep']['actual_nonword'];old=sep['ROLL252']['metrics'];new=sep['PULLBACK_PERSONALITY']['metrics']
    checks=dict(nonword_F1_all_folds=all(v['actual_nonword']['PULLBACK_PERSONALITY']['metrics']['F1']>=v['actual_nonword']['ROLL252']['metrics']['F1'] for v in report['folds'].values()),sep_recall_plus10pp=new['Recall']>=old['Recall']+.10,sep_precision_not_less5pp=new['Precision'] is not None and old['Precision'] is not None and new['Precision']>=old['Precision']-.05,sep_nonword_top2_not_worse=sep['PULLBACK_PERSONALITY']['top']['2']['hits']>=sep['ROLL252']['top']['2']['hits'])
    report.update(fixed_screen=checks,screen_passed=all(checks.values()),statistically_supported=report['folds']['2026Sep']['nonword_F1_delta95']['lower']>0,deployed=False,new_forward_labels=0,calibrated_probability=False,evaluation_repair_only=True)
    s.verify(frozen);s.verify(e['bindings'])
    s.write('results.json',report)
    compact={k:{g:{a:v[g][a]['metrics'] for a in ('ROLL252','PULLBACK_PERSONALITY','FROZEN_BASE')} for g in ('all','actual_nonword','cap_pullback')} for k,v in report['folds'].items()}
    print(json.dumps(dict(results=compact,screen=checks,F1_delta95={k:v['nonword_F1_delta95'] for k,v in report['folds'].items()})),flush=True)


if __name__=='__main__':evaluate()
