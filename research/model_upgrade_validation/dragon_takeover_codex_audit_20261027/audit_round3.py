"""Read-only review of saved round-three research. No estimator fitting."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'research/model_upgrade_validation/dragon_takeover_20261027T074500Z'
DATA=ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def choose(g,key,ascending=False):
    return g.sort_values(['date',key,'code'],ascending=[True,ascending,True]).groupby('date').head(1).set_index('date')
def rates(g):
    return {'n':len(g),**{f:{'hits':int(g[f].sum()),'rate':float(g[f].mean())}
        for f in ['is_market_space_leader','target_B_takeover_space','is_emotion_dragon_proxy','is_3to4','is_3to5','is_deep_6','is_monster_7']}}
def paired(a,b,target):
    z=a[['code',target]].join(b[['code',target]],lsuffix='_model',rsuffix='_control',how='inner')
    win=int((z[target+'_model'].eq(1)&z[target+'_control'].eq(0)).sum())
    loss=int((z[target+'_model'].eq(0)&z[target+'_control'].eq(1)).sum())
    return {'n':len(z),'model_only_hits':win,'control_only_hits':loss,'net_hits':win-loss,
        'two_sided_discordant_binomial_p':float(binomtest(win,win+loss).pvalue) if win+loss else 1.0,
        'limitation':'Diagnostic paired-day test; does not adjust for repeated research or dependence between dates.'}

def main():
    files=[p for p in SRC.iterdir() if p.is_file()]+[DATA,ROOT/'compute_round3_dataset.py',ROOT/'train_and_evaluate_round3.py',ROOT/'evaluate_round3_comprehensive.py',ROOT/'audit_three_board_traps.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    d=pd.read_csv(SRC/'round3_dataset.csv',dtype={'code':str},low_memory=False)
    o=pd.read_csv(SRC/'oof_round3_predictions.csv',dtype={'code':str})
    d.code=d.code.str.zfill(6);o.code=o.code.str.zfill(6)
    prev_path=ROOT/'research/model_upgrade_validation/dragon_stage_leader_codex_audit_20261027/all1851_label_diagnostics.csv'
    prev=pd.read_csv(prev_path,dtype={'code':str});prev.code=prev.code.str.zfill(6)
    # Reuse separately verified exact official-session chain diagnostics; bind its hash.
    z=d.merge(prev[['date','code','strict_chain_completed','strict_chain_max','strict_future_space','strict_dragon_proxy','already_space_on_signal','failure_known_on']],on=['date','code'],validate='one_to_one')
    mature=z[z.strict_chain_completed].copy()
    mature['expected_unified_space']=np.where(mature.already_space_on_signal.eq(1),mature.strict_chain_max.ge(4).astype(int),mature.strict_future_space)
    label_checks={'mature_streak_rows':len(mature),'incomplete_streak_rows':len(z)-len(mature),
        'space_label_disagreements':int(mature.is_market_space_leader.ne(mature.expected_unified_space).sum()),
        'emotion_proxy_disagreements':int(mature.is_emotion_dragon_proxy.ne(mature.strict_dragon_proxy).sum()),
        'final_maturity_date_disagreements':int(mature.determined_date_final.ne(mature.failure_known_on).sum())}
    for n,label in [(4,'is_3to4'),(5,'is_3to5'),(6,'is_deep_6'),(7,'is_monster_7')]:
        label_checks[label+'_disagreements']=int(mature[label].ne(mature.strict_chain_max.ge(n).astype(int)).sum())
    mism=mature[mature.is_market_space_leader.ne(mature.expected_unified_space)|mature.is_emotion_dragon_proxy.ne(mature.strict_dragon_proxy)|mature.determined_date_final.ne(mature.failure_known_on)]
    mism.to_csv(OUT/'mature_label_disagreements.csv',index=False,encoding='utf-8-sig')
    x=o.merge(d[['date','code','amplitude','price','turnover','signal_amount','first_seal_minute','last_seal_minute','determined_date_final']],on=['date','code'],validate='one_to_one')
    c=x.groupby('date').filter(lambda g:len(g)>=2)
    top=choose(c,'score_round3_composite')
    rec=top[top.score_round3_composite.ge(.4)]
    b=c[c.is_already_space_leader.eq(0)].groupby('date').filter(lambda g:len(g)>=2)
    btop=choose(b,'score_round3_composite')
    baselines={};diagnostic_pairs={}
    for name,col,asc in [('min_amplitude','amplitude',True),('min_turnover','turnover',True),('min_price','price',True),('max_amount','signal_amount',False)]:
        picked=choose(c,col,asc)
        baselines[name]=rates(picked)
        diagnostic_pairs[name]={target:paired(top,picked,target) for target in ['is_market_space_leader','is_emotion_dragon_proxy','is_3to5']}
    complete=c[c.base_cohort_match_status.eq('COMPLETE_MATCH')]
    mt=choose(complete,'score_round3_composite');bt=choose(complete,'base_historical_score')
    base_pairs={target:paired(mt,bt,target) for target in ['is_market_space_leader','is_emotion_dragon_proxy','is_3to5']}
    top.reset_index().to_csv(OUT/'top1_recomputed.csv',index=False,encoding='utf-8-sig')
    d[d.terminal_height_censored.eq(1)].to_csv(OUT/'cutoff_rows_still_censored.csv',index=False,encoding='utf-8-sig')
    daily=pd.read_csv(SRC/'daily_competition_round3.csv',dtype={'top1_code':str})
    manifest=json.loads((SRC/'manifest_round3.json').read_text(encoding='utf-8'))
    metrics=json.loads((SRC/'round3_comprehensive_metrics.json').read_text(encoding='utf-8'))
    model_info={}
    for p in SRC.glob('model*.joblib'):
        m=joblib.load(p)
        model_info[p.name]={'loaded':True,'feature_count':m.n_features_in_,'sha256':sha(p)}
    raw=pd.read_csv(DATA,usecols=['date','code','boards'],dtype={'code':str})
    cal=json.loads((ROOT/'data/v279/exchange_calendar_audit_20261002_v1/calendar.json').read_text(encoding='utf-8'))['sessions']
    s=sorted(raw.date.unique());official=[v for v in cal if s[0]<=v<=s[-1]]
    folds=[]
    for end,start in [('2024-08-31','2024-09-01'),('2024-12-31','2025-01-01'),('2025-05-31','2025-06-01'),('2025-11-30','2025-12-01')]:
        train=z[z.date.le(end)&z.determined_date_final.le(end)]
        folds.append({'test_start':start,'train_n':len(train),'unknown_chain_rows_in_train':int((~train.strict_chain_completed).sum()),
            'actual_maturity_after_train_end':int(train.failure_known_on.gt(end).sum())})
    bmanifest=ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/manifest.json'
    bm=json.loads(bmanifest.read_text(encoding='utf-8'))
    x['omitted_takeover_score']=.35*x.score_space_leader+.25*x.score_emotion_dragon+.15*x.score_height_5
    omit=choose(x.groupby('date').filter(lambda g:len(g)>=2),'omitted_takeover_score')
    report={'audit_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'NO_FIT_NO_SOURCE_MUTATION',
        'source_hashes':hashes,'prior_exact_chain_diagnostics_sha256':sha(prev_path),
        'label_checks_mature_1843':label_checks,'calendar_matches_official':s==official,
        'fold_maturity_checks':folds,'original_score_top1':rates(top),'selective_040':rates(rec),
        'selected_coverage':len(rec)/len(top),'case_b_takeover':rates(btop),
        'all_signal_days_top1':rates(choose(x,'score_round3_composite')),
        'simple_benchmarks_same_pool':baselines,'paired_diagnostic_vs_simple':diagnostic_pairs,
        'complete_base':{'model':rates(mt),'base':rates(bt),'paired_diagnostics':base_pairs},
        'artifact_completeness':{'loaded_models':model_info,'takeover_model_saved':bool(list(SRC.glob('*takeover*.joblib'))),
            'takeover_weight':.25,'top1_changes_if_component_omitted':int(top.code.ne(omit.code).sum()),
            'preprocessing_thresholds_saved_in_manifest':('med_price' in manifest or 'med_cap' in manifest)},
        'execution_audit':{'veto_n':int(daily.top1_audit_veto.sum()),'tag_counts':daily.top1_audit_tags.value_counts().to_dict(),
            'top1_missing_first_seal':int(top.first_seal_minute.isna().sum()),
            'top1_known_first_seal_values':top.first_seal_minute.dropna().unique().tolist(),
            'base_missing_defaults_to_50_code_still_present':True,'historical_risk_defaults_to_false_code_still_present':True,
            'unknown_string_compares_after_1400':bool('UNKNOWN'>='14:00')},
        'base_hash_matches':sha(bmanifest.parent/'model.joblib')==bm['sha256'],
        'new_fit_count':0,'new_trade_count':0}
    report['source_files_preserved']=hashes=={str(p.relative_to(ROOT)):sha(p) for p in files}
    report['checks']={'preserved_sources':report['source_files_preserved'],'base_hash':report['base_hash_matches'],
        'raw_top1_space_matches_json':round(top.is_market_space_leader.mean(),4)==metrics['competition_overall']['top1_space_rate'],
        'raw_top1_dragon_matches_json':round(top.is_emotion_dragon_proxy.mean(),4)==metrics['competition_overall']['top1_dragon_rate'],
        'selective_space_matches_json':round(rec.is_market_space_leader.mean(),4)==metrics['selective_recommendation']['rec_space_rate'],
        'all_1843_mature_targets_match':all(label_checks[k]==0 for k in label_checks if k.endswith('disagreements')),
        'calendar_matches':report['calendar_matches_official'],
        'no_observed_late_training_labels':all(f['actual_maturity_after_train_end']==0 and f['unknown_chain_rows_in_train']==0 for f in folds)}
    assert all(report['checks'].values()),report['checks']
    def default(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=default),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','paired_diagnostic_vs_simple','simple_benchmarks_same_pool']},ensure_ascii=False,indent=2,default=default))

if __name__=='__main__':main()
