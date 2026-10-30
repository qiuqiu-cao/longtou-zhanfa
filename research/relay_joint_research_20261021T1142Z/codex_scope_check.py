"""Read-only relay scope and saved-baseline checks. No training or downloads."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'research/model_upgrade_validation/short_term_thinking_20261015T071000Z'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
    if isinstance(v,np.generic): return v.item()
    return v

def metric(x,all_dates,k=1):
    picks=x.sort_values(['date','raw_score','code'],ascending=[True,False,True]).groupby('date',sort=False).head(k)
    known=picks.target.notna(); hits=int(picks.target.eq(1).sum()); n=len(picks)
    return dict(selected=n,selected_dates=int(picks.date.nunique()),known=int(known.sum()),unknown=int((~known).sum()),hits=hits,
                full_date_slot_hit_rate=hits/(k*all_dates),hits_per_date_count=hits/all_dates,
                dates_with_any_hit_rate=picks[picks.target.eq(1)].date.nunique()/all_dates,
                known_precision=hits/int(known.sum()) if known.any() else None,
                all_dates=all_dates,selection_date_coverage=picks.date.nunique()/all_dates)

def main():
    d=joblib.load(SOURCE/'scope_correction/dataset.joblib')
    old=joblib.load(SOURCE/'dataset.joblib')
    executed=joblib.load(SOURCE/'historical_final/dataset.joblib')
    pd.testing.assert_frame_equal(d['features'].sort_values(['date','code']).reset_index(drop=True),
                                  executed['features'].sort_values(['date','code']).reset_index(drop=True))
    a=d['features'].sort_values(['date','code']).reset_index(drop=True)
    b=old['features'].sort_values(['date','code']).reset_index(drop=True)
    changed={}
    for col in a.columns:
        # Object list-valued fields require a canonical representation.
        av=a[col].map(lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False) if isinstance(x,(dict,list)) else x)
        bv=b[col].map(lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False) if isinstance(x,(dict,list)) else x)
        neq=~(av.eq(bv)|(av.isna()&bv.isna()))
        if neq.any():changed[col]=int(neq.sum())
    relay=d['prepared']['relay']
    for key in ['date','code']:
        assert relay[key].notna().all()
    full=relay.merge(d['features'][['date','code','daily_one_word']],on=['date','code'],validate='one_to_one')
    basepath=ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/relay_history/predictions.csv'
    rollpath=ROOT/'research/model_upgrade_validation/recency_walkforward_20261013T134351Z/full_candidate_predictions.csv'
    base=pd.read_csv(basepath,dtype={'code':str},low_memory=False)
    base=base[base.arm.eq('BASE')]
    roll=pd.read_csv(rollpath,dtype={'code':str},low_memory=False)
    roll=roll[roll.arm.eq('ROLL252')].rename(columns={'signal_date':'date','truth':'target'})
    evidence=dict(at=datetime.now(timezone.utc).isoformat(),fits=0,http_requests=0,
        candidate_scope='EXISTING_ARCHIVED_MAINBOARD_RELAY_COHORT_NOT_NEW_FIRST_BOARD_RESEARCH',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'dataset.joblib',SOURCE/'scope_correction/dataset.joblib',
                   SOURCE/'historical_final/dataset.joblib',SOURCE/'protocol_BEFORE_ANY_FIT.json',basepath,rollpath]},
        scope_correction_features_equal_actual_executed_historical_final=True,
        original_and_scope_correction_9245_features_equal=not changed,
        original_vs_scope_correction_changed_fields=changed,
        selected_feature_bundle='scope_correction/dataset.joblib; bind against actual executed training bundle before freeze',
        rows=len(full),signal_dates=int(full.date.nunique()),known_targets=int(full.target.notna().sum()),unknown_targets=int(full.target.isna().sum()),
        boards=full.boards.value_counts().sort_index().to_dict(),
        feature_coverage={k:dict(known=int(relay[k].notna().sum()),unknown=int(relay[k].isna().sum())) for k in ['industry','theme','first_seal_minute','last_seal_minute','bomb_count','turnover','signal_amount']},
        coverage_is_not_source_or_time_qualification=True,stored_baselines={})
    for name,pred in [('BASE',base),('ROLL252',roll)]:
        assert not pred.duplicated(['date','code']).any()
        m=pred[['date','code','target','raw_score']].merge(full[['date','code','target','daily_one_word','boards']],
                            on=['date','code'],validate='one_to_one',suffixes=('_saved',''))
        assert len(m)==len(pred)
        assert (m.target.fillna(-1)==m.target_saved.fillna(-1)).all()
        n=m.date.nunique()
        results=dict(rows=len(m),dates=int(n),start=m.date.min(),end=m.date.max(),known=int(m.target.notna().sum()),unknown=int(m.target.isna().sum()),
            target_values_exactly_match_current_dataset=True,
            full_top1=metric(m,n),full_top2=metric(m,n,2),
            non_one_price_top1=metric(m[m.daily_one_word.eq(0)],n),
            non_one_price_top2=metric(m[m.daily_one_word.eq(0)],n,2),
            per_month={})
        for month,g in m.groupby(m.date.str[:7]):
            total=g.date.nunique()
            results['per_month'][month]=dict(full_top1=metric(g,total),non_one_price_top1=metric(g[g.daily_one_word.eq(0)],total))
        evidence['stored_baselines'][name]=results
    # Save exact source qualification limitations separately from null coverage.
    evidence['time_policy']='Historical causal reconstruction authorized; no invented original acquisition times or independent role ground truth.'
    evidence['baseline_policy']='BASE and ROLL252 stay separate; original first-board BASE62/52-of-117 is not a relay baseline.'
    p=OUT/'CODEX_RELAY_DATA_SCOPE.json'
    p.write_text(json.dumps(clean(evidence),ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(clean({k:v for k,v in evidence.items() if k!='sources' and k!='stored_baselines'}),ensure_ascii=False))
    print(json.dumps(clean({k:{t:v for t,v in item.items() if t!='per_month'} for k,item in evidence['stored_baselines'].items()}),ensure_ascii=False))

if __name__=='__main__':main()
