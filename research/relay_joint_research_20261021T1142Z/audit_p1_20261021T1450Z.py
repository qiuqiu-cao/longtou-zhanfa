"""Independent zero-fit checks of Gemini's P1; preserve original artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
P1=ROOT/'p1_zero_fit'
OUT=ROOT/'codex_p1_review_20261021T1450Z'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    if isinstance(v,np.generic):return v.item()
    return v

def day_gap(df,nonword=False):
    x=df[df.boards.eq(2)&df.target.notna()&df.same_height_amount_rank_delta.notna()].copy()
    if nonword:x=x[x.daily_one_word.eq(0)]
    original_dates=df[df.boards.eq(2)].date.nunique()
    g=x.groupby(['date','target']).same_height_amount_rank_delta.agg(['mean','count']).unstack('target')
    g=g[g['mean'][0].notna()&g['mean'][1].notna()]
    # Both groups must exist for a descriptive paired difference; no selection
    # rule or hit-rate denominator is changed by this diagnostic subset.
    diffs=(g['mean'][0]-g['mean'][1]).to_numpy()
    rng=np.random.default_rng(20261021);draws=[];n=len(diffs);block=5
    for _ in range(4000):
        starts=rng.integers(0,n-block+1,(n+block-1)//block)
        idx=(starts[:,None]+np.arange(block)).ravel()[:n]
        draws.append(diffs[idx].mean())
    return dict(original_board2_dates=int(original_dates),paired_observed_dates=n,unpaired_dates=int(original_dates-n),
                loss_minus_win_delta_mean_by_day=float(diffs.mean()),
                moving_block5_observed_days_descriptive_ci95=np.quantile(draws,[.025,.975]).tolist(),
                caveat='Exploratory, outcome-conditioned mixed days only; not selection performance, independent causality or new blind evidence.')

def main():
    OUT.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('p1_under_review',P1/'run_p1_zero_fit_research.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sessions=[f'd{i}' for i in range(7)]
    # Synthetic source has a prior high anchor and middle-tier member. There
    # is no current price/board observation for that code.
    attempts=pd.DataFrame([dict(signal_date='d3',code='000001',board_height=4,pool_assignment='CLOSING_LIMIT_UP'),
                           dict(signal_date='d3',code='000002',board_height=1,pool_assignment='CLOSING_LIMIT_UP'),
                           dict(signal_date='d4',code='000002',board_height=2,pool_assignment='CLOSING_LIMIT_UP')])
    bars=pd.DataFrame([dict(date='d3',code='000001',amount=100.),dict(date='d3',code='000002',amount=50.),
                       dict(date='d4',code='000002',amount=70.)])
    f=pd.DataFrame([dict(date='d4',code='000002',boards=2)])
    unknown_result=module.build_p1_features(sessions,attempts,bars,f).iloc[0].to_dict()
    assert unknown_result['anchor_break_today']==1
    assert unknown_result['mid_tier_failed_count']==1
    assert unknown_result['mid_tier_damaged_ratio']==1
    attempts2=pd.DataFrame([dict(signal_date=day,code=code,board_height=level,pool_assignment='CLOSING_LIMIT_UP')
                            for day,level in [('d3',1),('d4',2)] for code in ['000002','000003']])
    bars2=pd.DataFrame([dict(date=day,code=code,amount=amount)
                       for day in ['d3','d4'] for code,amount in [('000002',100.),('000003',np.nan)]])
    f2=pd.DataFrame([dict(date='d4',code=c,boards=2) for c in ['000002','000003']])
    missing_rank=module.build_p1_features(sessions,attempts2,bars2,f2)
    assert missing_rank.loc[missing_rank.code.eq('000002'),'same_height_amount_rank_delta'].notna().all()
    original=json.loads((P1/'COMPLETION_P1_RESEARCH.json').read_text('utf-8'))
    assert digest(P1/'p1_features_9245.csv')==original['artifacts_produced']['features_sha256']
    data=pd.read_csv(P1/'p1_features_9245.csv',dtype={'code':str},float_precision='round_trip')
    assert len(data)==9245 and not data.duplicated(['date','code']).any()
    computed={}
    for age,g in data.groupby('anchor_break_age_sessions',dropna=False):
        computed[str(age)]=dict(n=len(g),dates=int(g.date.nunique()),known=int(g.target.notna().sum()),
                               unknown=int(g.target.isna().sum()),success=int(g.target.eq(1).sum()),known_rate=float(g.target.mean()))
    tables=json.loads((P1/'P1_CONDITION_TABLES.json').read_text('utf-8'))
    for age,cell in tables['by_anchor_break_age'].items():
        assert computed[age]['success']==cell['k'] and computed[age]['n']==cell['n']
    result=dict(at=datetime.now(timezone.utc).isoformat(),status='P1_REQUIRES_CORRECTION_NO_MODEL_UPGRADE_ESTABLISHED',
        new_fits=0,source_csv_hash_verified=True,original_p1_fits=original['fits'],
        source_hashes={p.name:digest(p) for p in P1.iterdir() if p.is_file()},
        missing_current_anchor_and_mid_price_counterexample=unknown_result,
        missing_competitor_amount_counterexample=missing_rank.to_dict('records'),
        original_cells_arithmetic_reproduced=True,age_counts_and_observed_dates=computed,
        original_pooled_same_day_diagnostic=tables['amount_rank_delta_discrimination_on_same_day_peers'],
        equal_date_board2_diagnostic=day_gap(data),equal_date_real_nonword_board2_diagnostic=day_gap(data,True),
        qualification_passed=False,new_training_freeze_approved=False,
        defects=['Missing source/price interpreted as observed anchor break and middle-tier failure',
                 'Missing competitors discarded from amount ranks instead of interval/unknown',
                 'Narrative contradicts actual age table; fixed causal claims not supported',
                 'Pooled winner/loser means across mixed dates are not within-date equal-weighted control',
                 'B absence only checked by blank theme columns; admitted name/narrative qualification not completed',
                 'Saved baseline in P1 script is copied constants; Codex independent replay remains separately valid'])
    (OUT/'INDEPENDENT_P1_AUDIT.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(clean({k:v for k,v in result.items() if k not in ['source_hashes','missing_competitor_amount_counterexample']}),ensure_ascii=False))

if __name__=='__main__':main()
