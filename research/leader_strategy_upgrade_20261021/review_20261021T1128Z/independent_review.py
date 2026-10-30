"""Independent descriptive audit. No model fit, HTTP, trading or source rewrite."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
UP = ROOT / 'research/leader_strategy_upgrade_20261021'
RELAY = ROOT / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/dataset.joblib'
S1 = ROOT / 'research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv'
Y = 'promotion_next_LABEL_ONLY'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.generic): return value.item()
    return value

def summary(frame, label):
    v = pd.to_numeric(frame[label], errors='coerce')
    assert v.dropna().isin([0, 1]).all()
    n, known, k = len(v), int(v.notna().sum()), int(v.eq(1).sum())
    return dict(n=n, dates=int(frame.iloc[:, 0].nunique()), known=known,
                unknown=n-known, successes=k, known_rate=k/known if known else None,
                all_rows_success_lower_bound=k/n if n else None,
                all_rows_success_upper_bound=(k+n-known)/n if n else None)

def contrast(frame, condition, label, date, reps=4000):
    """Descriptive five-observed-session moving-block bootstrap, no model selection."""
    frame=frame.copy()
    frame['_g']=np.asarray(condition, dtype=bool)
    v=pd.to_numeric(frame[label], errors='coerce')
    frame['_y']=v.fillna(0)
    frame['_n']=v.notna().astype(int)
    dates=sorted(frame[date].unique())
    a=[]
    for flag in [True,False]:
        q=frame[frame['_g'].eq(flag)].groupby(date)[['_y','_n']].sum().reindex(dates, fill_value=0)
        a.append(q.to_numpy(float))
    data=np.column_stack(a)
    totals=data.sum(axis=0)
    observed=totals[0]/totals[1]-totals[2]/totals[3]
    rng=np.random.default_rng(20261021)
    length=len(dates); block=5; bs=[]
    for _ in range(reps):
        starts=rng.integers(0, length-block+1, size=(length+block-1)//block)
        ix=(starts[:, None]+np.arange(block)).ravel()[:length]
        t=data[ix].sum(axis=0)
        if t[1] and t[3]: bs.append(t[0]/t[1]-t[2]/t[3])
    low, high=np.quantile(bs,[.025,.975])
    return dict(group=summary(frame[frame['_g']],label), complement=summary(frame[~frame['_g']],label),
                known_rate_difference=observed, moving_block5_descriptive_ci95=[float(low),float(high)],
                bootstrap_repetitions=reps, seed=20261021,
                inferential_status='EXPLORATORY_LABELS_ALREADY_SEEN_NOT_CAUSAL_NOT_A_NEW_BLIND_TEST')

def main():
    OUT.mkdir(exist_ok=True)
    originals=[UP/'line_a_cohort_feedback_study.py',UP/'line_b_seal_quality_audit.py',
        UP/'line_a_cohort_feedback/线A_梯队与断板条件研究报告.md',
        UP/'line_a_cohort_feedback/LINE_A_CONDITION_STUDY.json',
        UP/'line_b_seal_quality/线B_封板质量与换手分桶资格审计报告.md',
        UP/'line_b_seal_quality/LINE_B_AUDIT_RECEIPT.json',
        UP/'model_comparison/report.json',UP/'model_comparison/JOINT_FROZEN_PROTOCOL.json',
        ROOT/'研究计划_梯队封板情绪_20261021.md',
        ROOT/'research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json']
    sources={}
    for p in originals:
        rel=p.relative_to(ROOT)
        dest=OUT/'originals'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): shutil.copy2(p,dest)
        sources[str(rel)]=sha(p)
    data=joblib.load(RELAY); f=data['features']; r=data['prepared']['relay']
    positional={key:int(f[key].astype(str).ne(r[key].astype(str)).sum()) for key in ['date','code','boards']}
    assert set(zip(f.date,f.code))==set(zip(r.date,r.code))
    assert not f.duplicated(['date','code']).any()
    # Validated one-to-one key join instead of relying on position.
    d=f.merge(r[['date','code','promoted_next_day','net_return','large_loss','next_open_gap',
                 'execution_status','execution_source','execution_assumption','large_loss_label_status',
                 'mark_to_market_return','target','trade_date','exit_date','entry_price','t1_close']],
              on=['date','code'], validate='one_to_one',sort=False)
    s=pd.read_csv(S1,dtype={'code':str},float_precision='round_trip')
    assert not s.duplicated(['signal_date','code']).any()
    info=dict(at=datetime.now(timezone.utc).isoformat(),models_fit=0,network_requests=0,
              sources=sources|{str(RELAY.relative_to(ROOT)):sha(RELAY),str(S1.relative_to(ROOT)):sha(S1)},
              relay_identity_alignment_verified=not any(positional.values()),
              original_positional_mismatches=positional,key_join_verified=True,
              relay_scope=dict(start=str(d.date.min()),end=str(d.date.max()),dates=int(d.date.nunique()),
                               board_counts=d.boards.value_counts().sort_index().to_dict()),
              first_scope=dict(start=str(s.signal_date.min()),end=str(s.signal_date.max()),dates=int(s.signal_date.nunique()),
                               model_months=sorted(s.model_month.unique().tolist())))
    info['relay_overall']=summary(d,'promoted_next_day')
    info['relay_official_target_overall']=summary(d,'target')
    info['s1_overall']=summary(s,Y)
    info['relay_target_vs_promoted']=dict(unknown_target=int(d.target.isna().sum()),unknown_promoted=int(d.promoted_next_day.isna().sum()),
                                      disagreements_when_both_known=int((d.target.notna()&d.promoted_next_day.notna()&d.target.ne(d.promoted_next_day)).sum()))
    mismatch=f.code.astype(str).ne(r.code.astype(str))
    info['misalignment_impact']=dict(misaligned_raw_labels_known=int(r.loc[mismatch,'promoted_next_day'].notna().sum()),
                                   misaligned_official_labels_known=int(r.loc[mismatch,'target'].notna().sum()),
                                   explanation='Known raw promotion labels are not proven swapped: all 173 misaligned raw labels are unknown. Official target must be joined on date/code.')
    info['corrected_official_target_tables']={
        'parent_break':{str(k):summary(v,'target') for k,v in d.groupby(d.parent_break_count.ge(1))},
        'parent_break_by_regime':{str(reg):{str(k):summary(v,'target') for k,v in g.groupby(g.parent_break_count.ge(1))} for reg,g in d.groupby('cycle_regime')},
        'handover':{str(k):summary(v,'target') for k,v in d.groupby('handover_on_anchor_break')},
        'roles':{str(k):summary(v,'target') for k,v in d.groupby('primary_role')},
        'board2_parent_break':contrast(d[d.boards.eq(2)],d.loc[d.boards.eq(2),'parent_break_count'].ge(1),'target','date')}
    info['relay_official_target_by_month']={str(k):dict(raw=summary(g,'promoted_next_day'),official=summary(g,'target')) for k,g in d.groupby(d.date.str.slice(0,7))}
    info['relay_parent_break']={str(k):summary(v,'promoted_next_day') for k,v in d.groupby(d.parent_break_count.ge(1))}
    info['relay_parent_break_by_regime']={str(reg):{str(k):summary(v,'promoted_next_day') for k,v in g.groupby(g.parent_break_count.ge(1))}
                                        for reg,g in d.groupby('cycle_regime')}
    info['relay_board2_parent_break']=contrast(d[d.boards.eq(2)],d.loc[d.boards.eq(2),'parent_break_count'].ge(1),
                                               'promoted_next_day','date')
    info['s1_height_all_values']={str(k):summary(v,Y) for k,v in s.groupby('cycle_height_change',dropna=False)}
    info['s1_height_le_minus2']=contrast(s,s.cycle_height_change.le(-2),Y,'signal_date')
    info['s1_height_eq_minus3']=contrast(s,s.cycle_height_change.eq(-3),Y,'signal_date')
    info['s1_anchor_break']=contrast(s,s.cycle_anchor_breaks.ge(1),Y,'signal_date')
    nop=s[s.one_price.eq(0)].copy()
    nop['turnover_bin']=pd.cut(nop.turnover,[-1,3,5,10,15,20,100],right=True)
    info['s1_non_one_price']=summary(nop,Y)
    info['s1_turnover_bins']={str(k):summary(v,Y) for k,v in nop.groupby('turnover_bin',observed=True,dropna=False)}
    info['s1_non_one_price_high_turnover']=contrast(nop,nop.turnover.gt(15),Y,'signal_date')
    # Same observed date, same within-date float-cap quartile: descriptive
    # comparison only, not causal identification or a model fit.
    nop['cap_quartile']=np.ceil(nop.float_cap.groupby(nop.signal_date).rank(pct=True)*4)
    diffs=[]
    for (dt,cap),g in nop[nop[Y].notna()&nop.turnover.notna()].groupby(['signal_date','cap_quartile']):
        a=g[g.turnover.gt(15)][Y]; b=g[g.turnover.le(15)][Y]
        if len(a) and len(b):
            diffs.append(dict(signal_date=dt,cap_quartile=int(cap),high_n=len(a),other_n=len(b),
                              known_rate_diff=float(a.mean()-b.mean())))
    matched=pd.DataFrame(diffs)
    info['high_turnover_same_date_capquartile_diagnostic']=dict(cells=len(matched),dates=int(matched.signal_date.nunique()),
        cell_equal_weighted_diff=float(matched.known_rate_diff.mean()),
        date_equal_weighted_diff=float(matched.groupby('signal_date').known_rate_diff.mean().mean()),
        scope='Only common-support cells, descriptive; unmatched cells/unknown outcomes retained in full source summaries.')
    info['height_condition_within_date_variation']=int(s.groupby('signal_date').cycle_height_change.nunique().gt(1).sum())
    info['s1_unknown_feature_rows']={k:int(s[k].isna().sum()) for k in ['cycle_anchor_breaks','cycle_height_change','turnover','one_price']}
    info['s1_bin_edges_exact_counts']={str(k):int(nop.turnover.eq(k).sum()) for k in [3,5,10,15,20,100]}
    info['s1_missing_bin_rows']=nop[nop.turnover_bin.isna()][['signal_date','code','turnover',Y]].to_dict('records')
    info['verified_turnover_2_height_counts']=d[d.verified_turnover_boards.eq(2)].boards.value_counts().sort_index().to_dict()
    info['partial_observation']=dict(rows=int(d.streak_observed_fraction.lt(1).sum()),
        nonnull_verified_with_partial=int((d.streak_observed_fraction.lt(1)&d.verified_turnover_boards.notna()).sum()))
    info['role_status_counts']=d.role_status.value_counts().to_dict()
    info['relay_execution_sources']=d.execution_source.value_counts(dropna=False).to_dict()
    info['relay_execution_status']=d.execution_status.value_counts(dropna=False).to_dict()
    info['return_label_sources']=d.large_loss_label_status.value_counts(dropna=False).to_dict()
    info['relay_one_word_risk']={}
    for k,g in d.groupby('daily_one_word',dropna=False):
        fails=g[g.promoted_next_day.eq(0)]
        info['relay_one_word_risk'][str(k)]=dict(promotion=summary(g,'promoted_next_day'),
            all_loss_known=int(g.large_loss.notna().sum()),all_loss_mean=float(g.large_loss.mean()),
            failure_count=len(fails),failure_loss_known=int(fails.large_loss.notna().sum()),
            failure_loss_mean=float(fails.large_loss.mean()),net_return_known=int(g.net_return.notna().sum()),
            proxy_net_return_mean=float(g.net_return.mean()),proxy_entry_nonnull=int(g.entry_price.notna().sum()))
    # Replay the report's actual selection of unpromoted examples with price labels attached.
    cases=d[d.date.isin(['2024-10-09','2024-10-08','2026-02-04','2026-01-29'])]
    cases=pd.concat([g[g.promoted_next_day.eq(0)].head(5) for _,g in cases.groupby('date')])
    cases[['date','code','name','boards','promoted_next_day','next_open_gap','large_loss',
           'execution_status','net_return','trade_date','exit_date']].to_csv(OUT/'COUNTEREXAMPLE_PRICE_LABELS.csv',index=False,encoding='utf-8-sig')
    info['report_counterexample_price_checks']=cases[['date','code','next_open_gap','large_loss','net_return','execution_status']].to_dict('records')
    info['s1_original_eligible_counts']=s.eligible.value_counts(dropna=False).to_dict()
    info['completed_lifecycle_replay']={}
    for arm in ['THEME_LIFECYCLE_75','THEME_LIFECYCLE_STATUS_79']:
        p=UP/'model_comparison'/arm
        csv=p/'VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv'
        v=pd.read_csv(csv,dtype={'code':str},float_precision='round_trip')
        bundle_file=p/'models'/f'validation_{arm}_bundle.joblib'
        bundle=joblib.load(bundle_file)
        scores=bundle['model'].predict_proba(v[bundle['features']])[:,1]
        np.testing.assert_allclose(scores,v.model_score,rtol=0,atol=0)
        report=json.loads((p/'report.json').read_text('utf-8'))
        arm_result=dict(rows=len(v),dates=int(v.signal_date.nunique()),source_unknown=int(v[Y].isna().sum()),
                        csv_sha256=sha(csv),bundle_sha256=sha(bundle_file),saved_scores_exactly_reproduced=True,new_fits=0)
        for name,pool in [('primary',v),('non_one_price',v[v.one_price.eq(0)])]:
            allowed=pool[pool.research_selection_allowed]
            take=allowed.sort_values(['signal_date','model_score','code'],ascending=[True,False,True]).groupby('signal_date',sort=False).head(1)
            hits=int(take[Y].eq(1).sum())
            assert hits==report[name]['hits']
            arm_result[name]=dict(selected=len(take),known=int(take[Y].notna().sum()),unknown=int(take[Y].isna().sum()),hits=hits,
                                  successes_per_complete_dates=hits/v.signal_date.nunique())
        info['completed_lifecycle_replay'][arm]=arm_result
    (OUT/'INDEPENDENT_RECOMPUTATION.json').write_text(json.dumps(clean(info),ensure_ascii=False,indent=2,default=str,allow_nan=False),'utf-8')
    print(json.dumps(clean({k:v for k,v in info.items() if k not in ['sources','report_counterexample_price_checks','s1_height_all_values','relay_parent_break_by_regime','relay_official_target_by_month','corrected_official_target_tables']}),ensure_ascii=False,default=str,allow_nan=False))

if __name__=='__main__': main()
