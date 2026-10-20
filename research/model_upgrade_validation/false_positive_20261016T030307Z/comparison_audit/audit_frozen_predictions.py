"""Read-only audit of frozen predictions; no fit, fetch, threshold search or writes to old artifacts."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OLD = REPO / 'research/model_upgrade_validation/ranking_optimization_20261015T173045Z'
FINAL = OLD / 'source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv'
ORIGINAL = OLD / 'source_repair/verified_22662_final/all_predictions_with_isolated_qualification.csv'
ATTR = OLD / 'source_feature_training/independent_verification/四层来源归因_完整指标.csv'
NEW_ARM = 'SOURCE_PATCHED_FEATURE_CLASSIFIER'
PERIODS = ['2025H2', '2026JanAug', '2026Sep', '2026Oct', 'HISTORICAL_POOLED', 'ALL_WITH_OCT_UNMATURED']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def period_mask(data, period):
    date = data.date
    if period == '2025H2': return date.between('2025-07-01', '2025-12-31')
    if period == '2026JanAug': return date.between('2026-01-01', '2026-08-31')
    if period == '2026Sep': return date.str.startswith('2026-09')
    if period == '2026Oct': return date.str.startswith('2026-10')
    if period == 'HISTORICAL_POOLED': return date.le('2026-09-30')
    return pd.Series(True, index=data.index)

def load(path):
    data = pd.read_csv(path, dtype={'code': str}, low_memory=False)
    data['known_audit'] = data.label_valid.astype(str).str.lower().eq('true') & pd.to_numeric(data.target, errors='coerce').isin([0, 1])
    assert not data.duplicated(['date','code','arm']).any()
    assert data.date.max() == '2026-10-15'
    return data

def metrics(group, call_column):
    known = group[group.known_audit]
    positive = known.target.eq(1)
    negative = known.target.eq(0)
    called_pos = known[call_column].eq('PROMOTE')
    called_neg = known[call_column].eq('FAIL')
    tp = int((called_pos & positive).sum()); fp = int((called_pos & negative).sum())
    fn = int((called_neg & positive).sum()); tn = int((called_neg & negative).sum())
    missing_pos = int((~(called_pos | called_neg) & positive).sum())
    missing_neg = int((~(called_pos | called_neg) & negative).sum())
    denom = lambda numerator, denominator: numerator/denominator if denominator else np.nan
    pair = group.arm.iloc[0] == 'PAIRWISE_DAY_RANK'
    return dict(candidates=len(group),known=len(known),unknown_outcomes=len(group)-len(known),positives=int(positive.sum()),negatives=int(negative.sum()),
        all_positive_calls=int(group[call_column].eq('PROMOTE').sum()),positive_calls_on_unknown_outcomes=int(group.loc[~group.known_audit,call_column].eq('PROMOTE').sum()),
        TP=tp if not pair else np.nan,FP=fp if not pair else np.nan,FN_called=fn if not pair else np.nan,TN=tn if not pair else np.nan,
        abstained_positive=missing_pos,abstained_negative=missing_neg,
        Recall=denom(tp,int(positive.sum())) if not pair else np.nan,Precision=denom(tp,tp+fp) if not pair else np.nan,
        F1=denom(2*tp,2*tp+fp+fn+missing_pos) if not pair else np.nan,
        false_positive_rate_among_actual_negatives=denom(fp,int(negative.sum())) if not pair else np.nan,
        error_share_among_known_positive_calls=denom(fp,tp+fp) if not pair else np.nan,
        prediction_coverage_known=denom(int((called_pos|called_neg).sum()),len(known)) if not pair else np.nan,
        classification_applicability='NO_ABSOLUTE_EVENT_HEAD' if pair else 'ABSOLUTE_EVENT_HEAD')

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    output = HERE/'classification_metrics_full_nonword.csv'
    if output.exists(): raise FileExistsError('immutable outputs already present')
    final = load(FINAL); old = load(ORIGINAL)
    assert len(final)==26439 and len(old)==22662
    original_columns=[c for c in old.columns if c != 'known_audit']
    pd.testing.assert_frame_equal(old[original_columns], final.iloc[:len(old)][original_columns],check_dtype=False,check_exact=True)
    b = final[final.frozen_original_nonword_identity.eq(0)].copy()
    old_b = old[old.frozen_original_nonword_identity.eq(0)].copy()
    rows=[]
    for period in PERIODS:
        for stage,data,column in [('ORIGINAL_CALL_BEFORE_SOURCE_RECOVERY',old_b,'call'),('FINAL_SAME_SOURCE_QUALIFICATION',b,'aligned_evaluation_call')]:
            for arm,group in data[period_mask(data,period)].groupby('arm',sort=True):
                rows.append(dict(period=period,stage=stage,arm=arm,**metrics(group,column)))
    stats=pd.DataFrame(rows)
    stats.to_csv(output,index=False,encoding='utf-8-sig')
    key=['date','code','name','boards','target_date','target','known_audit','frozen_original_nonword_identity']
    wide=b[b.arm.eq('BASE')][key].copy().set_index(['date','code'])
    for arm,group in b.groupby('arm',sort=True):
        g=group.set_index(['date','code'])
        for col in ['call','aligned_evaluation_call','b_score','ranking_score','current_price_shape','verified_current_price_shape']:
            wide[f'{arm}__{col}']=g[col]
    wide.reset_index().to_csv(HERE/'all_original_nonword_rows_all_calls_and_scores.csv',index=False,encoding='utf-8-sig')
    overlap=[]; changed=[]
    arms=[a for a in sorted(b.arm.unique()) if a!='PAIRWISE_DAY_RANK']
    for period in PERIODS:
        pool=b[period_mask(b,period)]
        for left,right in combinations(arms,2):
            gl=pool[pool.arm.eq(left)].set_index(['date','code']); gr=pool[pool.arm.eq(right)].set_index(['date','code'])
            assert gl.index.equals(gr.index)
            for truth,event in [(1,'TP'),(0,'FP')]:
                ls=set(gl.index[gl.known_audit & gl.target.eq(truth) & gl.aligned_evaluation_call.eq('PROMOTE')])
                rs=set(gr.index[gr.known_audit & gr.target.eq(truth) & gr.aligned_evaluation_call.eq('PROMOTE')])
                overlap.append(dict(period=period,left=left,right=right,event=event,left_count=len(ls),right_count=len(rs),both=len(ls&rs),left_only=len(ls-rs),right_only=len(rs-ls),union=len(ls|rs)))
            changed_pool=gl.aligned_evaluation_call.ne(gr.aligned_evaluation_call)
            for ident in gl.index[changed_pool]:
                record=gl.loc[ident]
                changed.append(dict(period=period,left=left,right=right,date=ident[0],code=ident[1],name=record['name'],boards=record.boards,target=record.target,known=bool(record.known_audit),left_call=record.aligned_evaluation_call,right_call=gr.loc[ident,'aligned_evaluation_call']))
    pd.DataFrame(overlap).to_csv(HERE/'TP_FP_pairwise_overlap_all_periods.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(changed).to_csv(HERE/'all_disagreements_all_periods.csv',index=False,encoding='utf-8-sig')
    transitions=[]
    for period in PERIODS:
        corrected=b[b.arm.eq('CORRECTED_CAUSAL') & period_mask(b,period)].set_index(['date','code'])
        trained=b[b.arm.eq(NEW_ARM) & period_mask(b,period)].set_index(['date','code'])
        for stage,left,right in [('SOURCE_QUALIFICATION_RESTORED_WITH_IDENTICAL_RAW_MODEL',corrected.call,corrected.aligned_evaluation_call),('NEW_TRAINING_AND_PATCHED_INFERENCE_SAME_QUALIFICATION',corrected.aligned_evaluation_call,trained.aligned_evaluation_call)]:
            for ident in corrected.index[left.ne(right)]:
                r=corrected.loc[ident]
                transitions.append(dict(period=period,stage=stage,date=ident[0],code=ident[1],name=r['name'],boards=r.boards,target=r.target,known=bool(r.known_audit),before_call=left.loc[ident],after_call=right.loc[ident]))
    tr=pd.DataFrame(transitions)
    tr.to_csv(HERE/'source_recovery_vs_training_all_changed_rows.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for (period,stage),group in tr.groupby(['period','stage'],sort=False):
        summaries.append(dict(period=period,stage=stage,changed_rows=len(group),unknown_changed=int((~group.known).sum()),new_TP=int((group.known&group.target.eq(1)&group.after_call.eq('PROMOTE')).sum()),removed_TP=int((group.known&group.target.eq(1)&group.before_call.eq('PROMOTE')).sum()),new_FP=int((group.known&group.target.eq(0)&group.after_call.eq('PROMOTE')).sum()),removed_FP=int((group.known&group.target.eq(0)&group.before_call.eq('PROMOTE')).sum())))
    pd.DataFrame(summaries).to_csv(HERE/'source_recovery_vs_training_change_totals.csv',index=False,encoding='utf-8-sig')
    # Multiple winners and multiple losers are retained rather than forcing a single winner.
    same=[]
    for (arm,date,boards),group in b.groupby(['arm','date','boards'],sort=True):
        known=group[group.known_audit]
        same.append(dict(arm=arm,date=date,boards=boards,candidates=len(group),known=len(known),unknown=len(group)-len(known),true_promotions=int(known.target.eq(1).sum()),positive_calls=int(group.aligned_evaluation_call.eq('PROMOTE').sum()),positive_calls_known=int(known.aligned_evaluation_call.eq('PROMOTE').sum()),TP=int((known.target.eq(1)&known.aligned_evaluation_call.eq('PROMOTE')).sum()) if arm!='PAIRWISE_DAY_RANK' else np.nan,FP=int((known.target.eq(0)&known.aligned_evaluation_call.eq('PROMOTE')).sum()) if arm!='PAIRWISE_DAY_RANK' else np.nan))
    pd.DataFrame(same).to_csv(HERE/'complete_same_day_same_height_groups.csv',index=False,encoding='utf-8-sig')
    sep=stats[(stats.period=='2026Sep') & (stats.stage=='FINAL_SAME_SOURCE_QUALIFICATION') & stats.arm.eq(NEW_ARM)].iloc[0]
    assert (sep.candidates,sep.known,sep.unknown_outcomes,sep.positives,sep.negatives,sep.TP,sep.FP)==(143,107,36,40,67,22,24)
    # Independently compare the published same-pool result, without calling its metric implementation.
    expected=pd.read_csv(OLD/'source_feature_training/same_source_route_alignment/same_pool_classification_metrics.csv',low_memory=False)
    expected=expected[(expected.scope=='ORIGINAL_NONWORD')&expected.period.isin(PERIODS)].copy()
    ours=stats[stats.stage=='FINAL_SAME_SOURCE_QUALIFICATION']
    for row in expected.itertuples(index=False):
        audit=ours[(ours.period==row.period)&ours.arm.eq(row.arm)].iloc[0]
        if row.arm=='PAIRWISE_DAY_RANK':continue
        for x,y in [('TP','tp'),('FP','fp'),('Recall','Recall'),('Precision','Precision'),('F1','F1')]:
            assert np.isclose(audit[x],getattr(row,y),rtol=0,atol=1e-14,equal_nan=True),(row.period,row.arm,x)
    receipt=dict(scope='READ_ONLY_FROZEN_PREDICTIONS; NO_FIT_NO_FETCH_NO_THRESHOLD_SEARCH',original_rows=len(old),final_rows=len(final),original_six_arm_cells_preserved=True,
        target_data_max_signal_date=final.date.max(),no_read_of_oct16_market_results=True,
        source_bindings={str(p):digest(p) for p in [FINAL,ORIGINAL,ATTR,Path(__file__)]},
        september_latest_nonword=dict(total=143,known=107,unknown_outcomes=36,positives=40,negatives=67,TP=22,FP=24,known_positive_signals=46,total_positive_signals_including_unknown=int(sep.all_positive_calls),positive_signals_with_unknown_outcome=int(sep.positive_calls_on_unknown_outcomes),false_positive_rate=24/67,wrong_share_of_mature_positive_signals=24/46,Recall=22/40,Precision=22/46,F1=44/86),
        pairwise_classification_metrics='NOT_APPLICABLE: preference head has no absolute event decision',outputs={str(p):digest(p) for p in HERE.glob('*.csv')})
    (HERE/'independent_comparison_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt['september_latest_nonword'],ensure_ascii=False))
    print(stats[(stats.period=='2026Sep') & stats.stage.eq('FINAL_SAME_SOURCE_QUALIFICATION')][['arm','TP','FP','Recall','Precision','F1']].to_string(index=False))

if __name__=='__main__': main()
