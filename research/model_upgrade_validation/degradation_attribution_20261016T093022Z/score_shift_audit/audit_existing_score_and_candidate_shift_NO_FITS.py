"""Descriptive existing-artifact shift audit. No fit or proposed outer scorer."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))

PRIOR=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z'
ARMS=['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']
INDIVIDUAL=['boards','height_gap','same_height_rivals','turnover','amplitude','log_signal_amount',
    'price_same_height_percentile','signal_amount_same_height_percentile','turnover_same_height_percentile']
FIELDS=['base_score_logit']+INDIVIDUAL
KEYS=['date','code']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)
def observed(f):return f.label_valid.fillna(False).astype(bool)&f.target.isin([0,1])
def positive(f):return f.verified_current_price_shape.eq(0)&f.raw_score.gt(.5)
def finite_summary(value):
    s=pd.to_numeric(value,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    if s.empty:return dict(observed=0,mean=None,std=None,q05=None,q25=None,q50=None,q75=None,q95=None)
    return dict(observed=len(s),mean=float(s.mean()),std=float(s.std(ddof=0)),
        **{f'q{int(q*100):02d}':float(s.quantile(q)) for q in [.05,.25,.5,.75,.95]})


def run():
    protocol_path=PRIOR/'protocol_BEFORE_ANY_FIT.json';protocol=read(protocol_path)
    input_path=PRIOR/'fixed_input_BEFORE_ANY_FIT.joblib';data=joblib.load(input_path)
    reference_candidates=[Path(p) for p in protocol['source_bindings'] if Path(p).name=='all41547_predictions_BEFORE_EVALUATION.csv']
    assert len(reference_candidates)==1
    reference_path=reference_candidates[0]
    references=csv(reference_path)
    baseline=references[references.arm.eq(protocol['baseline'])].copy()
    full=data['full'].set_index(KEYS)
    bindings={str(p.resolve()):sha(p) for p in [protocol_path,input_path,reference_path,Path(__file__)]}
    primary_records=[];meta_records=[];same_third=[];outer_feature=[];oof_long=[]
    for plan in data['plans']:
        month=plan['month']
        oof_path=PRIOR/f'execution/{month}_complete_B_OOF_BEFORE_SELECTION.csv'
        prefix_path=PRIOR/f'execution/{month}_meta_prefix_training.csv'
        final_path=PRIOR/f'execution/{month}_meta_final_training.csv'
        selection_path=PRIOR/f'execution/{month}_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json'
        oof,prefix,final=map(csv,[oof_path,prefix_path,final_path]);selection=read(selection_path)
        boundary_path=PRIOR/'execution/all48_first_stage_cache_replays_and_completion.csv'
        boundary=csv(boundary_path);lengths={1:62,2:125,3:188}
        for block,part in oof.groupby('inner_block'):
            block=int(block)
            oof_long.append(part[KEYS+['target','label_valid','raw_score','verified_current_price_shape']].assign(outer_month=month,inner_block=block,training_days=lengths[block]))
        outer=baseline[baseline.date.str[:7].eq(month)&baseline.daily_one_word.eq(0)].copy()
        outer['raw_score']=outer.b_score
        outer['base_score_logit']=np.log(np.clip(outer.raw_score,1e-12,1-1e-12))-np.log1p(-np.clip(outer.raw_score,1e-12,1-1e-12))
        outer_index=pd.MultiIndex.from_frame(outer[KEYS])
        for c in INDIVIDUAL:outer[c]=full.loc[outer_index,c].to_numpy()
        stages=[(f'OOF_BLOCK{int(k)}',lengths[int(k)],f) for k,f in oof.groupby('inner_block')]+[('OUTER_EXISTING_SOURCE',252,outer)]
        for name,length,part in stages:
            for scope,sub in [('ALL_ORIGINAL_B',part),('QUALIFIED_B',part[part.verified_current_price_shape.eq(0)]),('PRIMARY_POSITIVE',part[positive(part)])]:
                k=observed(sub);n=int(k.sum());p=int((k&sub.target.eq(1)).sum())
                primary_records.append(dict(month=month,stage=name,training_days=length,scope=scope,rows=len(sub),
                    known=n,unknown=len(sub)-n,positives=p,positive_rate=p/n if n else None,
                    primary_positive_count=int(positive(sub).sum()),primary_positive_fraction=float(positive(sub).mean()) if len(sub) else None,
                    **finite_summary(sub.raw_score)))
            for truth,label in [(1,'KNOWN_TRUE_POSITIVE'),(0,'KNOWN_FALSE_POSITIVE')]:
                sub=part[positive(part)&observed(part)&part.target.eq(truth)]
                primary_records.append(dict(month=month,stage=name,training_days=length,scope=label,rows=len(sub),
                    known=len(sub),unknown=0,positives=len(sub) if truth else 0,positive_rate=float(truth),
                    primary_positive_count=len(sub),primary_positive_fraction=1. if len(sub) else None,**finite_summary(sub.raw_score)))
        for stage,part in [('PREFIX_TWO_BLOCKS',prefix),('FINAL_THREE_BLOCKS',final)]:
            for c in FIELDS:
                meta_records.append(dict(month=month,stage=stage,field=c,rows=len(part),
                    positives=int(part.target.eq(1).sum()),false_positives=int(part.target.eq(0).sum()),
                    true_fraction=float(part.target.mean()),missing=int(part[c].isna().sum()),**finite_summary(part[c])))
        oof_positive=oof[positive(oof)]
        outer_positive=outer[positive(outer)]
        for c in FIELDS:
            a,b=finite_summary(oof_positive[c]),finite_summary(outer_positive[c])
            outer_feature.append(dict(month=month,field=c,OOF_positive_rows=len(oof_positive),outer_positive_rows=len(outer_positive),
                OOF_mean=a['mean'],outer_mean=b['mean'],OOF_q50=a['q50'],outer_q50=b['q50'],OOF_std=a['std'],
                mean_shift_in_OOF_std=(b['mean']-a['mean'])/a['std'] if a['std'] and b['mean'] is not None else None,
                OOF_missing_fraction=float(oof_positive[c].isna().mean()) if len(oof_positive) else None,
                outer_missing_fraction=float(outer_positive[c].isna().mean()) if len(outer_positive) else None))
        third=oof[oof.inner_block.eq(3)&positive(oof)].copy()
        for arm in ARMS:
            paths=[PRIOR/f'execution/{month}_{arm}_{phase}_model.joblib' for phase in ['PREFIX','FINAL']]
            before,after=map(joblib.load,paths)
            columns=before['features'];assert columns==after['features']
            with threadpool_limits(limits=2):
                pre=before['model'].predict_proba(third[columns])[:,1]
                fin=after['model'].predict_proba(third[columns])[:,1]
            threshold=selection['choices'][arm]['selected_threshold']
            pre_keep=pre>threshold if threshold else np.ones(len(pre),bool)
            fin_keep=fin>threshold if threshold else np.ones(len(fin),bool)
            labeled=observed(third).to_numpy();y=third.target.eq(1).to_numpy()
            same_third.append(dict(month=month,arm=arm,selected_threshold=threshold,eligible_third_rows=len(third),
                known_third=int(labeled.sum()),unknown_third=int((~labeled).sum()),
                prefix_score_mean=float(pre.mean()) if len(pre) else None,final_score_mean=float(fin.mean()) if len(fin) else None,
                prefix_q50=float(np.median(pre)) if len(pre) else None,final_q50=float(np.median(fin)) if len(fin) else None,
                mean_same_input_score_change=float((fin-pre).mean()) if len(pre) else None,
                max_same_input_absolute_change=float(np.abs(fin-pre).max()) if len(pre) else None,
                retained_to_rejected=int((pre_keep&~fin_keep).sum()),rejected_to_retained=int((~pre_keep&fin_keep).sum()),
                retained_to_rejected_TP=int((pre_keep&~fin_keep&labeled&y).sum()),
                retained_to_rejected_FP=int((pre_keep&~fin_keep&labeled&~y).sum()),
                rejected_to_retained_TP=int((~pre_keep&fin_keep&labeled&y).sum()),
                rejected_to_retained_FP=int((~pre_keep&fin_keep&labeled&~y).sum()),
                coefficient_L2_change=float(np.linalg.norm(after['model'].named_steps['logistic'].coef_-before['model'].named_steps['logistic'].coef_)),
                prefix_intercept=float(before['model'].named_steps['logistic'].intercept_[0]),final_intercept=float(after['model'].named_steps['logistic'].intercept_[0]),
                final_third_is_in_sample=True,not_an_unbiased_effect_comparison=True,
                proposed_NO_META_REFIT_outer_predictions_generated=False))
            for p in paths:bindings[str(p.resolve())]=sha(p)
        for p in [oof_path,prefix_path,final_path,selection_path,boundary_path]:bindings[str(p.resolve())]=sha(p)
    primary=pd.DataFrame(primary_records);meta=pd.DataFrame(meta_records);same=pd.DataFrame(same_third);outer_drift=pd.DataFrame(outer_feature)
    long=pd.concat(oof_long,ignore_index=True)
    long['positive_call']=positive(long)
    grouped=long.groupby(KEYS+['training_days']).agg(raw_mean=('raw_score','mean'),positive_call_fraction=('positive_call','mean'),occurrences=('raw_score','size')).reset_index()
    pairs=[]
    for short,length in [(62,125),(62,188),(125,188)]:
        one=grouped[grouped.training_days.eq(short)].merge(grouped[grouped.training_days.eq(length)],on=KEYS,suffixes=('_short','_long'))
        if len(one):
            delta=one.raw_mean_long-one.raw_mean_short
            pairs.append(dict(short_days=short,long_days=length,same_signal_code_pairs=len(one),
                average_raw_score_change=float(delta.mean()),median_raw_score_change=float(delta.median()),
                score_change_q05=float(delta.quantile(.05)),score_change_q95=float(delta.quantile(.95)),
                average_primary_positive_fraction_change=float((one.positive_call_fraction_long-one.positive_call_fraction_short).mean()),
                longer_score_higher=int(delta.gt(0).sum()),longer_score_lower=int(delta.lt(0).sum()),
                independent_pairs=False,training_cutoff_and_members_also_differ=True))
    primary.to_csv(OUT/'all80_primary_stage_and_label_conditional_score_distributions.csv',index=False,encoding='utf-8-sig')
    meta.to_csv(OUT/'all16_prefix_to_final_meta_training_feature_distributions.csv',index=False,encoding='utf-8-sig')
    same.to_csv(OUT/'all32_PREFIX_FINAL_same_third_input_scale_changes_IN_SAMPLE_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
    outer_drift.to_csv(OUT/'all16_OOF_to_existing_outer_positive_feature_shift.csv',index=False,encoding='utf-8-sig')
    grouped.to_csv(OUT/'all_repeated_same_member_by_training_length.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(OUT/'same_member_length_paired_summary_NOT_CAUSAL.csv',index=False,encoding='utf-8-sig')
    active=same[same.selected_threshold.gt(0)]
    summary=dict(new_fits=0,network_calls=0,old_files_modified=0,official_target16_new_results_read=False,
        proposed_NO_META_REFIT_outer_predictions_generated_or_evaluated=False,
        monthly_meta_choices=32,active_threshold_choices=len(active),
        active_thresholds=active[['month','arm','selected_threshold']].to_dict(orient='records'),
        original_OOF_rows_with_monthly_repetition=len(long),unique_OOF_signal_codes=len(long[KEYS].drop_duplicates()),
        same_input_scale_change_max=float(same.max_same_input_absolute_change.max()),
        prefix_final_input_drift_and_coefficient_change_descriptive_only=True,
        primary_training_length_not_causally_isolated=True,
        primary_length_pair_summary=pairs,
        source_bindings=bindings,
        output_bindings={str(p.resolve()):sha(p) for p in OUT.iterdir() if p.suffix=='.csv'})
    (OUT/'existing_score_shift_audit_NO_FITS_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_bindings','output_bindings']},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
