"""Read-only complete historical error decomposition; no fit or source fetch."""
from pathlib import Path
import hashlib, json, sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import model_upgrade_history as history
OUT=Path(__file__).resolve().parent
OLD=ROOT/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z'
PRED=OLD/'source_feature_training/same_source_route_alignment/complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv'
DATA=OLD/'source_feature_training/prepared.joblib'
CONTEXT=ROOT/'research/model_upgrade_validation/causal_scope_alignment_20261015T121000Z/context/review_scope_context.joblib'
ARM='SOURCE_PATCHED_FEATURE_CLASSIFIER'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    def default(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=default),encoding='utf-8')
def csv(name,frame):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def counts(f):
    known=f.label_valid.eq(True)&f.target.notna()
    return dict(candidates=len(f),known=int(known.sum()),unknown_outcomes=int((~known).sum()),
                actual_positive=int((known&f.target.eq(1)).sum()),
                TP=int((known&f.evaluation_call.eq('PROMOTE')&f.target.eq(1)).sum()),
                FP=int((known&f.evaluation_call.eq('PROMOTE')&f.target.eq(0)).sum()),
                FN=int((known&f.evaluation_call.eq('FAIL')&f.target.eq(1)).sum()),
                TN=int((known&f.evaluation_call.eq('FAIL')&f.target.eq(0)).sum()),
                known_abstention=int((known&f.evaluation_call.eq('UNKNOWN')).sum()),
                unknown_promote=int((~known&f.evaluation_call.eq('PROMOTE')).sum()))

def main():
    # Deliberately exclude every October row before accessing any outcome value.
    raw=pd.read_csv(PRED,dtype={'code':str},low_memory=False)
    raw=raw[raw.date.le('2026-09-30')].copy()
    data=joblib.load(DATA);fields=data['fields']
    full=data['full'][data['full'].date.le('2026-09-30')].copy()
    context=joblib.load(CONTEXT)
    sem=context['rows'];sem=sem[sem.date.le('2026-09-30')].copy()
    parents=context['parents'];parents=parents[parents.date.le('2026-09-30')].copy()
    # Preserve original frozen B identity from prediction table rather than latest engine.
    head=raw[raw.arm.eq(ARM)].copy()
    feat=full[['date','code']+fields].copy()
    feat=feat.rename(columns={x:'feature_'+x for x in fields})
    head=head.merge(feat,on=['date','code'],validate='one_to_one')
    semcols=['review_primary_role','review_roles','independent_role_identity_truth','theme_core_status','active_anchor_discovery_scope','full_cohort_status','full_market_pit_verified','theme_first_support','theme_relay_support']
    head=head.merge(sem[['date','code']+semcols],on=['date','code'],validate='one_to_one')
    head['original_frozen_B']=head.daily_one_word.eq(0)
    head['outcome_known']=head.label_valid.eq(True)&head.target.notna()
    head['disposition']=np.select([
        ~head.outcome_known,
        head.evaluation_call.eq('UNKNOWN'),
        head.evaluation_call.eq('PROMOTE')&head.target.eq(1),
        head.evaluation_call.eq('PROMOTE')&head.target.eq(0),
        head.evaluation_call.eq('FAIL')&head.target.eq(1),
        head.evaluation_call.eq('FAIL')&head.target.eq(0)],
        ['OUTCOME_UNKNOWN','ABSTAIN','TP','FP','FN','TN'],default='OTHER')
    head['period']=None
    for period in history.FOLDS:head.loc[head.date.between(period['start'],period['end']),'period']=period['id']
    head=head[head.period.notna()].copy()
    B=head[head.original_frozen_B].copy()
    # Field equality treats missing==missing but never imputes unknown values.
    daily=[];field_rows=[]
    for field in fields:
        group=head.groupby('date')['feature_'+field]
        n=group.nunique(dropna=False)
        eligible=head.groupby('date').size().gt(1)
        field_rows.append(dict(field=field,days=len(n),multi_candidate_days=int(eligible.sum()),
          same_value_multi_candidate_days=int((n.eq(1)&eligible).sum()),
          differs_within_day_days=int((n.gt(1)&eligible).sum()),
          all_days_common=bool(n[eligible].eq(1).all()),
          missing_rows=int(head['feature_'+field].isna().sum()),
          distinct_values_including_missing=int(head['feature_'+field].nunique(dropna=False))))
    for day,f in B.groupby('date',sort=True):
        known=f[f.outcome_known];same_day=head[head.date.eq(day)]
        shared=[x for x in fields if same_day['feature_'+x].nunique(dropna=False)==1]
        daily.append(dict(date=day,period=f.period.iloc[0],**counts(f),
          shared_field_count_full_day=len(shared),shared_fields='|'.join(shared),
          distinct_board_groups=int(f.boards.nunique())))
    csv('complete_original_B_day_diagnostics.csv',pd.DataFrame(daily))
    field_df=pd.DataFrame(field_rows);csv('all92_feature_within_day_variation.csv',field_df)
    daily_map={x['date']:x for x in daily}
    relation_groups={key:f for key,f in parents.groupby(['date','code'])}
    entries=[]
    for row in B.itertuples():
        day=B[B.date.eq(row.date)];peers=day[day.boards.eq(row.boards)]
        sameheight_all=head[head.date.eq(row.date)&head.boards.eq(row.boards)]
        p=relation_groups.get((row.date,row.code),parents.iloc[:0])
        entries.append(dict(date=row.date,code=row.code,
          B_day_candidates=len(day),B_sameheight_candidates=len(peers),all_sameheight_candidates=len(sameheight_all),
          B_day_TP=int(day.disposition.eq('TP').sum()),B_day_FP=int(day.disposition.eq('FP').sum()),
          B_sameheight_TP=int(peers.disposition.eq('TP').sum()),B_sameheight_FP=int(peers.disposition.eq('FP').sum()),
          sameheight_competing_codes='|'.join(sameheight_all.code),
          source_verified_price_only=bool(pd.isna(row.current_price_shape) and pd.notna(row.effective_verified_shape)),
          independent_role_truth_present=pd.notna(row.independent_role_identity_truth),
          bound_parent_relations=len(p),author_supported_parent_relations=int(p.author_supported_hypothesis.eq(True).sum()),
          parent_industry_known=int(p.industry_relation.ne('UNKNOWN_UNLESS_SEPARATELY_BOUND').sum()),
          parent_funds_causality_known=int(p.funds_causality.ne('UNKNOWN').sum()),
          parent_missing_current_price=int(p.anchor_current_state.eq('UNKNOWN_CURRENT_ANCHOR_PRICE').sum()),
          shared_field_count_full_day=daily_map[row.date]['shared_field_count_full_day']))
    B=B.merge(pd.DataFrame(entries),on=['date','code'],validate='one_to_one')
    B['score_band_DIAGNOSTIC_ONLY']=pd.cut(B.b_score,bins=[-np.inf,.5,.6,.7,.8,.9,np.inf],labels=['<=0.5','(0.5,0.6]','(0.6,0.7]','(0.7,0.8]','(0.8,0.9]','>0.9']).astype(str)
    csv('complete_three_period_original_B_all_outcomes.csv',B)
    csv('september_complete143_original_B.csv',B[B.period.eq('2026Sep')])
    csv('september_all24_FP_and22_TP.csv',B[B.period.eq('2026Sep')&B.disposition.isin(['FP','TP'])])
    csv('september_all24_FP.csv',B[B.period.eq('2026Sep')&B.disposition.eq('FP')])
    csv('september_all22_TP.csv',B[B.period.eq('2026Sep')&B.disposition.eq('TP')])
    # Fixed structural partitions, no target-guided threshold construction.
    dims=['boards','review_cycle_state','review_primary_role','current_price_path_status','full_cohort_status','active_anchor_discovery_scope','score_band_DIAGNOSTIC_ONLY','source_verified_price_only']
    decomposed=[]
    for period,f in B.groupby('period'):
        for dimension in dims:
            for value,part in f.groupby(dimension,dropna=False,observed=False):
                decomposed.append(dict(period=period,dimension=dimension,stratum=str(value),**counts(part)))
    csv('all_three_period_fixed_structural_decompositions.csv',pd.DataFrame(decomposed))
    summary=[]
    for period,f in B.groupby('period',sort=False):
        c=counts(f);fp=f[f.disposition.eq('FP')];tp=f[f.disposition.eq('TP')]
        summary.append(dict(period=period,**c,
          recall=c['TP']/c['actual_positive'] if c['actual_positive'] else None,
          precision=c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] else None,
          actual_positive_rate=c['actual_positive']/c['known'],
          error_share_of_positive_calls=c['FP']/(c['TP']+c['FP']),
          FPs_on_day_also_has_TP=int(fp.B_day_TP.gt(0).sum()),
          FPs_same_day_height_also_has_TP=int(fp.B_sameheight_TP.gt(0).sum()),
          FPs_days=int(fp.date.nunique()),days=int(f.date.nunique()),
          FPs_with_2plus_same_day_FP=int(fp.B_day_FP.ge(2).sum()),
          FPs_path_unknown=int(fp.current_price_path_status.ne('OBSERVED_CONTINUOUS_10PCT_PRICE_PATH_PROXY').sum()),
          TPs_path_unknown=int(tp.current_price_path_status.ne('OBSERVED_CONTINUOUS_10PCT_PRICE_PATH_PROXY').sum()),
          FPs_cycle_incomplete=int(fp.feature_causal_cycle_complete.ne(1).sum()),
          TPs_cycle_incomplete=int(tp.feature_causal_cycle_complete.ne(1).sum()),
          FPs_independent_role_truth=int(fp.independent_role_truth_present.sum()),
          TPs_independent_role_truth=int(tp.independent_role_truth_present.sum()),
          FPs_author_supported_parent_rows=int(fp.author_supported_parent_relations.gt(0).sum()),
          TPs_author_supported_parent_rows=int(tp.author_supported_parent_relations.gt(0).sum()),
          FPs_theme_first_support_known=int(fp.theme_first_support.notna().sum()),
          TPs_theme_first_support_known=int(tp.theme_first_support.notna().sum()),
          FPs_price_source_only_repaired=int(fp.source_verified_price_only.sum()),
          TPs_price_source_only_repaired=int(tp.source_verified_price_only.sum())))
    csv('three_period_summary.csv',pd.DataFrame(summary))
    comparison=[]
    for (period,arm),f in raw[raw.daily_one_word.eq(0)].assign(period=lambda f: np.select([f.date.between(p['start'],p['end']) for p in history.FOLDS],[p['id'] for p in history.FOLDS],default='OUTSIDE')).groupby(['period','arm']):
        if period=='OUTSIDE' or arm=='PAIRWISE_DAY_RANK':continue
        c=counts(f);comparison.append(dict(period=period,arm=arm,**c))
    csv('complete_same_B_classifier_comparison.csv',pd.DataFrame(comparison))
    # Class-balance base-rate diagnostic uses saved window members, not refitting.
    weights=[]
    for check in data['training_checks']:
        if check['cutoff']>'2026-09-01':continue
        train=history.training_rows(full,check['cutoff'],'relay')
        initial=1./train.groupby('date').date.transform('size')
        mass=initial.groupby(train.target).sum()
        weights.append(dict(month=check['month'],rows=len(train),positive_rows=int(train.target.eq(1).sum()),
          positive_raw_row_rate=float(train.target.eq(1).mean()),positive_inverse_day_mass=float(mass[1]/mass.sum()),
          final_positive_mass_fraction=.5,final_negative_mass_fraction=.5,
          strict_decision_threshold=.5,probability_calibration=False))
    csv('all_historical_training_windows_class_balance_diagnostic.csv',pd.DataFrame(weights))
    sep=B[B.period.eq('2026Sep')]
    assert len(sep)==143 and sep.outcome_known.sum()==107 and sep.target.eq(1).sum()==40
    assert sep.disposition.eq('FP').sum()==24 and sep.disposition.eq('TP').sum()==22
    assert all(x['FP']>=0 and x['TP']>=0 for x in summary)
    assert B.independent_role_truth_present.sum()==0
    assert parents.author_supported_hypothesis.eq(True).sum()==0
    assert all(not x.startswith('theme') for x in fields)
    source_paths=[PRED,DATA,CONTEXT,ROOT/'expectation_345_features.py',ROOT/'expectation_345_research.py',ROOT/'relay_causal_review_context.py',ROOT/'model_upgrade_history.py',Path(__file__)]
    bindings={str(p.resolve()):sha(p) for p in source_paths}
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),arm=ARM,new_fits=0,fetches=0,source_writeback=0,
      last_signal_date_read='2026-09-30',October_outcomes_accessed=False,classification_threshold_changed=False,
      complete_original_B_rows=len(B),period_summaries=summary,
      every_day_constant_fields=field_df[field_df.all_days_common].field.tolist(),
      no_independent_role_truth=True,no_author_supported_parent_relations=True,
      no_independent_typed_theme_features=True,no_orderbook_or_auction_features=True,
      caveat='Descriptive overlap and data gaps do not establish cause. Shared fields do not mean complete92 identical; remaining individual fields differ. No filter chosen from September.',
      source_bindings=bindings)
    write_json(OUT/'audit_receipt.json',receipt)
    print(json.dumps(dict(rows=len(B),summaries=summary,all_day_common=len(receipt['every_day_constant_fields'])),ensure_ascii=False,default=int))

if __name__=='__main__':main()
