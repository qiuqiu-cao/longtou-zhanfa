"""Independent pooled exploratory paired circular block intervals; no fits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
ARMS=['POSITIVE_FILTER_SCORE_ONLY','POSITIVE_FILTER_INDIVIDUAL_CONTEXT']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    output=HERE/'独立复算_历史合计5日块探索区间.csv'
    receipt=HERE/'pooled_exploratory_intervals_independent_receipt.json'
    if output.exists() or receipt.exists():raise FileExistsError('immutable interval audit already exists')
    source=ROOT/'execution/all49101_predictions_BEFORE_EVALUATION.csv'
    published=ROOT/'evaluation/exploratory_5day_block_intervals.csv'
    frame=pd.read_csv(source,dtype={'code':str},float_precision='round_trip',low_memory=False)
    sample=frame[frame.date.between('2025-07-01','2026-09-30')&frame.daily_one_word.eq(0)&frame.arm.isin([BASE]+ARMS)].copy()
    dates=sorted(sample.date.unique());assert len(dates)==302
    assert sample.groupby('arm').size().eq(2980).all()
    known=sample.label_valid.astype(str).str.lower().eq('true')&sample.target.isin([0,1])
    sample['TP']=(known&sample.target.eq(1)&sample.evaluation_call.eq('PROMOTE')).astype(int)
    sample['FP']=(known&sample.target.eq(0)&sample.evaluation_call.eq('PROMOTE')).astype(int)
    sample['positive']=(known&sample.target.eq(1)).astype(int)
    sample['negative']=(known&sample.target.eq(0)).astype(int)
    matrices={arm:part.groupby('date')[['TP','FP','positive','negative']].sum().reindex(dates,fill_value=0).to_numpy(float) for arm,part in sample.groupby('arm')}
    for value in matrices.values():assert value[:,2].sum()==871 and value[:,3].sum()==2061
    # Each paired resample keeps entire observed signal days, including those
    # with no known labels. Unknown rows never contribute a negative label.
    rng=np.random.default_rng(20261016)
    starts=rng.integers(0,302,size=(2000,61))
    resamples=(starts[:,:,None]+np.arange(5)[None,None,:])%302
    resamples=resamples.reshape(2000,-1)[:,:302]
    def measures(values):
        summed=values[resamples].sum(axis=1);tp=summed[:,0];fp=summed[:,1];pos=summed[:,2]
        divide=lambda numerator,denominator:np.divide(numerator,denominator,out=np.full(2000,np.nan),where=denominator>0)
        return dict(Precision=divide(tp,tp+fp),Recall=divide(tp,pos),F1=divide(2*tp,pos+tp+fp),FP=fp)
    base=measures(matrices[BASE]);rows=[]
    official=pd.read_csv(published,float_precision='round_trip')
    official=official[official.period.eq('HISTORICAL_POOLED')]
    assert len(official)==8
    for arm in ARMS:
        values=measures(matrices[arm])
        for metric,z in values.items():
            delta=z-base[metric];finite=delta[np.isfinite(delta)]
            lo,hi=np.quantile(finite,[.0125,.9875]) if len(finite) else [np.nan,np.nan]
            row=official[official.arm.eq(arm)&official.metric.eq(metric)].iloc[0]
            np.testing.assert_allclose([lo,hi],[row.lower,row.upper],atol=1e-14,rtol=0,equal_nan=True)
            assert int(row.finite_replicates)==len(finite) and int(row.total_replicates)==2000 and int(row.block_signal_days)==5
            rows.append(dict(period='HISTORICAL_POOLED',arm=arm,metric=metric,lower=lo,upper=hi,finite_replicates=len(finite),
                total_replicates=2000,observed_signal_days=302,block_signal_days=5,seed=20261016,quantile_low=.0125,quantile_high=.9875,
                unknown_outcomes_as_negative=False,exploratory_not_confirmatory=True,prior_sequential_trials_not_corrected=True))
    pd.DataFrame(rows).to_csv(output,index=False,encoding='utf-8-sig')
    result=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),verification_fits=0,network_calls=0,intervals_independently_reproduced=8,
        complete_observed_signal_days=302,complete_original_B_rows_per_arm=2980,known2932_unknown48=True,unknown_not_negative=True,
        paired_circular_blocks5_replicates2000_seed20261016=True,two_arm_quantiles_only=[.0125,.9875],
        sequential_research_multiplicity_not_corrected=True,not_blind_not_confirmatory=True,
        source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__),source,published,ROOT/'protocol_BEFORE_ANY_FIT.json']},
        output_bindings={str(output.resolve()):sha(output)})
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['arm','metric','lower','upper']].to_string(index=False))

if __name__=='__main__':main()
