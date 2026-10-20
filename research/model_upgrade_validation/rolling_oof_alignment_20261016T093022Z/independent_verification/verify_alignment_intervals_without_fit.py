"""Independent paired block-CI recount for the completed monthly252 study."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
BASE='SOURCE_PATCHED_FEATURE_CLASSIFIER'
ARMS=['MONTHLY252_OOF_FROZEN_SCORE_FILTER','MONTHLY252_OOF_FROZEN_INDIVIDUAL_FILTER']
PERIODS=[('2025H2','2025-07-01','2025-12-31'),('2026JanAug','2026-01-01','2026-08-31'),('2026Sep','2026-09-01','2026-09-30'),('2026Oct','2026-10-01','2026-10-15'),('HISTORICAL_POOLED','2025-07-01','2026-09-30')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    receipt=HERE/'monthly252_interval_independent_receipt.json';output=HERE/'独立复算_全部40项5日块探索区间.csv'
    if receipt.exists() or output.exists():raise FileExistsError('immutable interval audit already exists')
    source=ROOT/'execution/all56655_predictions_BEFORE_EVALUATION.csv';published=ROOT/'evaluation/exploratory_5day_block_intervals.csv'
    frame=pd.read_csv(source,dtype={'code':str},float_precision='round_trip',low_memory=False)
    official=pd.read_csv(published,float_precision='round_trip');assert len(official)==40
    rows=[]
    for period,first,last in PERIODS:
        pool=frame[frame.date.between(first,last)&frame.daily_one_word.eq(0)&frame.arm.isin([BASE]+ARMS)].copy()
        dates=sorted(pool.date.unique());n=len(dates)
        valid=pool.label_valid.astype(str).str.lower().eq('true')&pool.target.isin([0,1])
        pool['TP']=(valid&pool.target.eq(1)&pool.evaluation_call.eq('PROMOTE')).astype(int)
        pool['FP']=(valid&pool.target.eq(0)&pool.evaluation_call.eq('PROMOTE')).astype(int)
        pool['positive']=(valid&pool.target.eq(1)).astype(int)
        pool['negative']=(valid&pool.target.eq(0)).astype(int)
        daily={arm:part.groupby('date')[['TP','FP','positive','negative']].sum().reindex(dates,fill_value=0).to_numpy(float) for arm,part in pool.groupby('arm')}
        for arm in ARMS:np.testing.assert_array_equal(daily[BASE],daily[arm])
        rng=np.random.default_rng(20261016)
        starts=rng.integers(0,n,size=(2000,int(np.ceil(n/5))))
        ids=((starts[:,:,None]+np.arange(5))%n).reshape(2000,-1)[:,:n]
        def measures(a):
            summed=a[ids].sum(axis=1);tp=summed[:,0];fp=summed[:,1];pos=summed[:,2]
            div=lambda x,y:np.divide(x,y,out=np.full(2000,np.nan),where=y>0)
            return dict(Precision=div(tp,tp+fp),Recall=div(tp,pos),F1=div(2*tp,pos+tp+fp),FP=fp)
        base=measures(daily[BASE])
        for arm in ARMS:
            new=measures(daily[arm])
            for metric,value in new.items():
                delta=value-base[metric];finite=delta[np.isfinite(delta)]
                bounds=np.quantile(finite,[.0125,.9875]) if len(finite) else [np.nan,np.nan]
                row=official[official.period.eq(period)&official.arm.eq(arm)&official.metric.eq(metric)].iloc[0]
                np.testing.assert_allclose(bounds,[row.lower,row.upper],rtol=0,atol=1e-14,equal_nan=True)
                assert int(row.finite_replicates)==len(finite) and int(row.total_replicates)==2000 and int(row.block_signal_days)==5
                rows.append(dict(period=period,arm=arm,metric=metric,lower=bounds[0],upper=bounds[1],finite_replicates=len(finite),
                    total_replicates=2000,signal_days=n,block_signal_days=5,seed=20261016,quantiles='.0125/.9875',
                    unknown_not_negative=True,identical_daily_counts_verified=True,exploratory_not_confirmatory=True))
    result=pd.DataFrame(rows);result.to_csv(output,index=False,encoding='utf-8-sig')
    saved=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),verification_fits=0,network_calls=0,independent_interval_rows=40,
        all_bounds_zero_due_to_identical_daily_predictions=True,unknown_not_negative=True,
        only_current_two_arm_adjustment_prior_sequence_uncorrected=True,not_blind_not_confirmatory=True,
        source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__),source,published,ROOT/'protocol_BEFORE_ANY_FIT.json']},
        output_bindings={str(output.resolve()):sha(output)})
    receipt.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Independent40 paired intervals verified, all bounds[0,0]; finite draw counts retained.')

if __name__=='__main__':main()
