"""Read back frozen probe outputs, directly count outcomes; no fits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
OLD=REPO/'research/model_upgrade_validation/positive_filter_20261016T091007Z/execution/all49101_predictions_BEFORE_EVALUATION.csv'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,dtype={'code':str},float_precision='round_trip',low_memory=False)

def main():
    target=ROOT/'post_save_readback_verification.json'
    if target.exists():raise FileExistsError('no overwrite of verification')
    result=json.loads((ROOT/'fixed_probe_result.json').read_text(encoding='utf8'))
    for p,h in result['source_bindings'].items():assert sha(p)==h
    for p,h in result['output_bindings'].items():assert sha(p)==h
    old=read(OLD);new=read(ROOT/'all56655_predictions_BEFORE_EVALUATION.csv')
    pd.testing.assert_frame_equal(old,new.iloc[:49101][old.columns],check_exact=True,check_dtype=False)
    assert len(new)==56655 and new.arm.nunique()==15 and new.groupby('arm').size().eq(3777).all()
    summary=read(ROOT/'all_same_pool_classification_metrics.csv')
    ranges={'2025H2':('2025-07-01','2025-12-31'),'2026JanAug':('2026-01-01','2026-08-31'),'2026Sep':('2026-09-01','2026-09-30'),'2026Oct':('2026-10-01','2026-10-15'),'HISTORICAL_POOLED':('2025-07-01','2026-09-30')}
    checked=0
    for row in summary.itertuples(index=False):
        first,last=ranges[row.period];s=new[new.arm.eq(row.arm)&new.date.between(first,last)]
        if row.scope=='ORIGINAL_NONWORD':s=s[s.daily_one_word.eq(0)]
        mask=s.label_valid.astype(str).str.lower().eq('true')&s.target.isin([0,1]);k=s[mask]
        positive=k.target.eq(1);promote=k.evaluation_call.eq('PROMOTE');tp=int((positive&promote).sum());fp=int((~positive&promote).sum());pos=int(positive.sum())
        assert (len(s),len(k),len(s)-len(k),tp,fp)==(row.candidates,row.known,row.unknown_outcomes,row.tp,row.fp)
        metrics=[tp/pos if pos else np.nan,tp/(tp+fp) if tp+fp else np.nan,2*tp/(pos+tp+fp) if pos+tp+fp else np.nan]
        np.testing.assert_allclose(metrics,[row.Recall,row.Precision,row.F1],atol=1e-14,rtol=0,equal_nan=True);checked+=1
    for arm in ['NO_META_REFIT_SCORE_ONLY','NO_META_REFIT_INDIVIDUAL_CONTEXT']:
        s=new[new.arm.eq(arm)]
        hist=s[s.daily_one_word.eq(0)&s.date.le('2026-09-30')];known=hist.label_valid.astype(str).str.lower().eq('true')&hist.target.isin([0,1])
        assert (len(hist),int(known.sum()),int((~known).sum()),int(hist.target.eq(1).sum()))==(2980,2932,48,871)
        sep=s[s.daily_one_word.eq(0)&s.date.str.startswith('2026-09')];known=sep.label_valid.astype(str).str.lower().eq('true')&sep.target.isin([0,1])
        assert (len(sep),int(known.sum()),int((~known).sum()),int(sep.target.eq(1).sum()))==(143,107,36,40)
        current=s[s.date.eq('2026-10-15')];assert len(current)==6 and current.target.isna().all() and not current.label_valid.astype(str).str.lower().eq('true').any()
    saved_before=json.loads((ROOT/'all_predictions_COMPLETED_BEFORE_EFFECTS.json').read_text(encoding='utf8'))
    assert saved_before['formed_at_UTC']<result['formed_at_UTC']
    receipt=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),verification_fits=0,old49101_readback_all_columns_exact=True,
        complete56655_fifteen_heads=True,full_B_and_september_denominators_exact=True,current6_each_new_head_unknown=True,
        complete_predictions_saved_before_effect_result=True,direct_metric_groups_recounted=checked,
        source_bindings={str(p.resolve()):sha(p) for p in [Path(__file__),OLD,ROOT/'all56655_predictions_BEFORE_EVALUATION.csv',ROOT/'all_same_pool_classification_metrics.csv',ROOT/'fixed_probe_result.json']},output_bindings={})
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print('Readback exact and direct counts verified:',checked)

if __name__=='__main__':main()
