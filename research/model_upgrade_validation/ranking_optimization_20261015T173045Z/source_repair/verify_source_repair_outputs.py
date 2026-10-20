"""Independent output invariants and metrics; never invokes a model fit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import json
import shutil
import numpy as np
import pandas as pd
import short_term_thinking_dataset_builder as b

OUT=Path(__file__).resolve().parent


def main():
    checks={};counts={};bindings={str(Path(__file__).resolve()):b.sha(__file__)}
    for name,expected in [('verified_18885',18885),('verified_22662',22662)]:
        folder=OUT/name;receipt=b.read(folder/'source_qualification_repair_receipt.json')
        for path,digest in receipt['output_bindings'].items():
            if b.sha(path)!=digest:raise ValueError('repair output changed')
            bindings[path]=digest
        if b.sha(receipt['implementation_path'])!=receipt['implementation_sha256']:
            raise ValueError('implementation changed since run')
        source=pd.read_csv(receipt['predictions_path'],dtype={'code':str},low_memory=False)
        attached=pd.read_csv(folder/'all_predictions_with_isolated_qualification.csv',dtype={'code':str},low_memory=False)
        qual=pd.read_csv(folder/'all244_verified_qualification_rows.csv',dtype={'code':str})
        # Exact equality over all old columns includes every score/call/label.
        pd.testing.assert_frame_equal(source,attached[source.columns],check_exact=True)
        checks[name+'_all_original_columns_exact']=True
        checks[name+'_complete_rows_retained']=len(source)==len(attached)==expected
        checks[name+'_all244_unknown_retained']=len(qual)==244 and qual.verified_current_price_shape.notna().sum()==47 and qual.verified_current_price_shape.isna().sum()==197
        checks[name+'_54_observed_nonclosing_remain_unknown']=len(qual[qual.qualification_status.eq('OBSERVED_NONCLOSING_10PCT_PRICE_PATH')])==54 and qual[qual.qualification_status.eq('OBSERVED_NONCLOSING_10PCT_PRICE_PATH')].verified_current_price_shape.isna().all()
        base=attached[attached.arm.eq('BASE')]
        checks[name+'_BASE_call_and_score_preserved']=base.recovered_B_call.equals(base.call) and np.array_equal(base.score,source[source.arm.eq('BASE')].score,equal_nan=True)
        pairwise=attached[attached.arm.str.startswith('PAIRWISE')]
        checks[name+'_PAIRWISE_no_absolute_call']=pairwise.recovered_B_call.eq('UNKNOWN').all()
        allowed=attached.arm.isin(['CORRECTED_CAUSAL','CORRECTED_CAUSAL_COST1P5'])
        changed=attached.call.ne(attached.recovered_B_call)
        checks[name+'_only_existing_absolute_B_heads_reclassified']=attached[changed].arm.isin(['CORRECTED_CAUSAL','CORRECTED_CAUSAL_COST1P5']).all()
        used=attached[changed]
        checks[name+'_strict_same_row_b_score_threshold']=used.b_score.notna().all() and used.recovered_B_call.equals(pd.Series(np.where(used.b_score.gt(.5),'PROMOTE','FAIL'),index=used.index))
        sep=attached[attached.date.ge('2026-09-01')&attached.date.le('2026-09-30')&attached.daily_one_word.eq(0)&attached.current_price_shape.isna()]
        counts[name+'_september_repaired_per_arm']=sep.groupby('arm').size().to_dict()
        checks[name+'_all35_September_per_head_restored']=all(x==35 for x in sep.groupby('arm').size()) and sep.verified_current_price_shape.eq(0).all()
        table=pd.read_csv(folder/'same_pool_before_after_nonword_metrics.csv')
        independent=[]
        for arm in attached.arm.unique():
            rows=attached[attached.arm.eq(arm)&attached.date.ge('2026-09-01')&attached.date.le('2026-09-30')&attached.daily_one_word.eq(0)]
            known=rows[rows.label_valid.fillna(False)&rows.target.notna()]
            pos=int(known.target.eq(1).sum())
            for mode,field in [('ORIGINAL_FROZEN_CALL','call'),('SOURCE_REPAIRED_RETROSPECTIVE','recovered_B_call')]:
                tp=int((known.target.eq(1)&known[field].eq('PROMOTE')).sum())
                fp=int((known.target.eq(0)&known[field].eq('PROMOTE')).sum())
                record=table[table.period.eq('2026Sep')&table.arm.eq(arm)&table['mode'].eq(mode)].iloc[0]
                assert record.tp==tp and record.fp==fp and record.positives==pos
                assert abs(record.Recall-tp/pos)<1e-14
                if tp+fp:assert abs(record.Precision-tp/(tp+fp))<1e-14
                else:assert pd.isna(record.Precision)
                if pos+tp+fp:assert abs(record.F1-2*tp/(pos+tp+fp))<1e-14
                independent.append(dict(arm=arm,mode=mode,tp=tp,fp=fp,positives=pos))
        checks[name+'_September_metrics_independently_recomputed']=True
        counts[name+'_independent_September_metrics']=independent
        counts[name+'_heads_with_actual_b_score']=attached.groupby('arm').b_score.count().to_dict()
        bindings[str(folder/'source_qualification_repair_receipt.json')]=b.sha(folder/'source_qualification_repair_receipt.json')
    if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
    preimage=OUT/'source_preimages';preimage.mkdir(exist_ok=False)
    for file in [ROOT/'relay_source_qualification_repair.py',ROOT/'tests/test_relay_source_qualification_repair.py']:
        dest=preimage/file.name;shutil.copyfile(file,dest);bindings[str(dest)]=b.sha(dest)
    test_xml=OUT.parent/'source_repair_tests.xml';bindings[str(test_xml)]=b.sha(test_xml)
    b.write_new(OUT/'independent_verification.json',dict(formed_at=b.now(),checks=checks,checks_passed=len(checks),counts=counts,
        immutable_implementation_snapshots=str(preimage),source_bindings=bindings,new_fits=0,network_calls=0,original_writebacks=0))
    print(json.dumps(dict(checks_passed=len(checks),bindings=len(bindings)),ensure_ascii=False))


if __name__=='__main__':main()
