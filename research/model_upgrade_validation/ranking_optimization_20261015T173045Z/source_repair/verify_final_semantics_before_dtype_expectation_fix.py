"""Verify source repair v2 without rewriting its v1 receipts or artifacts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import shutil
import pandas as pd
import short_term_thinking_dataset_builder as b

OUT=Path(__file__).resolve().parent


def main():
    old=OUT/'verified_22662';final=OUT/'verified_22662_final'
    old_meta=b.read(old/'source_qualification_repair_receipt.json')
    final_meta=b.read(final/'source_qualification_repair_receipt.json')
    checks={}
    old_snapshot=OUT/'source_preimages/relay_source_qualification_repair.py'
    checks['old_implementation_exact_preimage_preserved']=b.sha(old_snapshot)==old_meta['implementation_sha256']
    checks['final_implementation_hash_current']=b.sha(final_meta['implementation_path'])==final_meta['implementation_sha256']
    bindings={str(old_snapshot):b.sha(old_snapshot),str(Path(__file__).resolve()):b.sha(__file__)}
    for name in ['all_predictions_with_isolated_qualification.csv','all244_verified_qualification_rows.csv','all_repaired_prediction_rows.csv']:
        left=pd.read_csv(old/name,dtype={'code':str},low_memory=False)
        right=pd.read_csv(final/name,dtype={'code':str},low_memory=False)
        pd.testing.assert_frame_equal(left,right,check_exact=True)
        checks[name+'_exact_v1_reproduction']=True
    old_table=pd.read_csv(old/'same_pool_before_after_nonword_metrics.csv')
    table=pd.read_csv(final/'same_pool_before_after_nonword_metrics.csv')
    pairwise=table.arm.str.startswith('PAIRWISE')
    checks['PAIRWISE_no_absolute_head_explicit']=table[pairwise].metric_applicability.eq('NO_ABSOLUTE_EVENT_HEAD').all()
    checks['PAIRWISE_TP_FP_R_P_F1_undefined']=table.loc[pairwise,['tp','fp','Recall','Precision','F1']].isna().all().all()
    pd.testing.assert_frame_equal(old_table[~old_table.arm.str.startswith('PAIRWISE')].reset_index(drop=True),table.loc[~pairwise,old_table.columns].reset_index(drop=True),check_exact=True)
    checks['all_absolute_head_metrics_unchanged']=True
    data=pd.read_csv(final/'all_predictions_with_isolated_qualification.csv',dtype={'code':str},low_memory=False)
    source=pd.read_csv(final_meta['predictions_path'],dtype={'code':str},low_memory=False)
    pd.testing.assert_frame_equal(source,data[source.columns],check_exact=True)
    checks['all22662_original_columns_exact']=True
    checks['BASE_direction_retained']=data[data.arm.eq('BASE')].call.equals(data[data.arm.eq('BASE')].recovered_B_call)
    checks['PAIRWISE_no_new_direction']=data[data.arm.str.startswith('PAIRWISE')].recovered_B_call.eq('UNKNOWN').all()
    sep=data[data.date.ge('2026-09-01')&data.date.le('2026-09-30')&data.daily_one_word.eq(0)&data.current_price_shape.isna()]
    checks['all35_September_per_head_qualification']=all(x==35 for x in sep.groupby('arm').size()) and sep.verified_current_price_shape.eq(0).all()
    checks['47_shapes_and197_unknown_reproduced']=final_meta['recovered_shapes']==47 and final_meta['remaining_unknown_shapes']==197
    audit=OUT.parent/'source_audit';audit_receipt=b.read(audit/'source_audit_delivery_receipt.json')
    b.verify(audit_receipt['all_source_audit_output_bindings'])
    checks['source_audit_all18_original_files_unchanged']=True
    if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
    snapshot=final/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for file in [ROOT/'relay_source_qualification_repair.py',ROOT/'tests/test_relay_source_qualification_repair.py']:
        dest=snapshot/file.name;shutil.copyfile(file,dest);bindings[str(dest)]=b.sha(dest)
    for p in [OUT/'tests_final.xml',old/'source_qualification_repair_receipt.json',final/'source_qualification_repair_receipt.json',audit/'source_audit_delivery_receipt.json']:
        bindings[str(p)]=b.sha(p)
    b.verify(final_meta['output_bindings'])
    bindings.update(final_meta['output_bindings'])
    b.write_new(final/'final_semantic_verification.json',dict(formed_at=b.now(),checks=checks,checks_passed=len(checks),targeted_tests=21,
        code_transition=dict(old_implementation_historical_path=old_meta['implementation_path'],old_implementation_sha256=old_meta['implementation_sha256'],
            old_implementation_preserved_at=str(old_snapshot.resolve()),new_implementation_path=final_meta['implementation_path'],new_implementation_sha256=final_meta['implementation_sha256'],
            old_receipt_not_a_claim_about_current_source=True),
        qualifications_changed=0,scores_changed=0,original_calls_changed=0,new_fits=0,network_calls=0,original_writebacks=0,source_bindings=bindings))
    print(dict(checks_passed=len(checks),tests=21,qualifications_changed=0))


if __name__=='__main__':main()
