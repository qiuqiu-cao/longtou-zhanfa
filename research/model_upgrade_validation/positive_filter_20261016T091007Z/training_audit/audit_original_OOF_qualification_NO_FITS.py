"""Read-only temporal/sample qualification for a fixed positive-only filter."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import model_upgrade_history as history

PRIOR=REPO/'research/model_upgrade_validation/false_positive_20261016T030307Z'
SOURCE=REPO/'research/model_upgrade_validation/ranking_optimization_20261015T173045Z/source_feature_training'
FIELDS=['boards','height_gap','same_height_rivals','turnover','amplitude','log_signal_amount','price_same_height_percentile','signal_amount_same_height_percentile','turnover_same_height_percentile']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    data_path=PRIOR/'fixed_input_BEFORE_ANY_FIT.joblib'
    source_path=SOURCE/'prepared.joblib'
    data=joblib.load(data_path);source=joblib.load(source_path)
    full=data['full'];shape=source['verified_shapes'].set_index(['date','code']).verified_current_price_shape
    assert len(full)==9315 and len(data['fields'])==92
    rows=[];bindings={str(p.resolve()):sha(p) for p in [data_path,source_path,Path(__file__)]}
    lookup=full.set_index(['date','code'])
    for plan in data['plans']:
        window=history.training_rows(full,plan['cutoff'],'relay')
        days=sorted(window.date.unique())
        assert len(days)==252 and history._frame_digest(window,data['fields'])==plan['training_digest']
        blocks=[]
        for number in (1,2,3):
            path=PRIOR/f"execution/{plan['month']}_inner{number}_all_validation_scores_BEFORE_EFFECTS.csv"
            cached=pd.read_csv(path,dtype={'code':str},float_precision='round_trip')
            validation_days=days[63*number:63*(number+1)]
            expected=window[window.date.isin(validation_days)]
            columns=['date','code','target_date','target','label_valid','frozen_original_nonword_identity']
            pd.testing.assert_frame_equal(cached[columns].reset_index(drop=True),expected[columns].reset_index(drop=True),check_exact=True,check_dtype=False)
            cached['verified_current_price_shape']=[shape.loc[(r.date,r.code)] for r in cached.itertuples()]
            for field in FIELDS:
                cached[field]=[lookup.loc[(r.date,r.code),field] for r in cached.itertuples()]
            cached['block_number']=number
            blocks.append(cached)
            bindings[str(path.resolve())]=sha(path)
        all_oof=pd.concat(blocks,ignore_index=True)
        third_first=blocks[2].date.min()
        two_before=pd.concat(blocks[:2],ignore_index=True)
        meta_train=two_before[two_before.target_date.lt(third_first)].copy()
        assert meta_train.target_date.max()<third_first
        eligibility=lambda frame:frame.frozen_original_nonword_identity.eq(0)&frame.verified_current_price_shape.eq(0)&frame.raw_score.gt(.5)
        pre=meta_train[eligibility(meta_train)]
        third=blocks[2][blocks[2].frozen_original_nonword_identity.eq(0)]
        threshold_candidates=third[eligibility(third)]
        final=all_oof[eligibility(all_oof)]
        assert final.target_date.max()<plan['cutoff']
        assert len(pre)>0 and len(final)>0 and pre.target.nunique()==2 and final.target.nunique()==2
        unknown_before=int((two_before.frozen_original_nonword_identity.eq(0)&two_before.verified_current_price_shape.isna()).sum())
        rows.append(dict(month=plan['month'],outer_cutoff=plan['cutoff'],window_days=252,window_rows=len(window),
            third_first=third_first,pre_two_block_rows=len(two_before),strict_maturity_removed_rows=len(two_before)-len(meta_train),
            meta_train_positive_eligible=len(pre),meta_train_true_positive=int(pre.target.eq(1).sum()),meta_train_false_positive=int(pre.target.eq(0).sum()),
            meta_train_max_target=pre.target_date.max(),third_full_original_B_rows=len(third),third_full_original_B_positives=int(third.target.eq(1).sum()),
            third_primary_positive_calls=len(threshold_candidates),third_primary_TP=int(threshold_candidates.target.eq(1).sum()),third_primary_FP=int(threshold_candidates.target.eq(0).sum()),
            final_meta_eligible=len(final),final_meta_true_positive=int(final.target.eq(1).sum()),final_meta_false_positive=int(final.target.eq(0).sum()),
            final_meta_max_target=final.target_date.max(),pre_two_B_qualification_unknown=unknown_before,
            individual9_train_missing_cells=int(pre[FIELDS].isna().sum().sum()),individual9_final_missing_cells=int(final[FIELDS].isna().sum().sum()),
            original_pool_fields_copied_before_positive_selection=True))
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'all16_positive_meta_training_qualification_NO_FITS.csv',index=False,encoding='utf-8-sig')
    receipt=dict(new_fits=0,old_files_modified=0,target16_results_read=False,windows_checked=16,cached_OOF_blocks_checked=48,
        all_original252_members_and92_fields_fixed=True,meta_train_targets_strictly_before_third_first=True,
        final_meta_targets_strictly_before_outer_cutoff=True,all_meta_train_and_final_two_classes=True,
        min_meta_train_rows=int(frame.meta_train_positive_eligible.min()),min_meta_train_TP=int(frame.meta_train_true_positive.min()),min_meta_train_FP=int(frame.meta_train_false_positive.min()),
        min_final_meta_rows=int(frame.final_meta_eligible.min()),min_final_meta_TP=int(frame.final_meta_true_positive.min()),min_final_meta_FP=int(frame.final_meta_false_positive.min()),
        nine_individual_fields=FIELDS,same_height_fields_from_complete_fixed_source_not_recomputed_after_positive_selection=True,
        row_equal_meta_training_no_sample_weight_no_class_weight=True,source_bindings=bindings)
    (OUT/'original_OOF_qualification_NO_FITS_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_bindings'},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
