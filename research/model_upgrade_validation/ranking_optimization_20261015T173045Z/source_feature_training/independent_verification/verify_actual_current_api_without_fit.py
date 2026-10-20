"""Last audit: invoke the real current API and test target insensitivity."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("source_audit_helpers",OUT/"verify_source_feature_independent.py")
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
sys.path.insert(0,str(helpers.REPO))
import relay_source_feature_research as research
import relay_source_feature_alignment as alignment


def main():
    directory=helpers.WORK/"same_source_route_alignment"
    fixed=helpers.read(directory/"same_source_aligned_fixed_result.json")
    for path,wanted in fixed["source_bindings"].items():assert helpers.sha(path)==wanted,path
    original=joblib.load(helpers.OLD/"trainable_corrected_dataset_QUALIFIED.joblib")
    verified=helpers.csv(helpers.BATCH/"source_repair/verified_22662_final/all244_verified_qualification_rows.csv")
    current=original["full"][original["full"].date.eq("2026-10-15")].copy()
    assert len(current)==6 and current.target.isna().all() and not current.label_valid.any()
    qualified=verified[verified.date.eq("2026-10-15")]
    copy,changes=research.patched_feature_copy(current,original["fields"],qualified,0)
    pd.testing.assert_frame_equal(current,copy,check_exact=True)
    shape=research.verified_shapes(copy,qualified)
    final_path=directory/"complete26439_predictions_WITH_SAME_SOURCE_ROUTES.csv"
    final=helpers.csv(final_path)
    ref=final[final.arm.eq("CORRECTED_CAUSAL")&final.date.eq("2026-10-15")].copy()
    ref["evaluation_call"]=ref.recovered_B_call
    end=helpers.read(helpers.WORK/"execution/2026-10_COMPLETED.json")
    assert helpers.sha(end["model_path"])==end["model_sha256"]
    model=joblib.load(end["model_path"])["model"]
    with patch.object(HistGradientBoostingClassifier,"fit",side_effect=AssertionError("current audit cannot fit")):
        with threadpool_limits(limits=2):
            score=model.predict_proba(copy[original["fields"]])[:,list(model.classes_).index(1)]
        first=alignment.predict_with_same_source_routes(copy,score,shape,ref)
        # Full-data metadata copied from reference may carry targets for later
        # settlement; neither the score nor route may depend on those values.
        perturbed=copy.copy();perturbed["target"]=[0.,1.,0.,1.,0.,1.]
        perturbed["target_date"]="2099-12-31";perturbed["label_valid"]=True
        perturbed_ref=ref.copy();perturbed_ref["target"]=[1.,0.,1.,0.,1.,0.]
        perturbed_ref["target_date"]="2099-12-31";perturbed_ref["label_valid"]=True
        with threadpool_limits(limits=2):
            changed_score=model.predict_proba(perturbed[original["fields"]])[:,list(model.classes_).index(1)]
        assert np.array_equal(score,changed_score)
        second=alignment.predict_with_same_source_routes(perturbed,changed_score,shape,perturbed_ref)
    calls=["score","b_score","ranking_score","call","evaluation_call","recovered_B_call","aligned_evaluation_call","effective_verified_shape","same_source_aligned_score"]
    helpers.equal_cells(first,second,["date","code"],calls)
    saved=final[final.arm.eq(helpers.ARM)&final.date.eq("2026-10-15")]
    z=first.merge(saved,on=["date","code"],validate="one_to_one",suffixes=("_api","_saved"))
    assert len(z)==6
    for field in ["score","b_score","ranking_score","same_source_aligned_score"]:
        delta=(z[field+"_api"]-z[field+"_saved"]).abs()
        assert (delta.le(1e-12)|(z[field+"_api"].isna()&z[field+"_saved"].isna())).all()
    for field in ["call","evaluation_call","aligned_evaluation_call"]:assert z[field+"_api"].eq(z[field+"_saved"]).all()
    prior=helpers.read(OUT/"训练与原始分数独立核验收据.json")
    raw_defects=helpers.csv(OUT/"原执行附件A资格差异_保留原失败.csv")
    new=final[final.arm.eq(helpers.ARM)]
    baseline=final[final.arm.eq("CORRECTED_CAUSAL")]
    nine=raw_defects[["date","code"]].merge(new,on=["date","code"],validate="one_to_one").merge(baseline[["date","code","score","recovered_B_call"]],on=["date","code"],validate="one_to_one",suffixes=("_new","_baseline"))
    assert len(nine)==9 and nine.call.eq(nine.recovered_B_call_baseline).all()
    assert (nine.score_new.eq(nine.score_baseline)|(nine.score_new.isna()&nine.score_baseline.isna())).all()
    assert int((raw_defects.evaluation_call_ref.eq("UNKNOWN")&raw_defects.evaluation_call_new.eq("PROMOTE")).sum())==7
    helpers.write_csv(OUT/"实际current6_API与留档逐股核对.csv",z)
    helpers.write_csv(OUT/"A通道9例修正守恒独立核对.csv",nine)
    learning=helpers.csv(OUT/"全部3777原模型patched输入与新模型分数归因.csv")
    reuse=learning[learning.reused_model]
    assert reuse.month.nunique()==7 and reuse.fitted_weight_score_change.abs().le(1e-12).all()
    for path,wanted in fixed["source_bindings"].items():assert helpers.sha(path)==wanted,path
    receipt=dict(formed_at=datetime.now(timezone.utc).isoformat(),passed=True,new_fits=0,
        actual_shared_feature_apply_called=True,current_cells_patched=0,actual_wrapper_called_twice=True,
        current_rows=6,current_future_outcomes_known=0,model_and_route_target_metadata_insensitive=True,
        target_fields_never_in92_feature_allowlist=not any("target" in field for field in original["fields"]),
        all_current_api_scores_and_calls_match_final_saved=True,
        original_A_route_failure_nine_rows_retained=True,original_wrong_new_positive_seven_retained=True,
        final_A_route_nine_exactly_inherit_old_qualified_score_and_call=True,
        reused_seven_windows_learning_score_difference_zero=True,
        previous22662_conservation_proven_by=str((OUT/"最终同资格独立核验收据.json").resolve()),
        restrictions="Target perturbation checks score/route sensitivity only; no new predictive accuracy, original-time eligibility or probability calibration claim",
        source_bindings={str(p.resolve()):helpers.sha(p) for p in [Path(__file__),Path(research.__file__),Path(alignment.__file__),final_path,OUT/"实际current6_API与留档逐股核对.csv",OUT/"A通道9例修正守恒独立核对.csv",OUT/"三段增益归因独立核验收据.json",OUT/"最终同资格独立核验收据.json",directory/"current15_six_SHARED_FEATURE_API_receipt.json"]})
    helpers.write_new(OUT/"实际API无目标泄漏与A守恒收尾核验收据.json",receipt)
    print(json.dumps({k:receipt[k] for k in ["passed","current_rows","model_and_route_target_metadata_insensitive","final_A_route_nine_exactly_inherit_old_qualified_score_and_call","reused_seven_windows_learning_score_difference_zero"]}))


if __name__=="__main__":main()
