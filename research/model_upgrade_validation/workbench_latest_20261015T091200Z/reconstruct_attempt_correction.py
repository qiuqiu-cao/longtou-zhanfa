"""Reconstruct corrected attempt context while freezing every old result."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_attempt_reconstruction as repair
import short_term_thinking_research as old

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/"attempt_reclassification_correction"
    if out.exists():raise FileExistsError("immutable attempt correction output")
    source=b.ROOT/"results/v215/historical_board_attempts_20260920_v2/historical_closing_and_failed_board_attempts.csv"
    original=joblib.load(b.DEFAULT_RUN/"scope_correction/dataset.joblib")
    raw=pd.read_csv(source,dtype={"code":str})
    attempts,excluded=repair.reconstruct(raw,original["sessions"])
    old_attempts=original["attempts"]
    new_keys=set(zip(attempts.signal_date,attempts.code));old_keys=set(zip(old_attempts.signal_date,old_attempts.code))
    assert old_keys<=new_keys and len(new_keys-old_keys)==362 and len(attempts)==56008
    added=attempts[attempts.reclassified]
    assert len(added)==362 and added.pool_assignment.eq("FAILED_BOARD_OBSERVATION").all()
    matching=old_attempts.merge(attempts,on=["signal_date","code"],suffixes=("_old","_new"),validate="one_to_one")
    assert matching.pool_assignment_old.eq(matching.pool_assignment_new).all()
    assert matching.board_height_old.eq(matching.board_height_new).all()
    # This is a new evidence reconstruction with the same original causal
    # function. Do not modify or refit models trained on the old snapshots.
    features,edges=b.causal_features(original["prepared"]["relay"],original["prepared"]["market"],attempts,original["bars"],original["sessions"])
    before=original["features"]
    assert list(zip(features.date,features.code))==list(zip(before.date,before.code))
    changes=[]
    for field in b.DYNAMIC_FIELDS+["primary_role","roles"]:
        for a,z in zip(before[["date","code",field]].to_dict("records"),features[["date","code",field]].to_dict("records")):
            left,right=b.clean(a[field]),b.clean(z[field])
            if left!=right:changes.append(dict(date=a["date"],code=a["code"],field=field,before=left,after=right))
    bindings=original["sources"]|{str(source.resolve()):b.sha(source),str(Path(repair.__file__).resolve()):b.sha(repair.__file__),str(Path(__file__).resolve()):b.sha(__file__)}
    b.verify(bindings)
    out.mkdir()
    attempts.to_csv(out/"complete56008_exact_closing_and_failed_attempts.csv",index=False,encoding="utf-8-sig")
    excluded.to_csv(out/"excluded2764_observations_WITH_REASONS.csv",index=False,encoding="utf-8-sig")
    added.to_csv(out/"reclassified362_closing_TO_failed_WITH_HEIGHT.csv",index=False,encoding="utf-8-sig")
    features.to_csv(out/"complete9245_corrected_attempt_context_NO_LABELS.csv",index=False,encoding="utf-8-sig")
    b.write_new(out/"all_context_differences.json",dict(formed_at=b.now(),changes=changes,old_inputs_and_scores_not_rewritten=True))
    windows={}
    full=old.merged(original)
    for fold in old.history.FOLDS:
        train=old.history.training_rows(full,fold["start"],"relay")
        changed_keys={(x["date"],x["code"]) for x in changes}
        f=features.merge(train[["date","code"]],on=["date","code"],validate="one_to_one")
        windows[fold["id"]]=dict(rows=len(f),rule_role_nonempty_coverage=float(f.primary_role.notna().mean()),
            candidates_with_changed_context=sum((x.date,x.code) in changed_keys for x in f.itertuples()),
            same_height_elimination_observed=int(f.same_height_elimination_rate.notna().sum()))
    data=dict(version=b.VERSION,features=features,prepared=original["prepared"],attempts=attempts,bars=original["bars"],sessions=original["sessions"],sources=bindings,
        correction_version=repair.VERSION,source_qualification=original.get("source_qualification"),no_old_model_fit_or_predictions_replaced=True)
    joblib.dump(data,out/"corrected_dataset_RESEARCH_NOT_FITTED.joblib",compress=3)
    report=dict(formed_at=b.now(),status="EXACT_FAILED_ATTEMPT_RECLASSIFICATION_CORRECTION_NOT_NEW_FIT",original58772_rows=len(raw),previous_qualified_rows=len(old_attempts),
        corrected_qualified_rows=len(attempts),reclassified_and_restored=len(added),reclassified_dates=added.signal_date.nunique(),
        raw_reclassified_original_heights=added.original_board_height.value_counts().sort_index().to_dict(),corrected_attempt_heights=added.board_height.value_counts().sort_index().to_dict(),
        exclusions=excluded.qualification_reason.value_counts().to_dict(),source_scope_unknown_dates=sorted(set(raw.signal_date)-set(attempts.attrs["failed_scope_complete_dates"])),
        original55646_assignments_and_heights_exact=True,full_original_candidate_rows=len(features),context_changes=len(changes),
        changed_candidates=len({(x["date"],x["code"]) for x in changes}),changed_fields=pd.Series([x["field"] for x in changes]).value_counts().to_dict(),training_windows=windows,
        original_market_aggregate_features_not_rebuilt=True,market_bomb_rate_may_have_independent_source_semantics=True,
        original_model_results_remain_with_original_inputs_not_reinterpreted_as_corrected_data_results=True,
        new_numerical_hypothesis_or_fit_started=False,actual_new_fits=0,new_formal_forecasts=0,new_mature_results=0,source_bindings=bindings,
        corrected_dataset_sha256=b.sha(out/"corrected_dataset_RESEARCH_NOT_FITTED.joblib"))
    b.write_new(out/"correction_and_impact_report.json",report)
    print({k:v for k,v in report.items() if k!="source_bindings"})


if __name__=="__main__":main()
