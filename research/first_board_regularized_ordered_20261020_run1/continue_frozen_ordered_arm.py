"""Complete the already-frozen, previously unfitted ordered arm exactly once."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import first_board_regularized_ordered as experiment

OUT=Path(__file__).resolve().parent

def run():
    old=json.loads((OUT/"protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
    assert base.sha(experiment.__file__)==old["source_sha256"]
    assert old["deferred_arm"]["arm"]=="CHRONO_ORDERED"
    assert old["deferred_arm"]["params"]==experiment.CAT_PARAMS
    for path,digest in {**old["inputs"],**old["protected"],**old["raw_source_hashes"]}.items():assert base.sha(path)==digest,path
    assert not list((OUT/"models").glob("*CHRONO_ORDERED*")),"Ordered arm already attempted;do not refit"
    receipt=OUT/"ordered_extension_BEFORE_ANY_ORDERED_FIT.json"
    if receipt.exists():raise FileExistsError("Preserve ordered attempt")
    acquisition=json.loads((experiment.DEP/"wheel_acquisition.json").read_text("utf-8"))
    assert acquisition["import_status"]=="PASS" and acquisition["imported_version"]=="1.2.10"
    experiment.cat_class()
    import catboost._catboost as native
    inputs=[experiment.DEP/"wheel_acquisition.json",experiment.DEP/"catboost-1.2.10-cp313-cp313-win_amd64.whl",
        experiment.DEP/"runtime/catboost/core.py",Path(native.__file__),
        OUT/"protocol_BEFORE_ANY_FIT.json",OUT/"report.json",OUT/"selection_BEFORE_EVALUATION.json",OUT/"verification.json",
        OUT/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_SCREEN.csv",OUT/"validation_REGULARIZED_LINEAR_ALL_BEFORE_METRICS.csv"]
    extension=dict(formed_at=base.now(),task="Finish frozen previouslyunfittedsecondarm;notnewthirdarmorhyperparameterretry",
        parent_protocol_sha256=base.sha(OUT/"protocol_BEFORE_ANY_FIT.json"),params=experiment.CAT_PARAMS,
        unchanged_original_fields=experiment.FIELDS,original_deadline="2026-10-20T15:14:53+00:00",
        baseline_and_rule="Unchanged original52/117screened;need>52hits,>=48non-onepricehits,>=95%datecoverage;linearfailed40/117isretained",
        no_linear_refit=True,no_production_changes=True,inputs={str(p.resolve()):base.sha(p) for p in inputs},
        runner_sha256=base.sha(__file__))
    base.write_json(receipt,extension)
    rows=pd.read_csv(OUT/"ALL_FIRST_BOARD_UNCHANGED_FEATURES_LABELS_SCREEN.csv",dtype={"code":str},float_precision="round_trip")
    train=rows[rows[stage.TARGET].notna()&rows.signal_date.lt("2025-01-01")&rows.label_end.lt("2025-01-01")]
    val=rows[rows.signal_date.between("2025-01-01","2025-06-30")]
    bundle=experiment.fit_head(train,"CHRONO_ORDERED",OUT,"validation_CHRONO_ORDERED","2025-01-01")
    predictions=experiment.predict(bundle,val)
    predictions.to_csv(OUT/"validation_CHRONO_ORDERED_ALL_BEFORE_METRICS.csv",index=False,encoding="utf-8-sig")
    primary=risk.screen_metric(predictions);non=risk.screen_metric(predictions[predictions.one_price.eq(0)])
    passed=bool(primary["hits"]>52 and non["hits"]>=48 and primary["decision_date_coverage"]>=.95)
    decision=dict(formed_at=base.now(),arm="CHRONO_ORDERED",primary=primary,non_one_price=non,
        unfiltered_diagnostic=stage.metric(predictions),pass_validation=passed,
        monthly_evaluation_status="PENDING_FIXED_EVALUATION" if passed else "REJECTED_NO_MONTHLY_FITS")
    base.write_json(OUT/"ordered_selection_BEFORE_EVALUATION.json",decision)
    report=dict(formed_at=base.now(),ordered=decision,linear_preserved="40/117=34.19%,no refit",
        goal_60_passed=False,ordered_core_fit_count=len(list((OUT/"models").glob("*CHRONO_ORDERED*_COMPLETED.json"))),
        total_core_fit_count=len(list((OUT/"models").glob("*_COMPLETED.json"))),
        status="ORDERED_VALIDATION_PASSED_EVALUATION_PENDING" if passed else "BOTH_FROZEN_ARMS_REJECTED_NO_MONTHLY_FITS",
        new_forward_results=0,actual_trades=0)
    for path,digest in extension["inputs"].items():assert base.sha(path)==digest,path
    base.write_json(OUT/"combined_report_APPEND_ONLY.json",report)
    print(json.dumps(base.clean(report),ensure_ascii=True,indent=2),flush=True)

if __name__=="__main__":run()
