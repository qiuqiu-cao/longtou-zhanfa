"""Generate a new, explicitly partial correction layer, never a new fitted model."""
from pathlib import Path
import json
import pandas as pd
import first_board_leader_model as base
import first_board_corporate_action_overlay as overlay
OUT=Path(__file__).resolve().parent
FACTS=base.ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
destination=OUT/"COMPLETE28503_ORIGINAL_PLUS_PARTIAL_CORRECTION_RESEARCH_ONLY.csv"
if destination.exists():raise FileExistsError("No overwritten research dataset")
before=base.sha(FACTS)
facts=pd.read_csv(FACTS,dtype={"code":str},float_precision="round_trip")
facts=facts[facts.stage.eq(1)].reset_index(drop=True)
correction=json.loads((OUT/"ONE_CONFIRMED_CORPORATE_ACTION_CORRECTION.json").read_text("utf-8"))
result=overlay.build_research_overlay(facts,[correction],OUT)
assert len(result)==28503 and result[overlay.CORRECTED].isna().sum()==126
result.to_csv(destination,index=False,encoding="utf-8-sig")
assert base.sha(FACTS)==before
base.write_json(OUT/"PARTIAL_CORRECTION_OVERLAY_RECEIPT.json",dict(at=base.now(),original_rows=len(facts),
    source_sha256=before,original_columns_targets_prices_unchanged=True,original_unknown126_retained=True,
    corrected_research_label_column=overlay.CORRECTED,correction_count=1,
    complete_corporate_action_audit=False,model_fits=0,old_predictions_or_metrics_rewritten=False,
    feature_pipeline_exrights_fixed=False,
    output_sha256=base.sha(destination),overlay_source_sha256=base.sha(overlay.__file__),
    limitation="This one-case label layer is not a fully repaired training set. No failed arm retraining or claimed win-rate improvement."))
print("PARTIAL_OVERLAY_FULL28503_ORIGINAL_UNCHANGED_UNKNOWN126_ONE_CORRECTION_NO_FITS")
