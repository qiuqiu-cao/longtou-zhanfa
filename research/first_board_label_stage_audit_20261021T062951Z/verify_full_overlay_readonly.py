"""Verify serialization, all original fields, exact one-case scope and source hashes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
import first_board_corporate_action_overlay as overlay

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
protocol=json.loads((OUT/"COLLECTION_PROTOCOL_BEFORE_REQUESTS.json").read_text("utf-8"))
for path,expected in protocol["protected_hashes"].items():
    assert sha(path)==expected,path
facts=pd.read_csv(ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv",
    dtype={"code":str},float_precision="round_trip")
facts=facts[facts.stage.eq(1)].reset_index(drop=True)
path=OUT/"COMPLETE28503_ORIGINAL_PLUS_PARTIAL_CORRECTION_RESEARCH_ONLY_V2.csv"
new=pd.read_csv(path,dtype={"code":str},float_precision="round_trip")
pd.testing.assert_frame_equal(new[facts.columns],facts,check_exact=True)
assert len(new)==28503 and new[overlay.ORIGINAL].isna().sum()==126
assert new[overlay.CORRECTED].isna().sum()==126
both_known=new[overlay.ORIGINAL].notna() & new[overlay.CORRECTED].notna()
changed=new.loc[both_known & new[overlay.ORIGINAL].ne(new[overlay.CORRECTED])]
assert len(changed)==1 and changed.iloc[0].code=="603214"
assert changed.iloc[0].signal_date=="2024-09-30"
assert changed.iloc[0].label_end=="2024-10-08"
assert changed.iloc[0][overlay.ORIGINAL]==0 and changed.iloc[0][overlay.CORRECTED]==1
assert new.corporate_action_correction_applied.sum()==1
assert new.corporate_action_label_independently_verified.sum()==1
assert "corporate_action_other_labels_independently_verified" not in new.columns
item=json.loads((OUT/"ONE_CONFIRMED_CORPORATE_ACTION_CORRECTION.json").read_text("utf-8"))
overlay.check_evidence(item,OUT)
result=dict(at=datetime.now(timezone.utc).isoformat(),status="PASS_COMPLETE28503_PARTIAL_OVERLAY_READONLY",
    rows=28503,original_columns_prices_labels_exact=True,original_unknown=126,research_unknown=126,
    corrected_labels=1,independently_verified_flags=1,all_other_labels_independently_verified=False,
    protected_hashes_unchanged=True,source_pdf_and_target_report_bound=True,
    model_fits=0,fully_repaired_training_set=False,feature_pipeline_exrights_fixed=False,
    output_sha256=sha(path),overlay_source_sha256=sha(overlay.__file__),targeted_tests="8 passed")
with (OUT/"FULL28503_OVERLAY_READONLY_VERIFICATION.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
print("FULL28503_EXACT_ORIGINAL_FIELDS_UNKNOWN126_ONE_CORRECTION_VERIFIED_NO_FITS")
