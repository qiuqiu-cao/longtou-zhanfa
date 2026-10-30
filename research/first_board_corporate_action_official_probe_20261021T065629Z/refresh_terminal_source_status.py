"""Record the real terminal source collection and pending actual-peer review."""
from pathlib import Path
from datetime import datetime,timezone
import json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sources=json.loads((OUT/"ALL_THREE_CASE_SOURCE_RECEIPTS.json").read_text("utf-8"))
assert sources["actual_direct_http_requests"]==5
pointer_path=ROOT/"research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
previous=json.loads(pointer_path.read_text("utf-8"))
assert previous["experiment_root"]==str(OUT)
with (OUT/"POINTER_BEFORE_TERMINAL_COLLECTION_UPDATE.json").open("x",encoding="utf-8") as stream:
    json.dump(previous,stream,ensure_ascii=False,indent=2)
current=dict(previous,updated_at=datetime.now(timezone.utc).isoformat(),
    status="OFFICIAL_THREE_CASE_COLLECTION_TERMINAL_AWAIT_PEER_REVIEW",
    code_execution_terminal=True,process_handle=None,prior_exec_session=89109,prior_exec_exit_code=0,
    source_collection_completed_at=sources["at"],actual_direct_http_requests=5,pdfs_captured=2,
    official_field_cases_confirmed=2,unknown_implementation_cases=1,confirmed_first_cohort_omissions=1,
    target_corrections_in_this_batch=0,active_model_training=False,model_fits=0,goal_60_achieved=False,
    report=str(OUT/"THREE_CASE_FIELD_AND_PRICE_QUALIFICATION.json"),delivery=str(OUT/"三例公告核验与首板漏入_结果.md"),
    pending_peer_review=str(OUT/"gemini/GEMINI_OFFICIAL_THREE_CASE_REVIEW.json"),
    next_scope_agreement_expected=str(OUT/"gemini/NEXT_STRUCTURED_SOURCE_SCOPE_AGREEMENT.json"),
    next_scope_is_frozen=False,no_failed_model_retry=True)
pointer_path.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding="utf-8")
print("SOURCE_COLLECTION_TERMINAL_5_HTTP_0_FIT_1_COHORT_GAP_NO_TARGET_EDITS")
