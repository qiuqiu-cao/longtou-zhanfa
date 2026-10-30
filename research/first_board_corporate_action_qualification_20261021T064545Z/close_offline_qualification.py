"""Close a jointly reviewed offline audit and preserve the next-scope disagreement."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def read(path):return json.loads(Path(path).read_text("utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_new(path,data):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(data,stream,ensure_ascii=False,indent=2)
report=read(OUT/"report.json")
peer=read(OUT/"gemini/GEMINI_OFFLINE_ACTION_QUALIFICATION_REVIEW.json")
assert peer["qualification_status"]=="OFFLINE_SCAN_AUDIT_PASSED"
assert report["candidate_rows"]==peer["scanned_source_metrics"]["candidate_records"]==32
assert report["source_rows"]==peer["scanned_source_metrics"]["total_source_records"]==35292
assert report["source_rows_without_receive_clock"]==peer["technical_amendment_audit"]["missing_clock_rows"]==142
names=["JOINT_FROZEN_PROTOCOL.json","report.json","ALL35292_SOURCE_NAMES_AND_PROVENANCE.csv",
    "XD_DR_XR_CANDIDATE_ACTION_LEDGER.csv","COMPLETE668_SOURCE_DATE_COVERAGE.csv",
    "SCAN_FAILURE_RECEIPT.json","scan_source_before_missing_clock_fix.py",
    "TECHNICAL_CLOCK_AMENDMENT_BEFORE_RESUME.json","668日除权线索资格_结果.md",
    "gemini/GEMINI_OFFLINE_ACTION_QUALIFICATION_REVIEW.md",
    "gemini/GEMINI_OFFLINE_ACTION_QUALIFICATION_REVIEW.json","CODEX_PEER_REVIEW_AND_NEXT_PROPOSAL.md"]
actual_now=datetime.now(timezone.utc).isoformat()
completion=dict(completed_at=actual_now,status="JOINT_OFFLINE_SOURCE_QUALIFICATION_SETTLED",
    source_scan_completed_at=report["at"],actual_peer="Antigravity Gemini 3.8 Flash High",
    actual_peer_independent_review_read=True,source_dates=668,source_rows=35292,
    candidate_rows=32,candidate_dates=28,candidate_codes=32,prefix_counts=report["prefix_counts"],
    source_dates_missing_receive_clock=3,source_rows_missing_receive_clock=142,
    first_technical_failure_preserved=True,manual_metadata_only_resume=True,
    full_source_date_hash_and_name_byte_spans_verified=True,target_labels_read=False,
    new_official_actions_verified=0,model_fits=0,external_requests=0,production_writes=0,
    original_targets_rewritten=0,complete_action_database=False,goal_60_achieved=False,
    numerical_training_arms_approved=[],
    next_scope_status="Codex proposes3 official cases instead of Gemini8;actual reply requested,not yet frozen",
    artifact_sha256={name:sha(OUT/name) for name in names})
write_new(OUT/"completion.json",completion)
pointer_path=ROOT/"research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
previous=read(pointer_path)
assert previous["experiment_root"]==str(OUT)
write_new(OUT/"POINTER_BEFORE_SETTLEMENT.json",previous)
pointer=dict(previous,updated_at=actual_now,status="SOURCE_QUALIFICATION_SETTLED_NEXT_SCOPE_AGREEMENT_PENDING",
    completion=str(OUT/"completion.json"),pending_peer_review=None,
    actual_peer_review=str(OUT/"gemini/GEMINI_OFFLINE_ACTION_QUALIFICATION_REVIEW.json"),
    next_scope_proposal=str(OUT/"CODEX_PEER_REVIEW_AND_NEXT_PROPOSAL.md"),
    next_scope_agreement_expected=str(OUT/"gemini/NEXT_BOUNDED_OFFICIAL_SCOPE_AGREEMENT.json"),
    next_scope_is_frozen=False,training_arms={},active_model_training=False,
    no_live_training=True,code_execution_terminal=True,process_handle=None,
    current_research_goal_still_active=True,goal_60_achieved=False)
pointer_path.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding="utf-8")
print("JOINT_OFFLINE_QUALIFICATION_SETTLED_NEXT_OFFICIAL_SCOPE_NOT_FROZEN_NO_FITS")
