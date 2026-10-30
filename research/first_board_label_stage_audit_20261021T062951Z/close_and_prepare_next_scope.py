"""Finalize the verified audit and prepare an agreed, outcome-free offline scope."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json
import hashlib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text("utf-8"))
def write_new(path,data):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(data,stream,ensure_ascii=False,indent=2)

verification=read(OUT/"FULL28503_OVERLAY_READONLY_VERIFICATION.json")
peer=read(OUT/"gemini/GEMINI_OVERLAY_AND_NEXT_SCOPE_REVIEW.json")
assert verification["status"]=="PASS_COMPLETE28503_PARTIAL_OVERLAY_READONLY"
assert peer["v2_overlay_audit"]["status"]=="APPROVED_AS_SEPARATE_RESEARCH_LAYER"
assert peer["next_batch_proposal_verdict"]["status"]=="ACCEPTED"
actual_now=datetime.now(timezone.utc)
names=["COLLECTION_PROTOCOL_BEFORE_REQUESTS.json","COLLECTION_COMPLETION.json","BOUNDED_STAGE_COMPARISON.json",
    "ONE_CONFIRMED_CORPORATE_ACTION_CORRECTION.json","CORPORATE_ACTION_CORRECTIONS_SEPARATE_LAYER.jsonl",
    "OFFICIAL_DIVIDEND_PDF_RECEIPT.json","PARTIAL_CORRECTION_OVERLAY_RECEIPT_V2.json",
    "COMPLETE28503_ORIGINAL_PLUS_PARTIAL_CORRECTION_RESEARCH_ONLY_V2.csv",
    "FULL28503_OVERLAY_READONLY_VERIFICATION.json","标签核验与除息误标修复_结果.md",
    "gemini/GEMINI_LIMITED_STAGE_AUDIT.md","gemini/GEMINI_LIMITED_STAGE_AUDIT.json",
    "gemini/GEMINI_OVERLAY_AND_NEXT_SCOPE_REVIEW.md","gemini/GEMINI_OVERLAY_AND_NEXT_SCOPE_REVIEW.json"]
completion=dict(completed_at=actual_now.isoformat(),status="BOUNDED_STAGE_AUDIT_SETTLED_ONE_ERROR_PARTIALLY_CORRECTED",
    batch_started_at="2026-10-21T06:29:51+00:00",deadline_utc="2026-10-21T06:59:51+00:00",
    actual_peer="Antigravity Gemini 3.8 Flash High",actual_peer_case_and_overlay_review_read=True,
    requested_dates=21,qualified_dates=21,source_second_reports=636,original_scope_rows=1981,
    corroborated_positives=462,unresolved_stage_conflicts=36,confirmed_corporate_action_errors=1,
    no_positive_matching_reports_unknown=1482,source_reports_outside_original_first_cohort=173,
    outcome_seen_diagnostic_dates=15,fixed_chronological_dates=6,
    separate_research_overlay_rows=28503,original_unknown_labels_preserved=126,corrected_research_labels=1,
    original_labels_rewritten=0,complete_label_verification=False,complete_corporate_action_repair=False,
    feature_pipeline_exrights_fixed=False,model_fits=0,actual_trades=0,production_writes=0,
    goal_60_achieved=False,old_failures_retrained=False,
    artifact_sha256={name:sha(OUT/name) for name in names})
write_new(OUT/"completion.json",completion)
next_root=ROOT/"research"/actual_now.strftime("first_board_corporate_action_qualification_%Y%m%dT%H%M%SZ")
next_root.mkdir(exist_ok=False)
source=ROOT/"research/first_board_joint_design_20261020T185417Z/ALL668_SOURCE_RECEIPTS.json"
protocol=dict(frozen_at=actual_now.isoformat(),deadline_utc=(actual_now+timedelta(minutes=30)).isoformat(),
    status="JOINTLY_ACCEPTED_OFFLINE_SOURCE_QUALIFICATION_ONLY",budget_minutes=30,
    current_owner="Codex scans;actual Antigravity Gemini independently reviews",actual_peer="Gemini 3.8 Flash High",
    peer_review=str(OUT/"gemini/GEMINI_OVERLAY_AND_NEXT_SCOPE_REVIEW.json"),
    peer_review_sha256=sha(OUT/"gemini/GEMINI_OVERLAY_AND_NEXT_SCOPE_REVIEW.json"),
    source_receipts=str(source),source_receipts_sha256=sha(source),
    source_dates=668,source_rows=35292,source_scope="All existing dated PidType1 source records,including20pct/outside original cohort",
    selection="All668 sources and every row;normalizedname.strip.upper begins XD|XR|DR;no outcome-based selection",
    provenance="Rawname,code,date,SHA256,JSONpointer and exact UTF8 byte span; preserve all unmatching rows",
    scan_source=str(ROOT/"first_board_corporate_action_scan.py"),scan_source_sha256=sha(ROOT/"first_board_corporate_action_scan.py"),
    model_fits_allowed=0,external_requests_allowed=0,target_labels_allowed=False,
    original_target_writes_allowed=False,production_writes_allowed=False,
    absence_of_prefix_never_certifies_no_action=True,
    late_capture_not_original_signal_time=True,
    no_official_action_claim_from_prefix=True,
    next_official_sampling_requires_separately_frozen_scope=True)
write_new(next_root/"JOINT_FROZEN_PROTOCOL.json",protocol)
pointer_path=ROOT/"research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
write_new(next_root/"PREVIOUS_JOINT_EXPERIMENT_POINTER.json",read(pointer_path))
pointer=dict(updated_at=actual_now.isoformat(),status="OFFLINE668_SOURCE_SCAN_AUTHORIZED_NOT_STARTED",
    experiment_root=str(next_root),protocol=str(next_root/"JOINT_FROZEN_PROTOCOL.json"),
    prior_batch_settled_no_retries=True,prior_audit_root=str(OUT),prior_audit_completion=str(OUT/"completion.json"),
    prior_event_batch_settled=True,actual_owner="Codex",actual_peer="Antigravity Gemini 3.8 Flash High",
    model_fits=0,external_requests=0,active_model_training=False,goal_60_achieved=False,
    process_handle=None,source_scan_not_yet_run=True,deadline_utc=protocol["deadline_utc"])
pointer_path.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding="utf-8")
print(str(next_root))
