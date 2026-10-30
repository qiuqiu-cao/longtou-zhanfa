"""Refresh actual terminal scan state, never interpret old status as a live job."""
from pathlib import Path
from datetime import datetime,timezone
import json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
report=json.loads((OUT/"report.json").read_text("utf-8"))
assert report["status"]=="COMPLETE668_OFFLINE_PREFIX_QUALIFICATION_NO_FITS"
pointer_path=ROOT/"research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
previous=json.loads(pointer_path.read_text("utf-8"))
assert previous["experiment_root"]==str(OUT)
pointer=dict(previous,updated_at=datetime.now(timezone.utc).isoformat(),
    status="OFFLINE668_SCAN_COMPLETE_AWAIT_GEMINI_INDEPENDENT_REVIEW",
    source_scan_not_yet_run=False,active_model_training=False,
    code_execution_terminal=True,process_handle=None,
    source_scan_completed_at=report["at"],source_dates=668,source_rows=35292,
    prefix_candidates=32,source_dates_without_receive_clock=3,source_rows_without_receive_clock=142,
    model_fits=0,external_requests=0,target_labels_read=False,
    report=str(OUT/"report.json"),delivery=str(OUT/"668日除权线索资格_结果.md"),
    prior_label_audit_delivery=str(ROOT/"research/first_board_label_stage_audit_20261021T062951Z/标签核验与除息误标修复_结果.md"),
    pending_peer_review=str(OUT/"gemini/GEMINI_OFFLINE_ACTION_QUALIFICATION_REVIEW.json"),
    peer_request="Sent to actual authorized Antigravity conversation;no additional jobs requested",
    no_live_training=True,no_failed_model_retry=True,goal_60_achieved=False)
pointer_path.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding="utf-8")
print("ACTUAL_TERMINAL_SCAN_STATE_UPDATED_NO_MODEL_FITS")
