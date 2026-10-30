"""Remember the material results reported in this heartbeat, not routine polling."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat()
summary="117日验证两臂41.03%/39.32%低于BASE44.44%，不采用；研究标签层独立更正1条除息误标，32条名称线索待公告核验；60%未达。"
items=[
    (ROOT/"research/gemini_codex_joint_event_path_20261021T030334Z","report.json"),
    (ROOT/"research/first_board_label_stage_audit_20261021T062951Z","completion.json"),
    (OUT,"report.json")]
for folder,report in items:
    marker=folder/"MONITOR_LAST_NOTIFIED_RESULT.json"
    if marker.exists():raise FileExistsError("Do not overwrite an existing notification receipt")
    marker.write_text(json.dumps(dict(at=now,automation_id="gemini",result_reported_in_current_thread=True,
        result_report=str(folder/report),root_report_sha256=sha(folder/report),
        notification_summary=summary,same_result_should_remain_quiet=True,goal_60_achieved=False,
        ongoing_collaboration="Next bounded official sample scope is under actual peer review;not a live model job"),
        ensure_ascii=False,indent=2),encoding="utf-8")
print("THREE_MATERIAL_RESULT_MARKERS_RECORDED_AVOID_DUPLICATE_NOTIFICATIONS")
