"""Preserve a metadata-only crash and freeze its narrow repair before resume."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
protocol=json.loads((OUT/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
original=OUT/"scan_source_before_missing_clock_fix.py"
assert sha(original)==protocol["scan_source_sha256"]
receipts=json.loads(Path(protocol["source_receipts"]).read_text("utf-8"))
missing=[r["date"] for r in receipts if not r.get("received_at")]
assert len(missing)==3
data=dict(at=datetime.now(timezone.utc).isoformat(),reason="3 original reused probes have no receipt received_at key",
    before_source_sha256=sha(original),after_source_sha256=sha(ROOT/"first_board_corporate_action_scan.py"),
    failure_receipt_sha256=sha(OUT/"SCAN_FAILURE_RECEIPT.json"),missing_clock_dates=missing,
    fix="Missing receive clock stays null;explicit clock_known flag false;retain all rows and sources",
    source_or_selection_changes=False,original_protocol_unchanged=True,original_deadline_unchanged=True,
    model_fits=0,external_requests=0,no_automatic_retry=True,
    manual_technical_resume_of_offline_audit_only=True)
with (OUT/"TECHNICAL_CLOCK_AMENDMENT_BEFORE_RESUME.json").open("x",encoding="utf-8") as stream:
    json.dump(data,stream,ensure_ascii=False,indent=2)
print("MISSING_CLOCK_LEFT_UNKNOWN_NARROW_OFFLINE_REPAIR_FROZEN")
