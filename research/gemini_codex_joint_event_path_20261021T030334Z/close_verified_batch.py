"""Record a terminal fixed batch; never fit or replay a model."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent

def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))

report = read("report.json")
verification = read("JOINT_SAVED_RESULT_VERIFICATION.json")
clock = read("gemini/GEMINI_EVENT_PATH_SETTLEMENT_RECEIPT_CLOCK_FIX.json")
assert report["status"] == "BOTH_FIXED_ARMS_REJECTED"
assert report["selection"]["selected"] is None
assert verification["status"] == "PASS_SAVED_VALIDATION_REPLAY"
assert clock["fit_counts_clarification"]["batch_actual_core_fits"] == 4
assert clock["fit_counts_clarification"]["batch_actual_calibrator_fits"] == 2
names = ["JOINT_FROZEN_PROTOCOL.json", "report.json", "JOINT_SELECTION_BEFORE_EVALUATION.json",
    "JOINT_SAVED_RESULT_VERIFICATION.json", "COMPLETE4724_BOTH_ARMS_WITH_FROZEN_BASE.csv",
    "COMPLETE117_DAILY_PAIRED_SELECTIONS_ALL_ARMS.csv", "联合训练_涨停回调与试盘路径_结果.md",
    "gemini/GEMINI_EVENT_PATH_FINAL_SETTLEMENT.md", "gemini/GEMINI_EVENT_PATH_FINAL_SETTLEMENT.json",
    "gemini/GEMINI_EVENT_PATH_SETTLEMENT_RECEIPT_CLOCK_FIX.md",
    "gemini/GEMINI_EVENT_PATH_SETTLEMENT_RECEIPT_CLOCK_FIX.json", "CODEX_PEER_SETTLEMENT_REVIEW.md"]
completion = dict(completed_at=datetime.now(timezone.utc).isoformat(),
    status="JOINT_FIXED_BATCH_SETTLED_BOTH_ARMS_REJECTED",
    model_results_completed_at=report["at"], batch_deadline_utc="2026-10-21T03:33:34+00:00",
    model_results_within_original_deadline=True,
    actual_peer="Antigravity Gemini 3.8 Flash High", actual_peer_settlement_and_clock_correction_read=True,
    validation_heads=2, actual_hgb_core_fits=4, actual_temporal_calibration_fits=2,
    baseline_refits=0, second_stage_refits=0, monthly_fits=0,
    selected=None, goal_60_achieved=False, deployment_allowed=False,
    actual_trades=0, new_forward_results=0, no_live_training=True,
    full28503_source_targets_and_screen_preserved=True, original_unknown126_retained=True,
    no_failed_arm_retry=True,
    primary_hits={"BASE":52,"EVENT_SEALED_PATH_67":48,"EVENT_DUAL_PATH_72":46},
    non_one_price_hits={"BASE":48,"EVENT_SEALED_PATH_67":43,"EVENT_DUAL_PATH_72":40},
    complete_validation_dates=117, complete_validation_rows=4724,
    next_scope="Bounded source/label audit, separate correction layer; no new model fit",
    artifact_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names})
target=OUT/"completion.json"
with target.open("x",encoding="utf-8") as stream:
    json.dump(completion,stream,ensure_ascii=False,indent=2)
print("EVENT_BATCH_SETTLED_4_CORE_2_CAL_0_MONTHLY_BOTH_REJECTED_NO_FITS")
