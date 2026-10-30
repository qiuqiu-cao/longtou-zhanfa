"""Correct stale coordinator status from existing terminal model artifacts; never fit."""
from pathlib import Path
import json
import first_board_leader_model as base
OUT = Path(__file__).resolve().parent
POINTER = base.ROOT / "research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
before = json.loads(POINTER.read_text("utf-8"))
report = json.loads((OUT / "report.json").read_text("utf-8"))
verification = json.loads((OUT / "JOINT_SAVED_RESULT_VERIFICATION.json").read_text("utf-8"))
assert report["status"] == "BOTH_FIXED_ARMS_REJECTED" and report["monthly_fits"] == 0
assert report["selection"]["selected"] is None
assert verification["status"] == "PASS_SAVED_VALIDATION_REPLAY" and verification["model_fits"] == 0
with (OUT / "STALE_POINTER_BEFORE_TERMINAL_RECONCILIATION.json").open("x",encoding="utf-8") as f:
    json.dump(before,f,ensure_ascii=False,indent=2)
after = dict(before, updated_at=base.now(), status="BOTH_EVENT_PATH_ARMS_REJECTED_AWAIT_PEER_SETTLEMENT_NO_LIVE_TRAINING",
    agreed_at=json.loads((OUT / "JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))["agreed_at"],
    actual_validation_heads=2, actual_core_fits=4, actual_calibrator_fits=2, monthly_fits=0, fits=4,
    selected=None, goal_60_achieved=False, baseline_refits=0, second_stage_refits=0,
    reports_completed_at=report["at"], code_execution_terminal=True,
    primary_hits={"BASE":52,**{k:v["primary"]["hits"] for k,v in report["selection"]["reports"].items()}},
    non_one_price_hits={"BASE":48,**{k:v["non_one_price"]["hits"] for k,v in report["selection"]["reports"].items()}},
    report=str(OUT / "report.json"),delivery=str(OUT / "联合训练_涨停回调与试盘路径_结果.md"),
    read_only_verification=str(OUT / "JOINT_SAVED_RESULT_VERIFICATION.json"))
base.write_json(POINTER,after)
print(json.dumps(dict(status=after["status"],primary_hits=after["primary_hits"],monthly_fits=0)))
