"""Archive a completed fixed experiment and advance the research pointer only."""
import json
from pathlib import Path
from datetime import datetime
import first_board_leader_model as base

ROOT = base.ROOT
OUT = Path(__file__).resolve().parent
OLD = ROOT / "research/gemini_codex_joint_long_high_20261021T022449Z"
POINTER = ROOT / "research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
report = json.loads((OLD / "report.json").read_text("utf-8"))
protocol = json.loads((OLD / "JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
verify = json.loads((OLD / "JOINT_SAVED_RESULT_VERIFICATION.json").read_text("utf-8"))
correction = json.loads((OLD / "gemini/GEMINI_LONG_HIGH_SETTLEMENT_CORRECTION.json").read_text("utf-8"))
assert report["selection"]["selected"] == correction["corrections"]["validation_selection_status"]["exact_selected_arm"] == "LONG_HIGH120_63"
assert base.sha(ROOT / "first_board_joint_long_high_model.py") == protocol["source_sha256"]
assert verify["status"] == "PASS_SAVED_VALIDATION_REPLAY"
assert not report["goal_60_achieved"] and not report["production_deployment_allowed"]
core = [p for p in OLD.rglob("*_BEFORE_FIT.json") if p.parent.name == "models"]
calibration = list(OLD.rglob("*_calibration_receipt.json"))
assert len(core) == 32 and len(calibration) == 16
artifacts = [OLD / name for name in ["JOINT_FROZEN_PROTOCOL.json", "report.json", "JOINT_SELECTION_BEFORE_EVALUATION.json",
    "JOINT_SAVED_RESULT_VERIFICATION.json", "COMPLETE4724_BOTH_ARMS_WITH_FROZEN_BASE.csv", "COMPLETE117_DAILY_PAIRED_SELECTIONS_ALL_ARMS.csv",
    "EVALUATION_ALL13149_SELECTED_ARM_WITH_FROZEN_BASE.csv", "EVALUATION286_DAILY_PAIRED_SELECTIONS.csv", "evaluation/SAME_STOCK_SECOND_RECHECK.csv",
    "gemini/GEMINI_LONG_HIGH_FINAL_SETTLEMENT.md", "gemini/GEMINI_LONG_HIGH_FINAL_SETTLEMENT.json",
    "gemini/GEMINI_LONG_HIGH_SETTLEMENT_CORRECTION.md", "gemini/GEMINI_LONG_HIGH_SETTLEMENT_CORRECTION.json", "CODEX_PEER_SETTLEMENT_REVIEW.md",
    "联合训练_长周期前高_结果.md"]]
complete = dict(completed_at=base.now(), status="JOINT_FIXED_BATCH_SETTLED_TARGET_FAILED", model_results_completed_at=report["at"],
    model_results_within_original_deadline=datetime.fromisoformat(report["at"]) < datetime.fromisoformat(protocol["batch_deadline_utc"]),
    documentation_followthrough_after_model_results=True, validation_selected="LONG_HIGH120_63", monthly_heads=14, validation_heads=2,
    actual_hgb_core_fits=len(core), actual_temporal_calibration_fits=len(calibration), baseline_refits=0, second_stage_refits=0,
    goal_60_achieved=False, deployment_allowed=False, actual_trades=0, new_forward_results=0,
    actual_peer="Antigravity Gemini 3.8 Flash High", peer_independent_settlement_and_correction_read=True,
    evaluation=report["evaluation"]["all"], same_stock_second_recheck=report["same_stock_second_recheck"],
    protected_original_labels_and_full_pool_verified=True, no_failed_arm_retry=True,
    next_scope="User-directed forum six-factor design and target-free event-path qualification, bounded source/label audit alongside;no new fit in this design batch",
    artifact_sha256={str(p.relative_to(OLD)):base.sha(p) for p in artifacts})
with (OLD / "completion.json").open("x", encoding="utf-8") as f:
    json.dump(base.clean(complete), f, ensure_ascii=False, indent=2, allow_nan=False)
previous = json.loads(POINTER.read_text("utf-8"))
base.write_json(OUT / "PREVIOUS_JOINT_EXPERIMENT_POINTER_BEFORE_DESIGN.json", previous)
base.write_json(OLD / "MONITOR_LAST_NOTIFIED_RESULT.json", dict(at=base.now(), status="FAILED_TARGET_RESULT_AND_PEER_CORRECTION_REPORTED",
    report_sha256=base.sha(OLD / "report.json"), completion_sha256=base.sha(OLD / "completion.json"), goal_60_achieved=False))
base.write_json(POINTER, dict(updated_at=base.now(), status="FORUM_FACTOR_DESIGN_QUALIFICATION_AWAIT_PEER_CONSENSUS_NO_FITS",
    experiment_root=str(OUT.resolve()), prior_batch=str(OLD.resolve()), prior_batch_settled_no_retries=True,
    batch_deadline_utc="2026-10-21T03:21:39+00:00", fits=0, goal_60_achieved=False,
    qualification=str(OUT / "EVENT_PATH_QUALIFICATION.json"), model_fits_authorized_in_this_batch=False,
    proposed_arms="Only pending design; original62+sealed7 versus original62+sealed7+probe7. Not frozen.",
    delivery=str(OUT / "股吧经验与六类因子候选.md"), peer_review=str(OUT / "gemini/GEMINI_FORUM_FACTOR_DESIGN.md")))
print(json.dumps(dict(old_status=complete["status"], core_fits=len(core), calibrations=len(calibration), pointer_advanced=True)))
