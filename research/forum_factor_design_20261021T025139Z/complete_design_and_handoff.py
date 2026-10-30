"""Record completed design artifacts and advance the existing collaboration pointer."""
import json
from pathlib import Path
import first_board_leader_model as base
ROOT = base.ROOT
OUT = Path(__file__).resolve().parent
NEW = ROOT / "research/gemini_codex_joint_event_path_20261021T030334Z"
POINTER = ROOT / "research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
assert json.loads((NEW / "JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))["status"] == "AWAIT_GEMINI_ACK_NO_FITS"
artifacts = [OUT/name for name in ["sources.json", "股吧经验与六类因子候选.md", "共同方案与资格结论.md",
    "EVENT_PATH_QUALIFICATION.json", "COMPLETE28503_EVENT_PATH_FEATURES_NO_TARGETS.csv",
    "EVENT_FEATURE_EXISTING_TRANSFORM_AUDIT.json", "已有变量变换审查与候选修订.md",
    "LABEL_REPORT_CROSSCHECK.json", "LABEL_CROSSCHECK_SCOPE_CORRECTION.md", "COMPLETE28503_LABEL_REPORT_CROSSCHECK.csv",
    "gemini/GEMINI_FORUM_FACTOR_DESIGN.md", "gemini/GEMINI_FORUM_FACTOR_DESIGN_RESPONSE.md",
    "gemini/FINAL_CANDIDATE_AGREEMENT.md", "gemini/FINAL_CANDIDATE_AGREEMENT.json", "ACTUAL_GEMINI_DESIGN_REVIEW.jpg"]]
completion = dict(completed_at=base.now(), status="JOINT_FORUM_DESIGN_QUALIFICATION_COMPLETE_NO_FITS",
    fits=0, sources_actually_read_by="Codex web primary pages", peer_review="Actual Antigravity Gemini reviewed locally extracted material,sourcecode,wholepool,tests,and proposed independent differences",
    primary_articles=3, requested_factor_groups=6, complete_rows=28503, signal_dates=668, candidate_source_variables=14,
    final_new_model_variables=10, redundant_age_presence_not_added_to_models=True, tests_passed=6, full_prefixes_verified=[15354,20229],
    target_rewrites=0, original_unknown126_preserved=True, source_crosscheck_limited_to_first_board_scope=True,
    goal_60_achieved=False, production_deployment_allowed=False, actual_trades=0,
    final_candidate_arms={"EVENT_SEALED_PATH_67":"Gemini", "EVENT_DUAL_PATH_72":"Codex"},
    next_numerical_batch=str(NEW.resolve()), artifact_sha256={str(p.relative_to(OUT)):base.sha(p) for p in artifacts})
with (OUT/"completion.json").open("x",encoding="utf-8") as f:
    json.dump(completion,f,ensure_ascii=False,indent=2)
base.write_json(OUT/"MONITOR_LAST_NOTIFIED_RESULT.json",dict(at=base.now(),status="DESIGN_AND_QUALIFICATION_RESULT_REPORTED",
    completion_sha256=base.sha(OUT/"completion.json"), fits=0, goal_60_achieved=False))
base.write_json(NEW/"PREVIOUS_JOINT_EXPERIMENT_POINTER.json",json.loads(POINTER.read_text("utf-8")))
base.write_json(POINTER,dict(updated_at=base.now(),status="EVENT_PATH_PROTOCOL_AWAIT_PEER_ACK_NO_FITS",experiment_root=str(NEW.resolve()),
    design_root=str(OUT.resolve()),prior_batch_settled_no_retries=True,batch_deadline_utc="2026-10-21T03:33:34+00:00",
    proposed_arms={"EVENT_SEALED_PATH_67":"Gemini","EVENT_DUAL_PATH_72":"Codex"}, fits=0,goal_60_achieved=False,
    protocol=str(NEW/"JOINT_FROZEN_PROTOCOL.json"),preflight=str(NEW/"CODEX_PREFLIGHT_NO_FITS.json")))
print("DESIGN_COMPLETE_NO_FITS; NEW_NUMERICAL_PROTOCOL_PENDING_ACTUAL_PEER_ACK")
