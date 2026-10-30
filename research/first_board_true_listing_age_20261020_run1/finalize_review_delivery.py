"""Bind the received, independent browser review without altering frozen results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
AUDIT = RUN / "gemini4_review"
manifest = json.loads((AUDIT / "UPLOAD_MANIFEST.json").read_text("utf-8"))
for item in manifest["files"] + [manifest["packet"]]:
    path = ROOT / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
expanded = (AUDIT / "GEMINI4_REVIEW_WITH_TOOL_EXECUTION.txt").read_text("utf-8")
for basename in ("AUDIT_COMPLETE4724_SCORE_COMPARISON_WITH_UNKNOWN.csv", "INDEPENDENT_ALL117_DAILY_PAIRED_SELECTIONS.csv"):
    assert hashlib.sha256((RUN / basename).read_bytes()).hexdigest() in expanded
for statement in ("c_hits=48/117", "b_hits=52/117", "c_hits=43/117", "b_hits=48/117", "known labels: 4714", "unknown: 10"):
    assert statement in expanded, statement
supplement = (AUDIT / "GEMINI4_REVIEW_SCOPE_CORRECTION.txt").read_text("utf-8")
assert "同意淘汰" in supplement and "机制假设" in supplement and "60%目标未达到" in supplement
assert (AUDIT / "GEMINI4_REVIEW_FINAL.jpg").read_bytes().startswith(bytes.fromhex("ffd8ff"))
report = json.loads((RUN / "report.json").read_text("utf-8"))
assert report["reviewer"] == "Gemini4_PENDING_REVIEW" and not report["goal_60_passed"]
receipt = dict(at=datetime.now(timezone.utc).isoformat(), previous_goal_turn_classification="PROGRESS:qualified actualIPOdates,one fixed head trained and failed,independent replay passed",
    this_goal_turn_classification="PROGRESS:actual Gemini4 uploaded-source audit and independent Python CSV replay received;causal and source-scope overclaims corrected in append-only reply",
    status="FIXED_EXPERIMENT_REJECTED_AND_GEMINI4_REVIEW_COMPLETED", trainer="Codex", reviewer="Chrome Gemini 4 Argon", conversation="https://gemini.google.com/app/ec17e48589786054",
    auditor_executed_python_on_uploaded_full_validation_csvs=True, auditor_tool_output_sha_matches_local=True, auditor_local_model_or_private_repo_access=False,
    auditor_fits=0, core_fits=2, temporal_calibrator_fits=1, monthly_fits=0, baseline_refits=0, second_stage_refits=0, production_deployments=0, actual_trades=0, new_forward_results=0,
    screened_top1=dict(new_hits=48, baseline_hits=52, known_each=117), non_one_price_top1=dict(new_hits=43, baseline_hits=48, known_each=117),
    outcome="REJECTED_NO_GAIN", audit_scope="Static supplied-code review and independently executed full CSV settlement;model training/score replay and raw source qualification remain the separately bound Codex local verification",
    goal_60_achieved=False, goal_kept_active=True, frozen_report_and_manifest_hashes_unchanged=True, source_survivorship_and_late_acquisition_limits_retained=True,
    review_followup="No substantive codebug found in supplied scope;causal mechanism and whole-dataset no-leakage overclaims explicitly corrected;original replies preserved",
    batch_end="Single fixed comparison and its requested independent audit complete;no result-driven age transformation,cutoff or retry",
    next_research="Seek distinct qualified first-close information under the original complete-date Top1 target before another separately preregistered batch;do not retry failed arms or change60%denominator",
    screenshot_format_correction="Browser screenshot returnedJPEG;initialPNG-suffixassertion failed beforecompletionwaswritten;renamedunalteredbytes.jpgandverifiedJPEGsignature,no model code or result change",
    artifacts={p.relative_to(RUN).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [AUDIT / "GEMINI4_REVIEW_REPLY.txt", AUDIT / "GEMINI4_REVIEW_WITH_TOOL_EXECUTION.txt", AUDIT / "GEMINI4_REVIEW_SCOPE_CORRECTION.txt", AUDIT / "GEMINI4_REVIEW_FINAL.jpg", RUN / "真实上市年限固定训练与Gemini4审查.md"]})
(AUDIT / "REVIEW_COMPLETED.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
(RUN / "completion.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key:receipt[key] for key in ("status", "auditor_tool_output_sha_matches_local", "outcome", "goal_60_achieved", "goal_kept_active")}, ensure_ascii=False))
