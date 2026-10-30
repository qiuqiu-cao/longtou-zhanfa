"""Freeze only after actual peer discussion, code review and zero-fit checks."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import first_board_joint_ladder_emotion_model as m

design = Path(__file__).resolve().parent
review_path = design / "gemini/IMPLEMENTATION_REVIEW_BEFORE_FIT.json"
review = json.loads(review_path.read_text("utf-8"))
assert review["status"] == "IMPLEMENTATION_AND_FIT_PLAN_AUDIT_PASSED" and review["ready_for_protocol_freeze"]
agreement = json.loads((design / "gemini/FINAL_DESIGN_AGREEMENT.json").read_text("utf-8"))
assert agreement["status"] == "AGREED_ON_FINAL_DESIGN_BEFORE_FREEZE"
theme = json.loads((design / "gemini/THEME_EMOTION_SCOPE_DECISION.json").read_text("utf-8"))
assert theme["decision_on_two_arms"]["action"] == "MAINTAIN_ARMS_80_81_AS_MARKET_EMOTION_BASELINE"
preflight = json.loads((design / "PRE_FREEZE_VERIFICATION.json").read_text("utf-8"))
assert preflight["status"] == "READ_ONLY_PREFLIGHT_PASS"
for file, digest in preflight["bindings"].items():
    assert m.base.sha(file) == digest, file
now = datetime.now(timezone.utc)
out = m.ROOT / "research" / ("gemini_codex_joint_ladder_emotion_" + now.strftime("%Y%m%dT%H%M%SZ"))
out.mkdir()
pointer_path = m.ROOT / "research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
shutil.copy2(pointer_path, out / "PREVIOUS_JOINT_EXPERIMENT_POINTER.json")
plan = json.loads((design / "EXACT_FIT_PLAN_BEFORE_ANY_FIT.json").read_text("utf-8"))
bindings = preflight["bindings"].copy()
extra = [design / "gemini/FINAL_DESIGN_AGREEMENT.json", review_path,
    design / "gemini/THEME_EMOTION_SCOPE_DECISION.json", design / "CODEX_THEME_DECISION_AND_CORRECTIONS.md",
    design / "PRE_FREEZE_VERIFICATION.json", m.ROOT / "tests/test_first_board_emotion_forecast.py",
    m.ROOT / "tests/test_first_board_ladder_emotion_features.py"]
protected = [m.prior.REFERENCE / x for x in ("report.json", "S1_ALL_PREDICTIONS.csv", "S2_ALL_PREDICTIONS.csv")]
for file in extra + protected + list(m.base.INPUTS.values()):
    bindings[str(file.resolve())] = m.base.sha(file)
protocol = dict(status="AGREED_BEFORE_NEW_FITS", formed_at=now.isoformat(), batch_started_at=now.isoformat(),
    batch_deadline_utc=(now+timedelta(minutes=30)).isoformat(), budget_minutes=30,
    actual_peer="Antigravity Gemini 3.8 Flash High", design_root=str(design),
    source_sha256=m.base.sha(m.__file__),
    arms={arm: dict(trainer=m.OWNERS[arm], fields=fields) for arm,fields in m.ARMS.items()},
    bindings=bindings, fit_plan=plan,
    scope="Market-wide ladder and sequential acceptance component only; theme/narrative recognition remains separate and unsolved",
    hypothesis="New tier structure and tier-specific feedback paths can condition existing sealing observations; an independently tested forward pool-rate head may improve ranking",
    qualified_new_inputs=13, reused_sealing_inputs=5, actor_identity_truth=False,
    macro_target="Complete original first-board cohort next-official-close promotion rate if known coverage>=.95; otherwise unknown",
    macro_pipeline="median keep_empty_features; StandardScaler; Ridge alpha1; clip[0,1]; one date one weight",
    macro_causal_rule="252 official positions, all label_end<cut; minimum63 qualified dates; monthly forward OOF; five cold startsNaN, no stock removed",
    nested_calibration="42 signal dates; stock prefix and one frozen macro prefix exclude all calibration outcomes; 2025H1 uses a single pre2025 macro head",
    baseline="Frozen original62 HGB and original capital/thin-turnover screen; no baseline or S2 refit",
    stock_parameters="160 iterations,15 leaves,minleaf60,lr.06,L2=8,balanced,seed42,no early stop,equal day total weights,Platt C1 positive slope",
    validation="4724 rows/117 dates in2025H1; exact10586 original pre2025 members; >52 mainTop1 hits,>=48 non-one-price hits,datecoverage>=.95; BASE wins ties",
    B_additional_gate="Emotion paired-date coverage>=.95 and both MAE/RMSE strictly lower than prior known first-cohort promotion naive",
    winner="Only passing arms; highest mainTop1 hits; ties fewer inputs. Only this winner enters original14 monthly windows",
    final_goal="Complete-dateTop1knownhit>=.60;>=200known;outcomecoverage>=.95;noabstention;2025H2and2026each>=.55;noauctionfields",
    second_stage="Same selected stock after actual first promotion; frozen S2 recheck at its second close, reported separately",
    unknown_and_excluded_denominators="Keep all28503 original first rows,126unknown,all dates,exclusions and positive exclusions; no threshold/cap changes",
    data_boundary="Same original nominal-price/surviving universe limitations for both sides; known partial corporate-action correction not silently applied; no fully verified universe claim",
    historical_replay_not_pristine_blind=True, expected_validation_HGB_fits=4,
    expected_validation_Platt_fits=2, expected_validation_Ridge_fits=10,
    stop="Once per declared component; no retry/retune after effects. Preserve partial failures.30minutes",
    theme_next="Dynamic earlier anchor->leave-one-out narrative diffusion and cohort feedback, distinct from failed static17; no third arm in this batch",
    production_changes=False, actual_trades=0, fits_at_freeze=0)
m.base.write_json(out / "JOINT_FROZEN_PROTOCOL.json", protocol)
for file in [Path(m.__file__), Path(m.macro.__file__), Path(m.features.__file__)]:
    shutil.copy2(file, out / file.name)
m.base.write_json(out / "CODEX_FREEZE_ACK_BEFORE_FIT.json", dict(at=m.base.now(),
    protocol_sha256=m.base.sha(out / "JOINT_FROZEN_PROTOCOL.json"), review_sha256=m.base.sha(review_path),
    actual_peer_agreement_and_review_exists=True, fits=0, corrections=str(design / "CODEX_THEME_DECISION_AND_CORRECTIONS.md")))
m.base.write_json(design / "completion.json", dict(at=m.base.now(), status="JOINT_DESIGN_AND_PREFLIGHT_COMPLETE_NO_FITS",
    numerical_root=str(out), protocol_sha256=m.base.sha(out / "JOINT_FROZEN_PROTOCOL.json"), fits=0,
    full_rows=28503, observed_dates=668, new_features=13, relevant_tests_passed=11,
    macro_scope_only=True, theme_emotion_still_to_be_tested=True))
m.base.write_json(pointer_path, dict(updated_at=m.base.now(), status="JOINT_PROTOCOL_FROZEN_AWAIT_ACTUAL_TRAINERS",
    experiment_root=str(out), protocol=str(out / "JOINT_FROZEN_PROTOCOL.json"), actual_peer=protocol["actual_peer"],
    training_arms=protocol["arms"], active_model_training=False, process_handles={}, model_fits=0,
    goal_60_achieved=False, data_audit_expansion_deferred=True, current_research_goal_still_active=True))
print(json.dumps(dict(numerical_root=str(out), deadline=protocol["batch_deadline_utc"],
    protocol_sha256=m.base.sha(out / "JOINT_FROZEN_PROTOCOL.json")), ensure_ascii=True), flush=True)
