"""Fixed joint theme prior-anchor feedback experiment; research only, no actor labels."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import first_board_joint_seal_model as prior
import first_board_joint_reopen_model as reopen

base, stage, risk, history, members = prior.base, prior.stage, prior.risk, prior.history, prior.members
ROOT = prior.ROOT
OUTPUT = None
KEYS = prior.KEYS

DESIGN = ROOT / "research/gemini_codex_joint_theme_feedback_design_20261021T074448Z"
NEW_FIELDS_FILE = DESIGN / "COMPLETE28503_PRIOR_NAME_FEEDBACK_NO_TARGETS.csv"

THEME_FIELDS = [
    "prior_name_anchor_present",
    "name_anchor_current_mean_return",
    "name_anchor_current_limit_survival",
    "name_prev_followers_current_mean_return",
    "name_prev_followers_current_loss7_share",
]

SEAL_FIELDS = prior.TIME + prior.FUND + [reopen.FIELD]

A = "THEME_FEEDBACK_67"
B = "THEME_FEEDBACK_SEAL_72"

ARMS = {
    A: prior.ORIGINAL + THEME_FIELDS,
    B: prior.ORIGINAL + THEME_FIELDS + SEAL_FIELDS,
}
OWNERS = {A: "Gemini", B: "Gemini"}


def load_rows():
    rows = reopen.load_rows()
    new = pd.read_csv(NEW_FIELDS_FILE, dtype={"code": str}, float_precision="round_trip")
    assert stage.TARGET not in new and "label_end" not in new
    assert not new.duplicated(KEYS).any()
    assert set(map(tuple, rows[KEYS].to_numpy())) == set(map(tuple, new[KEYS].to_numpy()))
    result = rows.merge(new[KEYS + THEME_FIELDS], on=KEYS, validate="one_to_one", sort=False)
    np.testing.assert_allclose(result[prior.ORIGINAL + [stage.TARGET]], rows[prior.ORIGINAL + [stage.TARGET]],
                               atol=0, rtol=0, equal_nan=True)
    assert result[KEYS].equals(rows[KEYS]) and len(result) == 28503
    return result


def read_freeze():
    path = OUTPUT / "JOINT_FROZEN_PROTOCOL.json"
    freeze = json.loads(path.read_text("utf-8"))
    assert freeze["status"] == "AGREED_BEFORE_NEW_FITS"
    assert freeze["source_sha256"] == base.sha(__file__)
    assert {arm: item["fields"] for arm, item in freeze["arms"].items()} == ARMS
    assert {arm: item["trainer"] for arm, item in freeze["arms"].items()} == OWNERS
    for file, digest in freeze["bindings"].items():
        assert base.sha(file) == digest, file
    return freeze


def check_budget(freeze):
    if datetime.now(timezone.utc) >= datetime.fromisoformat(freeze["batch_deadline_utc"]):
        raise TimeoutError("Fixed 30-minute batch budget ended; preserve all attempts")


def train_arm(arm, trainer):
    freeze = read_freeze()
    check_budget(freeze)
    assert trainer == OWNERS[arm]
    out = OUTPUT / arm
    if out.exists():
        raise FileExistsError("Arm already attempted; preserve failure, no retune or retry")
    out.mkdir()
    base.write_json(out / "protocol_BEFORE_ANY_FIT.json", dict(at=base.now(), arm=arm, trainer=trainer,
                    freeze_sha256=base.sha(OUTPUT / "JOINT_FROZEN_PROTOCOL.json"), fields=ARMS[arm], fits_before=0))
    shutil.copy2(__file__, out / "source_before_fit.py")
    try:
        rows = load_rows()
        train = members.train_members(rows, "2025-01-01")
        old = pd.read_csv(prior.REFERENCE / "models/validation_S1_STAGE_CYCLE_GENE_full_members.csv",
                          dtype={"code": str}, float_precision="round_trip")
        membership_fields = KEYS + ["label_end", stage.TARGET]
        assert set(map(tuple, train[membership_fields].to_numpy())) == set(map(tuple, old[membership_fields].to_numpy()))
        val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")].copy()
        assert len(val) == 4724 and val.signal_date.nunique() == 117
        stage.VARIANTS[arm] = ARMS[arm]
        head = stage.fit_head(train, arm, out, "validation_" + arm)
        scored = stage.predict(head, val)
        scored.to_csv(out / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", index=False, encoding="utf-8-sig")
        primary = risk.screen_metric(scored)
        non = risk.screen_metric(scored[scored.one_price.eq(0)])
        passed = bool(primary["hits"] > 52 and non["hits"] >= 48 and primary["decision_date_coverage"] >= .95)
        report = dict(at=base.now(), trainer=trainer, arm=arm, primary=primary, non_one_price=non,
                      unfiltered_diagnostic=stage.metric(scored), pass_validation=passed,
                      core_fits=2, temporal_calibrator_fits=1, baseline_refits=0, second_stage_refits=0,
                      new_forward_results=0, actual_trades=0, monthly_fits=0, goal_60_achieved=False,
                      status="VALIDATION_PASS_AWAIT_JOINT_SELECTION" if passed else "REJECTED_VALIDATION",
                      source_sha256=base.sha(__file__))
        base.write_json(out / "report.json", report)
        read_freeze()
        print(json.dumps(base.clean(report), ensure_ascii=True), flush=True)
        return report
    except Exception as exc:
        base.write_json(out / "FAILURE_PRESERVED_NO_RETRY.json", dict(at=base.now(), error=repr(exc)))
        raise


def evaluate():
    freeze = read_freeze()
    check_budget(freeze)
    reports = {arm: json.loads((OUTPUT / arm / "report.json").read_text("utf-8")) for arm in ARMS}
    passed = [arm for arm in ARMS if reports[arm]["pass_validation"]]
    winner = sorted(passed, key=lambda arm: (-reports[arm]["primary"]["hits"], len(ARMS[arm]), arm))[0] if passed else None
    selection = dict(at=base.now(), reports=reports, selected=winner,
                     baseline_wins_ties=True, baseline_hits=52, baseline_non_one_price_hits=48)
    with (OUTPUT / "JOINT_SELECTION_BEFORE_EVALUATION.json").open("x", encoding="utf-8") as file:
        json.dump(base.clean(selection), file, ensure_ascii=False, indent=2)
    result = dict(at=base.now(), selection=selection, goal_60_achieved=False,
                  production_deployment_allowed=False, baseline_refits=0, second_stage_refits=0,
                  new_forward_results=0, actual_trades=0)
    if winner is None:
        result.update(status="BOTH_FIXED_ARMS_REJECTED", monthly_fits=0)
    else:
        out = OUTPUT / "evaluation"
        out.mkdir()
        rows = load_rows()
        sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
        stage.VARIANTS[winner] = ARMS[winner]
        parts = []
        began = time.monotonic()
        for month, part in rows[rows.signal_date.ge("2025-07-01")].groupby(rows.signal_date.str[:7], sort=True):
            check_budget(freeze)
            if time.monotonic() - began > 20 * 60:
                raise TimeoutError("Preserve checkpoints; no repeated monthly fits")
            boundary = month + "-01"
            position = int(np.searchsorted(sessions, boundary))
            train = members.train_members(rows, boundary, position - 252)
            old = pd.read_csv(prior.REFERENCE / "models" / (month + "_S1_STAGE_CYCLE_GENE_full_members.csv"),
                              dtype={"code": str}, float_precision="round_trip")
            membership_fields = KEYS + ["label_end", stage.TARGET]
            assert set(map(tuple, train[membership_fields].to_numpy())) == set(map(tuple, old[membership_fields].to_numpy()))
            head = stage.fit_head(train, winner, out, month + "_" + winner)
            assert head["train_max_label_end"] < part.signal_date.min()
            scored = stage.predict(head, part)
            scored["model_month"] = month
            scored.to_csv(out / (month + "_ALL_PREDICTIONS_BEFORE_METRICS.csv"), index=False, encoding="utf-8-sig")
            parts.append(scored)
            print(month + " saved", flush=True)
        assert len(parts) == 14
        scored = pd.concat(parts, ignore_index=True)
        scored.to_csv(out / "ALL14_MONTHS_PREDICTIONS.csv", index=False, encoding="utf-8-sig")
        frozen = risk.attach_frozen(rows, prior.REFERENCE / "S1_ALL_PREDICTIONS.csv")
        result["evaluation"] = history.period_metrics(scored, frozen)
        second = pd.read_csv(prior.REFERENCE / "S2_ALL_PREDICTIONS.csv", dtype={"code": str}, float_precision="round_trip")
        chain, result["same_stock_second_recheck"] = stage.chain_recheck(scored[scored.research_selection_allowed], second)
        chain.to_csv(out / "SAME_STOCK_SECOND_RECHECK.csv", index=False, encoding="utf-8-sig")
        result["calibration"] = stage.calibration_metrics(scored[scored.research_selection_allowed])
        metric = result["evaluation"]["all"]["new_screened"]
        result["goal_60_achieved"] = bool(metric["hit_rate"] >= .60 and metric["known"] >= 200
            and metric["outcome_coverage"] >= .95 and metric["abstention_dates"] == 0
            and all(result["evaluation"][period]["new_screened"]["hit_rate"] >= .55 for period in ("2025H2", "2026")))
        result.update(at=base.now(), status="FIXED_JOINT_MONTHLY_EVALUATION_COMPLETE", monthly_fits=14)
    base.write_json(OUTPUT / "report.json", result)
    read_freeze()
    print(json.dumps(base.clean(result), ensure_ascii=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--trainer", choices=["Codex", "Gemini"])
    parser.add_argument("--output", required=True, help="Exact new batch directory bound by its frozen protocol")
    args = parser.parse_args()
    OUTPUT = Path(args.output).resolve()
    assert OUTPUT.is_relative_to((ROOT / "research").resolve())
    if args.command == "evaluate":
        evaluate()
    elif not args.arm or not args.trainer:
        parser.error("train requires arm and actual trainer")
    else:
        train_arm(args.arm, args.trainer)
