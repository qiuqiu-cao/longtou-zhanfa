"""One jointly reviewed 62+8 first-close challenge; no production deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import first_board_leader_model as base
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
import numpy as np
import pandas as pd

JOINT = base.ROOT / "research/gemini_codex_joint_training_20261020"
NETWORK = JOINT / "codex/network_qualification_precision_repair"
NETWORK_FIELDS = ["co_limit_peer_n", "co_limit_history_weight", "co_limit_current_observed_weight_fraction",
                  "co_limit_peer_limit_rate", "co_limit_peer_first_rate", "co_limit_peer_failed_fraction",
                  "co_limit_peer_excess_return", "co_limit_previous_high_survival"]
ORIGINAL = stage.VARIANTS["STAGE_CYCLE_GENE"].copy()
FIELDS = ORIGINAL + NETWORK_FIELDS
VARIANT = "STAGE_CYCLE_GENE_CO_LIMIT_70"
KEYS = ["signal_date", "code"]
SCREEN_FILE = base.ROOT / "research/first_board_capital_risk_20261020_run1/COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv"


def attach_network(rows, network):
    if set(network) != set(KEYS + NETWORK_FIELDS):
        raise ValueError("Network must contain exactly keys and the eight approved fields; no labels or extra inputs")
    if rows.duplicated(KEYS).any() or network.duplicated(KEYS).any():
        raise ValueError("Duplicate complete-cohort keys")
    if set(map(tuple, rows[KEYS].to_numpy())) != set(map(tuple, network[KEYS].to_numpy())):
        raise ValueError("Network does not cover the unchanged full first-board cohort")
    if np.isinf(network[NETWORK_FIELDS].to_numpy(float)).any():
        raise ValueError("Invalid infinite network input")
    result = rows.merge(network, on=KEYS, validate="one_to_one", sort=False)
    assert result[KEYS].equals(rows[KEYS].reset_index(drop=True))
    np.testing.assert_allclose(result[ORIGINAL + [stage.TARGET]], rows[ORIGINAL + [stage.TARGET]], equal_nan=True, atol=0, rtol=0)
    return result


def same_members(train, original):
    cols = KEYS + ["stage", "label_end", stage.TARGET]
    a = train[cols].sort_values(KEYS).reset_index(drop=True)
    b = original[cols].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    if not train[stage.TARGET].notna().all():
        raise ValueError("Unknown labels cannot enter training")


def validation_passes(primary, non_one_price):
    return bool(primary["hits"] > 52 and primary["decision_date_coverage"] >= .95 and non_one_price["hits"] >= 48)


def comparison(rows, frozen):
    assert set(map(tuple, rows[KEYS].to_numpy())) == set(map(tuple, frozen[KEYS].to_numpy()))
    return dict(challenger=risk.screen_metric(rows), frozen_screened=risk.screen_metric(frozen),
                non_one_price_challenger=risk.screen_metric(rows[rows.one_price.eq(0)]),
                non_one_price_frozen=risk.screen_metric(frozen[frozen.one_price.eq(0)]),
                unfiltered_challenger=stage.metric(rows), unfiltered_frozen=stage.metric(frozen))


def run(output):
    started = time.monotonic()
    out = Path(output)
    if out.exists():
        raise FileExistsError("Preserve the one attempt, all failures and original results; no rerun")
    assert len(ORIGINAL) == 62 and len(FIELDS) == len(set(FIELDS)) == 70
    assert not set(risk.CAP_FIELDS) & set(FIELDS)
    verified = json.loads((NETWORK / "DIRECT_ENUMERATION_VERIFICATION.json").read_text("utf-8"))
    assert verified["status"] == "PASS" and verified["complete_rows"] == 28503
    assert verified["feature_mismatch_count"] == verified["peer_selection_mismatch_count"] == 0
    consensus = json.loads((JOINT / "chrome/GEMINI_REVIEW_RECEIPT.json").read_text("utf-8"))
    assert consensus["response_complete"] and consensus["verdict"].startswith("Approve for One Fixed Comparison")
    inputs = [risk.REFERENCE / "all_first_and_second_boards_WITH_SEPARATE_LABELS.csv", SCREEN_FILE,
              NETWORK / "COMPLETE_FIRST_BOARD_CAUSAL_NETWORK_NO_LABELS.csv", NETWORK / "qualification.json",
              NETWORK / "DIRECT_ENUMERATION_VERIFICATION.json", JOINT / "chrome/GEMINI_REVIEW_REPLY.txt",
              JOINT / "chrome/GEMINI_REVIEW_RECEIPT.json", JOINT / "codex/CODEX_PROPOSAL_BEFORE_GEMINI_REVIEW.md"]
    protected = [Path(base.__file__), Path(stage.__file__), Path(risk.__file__),
                 risk.REFERENCE / "S1_ALL_PREDICTIONS.csv", risk.REFERENCE / "S2_ALL_PREDICTIONS.csv",
                 risk.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv", risk.REFERENCE / "report.json"]
    protocol = dict(formed_at=base.now(), budget_minutes=30, hypothesis="Past-only co-limit peers' current-close diffusion adds predictive information to unchanged62-field first-board model",
                    fields=FIELDS, network_parameters=dict(history_official_positions=126, minimum_shared_limits=2, minimum_joint_observations=60, maximum_peers=8),
                    baseline="Reuse exact saved62-field scores with unchanged existing capital screen; no baseline or second-stage refit",
                    challenger=VARIANT, no_extra_capital_or_symbolic_fields=True,
                    parameters="Unchanged HGB160/15leaf/min60/lr.06/l2=8/balanced/seed42; equal-day weights; last42-date temporal Platt C1",
                    validation="Original2025H1 all4724rows/117days. >52 screened Top1 hits,>=48 non-one-price Top1 hits,>=95% decision-date coverage. Baseline wins ties. One validation head with prefix+full cores and one temporal calibrator.",
                    evaluation="Only if validation passes:14 original monthly252-official-position windows, same mature full members as original model. Save complete predictions before metrics. Reuse same-stock second-stage frozen scores.",
                    acceptance="Every original candidate day retained; Top1 next-official-close hit rate>=60%,>=200known,>=95% outcome coverage,no abstention dates;2025H2 and2026 each>=55%",
                    review_response="Chrome Gemini Argon approved fixed comparison; Codex accepts exact70-field guard, original screen and strict-tie gates. Bind repaired network CSV SHA; do not introduce rounding or change network selection.",
                    input_hashes={str(p.resolve()): base.sha(p) for p in inputs}, protected={str(p.resolve()): base.sha(p) for p in protected},
                    raw_sources={str(p.resolve()): base.sha(p) for p in base.INPUTS.values()}, source_sha256=base.sha(__file__),
                    historical_not_pristine_blind=True, external_quote_certification=False, production_changes=False, actual_trades=0)
    out.mkdir(parents=True)
    base.write_json(out / "protocol_BEFORE_ANY_FIT.json", protocol)
    shutil.copy2(__file__, out / "source_before_fit.py")
    once = JOINT / "codex/CO_LIMIT_CHALLENGE_ONCE.json"
    with once.open("x", encoding="utf-8") as stream:
        json.dump(dict(at=base.now(), output=str(out.resolve()), protocol_sha256=base.sha(out / "protocol_BEFORE_ANY_FIT.json")), stream, indent=2)
    rows = pd.read_csv(inputs[0], dtype={"code": str}, float_precision="round_trip")
    rows = rows[rows.stage.eq(1)].reset_index(drop=True)
    assert len(rows) == 28503
    screen = pd.read_csv(SCREEN_FILE, dtype={"code": str}, float_precision="round_trip")
    same_members(rows[rows[stage.TARGET].notna()], screen[screen[stage.TARGET].notna()])
    indexed_screen = screen.set_index(KEYS).loc[pd.MultiIndex.from_frame(rows[KEYS])]
    np.testing.assert_allclose(rows[ORIGINAL], indexed_screen[ORIGINAL], equal_nan=True, atol=0, rtol=0)
    assert screen.research_selection_allowed.isin([True, False]).all()
    rows = rows.merge(screen[KEYS + ["research_selection_allowed", "exclusion_reason", "threeboard_evidence_status"]], on=KEYS, validate="one_to_one", sort=False)
    rows = attach_network(rows, pd.read_csv(inputs[2], dtype={"code": str}, float_precision="round_trip"))
    rows.to_csv(out / "ALL_FIRST_BOARD_70_FIELDS_UNCHANGED_FACTS_LABELS_SCREEN.csv", index=False, encoding="utf-8-sig")
    known = rows[rows[stage.TARGET].notna()]
    train = known[known.signal_date.lt("2025-01-01") & known.label_end.lt("2025-01-01")]
    member_file = risk.REFERENCE / "models/validation_S1_STAGE_CYCLE_GENE_full_members.csv"
    same_members(train, pd.read_csv(member_file, dtype={"code": str}, float_precision="round_trip"))
    val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")]
    assert len(val) == 4724 and val.signal_date.nunique() == 117
    frozen = risk.attach_frozen(rows, risk.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    assert risk.screen_metric(frozen)["hits"] == 52 and risk.screen_metric(frozen[frozen.one_price.eq(0)])["hits"] == 48
    stage.VARIANTS[VARIANT] = FIELDS
    head = stage.fit_head(train, VARIANT, out, "validation_network70")
    predictions = stage.predict(head, val)
    predictions.to_csv(out / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", index=False, encoding="utf-8-sig")
    decision = comparison(predictions, frozen)
    decision.update(formed_at=base.now(), pass_validation=validation_passes(decision["challenger"], decision["non_one_price_challenger"]))
    base.write_json(out / "selection_BEFORE_EVALUATION.json", decision)
    print(json.dumps(base.clean(decision), ensure_ascii=True), flush=True)
    report = dict(completed_at=None, validation=decision, goal_60_passed=False, new_forward_results=0, actual_trades=0, production_deployment_allowed=False)
    if decision["pass_validation"]:
        sessions = json.loads(base.INPUTS["calendar"].read_text("utf-8"))["sessions"]
        test = rows[rows.signal_date.ge("2025-07-01")]
        frozen_test = risk.attach_frozen(rows, risk.REFERENCE / "S1_ALL_PREDICTIONS.csv")
        parts = []
        for month, part in test.groupby(test.signal_date.str[:7], sort=True):
            if time.monotonic() - started > 29 * 60:
                base.write_json(out / "CHECKPOINT_30MIN_NO_REPEATED_FITS.json", dict(at=base.now(), completed_months=len(parts), next_month=month, status="INCOMPLETE_FIXED_PROTOCOL_NO_EFFECT_TUNING"))
                return None
            boundary = month + "-01"
            position = int(np.searchsorted(sessions, boundary))
            tr = known[known.p.ge(position - 252) & known.label_end.lt(boundary)]
            old = pd.read_csv(risk.REFERENCE / "models" / (month + "_S1_STAGE_CYCLE_GENE_full_members.csv"), dtype={"code": str}, float_precision="round_trip")
            same_members(tr, old)
            bundle = stage.fit_head(tr, VARIANT, out, month + "_network70")
            assert bundle["train_max_label_end"] < part.signal_date.min()
            scored = stage.predict(bundle, part)
            scored["model_month"] = month
            scored.to_csv(out / (month + "_ALL_PREDICTIONS_BEFORE_METRICS.csv"), index=False, encoding="utf-8-sig")
            parts.append(scored)
            print(month + " saved", flush=True)
        scores = pd.concat(parts, ignore_index=True)
        assert len(parts) == 14
        scores.to_csv(out / "EVALUATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", index=False, encoding="utf-8-sig")
        report["evaluation"] = {}
        for period, mask in [("all", pd.Series(True, index=scores.index)), ("2025H2", scores.signal_date.lt("2026-01-01")), ("2026", scores.signal_date.ge("2026-01-01"))]:
            subset = scores[mask]
            baseline = frozen_test[frozen_test.signal_date.isin(subset.signal_date)]
            report["evaluation"][period] = comparison(subset, baseline)
        allowed = scores[scores.research_selection_allowed]
        second = pd.read_csv(risk.REFERENCE / "S2_ALL_PREDICTIONS.csv", dtype={"code": str}, float_precision="round_trip")
        chain, report["same_stock_second_recheck"] = stage.chain_recheck(allowed, second)
        chain.to_csv(out / "FIRST_SELECTION_TO_SAME_STOCK_SECOND_RECHECK.csv", index=False, encoding="utf-8-sig")
        report["calibration_on_allowed_pool"] = stage.calibration_metrics(allowed)
        m = report["evaluation"]["all"]["challenger"]
        report["goal_60_passed"] = bool(m["hit_rate"] >= .60 and m["known"] >= 200 and m["outcome_coverage"] >= .95 and m["abstention_dates"] == 0
            and all(report["evaluation"][p]["challenger"]["hit_rate"] >= .55 for p in ("2025H2", "2026")))
    report.update(completed_at=base.now(), core_fit_count=len(list((out / "models").glob("*_COMPLETED.json"))),
                  logical_heads=len(list((out / "models").glob("*_bundle.joblib"))), baseline_refits=0, second_stage_refits=0,
                  status="FIXED_COMPARISON_FINISHED" if decision["pass_validation"] else "VALIDATION_FAILED_NO_MONTHLY_FITS_NO_DEPLOYMENT")
    for filename, digest in {**protocol["input_hashes"], **protocol["protected"], **protocol["raw_sources"]}.items():
        assert base.sha(filename) == digest
    base.write_json(out / "report.json", report)
    print(json.dumps(base.clean({k: report[k] for k in ("status", "goal_60_passed", "core_fit_count", "logical_heads")}), ensure_ascii=True), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
