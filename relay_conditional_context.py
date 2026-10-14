"""Read-only conditional relay errors and pre-month composition decomposition.

Keep every candidate, source state and unknown. Historical target values are
used for diagnostics only; training comparisons enforce maturity before cutoff.
No models, scores, thresholds or production predictions are edited here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import relay_context_research as common

CELL = ["boards", "original_emotion", "legacy_proxy_cell"]
PHASES = ("启动", "升温", "加速", "一致", "大分歧", "退潮", "修复")


def phase_context(relay, market):
    """Signal-date-only archived algorithm states; labels/forecasts are not read."""
    if relay.duplicated(["date", "code"]).any() or market.date.duplicated().any():
        raise ValueError("unique candidate and signal-date market sources required")
    left = relay[["date", "code", "emotion_state"]].rename(columns={"emotion_state": "relay_phase"})
    right = market[["date", "emotion_state"]].rename(columns={"emotion_state": "market_phase"})
    joined = left.merge(right, on="date", how="left", validate="many_to_one", sort=False)
    records = []
    for row in joined.to_dict("records"):
        a = None if pd.isna(row["relay_phase"]) else row["relay_phase"]
        b = None if pd.isna(row["market_phase"]) else row["market_phase"]
        if a is not None and a not in PHASES or b is not None and b not in PHASES:
            raise ValueError("unregistered archived phase label")
        if a is not None and b is not None and a != b:
            state, status = None, "ARCHIVED_PHASE_CONFLICT"
        elif a is None and b is None:
            state, status = None, "ARCHIVED_PHASE_UNAVAILABLE"
        else:
            state = a if a is not None else b
            status = "AGREES_ORIGINAL_AND_MARKET" if a is not None and b is not None else (
                "SIGNAL_MARKET_SUPPLEMENT_ORIGINAL_RELAY_MISSING" if b is not None else "ORIGINAL_RELAY_ONLY")
        records.append({"date": row["date"], "code": row["code"], "original_relay_phase": a,
            "source_market_phase": b, "phase_feature": state, "status": status,
            "algorithm_state_not_independent_cycle_truth": True, "original_relay_field_modified": False})
    return records


def cells(frame):
    result = frame.copy()
    if result.boards.isna().any() or not result.boards.ge(2).all():
        raise ValueError("original relay board height required")
    result["original_emotion"] = result.emotion_state.fillna("UNKNOWN").astype(str)
    result["legacy_proxy_cell"] = result.signal_one_word.map(
        lambda v: "UNKNOWN" if pd.isna(v) else "PROXY" + str(int(v)) if v in (0., 1.) else "INVALID")
    if result.legacy_proxy_cell.eq("INVALID").any():
        raise ValueError("original proxy must be binary or unknown")
    return result


def training_parts(frame, cutoff):
    train = common.history.training_rows(frame, cutoff, "relay")
    if train.date.nunique() != 252 or train.target_date.isna().any() or train.target_date.ge(cutoff).any():
        raise ValueError("252 prior eligible dates and mature labels required")
    train = cells(train)
    train["full_day_weight"] = 1 / train.groupby("date").date.transform("size")
    last63 = set(sorted(train.date.unique())[-63:])
    recent, earlier = train[train.date.isin(last63)].copy(), train[~train.date.isin(last63)].copy()
    if recent.date.nunique() != 63 or earlier.date.nunique() != 189:
        raise ValueError("fixed historical split changed")
    return recent, earlier


def totals(frame):
    return {"n": len(frame), "signal_days": int(frame.date.nunique()),
        "positive": int(frame.target.eq(1).sum()), "weight_mass": float(frame.full_day_weight.sum()),
        "positive_weight_mass": float((frame.full_day_weight * frame.target).sum())}


def decomposition(recent, earlier):
    """Exact full-mass within/mix/unmatched decomposition, including empty cells."""
    if recent.empty or earlier.empty:
        return {"status": "UNDEFINED_ONE_PERIOD_EMPTY", "recent": totals(recent), "earlier": totals(earlier),
                "raw_rows": None, "equal_full_day_weight": None, "cell_rows": []}
    r = {key: totals(part) for key, part in recent.groupby(CELL, dropna=False)}
    o = {key: totals(part) for key, part in earlier.groupby(CELL, dropna=False)}
    output, rows = {}, []
    empty = {"n": 0, "signal_days": 0, "positive": 0, "weight_mass": 0., "positive_weight_mass": 0.}
    for mode, mass, positive in [("raw_rows", "n", "positive"),
                                ("equal_full_day_weight", "weight_mass", "positive_weight_mass")]:
        nr, no = sum(v[mass] for v in r.values()), sum(v[mass] for v in o.values())
        within = composition = unmatched = adequate_within = 0.
        common_r_mass = common_o_mass = adequate_r_mass = adequate_o_mass = 0.
        for key in sorted(set(r) | set(o)):
            a, b = r.get(key, empty), o.get(key, empty)
            wr, wo = a[mass] / nr, b[mass] / no
            rr, ro = a[positive] / a[mass] if a[mass] else None, b[positive] / b[mass] if b[mass] else None
            adequate = (a["n"] >= 20 and b["n"] >= 20 and a["signal_days"] >= 5 and b["signal_days"] >= 5)
            cw = cc = cu = 0.
            if rr is not None and ro is not None:
                cw = .5 * (wr + wo) * (rr - ro)
                cc = .5 * (rr + ro) * (wr - wo)
                common_r_mass += wr
                common_o_mass += wo
                if adequate:
                    adequate_within += cw
                    adequate_r_mass += wr
                    adequate_o_mass += wo
            else:
                cu = wr * (rr or 0.) - wo * (ro or 0.)
            within += cw
            composition += cc
            unmatched += cu
            rows.append({"mode": mode, **dict(zip(CELL, key)), "recent": a, "earlier": b,
                "recent_share": wr, "earlier_share": wo, "recent_rate": rr, "earlier_rate": ro,
                "rate_difference": rr - ro if rr is not None and ro is not None else None,
                "within_contribution": cw, "composition_contribution": cc, "unmatched_contribution": cu,
                "adequate_descriptive_support_both": adequate})
        recent_rate = sum(v[positive] for v in r.values()) / nr
        earlier_rate = sum(v[positive] for v in o.values()) / no
        delta = recent_rate - earlier_rate
        if abs(delta - within - composition - unmatched) > 1e-12:
            raise ValueError("complete cell decomposition does not add up")
        output[mode] = {"recent_rate": recent_rate, "earlier_rate": earlier_rate, "delta": delta,
            "within_cell_rate_change": within, "candidate_composition_change": composition,
            "unmatched_cell_contribution": unmatched, "common_recent_mass": common_r_mass,
            "common_earlier_mass": common_o_mass, "adequate_recent_mass": adequate_r_mass,
            "adequate_earlier_mass": adequate_o_mass, "adequate_within_contribution": adequate_within,
            "not_causal_or_forecast_effect": True}
    return {"status": "COMPLETE_DESCRIPTIVE_DECOMPOSITION", "recent": totals(recent), "earlier": totals(earlier),
            **output, "cell_rows": rows}


def safe_truth(value):
    return None if pd.isna(value) else bool(value)


def outcomes(rows):
    known = [r for r in rows if type(r["truth"]) is bool and type(r["prediction"]) is bool]
    tp = sum(r["truth"] and r["prediction"] for r in known)
    tn = sum(not r["truth"] and not r["prediction"] for r in known)
    fp = sum(not r["truth"] and r["prediction"] for r in known)
    fn = sum(r["truth"] and not r["prediction"] for r in known)
    positives = sum(r["truth"] is True for r in rows)
    return {"n": len(rows), "signal_days": len({r["signal_date"] for r in rows}),
        "scoreable": len(known), "unknown": len(rows) - len(known), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "positive": positives, "Accuracy": (tp + tn) / len(known) if known else None,
        "Precision": tp / (tp + fp) if tp + fp else None, "Recall": tp / positives if positives else None,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "coverage": len(known) / len(rows) if rows else None}


def build(plan_path, output_dir):
    plan = common.read(plan_path)
    common.verify(plan["source_bindings"])
    source_root = Path(__file__).resolve().parent / "research/model_upgrade_validation/recency_walkforward_20261013T134351Z"
    receipt = common.read(source_root / "implementation_BEFORE_FITS.json")
    common.verify(receipt["sources"])
    original_receipt = common.read("research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json")
    prepared = joblib.load(original_receipt["prepared_path"])
    full, market = prepared["relay"], prepared["market"]
    original = cells(full).set_index(["date", "code"])
    forecasts = pd.read_csv(source_root / "full_candidate_predictions.csv", dtype={"code": str})
    benchmark = forecasts.loc[forecasts.arm.eq("ROLL252")].copy()
    if benchmark.duplicated(["signal_date", "code"]).any() or len(benchmark) != 3707:
        raise ValueError("complete saved benchmark required")
    if set(zip(benchmark.signal_date, benchmark.code)) != set(zip(full.loc[full.date.ge("2025-07-01")].date, full.loc[full.date.ge("2025-07-01")].code)):
        raise ValueError("original evaluation members changed")
    state_sources = market.set_index("date")
    rows = []
    for r in benchmark.to_dict("records"):
        source = original.loc[(r["signal_date"], r["code"])]
        if ((pd.notna(source.target_date) and source.target_date != r["target_date"])
                or (pd.isna(source.target_date) and pd.notna(r["target_date"]))):
            raise ValueError("original target date changed")
        truth = safe_truth(source.target) if source.label_valid else None
        if truth is not safe_truth(r["truth"]):
            raise ValueError("original labels changed")
        r["truth"] = truth
        r["prediction"] = safe_truth(r["prediction"])
        r["error"] = "UNKNOWN" if truth is None or r["prediction"] is None else (
            "TP" if truth and r["prediction"] else "FN" if truth else "FP" if r["prediction"] else "TN")
        market_state = state_sources.loc[r["signal_date"], "emotion_state"] if r["signal_date"] in state_sources.index else None
        r.update(original_emotion=source.original_emotion, legacy_proxy_cell=source.legacy_proxy_cell,
            market_original_emotion="UNKNOWN" if pd.isna(market_state) else str(market_state),
            state_was_filled=False, diagnostic_generated_at=common.now())
        rows.append(r)
    decompositions, cell_rows, matrix = [], [], []
    for expected in receipt["checks"]:
        recent, earlier = training_parts(full, expected["cutoff"])
        train = pd.concat([recent, earlier]).sort_index()
        if common.history._frame_digest(train, prepared["relay_fields"]) != expected["training_digest"]:
            raise ValueError("original monthly input/labels changed")
        states = ["ALL"] + sorted(set(recent.original_emotion) | set(earlier.original_emotion))
        for state in states:
            a = recent if state == "ALL" else recent.loc[recent.original_emotion.eq(state)]
            b = earlier if state == "ALL" else earlier.loc[earlier.original_emotion.eq(state)]
            d = decomposition(a, b)
            cell_rows.extend({"month": expected["month"], "state_scope": state, **r} for r in d.pop("cell_rows"))
            decompositions.append({"month": expected["month"], "model_cutoff": expected["cutoff"], "state_scope": state,
                "training_max_target": train.target_date.max(), **d})
    frame = pd.DataFrame(rows)
    definitions = {"month_height_state_proxy": ["month", *CELL], "height_state_proxy": CELL,
        "emotion": ["original_emotion"], "board_height": ["boards"], "legacy_proxy": ["legacy_proxy_cell"]}
    for scope, keys in definitions.items():
        for key, group in frame.groupby(keys, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            matrix.append({"scope": scope, **dict(zip(keys, key)), **outcomes(group.to_dict("records"))})
    changes = []
    for arm in ("HL63", "HL126"):
        candidate = forecasts.loc[forecasts.arm.eq(arm)]
        joined = frame.merge(candidate[["signal_date", "code", "prediction", "raw_score"]], on=["signal_date", "code"],
            suffixes=("_benchmark", "_challenge"), validate="one_to_one")
        if len(joined) != len(frame):
            raise ValueError("saved challenger omitted original members")
        for raw in joined.to_dict("records"):
            raw.update(arm_challenge=arm, call_changed=raw["prediction_benchmark"] != raw["prediction_challenge"])
            changes.append(raw)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("unique conditional diagnosis output required")
    output.mkdir(parents=True)
    phase_rows = phase_context(full, market)
    common.write_new(output / "source_phase_context.json", {"formed_at": common.now(),
        "source_track": "CURRENT_RETROSPECTIVE_ARCHIVED_ALGORITHM_PHASE_NOT_ORIGINAL_TIME_REPAIR",
        "original_receipt_path": str(Path("research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json").resolve()),
        "source_receipt_sha256": common.sha("research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json"),
        "prepared_path": original_receipt["prepared_path"], "prepared_sha256": original_receipt["prepared_sha256"],
        "rows": phase_rows})
    frame.to_csv(output / "full3707_original_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix).to_csv(output / "complete_error_cells.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**r, "recent": str(r["recent"]), "earlier": str(r["earlier"])} for r in cell_rows]).to_csv(
        output / "full_pre_month_condition_cells.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(output / "all7414_original_challenger_pairs.csv", index=False, encoding="utf-8-sig")
    report = {"formed_at": common.now(), "original_complete_errors": outcomes(rows),
        "complete_error_cell_count": len(matrix), "pre_month_decompositions": decompositions,
        "pre_month_cell_rows": len(cell_rows), "legacy_proxy_is_not_actual_identity": True,
        "missing_states_never_relabelled_from_outcomes": True, "new_fits": 0,
        "new_mature_forward_samples": 0, "prediction_gain_verified": False,
        "source_bindings": {**plan["source_bindings"], str(Path(__file__).resolve()): common.sha(__file__),
            str(Path("tests/test_relay_conditional_context.py").resolve()): common.sha("tests/test_relay_conditional_context.py")}}
    report["output_hashes"] = {path.name: common.sha(path) for path in output.iterdir()}
    common.write_new(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build(args.plan, args.output_dir)
    print({k: result[k] for k in ("original_complete_errors", "complete_error_cell_count", "pre_month_cell_rows", "new_fits")})
