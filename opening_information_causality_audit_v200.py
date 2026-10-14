"""Audit whether opening information is paired with a causally valid entry.

The official opening gap is useful opening-confirmation evidence, but it is not
known at the prior close.  If a rule observes that gap and then still books a
fill at the same official open, the feature and execution price are paired at
the same event and the backtest is not executable.  This audit describes the
gap/loss relationship without granting such a fill and checks whether a later
09:30/09:35 confirmation dataset currently exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]
BANDS = ["LE_0", "GT_0_LE_5", "GT_5_LE_7", "GT_7"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opening_gap_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce") * 100
    return pd.cut(
        numeric, [-np.inf, 0, 5, 7, np.inf], labels=BANDS,
        right=True).astype("string")


def join_opening_evidence(predictions: pd.DataFrame,
                          outcomes: pd.DataFrame) -> pd.DataFrame:
    required_predictions = set(KEYS + [
        "name", "feature_available_at", "fill_status", "big_loss_target",
        "big_loss_target_available", "net_return_B_plus_1",
        "net_return_B_plus_2", "risk_base_probability",
        "risk_emotion_probability", "risk_emotion_balanced_probability",
    ])
    required_outcomes = set(KEYS + [
        "buy_date_B", "entry_rule", "fill_status", "B_open_gap",
        "B_open_upper_limit", "B_one_price_upper_limit",
    ])
    if required_predictions - set(predictions):
        raise ValueError("prediction fields missing")
    if required_outcomes - set(outcomes):
        raise ValueError("outcome fields missing")
    if predictions.duplicated(KEYS).any() or outcomes.duplicated(KEYS).any():
        raise ValueError("opening evidence keys must be unique")
    right = outcomes[list(required_outcomes)].rename(columns={
        "fill_status": "outcome_fill_status"})
    joined = predictions.merge(right, on=KEYS, how="left", validate="one_to_one")
    mismatch = (
        joined.outcome_fill_status.notna()
        & joined.fill_status.ne(joined.outcome_fill_status))
    if mismatch.any():
        raise ValueError("prediction and outcome fill status disagree")
    joined["B_open_gap_pct"] = (
        pd.to_numeric(joined.B_open_gap, errors="coerce") * 100)
    joined["opening_gap_band"] = opening_gap_band(joined.B_open_gap)
    return joined


def band_summary(frame: pd.DataFrame) -> dict:
    known = frame[
        frame.big_loss_target_available.eq(1)
        & frame.opening_gap_band.notna()].copy()
    result = {}
    for band in BANDS:
        part = known[known.opening_gap_band.eq(band)].copy()
        big = pd.to_numeric(part.big_loss_target, errors="coerce").dropna()
        b1 = pd.to_numeric(part.net_return_B_plus_1, errors="coerce").dropna()
        b2 = pd.to_numeric(part.net_return_B_plus_2, errors="coerce").dropna()
        risk = pd.to_numeric(
            part.risk_emotion_balanced_probability, errors="coerce").dropna()
        result[band] = {
            "rows": int(len(part)),
            "sessions": int(part.signal_date.nunique()),
            "big_loss_count": int(big.sum()) if len(big) else 0,
            "big_loss_rate": float(big.mean()) if len(big) else None,
            "B_plus_1_mean_net_return": float(b1.mean()) if len(b1) else None,
            "B_plus_2_mean_net_return": float(b2.mean()) if len(b2) else None,
            "mean_prior_close_balanced_risk_probability": (
                float(risk.mean()) if len(risk) else None),
        }
    return result


def audit(predictions: pd.DataFrame, outcomes: pd.DataFrame,
          frozen_top1: pd.DataFrame, readiness: dict) -> tuple[pd.DataFrame, dict]:
    joined = join_opening_evidence(predictions, outcomes)
    frozen = frozen_top1[
        frozen_top1.ranker.eq("profit_emotion_probability")].copy()
    if frozen.empty or frozen.duplicated("signal_date").any():
        raise ValueError("frozen emotion top-one is invalid")
    top = frozen[KEYS].merge(joined, on=KEYS, how="left", validate="one_to_one")
    top_big = top[pd.to_numeric(
        top.big_loss_target, errors="coerce").eq(1)].copy()
    big_cases = top_big[[
        "signal_date", "code", "name", "B_open_gap_pct", "opening_gap_band",
        "risk_base_probability", "risk_emotion_probability",
        "risk_emotion_balanced_probability", "net_return_B_plus_1",
        "net_return_B_plus_2",
    ]].to_dict("records")

    technical_sequences = int(readiness.get("technical_complete_sequences", 0) or 0)
    full_sequences = int(readiness.get("full_complete_sequences", 0) or 0)
    signal_dates = pd.to_datetime(
        joined.feature_available_at, errors="coerce", utc=True
    ).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    same_signal_day = signal_dates.eq(joined.signal_date.astype(str))
    known = joined[joined.big_loss_target_available.eq(1)].copy()
    report = {
        "version": "v200-opening-information-causality-audit-2",
        "status": "DIAGNOSTIC_ONLY",
        "scope": {
            "prequential_rows": int(len(joined)),
            "prequential_sessions": int(joined.signal_date.nunique()),
            "known_filled_risk_rows": int(len(known)),
            "frozen_top1_sessions": int(len(top)),
            "frozen_top1_big_loss_cases": int(len(top_big)),
        },
        "availability_contract": {
            "prior_close_features_available_at": "signal_date_15:05_or_earlier",
            "official_open_gap_available_at": (
                "after_next_session_opening_call_auction_match_approximately_09:25"),
            "continuous_auction_begins_at": "09:30",
            "opening_entry_proxy_price_time": (
                "same_next_session_opening_call_auction_match"),
            "opening_gap_is_prior_close_feature": False,
            "opening_gap_can_filter_same_official_open_fill": False,
            "09_25_indicative_price_equals_official_open_claim": False,
            "opening_gap_can_condition_later_09_30_or_09_35_entry_in_principle": True,
            "later_entry_requires_new_price_and_return_labels": True,
        },
        "candidate_opening_gap_bands": band_summary(joined),
        "frozen_top1_opening_gap_bands": band_summary(top),
        "frozen_top1_big_loss_case_details": big_cases,
        "intraday_confirmation_readiness": {
            "technical_complete_sequences": technical_sequences,
            "full_complete_sequences": full_sequences,
            "09_35_execution_model_ready": bool(
                technical_sequences > 0 and full_sequences > 0),
        },
        "diagnosis": {
            "large_losses_span_multiple_opening_gap_bands": bool(
                top_big.opening_gap_band.dropna().nunique() > 1),
            "opening_gap_is_safe_oracle_filter": False,
            "same_open_gap_filter_with_same_open_fill_is_executable": False,
            "validated_09_35_risk_reduction_demonstrated": False,
        },
        "governance": {
            "signal_features_timestamped_on_signal_date": bool(same_signal_day.all()),
            "opening_gap_allowed_in_prior_close_model": False,
            "same_open_counterfactual_rule_allowed": False,
            "opening_gap_direct_entry_allowed": False,
            "09_35_entry_authorized": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "The official opening gap is available only after the opening call-auction "
            "match, not at the prior close. Observing it and retaining a fill at that same "
            "auction price is a same-event execution error. A valid 09:30/09:35 model must "
            "use separately timestamped evidence and a new post-confirmation fill price."),
        "limitations": [
            "daily official open cannot describe 09:30-09:35 absorption, VWAP or reseal",
            "opening-gap bands are descriptive and were not selected as trading thresholds",
            "the frozen top-one evaluation contains only ten inspected sessions",
            "there are currently no complete forward intraday confirmation sequences",
        ],
    }
    return joined, report


def run(predictions_path, outcomes_path, top1_path, readiness_path, output_dir) -> dict:
    paths = [Path(predictions_path), Path(outcomes_path), Path(top1_path),
             Path(readiness_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V200 audit")
    predictions = pd.read_csv(paths[0], dtype={"code": str})
    outcomes = pd.read_csv(paths[1], dtype={"code": str})
    top1 = pd.read_csv(paths[2], dtype={"code": str})
    readiness = json.loads(paths[3].read_text(encoding="utf-8"))
    rows, report = audit(predictions, outcomes, top1, readiness)
    target.mkdir(parents=True)
    rows_path = target / "opening_causality_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths}
    report["artifact_hashes"] = {rows_path.name: _sha256(rows_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--top1", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.predictions, args.outcomes, args.top1, args.readiness, args.output),
        ensure_ascii=False, indent=2))
