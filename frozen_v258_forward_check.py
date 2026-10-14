"""Historical replay with independent next-session checks, NOT a forward test.

The legacy filename is retained for compatibility. No pre-outcome registration
exists for this replay. V258's code-order fallback is withdrawn: preserve the
model ranking and report market risk separately. Wrong-session labels are
unavailable, not failures and not usable evidence of next-day promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from relay_promotion_holdout_v256 import SPECS, _fit, _top, load_rows


VERSION = "v259-historical-replay-calendar-audit-2"
TRAIN_END_EXCLUSIVE = pd.Timestamp("2026-01-01")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_target_calendar(rows: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "official_query_success", "provider_cross_section_complete"}
    if not required <= set(calendar):
        raise ValueError("calendar requires verified session evidence columns")
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal.date, errors="raise").dt.strftime("%Y-%m-%d")
    if cal.date.duplicated().any():
        raise ValueError("duplicate calendar sessions")
    if not (cal.official_query_success.eq(True) & cal.provider_cross_section_complete.eq(True)).all():
        raise ValueError("incomplete calendar evidence; cannot skip unverified sessions")
    sessions = sorted(cal.date.tolist())
    successor = dict(zip(sessions[:-1], sessions[1:]))
    checked = rows.copy()
    checked["expected_trade_date"] = checked.date.map(successor)
    checked["calendar_label_status"] = "NEXT_SESSION_UNVERIFIED"
    available = checked.expected_trade_date.notna()
    matched = available & checked.trade_date.eq(checked.expected_trade_date)
    checked.loc[available & ~matched, "calendar_label_status"] = "WRONG_SESSION_LABEL"
    checked.loc[matched, "calendar_label_status"] = "NEXT_SESSION_MATCHED"
    label_valid = checked.promoted_next_day.isin([0, 1]) & checked.promotion_label_status.eq(
        "verified_from_next_daily_limit_close")
    checked["evaluation_label_valid"] = matched & label_valid
    checked["archived_promotion_label"] = checked.promoted_next_day
    checked.loc[~checked.evaluation_label_valid, "promoted_next_day"] = float("nan")
    return checked


def records(frame: pd.DataFrame, fields: list[str]) -> list[dict]:
    # pandas serializes missing values as JSON null, never as a false failure.
    return json.loads(frame[fields].to_json(orient="records", double_precision=15))


def run(source_path: str, target_date: str, output_dir: str, calendar_path: str) -> dict:
    source = Path(source_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("immutable output directory already exists")
    rows, audit = load_rows(source)
    target_date = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    if pd.Timestamp(target_date) < TRAIN_END_EXCLUSIVE:
        raise ValueError("replay target overlaps training window")
    valid = rows.promoted_next_day.isin([0, 1]) & rows.promotion_label_status.eq(
        "verified_from_next_daily_limit_close"
    )
    next_date = pd.to_datetime(rows.trade_date, errors="coerce")
    dates = pd.to_datetime(rows.date, errors="coerce")
    train = rows[(dates < TRAIN_END_EXCLUSIVE) & valid
                 & (next_date < TRAIN_END_EXCLUSIVE) & (next_date > dates)].copy()
    target_rows = rows[rows.date.eq(target_date)].copy()
    if target_rows.empty:
        raise ValueError(f"target date not found: {target_date}")
    calendar_source = Path(calendar_path)
    target_rows = audit_target_calendar(target_rows, pd.read_csv(calendar_source))
    complete = bool(target_rows.evaluation_label_valid.all())
    if train.empty:
        raise ValueError("empty training set")

    scored = target_rows.copy()
    for name, fields in SPECS.items():
        scored["score_" + name] = _fit(train, scored, fields)
    scored["identity_channel"] = "LEADER_IDENTITY_DIAGNOSTIC"
    scored["trade_authorization"] = "CASH"
    scored["trade_authorization_reason"] = "theme_intraday_actual_fill_evidence_missing"

    selections = {}
    for scope, predicate in {
        "two_board": scored.boards.eq(2),
        "three_plus": scored.boards.ge(3),
        "all_boards": scored.boards.ge(2),
    }.items():
        pool = scored[predicate & scored.one_word_proxy.eq(0)].copy()
        if pool.empty:
            selections[scope + ":non_one_price"] = {"status": "NO_CANDIDATES"}
            continue
        baseline = _top(pool)
        model = _top(pool, "score_identity_emotion_price_volume")
        # Emotion risk does not authorize arbitrary code-order reranking.
        chosen = model
        state = pool.state_divergence
        risk = ("DIVERGENCE" if state.eq(1).all() else
                "NON_DIVERGENCE" if state.eq(0).all() else "UNKNOWN")
        selections[scope + ":non_one_price"] = {
            "candidate_rows": int(len(pool)),
            "market_risk_state": risk,
            "baseline_comparator_only": records(baseline, ["date", "code", "boards", "promoted_next_day"]),
            "identity_model": records(chosen, ["date", "code", "boards", "promoted_next_day",
                "score_identity_emotion_price_volume", "calendar_label_status"]),
        }

    report = {
        "version": VERSION,
        "status": "HISTORICAL_REPLAY_DIAGNOSTIC_ONLY" if complete else "LABEL_INVALID_DIAGNOSTIC_ONLY",
        "source_audit": audit,
        "target_date": target_date,
        "train_rows": int(len(train)),
        "max_training_label_date": str(train.trade_date.max()),
        "target_date_complete": complete,
        "calendar_audit": records(scored, ["date", "code", "trade_date", "expected_trade_date",
            "calendar_label_status", "evaluation_label_valid"]),
        "selections": selections,
        "interpretation": {
            "is_pristine_blind_test": False,
            "is_forward_test": False,
            "reason": "historical date and outcomes inspected before prediction registration",
            "ranking_rule": "preserve diagnostic model ranking; V258 code-order fallback withdrawn",
            "calibrated_probability": False,
            "training_calendar_coverage": "NOT_FULLY_AUDITED_BY_THIS_TARGET_DATE_CHECK",
            "actual_fill_count": 0,
            "theme_and_intraday_missing": True,
            "sample_warning": "one target date; not evidence of production win rate",
            "entry_authorized": False,
        },
        "source_hashes": {str(source.resolve()): sha(source),
            str(calendar_source.resolve()): sha(calendar_source),
            str(Path(__file__).resolve()): sha(Path(__file__)),
            str(Path('relay_promotion_holdout_v256.py').resolve()): sha(Path('relay_promotion_holdout_v256.py'))},
    }
    target.mkdir(parents=True)
    scored.to_csv(target / "target_predictions_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calendar", required=True)
    args = parser.parse_args()
    result = run(args.source, args.target_date, args.output, args.calendar)
    print(json.dumps({
        "version": result["version"],
        "status": result["status"],
        "target_date": result["target_date"],
        "selections": result["selections"],
    }, ensure_ascii=False))
