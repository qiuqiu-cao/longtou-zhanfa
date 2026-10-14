"""V267 cycle-transition and fundamental-cleanliness research gate.

This module turns the latest review lessons into explicit, point-in-time
fields.  It distinguishes old-cycle repair, old-cycle supplementation, a
new-cycle trial, and a confirmed new cycle.  Fundamental cleanliness is a
risk filter for relay research, not a value-investing score and never an
order authorization.

Missing fields remain UNKNOWN.  The gate checks field coverage and timestamps
instead of treating the presence of an empty column as usable evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "v267-cycle-fundamental-gate-1"

BASE_FIELDS = {"code", "signal_date", "boards"}
CYCLE_FIELDS = {
    "old_cycle_anchor",
    "old_leader_break_date",
    "middle_board_clearance_status",
    "old_cycle_negative_feedback",
    "supplement_window",
    "new_theme_independent_status",
    "new_ladder_first_board",
    "new_ladder_middle_support",
    "new_cycle_breakout_status",
    "cycle_evidence_asof",
}
FUNDAMENTAL_FIELDS = {
    "fundamental_clean_status",
    "hard_risk_flags",
    "business_relevance",
    "theme_business_share",
    "earnings_quality",
    "cashflow_quality",
    "capital_structure_status",
    "audit_litigation_status",
    "fundamental_asof",
}

FIELD_GROUPS = {
    "base": BASE_FIELDS,
    "cycle": CYCLE_FIELDS,
    "fundamental": FUNDAMENTAL_FIELDS,
}

CYCLE_STATES = {
    "OLD_CYCLE_REPAIR",
    "SUPPLEMENT_WINDOW",
    "NEW_CYCLE_TRIAL",
    "NEW_CYCLE_CONFIRMED",
    "RETREAT_OR_MIXED",
    "UNKNOWN",
}
CLEAN_STATES = {"CLEAN", "CLEAN_BUT_NO_CATALYST", "DIRTY", "UNKNOWN"}


def _truth(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "是", "有", "成立", "cleared", "independent", "broken"}:
        return True
    if text in {"0", "false", "no", "n", "否", "无", "不成立", "not_cleared", "dependent", "not_broken"}:
        return False
    return None


def _known(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    # An explicit "NONE/无硬风险" is evidence that the field was reviewed;
    # it must not be confused with an empty value.
    return str(value).strip().lower() not in {"", "unknown", "unavailable", "nan"}


def _flag_tokens(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NO_HARD_RISK", "无"}:
        return []
    return [token.strip() for token in text.replace("|", ";").replace(",", ";").split(";") if token.strip()]


def _coverage(frame: pd.DataFrame, fields: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    n = len(frame)
    for field in sorted(fields):
        if field not in frame.columns:
            result[field] = {"status": "UNAVAILABLE", "coverage": 0.0, "known_rows": 0}
            continue
        values = frame[field]
        known = int(values.map(_known).sum())
        result[field] = {
            "status": "AVAILABLE" if known else "UNAVAILABLE",
            "coverage": known / n if n else 0.0,
            "known_rows": known,
        }
    return result


def classify_cycle(row: pd.Series) -> str:
    """Classify only when the transition evidence is sufficiently explicit."""
    old_anchor = _truth(row.get("old_cycle_anchor"))
    old_negative = _truth(row.get("old_cycle_negative_feedback"))
    supplement = _truth(row.get("supplement_window"))
    independent = str(row.get("new_theme_independent_status", "")).strip().upper()
    first = _truth(row.get("new_ladder_first_board"))
    middle = _truth(row.get("new_ladder_middle_support"))
    clearance = str(row.get("middle_board_clearance_status", "")).strip().upper()
    breakout = str(row.get("new_cycle_breakout_status", "")).strip().upper()

    if any(value is None for value in (old_anchor, old_negative, first, middle)):
        return "UNKNOWN"
    if old_negative and clearance == "CLEARED" and independent in {"INDEPENDENT", "TRUE", "YES"} and first and middle:
        return "NEW_CYCLE_CONFIRMED" if breakout in {"BROKEN", "CONFIRMED", "TRUE", "YES"} else "NEW_CYCLE_TRIAL"
    if supplement and old_negative is False and independent in {"DEPENDENT", "TRUE", "YES"}:
        return "SUPPLEMENT_WINDOW"
    if old_anchor and old_negative is False:
        return "OLD_CYCLE_REPAIR"
    if old_negative and not (independent in {"INDEPENDENT", "TRUE", "YES"} and first and middle):
        return "RETREAT_OR_MIXED"
    return "UNKNOWN"


def classify_cleanliness(row: pd.Series) -> str:
    explicit = str(row.get("fundamental_clean_status", "")).strip().upper()
    flags = _flag_tokens(row.get("hard_risk_flags"))
    if flags:
        return "DIRTY"
    if explicit in CLEAN_STATES - {"UNKNOWN"}:
        return explicit
    return "UNKNOWN"


def _future_rows(frame: pd.DataFrame, asof: str | None) -> int:
    if not asof:
        return 0
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("Asia/Shanghai")
    else:
        cutoff = cutoff.tz_convert("Asia/Shanghai")
    total = 0
    for field in ("cycle_evidence_asof", "fundamental_asof"):
        if field not in frame:
            continue
        stamps = pd.to_datetime(frame[field], errors="coerce", utc=True)
        total += int(stamps.gt(cutoff.tz_convert("UTC")).sum())
    return total


def audit(frame: pd.DataFrame, *, asof: str | None = None, min_coverage: float = 0.8) -> dict[str, Any]:
    rows = frame.copy()
    if "code" in rows:
        rows["code"] = rows["code"].astype(str).str.zfill(6)
    if "signal_date" in rows:
        rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="coerce")
    if "boards" in rows:
        rows["boards"] = pd.to_numeric(rows["boards"], errors="coerce")

    field_status = {group: _coverage(rows, fields) for group, fields in FIELD_GROUPS.items()}
    base_ok = BASE_FIELDS <= set(rows.columns)
    cycle_coverage = {name: item["coverage"] for name, item in field_status["cycle"].items()}
    fundamental_coverage = {name: item["coverage"] for name, item in field_status["fundamental"].items()}
    cycle_ok = bool(CYCLE_FIELDS <= set(rows.columns) and all(value >= min_coverage for value in cycle_coverage.values()))
    fundamental_ok = bool(FUNDAMENTAL_FIELDS <= set(rows.columns) and all(value >= min_coverage for value in fundamental_coverage.values()))
    future_rows = _future_rows(rows, asof)

    if len(rows):
        rows["cycle_state_v267"] = rows.apply(classify_cycle, axis=1)
        rows["fundamental_clean_state_v267"] = rows.apply(classify_cleanliness, axis=1)
        rows["cycle_fundamental_context_v267"] = rows.apply(
            lambda row: (
                "DIRTY_OR_UNKNOWN" if row["fundamental_clean_state_v267"] in {"DIRTY", "UNKNOWN"}
                else "NEW_CYCLE_CLEAN_CANDIDATE" if row["cycle_state_v267"] in {"NEW_CYCLE_TRIAL", "NEW_CYCLE_CONFIRMED"}
                else "SUPPLEMENT_CLEAN_CANDIDATE" if row["cycle_state_v267"] == "SUPPLEMENT_WINDOW"
                else "OLD_CYCLE_CLEAN_CORE" if row["cycle_state_v267"] == "OLD_CYCLE_REPAIR"
                else "CYCLE_UNKNOWN"
            ), axis=1,
        )
    else:
        rows["cycle_state_v267"] = pd.Series(dtype=str)
        rows["fundamental_clean_state_v267"] = pd.Series(dtype=str)
        rows["cycle_fundamental_context_v267"] = pd.Series(dtype=str)

    context_allowed = bool(base_ok and cycle_ok and fundamental_ok and future_rows == 0)
    return {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "rows": int(len(rows)),
        "latest_signal_date": rows.signal_date.max().strftime("%Y-%m-%d") if "signal_date" in rows and rows.signal_date.notna().any() else None,
        "field_status": field_status,
        "coverage_threshold": min_coverage,
        "cycle_review": {"allowed": bool(base_ok and cycle_ok and future_rows == 0), "coverage_complete": cycle_ok},
        "fundamental_cleanliness_review": {"allowed": bool(base_ok and fundamental_ok and future_rows == 0), "coverage_complete": fundamental_ok},
        "cycle_fundamental_context_review": {
            "allowed": context_allowed,
            "reason": "requires complete cycle and fundamental fields at the configured coverage threshold",
        },
        "future_timestamp_rows": future_rows,
        "row_state_counts": {
            "cycle": rows["cycle_state_v267"].value_counts(dropna=False).to_dict(),
            "fundamental": rows["fundamental_clean_state_v267"].value_counts(dropna=False).to_dict(),
            "context": rows["cycle_fundamental_context_v267"].value_counts(dropna=False).to_dict(),
        },
        "trade_authorization": {"allowed": False, "reason": "V267 is a research gate; live confirmation, fills and exits remain separate."},
        "governance": {
            "clean_fundamentals_are_a_risk_filter": True,
            "cycle_state_is_not_a_buy_signal": True,
            "unknown_is_not_dirty": True,
            "forum_experience_requires_forward_validation": True,
            "empty_columns_cannot_pass_coverage_gate": True,
        },
        "rows_for_output": rows.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asof")
    parser.add_argument("--min-coverage", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable V267 output")
    frame = pd.read_csv(args.input, dtype={"code": str}, low_memory=False)
    report = audit(frame, asof=args.asof, min_coverage=args.min_coverage)
    args.output.mkdir(parents=True)
    rows = pd.DataFrame(report.pop("rows_for_output"))
    rows.to_csv(args.output / "cycle_fundamental_rows.csv", index=False, encoding="utf-8-sig")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "context_review_allowed": report["cycle_fundamental_context_review"]["allowed"],
        "trade_authorization": report["trade_authorization"]["allowed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
