"""Standardize a reconciled daily limit-up universe for the daily review.

Closing/failed classification comes from the independent quote reconciliation;
point-in-time eligibility comes from official SSE/SZSE roster evidence. Missing
microstructure fields remain unavailable rather than being filled with zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from collect_security_master_v4 import name_status


SSE_PREFIXES = ("600", "601", "603", "605")
SZSE_PREFIXES = ("000", "001", "002", "003")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code(value) -> str:
    return str(value).split(".")[0].strip().zfill(6)


def _sse_security(current: pd.DataFrame, signal_date: str, captured_at: str) -> pd.DataFrame:
    required = {"A_STOCK_CODE", "SEC_NAME_CN", "LIST_DATE"}
    if required - set(current):
        raise ValueError("SSE official current roster fields missing")
    frame = current.copy()
    frame["code"] = frame["A_STOCK_CODE"].map(_code)
    if frame.code.duplicated().any():
        raise ValueError("SSE current roster contains duplicate codes")
    frame["name"] = frame["SEC_NAME_CN"].astype(str).str.strip()
    frame["st_status"] = frame.name.map(name_status)
    frame["valid_from"] = signal_date
    frame["valid_to"] = signal_date
    frame["exchange"] = "SSE"
    frame["board"] = "MAINBOARD"
    frame["listing_status"] = "LISTED"
    frame["price_limit_regime"] = frame.st_status.map(
        lambda value: "ST_5" if value in {"ST", "*ST"} else
        "OTHER" if value == "DELISTING_RISK" else "MAINBOARD_10"
    )
    frame["source_type"] = "sse_official_current_roster"
    frame["source_ref"] = "https://www.sse.com.cn/assortment/stock/list/share/"
    frame["source_timestamp"] = captured_at
    frame["data_status"] = "verified_current_snapshot"
    return frame[["code", "name", "valid_from", "valid_to", "exchange", "board",
                  "listing_status", "st_status", "price_limit_regime", "source_type",
                  "source_ref", "source_timestamp", "data_status"]]


def _szse_security(intervals: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    required = {"code", "name", "valid_from", "valid_to", "exchange", "board",
                "listing_status", "st_status", "price_limit_regime", "source_type",
                "source_ref", "source_timestamp", "data_status"}
    if required - set(intervals):
        raise ValueError("SZSE official interval fields missing")
    frame = intervals.copy()
    frame["code"] = frame.code.map(_code)
    frame = frame[(frame.valid_from.astype(str) <= signal_date)
                  & (frame.valid_to.astype(str) >= signal_date)].copy()
    if frame.code.duplicated().any():
        raise ValueError("SZSE interval match contains duplicate codes")
    return frame[list(required)].copy()


def _eligible(row) -> tuple[str, str]:
    if row is None:
        return "unavailable", "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE"
    verified = (str(row.get("data_status", "")).startswith("verified")
                and row.get("board") == "MAINBOARD"
                and row.get("exchange") in {"SSE", "SZSE"})
    allowed = (verified and row.get("listing_status") == "LISTED"
               and row.get("st_status") == "NORMAL"
               and row.get("price_limit_regime") == "MAINBOARD_10")
    return ("verified", "VERIFIED_ELIGIBLE") if allowed else (
        "verified" if verified else "unavailable",
        "VERIFIED_INELIGIBLE" if verified else "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE",
    )


def build(closing: pd.DataFrame, failed: pd.DataFrame, reconciled_manifest: dict,
          raw_failed: pd.DataFrame, sse_current: pd.DataFrame, szse_intervals: pd.DataFrame,
          signal_date: str, sse_captured_at: str):
    if reconciled_manifest.get("signal_date") != signal_date:
        raise ValueError("reconciled manifest date mismatch")
    if len(closing) != int(reconciled_manifest["closing_count"]):
        raise ValueError("closing count differs from reconciled manifest")
    if len(failed) != int(reconciled_manifest["failed_count"]):
        raise ValueError("failed count differs from reconciled manifest")
    closing = closing.copy(); failed = failed.copy(); raw_failed = raw_failed.copy()
    for frame in (closing, failed):
        frame["code"] = frame.code.map(_code)
    raw_failed["code"] = raw_failed["代码"].map(_code)
    raw_failed_map = raw_failed.set_index("code").to_dict("index")
    security = pd.concat([
        _sse_security(sse_current, signal_date, sse_captured_at),
        _szse_security(szse_intervals, signal_date),
    ], ignore_index=True)
    security_map = {row["code"]: row for row in security.to_dict("records")}
    rows = []
    for record in closing.to_dict("records"):
        code = record["code"]
        sec = security_map.get(code)
        security_status, eligibility_status = _eligible(sec)
        rows.append({
            "candidate_id": f"{signal_date}:{code}:CLOSING_LIMIT_UP",
            "signal_date": signal_date, "code": code, "name": record["name"],
            "boards": int(record["boards"]),
            "board_group": "FIRST_BOARD" if int(record["boards"]) == 1 else "CONSECUTIVE_BOARD",
            "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True,
            "is_limit_up": True, "touched_limit": True,
            "touch_status": record["classification_status"],
            "limit_up_status": record["classification_status"], "identity_status": "verified",
            "snapshot_kind": "closing", "observed_at": signal_date + "T15:00:00+08:00",
            "available_at": reconciled_manifest["created_at"],
            "provider_data_status": "verified_or_reconciled",
            "source": "Eastmoney pool reconciled against an independent archived-roster close-quote scan",
            "change_pct": None, "last_price": record.get("close"),
            "amount": record.get("amount"), "float_cap": None,
            "turnover": record.get("turnover"), "first_seal_time": record.get("first_seal_time"),
            "last_seal_time": record.get("last_seal_time"),
            "failed_seal_count": record.get("failed_seal_count"),
            "limit_statistics": None, "industry_provider": record.get("provider_industry"),
            "security_status": security_status,
            "security_evidence_status": "official_current_day_match" if sec else "missing",
            "is_st": None if not sec else sec["st_status"] != "NORMAL",
            "is_delisting_period": None if not sec else sec["listing_status"] != "LISTED",
            "no_price_limit": None if not sec else sec["price_limit_regime"] != "MAINBOARD_10",
            "security_source": None if not sec else sec["source_ref"],
            "security_source_type": None if not sec else sec["source_type"],
            "eligibility_status": eligibility_status,
        })
    for record in failed.to_dict("records"):
        code = record["code"]
        sec = security_map.get(code)
        security_status, eligibility_status = _eligible(sec)
        raw = raw_failed_map.get(code, {})
        rows.append({
            "candidate_id": f"{signal_date}:{code}:FAILED_BOARD_OBSERVATION",
            "signal_date": signal_date, "code": code, "name": record["name"],
            "boards": 0, "board_group": "FAILED_BOARD",
            "pool_assignment": "FAILED_BOARD_OBSERVATION", "is_mainboard": True,
            "is_limit_up": False, "touched_limit": True,
            "touch_status": record["classification_status"],
            "limit_up_status": "FAILED_TO_CLOSE_AT_LIMIT", "identity_status": "not_applicable",
            "snapshot_kind": "closing", "observed_at": signal_date + "T15:00:00+08:00",
            "available_at": reconciled_manifest["created_at"],
            "provider_data_status": "inferred_daily_high_proxy",
            "source": "Independent public daily-high quote proxy reconciled against Eastmoney failed pool",
            "change_pct": record.get("pct"), "last_price": record.get("close"),
            "amount": float(record["amount_wan"]) * 10000 if pd.notna(record.get("amount_wan")) else None,
            "float_cap": None, "turnover": record.get("turnover"),
            "first_seal_time": raw.get("首次封板时间"), "last_seal_time": None,
            "failed_seal_count": raw.get("炸板次数"), "limit_statistics": raw.get("涨停统计"),
            "industry_provider": raw.get("所属行业"),
            "security_status": security_status,
            "security_evidence_status": "official_current_day_match" if sec else "missing",
            "is_st": None if not sec else sec["st_status"] != "NORMAL",
            "is_delisting_period": None if not sec else sec["listing_status"] != "LISTED",
            "no_price_limit": None if not sec else sec["price_limit_regime"] != "MAINBOARD_10",
            "security_source": None if not sec else sec["source_ref"],
            "security_source_type": None if not sec else sec["source_type"],
            "eligibility_status": eligibility_status,
        })
    output = pd.DataFrame(rows)
    if output.candidate_id.duplicated().any():
        raise ValueError("duplicate standardized candidate id")
    close_codes = set(output.loc[output.pool_assignment.eq("CLOSING_LIMIT_UP"), "code"])
    failed_codes = set(output.loc[output.pool_assignment.eq("FAILED_BOARD_OBSERVATION"), "code"])
    if close_codes & failed_codes:
        raise ValueError("closing and failed pools overlap")
    closing_out = output[output.pool_assignment.eq("CLOSING_LIMIT_UP")]
    failed_out = output[output.pool_assignment.eq("FAILED_BOARD_OBSERVATION")]
    counts = closing_out.boards.value_counts().sort_index()
    report = {
        "status": "RECONCILED_POOL_WITH_OFFICIAL_CURRENT_DAY_ELIGIBILITY",
        "signal_date": signal_date, "available_at": reconciled_manifest["created_at"],
        "mainboard_closing_limit_rows": int(len(closing_out)),
        "mainboard_first_board_rows": int(closing_out.boards.eq(1).sum()),
        "mainboard_consecutive_board_rows": int(closing_out.boards.ge(2).sum()),
        "mainboard_board_level_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "mainboard_failed_board_rows": int(len(failed_out)),
        "verified_eligible_closing_rows": int(closing_out.eligibility_status.eq("VERIFIED_ELIGIBLE").sum()),
        "security_unavailable_or_ineligible_closing_rows": int(
            closing_out.eligibility_status.ne("VERIFIED_ELIGIBLE").sum()),
        "verified_eligible_failed_board_rows": int(failed_out.eligibility_status.eq("VERIFIED_ELIGIBLE").sum()),
        "security_unavailable_or_ineligible_failed_board_rows": int(
            failed_out.eligibility_status.ne("VERIFIED_ELIGIBLE").sum()),
        "point_in_time_security_scope_complete": bool(output.security_status.eq("verified").all()),
        "main_and_failed_pools_disjoint": True,
        "provider_pool_status": "reconciled_two_source_with_quote_proxy",
        "full_market_completeness": "not_proven_archived_roster_only",
        "complete_within_archived_roster": bool(reconciled_manifest.get("complete_within_archived_roster")),
        "entry_authorized": False,
        "unavailable_fields": ["Level-2 queue", "auction cancellation sequence", "tick execution",
                               "last seal/open count for quote-only rows", "float cap for quote-only rows"],
        "limitations": [
            "Independent quote scan uses a current official roster and cannot by itself prove historical full-market completeness",
            "failed-board classification is inferred from daily high equaling the 10% limit price",
            "SSE/SZSE official current roster was collected after close for next-session planning",
            "no next-session auction or opening confirmation exists at closing-plan stage",
        ],
    }
    return output, report


def run(args):
    paths = [Path(args.closing), Path(args.failed), Path(args.reconciled_manifest),
             Path(args.raw_failed), Path(args.sse_current), Path(args.szse_intervals)]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable review snapshot")
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    sse_captured = datetime.fromtimestamp(paths[4].stat().st_mtime, timezone.utc).isoformat()
    frame, report = build(
        pd.read_csv(paths[0], dtype={"code": str}),
        pd.read_csv(paths[1], dtype={"code": str}), manifest,
        pd.read_csv(paths[3], dtype={"代码": str}),
        pd.read_csv(paths[4], dtype={"A_STOCK_CODE": str}),
        pd.read_csv(paths[5], dtype={"code": str}), args.date, sse_captured,
    )
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "all_closing_and_failed_candidates.csv", index=False, encoding="utf-8-sig")
    report["created_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["source_hashes"] = {str(path.resolve()): sha256_file(path) for path in paths + [Path(__file__)]}
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closing", required=True)
    parser.add_argument("--failed", required=True)
    parser.add_argument("--reconciled-manifest", required=True)
    parser.add_argument("--raw-failed", required=True)
    parser.add_argument("--sse-current", required=True)
    parser.add_argument("--szse-intervals", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
