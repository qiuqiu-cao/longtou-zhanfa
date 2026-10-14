"""Collect point-in-time severe-abnormal reset evidence for current candidates.

SSE main-board coverage comes from the exchange's official daily severe-
abnormal feed (refType Z3-Z8).  When every exchange trading session from the
active rule's effective date through the signal date is successfully queried,
the absence of a security is valid negative evidence for that bounded period.

SZSE coverage currently uses the official CNINFO company-disclosure index.
An explicit ``严重异常波动`` notice is positive evidence that an event occurred,
but a zero-result company search is *not* exchange-exhaustive and must never be
promoted to a verified no-reset statement.  Ordinary ``异常波动`` notices do not
count as severe events on either exchange.

This module produces reset chronology evidence only.  It does not calculate an
official regulatory distance, rank candidates, estimate probabilities, or
authorize a trade.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


VERSION = "v192-current-severe-abnormal-reset-chronology-1"
RULE_EFFECTIVE_DATE = "2026-07-06"
SSE_DAILY_URL = (
    "https://query.sse.com.cn/marketdata/tradedata/"
    "queryAllTradeOpenDate.do"
)
SSE_PAGE = "https://www.sse.com.cn/disclosure/diclosure/ycjyxx/main/"
CNINFO_STOCK_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_PAGE = "https://www.cninfo.com.cn/new/index"
SEVERE_TITLE = "严重异常波动"
SSE_REF_TYPES = {
    "Z3": "10D_FOUR_SAME_DIRECTION_POSITIVE",
    "Z4": "10D_FOUR_SAME_DIRECTION_NEGATIVE",
    "Z5": "10D_POSITIVE_100",
    "Z6": "10D_NEGATIVE_50",
    "Z7": "30D_POSITIVE_200",
    "Z8": "30D_NEGATIVE_70",
}
SSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": SSE_PAGE}
CNINFO_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": CNINFO_PAGE}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_markup(value: Any) -> str:
    return re.sub(r"<[^>]+>", "", str(value or "")).strip()


def _normalise_date(value: Any) -> str:
    text = re.sub(r"\D", "", str(value or ""))
    if len(text) != 8:
        raise ValueError(f"invalid date: {value!r}")
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def parse_sse_daily_payload(payload: dict[str, Any], query_date: str) -> list[dict[str, Any]]:
    """Extract severe events while rejecting a response for the wrong date."""
    page_help = payload.get("pageHelp")
    if not isinstance(page_help, dict):
        raise ValueError("SSE response lacks pageHelp")
    raw_rows = page_help.get("data") or []
    if not isinstance(raw_rows, list):
        raise ValueError("SSE pageHelp.data is not a list")
    compact_query_date = query_date.replace("-", "")
    events: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        ref_type = str(row.get("refType") or "")
        if ref_type not in SSE_REF_TYPES:
            continue
        trade_date = re.sub(r"\D", "", str(row.get("tradeDate") or ""))
        if trade_date != compact_query_date:
            raise ValueError(
                f"SSE severe row date {trade_date!r} does not match query {compact_query_date}"
            )
        abnormal_start = row.get("abnormalStart") or row.get("unnormalStart")
        abnormal_end = row.get("abnormalEnd") or row.get("unnormalEnd")
        events.append({
            "code": str(row.get("secCode") or "").zfill(6),
            "name": str(row.get("secAbbr") or ""),
            "exchange_publication_date": _normalise_date(trade_date),
            "abnormal_start": _normalise_date(abnormal_start),
            "abnormal_end": _normalise_date(abnormal_end),
            "ref_type": ref_type,
            "event_type": SSE_REF_TYPES[ref_type],
            "sec_value": row.get("secValue") or row.get("secIndct"),
            "source": SSE_PAGE,
            "source_level": "SSE_OFFICIAL_EXCHANGE_DAILY_FEED",
        })
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        key = (event["code"], event["exchange_publication_date"], event["ref_type"])
        unique[key] = event
    return sorted(unique.values(), key=lambda x: (x["exchange_publication_date"], x["code"], x["ref_type"]))


def parse_cninfo_payload(payload: dict[str, Any], code: str) -> list[dict[str, Any]]:
    announcements = payload.get("announcements") or []
    if not isinstance(announcements, list):
        raise ValueError("CNINFO announcements is not a list")
    events: list[dict[str, Any]] = []
    for row in announcements:
        if not isinstance(row, dict):
            continue
        row_code = str(row.get("secCode") or "").zfill(6)
        if row_code != code:
            raise ValueError(f"CNINFO returned {row_code} for requested {code}")
        title = _strip_markup(row.get("announcementTitle"))
        if SEVERE_TITLE not in title:
            continue
        timestamp = pd.to_datetime(row.get("announcementTime"), unit="ms", utc=True)
        adjunct = str(row.get("adjunctUrl") or "")
        events.append({
            "code": code,
            "name": str(row.get("secName") or ""),
            "company_announcement_date": timestamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d"),
            "title": title,
            "announcement_id": str(row.get("announcementId") or ""),
            "url": "https://static.cninfo.com.cn/" + adjunct.lstrip("/"),
            "source": CNINFO_PAGE,
            "source_level": "CNINFO_OFFICIAL_COMPANY_DISCLOSURE_INDEX",
        })
    unique: dict[str, dict[str, Any]] = {}
    for event in events:
        unique[event["announcement_id"] or event["url"]] = event
    return sorted(unique.values(), key=lambda x: (x["company_announcement_date"], x["code"]))


def next_session(calendar: list[str], event_date: str) -> str | None:
    return next((date for date in calendar if date > event_date), None)


def build_candidate_evidence(
    candidates: pd.DataFrame,
    sse_calendar: list[str],
    sse_query_success: dict[str, bool],
    sse_events: list[dict[str, Any]],
    cninfo_query_status: dict[str, dict[str, Any]],
    cninfo_events: list[dict[str, Any]],
) -> pd.DataFrame:
    complete_sse = bool(sse_calendar) and all(sse_query_success.get(date) is True for date in sse_calendar)
    rows: list[dict[str, Any]] = []
    for item in candidates.sort_values("code").itertuples(index=False):
        code = str(item.code).zfill(6)
        is_sse = code.startswith("6")
        if is_sse:
            events = [event for event in sse_events if event["code"] == code]
            latest = events[-1] if events else None
            if complete_sse:
                status = (
                    "verified_exchange_events_present"
                    if events else "verified_no_exchange_event_since_rule_effective"
                )
                chronology_complete = True
            else:
                status = "unavailable_incomplete_sse_daily_coverage"
                chronology_complete = False
            publication_date = latest["exchange_publication_date"] if latest else None
            reset_session = next_session(sse_calendar, publication_date) if publication_date else None
            source_level = "SSE_OFFICIAL_EXCHANGE_DAILY_FEED"
            query_coverage = f"{min(sse_calendar) if sse_calendar else ''}~{max(sse_calendar) if sse_calendar else ''}"
        else:
            events = [event for event in cninfo_events if event["code"] == code]
            latest = events[-1] if events else None
            query = cninfo_query_status.get(code) or {}
            success = query.get("status") == "SUCCESS"
            if not success:
                status = "unavailable_cninfo_query_failed"
            elif events:
                status = "company_notice_confirms_event_exchange_publication_date_unavailable"
            else:
                status = "company_index_no_event_found_not_exchange_exhaustive"
            chronology_complete = False
            publication_date = None
            reset_session = None
            source_level = "CNINFO_OFFICIAL_COMPANY_DISCLOSURE_INDEX"
            query_coverage = str(query.get("query_coverage") or "")
        rows.append({
            "code": code,
            "name": getattr(item, "name", ""),
            "exchange": "SSE" if is_sse else "SZSE",
            "reset_chronology_status": status,
            "official_exchange_chronology_complete": chronology_complete,
            "severe_event_count_since_rule_effective": len(events),
            "latest_exchange_publication_date": publication_date,
            "latest_reset_effective_session": reset_session,
            "source_level": source_level,
            "query_coverage": query_coverage,
            "ranking_allowed": False,
            "buy_signal": False,
        })
    return pd.DataFrame(rows)


def _query_sse_day(date: str, timeout: int = 35) -> tuple[str, dict[str, Any], str, str]:
    params = {
        "jsonCallBack": "",
        "token": "QUERY",
        "tradeDate": date.replace("-", ""),
        "flag": "",
    }
    response = requests.get(SSE_DAILY_URL, params=params, headers=SSE_HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = "utf-8"
    payload = response.json()
    parse_sse_daily_payload(payload, date)
    return date, payload, _sha256_bytes(response.content), response.url


def _cninfo_stock_map(timeout: int = 35) -> tuple[dict[str, str], dict[str, Any], str]:
    response = requests.get(CNINFO_STOCK_URL, headers=CNINFO_HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = "utf-8"
    payload = response.json()
    rows = payload.get("stockList") or []
    mapping = {
        str(row.get("code") or "").zfill(6): str(row.get("orgId") or "")
        for row in rows if row.get("code") and row.get("orgId")
    }
    return mapping, payload, _sha256_bytes(response.content)


def _query_cninfo_code(
    code: str, org_id: str, start: str, end: str, timeout: int = 35,
) -> tuple[str, dict[str, Any], str]:
    body = {
        "pageNum": "1", "pageSize": "100", "column": "szse",
        "tabName": "fulltext", "plate": "sz", "stock": f"{code},{org_id}",
        "searchkey": SEVERE_TITLE, "secid": "", "category": "", "trade": "",
        "seDate": f"{start}~{end}", "sortName": "", "sortType": "",
        "isHLtitle": "true",
    }
    response = requests.post(
        CNINFO_QUERY_URL, data=body, headers=CNINFO_HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = "utf-8"
    payload = response.json()
    if payload.get("hasMore"):
        raise ValueError(f"CNINFO severe query unexpectedly exceeds 100 rows for {code}")
    parse_cninfo_payload(payload, code)
    return code, payload, _sha256_bytes(response.content)


def run(
    candidates_path: str | Path,
    sse_benchmark_path: str | Path,
    signal_date: str,
    query_asof: str,
    output_dir: str | Path,
    workers: int = 4,
) -> dict[str, Any]:
    candidates_path = Path(candidates_path)
    benchmark_path = Path(sse_benchmark_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V192 archive")
    candidates = pd.read_csv(candidates_path, dtype={"code": str})
    if not {"code", "name"}.issubset(candidates):
        raise ValueError("candidate file lacks code/name")
    candidates["code"] = candidates["code"].astype(str).str.zfill(6)
    if candidates["code"].duplicated().any():
        raise ValueError("candidate codes are not unique")
    benchmark = pd.read_csv(benchmark_path)
    if "date" not in benchmark:
        raise ValueError("benchmark calendar lacks date")
    all_calendar = sorted(pd.to_datetime(benchmark["date"], errors="raise").dt.strftime("%Y-%m-%d").unique())
    sse_calendar = [date for date in all_calendar if RULE_EFFECTIVE_DATE <= date <= signal_date]
    if not sse_calendar or sse_calendar[-1] != signal_date:
        raise ValueError("SSE benchmark does not cover signal date")

    max_workers = max(1, min(4, int(workers)))
    sse_raw: dict[str, Any] = {}
    sse_status: dict[str, bool] = {date: False for date in sse_calendar}
    sse_audit: list[dict[str, Any]] = []
    sse_events: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        jobs = {executor.submit(_query_sse_day, date): date for date in sse_calendar}
        for future in futures.as_completed(jobs):
            date = jobs[future]
            try:
                _, payload, digest, url = future.result()
                sse_raw[date] = payload
                events = parse_sse_daily_payload(payload, date)
                sse_events.extend(events)
                sse_status[date] = True
                sse_audit.append({
                    "date": date, "status": "SUCCESS", "response_sha256": digest,
                    "event_rows": len(events), "url": url, "error": "",
                })
            except Exception as exc:
                sse_audit.append({
                    "date": date, "status": "FAILED", "response_sha256": "",
                    "event_rows": 0, "url": SSE_DAILY_URL,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                })

    szse_codes = sorted(code for code in candidates["code"] if not code.startswith("6"))
    cninfo_raw: dict[str, Any] = {}
    cninfo_events: list[dict[str, Any]] = []
    cninfo_status: dict[str, dict[str, Any]] = {}
    stock_map_error = ""
    stock_map_payload: dict[str, Any] = {}
    stock_map_hash = ""
    try:
        stock_map, stock_map_payload, stock_map_hash = _cninfo_stock_map()
    except Exception as exc:
        stock_map = {}
        stock_map_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        jobs = {}
        for code in szse_codes:
            org_id = stock_map.get(code)
            if not org_id:
                cninfo_status[code] = {
                    "status": "FAILED", "query_coverage": f"{RULE_EFFECTIVE_DATE}~{query_asof}",
                    "response_sha256": "", "result_count": 0,
                    "error": stock_map_error or "ORG_ID_MISSING",
                }
                continue
            jobs[executor.submit(
                _query_cninfo_code, code, org_id, RULE_EFFECTIVE_DATE, query_asof
            )] = code
        for future in futures.as_completed(jobs):
            code = jobs[future]
            try:
                _, payload, digest = future.result()
                events = parse_cninfo_payload(payload, code)
                cninfo_raw[code] = payload
                cninfo_events.extend(events)
                cninfo_status[code] = {
                    "status": "SUCCESS", "query_coverage": f"{RULE_EFFECTIVE_DATE}~{query_asof}",
                    "response_sha256": digest, "result_count": len(events), "error": "",
                }
            except Exception as exc:
                cninfo_status[code] = {
                    "status": "FAILED", "query_coverage": f"{RULE_EFFECTIVE_DATE}~{query_asof}",
                    "response_sha256": "", "result_count": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }

    evidence = build_candidate_evidence(
        candidates, sse_calendar, sse_status, sse_events, cninfo_status, cninfo_events)
    target.mkdir(parents=True, exist_ok=False)
    evidence_path = target / "candidate_reset_chronology.csv"
    sse_event_path = target / "sse_official_severe_events.csv"
    sse_audit_path = target / "sse_daily_query_audit.csv"
    cninfo_event_path = target / "szse_company_severe_notices.csv"
    cninfo_audit_path = target / "szse_company_query_audit.csv"
    sse_raw_path = target / "sse_daily_raw.json"
    cninfo_raw_path = target / "szse_company_query_raw.json"
    stock_map_path = target / "cninfo_stock_map_raw.json"
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sse_events).to_csv(sse_event_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sse_audit).sort_values("date").to_csv(
        sse_audit_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(cninfo_events).to_csv(cninfo_event_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"code": code, **status} for code, status in sorted(cninfo_status.items())
    ]).to_csv(cninfo_audit_path, index=False, encoding="utf-8-sig")
    sse_raw_path.write_text(json.dumps(sse_raw, ensure_ascii=False), encoding="utf-8")
    cninfo_raw_path.write_text(json.dumps(cninfo_raw, ensure_ascii=False), encoding="utf-8")
    stock_map_path.write_text(json.dumps(stock_map_payload, ensure_ascii=False), encoding="utf-8")

    top_codes = {"000993", "605058"}
    report = {
        "version": VERSION,
        "status": "PARTIAL_OFFICIAL_RESET_EVIDENCE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": signal_date,
        "query_asof": query_asof,
        "rule_effective_date": RULE_EFFECTIVE_DATE,
        "candidate_count": int(len(evidence)),
        "sse_candidate_count": int(evidence["exchange"].eq("SSE").sum()),
        "szse_candidate_count": int(evidence["exchange"].eq("SZSE").sum()),
        "sse_exchange_sessions_expected": len(sse_calendar),
        "sse_exchange_sessions_successful": int(sum(sse_status.values())),
        "sse_official_severe_event_count_all_codes": len(sse_events),
        "candidate_official_exchange_chronology_complete_count": int(
            evidence["official_exchange_chronology_complete"].sum()),
        "candidate_verified_no_reset_count": int(evidence[
            "reset_chronology_status"].eq(
                "verified_no_exchange_event_since_rule_effective").sum()),
        "candidate_verified_exchange_event_count": int(evidence[
            "reset_chronology_status"].eq("verified_exchange_events_present").sum()),
        "szse_company_query_success_count": int(sum(
            value.get("status") == "SUCCESS" for value in cninfo_status.values())),
        "szse_company_severe_notice_count": len(cninfo_events),
        "priority_candidates": evidence[evidence["code"].isin(top_codes)].replace(
            {pd.NA: None, float("nan"): None}).to_dict("records"),
        "official_regulatory_distance_count": 0,
        "candidate_reranking_allowed": False,
        "buy_signal": False,
        "entry_authorized": False,
        "governance": {
            "ordinary_abnormal_notice_is_reset": False,
            "sse_full_daily_coverage_can_verify_bounded_absence": True,
            "cninfo_zero_result_can_verify_exchange_absence": False,
            "company_notice_date_may_replace_exchange_publication_date": False,
            "reset_evidence_alone_can_rank_or_buy": False,
        },
        "interpretation": (
            "SSE candidates may receive bounded official reset chronology when every "
            "exchange session is covered. SZSE company-notice searches remain partial: "
            "positive severe notices confirm an event, while zero results do not prove "
            "exchange-level absence. No row is an official distance or trading signal."
        ),
        "sources": [SSE_PAGE, SSE_DAILY_URL, CNINFO_PAGE, CNINFO_QUERY_URL],
        "source_hashes": {
            str(candidates_path.resolve()): _sha256(candidates_path),
            str(benchmark_path.resolve()): _sha256(benchmark_path),
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
            "cninfo_stock_list_response": stock_map_hash,
        },
        "files": {},
    }
    for path in [
        evidence_path, sse_event_path, sse_audit_path, cninfo_event_path,
        cninfo_audit_path, sse_raw_path, cninfo_raw_path, stock_map_path,
    ]:
        report["files"][path.name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    report_path = target / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sse-benchmark", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--query-asof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(
        args.candidates, args.sse_benchmark, args.signal_date,
        args.query_asof, args.output, args.workers,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
