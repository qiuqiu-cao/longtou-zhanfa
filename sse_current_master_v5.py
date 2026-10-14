#!/usr/bin/env python3
"""Build current-SSE point-in-time eligibility from official action notices.

The collector searches the complete development window month by month for
titles containing risk-warning or short-name terms, then restricts results to
SSE codes still missing from the candidate security audit. Exact transitions
must be proven by PDF text. Metadata-only changes remain inferred.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from collect_security_master_v4 import name_status
from security_master_v4 import MASTER_COLUMNS, validate_master
from sse_name_history_v4 import (
    ANNOUNCEMENT_PAGE, HEADERS, QUERY_URL, download_pdf, extract_pdf_text,
    flatten_result, is_event_title, parse_event_text,
)


ROOT = Path(__file__).resolve().parent
GAP_AUDIT = ROOT / "results" / "v4" / "security_master" / "combined_candidate_audit" / "security_eligibility_audit.csv"
BASE = ROOT / "data" / "v5" / "security_evidence" / "sse_current"
RAW = BASE / "raw"
PDF = BASE / "pdf"
TEXT = BASE / "text"
CURRENT_CSV = BASE / "sse_current_mainboard.csv"
INDEX_CSV = BASE / "sse_action_index.csv"
EVENTS_CSV = BASE / "sse_action_events.csv"
REVIEW_CSV = BASE / "sse_action_manual_review.csv"
INTERVALS_CSV = ROOT / "data" / "v5" / "security_master_sse_current_evidenced.csv"
MANIFEST = ROOT / "results" / "v5" / "sse_current_security_master_manifest.json"
CURRENT_URL = "https://query.sse.com.cn/sseQuery/commonQuery.do"
CURRENT_PAGE = "https://www.sse.com.cn/assortment/stock/list/share/"
VERSION = "v5-sse-current-security-1"
KEYWORDS = ("风险警示", "简称")
ACTION_INDEX_COLUMNS = [
    "BULLETIN_TYPE_DESC", "BULLETIN_YEAR", "IS_HOLDER_DISCLOSE", "ORG_BULLETIN_ID",
    "ORG_FILE_TYPE", "SECURITY_CODE", "SECURITY_NAME", "SSEDATE", "TITLE", "URL",
]
EVENT_COLUMNS = [
    "code", "announcement_date", "security_name_at_announcement", "title", "url",
    "bulletin_id", "pdf_sha256", "text_sha256", "parse_status", "before_name",
    "after_name", "effective_date", "after_status", "transition_pattern", "date_pattern",
]
REVIEW_RESOLUTIONS = {"VERIFIED_NO_STATE_TRANSITION", "VERIFIED_CORROBORATING_EVENT"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: Any) -> str:
    return "".join(str(value or "").upper().split())


def month_segments(start: str, end: str) -> list[tuple[str, str]]:
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("start cannot exceed end")
    result = []
    cursor = first
    while cursor <= last:
        segment_end = min(cursor + pd.offsets.MonthEnd(0), last)
        result.append((cursor.strftime("%Y-%m-%d"), segment_end.strftime("%Y-%m-%d")))
        cursor = segment_end + timedelta(days=1)
    return result


def gap_codes(path: Path | None = None) -> list[str]:
    # Resolve the configured archive at call time.  Research extensions use a
    # separately frozen scope and must not inherit the V5 path captured when
    # this function was defined.
    path = GAP_AUDIT if path is None else path
    audit = pd.read_csv(path, dtype={"code": "string"})
    missing = audit[audit["security_status"].eq("UNVERIFIED_POINT_IN_TIME_STATUS")]
    codes = missing["code"].astype(str).str.zfill(6)
    if (~codes.str.startswith(("600", "601", "603", "605"))).any():
        raise ValueError("gap audit contains non-SSE mainboard code")
    return sorted(codes.unique())


def collect_current_list() -> pd.DataFrame:
    params = {
        "STOCK_TYPE": "1", "REG_PROVINCE": "", "CSRC_CODE": "", "STOCK_CODE": "",
        "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L", "COMPANY_STATUS": "2,4,5,7,8",
        "type": "inParams", "isPagination": "true", "pageHelp.cacheSize": 1,
        "pageHelp.beginPage": 1, "pageHelp.pageSize": 2000, "pageHelp.pageNo": 1,
    }
    response = requests.get(CURRENT_URL, params=params, headers={**HEADERS, "Referer": CURRENT_PAGE}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("result") or (payload.get("pageHelp") or {}).get("data") or []
    if len(rows) < 1000:
        raise RuntimeError(f"official SSE current list unexpectedly short: {len(rows)}")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "current_list.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows).drop_duplicates("A_STOCK_CODE")
    frame["A_STOCK_CODE"] = frame["A_STOCK_CODE"].astype("string").str.zfill(6)
    frame.to_csv(CURRENT_CSV, index=False, encoding="utf-8-sig")
    return frame


def query_action_month(keyword: str, start: str, end: str, timeout: int = 40) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session = requests.Session()
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "isPagination": "true", "pageHelp.pageSize": 100, "pageHelp.pageNo": page,
            "pageHelp.beginPage": page, "pageHelp.cacheSize": 1, "pageHelp.endPage": page,
            "START_DATE": start, "END_DATE": end, "SECURITY_CODE": "", "TITLE": keyword,
            "BULLETIN_TYPE": "", "stockType": "1",
        }
        response = session.get(QUERY_URL, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        raw.append(payload)
        rows.extend(flatten_result(payload))
        page_count = int((payload.get("pageHelp") or {}).get("pageCount") or 0)
        if page >= page_count:
            break
        page += 1
        time.sleep(0.05)
    return rows, raw


def filter_action_rows(collected: list[dict[str, Any]], codes: set[str]) -> pd.DataFrame:
    """Filter archived official search results to a locked code scope."""
    frame = pd.DataFrame(collected)
    required = {"SECURITY_CODE", "SECURITY_NAME", "SSEDATE", "TITLE", "URL", "ORG_BULLETIN_ID"}
    if frame.empty:
        frame = pd.DataFrame(columns=ACTION_INDEX_COLUMNS)
    elif not required.issubset(frame.columns):
        raise RuntimeError("official SSE action search returned malformed rows")
    frame["SECURITY_CODE"] = frame["SECURITY_CODE"].astype("string").str.zfill(6)
    frame = frame[frame["SECURITY_CODE"].isin(codes)].drop_duplicates("ORG_BULLETIN_ID").copy()
    frame["event_candidate"] = frame["TITLE"].astype(str).map(is_event_title)
    return frame.sort_values(["SECURITY_CODE", "SSEDATE", "ORG_BULLETIN_ID"])


def rebuild_action_index_from_raw(raw_bundle_path: Path, codes: set[str], output_path: Path) -> dict[str, Any]:
    """Re-scope an immutable official raw bundle without repeating network calls."""
    bundle = json.loads(Path(raw_bundle_path).read_text(encoding="utf-8"))
    collected: list[dict[str, Any]] = []
    for pages in bundle.values():
        for payload in pages:
            collected.extend(flatten_result(payload))
    frame = filter_action_rows(collected, {str(code).zfill(6) for code in codes})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "scope_codes": len(codes), "action_rows": int(len(frame)),
        "event_candidates": int(frame["event_candidate"].astype(bool).sum()),
        "event_candidate_codes": int(frame.loc[
            frame["event_candidate"].astype(bool), "SECURITY_CODE"].nunique()),
        "raw_bundle_sha256": sha256_file(Path(raw_bundle_path)),
        "index": str(output_path.resolve()), "sha256": sha256_file(output_path),
    }


def collect_actions(start: str = "2023-11-28", end: str = "2026-08-31", workers: int = 4) -> dict[str, Any]:
    codes = set(gap_codes())
    current = collect_current_list()
    current_codes = set(current["A_STOCK_CODE"].astype(str))
    absent = sorted(codes.difference(current_codes))
    if absent:
        raise ValueError(f"candidate gap codes absent from official current SSE list: {absent[:20]}")
    jobspec = [(keyword, left, right) for keyword in KEYWORDS for left, right in month_segments(start, end)]
    collected: list[dict[str, Any]] = []
    raw_bundle: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
        jobs = {executor.submit(query_action_month, *spec): spec for spec in jobspec}
        for future in futures.as_completed(jobs):
            keyword, left, right = jobs[future]
            key = f"{keyword}_{left}_{right}"
            try:
                rows, raw = future.result()
                collected.extend(rows)
                raw_bundle[key] = raw
            except Exception as exc:
                failures.append({"query": key, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    if failures:
        raise RuntimeError(f"official SSE action searches failed: {failures[:5]}")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "action_search_pages.json").write_text(
        json.dumps(raw_bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame = filter_action_rows(collected, codes)
    BASE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(INDEX_CSV, index=False, encoding="utf-8-sig")
    return {
        "scope_codes": len(codes), "official_current_rows": int(len(current)),
        "search_jobs": len(jobspec), "action_rows": int(len(frame)),
        "event_candidates": int(frame["event_candidate"].astype(bool).sum()),
        "event_candidate_codes": int(frame.loc[frame["event_candidate"].astype(bool), "SECURITY_CODE"].nunique()),
        "index": str(INDEX_CSV.resolve()), "sha256": sha256_file(INDEX_CSV),
    }


def download_candidates(workers: int = 4) -> dict[str, Any]:
    index = pd.read_csv(INDEX_CSV, dtype={"SECURITY_CODE": "string", "ORG_BULLETIN_ID": "string"})
    selected = index[index["event_candidate"].astype("string").str.lower().eq("true")].copy()
    PDF.mkdir(parents=True, exist_ok=True)
    TEXT.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    succeeded = 0

    def one(row: pd.Series) -> None:
        stem = f"{row.SECURITY_CODE}_{str(row.SSEDATE)[:10].replace('-', '')}_{row.ORG_BULLETIN_ID}"
        pdf_path, text_path = PDF / f"{stem}.pdf", TEXT / f"{stem}.txt"
        if not pdf_path.exists() or pdf_path.read_bytes()[:4] != b"%PDF":
            download_pdf(str(row.URL), pdf_path)
        if not text_path.exists() or text_path.stat().st_size < 30:
            extract_pdf_text(pdf_path, text_path)

    with futures.ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
        jobs = {executor.submit(one, row): row for _, row in selected.iterrows()}
        for future in futures.as_completed(jobs):
            row = jobs[future]
            try:
                future.result()
                succeeded += 1
            except Exception as exc:
                failures.append({
                    "code": str(row.SECURITY_CODE), "bulletin_id": str(row.ORG_BULLETIN_ID),
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                })
    return {"selected": int(len(selected)), "succeeded": succeeded, "failed": len(failures), "failures": failures}


def extract_events() -> dict[str, Any]:
    index = pd.read_csv(INDEX_CSV, dtype={"SECURITY_CODE": "string", "ORG_BULLETIN_ID": "string"})
    selected = index[index["event_candidate"].astype("string").str.lower().eq("true")].copy()
    rows = []
    for _, item in selected.iterrows():
        stem = f"{item.SECURITY_CODE}_{str(item.SSEDATE)[:10].replace('-', '')}_{item.ORG_BULLETIN_ID}"
        text_path = TEXT / f"{stem}.txt"
        pdf_path = PDF / f"{stem}.pdf"
        parsed = parse_event_text(text_path.read_text(encoding="utf-8", errors="replace")) if text_path.exists() else {
            "parse_status": "DOWNLOAD_OR_TEXT_MISSING", "before_name": "", "after_name": "",
            "effective_date": "", "after_status": "", "transition_pattern": "", "date_pattern": "",
        }
        if parsed["parse_status"] == "VERIFIED_EVENT" and parsed["effective_date"]:
            announced, effective = pd.Timestamp(str(item.SSEDATE)[:10]), pd.Timestamp(parsed["effective_date"])
            if effective < announced - pd.Timedelta(days=7):
                parsed["parse_status"] = "REVIEW_REQUIRED_DATE_BEFORE_ANNOUNCEMENT"
        rows.append({
            "code": str(item.SECURITY_CODE).zfill(6), "announcement_date": str(item.SSEDATE)[:10],
            "security_name_at_announcement": item.SECURITY_NAME, "title": item.TITLE,
            "url": item.URL, "bulletin_id": item.ORG_BULLETIN_ID,
            "pdf_sha256": sha256_file(pdf_path) if pdf_path.exists() else "",
            "text_sha256": sha256_file(text_path) if text_path.exists() else "", **parsed,
        })
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values(
        ["code", "announcement_date", "bulletin_id"])
    EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    verified = events[events["parse_status"].eq("VERIFIED_EVENT")]
    conflicts = verified.groupby(["code", "effective_date"])["after_name"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"conflicting verified current SSE events: {list(conflicts[conflicts > 1].index)[:5]}")
    return {
        "event_candidates": int(len(events)), "verified_events": int(len(verified)),
        "verified_codes": int(verified["code"].nunique()),
        "review_required": int(events["parse_status"].ne("VERIFIED_EVENT").sum()),
        "events": str(EVENTS_CSV.resolve()), "sha256": sha256_file(EVENTS_CSV),
    }


def regime_for_status(status: str) -> str:
    if status in {"ST", "*ST"}:
        return "ST_5"
    if status == "DELISTING_RISK":
        return "OTHER"
    return "MAINBOARD_10"


def event_chain(code: str, current_name: str, events: pd.DataFrame, start: pd.Timestamp) -> tuple[list[dict[str, Any]], list[str]]:
    part = events[(events["code"].eq(code)) & events["parse_status"].eq("VERIFIED_EVENT")].copy()
    part["effective_date"] = pd.to_datetime(part["effective_date"], errors="raise")
    part = part.sort_values(["effective_date", "announcement_date", "bulletin_id"])
    if part.empty:
        return [], []
    warnings: list[str] = []
    deduped = []
    for _, row in part.iterrows():
        event = row.to_dict()
        if deduped and event["effective_date"] == deduped[-1]["effective_date"]:
            same = normalized_name(event["before_name"]) == normalized_name(deduped[-1]["before_name"]) and normalized_name(event["after_name"]) == normalized_name(deduped[-1]["after_name"])
            if same:
                continue
        deduped.append(event)
    current = str(deduped[0]["before_name"]).strip()
    for event in deduped:
        if normalized_name(event["before_name"]) != normalized_name(current):
            warnings.append(f"{code}:{pd.Timestamp(event['effective_date']).date()}:before-chain-mismatch")
        current = str(event["after_name"]).strip()
    if normalized_name(current) != normalized_name(current_name):
        warnings.append(f"{code}:current-tail-mismatch:{current}->{current_name}")
    return deduped, warnings


def chain_is_trusted(warnings: list[str], unresolved_action_count: int) -> bool:
    """A chain is verified only when every state-change candidate is resolved."""
    return not warnings and unresolved_action_count == 0


def verified_manual_resolution_ids(events: pd.DataFrame, path: Path | None = None) -> tuple[set[tuple[str, str]], list[str]]:
    """Accept a manual resolution only when it is bound to the extracted official text hash."""
    path = REVIEW_CSV if path is None else path
    if not path.exists():
        return set(), []
    review = pd.read_csv(path, dtype="string").fillna("")
    required = {"code", "bulletin_id", "text_sha256", "resolution", "reviewed_at", "reviewer", "evidence"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"manual review ledger missing columns: {missing}")
    event_lookup = {
        (str(row.code).zfill(6), str(row.bulletin_id)): str(row.text_sha256).lower()
        for row in events.itertuples(index=False)
        if str(row.parse_status) == "REVIEW_REQUIRED"
    }
    accepted: set[tuple[str, str]] = set()
    errors: list[str] = []
    for row in review.itertuples(index=False):
        code, bulletin_id = str(row.code).zfill(6), str(row.bulletin_id)
        resolution, digest = str(row.resolution), str(row.text_sha256).lower()
        expected = event_lookup.get((code, bulletin_id))
        if resolution not in REVIEW_RESOLUTIONS:
            errors.append(f"{code}:{bulletin_id}:invalid-resolution")
        elif expected is None:
            errors.append(f"{code}:{bulletin_id}:not-review-required")
        elif digest != expected:
            errors.append(f"{code}:{bulletin_id}:text-hash-mismatch")
        elif not str(row.reviewed_at).strip() or not str(row.reviewer).strip() or not str(row.evidence).strip():
            errors.append(f"{code}:{bulletin_id}:incomplete-review-record")
        else:
            accepted.add((code, bulletin_id))
    if errors:
        raise ValueError("invalid manual review ledger: " + ";".join(errors))
    return accepted, errors


def downgrade_untrusted_intervals(rows: list[dict[str, Any]], start_index: int, trusted: bool) -> None:
    """Prevent a later verified event from masking an unresolved announcement in the chain."""
    if trusted:
        return
    for row in rows[start_index:]:
        row["data_status"] = "inferred"


def build_intervals(start: str = "2023-11-28", asof: str = "2026-08-31") -> dict[str, Any]:
    coverage_start, coverage_end = pd.Timestamp(start), pd.Timestamp(asof)
    codes = set(gap_codes())
    current = pd.read_csv(CURRENT_CSV, dtype={"A_STOCK_CODE": "string"})
    current = current[current["A_STOCK_CODE"].isin(codes)].copy()
    events = pd.read_csv(EVENTS_CSV, dtype={"code": "string", "bulletin_id": "string"}).fillna("")
    index = pd.read_csv(INDEX_CSV, dtype={"SECURITY_CODE": "string", "ORG_BULLETIN_ID": "string"}).fillna("")
    manually_resolved_ids, _ = verified_manual_resolution_ids(events)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    source_timestamp = datetime.now(timezone.utc).isoformat()
    for stock in current.itertuples(index=False):
        code_row_start = len(rows)
        code = str(stock.A_STOCK_CODE).zfill(6)
        listing = pd.to_datetime(str(stock.LIST_DATE), format="%Y%m%d", errors="raise")
        listed_start = max(listing, coverage_start)
        current_name = str(stock.SEC_NAME_CN).strip()
        chain, warnings = event_chain(code, current_name, events, listed_start)
        action_rows = index[index["SECURITY_CODE"].eq(code)]
        unverified_actions = action_rows[action_rows["event_candidate"].astype("string").str.lower().eq("true")]
        verified_ids = set(events.loc[
            events["code"].eq(code) & events["parse_status"].eq("VERIFIED_EVENT"), "bulletin_id"
        ].astype(str))
        resolved_ids = verified_ids | {bulletin_id for review_code, bulletin_id in manually_resolved_ids
                                       if review_code == code}
        unverified_actions = unverified_actions[~unverified_actions["ORG_BULLETIN_ID"].astype(str).isin(resolved_ids)]
        chain_trusted = chain_is_trusted(warnings, len(unverified_actions))
        initial_name = str(chain[0]["before_name"]).strip() if chain else current_name
        current_state, cursor = initial_name, listed_start
        transitions = []
        for event in chain:
            when = pd.Timestamp(event["effective_date"])
            if listed_start <= when <= coverage_end:
                transitions.append((when, str(event["after_name"]).strip(), "verified_import", str(event["url"])))
        for action in unverified_actions.itertuples(index=False):
            when = pd.Timestamp(str(action.SSEDATE)[:10])
            observed = str(action.SECURITY_NAME).strip()
            if listed_start <= when <= coverage_end and normalized_name(observed) != normalized_name(current_state):
                transitions.append((when, observed, "inferred", ANNOUNCEMENT_PAGE))
        transitions.append((coverage_end, current_name, "verified_import" if chain_trusted else "inferred", CURRENT_PAGE))
        transitions.sort(key=lambda item: (item[0], 0 if item[2] == "verified_import" else 1))
        state_status = "verified_import" if chain_trusted else "inferred"
        state_ref = ANNOUNCEMENT_PAGE
        for when, proposed, status, ref in transitions:
            if normalized_name(proposed) == normalized_name(current_state):
                continue
            if when > cursor:
                rows.append(_interval_row(code, current_state, cursor, when - timedelta(days=1), listing, state_status, state_ref, source_timestamp))
            current_state, cursor, state_status, state_ref = proposed, when, status, ref
        if cursor <= coverage_end:
            rows.append(_interval_row(code, current_state, cursor, coverage_end, listing, state_status, state_ref, source_timestamp))
        downgrade_untrusted_intervals(rows, code_row_start, chain_trusted)
        audits.append({
            "code": code, "current_name": current_name, "verified_events": len(chain),
            "unverified_action_notices": int(len(unverified_actions)), "chain_trusted": chain_trusted,
            "warnings": ";".join(warnings),
        })
    intervals = validate_master(pd.DataFrame(rows, columns=MASTER_COLUMNS))
    INTERVALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    intervals.loc[:, MASTER_COLUMNS].to_csv(INTERVALS_CSV, index=False, encoding="utf-8-sig")
    audit_path = BASE / "sse_current_chain_audit.csv"
    audit = pd.DataFrame(audits)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    inferred = intervals["data_status"].ne("verified_import")
    metadata = {
        "version": VERSION, "status": "PARTIAL_VERIFIED_CURRENT_SSE",
        "coverage_start": start, "asof": asof, "scope_codes": len(codes),
        "current_codes_found": int(current["A_STOCK_CODE"].nunique()),
        "interval_rows": int(len(intervals)), "verified_interval_rows": int((~inferred).sum()),
        "inferred_interval_rows": int(inferred.sum()),
        "trusted_chain_codes": int(audit["chain_trusted"].sum()),
        "untrusted_chain_codes": int((~audit["chain_trusted"]).sum()),
        "source_timestamp": source_timestamp,
        "sources": [CURRENT_PAGE, ANNOUNCEMENT_PAGE],
        "deployment_effect": "only verified_import current-SSE intervals may pass candidate eligibility",
        "files": {
            "current": {"path": str(CURRENT_CSV.resolve()), "sha256": sha256_file(CURRENT_CSV)},
            "index": {"path": str(INDEX_CSV.resolve()), "sha256": sha256_file(INDEX_CSV)},
            "events": {"path": str(EVENTS_CSV.resolve()), "sha256": sha256_file(EVENTS_CSV)},
            "intervals": {"path": str(INTERVALS_CSV.resolve()), "sha256": sha256_file(INTERVALS_CSV)},
            "audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        },
    }
    if REVIEW_CSV.exists():
        metadata["files"]["manual_review"] = {
            "path": str(REVIEW_CSV.resolve()), "sha256": sha256_file(REVIEW_CSV),
        }
        metadata["accepted_manual_resolutions"] = len(manually_resolved_ids)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(MANIFEST.resolve())
    return metadata


def _interval_row(code: str, name: str, start: pd.Timestamp, end: pd.Timestamp, listing: pd.Timestamp,
                  data_status: str, source_ref: str, source_timestamp: str) -> dict[str, Any]:
    status = name_status(name)
    return {
        "code": code, "name": name, "valid_from": start.strftime("%Y-%m-%d"),
        "valid_to": end.strftime("%Y-%m-%d"), "exchange": "SSE", "board": "MAINBOARD",
        "listing_status": "LISTED", "st_status": status, "price_limit_regime": regime_for_status(status),
        "listing_date": listing.strftime("%Y-%m-%d"), "delisting_date": "",
        "source_type": "sse_official", "source_ref": f"{CURRENT_PAGE}|{source_ref}",
        "source_timestamp": source_timestamp, "data_status": data_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current SSE candidate point-in-time security master")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--start", default="2023-11-28")
    collect.add_argument("--end", default="2026-08-31")
    collect.add_argument("--workers", type=int, default=4)
    download = sub.add_parser("download")
    download.add_argument("--workers", type=int, default=4)
    sub.add_parser("extract")
    build = sub.add_parser("build")
    build.add_argument("--start", default="2023-11-28")
    build.add_argument("--asof", default="2026-08-31")
    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--start", default="2023-11-28")
    all_cmd.add_argument("--end", default="2026-08-31")
    all_cmd.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        result = collect_actions(args.start, args.end, args.workers)
    elif args.command == "download":
        result = download_candidates(args.workers)
    elif args.command == "extract":
        result = extract_events()
    elif args.command == "build":
        result = build_intervals(args.start, args.asof)
    else:
        result = {"collect": collect_actions(args.start, args.end, args.workers)}
        result["download"] = download_candidates(args.workers)
        result["extract"] = extract_events()
        result["build"] = build_intervals(args.start, args.end)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
