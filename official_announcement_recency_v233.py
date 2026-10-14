"""Scan official announcement indexes for every main-board driver claim.

This is a point-in-time triage layer between V231 secondary explanations and
V232 official full-text verification.  It queries SSE and CNINFO company
announcement indexes, classifies titles, and creates a complete verification
queue.  A title match is only a candidate source; it never proves the claim,
its novelty, or a buy signal.  Absence from the bounded company-announcement
window is not evidence that no policy/news/interactive catalyst exists.
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

from current_reset_chronology_v192 import (
    CNINFO_HEADERS, CNINFO_QUERY_URL, _cninfo_stock_map,
)
from sse_name_history_v4 import query_segment as query_sse_segment


VERSION = "v233-official-announcement-recency-1"
CNINFO_PAGE_SIZE = 100
TITLE_RULES = (
    ("CONTROL_OR_CAPITAL_ACTION", r"回购|增持|减持|控制权|实际控制人|控股股东|权益变动|股份转让|协议转让|定向增发|向特定对象发行|重大资产|收购|重组"),
    ("PERFORMANCE_DISCLOSURE", r"业绩预告|业绩快报|半年度报告|年度报告|季度报告"),
    ("ORDER_OR_PROJECT", r"中标|合同|订单|项目进展|合作协议"),
    ("NEGATIVE_OR_CLARIFICATION", r"澄清|风险提示|异常波动|监管|问询|立案|处罚|不涉及|未开展"),
    ("PERIODIC_OR_BUSINESS_DISCLOSURE", r"半年度报告|年度报告|季度报告|投资者关系活动记录"),
)
CLAIM_COMPATIBLE_TITLES = {
    "CONTROL_OR_CAPITAL_ACTION": {"CONTROL_OR_CAPITAL_ACTION"},
    "PERFORMANCE_CLAIM": {"PERFORMANCE_DISCLOSURE"},
    "ORDER_OR_PROJECT": {"ORDER_OR_PROJECT"},
    "NEGATIVE_OR_LIMITED_EXPOSURE": {"NEGATIVE_OR_CLARIFICATION"},
    "PRODUCT_OR_BUSINESS_DESCRIPTION": {"PERIODIC_OR_BUSINESS_DISCLOSURE"},
}
HIGH_IMPACT_CLAIMS = {
    "CONTROL_OR_CAPITAL_ACTION", "PERFORMANCE_CLAIM", "ORDER_OR_PROJECT",
    "NEGATIVE_OR_LIMITED_EXPOSURE", "POLICY_OR_LIST_INCLUSION",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strip_markup(value: Any) -> str:
    return re.sub(r"<[^>]+>", "", str(value or "")).strip()


def is_mainboard(code: str) -> bool:
    return bool(re.fullmatch(r"(?:60[0135]|00[0123])\d{3}", str(code).zfill(6)))


def classify_title(title: Any) -> str:
    text = _strip_markup(title)
    for category, pattern in TITLE_RULES:
        if re.search(pattern, text):
            return category
    return "OTHER_OFFICIAL_DISCLOSURE"


def title_matches_claim(claim_category: str, title_category: str) -> bool:
    return title_category in CLAIM_COMPATIBLE_TITLES.get(str(claim_category), set())


def parse_sse_rows(rows: list[dict[str, Any]], code: str,
                   collected_at: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        row_code = str(row.get("SECURITY_CODE") or "").zfill(6)
        if row_code != code:
            raise ValueError(f"SSE returned {row_code} for requested {code}")
        date = pd.Timestamp(str(row.get("SSEDATE"))[:10]).strftime("%Y-%m-%d")
        title = _strip_markup(row.get("TITLE"))
        url = str(row.get("URL") or "")
        if url and not url.startswith("http"):
            url = "https://www.sse.com.cn" + ("" if url.startswith("/") else "/") + url
        result.append({
            "code": code, "name": str(row.get("SECURITY_NAME") or ""),
            "announcement_date": date, "published_at_status": "DATE_ONLY",
            "title": title, "title_category": classify_title(title), "url": url,
            "announcement_id": str(row.get("ORG_BULLETIN_ID") or ""),
            "source_level": "SSE_OFFICIAL_COMPANY_DISCLOSURE_INDEX",
            "collected_at": collected_at,
        })
    return result


def parse_cninfo_rows(payload: dict[str, Any], code: str,
                      collected_at: str) -> list[dict[str, Any]]:
    result = []
    for row in payload.get("announcements") or []:
        row_code = str(row.get("secCode") or "").zfill(6)
        if row_code != code:
            raise ValueError(f"CNINFO returned {row_code} for requested {code}")
        timestamp = pd.to_datetime(row.get("announcementTime"), unit="ms", utc=True)
        date = timestamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
        title = _strip_markup(row.get("announcementTitle"))
        adjunct = str(row.get("adjunctUrl") or "")
        result.append({
            "code": code, "name": str(row.get("secName") or ""),
            "announcement_date": date, "published_at_status": "DATE_ONLY",
            "title": title, "title_category": classify_title(title),
            "url": "https://static.cninfo.com.cn/" + adjunct.lstrip("/"),
            "announcement_id": str(row.get("announcementId") or adjunct),
            "source_level": "CNINFO_OFFICIAL_COMPANY_DISCLOSURE_INDEX",
            "collected_at": collected_at,
        })
    return result


def _query_cninfo(code: str, org_id: str, start: str, end: str,
                  timeout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows, raw, hashes = [], [], []

    def one_range(range_start: str, range_end: str, depth: int = 0) -> None:
        if depth > 12:
            raise ValueError(f"CNINFO date segmentation exceeds safety cap for {code}")
        body = {
            "pageNum": "1", "pageSize": str(CNINFO_PAGE_SIZE), "column": "szse",
            "tabName": "fulltext", "plate": "sz", "stock": f"{code},{org_id}",
            "searchkey": "", "secid": "", "category": "", "trade": "",
            "seDate": f"{range_start}~{range_end}", "sortName": "", "sortType": "",
            "isHLtitle": "true",
        }
        response = requests.post(
            CNINFO_QUERY_URL, data=body, headers=CNINFO_HEADERS, timeout=timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        payload = response.json()
        raw.append(payload)
        hashes.append(_sha_bytes(response.content))
        if payload.get("hasMore"):
            first, last = pd.Timestamp(range_start), pd.Timestamp(range_end)
            if first >= last:
                raise ValueError(f"CNINFO one-day range exceeds {CNINFO_PAGE_SIZE} rows for {code}")
            middle = first + (last - first) // 2
            one_range(first.strftime("%Y-%m-%d"), middle.strftime("%Y-%m-%d"), depth + 1)
            one_range((middle + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      last.strftime("%Y-%m-%d"), depth + 1)
            return
        rows.extend(payload.get("announcements") or [])

    one_range(start, end)
    unique = {}
    for row in rows:
        key = str(row.get("announcementId") or row.get("adjunctUrl") or "")
        unique[key] = row
    return list(unique.values()), raw, hashes


def build_overlays(claims: pd.DataFrame, announcements: pd.DataFrame,
                   signal_date: str, cutoff: str,
                   query_status: dict[str, dict[str, Any]],
                   role_codes: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        raise ValueError("cutoff must include timezone")
    signal = pd.Timestamp(signal_date)
    if cutoff_ts.tz_convert("Asia/Shanghai").date() != signal.date():
        raise ValueError("cutoff date must equal signal date")
    claims = claims.copy()
    claims["code"] = claims.code.astype(str).str.zfill(6)
    announcements = announcements.copy()
    if len(announcements):
        announcements["code"] = announcements.code.astype(str).str.zfill(6)
        announcements["announcement_date"] = pd.to_datetime(
            announcements.announcement_date, errors="raise").dt.strftime("%Y-%m-%d")
        # Official indexes used here expose only the date.  Same-day rows are
        # conservatively unavailable at a 15:00 close decision.
        announcements["known_before_cutoff"] = (
            announcements.announcement_date < signal_date)
    else:
        announcements["known_before_cutoff"] = pd.Series(dtype=bool)
    role_codes = role_codes or set()
    overlays, queue = [], []
    for claim in claims.sort_values("code").to_dict("records"):
        code = str(claim["code"]).zfill(6)
        status = query_status.get(code, {})
        part = announcements[
            announcements.code.eq(code) & announcements.known_before_cutoff].copy()
        if len(part):
            part["claim_title_compatible"] = part.title_category.map(
                lambda value: title_matches_claim(claim.get("driver_claim_category"), value))
        matched = part[part.get("claim_title_compatible", pd.Series(False, index=part.index))]
        matched = matched.sort_values(["announcement_date", "announcement_id"], ascending=False)
        if status.get("status") != "SUCCESS":
            recency = "UNAVAILABLE_QUERY_FAILED"
        elif len(matched):
            recency = "RECENT_OFFICIAL_TITLE_CANDIDATE"
        else:
            recency = "NO_CATEGORY_MATCHED_TITLE_IN_BOUNDED_WINDOW"
        latest = matched.iloc[0] if len(matched) else None
        age = ((signal - pd.Timestamp(latest.announcement_date)).days
               if latest is not None else None)
        overlays.append({
            "signal_date": signal_date, "code": code, "name": claim.get("name"),
            "is_mainboard": is_mainboard(code),
            "driver_claim_category": claim.get("driver_claim_category"),
            "recency_language_status": claim.get("recency_language_status"),
            "official_index_query_status": status.get("status", "NOT_QUERIED"),
            "official_index_row_count": status.get("row_count", 0),
            "pre_cutoff_official_row_count": int(len(part)),
            "compatible_title_count": int(len(matched)),
            "official_recency_status": recency,
            "latest_compatible_announcement_date": (
                str(latest.announcement_date) if latest is not None else ""),
            "latest_compatible_title": str(latest.title) if latest is not None else "",
            "latest_compatible_title_category": (
                str(latest.title_category) if latest is not None else ""),
            "latest_compatible_url": str(latest.url) if latest is not None else "",
            "latest_compatible_age_days": age,
            "title_match_proves_claim": False,
            "bounded_absence_proves_no_catalyst": False,
            "entry_authorized": False,
        })
        role_priority = 0 if code in role_codes else 1
        impact_priority = 0 if claim.get("driver_claim_category") in HIGH_IMPACT_CLAIMS else 1
        recency_priority = 0 if recency == "RECENT_OFFICIAL_TITLE_CANDIDATE" else 1
        queue.append({
            "signal_date": signal_date, "code": code, "name": claim.get("name"),
            "driver_claim_category": claim.get("driver_claim_category"),
            "driver_claim": claim.get("driver_claim"),
            "role_priority": role_priority, "high_impact_priority": impact_priority,
            "official_title_candidate_priority": recency_priority,
            "verification_priority_tuple": f"{role_priority}-{impact_priority}-{recency_priority}",
            "candidate_official_url": str(latest.url) if latest is not None else "",
            "candidate_official_title": str(latest.title) if latest is not None else "",
            "required_next_step": (
                "DOWNLOAD_AND_MATCH_OFFICIAL_FULLTEXT" if latest is not None
                else "SEARCH_POLICY_NEWS_INTERACTIVE_OR_OLDER_DISCLOSURES"),
            "official_claim_verified": False, "entry_authorized": False,
        })
    overlay = pd.DataFrame(overlays)
    queue_frame = pd.DataFrame(queue).sort_values(
        ["role_priority", "high_impact_priority", "official_title_candidate_priority", "code"])
    return overlay, queue_frame


def collect(claims_path: str, signal_date: str, cutoff: str, output_dir: str,
            lookback_days: int = 120, workers: int = 4, timeout: int = 35,
            role_report_path: str | None = None) -> dict[str, Any]:
    claim_file, target = Path(claims_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V233 archive")
    claims = pd.read_csv(claim_file, dtype={"code": str})
    required = {"signal_date", "code", "name", "driver_claim",
                "driver_claim_category", "recency_language_status"}
    if required - set(claims):
        raise ValueError(f"claim fields missing: {sorted(required-set(claims))}")
    claims["code"] = claims.code.astype(str).str.zfill(6)
    if set(claims.signal_date.astype(str)) != {signal_date}:
        raise ValueError("claim signal date mismatch")
    candidates = claims[claims.code.map(is_mainboard)].copy()
    start = (pd.Timestamp(signal_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    collected_at = datetime.now(timezone.utc).isoformat()
    role_codes: set[str] = set()
    source_paths = [claim_file]
    if role_report_path:
        role_file = Path(role_report_path)
        source_paths.append(role_file)
        role_doc = json.loads(role_file.read_text(encoding="utf-8-sig"))
        for key in ("narrative_lane_candidate", "symbolic_pure_buyable_pioneer",
                    "symbolic_dual_identity_candidate"):
            if role_doc.get(key):
                role_codes.add(str(role_doc[key]["code"]).zfill(6))

    stock_map, stock_map_raw, stock_map_hash = _cninfo_stock_map(timeout=timeout)
    query_status: dict[str, dict[str, Any]] = {}
    all_rows, raw_by_code = [], {}

    def one(code: str):
        if code.startswith("6"):
            rows, raw = query_sse_segment(code, start, signal_date, timeout=timeout)
            parsed = parse_sse_rows(rows, code, collected_at)
            return code, parsed, raw, []
        org_id = stock_map.get(code)
        if not org_id:
            raise ValueError("CNINFO_ORG_ID_MISSING")
        rows, raw, hashes = _query_cninfo(code, org_id, start, signal_date, timeout)
        parsed = parse_cninfo_rows({"announcements": rows}, code, collected_at)
        return code, parsed, raw, hashes

    with futures.ThreadPoolExecutor(max_workers=max(1, min(int(workers), 5))) as executor:
        jobs = {executor.submit(one, code): code for code in sorted(candidates.code.unique())}
        for job in futures.as_completed(jobs):
            code = jobs[job]
            try:
                _, rows, raw, hashes = job.result()
                all_rows.extend(rows)
                raw_by_code[code] = raw
                query_status[code] = {
                    "status": "SUCCESS", "row_count": len(rows), "error": "",
                    "response_hashes": hashes,
                }
            except Exception as exc:
                query_status[code] = {
                    "status": "FAILED", "row_count": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    "response_hashes": [],
                }
    announcements = pd.DataFrame(all_rows)
    if announcements.empty:
        announcements = pd.DataFrame(columns=[
            "code", "name", "announcement_date", "published_at_status", "title",
            "title_category", "url", "announcement_id", "source_level", "collected_at"])
    announcements = announcements.drop_duplicates(
        ["code", "announcement_id", "announcement_date", "title"])
    overlay, queue = build_overlays(
        claims, announcements, signal_date, cutoff, query_status, role_codes)

    target.mkdir(parents=True)
    index_path = target / "official_announcement_index.csv"
    overlay_path = target / "candidate_official_recency_overlay.csv"
    queue_path = target / "official_fulltext_verification_queue.csv"
    audit_path = target / "query_audit.csv"
    raw_path = target / "raw_index_responses.json"
    stock_map_path = target / "cninfo_stock_map_raw.json"
    announcements.sort_values(["code", "announcement_date", "announcement_id"]).to_csv(
        index_path, index=False, encoding="utf-8-sig")
    overlay.to_csv(overlay_path, index=False, encoding="utf-8-sig")
    queue.to_csv(queue_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"code": code, **value} for code, value in sorted(query_status.items())]).to_csv(
        audit_path, index=False, encoding="utf-8-sig")
    raw_path.write_text(json.dumps(raw_by_code, ensure_ascii=False), encoding="utf-8")
    stock_map_path.write_text(json.dumps(stock_map_raw, ensure_ascii=False), encoding="utf-8")
    successful = sum(value["status"] == "SUCCESS" for value in query_status.values())
    title_candidates = int(overlay.compatible_title_count.gt(0).sum())
    report = {
        "version": VERSION, "status": (
            "COMPLETE_OFFICIAL_INDEX_TRIAGE" if successful == len(candidates)
            else "PARTIAL_OFFICIAL_INDEX_TRIAGE"),
        "signal_date": signal_date, "cutoff": cutoff,
        "lookback_start": start, "lookback_days": lookback_days,
        "secondary_claim_rows": int(len(claims)),
        "mainboard_claim_rows": int(len(candidates)),
        "excluded_non_mainboard_rows": int(len(claims) - len(candidates)),
        "successful_official_index_queries": successful,
        "failed_official_index_queries": int(len(candidates) - successful),
        "official_announcement_rows": int(len(announcements)),
        "candidates_with_compatible_official_titles": title_candidates,
        "candidates_requiring_non_announcement_or_older_search": int(len(candidates) - title_candidates),
        "cninfo_stock_map_sha256": stock_map_hash,
        "source_hashes": {str(path.resolve()): _sha(path) for path in source_paths},
        "artifact_hashes": {path.name: _sha(path) for path in
                            [index_path, overlay_path, queue_path, audit_path,
                             raw_path, stock_map_path]},
        "governance": {
            "same_day_date_only_rows_known_at_close": False,
            "title_match_proves_claim": False,
            "bounded_absence_proves_no_catalyst": False,
            "non_company_policy_news_coverage_complete": False,
            "entry_authorized": False,
        },
        "entry_authorized": False,
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--signal-date", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lookback-days", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--role-report")
    ns = ap.parse_args()
    print(json.dumps(collect(
        ns.claims, ns.signal_date, ns.cutoff, ns.output, ns.lookback_days,
        ns.workers, ns.timeout, ns.role_report), ensure_ascii=False, indent=2))
