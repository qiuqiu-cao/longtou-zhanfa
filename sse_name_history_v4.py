#!/usr/bin/env python3
"""Recover dated SSE short-name/ST events from official announcements.

The SSE announcement index is authoritative for bulletin metadata.  Candidate
PDFs are archived byte-for-byte.  An event is VERIFIED only when PDF text
contains both an explicit before/after short-name transition and an explicit
effective date.  Title-only inference is never promoted to a security-master
interval.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from collect_security_master_v4 import name_status


ROOT = Path(__file__).resolve().parent
DELISTS = ROOT / "data" / "v4" / "historical_universe" / "official_delisted_mainboard.csv"
BASE = ROOT / "data" / "v4" / "security_evidence" / "sse_announcements"
RAW_INDEX = BASE / "raw_index"
PDF_DIR = BASE / "pdf"
TEXT_DIR = BASE / "text"
INDEX_CSV = BASE / "sse_announcement_index.csv"
EVENTS_CSV = BASE / "sse_name_events.csv"
RESULT_DIR = ROOT / "results" / "v4" / "security_master"
MANIFEST = RESULT_DIR / "sse_name_history_manifest.json"
VERSION = "v4-sse-name-history-1"

QUERY_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
PDF_HOST = "https://www.sse.com.cn"
ANNOUNCEMENT_PAGE = "https://www.sse.com.cn/disclosure/listedinfo/announcement/"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": ANNOUNCEMENT_PAGE}

DATE_TOKEN = r"(20\d{2})年(\d{1,2})月(\d{1,2})日"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acw_cookie(arg1: str) -> str:
    positions = [15, 35, 29, 24, 33, 16, 1, 38, 10, 9, 19, 31, 40, 27, 22, 23, 25, 13, 6, 11,
                 39, 18, 20, 8, 14, 21, 32, 26, 2, 30, 7, 4, 17, 5, 3, 28, 34, 37, 12, 36]
    mask = "3000176000856006061501533003690027800375"
    reordered = "".join(arg1[index - 1] for index in positions)
    return "".join(
        f"{int(reordered[index:index + 2], 16) ^ int(mask[index:index + 2], 16):02x}"
        for index in range(0, min(len(reordered), len(mask)), 2)
    )


def sse_codes(path: Path = DELISTS, coverage_start: str = "2023-11-28") -> list[str]:
    frame = pd.read_csv(path, dtype={"code": "string"})
    frame["delisting_date"] = pd.to_datetime(frame["delisting_date"], errors="raise")
    selected = frame[frame["exchange"].eq("SSE") & frame["delisting_date"].ge(pd.Timestamp(coverage_start))]
    return sorted(selected["code"].astype(str).str.zfill(6).unique())


def flatten_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in payload.get("result") or []:
        for item in group or []:
            if item.get("TITLE"):
                rows.append(item)
    return rows


def is_event_title(title: str) -> bool:
    title = str(title)
    if re.search(r"(?:股票|证券)简称.{0,8}变更|变更.{0,8}(?:股票|证券)简称", title):
        return True
    if "终止上市暨摘牌" in title:
        return True
    if "进入退市整理期" in title:
        return not any(token in title for token in ["风险提示", "首日", "第"])
    if "风险警示" not in title or not any(token in title for token in ["实施", "撤销"]):
        return False
    if any(token in title for token in ["进展", "可能被", "可能存在"]):
        return False
    # “继续实施”本身通常不改变简称，但“实施退市风险警示并继续实施
    # 其他风险警示”包含一个真实的新实施事件，不能一并排除。
    has_new_implementation = bool(re.search(r"(?<!继续)实施(?:退市|其他)风险警示", title))
    if "继续实施" in title and "撤销" not in title and not has_new_implementation:
        return False
    if "风险提示" in title and "暨停牌" not in title:
        return False
    return True


def query_segment(code: str, start: str, end: str, timeout: int = 30) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    session = requests.Session()
    page = 1
    rows: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    while True:
        params = {
            "isPagination": "true", "pageHelp.pageSize": 100, "pageHelp.pageNo": page,
            "pageHelp.beginPage": page, "pageHelp.cacheSize": 1, "pageHelp.endPage": page,
            "START_DATE": start, "END_DATE": end, "SECURITY_CODE": code,
            "TITLE": "", "BULLETIN_TYPE": "", "stockType": "1",
        }
        response = session.get(QUERY_URL, params=params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        raw_pages.append(payload)
        rows.extend(flatten_result(payload))
        page_count = int((payload.get("pageHelp") or {}).get("pageCount") or 0)
        if page >= page_count:
            break
        page += 1
        time.sleep(0.08)
    return rows, raw_pages


def collect_index(start: str = "2022-11-28", end: str = "2026-08-31", workers: int = 4) -> dict[str, Any]:
    codes = sse_codes()
    # The public UI caps one code query at three years, so use non-overlapping segments.
    segments = [(start, "2025-11-27"), ("2025-11-28", end)]
    RAW_INDEX.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def one(code: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        code_rows: list[dict[str, Any]] = []
        code_raw: list[dict[str, Any]] = []
        for segment_start, segment_end in segments:
            rows, raw = query_segment(code, segment_start, segment_end)
            code_rows.extend(rows)
            code_raw.extend(raw)
        return code, code_rows, code_raw

    with futures.ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
        jobs = {executor.submit(one, code): code for code in codes}
        for future in futures.as_completed(jobs):
            code = jobs[future]
            try:
                _, rows, raw = future.result()
                (RAW_INDEX / f"{code}.json").write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                collected.extend(rows)
            except Exception as exc:
                failures.append({"code": code, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    frame = pd.DataFrame(collected)
    required = ["SECURITY_CODE", "SECURITY_NAME", "SSEDATE", "TITLE", "URL", "ORG_BULLETIN_ID"]
    if frame.empty or not set(required).issubset(frame.columns):
        raise RuntimeError("SSE announcement index returned no usable rows")
    frame = frame.drop_duplicates("ORG_BULLETIN_ID").copy()
    frame["event_candidate"] = frame["TITLE"].astype(str).map(is_event_title)
    frame = frame.sort_values(["SECURITY_CODE", "SSEDATE", "ORG_BULLETIN_ID"])
    BASE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(INDEX_CSV, index=False, encoding="utf-8-sig")
    return {
        "codes_requested": len(codes), "codes_failed": len(failures), "failures": failures,
        "announcement_rows": int(len(frame)), "event_candidates": int(frame["event_candidate"].sum()),
        "index": str(INDEX_CSV.resolve()), "sha256": sha256_file(INDEX_CSV),
    }


def classify_existing_index() -> dict[str, Any]:
    frame = pd.read_csv(INDEX_CSV, dtype={"SECURITY_CODE": "string"})
    frame["event_candidate"] = frame["TITLE"].astype(str).map(is_event_title)
    frame.to_csv(INDEX_CSV, index=False, encoding="utf-8-sig")
    return {
        "announcement_rows": int(len(frame)), "event_candidates": int(frame["event_candidate"].sum()),
        "codes_with_candidates": int(frame.loc[frame["event_candidate"], "SECURITY_CODE"].nunique()),
        "index": str(INDEX_CSV.resolve()), "sha256": sha256_file(INDEX_CSV),
    }


def download_pdf(url_path: str, output: Path, timeout: int = 35) -> None:
    session = requests.Session()
    url = url_path if url_path.startswith("http") else PDF_HOST + url_path
    response = session.get(url, headers=HEADERS, timeout=timeout)
    content = response.content
    if not content.startswith(b"%PDF"):
        text = response.text
        match = re.search(r"arg1=.([0-9A-F]+).", text)
        if not match:
            raise ValueError("SSE PDF response is neither PDF nor recognized cookie challenge")
        session.cookies.set("acw_sc__v2", acw_cookie(match.group(1)), domain="sse.com.cn", path="/")
        response = session.get(url, headers=HEADERS, timeout=timeout)
        content = response.content
    if response.status_code != 200 or not content.startswith(b"%PDF"):
        raise ValueError(f"invalid SSE PDF response status={response.status_code} bytes={len(content)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)


def extract_pdf_text(pdf_path: Path, text_path: Path) -> None:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(text_path)],
        capture_output=True, text=True, timeout=60,
    )
    if process.returncode != 0 or not text_path.exists() or text_path.stat().st_size < 30:
        raise RuntimeError(f"pdftotext failed: {process.stderr[:300]}")


def download_candidates(workers: int = 3) -> dict[str, Any]:
    index = pd.read_csv(INDEX_CSV, dtype={"SECURITY_CODE": "string"})
    selected = index[index["event_candidate"].astype("string").str.lower().eq("true")].copy()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    succeeded = 0

    def one(row: pd.Series) -> None:
        stem = f"{row.SECURITY_CODE}_{str(row.SSEDATE)[:10].replace('-', '')}_{row.ORG_BULLETIN_ID}"
        pdf_path, text_path = PDF_DIR / f"{stem}.pdf", TEXT_DIR / f"{stem}.txt"
        if not pdf_path.exists() or not pdf_path.read_bytes()[:4] == b"%PDF":
            download_pdf(str(row.URL), pdf_path)
        if not text_path.exists() or text_path.stat().st_size < 30:
            extract_pdf_text(pdf_path, text_path)

    with futures.ThreadPoolExecutor(max_workers=max(1, min(3, workers))) as executor:
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
    return {"selected": len(selected), "succeeded": succeeded, "failed": len(failures), "failures": failures}


def parse_date(match: re.Match[str]) -> str:
    year, month, day = map(int, match.groups()[-3:])
    return pd.Timestamp(year=year, month=month, day=day).strftime("%Y-%m-%d")


def parse_event_text(text: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", text).replace("（", "(").replace("）", ")")
    transition_patterns = [
        r"(?:日起)?由[“‘\"]([^“”‘’\"，,；;。]{1,15})[”’\"](?:变更为|变为)[“‘\"]([^“”‘’\"，,；;。]{1,15})[”’\"](?=，?(?:公司)?(?:股票)?证券代码|，?股票代码)",
        r"(?:A股)?(?:股票|证券)简称[:：]?(?:将)?(?:由)?[“‘\"]?([^“”‘’\"，,；;。]{1,15})[”’\"]?(?:变更为|变为)[“‘\"]?([^“”‘’\"，,；;。]{1,15})[”’\"]?",
        # pdftotext may interleave two columns so that the effective date falls
        # between “将自” and the quoted before/after names.
        r"(?:公司)?(?:股票|证券)简称将自.{0,30}?[“‘\"]([^“”‘’\"，,；;。]{1,15})[”’\"](?:变更为|变为)[“‘\"]([^“”‘’\"，,；;。]{1,15})[”’\"]",
        r"(?:变更前的)?(?:A股)?(?:股票|证券)简称[:：][“‘\"]?([^“”‘’\"，,；;。]{1,15})[”’\"]?.{0,30}变更后的(?:A股)?(?:股票|证券)简称[:：][“‘\"]?([^“”‘’\"，,；;。]{1,15})[”’\"]?",
    ]
    transition = None
    transition_pattern = ""
    for pattern in transition_patterns:
        matches = list(re.finditer(pattern, compact))
        if matches:
            transition = max(matches, key=lambda match: len(match.group(1)) + len(match.group(2)))
            transition_pattern = pattern
            break
    date_patterns = [
        rf"(?:实施起始日|撤销起始日)[:：]?为?{DATE_TOKEN}",
        rf"(?:证券简称变更日期|股票简称变更日期|实施退市风险警示的起始日|实施其他风险警示的起始日|撤销退市风险警示的起始日|撤销其他风险警示的起始日)[:：]?{DATE_TOKEN}",
        rf"(?:将于|自){DATE_TOKEN}起(?:被)?(?:实施|撤销|进入)",
        rf"{DATE_TOKEN}起(?:被)?(?:实施|撤销|进入)(?:退市风险警示|其他风险警示|风险警示|退市整理期)",
    ]
    date_match = None
    date_pattern = ""
    for pattern in date_patterns:
        date_match = re.search(pattern, compact)
        if date_match:
            date_pattern = pattern
            break
    if not transition or not date_match:
        return {
            "parse_status": "REVIEW_REQUIRED", "before_name": "", "after_name": "",
            "effective_date": "", "after_status": "", "transition_pattern": transition_pattern,
            "date_pattern": date_pattern,
        }
    before_name, after_name = transition.group(1).strip(":："), transition.group(2).strip(":：")
    if before_name == after_name or len(before_name) > 12 or len(after_name) > 12:
        return {
            "parse_status": "REVIEW_REQUIRED", "before_name": before_name, "after_name": after_name,
            "effective_date": parse_date(date_match), "after_status": "",
            "transition_pattern": transition_pattern, "date_pattern": date_pattern,
        }
    return {
        "parse_status": "VERIFIED_EVENT", "before_name": before_name, "after_name": after_name,
        "effective_date": parse_date(date_match), "after_status": name_status(after_name),
        "transition_pattern": transition_pattern, "date_pattern": date_pattern,
    }


def extract_events() -> dict[str, Any]:
    index = pd.read_csv(INDEX_CSV, dtype={"SECURITY_CODE": "string", "ORG_BULLETIN_ID": "string"})
    selected = index[index["event_candidate"].astype("string").str.lower().eq("true")].copy()
    rows: list[dict[str, Any]] = []
    for _, item in selected.iterrows():
        stem = f"{item.SECURITY_CODE}_{str(item.SSEDATE)[:10].replace('-', '')}_{item.ORG_BULLETIN_ID}"
        pdf_path, text_path = PDF_DIR / f"{stem}.pdf", TEXT_DIR / f"{stem}.txt"
        if not pdf_path.exists() or not text_path.exists():
            parsed = {"parse_status": "DOWNLOAD_OR_TEXT_MISSING", "before_name": "", "after_name": "",
                      "effective_date": "", "after_status": "", "transition_pattern": "", "date_pattern": ""}
        else:
            parsed = parse_event_text(text_path.read_text(encoding="utf-8", errors="replace"))
        if parsed["parse_status"] == "VERIFIED_EVENT" and parsed["effective_date"]:
            announced = pd.Timestamp(str(item.SSEDATE)[:10])
            effective = pd.Timestamp(parsed["effective_date"])
            if effective < announced - pd.Timedelta(days=7):
                parsed["parse_status"] = "REVIEW_REQUIRED_DATE_BEFORE_ANNOUNCEMENT"
        rows.append({
            "code": str(item.SECURITY_CODE).zfill(6), "announcement_date": str(item.SSEDATE)[:10],
            "security_name_at_announcement": item.SECURITY_NAME, "title": item.TITLE,
            "url": item.URL, "bulletin_id": item.ORG_BULLETIN_ID,
            "pdf_sha256": sha256_file(pdf_path) if pdf_path.exists() else "",
            "text_sha256": sha256_file(text_path) if text_path.exists() else "", **parsed,
        })
    events = pd.DataFrame(rows).sort_values(["code", "announcement_date", "bulletin_id"])
    verified = events[events["parse_status"].eq("VERIFIED_EVENT")]
    conflicts = verified.groupby(["code", "effective_date"])["after_name"].nunique()
    conflict_keys = conflicts[conflicts > 1]
    if not conflict_keys.empty:
        raise ValueError(f"conflicting SSE name events: {list(conflict_keys.index)[:5]}")
    BASE.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    metadata = {
        "version": VERSION, "status": "PARTIAL_VERIFIED_EVENTS" if len(verified) < len(events) else "ALL_VERIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "codes": int(events["code"].nunique()), "event_candidates": int(len(events)),
        "verified_events": int(len(verified)), "verified_codes": int(verified["code"].nunique()),
        "review_required": int(events["parse_status"].ne("VERIFIED_EVENT").sum()),
        "source": ANNOUNCEMENT_PAGE,
        "files": {
            "index": {"path": str(INDEX_CSV.resolve()), "sha256": sha256_file(INDEX_CSV)},
            "events": {"path": str(EVENTS_CSV.resolve()), "sha256": sha256_file(EVENTS_CSV)},
        },
        "deployment_effect": "no SSE interval may be verified from title-only evidence",
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(MANIFEST.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover SSE dated name/ST events from official announcements")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--start", default="2022-11-28")
    index.add_argument("--end", default="2026-08-31")
    index.add_argument("--workers", type=int, default=4)
    download = sub.add_parser("download")
    download.add_argument("--workers", type=int, default=3)
    sub.add_parser("extract")
    sub.add_parser("classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "index":
        result = collect_index(args.start, args.end, args.workers)
    elif args.command == "download":
        result = download_candidates(args.workers)
    elif args.command == "extract":
        result = extract_events()
    else:
        result = classify_existing_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
