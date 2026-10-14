"""Bounded official disclosure collection appended after the frozen forecast.

All frozen candidates are queried. Date-only publication never becomes an
invented clock time. Full-text receipt and manual review remain separate.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import fitz
import pandas as pd

from current_reset_chronology_v192 import CNINFO_QUERY_URL, CNINFO_HEADERS, CNINFO_STOCK_URL
from next_session_expectation_contract_v271 import load_registered, now
from official_announcement_recency_v233 import parse_cninfo_rows

VERSION = "v283-post-freeze-official-disclosure-1"
MAX_PAGES = 3
TIMEOUT = (4, 10)
MAX_PDF_BYTES = 16 * 1024 * 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock():
    return now().astimezone(ZoneInfo("Asia/Shanghai")).isoformat()


def validate_page(payload, code, start, end):
    if not isinstance(payload, dict) or not isinstance(payload.get("hasMore"), bool):
        raise ValueError("index has no explicit pagination status")
    if not isinstance(payload.get("totalAnnouncement"), int) or payload["totalAnnouncement"] < 0:
        raise ValueError("index has no explicit nonnegative row total")
    raw = payload.get("announcements")
    if raw is None and payload["totalAnnouncement"] == 0:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("index announcement list missing")
    rows = parse_cninfo_rows({"announcements": raw}, code, clock())
    if any(not start <= r["announcement_date"] <= end for r in rows):
        raise ValueError("announcement outside requested date window")
    for row in rows:
        if not re.fullmatch(r"https://static\.cninfo\.com\.cn/finalpage/\d{4}-\d{2}-\d{2}/\d+\.PDF", row["url"], re.I):
            raise ValueError("nonofficial or malformed document URL")
        row["published_at"] = None
        row["available_at"] = row.pop("collected_at")
    return rows


def choose_documents(rows):
    risk = [r for r in rows if re.search(r"异常波动|风险提示|立案|处罚|终止上市|减持计划|股票停牌", r["title"])]
    capital = [r for r in rows if re.search(r"购买资产.*(?:预案|报告书)", r["title"])
               and not re.search(r"说明|核查|决议|意见|回复|关于|前十", r["title"])]
    chosen = {(r["code"], r["announcement_id"]): {**r, "selection_reason": "RISK_NOTICE_TITLE"} for r in risk}
    for code in sorted({r["code"] for r in capital}):
        part = [r for r in capital if r["code"] == code]
        latest_date = max(r["announcement_date"] for r in part)
        part = [r for r in part if r["announcement_date"] == latest_date]
        # Preserve the full plan, prefer summary separately when one is supplied.
        for summary in (False, True):
            group = [r for r in part if ("摘要" in r["title"]) == summary]
            if group:
                r = sorted(group, key=lambda x: x["announcement_id"])[-1]
                chosen[(r["code"], r["announcement_id"])] = {**r, "selection_reason": "CAPITAL_ACTION_DOCUMENT"}
    return sorted(chosen.values(), key=lambda r: (r["code"], r["announcement_date"], r["announcement_id"]))


def query(code, org_id, start, end, raw_dir):
    result, total, complete = [], None, False
    requested_at = clock()
    response_hashes = []
    for page in range(1, MAX_PAGES + 1):
        body = dict(pageNum=str(page), pageSize="100", column="sse" if code.startswith("6") else "szse",
            tabName="fulltext", plate="sh" if code.startswith("6") else "sz", stock=code+","+org_id,
            searchkey="", secid="", category="", trade="", seDate=start+"~"+end,
            sortName="", sortType="", isHLtitle="true")
        response = requests.post(CNINFO_QUERY_URL, data=body, headers=CNINFO_HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        path = raw_dir / (code+"_page"+str(page)+".json")
        path.write_bytes(response.content)
        response_hashes.append(dict(path=str(path.resolve()), sha256=sha(path), received_at=clock()))
        response.encoding = "utf-8"
        payload = response.json()
        rows = validate_page(payload, code, start, end)
        if total is not None and total != payload["totalAnnouncement"]:
            raise ValueError("index total changed during pagination")
        total = payload["totalAnnouncement"]
        result.extend(rows)
        if not payload["hasMore"]:
            complete = True
            break
    unique = {(r["code"], r["announcement_id"]): r for r in result}
    if complete and len(unique) != total:
        raise ValueError("received index count differs from official total")
    return list(unique.values()), dict(code=code, status="BOUNDED_INDEX_COMPLETE" if complete else "TRUNCATED_AT_PAGE_CAP",
        total=total, rows_received=len(unique), request_started_at=requested_at, completed_at=clock(),
        response_hashes=response_hashes)


def document(row, directory):
    result = {**row, "fulltext_status": "UNAVAILABLE", "manual_review_status": "NOT_REVIEWED",
              "hard_veto": False, "trade_authorized": False, "request_started_at": clock()}
    try:
        with requests.get(row["url"], headers=CNINFO_HEADERS, timeout=TIMEOUT, stream=True) as response:
            response.raise_for_status()
            chunks, length = [], 0
            for chunk in response.iter_content(65536):
                length += len(chunk)
                if length > MAX_PDF_BYTES:
                    raise ValueError("PDF exceeds bounded size cap")
                chunks.append(chunk)
        content = b"".join(chunks)
        if not content.startswith(b"%PDF"):
            raise ValueError("official document response is not PDF")
        name = row["code"]+"_"+row["announcement_id"]
        pdf_path = directory / (name+".pdf")
        text_path = directory / (name+".txt")
        pdf_path.write_bytes(content)
        result.update(fulltext_status="OFFICIAL_PDF_RECEIVED", pdf_path=str(pdf_path.resolve()),
                      pdf_sha256=sha(pdf_path), pdf_available_at=clock())
        doc = fitz.open(pdf_path)
        try:
            if len(doc) > 400:
                raise ValueError("PDF exceeds bounded extraction page cap")
            texts = [page.get_text() for page in doc]
            text = "\n".join("\n[PAGE "+str(i+1)+"]\n"+t for i, t in enumerate(texts))
            if len(text.strip()) < 80:
                raise ValueError("text extraction too short; OCR/manual review needed")
            text_path.write_text(text, encoding="utf-8")
            result.update(fulltext_status="OFFICIAL_PDF_TEXT_RECEIVED", text_path=str(text_path.resolve()),
                text_sha256=sha(text_path), pages=len(doc), fulltext_available_at=clock())
        finally:
            doc.close()
    except Exception as exc:
        result.update(error=type(exc).__name__+": "+str(exc)[:350], completed_at=clock())
    return result


def run(registered_path, output, start="2026-09-15"):
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite official evidence archive")
    registered = load_registered(registered_path)
    original_hash = sha(registered_path)
    end = now().astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    if not start <= registered["plan"]["signal_date"] <= end < registered["plan"]["target_date"]:
        raise ValueError("collection must be between signal and exact future target session")
    codes = {r["code"]: r["name"] for r in registered["plan"]["candidates"]}
    target.mkdir(parents=True, exist_ok=False)
    raw_dir, docs_dir = target / "raw", target / "documents"
    raw_dir.mkdir(); docs_dir.mkdir()
    response = requests.get(CNINFO_STOCK_URL, headers=CNINFO_HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    (raw_dir / "stock_map.json").write_bytes(response.content)
    response.encoding = "utf-8"
    mapping = {str(r["code"]).zfill(6): str(r["orgId"]) for r in response.json().get("stockList", [])
               if r.get("code") and r.get("orgId")}
    rows, audit = [], []
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(query, code, mapping.get(code, ""), start, end, raw_dir): code for code in sorted(codes)}
        for job in futures.as_completed(jobs):
            code = jobs[job]
            try:
                if not mapping.get(code):
                    raise ValueError("official org ID unavailable")
                found, status = job.result()
                rows.extend(found); audit.append(status)
            except Exception as exc:
                audit.append(dict(code=code, status="QUERY_FAILED", rows_received=0, completed_at=clock(),
                                  error=type(exc).__name__+": "+str(exc)[:350]))
    selected_docs = choose_documents(rows)
    evidence = []
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(document, row, docs_dir): row for row in selected_docs}
        for job in futures.as_completed(jobs):
            evidence.append(job.result())
    rows.sort(key=lambda r: (r["code"], r["announcement_date"], r["announcement_id"]))
    evidence.sort(key=lambda r: (r["code"], r["announcement_date"], r["announcement_id"]))
    for name, value in (("index.json", rows), ("query_audit.json", sorted(audit, key=lambda r: r["code"])),
                        ("fulltext_evidence.json", evidence)):
        (target / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(target / "index.csv", index=False, encoding="utf-8-sig")
    completed = sum(r["status"] == "BOUNDED_INDEX_COMPLETE" for r in audit)
    if sha(registered_path) != original_hash:
        raise ValueError("frozen forecast changed during collection")
    report = dict(version=VERSION, status="COMPLETE_BOUNDED_INDEX" if completed == len(codes) else "PARTIAL_INDEX",
        signal_date=registered["plan"]["signal_date"], target_date=registered["plan"]["target_date"],
        registration_hash=registered["sha256"], frozen_plan_preserved=True, collected_until=clock(),
        window_start=start, window_end=end, candidate_codes=sorted(codes), candidate_count=len(codes),
        complete_index_codes=completed, query_failed_or_truncated_codes=len(codes)-completed,
        index_rows=len(rows), selected_documents=len(evidence),
        received_fulltexts=sum(r["fulltext_status"] == "OFFICIAL_PDF_TEXT_RECEIVED" for r in evidence),
        source_level="CNINFO_OFFICIAL_COMPANY_DISCLOSURE",
        source_hashes={str(Path(registered_path).resolve()): original_hash, str(Path(__file__).resolve()): sha(__file__)},
        artifact_hashes={str(p.relative_to(target)): sha(p) for p in target.rglob("*") if p.is_file()},
        limitations=["Bounded company disclosure index; not exhaustive all-policy/news/regulatory coverage",
            "Publication dates are date-only; availability uses actual receipt, never midnight invention",
            "Title selection and text receipt are not manual risk review or no-risk certification",
            "Current archive is post-freeze evidence; cannot alter original forecast or historical labels",
            "October 3-8 new disclosure/auction/intraday evidence remains unavailable"],
        goal_verified=False, trade_authorized=False, default_model_changed=False)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: report[k] for k in ("status", "candidate_count", "complete_index_codes", "index_rows", "selected_documents", "received_fulltexts", "frozen_plan_preserved")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2026-09-15")
    args = parser.parse_args()
    print(json.dumps(run(args.registered, args.output, args.start), ensure_ascii=True))
