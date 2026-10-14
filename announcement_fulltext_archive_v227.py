"""Archive all active hard-risk notice PDFs and extracted text for relay review.

The collector validates the announcement index against its manifest, selects
the complete active risk set, stores immutable hashes and records the actual
receipt time.  Eastmoney is a document mirror, so receipt is not represented
as exchange-domain authentication and never authorises an entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


VERSION = "v227-announcement-fulltext-archive-1"
TZ = ZoneInfo("Asia/Shanghai")
NOTICE_RE = re.compile(r"(AN\d+)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notice_id(url: str) -> str:
    match = NOTICE_RE.search(str(url))
    if not match:
        raise ValueError(f"notice id unavailable in URL: {url}")
    return match.group(1)


def collect(index_path: str, manifest_path: str, output_dir: str,
            timeout: int = 30) -> dict:
    index_file, manifest_file = Path(index_path), Path(manifest_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V227 archive")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    expected = manifest.get("files", {}).get("announcements", {}).get("sha256")
    if not expected or _sha(index_file) != expected:
        raise ValueError("announcement index hash differs from manifest")
    frame = pd.read_csv(index_file, dtype={"code": str})
    active = frame[
        pd.to_numeric(frame.risk_hard, errors="coerce").fillna(0).gt(0)
        & frame.risk_window_status.fillna("").astype(str).str.startswith("active_within_")
    ].copy()
    target.mkdir(parents=True)
    rows = []
    for row in active.to_dict("records"):
        notice_id = _notice_id(row["url"])
        started = datetime.now(TZ)
        pdf_url = f"https://pdf.dfcfw.com/pdf/H2_{notice_id}_1.pdf"
        result = {
            "code": str(row["code"]).zfill(6), "name": row.get("name"),
            "notice_id": notice_id, "title": row.get("title"),
            "index_url": row.get("url"), "pdf_url": pdf_url,
            "published_at": row.get("published_at"),
            "published_at_status": row.get("published_at_status"),
            "risk_type": row.get("risk_type"), "risk_window_status": row.get("risk_window_status"),
            "request_started_at": started.isoformat(),
            "document_source_tier": "PUBLIC_DOCUMENT_MIRROR_NOT_EXCHANGE_AUTHENTICATED",
            "full_text_status": "UNAVAILABLE", "entry_authorized": False,
        }
        try:
            response = requests.get(pdf_url, timeout=timeout)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            pdf_path = target / f"{notice_id}.pdf"
            text_path = target / f"{notice_id}.txt"
            pdf_path.write_bytes(response.content)
            result.update({
                "full_text_status": "MIRROR_PDF_RECEIVED_TEXT_UNAVAILABLE",
                "full_text_available_at": datetime.now(TZ).isoformat(),
                "pdf_path": str(pdf_path.resolve()), "pdf_sha256": _sha(pdf_path),
            })
            subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_path)],
                           check=True, capture_output=True, timeout=timeout)
            text = text_path.read_text(encoding="utf-8", errors="strict")
            if len(text.strip()) < 80:
                raise ValueError("extracted text too short; OCR or manual review required")
            received = datetime.now(TZ)
            result.update({
                "full_text_status": "MIRROR_PDF_TEXT_RECEIVED",
                "full_text_available_at": received.isoformat(),
                "pdf_path": str(pdf_path.resolve()), "text_path": str(text_path.resolve()),
                "pdf_sha256": _sha(pdf_path), "text_sha256": _sha(text_path),
                "text_length": len(text),
            })
        except Exception as exc:
            result.update({"full_text_available_at": datetime.now(TZ).isoformat(),
                           "error": f"{type(exc).__name__}: {exc}"})
        rows.append(result)
    evidence = pd.DataFrame(rows)
    evidence_path = target / "fulltext_evidence.csv"
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    report = {
        "version": VERSION, "status": "ARCHIVED_MIRROR_EVIDENCE_NOT_TRADE_CLEARANCE",
        "signal_date": manifest.get("signal_date"),
        "announcement_scope_exact": bool(
            manifest.get("candidate_count") == manifest.get("announcements", {}).get("candidate_codes")
            and manifest.get("announcements", {}).get("failed_codes") == 0),
        "active_hard_notice_rows": int(len(active)), "requested_rows": int(len(rows)),
        "received_pdf_rows": int(evidence.full_text_status.astype(str).str.startswith(
            "MIRROR_PDF").sum()) if len(evidence) else 0,
        "received_text_rows": int(evidence.full_text_status.eq("MIRROR_PDF_TEXT_RECEIVED").sum()) if len(evidence) else 0,
        "unavailable_text_rows": int(evidence.full_text_status.ne("MIRROR_PDF_TEXT_RECEIVED").sum()) if len(evidence) else 0,
        "source_hashes": {str(index_file.resolve()): _sha(index_file),
                          str(manifest_file.resolve()): _sha(manifest_file)},
        "evidence_sha256": _sha(evidence_path),
        "limitations": [
            "the PDF host is a public mirror rather than an exchange domain",
            "receipt time is recorded and is not backdated to the publication date",
            "PDF/text receipt alone is not a risk conclusion or entry authorisation",
        ],
        "entry_authorized": False,
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=30)
    ns = ap.parse_args()
    print(json.dumps(collect(ns.index, ns.manifest, ns.output, ns.timeout),
                     ensure_ascii=False, indent=2))
