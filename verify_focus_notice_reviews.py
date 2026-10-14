"""Verify hash-bound manual reviews of focus announcements."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ALLOWED_CONCLUSIONS = {
    "ACTIVE_REGULATORY_WARNING",
    "PLANNED_JV_NOT_OPERATING",
    "UNLOCK_CONDITIONS_MET_AWAIT_LISTING_NOTICE",
    "ORDINARY_ABNORMAL_NO_UNDISCLOSED_MATTER",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_path: Path, ledger_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    notices = {row["notice_id"]: row for row in manifest["notices"]}
    review_ids = [row.get("notice_id", "") for row in rows]
    errors: list[str] = []
    if len(review_ids) != len(set(review_ids)):
        errors.append("DUPLICATE_REVIEW_ID")
    if set(review_ids) != set(notices):
        errors.append("REVIEW_SET_DOES_NOT_MATCH_MANIFEST")
    for row in rows:
        notice_id = row.get("notice_id", "")
        notice = notices.get(notice_id)
        if not notice:
            continue
        pdf_path = Path(notice["pdf_path"])
        text_path = Path(notice["text_path"])
        current_pdf = sha256_file(pdf_path)
        current_text = sha256_file(text_path)
        if row.get("pdf_sha256") != notice["pdf_sha256"] or current_pdf != notice["pdf_sha256"]:
            errors.append(f"PDF_HASH_MISMATCH:{notice_id}")
        if row.get("text_sha256") != notice["text_sha256"] or current_text != notice["text_sha256"]:
            errors.append(f"TEXT_HASH_MISMATCH:{notice_id}")
        if row.get("conclusion") not in ALLOWED_CONCLUSIONS:
            errors.append(f"INVALID_CONCLUSION:{notice_id}")
        if row.get("entry_authorized", "").lower() != "false":
            errors.append(f"ENTRY_MUST_REMAIN_UNAUTHORIZED:{notice_id}")
        try:
            reviewed = datetime.fromisoformat(row["reviewed_at"])
            received = datetime.fromisoformat(notice["received_at"])
            if reviewed < received:
                errors.append(f"REVIEW_PRECEDES_RECEIPT:{notice_id}")
        except (KeyError, ValueError):
            errors.append(f"INVALID_REVIEW_TIME:{notice_id}")
        if not row.get("visual_pages_reviewed"):
            errors.append(f"MISSING_VISUAL_REVIEW:{notice_id}")
    return {
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": sha256_file(ledger_path),
        "manifest_notice_count": len(notices),
        "review_count": len(rows),
        "valid": not errors,
        "errors": errors,
        "scope": (
            "Integrity and timing verification of manual reviews. It does not "
            "authenticate the Eastmoney mirror against an exchange domain and "
            "does not authorize an entry."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify(Path(args.manifest), Path(args.ledger))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
