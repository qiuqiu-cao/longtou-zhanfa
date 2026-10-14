"""Hash-bind a manually read, post-freeze disclosure update to every candidate.

Source anchors verify provenance, not semantic truth; claims are human-style
review annotations. No regex score, probability adjustment, new order, or
historical call rewrite is produced.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import fitz

from next_session_expectation_contract_v271 import load_registered, now, stamp
from holiday_official_evidence_v283 import sha


def normalize(text):
    return re.sub(r"\s+", "", text)


def validate_candidate_scope(manual, frozen):
    codes = [r["code"] for r in manual["candidates"]]
    if len(codes) != len(set(codes)) or set(codes) != {r["code"] for r in frozen["plan"]["candidates"]}:
        raise ValueError("manual update must cover the exact complete frozen candidate denominator")
    policy = manual["policy"]
    if any(policy.get(k) is not False for k in ("all_candidates_certified_clean", "rank_change_authorized",
            "trade_authorized", "original_frozen_calls_change_authorized")) or policy.get("probability_adjustment") is not None:
        raise ValueError("manual evidence update cannot change calls, probabilities or trading authorization")


def source_review(row, evidence, recorded_at):
    code, ident = row["code"], row["announcement_id"]
    source = evidence.get((code, ident))
    if source is None:
        raise ValueError("reviewed document outside the archived official selection")
    pdf = Path(source["pdf_path"])
    if sha(pdf) != source["pdf_sha256"]:
        raise ValueError("review PDF hash mismatch")
    available = source.get("fulltext_available_at") or source.get("pdf_available_at")
    if not available or stamp(available) > stamp(recorded_at):
        raise ValueError("review source not yet available")
    document = fitz.open(pdf)
    try:
        pages = row.get("source_pages", [])
        if not pages or any(not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= len(document) for p in pages):
            raise ValueError("explicit valid source page numbers required")
        page_text = normalize("\n".join(document[p-1].get_text() for p in pages))
        for anchor in row.get("anchors", []):
            if normalize(anchor) not in page_text:
                raise ValueError("review anchor missing on selected source pages: " + anchor)
    finally:
        document.close()
    return dict(official_url=source["url"], publication_date=source["announcement_date"],
        publication_clock_status="DATE_ONLY", available_at=available,
        manually_reviewed_at=recorded_at, source_pages=row["source_pages"],
        pdf_path=str(pdf), pdf_sha256=source["pdf_sha256"],
        review_status="MANUAL_SELECTED_FULLTEXT" if row.get("anchors") else "MANUAL_VISUAL_OR_CONTEXT_REVIEW")


def run(archive, manual_path, registered_path, output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite manual official update")
    archive = Path(archive)
    report = json.loads((archive / "report.json").read_text(encoding="utf-8"))
    frozen = load_registered(registered_path)
    if report["registration_hash"] != frozen["sha256"]:
        raise ValueError("official archive belongs to another frozen forecast")
    for path, wanted in report["artifact_hashes"].items():
        if sha(archive / path) != wanted:
            raise ValueError("official archive artifact hash changed")
    for path, wanted in report["source_hashes"].items():
        if sha(path) != wanted:
            raise ValueError("official archive source hash changed")
    manual = json.loads(Path(manual_path).read_text(encoding="utf-8"))
    validate_candidate_scope(manual, frozen)
    evidence = {(r["code"], r["announcement_id"]): r
        for r in json.loads((archive / "fulltext_evidence.json").read_text(encoding="utf-8"))}
    recorded = now().isoformat()
    updates = []
    for row in manual["candidates"]:
        update = dict(row)
        if row["announcement_id"] is None:
            if any(code == row["code"] for code, _ in evidence):
                raise ValueError("unreviewed candidate has selected official documents")
            update.update(review_status="BOUNDED_NO_SELECTED_DOCUMENT_NOT_CLEAN", manually_reviewed_at=recorded)
        else:
            update.update(source_review(row, evidence, recorded))
        update.update(full_risk_coverage=False, hard_veto_from_keyword_scan=False,
            eligible_for_original_frozen_forecast=False, probability_adjustment=None,
            original_call_rewrite_allowed=False, trade_authorized=False)
        updates.append(update)
    visual = [{**row, **source_review(row, evidence, recorded)} for row in manual["supplementary_visual_reviews"]]
    keyword = manual["keyword_false_positive_case"]
    if keyword.get("assert_no_investigation_anywhere") is not False:
        raise ValueError("keyword-context review cannot become a whole-document no-risk certificate")
    keyword = {**keyword, **source_review(keyword, evidence, recorded), "hard_veto_authorized": False}
    result = dict(version="v283-hash-bound-manual-official-update-1", recorded_at=recorded,
        registration_hash=frozen["sha256"], status="POST_FREEZE_MANUAL_EVIDENCE_UPDATE",
        signal_date=frozen["plan"]["signal_date"], target_date=frozen["plan"]["target_date"],
        candidate_updates=updates, supplementary_visual_reviews=visual, keyword_context_case=keyword,
        candidate_denominator=len(updates), fulltext_reviewed_candidates=sum(r["announcement_id"] is not None for r in updates),
        policy=manual["policy"], accuracy_claim_allowed=False, source_hashes={str(Path(p).resolve()):sha(p) for p in
            [manual_path, registered_path, archive / "report.json", Path(__file__)]})
    target.mkdir(parents=True, exist_ok=False)
    (target / "manual_official_update.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: result[k] for k in ("status", "candidate_denominator", "fulltext_reviewed_candidates", "registration_hash", "accuracy_claim_allowed")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--registered", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.manual, args.registered, args.output), ensure_ascii=True))
