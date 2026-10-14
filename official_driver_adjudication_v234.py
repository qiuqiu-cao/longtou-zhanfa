"""Adjudicate high-impact first-board claims against official full text.

V233 finds candidate announcement titles across the complete main-board claim
pool.  V234 uses an explicit review ledger to archive selected official PDFs,
require claim-specific terms, enforce the decision cutoff, and preserve material
conflicts between the secondary explanation and the official document.

The adjudication is evidence only.  It does not rank candidates or authorize a
trade, and a same-day date-only disclosure is conservatively unavailable at a
15:00 close decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from official_driver_first_time_v232 import (
    _archive_source, _assert_terms, _sha, _terms,
)


VERSION = "v234-official-driver-adjudication-1"
REQUIRED = {
    "signal_date", "code", "official_source_url", "official_published_at",
    "published_time_precision", "availability_basis", "official_available_at",
    "source_level", "required_terms", "conflict_terms", "expected_verdict", "notes",
}
VERDICTS = {
    "OFFICIAL_FULLTEXT_MATCH_PREEXISTING",
    "OFFICIAL_NEGATIVE_CLARIFICATION_MATCHED",
    "OFFICIAL_DETAIL_CONFLICT",
    "OFFICIAL_MATCH_AFTER_CUTOFF_UNUSABLE",
}


def available_at(row: dict[str, Any]) -> pd.Timestamp:
    basis = str(row["availability_basis"])
    if basis == "DATE_ONLY_CONSERVATIVE_END_OF_DAY":
        stamp = pd.Timestamp(str(row["official_published_at"])).normalize()
        return stamp.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    if basis == "EXACT_AVAILABLE_AT":
        stamp = pd.Timestamp(str(row["official_available_at"]))
        if stamp.tzinfo is None:
            raise ValueError("exact official_available_at must include timezone")
        return stamp.tz_convert("Asia/Shanghai")
    raise ValueError(f"unsupported availability basis: {basis}")


def adjudicate(row: dict[str, Any], claim_category: str, cutoff: str) -> dict[str, Any]:
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts.tzinfo is None:
        raise ValueError("cutoff must include timezone")
    cutoff_cn = cutoff_ts.tz_convert("Asia/Shanghai")
    availability = available_at(row)
    if availability > cutoff_cn:
        verdict = "OFFICIAL_MATCH_AFTER_CUTOFF_UNUSABLE"
    elif _terms(row.get("conflict_terms")):
        verdict = "OFFICIAL_DETAIL_CONFLICT"
    elif claim_category == "NEGATIVE_OR_LIMITED_EXPOSURE":
        verdict = "OFFICIAL_NEGATIVE_CLARIFICATION_MATCHED"
    else:
        verdict = "OFFICIAL_FULLTEXT_MATCH_PREEXISTING"
    if verdict not in VERDICTS:
        raise ValueError(f"unsupported adjudication verdict: {verdict}")
    expected = str(row["expected_verdict"])
    if expected != verdict:
        raise ValueError(f"expected verdict {expected} differs from calculated {verdict}")
    signal = pd.Timestamp(str(row["signal_date"]))
    published = pd.Timestamp(str(row["official_published_at"]))
    return {
        "official_adjudication_verdict": verdict,
        "official_available_at_effective": availability.isoformat(),
        "official_evidence_usable_at_decision": availability <= cutoff_cn,
        "official_driver_age_days": int((signal.normalize() - published.normalize()).days),
        "secondary_claim_fully_supported": verdict in {
            "OFFICIAL_FULLTEXT_MATCH_PREEXISTING",
            "OFFICIAL_NEGATIVE_CLARIFICATION_MATCHED",
        },
        "material_detail_conflict": verdict == "OFFICIAL_DETAIL_CONFLICT",
        "after_cutoff_unusable": verdict == "OFFICIAL_MATCH_AFTER_CUTOFF_UNUSABLE",
        "positive_buy_reason_allowed": False,
        "entry_authorized": False,
    }


def run(claims_path: str, evidence_path: str, cutoff: str, output_dir: str,
        timeout: int = 45, prior_report_path: str | None = None) -> dict[str, Any]:
    claim_file, evidence_file, target = Path(claims_path), Path(evidence_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V234 archive")
    claims = pd.read_csv(claim_file, dtype={"code": str})
    evidence = pd.read_csv(evidence_file, dtype={"code": str}, keep_default_na=False)
    if REQUIRED - set(evidence):
        raise ValueError(f"adjudication evidence fields missing: {sorted(REQUIRED-set(evidence))}")
    claims["code"] = claims.code.astype(str).str.zfill(6)
    evidence["code"] = evidence.code.astype(str).str.zfill(6)
    if evidence.duplicated(["signal_date", "code"]).any():
        raise ValueError("adjudication evidence keys must be unique")
    dates = set(claims.signal_date.astype(str)) | set(evidence.signal_date.astype(str))
    if len(dates) != 1:
        raise ValueError("claim and adjudication signal dates differ")
    missing = sorted(set(evidence.code) - set(claims.code))
    if missing:
        raise ValueError(f"adjudication codes lack V231 claims: {missing}")

    prior_by_code = {}
    if prior_report_path:
        prior_file = Path(prior_report_path)
        prior_report = json.loads(prior_file.read_text(encoding="utf-8-sig"))
        prior_by_code = {str(row["code"]).zfill(6): row["source"]
                         for row in prior_report.get("source_audit", [])}
    target.mkdir(parents=True)
    source_dir = target / "official_sources"
    rows, audits = [], []
    for item in evidence.sort_values("code").to_dict("records"):
        code = item["code"]
        claim = claims[claims.code.eq(code)].iloc[0]
        archived = _archive_source(
            item["official_source_url"], f"{code}_official", source_dir,
            timeout, _terms(item["required_terms"]), prior_by_code.get(code))
        text = Path(archived["text_path"]).read_text(encoding="utf-8", errors="replace")
        conflicts = _terms(item.get("conflict_terms"))
        if conflicts:
            _assert_terms(text, conflicts, f"{code}_conflict")
        result = adjudicate(item, str(claim.driver_claim_category), cutoff)
        rows.append({
            "signal_date": item["signal_date"], "code": code, "name": claim["name"],
            "secondary_driver_claim": claim.driver_claim,
            "secondary_driver_claim_category": claim.driver_claim_category,
            "official_source_url": item["official_source_url"],
            "official_published_at": item["official_published_at"],
            "published_time_precision": item["published_time_precision"],
            "availability_basis": item["availability_basis"],
            "source_level": item["source_level"],
            "required_terms": item["required_terms"],
            "conflict_terms": item["conflict_terms"],
            "official_source_sha256": archived["source_sha256"],
            "official_text_sha256": archived["text_sha256"],
            "notes": item["notes"], **result,
        })
        audits.append({"code": code, "source": archived, "verdict": result[
            "official_adjudication_verdict"]})
    adjudications = pd.DataFrame(rows)
    overlay = claims[["signal_date", "code", "name"]].merge(
        adjudications, on=["signal_date", "code", "name"], how="left")
    overlay["official_adjudication_verdict"] = overlay[
        "official_adjudication_verdict"].fillna("NO_V234_ADJUDICATION")
    overlay["positive_buy_reason_allowed"] = False
    overlay["entry_authorized"] = False

    adjudication_path = target / "official_driver_adjudications.csv"
    overlay_path = target / "candidate_adjudication_overlay.csv"
    adjudications.to_csv(adjudication_path, index=False, encoding="utf-8-sig")
    overlay.to_csv(overlay_path, index=False, encoding="utf-8-sig")
    counts = adjudications.official_adjudication_verdict.value_counts().to_dict()
    report = {
        "version": VERSION, "status": "PARTIAL_HIGH_IMPACT_FULLTEXT_ADJUDICATION",
        "signal_date": dates.pop(), "cutoff": cutoff,
        "candidate_claims": int(len(claims)),
        "adjudicated_candidates": int(len(adjudications)),
        "verdict_counts": counts,
        "fully_supported_candidates": int(adjudications.secondary_claim_fully_supported.sum()),
        "material_detail_conflicts": int(adjudications.material_detail_conflict.sum()),
        "after_cutoff_unusable": int(adjudications.after_cutoff_unusable.sum()),
        "source_audit": audits,
        "source_hashes": {
            str(claim_file.resolve()): _sha(claim_file),
            str(evidence_file.resolve()): _sha(evidence_file),
            **({str(Path(prior_report_path).resolve()): _sha(Path(prior_report_path))}
               if prior_report_path else {}),
        },
        "artifact_hashes": {adjudication_path.name: _sha(adjudication_path),
                            overlay_path.name: _sha(overlay_path)},
        "governance": {
            "same_day_date_only_available_at_close": False,
            "material_detail_conflict_can_rank_positive": False,
            "official_fulltext_match_is_buy_signal": False,
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
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--prior-report",
                    help="reuse hash-verified official archives from an earlier V234 report")
    ns = ap.parse_args()
    print(json.dumps(run(ns.claims, ns.evidence, ns.cutoff, ns.output, ns.timeout,
                         ns.prior_report),
                     ensure_ascii=False, indent=2))
