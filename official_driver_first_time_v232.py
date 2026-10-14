"""Verify first-board driver age against archived official primary sources.

V231 preserves secondary-market explanations but cannot tell whether a claimed
driver is new.  V232 takes a small, explicit official-evidence ledger, archives
every cited source, verifies that its required terms occur in the source text,
and reports how long the core claim had already been public by the signal date.

The result is deliberately not a score.  An old driver may still trade, and an
ongoing corporate action may remain relevant, but neither may be described as a
new signal-date catalyst without separate evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from sse_name_history_v4 import download_pdf as download_sse_pdf


VERSION = "v232-official-driver-first-time-1"
REQUIRED_EVIDENCE = {
    "signal_date", "code", "official_source_url", "official_published_at",
    "published_time_precision", "first_public_time_basis", "source_level",
    "match_terms", "lifecycle_status", "latest_status_url",
    "latest_status_published_at", "latest_status_match_terms", "notes",
}
ALLOWED_BASES = {"EXACT_OFFICIAL_PAGE_TIME", "AT_LATEST_OFFICIAL_DISCLOSURE"}
ALLOWED_LIFECYCLES = {
    "LEGACY_FACT_OR_PRODUCT_DESCRIPTION", "ONGOING_ACTION_VERIFIED",
    "HISTORICAL_POLICY_INCLUSION",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("～", "~")


def _terms(value: Any) -> list[str]:
    return [term.strip() for term in str(value or "").split("|") if term.strip()]


def _assert_terms(text: str, terms: list[str], label: str) -> None:
    compact = _normalise(text)
    missing = [term for term in terms if _normalise(term) not in compact]
    if missing:
        raise ValueError(f"official source term mismatch for {label}: {missing}")


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Calculate only objective age/read-through fields after source matching."""
    signal = pd.Timestamp(str(row["signal_date"]))
    published = pd.Timestamp(str(row["official_published_at"]))
    if published.date() > signal.date():
        raise ValueError("official source was published after signal date")
    basis = str(row["first_public_time_basis"])
    if basis not in ALLOWED_BASES:
        raise ValueError(f"unsupported first-public basis: {basis}")
    lifecycle = str(row["lifecycle_status"])
    if lifecycle not in ALLOWED_LIFECYCLES:
        raise ValueError(f"unsupported lifecycle: {lifecycle}")
    age = int((signal.normalize() - published.normalize()).days)
    if age == 0:
        novelty = "SAME_DAY_OFFICIAL_DISCLOSURE"
    else:
        novelty = f"PREEXISTING_AT_LEAST_{age}_CALENDAR_DAYS"
    if lifecycle == "ONGOING_ACTION_VERIFIED":
        readthrough = ("ONGOING_ACTION_NEW_ON_SIGNAL_DATE" if age == 0
                       else "ONGOING_ACTION_BUT_NOT_NEW_ON_SIGNAL_DATE")
    else:
        readthrough = ("NEW_OFFICIAL_DRIVER_ON_SIGNAL_DATE" if age == 0
                       else "LEGACY_OR_RECYCLED_DRIVER_NOT_NEW")
    return {
        "official_driver_evidence_status": "OFFICIAL_CORE_CLAIM_MATCHED",
        "first_public_time_status": (
            "VERIFIED_EXACT" if basis == "EXACT_OFFICIAL_PAGE_TIME"
            else "VERIFIED_AT_LATEST_DATE"),
        "official_driver_age_lower_bound_days": age,
        "signal_date_novelty_status": novelty,
        "driver_freshness_readthrough": readthrough,
        "driver_is_new_on_signal_date": age == 0,
        "driver_age_is_ranking_score": False,
        "entry_authorized": False,
    }


def _download(url: str, output: Path, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if url.lower().endswith(".pdf") and "sse.com.cn" in url:
        download_sse_pdf(url, output, timeout=timeout)
        return
    response = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0", "Referer": url}, timeout=timeout)
    response.raise_for_status()
    output.write_bytes(response.content)


def _extract_text(source: Path) -> str:
    if source.read_bytes()[:4] == b"%PDF":
        text_path = source.with_suffix(source.suffix + ".txt")
        process = subprocess.run(
            ["pdftotext", "-layout", str(source), str(text_path)],
            capture_output=True, text=True, timeout=90)
        if process.returncode != 0 or not text_path.exists():
            raise RuntimeError(f"pdftotext failed for {source.name}: {process.stderr[:300]}")
        return text_path.read_text(encoding="utf-8", errors="replace")
    return BeautifulSoup(source.read_bytes(), "html.parser").get_text(" ", strip=True)


def _archive_source(url: str, stem: str, source_dir: Path, timeout: int,
                    terms: list[str], reuse_record: dict[str, Any] | None = None) -> dict[str, Any]:
    suffix = ".pdf" if url.lower().endswith(".pdf") else ".html"
    target = source_dir / f"{stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if reuse_record is not None:
        if reuse_record.get("url") != url:
            raise ValueError(f"prior official-source URL mismatch for {stem}")
        prior = Path(str(reuse_record.get("archived_path") or ""))
        if not prior.exists() or _sha(prior) != reuse_record.get("source_sha256"):
            raise ValueError(f"prior official-source hash mismatch for {stem}")
        target.write_bytes(prior.read_bytes())
    else:
        _download(url, target, timeout)
    text = _extract_text(target)
    _assert_terms(text, terms, stem)
    text_path = target.with_suffix(target.suffix + ".txt")
    if not text_path.exists():
        text_path.write_text(text, encoding="utf-8")
    return {
        "url": url, "archived_path": str(target.resolve()),
        "source_sha256": _sha(target), "text_path": str(text_path.resolve()),
        "text_sha256": _sha(text_path), "matched_terms": terms,
        "reused_prior_hash_verified_archive": reuse_record is not None,
    }


def verify(claims_path: str, evidence_path: str, output_dir: str,
           timeout: int = 35, prior_report_path: str | None = None) -> dict[str, Any]:
    claim_file, evidence_file, target = (Path(claims_path), Path(evidence_path),
                                         Path(output_dir))
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V232 archive")
    claims = pd.read_csv(claim_file, dtype={"code": str})
    evidence = pd.read_csv(evidence_file, dtype={"code": str}, keep_default_na=False)
    if REQUIRED_EVIDENCE - set(evidence):
        raise ValueError(f"official evidence fields missing: {sorted(REQUIRED_EVIDENCE-set(evidence))}")
    if evidence.duplicated(["signal_date", "code"]).any():
        raise ValueError("official evidence keys must be unique")
    claims["code"] = claims.code.astype(str).str.zfill(6)
    evidence["code"] = evidence.code.astype(str).str.zfill(6)
    signal_dates = set(claims.signal_date.astype(str)) | set(evidence.signal_date.astype(str))
    if len(signal_dates) != 1:
        raise ValueError("claim and official-evidence signal dates differ")
    missing_claims = sorted(set(evidence.code) - set(claims.code))
    if missing_claims:
        raise ValueError(f"official evidence lacks matching V231 claim rows: {missing_claims}")

    prior_by_code = {}
    if prior_report_path:
        prior_file = Path(prior_report_path)
        prior_report = json.loads(prior_file.read_text(encoding="utf-8-sig"))
        prior_by_code = {str(row["code"]).zfill(6): row
                         for row in prior_report.get("source_audit", [])}
    target.mkdir(parents=True)
    source_dir = target / "official_sources"
    rows, audits = [], []
    for item in evidence.sort_values("code").to_dict("records"):
        code = item["code"]
        primary = _archive_source(
            item["official_source_url"], f"{code}_primary", source_dir,
            timeout, _terms(item["match_terms"]),
            (prior_by_code.get(code) or {}).get("primary"))
        latest = None
        if item["latest_status_url"]:
            latest = _archive_source(
                item["latest_status_url"], f"{code}_latest_status", source_dir,
                timeout, _terms(item["latest_status_match_terms"]),
                (prior_by_code.get(code) or {}).get("latest_status"))
            if not item["latest_status_published_at"]:
                raise ValueError(f"latest-status date missing for {code}")
            if pd.Timestamp(item["latest_status_published_at"]).date() > pd.Timestamp(
                    item["signal_date"]).date():
                raise ValueError(f"latest-status source is after signal date for {code}")
        if item["lifecycle_status"] == "ONGOING_ACTION_VERIFIED" and latest is None:
            raise ValueError(f"ongoing action lacks a matched current-status source: {code}")
        evaluated = evaluate_row(item)
        claim = claims[claims.code.eq(code)].iloc[0]
        rows.append({
            "signal_date": item["signal_date"], "code": code, "name": claim["name"],
            "secondary_driver_claim": claim["driver_claim"],
            "secondary_driver_claim_category": claim["driver_claim_category"],
            "official_source_url": item["official_source_url"],
            "official_published_at": item["official_published_at"],
            "published_time_precision": item["published_time_precision"],
            "first_public_time_basis": item["first_public_time_basis"],
            "source_level": item["source_level"],
            "lifecycle_status": item["lifecycle_status"],
            "latest_status_url": item["latest_status_url"],
            "latest_status_published_at": item["latest_status_published_at"],
            "official_source_sha256": primary["source_sha256"],
            "latest_status_source_sha256": latest["source_sha256"] if latest else "",
            "notes": item["notes"], **evaluated,
        })
        audits.append({"code": code, "primary": primary, "latest_status": latest})

    verified = pd.DataFrame(rows)
    overlay = claims[["signal_date", "code", "name"]].copy()
    overlay = overlay.merge(verified, on=["signal_date", "code", "name"], how="left")
    overlay["official_driver_evidence_status"] = overlay[
        "official_driver_evidence_status"].fillna("NO_OFFICIAL_DRIVER_EVIDENCE")
    overlay["first_public_time_status"] = overlay[
        "first_public_time_status"].fillna("UNAVAILABLE")
    overlay["driver_is_new_on_signal_date"] = overlay[
        "driver_is_new_on_signal_date"].astype("boolean")
    overlay["driver_age_is_ranking_score"] = False
    overlay["entry_authorized"] = False

    verified_path = target / "verified_official_driver_evidence.csv"
    overlay_path = target / "candidate_driver_time_overlay.csv"
    verified.to_csv(verified_path, index=False, encoding="utf-8-sig")
    overlay.to_csv(overlay_path, index=False, encoding="utf-8-sig")
    report = {
        "version": VERSION, "status": "PARTIAL_OFFICIAL_DRIVER_TIME_VERIFICATION",
        "signal_date": signal_dates.pop(), "candidate_claims": int(len(claims)),
        "officially_matched_candidates": int(len(verified)),
        "candidates_without_official_driver_evidence": int(len(claims) - len(verified)),
        "same_day_new_official_drivers": int(verified.driver_is_new_on_signal_date.sum()),
        "preexisting_driver_candidates": int((~verified.driver_is_new_on_signal_date).sum()),
        "ongoing_actions_verified": int(verified.lifecycle_status.eq(
            "ONGOING_ACTION_VERIFIED").sum()),
        "source_audit": audits,
        "source_hashes": {str(path.resolve()): _sha(path) for path in
                          [claim_file, evidence_file] +
                          ([Path(prior_report_path)] if prior_report_path else [])},
        "artifact_hashes": {verified_path.name: _sha(verified_path),
                            overlay_path.name: _sha(overlay_path)},
        "governance": {
            "unverified_claim_becomes_new_catalyst": False,
            "age_used_as_ranking_score": False,
            "ongoing_action_equals_new_catalyst": False,
            "official_source_term_match_required": True,
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
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--prior-report",
                    help="reuse hash-verified official archives from an earlier V232 report")
    ns = ap.parse_args()
    print(json.dumps(verify(ns.claims, ns.evidence, ns.output, ns.timeout, ns.prior_report),
                     ensure_ascii=False, indent=2))
