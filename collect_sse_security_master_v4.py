#!/usr/bin/env python3
"""Build point-in-time SSE intervals from official announcement evidence.

Exact name-change dates come only from parsed announcement PDFs.  A name first
seen in announcement metadata without a matching PDF transition starts an
``inferred`` interval, so it can never pass the verified eligibility gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from collect_security_master_v4 import name_status
from security_master_v4 import MASTER_COLUMNS, validate_master


ROOT = Path(__file__).resolve().parent
DELISTS = ROOT / "data" / "v4" / "historical_universe" / "official_delisted_mainboard.csv"
INDEX = ROOT / "data" / "v4" / "security_evidence" / "sse_announcements" / "sse_announcement_index.csv"
EVENTS = ROOT / "data" / "v4" / "security_evidence" / "sse_announcements" / "sse_name_events.csv"
OUTPUT = ROOT / "data" / "v4" / "security_master_sse_delisted_evidenced.csv"
MANIFEST = ROOT / "results" / "v4" / "security_master" / "sse_delisted_interval_manifest.json"
SSE_ANNOUNCEMENTS = "https://www.sse.com.cn/disclosure/listedinfo/announcement/"
SSE_DELIST = "https://www.sse.com.cn/assortment/stock/list/delisting/"
VERSION = "v4-sse-delisted-security-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def regime_for_status(status: str) -> str:
    if status in {"ST", "*ST"}:
        return "ST_5"
    if status == "DELISTING_RISK":
        return "OTHER"
    return "MAINBOARD_10"


def observed_name_changes(index_part: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = index_part.sort_values(["SSEDATE", "ORG_BULLETIN_ID"]).copy()
    ordered["_norm"] = ordered["SECURITY_NAME"].map(normalized_name)
    changes: list[dict[str, Any]] = []
    prior = None
    for row in ordered.itertuples(index=False):
        current = normalized_name(row.SECURITY_NAME)
        if prior is not None and current != prior:
            changes.append({
                "date": pd.Timestamp(row.SSEDATE).normalize(),
                "name": str(row.SECURITY_NAME).strip(),
                "kind": "metadata_observation",
                "source_ref": SSE_ANNOUNCEMENTS,
            })
        prior = current
    return changes


def initial_name(index_part: pd.DataFrame, events_part: pd.DataFrame, coverage_start: pd.Timestamp) -> str:
    events_part = events_part.sort_values("effective_date")
    if not events_part.empty:
        first = events_part.iloc[0]
        current = str(first["before_name"]).strip()
        for event in events_part.itertuples(index=False):
            if pd.Timestamp(event.effective_date) <= coverage_start:
                current = str(event.after_name).strip()
        return current
    ordered = index_part.sort_values(["SSEDATE", "ORG_BULLETIN_ID"])
    if ordered.empty:
        raise ValueError("SSE code lacks official announcement metadata")
    return str(ordered.iloc[0]["SECURITY_NAME"]).strip()


def build_code_intervals(
    stock: pd.Series,
    index_part: pd.DataFrame,
    events_part: pd.DataFrame,
    coverage_start: pd.Timestamp,
    asof: pd.Timestamp,
    source_timestamp: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    listing = pd.Timestamp(stock["listing_date"]).normalize()
    delisting = pd.Timestamp(stock["delisting_date"]).normalize()
    listed_start = max(listing, coverage_start)
    listed_end = min(delisting - timedelta(days=1), asof)
    code = str(stock["code"]).zfill(6)
    events_part = events_part.sort_values("effective_date").copy()
    timeline: list[dict[str, Any]] = []
    for event in events_part.itertuples(index=False):
        when = pd.Timestamp(event.effective_date).normalize()
        if listed_start <= when <= listed_end:
            timeline.append({
                "date": when, "name": str(event.after_name).strip(), "before_name": str(event.before_name).strip(),
                "kind": "verified_pdf_event", "source_ref": str(event.url),
            })
    timeline.extend(
        change for change in observed_name_changes(index_part)
        if listed_start <= change["date"] <= listed_end
    )
    timeline.sort(key=lambda item: (item["date"], 0 if item["kind"] == "verified_pdf_event" else 1))

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if listed_start <= listed_end:
        current_name = initial_name(index_part, events_part, listed_start)
        current_start = listed_start
        current_status = "verified_import"
        current_ref = SSE_ANNOUNCEMENTS
        for transition in timeline:
            when = transition["date"]
            proposed = transition["name"]
            if transition["kind"] == "verified_pdf_event":
                chain_ok = normalized_name(current_name) == normalized_name(transition["before_name"])
                audit.append({
                    "code": code, "date": when.strftime("%Y-%m-%d"), "kind": transition["kind"],
                    "before_expected": current_name, "before_stated": transition["before_name"],
                    "after_name": proposed, "chain_ok": chain_ok,
                })
                next_status = "verified_import"
            else:
                if normalized_name(proposed) == normalized_name(current_name):
                    continue
                audit.append({
                    "code": code, "date": when.strftime("%Y-%m-%d"), "kind": transition["kind"],
                    "before_expected": current_name, "before_stated": "", "after_name": proposed,
                    "chain_ok": False,
                })
                next_status = "inferred"
            if when > current_start:
                rows.append(_interval_row(
                    code, current_name, current_start, when - timedelta(days=1), listing, delisting,
                    current_status, current_ref, source_timestamp,
                ))
            current_name = proposed
            current_start = when
            current_status = next_status
            current_ref = transition["source_ref"]
        rows.append(_interval_row(
            code, current_name, current_start, listed_end, listing, delisting,
            current_status, current_ref, source_timestamp,
        ))
    if delisting <= asof:
        rows.append({
            "code": code, "name": str(stock["final_name"]).strip(),
            "valid_from": delisting.strftime("%Y-%m-%d"), "valid_to": asof.strftime("%Y-%m-%d"),
            "exchange": "SSE", "board": "MAINBOARD", "listing_status": "DELISTED",
            "st_status": "DELISTING_RISK", "price_limit_regime": "NONE",
            "listing_date": listing.strftime("%Y-%m-%d"), "delisting_date": delisting.strftime("%Y-%m-%d"),
            "source_type": "sse_official", "source_ref": SSE_DELIST,
            "source_timestamp": source_timestamp, "data_status": "verified_import",
        })
    return rows, audit


def _interval_row(
    code: str, name: str, start: pd.Timestamp, end: pd.Timestamp, listing: pd.Timestamp,
    delisting: pd.Timestamp, data_status: str, source_ref: str, source_timestamp: str,
) -> dict[str, Any]:
    status = name_status(name)
    return {
        "code": code, "name": name, "valid_from": start.strftime("%Y-%m-%d"),
        "valid_to": end.strftime("%Y-%m-%d"), "exchange": "SSE", "board": "MAINBOARD",
        "listing_status": "LISTED", "st_status": status, "price_limit_regime": regime_for_status(status),
        "listing_date": listing.strftime("%Y-%m-%d"), "delisting_date": delisting.strftime("%Y-%m-%d"),
        "source_type": "sse_official", "source_ref": f"{SSE_ANNOUNCEMENTS}|{source_ref}",
        "source_timestamp": source_timestamp, "data_status": data_status,
    }


def build_intervals(
    delists: pd.DataFrame,
    index: pd.DataFrame,
    events: pd.DataFrame,
    coverage_start: pd.Timestamp,
    asof: pd.Timestamp,
    source_timestamp: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    verified_events = events[events["parse_status"].eq("VERIFIED_EVENT")].copy()
    verified_events["effective_date"] = pd.to_datetime(verified_events["effective_date"], errors="raise")
    for _, stock in delists.sort_values("code").iterrows():
        code = str(stock["code"]).zfill(6)
        index_part = index[index["SECURITY_CODE"].eq(code)]
        if index_part.empty:
            raise ValueError(f"SSE code lacks archived announcement index: {code}")
        event_part = verified_events[verified_events["code"].eq(code)]
        code_rows, code_audit = build_code_intervals(
            stock, index_part, event_part, coverage_start, asof, source_timestamp,
        )
        rows.extend(code_rows)
        audits.extend(code_audit)
    checked = validate_master(pd.DataFrame(rows, columns=MASTER_COLUMNS))
    return checked.loc[:, MASTER_COLUMNS], pd.DataFrame(audits)


def collect(
    coverage_start: str = "2023-11-28",
    asof: str = "2026-08-31",
    delists_path: Path = DELISTS,
    index_path: Path = INDEX,
    events_path: Path = EVENTS,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    source_timestamp = datetime.now(timezone.utc).isoformat()
    start, end = pd.Timestamp(coverage_start), pd.Timestamp(asof)
    delists = pd.read_csv(delists_path, dtype={"code": "string"})
    delists["listing_date"] = pd.to_datetime(delists["listing_date"], errors="raise")
    delists["delisting_date"] = pd.to_datetime(delists["delisting_date"], errors="raise")
    delists = delists[
        delists["exchange"].eq("SSE") & delists["delisting_date"].ge(start)
        & delists["listing_date"].le(end)
    ].copy()
    index = pd.read_csv(index_path, dtype={"SECURITY_CODE": "string", "ORG_BULLETIN_ID": "string"})
    index["SSEDATE"] = pd.to_datetime(index["SSEDATE"], errors="raise")
    archived_codes = set(index["SECURITY_CODE"].astype(str))
    missing_delist_rows = sorted(archived_codes.difference(delists["code"].astype(str)))
    if missing_delist_rows:
        raise ValueError(f"archived SSE code lacks official delisting row: {missing_delist_rows}")
    delists = delists[delists["code"].astype(str).isin(archived_codes)].copy()
    events = pd.read_csv(events_path, dtype={"code": "string", "bulletin_id": "string"}).fillna("")
    intervals, chain_audit = build_intervals(delists, index, events, start, end, source_timestamp)
    output.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(output, index=False, encoding="utf-8-sig")
    audit_path = output.with_name("security_master_sse_event_chain_audit.csv")
    chain_audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    inferred = intervals["data_status"].ne("verified_import")
    verified_chain_failures = chain_audit[
        chain_audit.get("kind", pd.Series(dtype=str)).eq("verified_pdf_event")
        & ~chain_audit.get("chain_ok", pd.Series(dtype=bool)).astype(bool)
    ] if not chain_audit.empty else chain_audit
    metadata = {
        "version": VERSION,
        "status": "PARTIAL_VERIFIED_SSE_INTERVALS" if inferred.any() else "VERIFIED_SSE_INTERVALS",
        "coverage_start": coverage_start, "asof": asof,
        "delisted_codes": int(delists["code"].nunique()), "interval_rows": int(len(intervals)),
        "verified_interval_rows": int((~inferred).sum()), "inferred_interval_rows": int(inferred.sum()),
        "normal_verified_rows": int((intervals["st_status"].eq("NORMAL") & ~inferred).sum()),
        "pdf_verified_events": int(events["parse_status"].eq("VERIFIED_EVENT").sum()),
        "verified_event_chain_mismatches": int(len(verified_chain_failures)),
        "metadata_only_transitions": int(chain_audit.get("kind", pd.Series(dtype=str)).eq("metadata_observation").sum()),
        "source_timestamp": source_timestamp,
        "sources": [SSE_ANNOUNCEMENTS, SSE_DELIST],
        "deployment_effect": "only verified_import intervals may pass the historical eligibility gate",
        "files": {
            "index": {"path": str(index_path.resolve()), "sha256": sha256_file(index_path)},
            "events": {"path": str(events_path.resolve()), "sha256": sha256_file(events_path)},
            "intervals": {"path": str(output.resolve()), "sha256": sha256_file(output)},
            "chain_audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(MANIFEST.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official-evidence SSE delisted intervals")
    parser.add_argument("--coverage-start", default="2023-11-28")
    parser.add_argument("--asof", default="2026-08-31")
    parser.add_argument("--delists", type=Path, default=DELISTS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect(args.coverage_start, args.asof, args.delists, args.index, args.events, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
