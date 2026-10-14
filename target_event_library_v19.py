"""Merge inferred discoveries with reviewed threshold evidence, without labels.

An official event can be missing from raw crossings after a prior unreset
breach. Add that event without inventing a full historical reset calendar.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def merge_events(scan, reconciled):
    rows = {}
    for row in scan.to_dict("records"):
        for target in (100, 200):
            flag = row.get(f"cross_{target}_proxy")
            if str(flag).lower() not in {"true", "1"}:
                continue
            key = (str(row["code"]).zfill(6), str(row["date"]), target)
            if key in rows:
                raise ValueError("duplicate scan event")
            rows[key] = dict(code=key[0], event_date=key[1], target_pct=target,
                             scanned=True, official_confirmed=False, official_evidence=[],
                             training_label_ready=False, evidence_available_at=None)
    for case in reconciled:
        if case.get("result") != "THRESHOLD_DIRECTION_MATCH":
            continue
        if case.get("evidence_status") != "official_company_full_text_read":
            raise ValueError("official event requires reviewed full text")
        if not case.get("source") or case.get("threshold_pct") not in (100, 200):
            raise ValueError("source and explicit supported threshold required")
        event, published = pd.Timestamp(case["end"]), pd.Timestamp(case["publication_date"])
        if pd.isna(event) or pd.isna(published) or published < event:
            raise ValueError("invalid publication chronology")
        key = (str(case["code"]).zfill(6), event.strftime("%Y-%m-%d"), case["threshold_pct"])
        row = rows.setdefault(key, dict(code=key[0], event_date=key[1], target_pct=key[2],
                                       scanned=False, official_confirmed=False, official_evidence=[],
                                       training_label_ready=False, evidence_available_at=None))
        available = published.strftime("%Y-%m-%d") + "T23:59:59+08:00"
        row["official_confirmed"] = True
        row["official_evidence"].append(dict(source=case["source"], start=case["start"],
                                             publication_date=case["publication_date"],
                                             calculated_deviation_pct=case["calculated_deviation_pct"]))
        row["evidence_available_at"] = min(row["evidence_available_at"] or available, available)
    return sorted(rows.values(), key=lambda r: (r["event_date"], r["code"], r["target_pct"]))


def known_official_events(events, asof):
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        raise ValueError("asof must include timezone")
    return [row for row in events if row["official_confirmed"]
            and pd.Timestamp(row["evidence_available_at"]) <= cutoff]


def load_reconciliations(paths):
    """Combine separately audited batches without dropping earlier evidence."""
    cases, seen = [], set()
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for filename, expected in report["sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"reconciliation input changed: {filename}")
        for case in report["cases"]:
            key = json.dumps(case, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                cases.append(case)
                seen.add(key)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite event library")
    cases = load_reconciliations(args.reconciliation)
    events = merge_events(pd.read_csv(args.scan, dtype={"code": str}), cases)
    output = dict(status="DIAGNOSTIC_ONLY", events=events,
                  event_count=len(events), official_confirmed_count=sum(r["official_confirmed"] for r in events),
                  official_missing_from_scan=sum(r["official_confirmed"] and not r["scanned"] for r in events),
                  verified_trade_labels=0, source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in [args.scan, *args.reconciliation]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k != "events"}, ensure_ascii=False, indent=2))
