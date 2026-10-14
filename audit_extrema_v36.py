"""Measure cross-source extrema discrepancies without selecting a preferred feed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def classify(row):
    differences = row.get("differences", {})
    if "observation" not in row:
        return dict(code=row["code"], date=row["entry_date"], state="UNAVAILABLE")
    fields = sorted(differences)
    inward = all((field == "high" and values["minute"] < values["daily"])
                 or (field == "low" and values["minute"] > values["daily"])
                 for field, values in differences.items())
    state = "MATCH_ONLY" if not fields else "MINUTE_RANGE_NARROWER" if inward else "OTHER_DISAGREEMENT"
    high = differences.get("high")
    limit = row["limit_price_proxy"]
    missed_touch = bool(high and high["daily"] >= limit-1e-6 and high["minute"] < limit-1e-6)
    return dict(code=row["code"], date=row["entry_date"], state=state,
                differing_fields=fields, daily_limit_touch_not_in_minute=missed_touch,
                extrema_reconciled=not fields, cross_source_cause="UNRESOLVED",
                exact_touch_time_usable=False, stop_fill_usable=False,
                note="Full-day quality audit is retrospective metadata, not an opening feature.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.replay.read_text(encoding="utf-8"))
    rows = [classify(row) for row in source["rows"]]
    report = dict(status="DIAGNOSTIC_ONLY", source_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                  states=dict(Counter(r["state"] for r in rows)),
                  differing_fields=dict(Counter(f for r in rows for f in r.get("differing_fields", []))),
                  daily_only_limit_touches=sum(r.get("daily_limit_touch_not_in_minute", False) for r in rows),
                  rows=rows, root_cause_confirmed=False,
                  implication="Do not treat absence of a minute high at limit as proof of no touch; do not reconstruct missed ticks.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
