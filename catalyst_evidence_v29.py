"""Point-in-time catalyst ledger, not automatic theme attribution or buy signals."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("explicit timezone timestamp required")
    return result.tz_convert("Asia/Shanghai")


def public_available(record):
    if record.get("published_at"):
        return timestamp(record["published_at"])
    # Date-only disclosure must not be silently backdated to that day's auction.
    return timestamp(record["publication_date"] + "T23:59:59+08:00")


def snapshot(ledger, code, asof, mode="forward"):
    if mode not in {"forward", "historical_reconstruction"}:
        raise ValueError("unknown mode")
    cutoff = timestamp(asof)
    code = str(code).zfill(6)
    selected, seen = {}, set()
    for record in ledger["records"]:
        if record["evidence_id"] in seen:
            raise ValueError("duplicate evidence id")
        seen.add(record["evidence_id"])
        if str(record["code"]).zfill(6) != code:
            continue
        available, captured = public_available(record), timestamp(record["captured_at"])
        if captured < available:
            raise ValueError("capture predates conservative public availability")
        if available > cutoff or (mode == "forward" and captured > cutoff):
            continue
        if record.get("evidence_status") != "verified" or not record.get("source"):
            continue
        key = record["event_id"]
        known_at = max(available, captured) if mode == "forward" else available
        previous = selected.get(key)
        if previous is not None and previous[0] == known_at:
            raise ValueError("ambiguous simultaneous event revisions")
        if previous is None or previous[0] < known_at:
            selected[key] = (known_at, {**record, "available_at": available.isoformat(),
                                      "known_at": known_at.isoformat(),
                                      "calendar_days_since_publication": (cutoff.date()-available.date()).days})
    records = [pair[1] for _, pair in sorted(selected.items())]
    return dict(code=code, asof=cutoff.isoformat(), mode=mode, status="DIAGNOSTIC_ONLY",
                coverage=ledger.get("coverage", "unknown"), records=records,
                observed_event_count=len(records) if records else None,
                evidence_state="historical_facts_available" if records else "unavailable_not_proven_absent",
                current_event_validity="not_established_by_historical_disclosure",
                confirmed_price_driver=None, theme_peer_count=None,
                leader_probability=None, buy_signal=False,
                note="Corporate event evidence does not prove price causation, peer support or ongoing validity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--mode", choices=["forward", "historical_reconstruction"], default="forward")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = snapshot(json.loads(Path(args.ledger).read_text(encoding="utf-8")), args.code, args.asof, args.mode)
    result["ledger_sha256"] = hashlib.sha256(Path(args.ledger).read_bytes()).hexdigest()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
