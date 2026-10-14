"""Map official event endpoints to relay segments, without assigning trade labels."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(events, observations, episodes, calendar):
    if calendar != sorted(set(calendar)):
        raise ValueError("invalid calendar")
    positions = {day: i for i, day in enumerate(calendar)}
    if episodes.episode_id.duplicated().any() or observations.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate episode or observation")
    rows, summaries = [], []
    for event in events:
        if event.get("official_confirmed") is not True:
            continue
        code, end = event["code"], event["event_date"]
        evidence = event["official_evidence"]
        if not evidence or end not in positions:
            raise ValueError("official evidence or event session missing")
        start = min(item["start"] for item in evidence)
        containing = episodes[(episodes.code == code) & (episodes.start_date <= end)
                              & (episodes.last_limit_date >= end)]
        if len(containing) > 1:
            raise ValueError("overlapping relay episodes")
        containing_id = containing.iloc[0].episode_id if len(containing) else None
        candidates = observations[(observations.code == code)
                                  & observations.signal_date.between(start, end)]
        counts = {}
        for candidate in candidates.to_dict("records"):
            day = candidate["signal_date"]
            gap = positions[end]-positions[day]
            if day == end:
                status = "EVENT_DAY_NOT_FUTURE_TARGET"
            elif str(candidate["start_known"]).lower() not in {"true", "1"}:
                status = "UNKNOWN_RELAY_START"
            elif containing_id is None:
                status = "NO_LIMIT_RELAY_AT_EVENT_ENDPOINT"
            elif candidate["episode_id"] != containing_id:
                status = "DIFFERENT_RELAY_NOT_SUCCESS_TO_THIS_EVENT"
            elif gap > 30:
                status = "BEYOND_TARGET_HORIZON"
            else:
                status = "SAME_RELAY_POTENTIAL_LABEL_REQUIRES_PRIOR_TARGET_AND_FILL_AUDIT"
            counts[status] = counts.get(status, 0) + 1
            rows.append(dict(code=code, signal_date=day, boards=candidate["boards"],
                             episode_id=candidate["episode_id"], event_date=end,
                             target_pct=event["target_pct"], sessions_to_event_endpoint=gap,
                             scope_status=status, training_label=None, live_feature_allowed=False,
                             evidence_available_at=event["evidence_available_at"]))
        summaries.append(dict(code=code, event_date=end, target_pct=event["target_pct"],
                              official_interval_start=start, endpoint_episode_id=containing_id,
                              candidate_rows=len(candidates), scope_counts=counts))
    return dict(status="DIAGNOSTIC_ONLY", event_summaries=summaries, rows=rows,
                verified_trade_labels=0,
                limitations=["Event endpoint is not independently verified first-hit date",
                             "Same relay does not prove no earlier 100/200 threshold was reached",
                             "Disconnected from this endpoint does not prove failure against every earlier event",
                             "No fills, fees or profit simulated; all mappings are hindsight labels-only",
                             "Only signals within disclosed intervals audited; not a complete candidate universe"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("events", "observations", "episodes", "calendar", "output"):
        parser.add_argument("--"+field, required=True)
    args = parser.parse_args()
    report = audit(json.loads(Path(args.events).read_text(encoding="utf-8"))["events"],
                   pd.read_csv(args.observations, dtype={"code": str}),
                   pd.read_csv(args.episodes, dtype={"code": str}),
                   pd.read_csv(args.calendar).date.tolist())
    report["source_hashes"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in (args.events, args.observations, args.episodes, args.calendar)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["event_summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
