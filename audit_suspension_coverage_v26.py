"""Evidence-only gap overlay. Never changes board counts, labels or model scores."""
import argparse
import json
from pathlib import Path

import pandas as pd


def assess_gap(bars, calendar, evidence, asof):
    """Require full market-session gap coverage and point-in-time evidence."""
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("Asia/Shanghai")
    else:
        cutoff = cutoff.tz_convert("Asia/Shanghai")
    available = pd.Timestamp(evidence["publication_date"] + "T23:59:59", tz="Asia/Shanghai")
    result = dict(code=evidence["code"], source=evidence["source"],
                  status="UNRESOLVED", available_at=available.isoformat(),
                  labels_changed=False, regulatory_reset_inferred=False)
    if available > cutoff:
        return {**result, "reason": "EVIDENCE_NOT_YET_AVAILABLE"}
    if calendar != sorted(set(calendar)):
        raise ValueError("calendar must be sorted and unique")
    frame = bars[bars.code == evidence["code"]].sort_values("date")
    if frame.date.duplicated().any():
        raise ValueError("duplicate stock date")
    resume = evidence["resumption_date"]
    before = frame[frame.date < resume]
    after = frame[frame.date == resume]
    if before.empty or len(after) != 1:
        return {**result, "reason": "MISSING_BOUNDARY_BAR"}
    previous, resumed = before.iloc[-1], after.iloc[0]
    missing = [d for d in calendar if previous.date < d < resume]
    suspended = [d for d in calendar if evidence["suspension_start"] <= d < resume]
    if not missing or missing != suspended:
        return {**result, "reason": "GAP_NOT_FULLY_COVERED"}
    if abs(float(resumed.pre_close) - float(previous.close)) > 1e-6:
        return {**result, "reason": "PRICE_REFERENCE_DISCONTINUITY"}
    from leader_episode_inventory_v20 import true
    flags = [previous.status_verified, previous.st, previous.new_listing]
    if (not all(str(x).lower() in {"true", "false", "0", "1"} for x in flags)
            or not true(flags[0]) or true(flags[1]) or true(flags[2])):
        return {**result, "reason": "PRIOR_SECURITY_STATUS_UNVERIFIED"}
    import math
    close, pre = float(previous.close), float(previous.pre_close)
    if not all(math.isfinite(x) and x > 0 for x in (close, pre)):
        return {**result, "reason": "PRIOR_PRICE_INVALID"}
    prior_limit = abs(close - math.floor(pre * 110 + .5 + 1e-8) / 100) < 1e-6
    return {**result, "status": "GAP_EXPLAINED_NOT_RELABELLED",
            "source_status": evidence["evidence_status"],
            "missing_sessions": missing, "prior_limit_proxy": prior_limit,
            "reason": "PRE_SUSPENSION_LIMIT_REQUIRES_COUNT_POLICY" if prior_limit
            else "PRIOR_NONLIMIT_SUPPORTS_NEW_RUN_SUBJECT_TO_SOURCE_VERIFICATION"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    bars = pd.read_csv(args.bars, dtype={"code": str})
    calendar = pd.read_csv(args.calendar).date.tolist()
    report = {"status": "DIAGNOSTIC_ONLY", "asof": args.asof,
              "labels_changed": 0,
              "cases": [assess_gap(bars, calendar, e, args.asof) for e in evidence["cases"]]}
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
