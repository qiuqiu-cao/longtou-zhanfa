"""Descriptive prior-session context for ALL supplied opening observations."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from leader_holding_review_v21 import stamp
from leader_workflow_v30 import clean

FIELDS = ["height_gap_to_known_top", "same_height_known_count", "sole_known_top_proxy",
          "turnover", "amount_to_float_cap", "one_price_day_proxy",
          "market_failed_rate", "market_down_limit_count", "market_yesterday_premium",
          "market_promotion_mid", "market_space_feedback", "market_field_coverage"]


def attach(rows, features):
    selected = features[["code", "signal_date", "feature_available_not_before", *FIELDS]].copy()
    if selected.duplicated(["code", "signal_date"]).any():
        raise ValueError("ambiguous feature identity")
    indexed = selected.set_index(["code", "signal_date"])
    output = []
    for row in rows:
        result = dict(row, prior_context=None, prior_context_status="MISSING")
        key = (row["code"], row["signal_date"])
        if key in indexed.index:
            feature = indexed.loc[key, :]
            available = stamp(feature["feature_available_not_before"])
            decision = stamp(row["entry_date"]+"T09:35:00+08:00")
            if available.date().isoformat() != row["signal_date"] or available >= decision:
                raise ValueError("feature date mismatch or future feature")
            result["prior_context"] = {k:feature[k] for k in FIELDS}
            result["prior_context_status"] = "HISTORICAL_RECONSTRUCTION_NOT_FORWARD_VERIFIED"
            result["prior_feature_available_not_before"] = available.isoformat()
        output.append(result)
    return clean(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    rows = attach(replay["rows"], pd.read_csv(args.features, dtype={"code":str}))
    sealed = [r for r in rows if r.get("observation", {}).get("state") == "CLOSE_AT_LIMIT_PROXY"]
    by_date = {}
    for r in sealed:
        group = by_date.setdefault(r["entry_date"], {"observations":0,"promotion_proxy_true":0,"promotion_proxy_false":0})
        group["observations"] += 1
        group["promotion_proxy_true" if r["promotion_LABEL_ONLY"] else "promotion_proxy_false"] += 1
    report = dict(status="DIAGNOSTIC_ONLY", rows=rows, sealed_by_date=by_date,
                  fitted_model=False, theme_evidence="UNAVAILABLE", calibrated_probability=None,
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.replay,args.features)},
                  limitations=["outcome-selected dates; explanatory comparison not causal proof or validation",
                               "shared date-level emotion cannot alone rank candidates on the same date",
                               "prior-session context is not live 09:35 market emotion"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(by_date=by_date, sealed_context=[{k:r[k] for k in
          ("code","entry_date","boards","promotion_LABEL_ONLY","prior_context")} for r in sealed]), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
