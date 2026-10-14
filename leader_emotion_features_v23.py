"""Join explicit contemporaneous emotion statistics, never legacy forecasts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["limit_count", "consecutive_count", "max_board", "failed_count", "down_limit_count",
          "seal_rate", "failed_rate", "height_change", "limit_change", "middle_density",
          "ladder_gap_count", "yesterday_premium", "previous_limit_tail_loss", "promotion_overall",
          "promotion_1to2", "promotion_2to3", "promotion_3to4", "promotion_4to5", "promotion_mid",
          "space_feedback", "previous_limit_feedback_coverage"]


def join_emotion(features, market):
    required = {"date", "source", "source_timestamp", "data_status", *FIELDS}
    if not required.issubset(market):
        raise ValueError(f"missing market columns: {sorted(required-set(market))}")
    if any(c.startswith("market_") for c in features):
        raise ValueError("market columns already joined")
    market = market[["date", "source", "source_timestamp", "data_status", *FIELDS]].copy()
    market["date"] = pd.to_datetime(market.date).dt.strftime("%Y-%m-%d")
    if market.date.duplicated().any():
        raise ValueError("duplicate market date")
    def local_time(value):
        try:
            time = pd.Timestamp(value)
            if pd.isna(time):
                return pd.NaT
            return time.tz_localize("Asia/Shanghai") if time.tzinfo is None else time.tz_convert("Asia/Shanghai")
        except (ValueError, TypeError):
            return pd.NaT
    raw_times = market.source_timestamp.map(local_time)
    # These are reconstructed EOD records, not genuine original arrival times.
    # Require recorded date agreement; availability is conservatively 15:05.
    date_agrees = market.source_timestamp.astype(str).str[:10].eq(market.date)
    cutoff = pd.to_datetime(market.date + " 15:05:00").dt.tz_localize("Asia/Shanghai")
    valid = raw_times.notna() & date_agrees & raw_times.le(cutoff) & market.source.notna()
    for field in FIELDS:
        market[field] = pd.to_numeric(market[field], errors="coerce").replace([np.inf, -np.inf], np.nan).where(valid)
    market["evidence_valid"] = valid
    market = market.rename(columns={c: "market_"+c for c in market if c != "date"})
    output = features.merge(market, left_on="signal_date", right_on="date", how="left", validate="many_to_one").drop(columns="date")
    output["market_field_coverage"] = output[["market_"+f for f in FIELDS]].notna().mean(axis=1)
    output["market_join_status"] = np.where(output.market_evidence_valid.eq(True),
                                             "historical_reconstruction", "unavailable_or_invalid")
    output["theme_evidence_status"] = "not_joined"
    output["model_status"] = "DIAGNOSTIC_ONLY"
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite research feature bundle")
    result = join_emotion(pd.read_csv(args.features, dtype={"code": str}), pd.read_csv(args.market))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    report = dict(rows=len(result), dates=result.signal_date.nunique(),
                  matched_rows=int(result.market_join_status.eq("historical_reconstruction").sum()),
                  average_field_coverage=float(result.market_field_coverage.mean()),
                  calibrated_target_probability_available=False,
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.features,args.market]},
                  output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
