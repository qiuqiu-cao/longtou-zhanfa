"""Same-date full archived pool replay; no fitted thresholds or return claims."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import pandas as pd
from audit_historical_5min_v33 import inspect
from opening_replay_v34 import observe, next_bar_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, help="explicit preselected cohort CSV")
    parser.add_argument("--extra-archive-root", type=Path, action="append", default=[])
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort or args.root/"same_date_cohort_20260904/cohort.csv", dtype={"code": str})
    daily = pd.read_csv(args.daily, dtype={"code": str}, usecols=["code", "date", "open", "high", "low", "close"])
    daily = daily.set_index(["code", "date"])
    if not daily.index.is_unique:
        raise ValueError("duplicate daily bars")
    available = {}
    manifests = [p for root in [args.root, *args.extra_archive_root] for p in root.glob("minute_*/manifest.json")]
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            if case["error_code"] != "0":
                continue
            path = manifest_path.parent/(case["code"]+"_"+case["date"]+".json")
            if hashlib.sha256(path.read_bytes()).hexdigest() != case["raw_sha256"]:
                raise ValueError("raw hash mismatch")
            key = (case["code"], case["date"])
            if key in available:
                raise ValueError("duplicate archive")
            available[key] = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for candidate in cohort.to_dict("records"):
        raw = available.get((candidate["provider_code"], candidate["entry_date"]))
        if raw is None:
            rows.append(dict(candidate, data_status="MISSING_ARCHIVE"))
            continue
        day_key = (candidate["code"], candidate["entry_date"])
        prior_key = (candidate["code"], candidate["signal_date"])
        if day_key not in daily.index or prior_key not in daily.index:
            rows.append(dict(candidate, data_status="MISSING_DAILY_COMPARATOR"))
            continue
        day_bar = daily.loc[day_key, :]
        previous = daily.loc[prior_key, :]
        limit = float((Decimal(str(previous.close))*Decimal("1.10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        audit = inspect(raw, day_bar)
        if not audit["status"].startswith("DAILY_OHLC_"):
            rows.append(dict(candidate, data_status=audit["status"]))
            continue
        bars = [dict(zip(raw["fields"], row)) for row in raw["rows"]]
        asof = candidate["entry_date"].replace("-", "")+"093500000"
        rows.append(dict(candidate, data_status=audit["status"], differences=audit["differences"],
                         limit_price_proxy=limit, observation=observe(bars, asof, limit),
                         execution_LABEL_ONLY=next_bar_diagnostic(bars, asof, limit),
                         promotion_LABEL_ONLY=abs(float(day_bar.close)-limit)<1e-6))
    groups = {}
    for row in rows:
        if "observation" not in row:
            continue
        key = row["observation"]["state"]
        group = groups.setdefault(key, dict(observations=0, promotion_proxy_true=0, promotion_proxy_false=0))
        group["observations"] += 1
        group["promotion_proxy_true" if row["promotion_LABEL_ONLY"] else "promotion_proxy_false"] += 1
    report = dict(status="DIAGNOSTIC_ONLY", rows=rows, groups=groups,
                  data_status_counts=dict(Counter(r["data_status"] for r in rows)),
                  verified_fills=0, target_100_200_probabilities=None,
                  limitations=["historical research cohort and reconstructed universe: not blind validation",
                               "promotion is daily price proxy, not profitable trade or 100/200 target",
                               "OHLC mismatches retained, not silently excluded",
                               "five-minute timestamps assumed bar-end; receipt latency unknown"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
