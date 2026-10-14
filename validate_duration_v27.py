"""Fixed-window calibration audit of the frozen training duration baseline.

All candidate rows remain in the report, including unknown identity/labels.
This is a reused validation window, not a new blind test or a trading backtest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from leader_duration_baseline_v27 import duration_rows


def horizon_label(row, horizon):
    if pd.isna(row.get("duration_sessions")):
        return None
    duration = int(row["duration_sessions"])
    broken = bool(row["broken_observed"])
    if broken and duration <= horizon:
        return 0
    if duration >= horizon:
        return 1
    return None


def score_pool(observations, model, start, end, horizons=(1, 3, 5)):
    if not model["training_exclusive_cutoff"] < start < end:
        raise ValueError("training/validation dates overlap or are invalid")
    if model["validation_start"] != start:
        raise ValueError("validation start differs from fixed training contract")
    pool = observations[(observations.signal_date >= start) & (observations.signal_date < end)].copy()
    if pool.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate candidate")
    for horizon in horizons:
        forecasts = []
        for row in pool.to_dict("records"):
            known = str(row["start_known"]).lower() in {"true", "1"}
            curve = model["per_height"].get(str(int(row["boards"])), {}).get("curve", []) if known else []
            forecasts.append(next((p["relay_survival_estimate"] for p in curve
                                   if p["future_session"] == horizon), None))
        pool[f"p_continue_{horizon}_NOT_LIVE"] = forecasts
    return pool


def metrics(pool, horizon):
    p = pd.to_numeric(pool[f"p_continue_{horizon}_NOT_LIVE"], errors="coerce")
    y = pd.to_numeric(pool[f"label_continue_{horizon}"], errors="coerce")
    valid = p.notna() & y.notna()
    n = len(pool)
    return dict(candidate_rows=n, forecast_rows=int(p.notna().sum()),
                unresolved_label_rows=int(y.isna().sum()), evaluated_rows=int(valid.sum()),
                evaluated_mean_forecast=float(p[valid].mean()) if valid.any() else None,
                evaluated_actual_rate=float(y[valid].mean()) if valid.any() else None,
                brier=float(((p[valid]-y[valid])**2).mean()) if valid.any() else None,
                all_candidate_rate_lower=float(y.eq(1).sum()/n) if n else None,
                all_candidate_rate_upper=float((y.eq(1).sum()+y.isna().sum())/n) if n else None)


def validate(observations, episodes, calendar, model, start, end):
    # Forecasts use only candidate fields and fixed training curves, before labels.
    pool = score_pool(observations, model, start, end)
    labels, _ = duration_rows(pool, episodes, calendar, end)
    columns = ["code", "signal_date", "duration_sessions", "broken_observed"]
    if labels.empty:
        labels = pd.DataFrame(columns=columns)
    pool = pool.merge(labels[columns], on=["code", "signal_date"], how="left", validate="one_to_one")
    for horizon in (1, 3, 5):
        pool[f"label_continue_{horizon}"] = [horizon_label(r, horizon) for r in pool.to_dict("records")]
    report = dict(status="DIAGNOSTIC_ONLY", validation_start=start, validation_exclusive_end=end,
                  training_exclusive_cutoff=model["training_exclusive_cutoff"],
                  refitted=False, blind_test=False, trades_simulated=0,
                  all_candidates={str(h): metrics(pool, h) for h in (1, 3, 5)},
                  per_height={str(int(height)): {str(h): metrics(group, h) for h in (1, 3, 5)}
                              for height, group in pool.groupby("boards")},
                  limitations=["Repeated observations of the same episode are correlated",
                               "Already used validation period, not independent blind evidence",
                               "OHLC inferred limit-up labels; incomplete universe and status evidence",
                               "Observed-subset Brier may be biased by informative censoring",
                               "Arithmetic missing-label bounds are not confidence intervals",
                               "No calibrated leader-target, holding-day or profitability claim"])
    return report, pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "episodes", "calendar", "model", "start", "end", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    # Input provenance must match the training artifact.
    for path in (args.observations, args.episodes, args.calendar):
        expected = model["source_hashes"].get(str(path))
        if expected != hashlib.sha256(Path(path).read_bytes()).hexdigest():
            raise ValueError("training input provenance mismatch: " + path)
    report, pool = validate(pd.read_csv(args.observations, dtype={"code": str}),
                            pd.read_csv(args.episodes, dtype={"code": str}),
                            pd.read_csv(args.calendar).date.tolist(), model, args.start, args.end)
    report["training_model_sha256"] = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pool.to_csv(output / "validation_forecasts_and_LABELS_NOT_LIVE.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: v for k, v in report.items() if k != "per_height"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
