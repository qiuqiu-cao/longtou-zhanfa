"""Fixed duration-factor experiment; no threshold tuning, no live probability.

Uses inferred continuous-board outcomes rather than the incomplete regulatory
event library. These outcomes cannot establish 100/200 target or profit rates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leader_duration_baseline_v27 import duration_rows
from leader_emotion_features_v23 import FIELDS
from validate_duration_v27 import horizon_label, metrics

IDENTITY = ["boards", "height_gap_to_known_top", "same_height_known_count",
            "sole_known_top_proxy", "candidate_pool_count", "unknown_height_pool_count"]
GROUPS = {"height": ["boards"], "identity_proxy": IDENTITY,
          "identity_plus_emotion": IDENTITY + ["market_" + f for f in FIELDS]}


def attach_labels(features, episodes, calendar, end):
    labels, _ = duration_rows(features, episodes, calendar, end)
    columns = ["code", "signal_date", "duration_sessions", "broken_observed"]
    if labels.empty:
        labels = pd.DataFrame(columns=columns)
    result = features.merge(labels[columns], on=["code", "signal_date"], how="left", validate="one_to_one")
    for h in (1, 3, 5):
        result[f"label_continue_{h}"] = [horizon_label(r, h) for r in result.to_dict("records")]
    return result


def top1_without_labels(pool, probabilities):
    # Unknown outcomes do not determine ranking eligibility.
    output = pool[["code", "signal_date", "episode_id"]].copy()
    output["probability_NOT_LIVE"] = probabilities
    return output.sort_values(["signal_date", "probability_NOT_LIVE", "code"],
                              ascending=[True, False, True]).drop_duplicates("signal_date")


def selected_summary(top, labeled_pool, horizon):
    column = f"label_continue_{horizon}"
    joined = top.merge(labeled_pool[["code", "signal_date", column]],
                       on=["code", "signal_date"], how="left", validate="one_to_one")
    first = joined.sort_values(["signal_date", "code"]).drop_duplicates("episode_id")
    y = first[column]
    n, unknown = len(first), int(y.isna().sum())
    hits = int(y.eq(1).sum())
    return dict(selected_days=len(top), distinct_selected_episodes=n,
                unresolved_episodes=unknown, hits=hits,
                all_episode_rate_lower=hits/n if n else None,
                all_episode_rate_upper=(hits+unknown)/n if n else None,
                actual_profit_win_rate=None)


def compare(features, episodes, calendar, groups=None):
    groups = GROUPS if groups is None else groups
    start, end = "2025-07-28", "2025-12-26"
    cutoff = calendar[calendar.index(start)-30]
    # Do not construct outcomes for the exploratory test period.
    features = features[features.signal_date < end].copy()
    allowed = set(column for columns in groups.values() for column in columns)
    if not allowed.issubset(features):
        raise ValueError("missing factor columns")
    feature_time = pd.to_datetime(features.feature_available_not_before, utc=True)
    expected_time = pd.to_datetime(features.signal_date + "T15:05:00+08:00", utc=True)
    if feature_time.isna().any() or (feature_time > expected_time).any():
        raise ValueError("invalid feature availability")
    for col in allowed:
        features[col] = pd.to_numeric(features[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    pool = features[features.signal_date >= start].copy()
    train_features = features[features.signal_date < cutoff].copy()
    train_features = train_features[~train_features.episode_id.isin(pool.episode_id)]
    train = attach_labels(train_features, episodes, calendar, cutoff)
    val = attach_labels(pool, episodes, calendar, end)
    reports, outputs = {}, []
    for horizon in (1, 3, 5):
        label = f"label_continue_{horizon}"
        complete = train[train[label].notna()].copy()
        if complete[label].nunique() != 2:
            raise ValueError("insufficient training classes")
        weights = 1/complete.groupby("episode_id").episode_id.transform("size")
        reports[str(horizon)] = {}
        for name, columns in groups.items():
            pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                     StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=28))
            pipeline.fit(complete[columns], complete[label], logisticregression__sample_weight=weights)
            probabilities = pipeline.predict_proba(pool[columns])[:, 1]
            evaluated = val.copy()
            evaluated[f"p_continue_{horizon}_NOT_LIVE"] = probabilities
            stats = metrics(evaluated, horizon)
            valid = evaluated[label].notna()
            stats["auc"] = (float(roc_auc_score(evaluated.loc[valid, label], probabilities[valid]))
                            if evaluated.loc[valid, label].nunique() == 2 else None)
            top = top1_without_labels(pool, probabilities)
            reports[str(horizon)][name] = dict(features=columns, train_labeled_rows=len(complete),
                                               validation=stats, daily_top1=selected_summary(top, val, horizon))
            output = pool[["code", "signal_date", "episode_id"]].copy()
            output["horizon"] = horizon
            output["group"] = name
            output["probability_NOT_LIVE"] = probabilities
            output["label_ONLY"] = val[label].to_numpy()
            outputs.append(output)
    return dict(status="DIAGNOSTIC_ONLY", training_exclusive_cutoff=cutoff,
                validation_start=start, validation_exclusive_end=end, pool_rows=len(pool),
                groups_by_horizon=reports, final_blind_test=False, hyperparameter_search=False,
                target_100_probability=None, target_200_probability=None, actual_trades=0,
                limitations=["identity factors measure height competition, not verified leader identity",
                             "retrospective EOD factors, not auction or intraday forecasts",
                             "all groups use identical candidates and per-horizon training labels",
                             "correlated observations, reused validation and informative censoring",
                             "unknown-start candidates scored using observed height lower bounds; watch only",
                             "no proof of profitability, regulatory-target success or stable improvement"]), pd.concat(outputs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "episodes", "calendar", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    report, predictions = compare(pd.read_csv(args.features, dtype={"code": str}),
                                  pd.read_csv(args.episodes, dtype={"code": str}),
                                  pd.read_csv(args.calendar).date.tolist())
    report["source_hashes"] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in (args.features, args.episodes, args.calendar, __file__)}
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(directory/"validation_predictions_NOT_LIVE.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({h: {g: dict(brier=v["validation"]["brier"], auc=v["validation"]["auc"],
                                 top1=v["daily_top1"]) for g, v in groups.items()}
                      for h, groups in report["groups_by_horizon"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
