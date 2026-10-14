"""Fixed validation-only ablation; never opens V24 exploratory test predictions."""
import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from target_proxy_experiment_v24 import FEATURES, proxy_labels, metrics, selection_stats


def score_full_pool(model, columns, pool, labels):
    scored = pool.copy()
    scored["probability"] = model.predict_proba(scored[columns])[:,1]
    scored = scored.merge(labels[["code","signal_date","proxy_target"]],
                          on=["code","signal_date"],how="left",validate="one_to_one")
    return scored.sort_values(["signal_date","probability","code"],ascending=[True,False,True]).drop_duplicates("signal_date")


def compare(data, calendar, features):
    validation_start, test_start = "2025-07-28", "2026-02-10"
    positions = {d:i for i,d in enumerate(calendar)}
    train_end = calendar[positions[validation_start]-30]
    validation_end = calendar[positions[test_start]-30]
    train = data[(data.signal_date < train_end) & (data.label_end < train_end)].copy()
    val = data[(data.signal_date >= validation_start) & (data.signal_date < validation_end)
               & (data.label_end < validation_end)].copy()
    pool = features[(features.signal_date >= validation_start) & (features.signal_date < validation_end)].copy()
    train = train[~train.episode_id.isin(pool.episode_id)]
    groups = {"height_only": ["boards"],
              "identity_and_price_volume": [f for f in FEATURES if not f.startswith("market_")],
              "plus_emotion": FEATURES}
    reports = {}
    for name, columns in groups.items():
        pipeline = make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),
                                 StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=24))
        weights = 1/train.groupby("episode_id").episode_id.transform("size")
        pipeline.fit(train[columns],train.proxy_target,logisticregression__sample_weight=weights)
        probabilities = pipeline.predict_proba(val[columns])[:,1]
        top = score_full_pool(pipeline,columns,pool,val)
        reports[name] = dict(features=columns,metrics=metrics(val,probabilities),
                             daily_top1=selection_stats(top),
                             top1_above_80pct=selection_stats(top[top.probability>=.8]))
    return dict(status="DIAGNOSTIC_ONLY",train_rows=len(train),validation_rows=len(val),
                validation_start=validation_start,validation_exclusive_end=validation_end,
                embargo_sessions=30,groups=reports,ranking_pool_rows=len(pool),
                ranking_policy="full_feature_pool_before_label_join",
                limitation="unverified event proxy; single validation segment; no proof of factor stability or profit")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    output = root/"results/v25/validation_ablation_full_pool_20260904.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite validation experiment")
    features = pd.read_csv(root/"data/v23/candidate_emotion_features_strict.csv",dtype={"code":str})
    episodes = pd.read_csv(root/"data/v20/episode_inventory_20260904_strict/episode_outcomes_LABEL_ONLY.csv",dtype={"code":str})
    events = json.loads((root/"data/v19/target_event_library_20260904.json").read_text(encoding="utf-8"))["events"]
    calendar = pd.read_csv(root/"data/v18/benchmarks_20260904/sz399107.csv").date.tolist()
    # Do not construct proxy labels for any signal in the test segment.
    features = features[features.signal_date < "2026-02-10"]
    report = compare(proxy_labels(features,episodes,events,calendar),calendar,features)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
