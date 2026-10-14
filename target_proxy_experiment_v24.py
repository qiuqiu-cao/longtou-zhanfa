"""Exploratory time-split baseline on UNVERIFIED event-library labels.

No result here can establish actual hit rate, tradability or profit. Negatives
mean no event found in the incomplete discovery library, not proven absence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leader_emotion_features_v23 import FIELDS

FEATURES = ["boards", "height_gap_to_known_top", "same_height_known_count", "sole_known_top_proxy",
            "candidate_pool_count", "unknown_height_pool_count", "turnover", "amount_to_float_cap",
            "open_gap_pct", "amplitude_pct", "one_price_day_proxy", "close"] + ["market_"+f for f in FIELDS]


def audit_proxy_labels(features, episodes, events, calendar):
    positions = {d:i for i,d in enumerate(calendar)}
    ends = episodes.set_index("episode_id")
    if not ends.index.is_unique:
        raise ValueError("duplicate episode metadata")
    event_map = {}
    for event in events:
        event_map.setdefault(event["code"], set()).add(event["event_date"])
    rows = []
    for row in features.to_dict("records"):
        ep = ends.loc[row["episode_id"]]
        if str(ep.start_known).lower() not in {"true", "1"}:
            rows.append({**row,"proxy_target":None,"label_end":None,"label_status":"START_UNKNOWN"})
            continue
        signal = row["signal_date"]
        dates = sorted(d for d in event_map.get(row["code"], set()) if ep.start_date <= d <= ep.last_limit_date)
        if any(d <= signal for d in dates):
            rows.append({**row,"proxy_target":None,"label_end":None,"label_status":"TARGET_ALREADY_OCCURRED_PROXY"})
            continue  # The first target in this relay has already occurred.
        future = [d for d in dates if d > signal and positions[d]-positions[signal] <= 30]
        if future:
            label, end = 1, future[0]
        elif ep.termination == "BROKEN_BOARD" and positions[ep.terminal_date]-positions[signal] <= 30:
            label, end = 0, ep.terminal_date
        else:
            reason = "HORIZON_EXCEEDED" if ep.termination == "BROKEN_BOARD" else str(ep.termination)
            rows.append({**row,"proxy_target":None,"label_end":None,"label_status":reason})
            continue
        rows.append({**row, "proxy_target":label, "label_end":end,
                     "label_status":"incomplete_event_library_proxy_NOT_VERIFIED"})
    return pd.DataFrame(rows)


def proxy_labels(features, episodes, events, calendar):
    audited = audit_proxy_labels(features, episodes, events, calendar)
    return audited[audited.proxy_target.notna()].copy() if len(audited) else audited


def metrics(frame, probabilities):
    y = frame.proxy_target.to_numpy()
    return dict(rows=len(frame), positive_rows=int(y.sum()),
                auc=float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
                brier=float(brier_score_loss(y, probabilities)))


def selection_stats(frame):
    # One earliest selected signal per episode for denominators.
    distinct = frame.sort_values(["signal_date", "code"]).drop_duplicates("episode_id")
    unresolved = int(distinct.proxy_target.isna().sum())
    wins = int(distinct.proxy_target.eq(1).sum())
    n = len(distinct)
    return dict(selected_rows=len(frame), selected_episodes=n, unresolved_episodes=unresolved,
                proxy_hit_rate=float(distinct.proxy_target.mean()) if n and not unresolved else None,
                proxy_hit_rate_lower=wins/n if n else None,
                proxy_hit_rate_upper=(wins+unresolved)/n if n else None,
                verified_hit_rate=None, actual_profit_win_rate=None)


def full_pool_predictions(model, features, labels, test_start):
    pool = features[features.signal_date >= test_start].copy()
    pool["proxy_probability_NOT_LIVE"] = model.predict_proba(pool[FEATURES])[:,1]
    # Outcome availability must never decide which candidate wins the ranking.
    metadata = labels[["code","signal_date","proxy_target","label_end","label_status"]]
    return pool.merge(metadata,on=["code","signal_date"],how="left",validate="one_to_one")


def experiment(features, episodes, events, calendar):
    data = proxy_labels(features, episodes, events, calendar)
    dates = sorted(data.signal_date.unique())
    if len(dates) < 150:
        raise ValueError("insufficient time coverage")
    val_start, test_start = dates[int(len(dates)*.6)], dates[int(len(dates)*.8)]
    position = {d:i for i,d in enumerate(calendar)}
    val_cut = calendar[max(0,position[val_start]-30)]
    test_cut = calendar[max(0,position[test_start]-30)]
    train = data[(data.signal_date < val_cut) & (data.label_end < val_cut)].copy()
    val = data[(data.signal_date >= val_start) & (data.signal_date < test_cut) & (data.label_end < test_cut)].copy()
    test = data[data.signal_date >= test_start].copy()
    later_episodes = set(val.episode_id) | set(test.episode_id)
    train = train[~train.episode_id.isin(later_episodes)]
    val = val[~val.episode_id.isin(set(test.episode_id))]
    if train.proxy_target.nunique() != 2 or min(len(train), len(val), len(test)) == 0:
        raise ValueError("insufficient split classes or rows")
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=24))
    # Each relay has total training weight one, rather than rewarding long runs.
    weights = 1/train.groupby("episode_id").episode_id.transform("size")
    model.fit(train[FEATURES], train.proxy_target, logisticregression__sample_weight=weights)
    report = dict(status="DIAGNOSTIC_ONLY", label_warning="incomplete event library; both positives and negatives may be wrong",
                  train_rows=len(train), validation_start=val_start, exploratory_test_start=test_start,
                  embargo_sessions=30, features=FEATURES, target_threshold_fixed_before_fit=0.8,
                  actual_trades_simulated=0, expected_holding_days=None, splits={})
    for name, frame in [("train",train),("validation",val),("exploratory_test",test)]:
        probabilities = model.predict_proba(frame[FEATURES])[:,1]
        frame["proxy_probability_NOT_LIVE"] = probabilities
        report["splits"][name] = metrics(frame, probabilities)
    pool = full_pool_predictions(model,features,data,test_start)
    report["ranking_pool_rows"] = len(pool)
    report["ranking_pool_unresolved_rows"] = int(pool.proxy_target.isna().sum())
    report["ranking_policy"] = "all_feature_candidates_before_outcome_join; eligibility and unresolved labels require further audit"
    top = pool.sort_values(["signal_date","proxy_probability_NOT_LIVE","code"], ascending=[True,False,True]).drop_duplicates("signal_date")
    height = pool.sort_values(["signal_date","boards","code"], ascending=[True,False,True]).drop_duplicates("signal_date")
    report["test_model_daily_top1"] = selection_stats(top)
    report["test_highest_board_baseline"] = selection_stats(height)
    report["test_model_top1_above_80pct"] = selection_stats(top[top.proxy_probability_NOT_LIVE >= .8])
    report["not_final_blind_test"] = True
    return report, pool[["code","signal_date","episode_id","proxy_target","label_end","proxy_probability_NOT_LIVE","label_status"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--episodes",type=Path,required=True)
    parser.add_argument("--events",type=Path,required=True)
    parser.add_argument("--calendar",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report, predictions=experiment(pd.read_csv(args.features,dtype={"code":str}),
                                   pd.read_csv(args.episodes,dtype={"code":str}),
                                   json.loads(args.events.read_text(encoding="utf-8"))["events"],
                                   pd.read_csv(args.calendar).date.tolist())
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    predictions.to_csv(args.output/"exploratory_predictions_NOT_LIVE.csv",index=False,encoding="utf-8-sig")
    print(json.dumps(report,ensure_ascii=False,indent=2))
