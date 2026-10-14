"""Full observed-pool price-opportunity experiment, NOT executable profit odds.

Fixed next-session limit-touch + following-session close-mark target. Auction
and fills are not known, so no model from this experiment is deployable.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from duration_factor_ablation_v28 import IDENTITY, top1_without_labels
from t1_scenario_v44 import net_return

EMOTION = ["market_yesterday_premium", "market_previous_limit_tail_loss",
           "market_failed_rate", "market_promotion_mid", "market_space_feedback"]
PAIRS = [("boards", "market_yesterday_premium"),
         ("sole_known_top_proxy", "market_failed_rate"),
         ("height_gap_to_known_top", "market_promotion_mid")]
COSTS = dict(shares=100, commission=.0003, minimum=5., sell_tax=.0005, slippage=.001)
START, END = "2025-07-28", "2025-12-26"


def rounded_limit(close, multiplier):
    return float((Decimal(str(close))*Decimal(multiplier)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))


def label_case(signal, entry, exit_bar):
    result = dict(label_ONLY=None, mark_return_ONLY=None, entry_one_price_ONLY=None,
                  exit_down_limit_ONLY=None, state="MISSING_OR_INVALID_DAILY", verified_trade=False)
    if any(x is None for x in (signal, entry, exit_bar)):
        return result
    for row in (signal, entry, exit_bar):
        vals = [float(row[k]) for k in ("open", "high", "low", "close", "pre_close", "volume")]
        o, h, l, c, p, v = vals
        if not all(np.isfinite(vals)) or min(o,h,l,c,p) <= 0 or v <= 0 or l > min(o,c) or h < max(o,c):
            return result
    if abs(float(entry["pre_close"])-float(signal["close"])) > .011 or abs(float(exit_bar["pre_close"])-float(entry["close"])) > .011:
        return dict(result, state="PRICE_BASIS_DISCONTINUITY")
    limit = rounded_limit(signal["close"], "1.10")
    touched = float(entry["high"]) >= limit-.00001
    # A high above the assumed limit means rule/status data is inconsistent.
    if float(entry["high"]) > limit+.011:
        return dict(result, state="LIMIT_RULE_UNCONFIRMED")
    mark = net_return(limit, float(exit_bar["close"]), **COSTS)
    return dict(result, state="PRICE_OPPORTUNITY_PROXY_ONLY", label_ONLY=int(touched and mark > 0),
                touched_limit_ONLY=touched, mark_return_ONLY=mark if touched else None,
                entry_one_price_ONLY=touched and float(entry["low"]) >= limit-.00001,
                exit_down_limit_ONLY=float(exit_bar["close"]) <= rounded_limit(exit_bar["pre_close"], ".90")+.00001)


def labels(pool, daily, calendar, end):
    if calendar != sorted(set(calendar)) or not daily.index.is_unique:
        raise ValueError("calendar or daily keys invalid")
    successor = dict(zip(calendar[:-1], calendar[1:]))
    rows = []
    for item in pool.to_dict("records"):
        d0 = item["signal_date"]
        d1 = successor.get(d0)
        d2 = successor.get(d1)
        bars = []
        for day in (d0,d1,d2):
            key = (item["code"],day)
            bars.append(daily.loc[key].to_dict() if day is not None and day < end and key in daily.index else None)
        rows.append(dict(code=item["code"], signal_date=d0, entry_date_ONLY=d1,
                         exit_date_ONLY=d2, **label_case(*bars)))
    return pd.DataFrame(rows)


def features_for_model(frame):
    frame = frame.copy()
    time = pd.to_datetime(frame.feature_available_not_before, utc=True)
    cutoff = pd.to_datetime(frame.signal_date+"T15:05:00+08:00", utc=True)
    if time.isna().any() or (time > cutoff).any() or (time.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d") != frame.signal_date).any():
        raise ValueError("invalid EOD feature availability")
    if frame.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate candidates")
    for col in IDENTITY+EMOTION:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf,-np.inf],np.nan)
    interactions = []
    for a,b in PAIRS:
        col = a+"_x_"+b
        frame[col] = frame[a]*frame[b]
        interactions.append(col)
    groups = {"height": ["boards"], "identity": IDENTITY,
              "identity_emotion_additive": IDENTITY+EMOTION,
              "identity_emotion_interactions": IDENTITY+EMOTION+interactions}
    return frame, groups


def summary(predictions):
    valid = predictions.label_ONLY.notna()
    y = predictions.loc[valid,"label_ONLY"]
    p = predictions.loc[valid,"probability_NOT_LIVE"]
    return dict(rows=len(predictions), labeled=int(valid.sum()), unresolved=int((~valid).sum()),
                price_opportunity_rate=float(y.mean()) if len(y) else None,
                auc=float(roc_auc_score(y,p)) if y.nunique()==2 else None,
                brier=float(brier_score_loss(y,p)) if len(y) else None,
                verified_trades=0, actual_profit_win_rate=None)


def run(frame, daily, calendar, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(status="DIAGNOSTIC_ONLY", target="next-session touch of inferred limit AND following-session net close mark positive",
                    validation_start=START, validation_end_exclusive=END,
                    training="one fixed pre-validation fit, two outcome sessions plus one purge session",
                    costs=COSTS, parameter_search=False, C=1, final_blind_test=False,
                    execution_verified=False, full_observed_pool=True)
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    frame = frame[frame.signal_date < END].copy()
    frame, groups = features_for_model(frame)
    cutoff = calendar[calendar.index(START)-3]
    train = frame[frame.signal_date < cutoff].copy()
    pool = frame[frame.signal_date >= START].copy()
    # Purge shared relay episodes; do not learn one episode then test its tail.
    train = train[~train.episode_id.isin(pool.episode_id)].copy()
    train = train.merge(labels(train,daily,calendar,START),on=["code","signal_date"],validate="one_to_one")
    complete = train[train.label_ONLY.notna()].copy()
    if complete.label_ONLY.nunique()!=2:
        raise ValueError("insufficient labeled training classes")
    weights = 1/complete.groupby("episode_id").episode_id.transform("size")
    outputs, report = [], {}
    # Rank all candidates before constructing validation labels.
    for name, columns in groups.items():
        pipeline = make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),
                                 StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=48))
        pipeline.fit(complete[columns],complete.label_ONLY,logisticregression__sample_weight=weights)
        prediction = pool[["code","signal_date","episode_id"]].copy()
        prediction["probability_NOT_LIVE"] = pipeline.predict_proba(pool[columns])[:,1]
        top = top1_without_labels(pool,prediction.probability_NOT_LIVE.to_numpy())
        top.to_csv(output/(name+"_selection_BEFORE_LABELS.csv"),index=False)
        prediction["selected"] = pd.MultiIndex.from_frame(prediction[["code","signal_date"]]).isin(pd.MultiIndex.from_frame(top[["code","signal_date"]]))
        prediction["group"] = name
        outputs.append(prediction)
    validation_labels = labels(pool,daily,calendar,END)
    for pred in outputs:
        name = pred.group.iloc[0]
        evaluated = pred.merge(validation_labels,on=["code","signal_date"],validate="one_to_one")
        top = evaluated[evaluated.selected]
        first = top.sort_values(["signal_date","code"]).drop_duplicates("episode_id")
        report[name] = dict(all_pool=summary(evaluated), daily_top1=summary(top), first_selected_episode=summary(first),
                            top1_one_price_cases=int(top.entry_one_price_ONLY.eq(True).sum()),
                            top1_down_limit_exit_cases=int(top.exit_down_limit_ONLY.eq(True).sum()))
        evaluated.to_csv(output/(name+"_evaluation_LABEL_ONLY.csv"),index=False)
    final = dict(protocol, train_rows=len(train), train_labeled_rows=len(complete), training_exclusive_cutoff=cutoff,
                 validation_rows=len(pool), validation_days=pool.signal_date.nunique(), groups=report,
                 limitations=["current-universe historical candidates are not complete historical market",
                              "reused historical validation, not fresh blind or walk-forward validation",
                              "daily high cannot prove auction timing, queue, touch order or actual entry",
                              "one-price and blocked-exit cases remain in proxy evaluation, never claimed profitable trades",
                              "retrospective EOD emotion, no actual contemporary receipt evidence or theme identity",
                              "fixed minimum fees for 100-share hypothetical marks, not a user's fee schedule",
                              "no calibrated executable-premium probability, deployed ranking, holding duration or portfolio P&L"])
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("features","daily","calendar","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.features,dtype={"code":str})
    daily=pd.read_csv(args.daily,dtype={"code":str})
    # Do not construct labels from the reserved later period.
    daily=daily[daily.date < END].set_index(["code","date"])
    report=run(frame,daily,pd.read_csv(args.calendar).date.tolist(),args.output)
    report["source_hashes"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.features,args.daily,args.calendar,Path(__file__))}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
