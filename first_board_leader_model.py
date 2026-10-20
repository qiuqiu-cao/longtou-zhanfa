"""Train a first-board-to-multi-board ranker with purged temporal replay.

Historical OHLC reconstruction is research evidence, never a fill or a verified
theme-leader label. The separate inference command loads only an explicit bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "first-board-multiboard-research-20261020"
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/v16/point_in_time_rebuild_cutoff_20260831_v2"
INPUTS = {
    "bars": SOURCE / "point_in_time_current_daily_bars.csv",
    "recovered": SOURCE / "verified_recovered_daily_bars.csv",
    "calendar": ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json",
    "security_intervals": ROOT / "data/v11/security_master_intervals.csv",
}
FEATURES = [
    "log_price", "log_amount", "log_float_cap", "turnover", "amount_cap",
    "gap", "amplitude", "body", "one_price", "close_to_open",
    "return_5", "return_10", "return_20", "return_60", "volume_ratio_5",
    "volume_ratio_20", "amount_ratio_5", "close_ma5", "close_ma20", "close_ma60",
    "breakout_20", "position_60", "prior_limits_20", "prior_failed_20",
    "previous_limit_age", "prior_volatility_20", "listing_age_sessions",
    "market_limits", "market_first", "market_relays", "market_failed",
    "market_failed_rate", "market_max_height", "market_down_limits",
    "market_positive_fraction", "market_median_return", "previous_first_promotion",
    "previous_relay_promotion", "previous_top_return", "previous_top_broken",
    "turnover_rank", "amount_rank", "volume_ratio_rank", "pre_momentum_rank",
]
HORIZON = 5
TARGET = "reach4_LABEL_ONLY"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def true_values(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def limit_cents(pre_close, up=True):
    return np.floor(np.asarray(pre_close, dtype=float) * (110 if up else 90) + .5 + 1e-8)


def protocol():
    return dict(version=VERSION, recorded_at=now(), target=TARGET,
        hypothesis="First-close price/volume, prior momentum, competition and lagged emotion distinguish durable four-board paths.",
        input_paths={k: str(p.resolve()) for k, p in INPUTS.items()},
        source_hashes={str(p.resolve()): sha(p) for p in INPUTS.values()},
        target_definition="Known first board at S close; S+1,S+2,S+3 all close at exact 10% limit price on adjacent official sessions.",
        secondary_targets="Reach 2/3/5 consecutive boards; five-session same-first-board-cohort highest continuous height >=4, ties retained. This is a height proxy, not verified theme leadership.",
        baseline="Lowest observed signal-day turnover, code ascending ties, missing last. Also report whole-pool base rate.",
        challengers={"logistic":dict(C=.25,class_weight="balanced",max_iter=1000),
                     "hist_gradient":dict(max_iter=160,max_leaf_nodes=15,min_samples_leaf=60,learning_rate=.06,l2_regularization=8,class_weight="balanced",early_stopping=False,random_state=42)},
        development="Train signals before 2025-01-01 with label end strictly before 2025-01-01; validate 2025H1 once. Choose largest Top3 reach4 hit count, logistic wins ties.",
        evaluation="2025-07-01 through last fully five-session-mature source date; selected family only, monthly rolling previous 252 official signal sessions, label end strictly before test month.",
        historical_test_status="TEMPORAL_OUT_OF_SAMPLE_REPLAY_NOT_PRISTINE_BLIND: archive previously explored, sources collected later.",
        horizons=HORIZON,features=FEATURES,topk=[1,3,5],positive_threshold=.5,
        acceptance="Top3 reach4 precision paired weekly-block 95% lower delta >0; both 2025H2 and 2026 positive deltas; non-one-price Top3 delta >=0. Historical pass does not deploy.",
        simulation="Prior-close Top1 only, no fallback; precommitted next-open proxy; skip limit-up/down opens and unknowns; 100-share lots, 20% equity cap, 1% prior-day amount cap, commission 3bp/min5 each side, transfer1e-5 each side, stamp5bp sell, 10bp adverse slippage each side. Exit at B+2 close or next non-limit close, no same-day stop; missing/blocked exits remain open and marked.",
        uncertainty="2,000 paired moving weekly-block bootstrap replicates, fixed seed42, five-week blocks; no probability calibration claim.",
        stop="One fixed comparison; do not retune on evaluation. Preserve old production, schedules, ledgers and failures.",
        budget_minutes=30)


def build_dataset(inputs=INPUTS, feature_cutoff=None):
    sessions = json.loads(Path(inputs["calendar"]).read_text(encoding="utf-8"))["sessions"]
    if sessions != sorted(set(sessions)):
        raise ValueError("calendar must be sorted and unique")
    pos = {d:i for i,d in enumerate(sessions)}
    cols = ["date","code","name","open","high","low","close","pre_close","volume","amount","turnover","float_cap","st","new_listing","status_verified","source"]
    bars = pd.concat([pd.read_csv(inputs[k], usecols=cols, dtype={"code":str}, low_memory=False) for k in ("bars","recovered")], ignore_index=True)
    audit = dict(input_bar_rows=len(bars),source_collected_later=True,full_historical_universe_complete=False,
                 survivorship_bias_resolved=False,theme_leader_truth_available=False,
                 price_basis="Archived unadjusted OHLC; exact rounded 10% price inference, not an independent official limit-price feed")
    bars["code"] = bars.code.str.zfill(6)
    bars["date"] = bars.date.astype(str).str[:10]
    if feature_cutoff is not None:
        bars=bars[bars.date.le(feature_cutoff)].copy()
    if bars.duplicated(["date","code"]).any():
        duplicated = bars[bars.duplicated(["date","code"],keep=False)]
        n = duplicated.groupby(["date","code"])[["open","high","low","close","pre_close"]].nunique().max(axis=1)
        if n.gt(1).any():
            raise ValueError("conflicting duplicate price sources")
        bars = bars.drop_duplicates(["date","code"], keep="last")
    numeric = ["open","high","low","close","pre_close","volume","amount","turnover","float_cap"]
    bars[numeric] = bars[numeric].apply(pd.to_numeric, errors="coerce")
    bars["scope_known"] = true_values(bars.status_verified)
    bars["eligible"] = bars.scope_known & ~true_values(bars.st) & ~true_values(bars.new_listing) & bars.code.str.startswith(("000","001","002","003","600","601","603","605"))
    bars["p"] = bars.date.map(pos)
    audit["outside_calendar_bars"] = int(bars.p.isna().sum())
    bars = bars[bars.p.notna()].sort_values(["code","p"]).reset_index(drop=True)
    bars["p"] = bars.p.astype(int)
    price_valid = bars[["open","high","low","close","pre_close"]].gt(0).all(axis=1) & bars.high.ge(bars[["open","close","low"]].max(axis=1)) & bars.low.le(bars[["open","close"]].min(axis=1))
    bars["valid_price"] = price_valid
    bars["cap"] = limit_cents(bars.pre_close)
    bars["floor"] = limit_cents(bars.pre_close, False)
    cents = (bars[["open","high","low","close"]]*100).round()
    bars["regular_price"] = price_valid & cents.high.le(bars.cap) & cents.low.ge(bars.floor)
    bars["limit"] = bars.eligible & bars.regular_price & cents.close.eq(bars.cap)
    bars["failed"] = bars.eligible & bars.regular_price & cents.high.eq(bars.cap) & ~bars.limit
    bars["down_limit"] = bars.eligible & bars.regular_price & cents.close.eq(bars.floor)
    g = bars.groupby("code",sort=False)
    adjacent = bars.p.sub(g.p.shift()).eq(1)
    previous_valid = g.valid_price.shift().fillna(False).astype(bool)
    previous_limit = g.limit.shift().fillna(False).astype(bool)
    link = bars.limit & previous_limit & adjacent
    run = (~link).groupby(bars.code).cumsum()
    bars["height"] = bars.limit.astype(int).groupby([bars.code,run]).cumsum()
    # A missing predecessor is unknown, never automatically a first board.
    bars["first"] = bars.limit & ~previous_limit & adjacent & previous_valid & g.eligible.shift().fillna(False).astype(bool)
    bars["previous_limit"] = (previous_limit & adjacent)
    bars["previous_height"] = g.height.shift().where(adjacent)
    bars["day_return"] = bars.close/bars.pre_close-1
    for c, raw in [("log_price","close"),("log_amount","amount"),("log_float_cap","float_cap")]:
        bars[c] = np.log(bars[raw].where(bars[raw].gt(0)))
    bars["amount_cap"] = bars.amount/bars.float_cap.where(bars.float_cap.gt(0))
    bars["gap"] = bars.open/bars.pre_close-1
    bars["amplitude"] = (bars.high-bars.low)/bars.pre_close
    bars["body"] = (bars.close-bars.open)/bars.pre_close
    bars["one_price"] = cents.high.eq(cents.low).astype(int)
    bars["close_to_open"] = bars.close/bars.open-1
    for h in (5,10,20,60):
        bars[f"return_{h}"] = (bars.close/g.close.shift(h)-1).where(bars.p.sub(g.p.shift(h)).eq(h))
    for h in (5,20,60):
        contiguous = bars.p.sub(g.p.shift(h-1)).eq(h-1)
        ma = g.close.transform(lambda s:s.rolling(h,min_periods=h).mean())
        bars[f"close_ma{h}"] = (bars.close/ma-1).where(contiguous)
    for h in (5,20):
        denom = g.volume.transform(lambda s:s.shift().rolling(h,min_periods=h).mean())
        bars[f"volume_ratio_{h}"] = (bars.volume/denom.where(denom.gt(0))).where(bars.p.sub(g.p.shift(h)).eq(h))
    prior_amount = g.amount.transform(lambda s:s.shift().rolling(5,min_periods=5).mean())
    bars["amount_ratio_5"] = (bars.amount/prior_amount.where(prior_amount.gt(0))).where(bars.p.sub(g.p.shift(5)).eq(5))
    hi20 = g.high.transform(lambda s:s.shift().rolling(20,min_periods=20).max())
    bars["breakout_20"] = (bars.close/hi20-1).where(bars.p.sub(g.p.shift(20)).eq(20))
    hi60 = g.high.transform(lambda s:s.shift().rolling(60,min_periods=60).max())
    lo60 = g.low.transform(lambda s:s.shift().rolling(60,min_periods=60).min())
    bars["position_60"] = ((bars.pre_close-lo60)/(hi60-lo60).where(hi60.gt(lo60))).where(bars.p.sub(g.p.shift(60)).eq(60))
    for out,col in [("prior_limits_20","limit"),("prior_failed_20","failed")]:
        bars[out] = g[col].transform(lambda s:s.shift().rolling(20,min_periods=20).sum()).where(bars.p.sub(g.p.shift(20)).eq(20))
    bars["prior_volatility_20"] = g.day_return.transform(lambda s:s.shift().rolling(20,min_periods=20).std()).where(bars.p.sub(g.p.shift(20)).eq(20))
    last_limit = bars.p.where(bars.limit).groupby(bars.code).ffill().groupby(bars.code).shift()
    bars["previous_limit_age"] = (bars.p-last_limit).clip(upper=253).fillna(253)
    master = pd.read_csv(inputs["security_intervals"],dtype={"code":str})
    listing = pd.to_datetime(master.groupby("code").listing_date.first(),errors="coerce")
    listing_pos = listing.map(lambda x:np.searchsorted(sessions,str(x.date())) if pd.notna(x) else np.nan)
    # Left-censored at this archive's start, NOT the full historical listing age.
    bars["listing_age_sessions"] = bars.p-bars.code.map(listing_pos)
    eligible = bars[bars.eligible & bars.valid_price]
    market = eligible.groupby("date").agg(market_limits=("limit","sum"),market_first=("first","sum"),market_relays=("height",lambda s:s.ge(2).sum()),market_failed=("failed","sum"),market_max_height=("height","max"),market_down_limits=("down_limit","sum"),market_positive_fraction=("day_return",lambda s:s.gt(0).mean()),market_median_return=("day_return","median"))
    market["market_failed_rate"] = market.market_failed/(market.market_failed+market.market_limits).replace(0,np.nan)
    previous_first = bars.previous_height.eq(1) & bars.previous_limit
    previous_relay = bars.previous_height.ge(2) & bars.previous_limit
    top = bars.previous_height.eq(bars.groupby("date").previous_height.transform("max")) & bars.previous_limit
    for name,mask,col in [("previous_first_promotion",previous_first,"limit"),("previous_relay_promotion",previous_relay,"limit"),("previous_top_return",top,"day_return")]:
        market[name] = bars[mask].groupby("date")[col].mean()
    market["previous_top_broken"] = 1-bars[top].groupby("date").limit.mean()
    first = bars[bars.first].copy()
    audit.update(first_board_rows=len(first),first_board_sessions=first.date.nunique(),scope_verified_rows=int(first.scope_known.sum()),first_date=first.date.min(),last_date=first.date.max(),source_distribution=first.source.value_counts().to_dict())
    first = first.merge(market,on="date",how="left",validate="many_to_one")
    for out,col in [("turnover_rank","turnover"),("amount_rank","amount"),("volume_ratio_rank","volume_ratio_5"),("pre_momentum_rank","return_20")]:
        first[out] = first.groupby("date")[col].rank(pct=True)
    # Join exact official successor keys. Never use the next observed stock bar.
    look = bars.set_index(["p","code"])[["limit","valid_price","eligible"]]
    alive = np.ones(len(first),dtype=bool)
    observed = np.ones(len(first),dtype=bool)
    heights = np.ones(len(first),dtype=int)
    for h in range(1,HORIZON+1):
        idx = pd.MultiIndex.from_arrays([first.p.to_numpy()+h,first.code.to_numpy()],names=["p","code"])
        future = look.reindex(idx)
        known = future.valid_price.fillna(False).to_numpy(dtype=bool) & future.eligible.fillna(False).to_numpy(dtype=bool)
        observed &= known
        alive &= known & future.limit.fillna(False).to_numpy(dtype=bool)
        heights += alive
        if h<=4:
            first[f"reach{h+1}_LABEL_ONLY"] = np.where(observed,alive.astype(float),np.nan)
    # Use one uniform mature/observable horizon for family selection and all heads.
    mature = first.p+HORIZON <= int(bars.p.max())
    first["label_end"] = first.p.map(lambda p:sessions[p+HORIZON] if p+HORIZON<len(sessions) else None)
    first["label_observed"] = observed & mature
    first["max_height_5_LABEL_ONLY"] = np.where(observed & mature,heights,np.nan)
    label_columns = [f"reach{h}_LABEL_ONLY" for h in (2,3,4,5)]
    first.loc[~first.label_observed,label_columns] = np.nan
    all_cohort_known = first.groupby("date").label_observed.transform("all")
    max_height = first.groupby("date").max_height_5_LABEL_ONLY.transform("max")
    first["cohort_leader_LABEL_ONLY"] = np.where(all_cohort_known,(first.max_height_5_LABEL_ONLY.eq(max_height) & max_height.ge(4)).astype(float),np.nan)
    first[FEATURES] = first[FEATURES].replace([np.inf,-np.inf],np.nan)
    first = first.rename(columns={"date":"signal_date"}).sort_values(["signal_date","code"]).reset_index(drop=True)
    audit.update(mature_observed_rows=int(first.label_observed.sum()),unknown_or_right_censored_rows=int((~first.label_observed).sum()),cohort_label_known_rows=int(first.cohort_leader_LABEL_ONLY.notna().sum()),features=FEATURES,
                 last_source_session=bars.date.max(),last_mature_signal=first.loc[first.label_observed,"signal_date"].max())
    return first,bars,sessions,audit


def make_model(family):
    if family=="logistic":
        return make_pipeline(SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.25,class_weight="balanced",max_iter=1000,random_state=42))
    if family=="hist_gradient":
        return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=60,learning_rate=.06,l2_regularization=8,class_weight="balanced",early_stopping=False,random_state=42)
    raise ValueError("unknown model family")


def selected(rows,score,k):
    return rows.sort_values(["signal_date",score,"code"],ascending=[True,False,True],na_position="last").groupby("signal_date",sort=False).head(k)


def metric(rows,score,k,target=TARGET):
    take = selected(rows,score,k)
    known = take[target].notna()
    positives = int(rows[target].eq(1).sum())
    hits = int(take[target].eq(1).sum())
    bydate = take.groupby("signal_date")[target].agg(["count","sum"])
    complete=bydate["count"].eq(take.groupby("signal_date").size())
    return dict(selected=len(take),selected_dates=take.signal_date.nunique(),known=int(known.sum()),unknown=int((~known).sum()),hits=hits,
                precision=hits/int(known.sum()) if known.any() else None,recall=hits/positives if positives else None,
                false_positives=int(take[target].eq(0).sum()),false_negatives=positives-hits,
                positive_pool=positives,pool_label_known=int(rows[target].notna().sum()),
                daily_confirmed_hit_fraction_all_registered_dates=float(bydate["sum"].gt(0).mean()) if len(bydate) else None,
                daily_any_hit_rate_complete_dates=float(bydate.loc[complete,"sum"].gt(0).mean()) if complete.any() else None,
                daily_unknown_dates=int((~complete).sum()))


def evaluate(rows,score):
    out = {f"top{k}":{f"reach{h}":metric(rows,score,k,f"reach{h}_LABEL_ONLY") for h in (2,3,4,5)} | {"cohort_leader_proxy":metric(rows,score,k,"cohort_leader_LABEL_ONLY")} for k in (1,3,5)}
    known = rows[TARGET].notna() & rows[score].notna()
    y = rows.loc[known,TARGET].astype(int)
    s = rows.loc[known,score]
    out["pool"] = dict(rows=len(rows),sessions=rows.signal_date.nunique(),known=len(y),positives=int(y.sum()),base_rate=float(y.mean()) if len(y) else None)
    if score=="model_score" and y.nunique()==2:
        p,r,f,_ = precision_recall_fscore_support(y,s.ge(.5),average="binary",zero_division=0)
        out["threshold_0_5_uncalibrated"] = dict(precision=p,recall=r,f1=f,accuracy=float((s.ge(.5).to_numpy()==y.to_numpy()).mean()),roc_auc=roc_auc_score(y,s),average_precision=average_precision_score(y,s))
    return out


def paired_interval(rows,k=3,reps=2000):
    dates = sorted(rows.signal_date.unique())
    a = selected(rows,"model_score",k).groupby("signal_date")[TARGET].agg(["count","sum"]).reindex(dates,fill_value=0)
    b = selected(rows,"baseline_score",k).groupby("signal_date")[TARGET].agg(["count","sum"]).reindex(dates,fill_value=0)
    weekly = pd.DataFrame({"a_hits":a["sum"],"a_n":a["count"],"b_hits":b["sum"],"b_n":b["count"]})
    weekly.index = pd.to_datetime(weekly.index).to_period("W")
    values = weekly.groupby(level=0).sum().to_numpy(dtype=float)
    if not len(values):
        return dict(delta=None,ci95=[None,None])
    rng = np.random.default_rng(42)
    estimates=[]
    n=len(values)
    for _ in range(reps):
        starts=rng.integers(0,n,size=math.ceil(n/5))
        indices=np.concatenate([(x+np.arange(5))%n for x in starts])[:n]
        ah,an,bh,bn=values[indices].sum(axis=0)
        estimates.append(ah/an-bh/bn if an and bn else np.nan)
    return dict(delta=float(a["sum"].sum()/a["count"].sum()-b["sum"].sum()/b["count"].sum()),ci95=np.nanquantile(estimates,[.025,.975]).tolist(),weeks=n,replicates=reps,block_weeks=5)


def train_model(train,family,out,name):
    if train[TARGET].nunique()!=2:
        raise ValueError("training requires both classes")
    fitdir=out/"models";fitdir.mkdir(exist_ok=True)
    receipt=dict(name=name,family=family,started_at=now(),rows=len(train),signals=[train.signal_date.min(),train.signal_date.max()],max_label_end=train.label_end.max(),positives=int(train[TARGET].sum()),features=FEATURES)
    write_json(fitdir/f"{name}_BEFORE_FIT.json",receipt)
    train[["signal_date","code","label_end",TARGET]].to_csv(fitdir/f"{name}_membership.csv",index=False)
    model=make_model(family)
    weights=1/train.groupby("signal_date").code.transform("size")
    weights=weights/weights.mean()
    if family=="logistic":
        model.fit(train[FEATURES],train[TARGET].astype(int),logisticregression__sample_weight=weights)
    else:
        model.fit(train[FEATURES],train[TARGET].astype(int),sample_weight=weights)
    joblib.dump(model,fitdir/f"{name}.joblib")
    receipt.update(completed_at=now(),model_sha256=sha(fitdir/f"{name}.joblib"),train_metric=evaluate(train.assign(model_score=model.predict_proba(train[FEATURES])[:,1]),"model_score"))
    write_json(fitdir/f"{name}_COMPLETED.json",receipt)
    return model,receipt


def simulation(rows,bars,sessions,score):
    """Daily-bar scenario with cash, lots and persistent blocked positions."""
    picks=selected(rows,score,1)
    by_entry={sessions[int(r.p)+1]:r for r in picks.itertuples() if int(r.p)+1<len(sessions)}
    needed=set(picks.code)
    look=bars[bars.code.isin(needed)].set_index(["date","code"]).to_dict("index")
    start=min(by_entry) if by_entry else sessions[-1]
    end=bars.date.max()
    cash=1_000_000.;positions=[];events=[];equity=[]
    last={}
    for p,day in enumerate(sessions):
        if not start<=day<=end:continue
        # Do not spend proceeds from this evening's exit at this morning's open.
        mark=cash+sum(t["shares"]*last.get(t["code"],t["entry_price"]) for t in positions)
        candidate=by_entry.get(day)
        if candidate is not None:
            bar=look.get((day,candidate.code))
            reason=None
            if bar is None or not bar["valid_price"] or not bar["eligible"]:reason="ENTRY_UNKNOWN_OR_INELIGIBLE"
            elif round(bar["open"]*100)>=bar["cap"]:reason="OPEN_AT_UPPER_LIMIT_NO_QUEUE_ASSUMPTION"
            elif round(bar["open"]*100)<=bar["floor"]:reason="OPEN_AT_LOWER_LIMIT_CANCEL"
            if reason:
                events.append(dict(signal_date=candidate.signal_date,code=candidate.code,entry_date=day,status=reason))
            else:
                price=math.ceil(bar["open"]*1.001*100-1e-8)/100
                budget=min(cash/1.001,mark*.20,max(0,float(candidate.amount))*.01 if pd.notna(candidate.amount) else 0)
                shares=int(budget/price/100)*100
                if shares and round(price*100)<bar["cap"]:
                    value=shares*price;fee=max(5,value*.0003)+value*.00001
                    cash-=value+fee
                    positions.append(dict(signal_date=candidate.signal_date,code=candidate.code,entry_date=day,entry_p=p,planned_exit_p=p+2,shares=shares,entry_price=price,entry_cost=value+fee,status="OPEN",blocked_days=0))
                else:events.append(dict(signal_date=candidate.signal_date,code=candidate.code,entry_date=day,status="ENTRY_LOT_OR_SLIPPAGE_LIMIT_CANCEL"))
        for t in list(positions):
            bar=look.get((day,t["code"]))
            if bar and bar["valid_price"]:last[t["code"]]=bar["close"]
            if p<t["planned_exit_p"]:continue
            if bar is None or not bar["valid_price"] or not bar["eligible"] or round(bar["close"]*100)<=bar["floor"] or not (bar["volume"]>0):
                t["blocked_days"]+=1;continue
            price=math.floor(bar["close"]*.999*100+1e-8)/100
            value=t["shares"]*price;fee=max(5,value*.0003)+value*(.00001+.0005)
            cash+=value-fee
            t.update(status="SCENARIO_CLOSED",exit_date=day,exit_price=price,net_pnl=value-fee-t["entry_cost"],net_return=(value-fee)/t["entry_cost"]-1)
            events.append(t.copy());positions.remove(t)
        equity.append(dict(date=day,equity=cash+sum(t["shares"]*last.get(t["code"],t["entry_price"]) for t in positions),cash=cash,open_positions=len(positions)))
    events.extend([{**t,"status":"OPEN_UNRESOLVED","mark":last.get(t["code"],t["entry_price"])} for t in positions])
    trades=pd.DataFrame(events);curve=pd.DataFrame(equity)
    closed=trades[trades.status.eq("SCENARIO_CLOSED")] if len(trades) else trades
    returns=closed.net_return if len(closed) else pd.Series(dtype=float)
    wins=returns[returns>0];losses=returns[returns<=0]
    report=dict(status="OHLC_EXECUTION_SCENARIO_NOT_VERIFIED_FILLS",selected=len(picks),closed_trades=len(closed),open_unresolved=len(positions),skipped=len(trades)-len(closed)-len(positions),
                total_return=curve.equity.iloc[-1]/1e6-1 if len(curve) else None,
                max_drawdown=float((curve.equity/curve.equity.cummax().clip(lower=1_000_000.)-1).min()) if len(curve) else None,
                win_rate=float((returns>0).mean()) if len(returns) else None,average_trade=returns.mean(),average_win=wins.mean(),average_loss=losses.mean(),worst_trade=returns.min(),
                profit_factor=closed.loc[closed.net_pnl>0,"net_pnl"].sum()/-closed.loc[closed.net_pnl<0,"net_pnl"].sum() if len(closed) and closed.net_pnl.lt(0).any() else None,
                blocked_exit_count=(int(closed.blocked_days.gt(0).sum()) if len(closed) else 0)+sum(t["blocked_days"]>0 for t in positions),
                actual_trade_count=0,annual_return_claimed=False)
    return trades,curve,report


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError("use a new output directory; never overwrite an experiment")
    out.mkdir(parents=True)
    pre=protocol();write_json(out/"protocol_BEFORE_ANY_FIT.json",pre)
    dataset,bars,sessions,audit=build_dataset()
    dataset.to_csv(out/"all_first_boards_features_and_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    write_json(out/"data_audit.json",audit)
    known=dataset[dataset[TARGET].notna()].copy()
    development=known[(known.signal_date<"2025-01-01") & (known.label_end<"2025-01-01")]
    validation=dataset[dataset.signal_date.between("2025-01-01","2025-06-30")].copy()
    choices={}
    for family in ("logistic","hist_gradient"):
        model,receipt=train_model(development,family,out,f"validation_{family}")
        val=validation.assign(model_score=model.predict_proba(validation[FEATURES])[:,1])
        val.to_csv(out/f"validation_{family}_ALL_PREDICTIONS.csv",index=False)
        choices[family]=evaluate(val,"model_score")
    winner=max(choices,key=lambda f:choices[f]["top3"]["reach4"]["hits"])
    write_json(out/"family_selection_BEFORE_EVALUATION.json",dict(recorded_at=now(),family=winner,selection="validation Top3 reach4 hits only, logistic wins ties",validation=choices))
    evaluation=dataset[dataset.signal_date.ge("2025-07-01")].copy()
    evaluation["baseline_score"]=-evaluation.turnover
    pieces=[];fits=[]
    for month in sorted(evaluation.signal_date.str[:7].unique()):
        boundary=f"{month}-01";boundary_pos=int(np.searchsorted(sessions,boundary))
        train=known[(known.label_end<boundary) & (known.p>=boundary_pos-252)]
        score=evaluation[evaluation.signal_date.str.startswith(month)].copy()
        model,receipt=train_model(train,winner,out,month)
        if not (receipt["max_label_end"]<score.signal_date.min()):raise AssertionError("label maturity leak")
        score["model_score"]=model.predict_proba(score[FEATURES])[:,1]
        score["model_month"]=month
        score.to_csv(out/f"predictions_{month}_BEFORE_METRICS.csv",index=False)
        fits.append(receipt);pieces.append(score)
    predictions=pd.concat(pieces,ignore_index=True)
    predictions.to_csv(out/"evaluation_ALL_PREDICTIONS.csv",index=False,encoding="utf-8-sig")
    comparisons={}
    for period,mask in [("all",pd.Series(True,index=predictions.index)),("2025H2",predictions.signal_date.lt("2026-01-01")),("2026",predictions.signal_date.ge("2026-01-01")),("non_one_price",predictions.one_price.eq(0))]:
        part=predictions[mask]
        comparisons[period]=dict(model=evaluate(part,"model_score"),baseline=evaluate(part,"baseline_score"),top3_reach4_paired=paired_interval(part))
    acceptance=bool(comparisons["all"]["top3_reach4_paired"]["ci95"][0]>0 and all(comparisons[p]["top3_reach4_paired"]["delta"]>0 for p in ("2025H2","2026")) and comparisons["non_one_price"]["top3_reach4_paired"]["delta"]>=0)
    sim={}
    for score in ("model_score","baseline_score"):
        trades,curve,info=simulation(predictions,bars,sessions,score)
        trades.to_csv(out/f"{score}_SCENARIO_TRADES.csv",index=False,encoding="utf-8-sig")
        curve.to_csv(out/f"{score}_SCENARIO_EQUITY.csv",index=False)
        sim[score]=info
    cutoff=bars.date.max()
    pos=int(np.searchsorted(sessions,cutoff))
    final_train=known[(known.label_end<cutoff)&(known.p>=pos-252)]
    final,receipt=train_model(final_train,winner,out,"latest_research")
    bundle=dict(version=VERSION,model=final,features=FEATURES,target=TARGET,family=winner,knowledge_cutoff=cutoff,train_max_label_end=receipt["max_label_end"],protocol_sha256=sha(out/"protocol_BEFORE_ANY_FIT.json"),calibrated=False,research_only=True,data_audit=audit)
    joblib.dump(bundle,out/"first_board_leader_bundle.joblib")
    last=dataset[dataset.signal_date.eq(cutoff)].copy()
    if len(last):
        last["model_score"]=final.predict_proba(last[FEATURES])[:,1]
        last=last.sort_values(["model_score","code"],ascending=[False,True])
        last.to_csv(out/"last_archived_session_RESEARCH_RANKING.csv",index=False,encoding="utf-8-sig")
    monthly=[]
    for month,part in predictions.groupby("model_month"):
        m=metric(part,"model_score",3);b=metric(part,"baseline_score",3)
        monthly.append(dict(month=month,model_hits=m["hits"],baseline_hits=b["hits"],known=m["known"],unknown=m["unknown"],positive_pool=m["positive_pool"],precision=m["precision"],baseline_precision=b["precision"]))
    pd.DataFrame(monthly).to_csv(out/"monthly_top3_reach4.csv",index=False)
    failures=selected(predictions,"model_score",3)
    failures[failures[TARGET].ne(1)].to_csv(out/"top3_FALSE_POSITIVES_AND_UNKNOWNS.csv",index=False,encoding="utf-8-sig")
    picked=set(zip(failures.signal_date,failures.code))
    missed=predictions[predictions[TARGET].eq(1) & ~predictions.apply(lambda r:(r.signal_date,r.code) in picked,axis=1)]
    missed.to_csv(out/"all_missed_four_board_paths.csv",index=False,encoding="utf-8-sig")
    original_hashes=pre["source_hashes"]
    if any(sha(p)!=h for p,h in original_hashes.items()):raise ValueError("source changed during experiment")
    report=dict(version=VERSION,completed_at=now(),family=winner,data=audit,evaluation=comparisons,scenario=sim,monthly_fits=len(fits),total_fits=len(fits)+3,
                historical_rank_gate_passed=acceptance,production_deployment_allowed=False,probability_status="UNCALIBRATED_SCORE_NOT_EVENT_PROBABILITY",actual_trades=0,new_forward_results=0,
                source_hashes=original_hashes,code_sha256=sha(__file__),bundle_sha256=sha(out/"first_board_leader_bundle.joblib"))
    write_json(out/"report.json",report)
    write_report(out,report)
    print(json.dumps(clean(dict(output=str(out.resolve()),family=winner,rank_gate=acceptance,model_top3=comparisons["all"]["model"]["top3"],baseline_top3=comparisons["all"]["baseline"]["top3"],interval=comparisons["all"]["top3_reach4_paired"],scenario=sim)),ensure_ascii=False,indent=2))
    return report


def write_report(out,report):
    def pct(x):return "未知" if x is None else f"{x*100:.2f}%"
    r=report;d=r["data"];all_=r["evaluation"]["all"]
    lines=["# 首板连板潜力模型训练与回测", "",f"实际完成：{r['completed_at']}。模型：{r['family']}。首板信号范围：{d['first_date']}—{d['last_date']}。", "",
           "目标是在首板收盘后排序，主目标为随后三个相邻官方交易日均精确收盘涨停，即连续达到四板。三板、五板单列；同期首板中未来五日连板高度并列最高且至少四板，只称高度龙头代理，不能认证真实题材龙头。", "",
           f"重建首板 {d['first_board_rows']} 条/{d['first_board_sessions']} 日，统一五日路径可观察成熟 {d['mature_observed_rows']} 条；缺失或右删失 {d['unknown_or_right_censored_rows']} 条。", "",
           "模型只读取首板收盘已可计算的44项价量、历史动量、同日竞争及当日已实现的昨日反馈。未来标签不进特征。2025上半年固定验证只用于两种预先固定算法择一；2025年7月起逐月以过去252个官方交易日训练，训练标签终点严格早于该月。历史曾被其他研究看过，属于时间样本外回放，非全新盲测。", "",
           "|连续目标|每日名额|模型命中/已知|模型命中率|基准命中率|模型完整正例召回|", "|---|---:|---:|---:|---:|---:|"]
    for h in (3,4,5):
        for k in (1,3,5):
            a=all_["model"][f"top{k}"][f"reach{h}"];b=all_["baseline"][f"top{k}"][f"reach{h}"]
            lines.append(f"|{h}板|{k}|{a['hits']}/{a['known']}|{pct(a['precision'])}|{pct(b['precision'])}|{pct(a['recall'])}|")
    ci=all_["top3_reach4_paired"]
    lines += ["",f"主指标Top3四板命中率差：{pct(ci['delta']).replace('%','')}个百分点；配对周块95%区间 [{pct(ci['ci95'][0]).replace('%','')}, {pct(ci['ci95'][1]).replace('%','')}] 个百分点。预定历史排序门：{'通过' if r['historical_rank_gate_passed'] else '未通过'}。", "",
              "基准固定为当日首板中低换手优先，不按结果换基准。完整候选、未知、误报、漏报、逐月训练成员和预测均保留。分数未经概率校准，不可写成未来晋级百分比。", "", "## 分段与非一字复核", "", "|范围|Top3四板模型|基准|差值（百分点）|", "|---|---:|---:|---:|"]
    for period in ("2025H2","2026","non_one_price"):
        x=r["evaluation"][period];lines.append(f"|{period}|{pct(x['model']['top3']['reach4']['precision'])}|{pct(x['baseline']['top3']['reach4']['precision'])}|{pct(x['top3_reach4_paired']['delta']).replace('%','')}|")
    lines += ["", "## 交易情景", "", "下一交易日开盘价格代理，涨停/跌停开盘取消、无替补；每笔最多20%净值且不超过信号日成交额1%，100股整手。B+2收盘计划退出；跌停或缺行情继续保留持仓。佣金双边万三且单边最低5元、过户费双边十万一、卖出印花税万五，双边各千一不利滑点。日线不能证明订单成交，因此以下为扣费情景，并非实盘或经确认成交收益。", "", "|方案|情景平仓笔数|期末未解决|组合回报|最大回撤|平均每笔|胜率|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,s in r["scenario"].items():
        lines.append(f"|{name}|{s['closed_trades']}|{s['open_unresolved']}|{pct(s['total_return'])}|{pct(s['max_drawdown'])}|{pct(s['average_trade'])}|{pct(s['win_rate'])}|")
    lines += ["", "## 使用和边界", "", "训练入口：`python first_board_leader_model.py train --output <全新目录>`。显式评分入口：`python first_board_leader_model.py score --bundle <本批bundle> --features <完整首板特征CSV> --output <新CSV>`。评分输入必须含全部44字段及signal_date/code，仅接受训练知识截止之后的日期；缺字段拒绝，不自动替换原模型。", "",
              "日线来自项目已存Sina/Tencent等不复权归档，历史证券状态沿项目逐日期核验字段；源取得晚于历史信号。现有股票范围和历史母池仍有幸存者/缺失风险，未核历史真实题材、催化、首封时刻或逐笔排队。listing_age_sessions只计算官方归档区间内的上市年龄，归档前已上市者在归档起点左删失，不能解释为完整上市年限。本次训练权重已保存，但不构成实际部署验收，不改原生产模型、冻结账本或定时任务。新增真实前向结果0、实际成交0。", "",
              "[上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml)要求交收前不得卖出，普通主板股票不适用当日回转；[财政部税务总局公告](https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html)自2023年8月28日起减半征收证券交易印花税。手续费和滑点是事前固定模拟参数，须以真实账户核对。"]
    (out/"首板连板潜力模型_训练回测报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def score_file(bundle_path,features_path,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    bundle=joblib.load(bundle_path)
    rows=pd.read_csv(features_path,dtype={"code":str},float_precision="round_trip")
    required=set(bundle["features"]+["signal_date","code"])
    if required-set(rows):raise ValueError(f"missing features: {sorted(required-set(rows))}")
    if rows.duplicated(["signal_date","code"]).any():raise ValueError("duplicate candidate keys")
    parsed=pd.to_datetime(rows.signal_date,format="%Y-%m-%d",errors="coerce")
    if parsed.isna().any() or not rows.code.str.fullmatch(r"\d{6}").all():raise ValueError("invalid signal dates or codes")
    if rows.empty:raise ValueError("empty first-board candidate pool")
    for field in ("first","eligible","scope_known"):
        if field not in rows or not true_values(rows[field]).all():raise ValueError(f"verified first-board scope required: {field}")
    if not rows.signal_date.gt(bundle["knowledge_cutoff"]).all():raise ValueError("only new signals after training cutoff can be scored by this command")
    if any("LABEL" in c or "future" in c.lower() for c in bundle["features"]):raise ValueError("forbidden feature")
    rows["research_score"]=bundle["model"].predict_proba(rows[bundle["features"]])[:,1]
    rows["probability_status"]="UNCALIBRATED"
    rows=rows[[c for c in rows if "LABEL_ONLY" not in c and c not in ("label_end","label_observed")]]
    rows.sort_values(["signal_date","research_score","code"],ascending=[True,False,True]).to_csv(out,index=False,encoding="utf-8-sig")


def rank_bars(bundle_path,inputs,signal_date,output):
    """Rebuild causal features from explicit raw inputs, then score one new date."""
    bundle=joblib.load(bundle_path)
    if signal_date<=bundle["knowledge_cutoff"]:raise ValueError("new date must be after bundle training cutoff")
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    rows,_,sessions,audit=build_dataset(inputs,feature_cutoff=signal_date)
    if signal_date not in sessions:raise ValueError("signal date outside official calendar")
    current=rows[rows.signal_date.eq(signal_date)].copy()
    if current.empty:raise ValueError("no verified first-board candidates at requested close")
    current["research_score"]=bundle["model"].predict_proba(current[bundle["features"]])[:,1]
    current["research_rank"]=current.research_score.rank(method="first",ascending=False).astype(int)
    current["probability_status"]="UNCALIBRATED"
    current=current[[c for c in current if "LABEL_ONLY" not in c and c not in ("label_end","label_observed")]]
    current.sort_values(["research_score","code"],ascending=[False,True]).to_csv(out,index=False,encoding="utf-8-sig")
    write_json(out.with_suffix(".receipt.json"),dict(created_at=now(),bundle_sha256=sha(bundle_path),source_hashes={k:sha(p) for k,p in inputs.items()},signal_date=signal_date,candidates=len(current),source_scope=audit,production_allowed=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train");t.add_argument("--output",required=True)
    s=sub.add_parser("score");s.add_argument("--bundle",required=True);s.add_argument("--features",required=True);s.add_argument("--output",required=True)
    rank=sub.add_parser("rank");rank.add_argument("--bundle",required=True);rank.add_argument("--bars",required=True);rank.add_argument("--recovered",required=True);rank.add_argument("--calendar",required=True);rank.add_argument("--security-intervals",required=True);rank.add_argument("--signal-date",required=True);rank.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="train":run(args.output)
    elif args.command=="score":score_file(args.bundle,args.features,args.output)
    else:rank_bars(args.bundle,dict(bars=Path(args.bars),recovered=Path(args.recovered),calendar=Path(args.calendar),security_intervals=Path(args.security_intervals)),args.signal_date,args.output)
