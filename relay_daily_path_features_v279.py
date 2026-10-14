"""Causal daily-bar identity clues; never verified theme or intraday absorption."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from next_session_forecast_lab_v271 import ROOT, DATA, sha
from historical_ohlc_archive_v278 import verify_archive

VERSION = "v279-causal-daily-path-clues-1"
BAR_FIELDS = ["date","code","open","high","low","close","pre_close","volume","amount","turnover"]
NEW_FIELDS = ["daily_open_gap","daily_low_gap","daily_amplitude","daily_true_one_word",
              "daily_turnover_previous_session_ratio","daily_amount_previous_session_ratio",
              "daily_volume_previous_session_ratio","previous_session_true_one_word",
              "daily_first_open_after_one_word","daily_amount_previous3_ratio",
              "first_board_turnover","first_board_open_gap","first_board_amplitude",
              "first_board_true_one_word","first_board_amount_to_signal_ratio"]


def bar_features(bars, calendar):
    if bars.duplicated(["date","code"]).any():
        raise ValueError("duplicate stock/date bars")
    f = bars.sort_values(["code","date"]).copy()
    f["session_index"] = f.date.map({d:i for i,d in enumerate(calendar)})
    if f.session_index.isna().any():
        raise ValueError("bar outside supplied observed-session calendar")
    complete = f[["open","high","low","close","pre_close"]].notna().all(axis=1)
    valid = (complete & f.pre_close.gt(0) & f.low.gt(0) & f.high.ge(f.low)
             & f.open.between(f.low,f.high) & f.close.between(f.low,f.high))
    f["daily_open_gap"] = (f.open/f.pre_close-1).where(valid)
    f["daily_low_gap"] = (f.low/f.pre_close-1).where(valid)
    f["daily_amplitude"] = ((f.high-f.low)/f.pre_close).where(valid)
    f["daily_true_one_word"] = np.where(valid,np.isclose(f.high,f.low,atol=.001,rtol=0).astype(float),np.nan)
    g = f.groupby("code",sort=False)
    contiguous = f.session_index.sub(g.session_index.shift(1)).eq(1)
    for field in ["turnover","amount","volume"]:
        previous = g[field].shift(1)
        name = f"daily_{field}_previous_session_ratio"
        f[name] = (f[field]/previous).where(contiguous & previous.gt(0))
    f["previous_session_true_one_word"] = g.daily_true_one_word.shift(1).where(contiguous)
    f["daily_first_open_after_one_word"] = np.where(
        f.previous_session_true_one_word.notna() & f.daily_true_one_word.notna(),
        (f.previous_session_true_one_word.eq(1)&f.daily_true_one_word.eq(0)).astype(float),np.nan)
    previous3 = g.amount.transform(lambda x:x.shift(1).rolling(3,min_periods=3).mean())
    contiguous3 = f.session_index.sub(g.session_index.shift(3)).eq(3)
    f["daily_amount_previous3_ratio"] = (f.amount/previous3).where(contiguous3 & previous3.gt(0))
    f["daily_path_status"] = np.where(valid,"OBSERVED_DAILY_PROXY_NOT_INTRADAY_EVIDENCE","UNAVAILABLE_OR_INVALID")
    return f


def attach(candidates, bars, calendar):
    features = bar_features(bars,calendar)
    daily_fields = [k for k in NEW_FIELDS if not k.startswith("first_board_")]
    f = candidates.merge(features[["date","code","close","session_index","daily_path_status",*daily_fields]],
                         on=["date","code"],how="left",validate="many_to_one")
    match = (f.price-f.close).abs().le(.011)
    f.loc[~match,daily_fields] = np.nan
    f.loc[~match,"daily_path_status"] = "UNAVAILABLE_OR_CLOSE_DISAGREES"
    # Episode beginning = current observed session minus boards-1. Require every
    # daily bar through signal to be present; no jump over suspensions/gaps.
    f["first_board_index"] = f.session_index-f.boards+1
    first = features[["code","session_index","daily_open_gap","daily_amplitude","daily_true_one_word","turnover","amount"]].rename(
        columns={"session_index":"first_board_index","daily_open_gap":"first_board_open_gap",
                 "daily_amplitude":"first_board_amplitude","daily_true_one_word":"first_board_true_one_word",
                 "turnover":"first_board_turnover","amount":"first_board_amount"})
    f = f.merge(first,on=["code","first_board_index"],how="left",validate="many_to_one")
    f["first_board_turnover"] /= 100
    f["first_board_amount_to_signal_ratio"] = (f.first_board_amount/f.signal_amount).where(f.signal_amount.gt(0))
    # Row rank counts actual observations while session index counts all market
    # sessions. Equal distance guarantees no missing bar inside the episode.
    positions = features[["code","session_index"]].copy()
    positions["bar_position"] = positions.groupby("code").cumcount()
    f = f.merge(positions,on=["code","session_index"],how="left",validate="many_to_one")
    f = f.merge(positions.rename(columns={"session_index":"first_board_index","bar_position":"first_position"}),
                on=["code","first_board_index"],how="left",validate="many_to_one")
    intact = f.bar_position.sub(f.first_position).eq(f.boards-1) & match.to_numpy()
    f.loc[~intact,[k for k in NEW_FIELDS if k.startswith("first_board_")]] = np.nan
    f["episode_daily_complete"] = intact
    f["feature_semantics"] = "DAILY_IDENTITY_CLUES_NOT_THEME_OR_ABSORPTION_VERIFICATION"
    return f.drop(columns=["close","first_board_amount","bar_position","first_position"])


def read_history(codes):
    chunks=[]; sources=[]
    for name in ["point_in_time_current_daily_bars.csv","verified_recovered_daily_bars.csv"]:
        path=DATA/name;sources.append(path)
        for part in pd.read_csv(path,usecols=BAR_FIELDS,dtype={"code":str},chunksize=150000):
            part["code"]=part.code.str.zfill(6)
            keep=part.code.isin(codes)&part.date.le("2026-08-31")
            if keep.any():chunks.append(part[keep].copy())
    return pd.concat(chunks,ignore_index=True),sources


def run(output):
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    historical=pd.read_csv(DATA/"leader_candidates.csv",dtype={"code":str},low_memory=False)
    historical=historical[historical.boards.ge(2)].copy()
    bars,sources=read_history(set(historical.code))
    calendar=pd.read_csv(DATA/"market_emotion.csv").date.tolist()
    hp=attach(historical,bars,calendar)
    hp.to_csv(target/"historical_relay_paths_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    september=ROOT/"results/v278/september_falsification_20261002_v6/relay_predictions_LABEL_ONLY.csv"
    current=pd.read_csv(september,dtype={"code":str})
    daily_dir=ROOT/"data/v278/september_unadjusted_20261002_v3"
    daily=verify_archive(daily_dir)
    daily=daily[~daily.symbol.isin(["sh000002","sz399107"])].copy()
    daily["code"]=daily.symbol.str[2:]
    daily["pre_close"]=daily.sort_values(["symbol","date"]).groupby("symbol").close.shift(1)
    # Current bars have no daily turnover/amount series. Signal-date pool values
    # alone are joined; missing earlier amounts/turnover stay unavailable.
    reported=ROOT/"data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv"
    pools=pd.read_csv(reported,dtype={"code":str})
    daily=daily.merge(pools[["date","code","turnover","amount"]],on=["date","code"],how="left",validate="one_to_one")
    from reported_seal_archive_v275 import sessions
    current_calendar=sorted(set(calendar+[d for d in sessions() if d>"2026-08-31"]))
    sp=attach(current,daily[BAR_FIELDS],current_calendar)
    sp.to_csv(target/"september_relay_paths_LABEL_ONLY.csv",index=False,encoding="utf-8-sig")
    report={"version":VERSION,"status":"DIAGNOSTIC_ONLY","fields":NEW_FIELDS,
        "historical_rows":len(hp),"historical_daily_rows":len(bars),"september_rows":len(sp),
        "coverage":{name:{k:float(f[k].notna().mean()) for k in NEW_FIELDS} for name,f in [("historical",hp),("september",sp)]},
        "source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in sources+[DATA/"leader_candidates.csv",DATA/"market_emotion.csv",
            september,reported,daily_dir/"unadjusted_daily.csv",daily_dir/"report.json",Path(__file__)]},
        "limitations":["historic universe/security status incomplete; reused historical labels",
            "historical observed-session calendar is not independently certified",
            "daily range/amount cannot establish intraday order, absorption or real reseal",
            "September only reported limit pools supply turnover/amount; prior non-limit days remain missing",
            "First-board identity is an inferred contiguous episode; no true catalyst or sector verification"],
        "goal_verified":False,"trade_authorized":False}
    (target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return {k:report[k] for k in ["historical_rows","historical_daily_rows","september_rows","coverage"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();print(json.dumps(run(args.output),ensure_ascii=True))
