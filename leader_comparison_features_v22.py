"""Explicit after-close candidate features; no labels or fabricated odds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(observations, bars, full_limit_pool=False):
    keys = ["code", "signal_date"]
    frame = observations[[*keys, "episode_id", "boards", "start_known"]].copy()
    if frame.duplicated(keys).any():
        raise ValueError("duplicate candidate")
    for key in ("signal_date",):
        frame[key] = pd.to_datetime(frame[key]).dt.strftime("%Y-%m-%d")
    frame["boards"] = pd.to_numeric(frame.boards, errors="raise")
    known = frame.start_known.map(lambda x: str(x).lower() in {"true", "1"})
    valid_board = frame.boards.ge(1 if full_limit_pool else 2) & frame.boards.mod(1).eq(0)
    if full_limit_pool:
        valid_board = valid_board | (~known & frame.boards.isna())
    if not valid_board.all():
        raise ValueError("invalid board counts")
    if full_limit_pool:
        frame["board_group"] = np.where(~known, "UNKNOWN_LEFT_CENSORED",
                                          np.where(frame.boards.eq(1), "FIRST_BOARD", "CONSECUTIVE_BOARD"))
    # Only declared fields may enter features, regardless of caller extras.
    columns = ["code", "date", "open", "high", "low", "close", "pre_close", "amount", "float_cap", "turnover"]
    prices = bars[columns].rename(columns={"date": "signal_date"}).copy()
    prices["signal_date"] = pd.to_datetime(prices.signal_date).dt.strftime("%Y-%m-%d")
    for col in columns[2:]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.merge(prices, on=keys, how="left", validate="one_to_one")
    known = frame.start_known.map(lambda x: str(x).lower() in {"true", "1"})
    frame["boards_for_comparison"] = frame.boards.where(known)
    daily = frame.groupby("signal_date")
    frame["known_pool_max_boards"] = daily.boards_for_comparison.transform("max")
    frame["candidate_pool_count"] = daily.code.transform("size")
    frame["unknown_height_pool_count"] = daily.boards_for_comparison.transform(lambda s: s.isna().sum())
    frame["height_gap_to_known_top"] = frame.known_pool_max_boards-frame.boards_for_comparison
    frame["same_height_known_count"] = frame.groupby(["signal_date", "boards_for_comparison"]).code.transform("size")
    frame["sole_known_top_proxy"] = (frame.height_gap_to_known_top.eq(0) & frame.same_height_known_count.eq(1)).astype(float).where(known)
    frame["open_gap_pct"] = (frame.open/frame.pre_close.where(frame.pre_close.gt(0))-1)*100
    frame["amplitude_pct"] = (frame.high-frame.low)/frame.pre_close.where(frame.pre_close.gt(0))*100
    frame["amount_to_float_cap"] = frame.amount/frame.float_cap.where(frame.float_cap.gt(0))
    ohlc = frame[["open", "high", "low", "close"]]
    complete = ohlc.notna().all(axis=1) & ohlc.gt(0).all(axis=1)
    frame["one_price_day_proxy"] = (ohlc.max(axis=1)-ohlc.min(axis=1)).abs().lt(1e-6).astype(float).where(complete)
    frame["feature_available_not_before"] = frame.signal_date + "T15:05:00+08:00"
    frame["feature_status"] = "historical_reconstruction_not_forward_archive"
    frame["pool_scope"] = "available_nonst_mainboard_history_not_complete_market"
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-limit-pool", action="store_true", help="include first boards and missing left-censored heights")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite feature archive")
    observations = pd.read_csv(args.observations, dtype={"code": str})
    bars = pd.read_csv(args.bars, dtype={"code": str}, usecols=["code", "date", "open", "high", "low", "close", "pre_close", "amount", "float_cap", "turnover"])
    # Restrict price join to candidate keys, while retaining strict duplicate checks.
    bars = bars.merge(observations[["code", "signal_date"]].rename(columns={"signal_date": "date"}), on=["code", "date"], validate="one_to_one")
    result = build(observations, bars, full_limit_pool=args.full_limit_pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(json.dumps(dict(rows=len(result), dates=result.signal_date.nunique(),
                         known_top_rows=int(result.height_gap_to_known_top.eq(0).sum()),
                         one_price_rows=int(result.one_price_day_proxy.eq(1).sum()),
                         probability_model_trained=False), indent=2))
