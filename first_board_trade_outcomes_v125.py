"""Build label-only, execution-aware outcomes for first-board candidates.

This module never feeds future prices back into the signal-date feature table.
It treats the next session as buy day B and uses a deliberately simple auction
order proxy: the B open is considered observable/executable only when it is
below the rounded upper-limit price.  Any upper-limit open is marked queue-
unverifiable and is not counted as a fill, even if it later opens intraday.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limit_price(pre_close: float, rate: str = "0.10") -> float:
    value = Decimal(str(pre_close)) * (Decimal("1") + Decimal(rate))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def load_relevant_bars(path: Path, codes: set[str], dates: set[str]) -> pd.DataFrame:
    usecols = ["date", "code", "open", "high", "low", "close", "pre_close",
               "st", "new_listing", "status_verified", "universe_complete"]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"code": str}, chunksize=250_000):
        chunk["code"] = chunk.code.astype(str).str.zfill(6)
        keep = chunk.code.isin(codes) & chunk.date.astype(str).isin(dates)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    if not parts:
        return pd.DataFrame(columns=usecols)
    result = pd.concat(parts, ignore_index=True)
    result["date"] = result.date.astype(str)
    return result.drop_duplicates(["date", "code"], keep="last")


def net_return(entry_open: float, exit_close: float, *, slippage: float = .001,
               commission: float = .00025, transfer_fee: float = .00001,
               stamp_tax: float = .0005) -> float:
    entry_cash = entry_open * (1 + slippage) * (1 + commission + transfer_fee)
    exit_cash = exit_close * (1 - slippage) * (1 - commission - transfer_fee - stamp_tax)
    return exit_cash / entry_cash - 1


def build_outcomes(observations: pd.DataFrame, bars: pd.DataFrame,
                   calendar: pd.DataFrame) -> pd.DataFrame:
    sessions = (pd.to_datetime(calendar.date, errors="coerce").dropna()
                .dt.strftime("%Y-%m-%d").drop_duplicates().tolist())
    pos = {date: index for index, date in enumerate(sessions)}
    bar_map = bars.set_index(["date", "code"], drop=False)
    records = []
    for row in observations.itertuples(index=False):
        signal_date, code = str(row.signal_date), str(row.code).zfill(6)
        base = pos.get(signal_date)
        schedule = {f"B_plus_{offset - 1}": sessions[base + offset]
                    for offset in range(1, 5)
                    if base is not None and base + offset < len(sessions)}
        buy_date = schedule.get("B_plus_0")
        record = {
            "signal_date": signal_date, "code": code,
            "name": getattr(row, "name", ""), "promotion_label": row.promotion_label,
            "buy_date_B": buy_date, "entry_rule": "B_OPEN_AUCTION_PROXY",
            "fill_status": "MISSING_B_BAR", "entry_price_gross": np.nan,
            "entry_price_with_slippage": np.nan, "B_open_gap": np.nan,
            "B_open_upper_limit": np.nan, "B_one_price_upper_limit": np.nan,
        }
        if not buy_date or (buy_date, code) not in bar_map.index:
            records.append(record)
            continue
        buy = bar_map.loc[(buy_date, code)]
        if isinstance(buy, pd.DataFrame):
            buy = buy.iloc[-1]
        prices = pd.to_numeric(buy[["open", "high", "low", "close", "pre_close"]],
                               errors="coerce")
        if prices.isna().any() or prices.pre_close <= 0:
            record["fill_status"] = "INVALID_B_PRICE"
            records.append(record)
            continue
        upper = limit_price(float(prices.pre_close))
        open_at_limit = abs(float(prices.open) - upper) < .005
        one_price = open_at_limit and all(abs(float(prices[key]) - upper) < .005
                                          for key in ("high", "low", "close"))
        record.update({
            "B_open_gap": float(prices.open / prices.pre_close - 1),
            "B_upper_limit_price": upper,
            "B_open_upper_limit": int(open_at_limit),
            "B_one_price_upper_limit": int(one_price),
        })
        if open_at_limit:
            record["fill_status"] = "UNVERIFIABLE_LIMIT_QUEUE"
            records.append(record)
            continue
        entry = float(prices.open)
        record.update({
            "fill_status": "FILLED_OPEN_PROXY",
            "entry_price_gross": entry,
            "entry_price_with_slippage": entry * 1.001,
        })
        for horizon in (1, 2, 3):
            exit_date = schedule.get(f"B_plus_{horizon}")
            record[f"exit_date_B_plus_{horizon}"] = exit_date
            close_value = np.nan
            if exit_date and (exit_date, code) in bar_map.index:
                exit_row = bar_map.loc[(exit_date, code)]
                if isinstance(exit_row, pd.DataFrame):
                    exit_row = exit_row.iloc[-1]
                close_value = pd.to_numeric(pd.Series([exit_row["close"]]), errors="coerce").iloc[0]
            record[f"close_B_plus_{horizon}"] = close_value
            record[f"net_return_B_plus_{horizon}"] = (
                net_return(entry, float(close_value)) if pd.notna(close_value) else np.nan)
        b1 = record.get("net_return_B_plus_1", np.nan)
        b2 = record.get("net_return_B_plus_2", np.nan)
        record["big_loss_B_plus_1"] = int(b1 <= -.07) if pd.notna(b1) else np.nan
        record["positive_B_plus_1"] = int(b1 > 0) if pd.notna(b1) else np.nan
        record["positive_B_plus_1_and_2"] = (
            int(b1 > 0 and b2 > 0) if pd.notna(b1) and pd.notna(b2) else np.nan)
        records.append(record)
    return pd.DataFrame(records)


def summarize(outcomes: pd.DataFrame) -> dict:
    filled = outcomes[outcomes.fill_status.eq("FILLED_OPEN_PROXY")].copy()
    summary = {
        "observations": int(len(outcomes)),
        "filled_open_proxy": int(len(filled)),
        "unverifiable_limit_queue": int(outcomes.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE").sum()),
        "missing_or_invalid": int(outcomes.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"]).sum()),
    }
    for horizon in (1, 2, 3):
        values = filled[f"net_return_B_plus_{horizon}"].dropna()
        summary[f"B_plus_{horizon}"] = {
            "n": int(len(values)), "mean_net_return": float(values.mean()) if len(values) else None,
            "median_net_return": float(values.median()) if len(values) else None,
            "positive_rate": float(values.gt(0).mean()) if len(values) else None,
        }
    b1 = filled.big_loss_B_plus_1.dropna()
    sustained = filled.positive_B_plus_1_and_2.dropna()
    summary["big_loss_B_plus_1_rate"] = float(b1.mean()) if len(b1) else None
    summary["positive_B_plus_1_and_2_rate"] = float(sustained.mean()) if len(sustained) else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--bars", type=Path, default=Path("data/v3/imported/daily-bars.csv"))
    parser.add_argument("--calendar", type=Path, default=Path("data/v3/market_emotion.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v125 output")
    observations = pd.read_csv(args.observations, dtype={"code": str})
    calendar = pd.read_csv(args.calendar, usecols=["date"], dtype={"date": str})
    sessions = pd.to_datetime(calendar.date, errors="coerce").dropna().dt.strftime("%Y-%m-%d").tolist()
    positions = {date: index for index, date in enumerate(sessions)}
    needed_dates = set()
    for date in observations.signal_date.astype(str).unique():
        index = positions.get(date)
        if index is not None:
            needed_dates.update(sessions[index + 1:index + 5])
    bars = load_relevant_bars(args.bars, set(observations.code.astype(str).str.zfill(6)), needed_dates)
    outcomes = build_outcomes(observations, bars, calendar)
    args.output.mkdir(parents=True, exist_ok=False)
    outcomes.to_csv(args.output / "first_board_trade_outcomes_LABEL_ONLY.csv",
                    index=False, encoding="utf-8-sig")
    report = {
        "version": "v125-first-board-execution-outcomes-1",
        "status": "DIAGNOSTIC_ONLY",
        "summary": summarize(outcomes),
        "fee_assumptions": {"buy_slippage": .001, "sell_slippage": .001,
                            "commission_each_side": .00025,
                            "transfer_fee_each_side": .00001, "sell_stamp_tax": .0005},
        "rules": {
            "B": "immediate next trading session after signal date D",
            "upper_limit_open": "unverifiable queue; never counted as filled",
            "other_open": "filled at official open plus 10bp buy slippage proxy",
            "exit": "B+1/B+2/B+3 official close less 10bp sell slippage and fees",
        },
        "limitations": [
            "daily bars cannot verify auction order submission, partial fill, or 09:25-10:00 confirmation",
            "this is an opening-entry baseline, not the final intraday execution strategy",
            "missing later bars remain missing and are never imputed as wins",
        ],
        "source_hashes": {str(path.resolve()): sha256(path) for path in
                          (args.observations, args.bars, args.calendar)},
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
