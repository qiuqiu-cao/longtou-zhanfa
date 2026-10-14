"""Audit T+1 unexitable loss and earliest next-session exit risk.

The candidate is assumed to have been filled by the existing B-open proxy.
Mainboard A shares bought on B cannot be sold on B.  Therefore an intraday
drawdown threshold on B is a risk observation, not an executable stop.  The
first daily-bar exit proxy is B+1 open.  If B+1 opens at the rounded lower
limit, queue position is unavailable: a one-price lower-limit day is marked
unsellable and an opening lower limit that later trades away is marked queue-
unverifiable rather than filled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]
THRESHOLDS = (0.04, 0.05, 0.07)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded_limit(pre_close: float, direction: int) -> float:
    multiplier = Decimal("1.10") if direction > 0 else Decimal("0.90")
    return float((Decimal(str(pre_close)) * multiplier).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def net_return(entry_open: float, exit_open: float, *, slippage: float = .001,
               commission: float = .00025, transfer_fee: float = .00001,
               stamp_tax: float = .0005) -> float:
    entry_cash = entry_open * (1 + slippage) * (1 + commission + transfer_fee)
    exit_cash = exit_open * (1 - slippage) * (
        1 - commission - transfer_fee - stamp_tax)
    return exit_cash / entry_cash - 1


def load_relevant_bars(path: Path, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    usecols = ["date", "code", "open", "high", "low", "close", "pre_close"]
    dates = {date for date, _ in pairs}
    codes = {code for _, code in pairs}
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"code": str},
                             chunksize=250_000):
        chunk["code"] = chunk.code.astype(str).str.zfill(6)
        keep = chunk.date.astype(str).isin(dates) & chunk.code.isin(codes)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    if not parts:
        return pd.DataFrame(columns=usecols)
    result = pd.concat(parts, ignore_index=True)
    result["date"] = result.date.astype(str)
    return result.drop_duplicates(["date", "code"], keep="last")


def build_rows(outcomes: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + [
        "name", "buy_date_B", "fill_status", "entry_price_gross",
        "exit_date_B_plus_1", "net_return_B_plus_1",
    ])
    if required - set(outcomes):
        raise ValueError("outcome fields missing")
    if outcomes.duplicated(KEYS).any():
        raise ValueError("outcome keys must be unique")
    bar_map = bars.set_index(["date", "code"], drop=False)
    records = []
    for row in outcomes.itertuples(index=False):
        record = {
            "signal_date": str(row.signal_date),
            "code": str(row.code).zfill(6),
            "name": row.name,
            "buy_date_B": row.buy_date_B,
            "exit_date_B_plus_1": row.exit_date_B_plus_1,
            "original_fill_status": row.fill_status,
            "entry_price_gross": pd.to_numeric(
                pd.Series([row.entry_price_gross]), errors="coerce").iloc[0],
            "same_day_stop_executable": False,
            "B_bar_status": "NOT_APPLICABLE_UNFILLED",
            "B_intraday_mae_net_of_buy_cost": np.nan,
            "B_plus_1_exit_status": "NOT_APPLICABLE_UNFILLED",
            "B_plus_1_open": np.nan,
            "B_plus_1_lower_limit": np.nan,
            "B_plus_1_open_at_lower_limit": np.nan,
            "B_plus_1_one_price_lower_limit": np.nan,
            "B_plus_1_open_exit_net_return": np.nan,
            "existing_B_plus_1_close_net_return": pd.to_numeric(
                pd.Series([row.net_return_B_plus_1]), errors="coerce").iloc[0],
        }
        for threshold in THRESHOLDS:
            points = int(threshold * 100)
            record[f"B_intraday_breach_minus_{points}pct"] = np.nan
        if row.fill_status != "FILLED_OPEN_PROXY":
            records.append(record)
            continue
        entry = record["entry_price_gross"]
        buy_key = (str(row.buy_date_B), record["code"])
        exit_key = (str(row.exit_date_B_plus_1), record["code"])
        if pd.isna(entry) or buy_key not in bar_map.index:
            record["B_bar_status"] = "MISSING_OR_INVALID_B_BAR"
            records.append(record)
            continue
        buy = bar_map.loc[buy_key]
        if isinstance(buy, pd.DataFrame):
            buy = buy.iloc[-1]
        buy_low = pd.to_numeric(pd.Series([buy.low]), errors="coerce").iloc[0]
        if pd.isna(buy_low) or buy_low <= 0:
            record["B_bar_status"] = "MISSING_OR_INVALID_B_BAR"
            records.append(record)
            continue
        entry_cash = float(entry) * 1.001 * (1 + .00025 + .00001)
        mae = float(buy_low) / entry_cash - 1
        record["B_bar_status"] = "OBSERVED_T1_LOCKED"
        record["B_intraday_mae_net_of_buy_cost"] = mae
        for threshold in THRESHOLDS:
            points = int(threshold * 100)
            record[f"B_intraday_breach_minus_{points}pct"] = int(mae <= -threshold)

        if exit_key not in bar_map.index:
            record["B_plus_1_exit_status"] = "MISSING_B_PLUS_1_BAR"
            records.append(record)
            continue
        exit_row = bar_map.loc[exit_key]
        if isinstance(exit_row, pd.DataFrame):
            exit_row = exit_row.iloc[-1]
        prices = pd.to_numeric(
            exit_row[["open", "high", "low", "close", "pre_close"]],
            errors="coerce")
        if prices.isna().any() or prices.pre_close <= 0:
            record["B_plus_1_exit_status"] = "INVALID_B_PLUS_1_BAR"
            records.append(record)
            continue
        lower = rounded_limit(float(prices.pre_close), -1)
        open_at_lower = abs(float(prices.open) - lower) < .005
        one_price_lower = open_at_lower and all(
            abs(float(prices[key]) - lower) < .005
            for key in ("high", "low", "close"))
        record.update({
            "B_plus_1_open": float(prices.open),
            "B_plus_1_lower_limit": lower,
            "B_plus_1_open_at_lower_limit": int(open_at_lower),
            "B_plus_1_one_price_lower_limit": int(one_price_lower),
        })
        if one_price_lower:
            record["B_plus_1_exit_status"] = "ONE_PRICE_DOWN_LIMIT_UNSELLABLE"
        elif open_at_lower:
            record["B_plus_1_exit_status"] = "OPEN_DOWN_LIMIT_QUEUE_UNVERIFIABLE"
        else:
            record["B_plus_1_exit_status"] = "B_PLUS_1_OPEN_EXIT_PROXY"
            record["B_plus_1_open_exit_net_return"] = net_return(
                float(entry), float(prices.open))
        records.append(record)
    return pd.DataFrame(records)


def _summary(rows: pd.DataFrame) -> dict:
    filled = rows[rows.original_fill_status.eq("FILLED_OPEN_PROXY")].copy()
    observed = filled[filled.B_bar_status.eq("OBSERVED_T1_LOCKED")].copy()
    result = {
        "filled_open_proxy_rows": int(len(filled)),
        "B_intraday_path_observed_rows": int(len(observed)),
        "same_day_stop_executable_rows": int(
            observed.same_day_stop_executable.astype(bool).sum()),
    }
    for threshold in THRESHOLDS:
        points = int(threshold * 100)
        values = pd.to_numeric(
            observed[f"B_intraday_breach_minus_{points}pct"], errors="coerce"
        ).dropna()
        result[f"B_intraday_breach_minus_{points}pct"] = {
            "known_rows": int(len(values)),
            "count": int(values.sum()) if len(values) else 0,
            "rate": float(values.mean()) if len(values) else None,
            "executable_same_day_exit_count": 0,
        }
    statuses = filled.B_plus_1_exit_status.value_counts(dropna=False)
    result["B_plus_1_exit_status_counts"] = {
        str(key): int(value) for key, value in statuses.items()
    }
    executable = pd.to_numeric(
        filled.B_plus_1_open_exit_net_return, errors="coerce").dropna()
    result["B_plus_1_open_exit_proxy"] = {
        "known_executable_rows": int(len(executable)),
        "mean_net_return": float(executable.mean()) if len(executable) else None,
        "median_net_return": float(executable.median()) if len(executable) else None,
        "big_loss_rate": float(executable.le(-.07).mean()) if len(executable) else None,
    }
    return result


def _records_without_nan(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def audit(outcomes: pd.DataFrame, bars: pd.DataFrame,
          frozen_top1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = build_rows(outcomes, bars)
    frozen = frozen_top1[
        frozen_top1.ranker.eq("profit_emotion_probability")][KEYS].copy()
    if frozen.empty or frozen.duplicated("signal_date").any():
        raise ValueError("frozen emotion top-one is invalid")
    top = frozen.merge(rows, on=KEYS, how="left", validate="one_to_one")
    top_cases = _records_without_nan(top[
        pd.to_numeric(top.B_intraday_breach_minus_7pct, errors="coerce").eq(1)
        | top.B_plus_1_exit_status.isin([
            "ONE_PRICE_DOWN_LIMIT_UNSELLABLE",
            "OPEN_DOWN_LIMIT_QUEUE_UNVERIFIABLE",
        ])
    ][[
        "signal_date", "code", "name", "buy_date_B",
        "B_intraday_mae_net_of_buy_cost", "B_plus_1_exit_status",
        "B_plus_1_open_exit_net_return", "existing_B_plus_1_close_net_return",
    ]])
    report = {
        "version": "v201-t1-unexitable-loss-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "scope": {
            "outcome_rows": int(len(rows)),
            "outcome_sessions": int(rows.signal_date.nunique()),
            "frozen_top1_sessions": int(len(top)),
        },
        "all_candidates": _summary(rows),
        "frozen_emotion_top1": _summary(top),
        "frozen_top1_tail_case_details": top_cases,
        "execution_contract": {
            "B_is_buy_session": True,
            "mainboard_A_share_same_day_sale_allowed": False,
            "B_intraday_threshold_is_executable_stop": False,
            "earliest_daily_bar_exit_proxy": "B_plus_1_open",
            "one_price_B_plus_1_down_limit_is_sellable": False,
            "open_down_limit_then_break_queue_fill_assumed": False,
            "daily_bar_can_verify_queue_position": False,
        },
        "official_basis": {
            "sse_rule": (
                "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/"
                "exchange/c/c_20260424_10816482.shtml"),
            "szse_rule": (
                "https://docs.static.szse.cn/www/lawrules/rule/trade/current/"
                "W020260424690713155663.pdf"),
            "rule_summary": (
                "Securities bought by an investor may not be sold before settlement "
                "unless the product is explicitly eligible for turnaround trading; "
                "ordinary mainboard A shares are not in the same-day list."),
        },
        "governance": {
            "same_day_stop_may_be_backtested_as_executed": False,
            "B_plus_1_down_limit_queue_may_be_assumed_filled": False,
            "fixed_stop_loss_guarantee_allowed": False,
            "pre_entry_veto_or_position_limit_required": True,
            "entry_authorized": False,
        },
        "interpretation": (
            "A B-session drawdown is an unexitable exposure under T+1, not a stop-fill. "
            "Risk control must therefore act before entry through abstention and position "
            "limits, then use the earliest actually executable B+1 evidence. A lower-limit "
            "opening cannot be treated as a guaranteed sale without queue evidence."),
        "limitations": [
            "daily bars do not identify whether the intraday low occurred before or after any later confirmation entry",
            "daily bars cannot verify B+1 lower-limit queue position, partial fills or cancellation sequence",
            "the entry remains an opening-auction proxy and not a broker-confirmed fill",
            "the inspected sample is not a sealed final blind test",
        ],
    }
    return rows, report


def run(outcomes_path, bars_path, top1_path, output_dir) -> dict:
    paths = [Path(outcomes_path), Path(bars_path), Path(top1_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V201 audit")
    outcomes = pd.read_csv(paths[0], dtype={"code": str})
    top1 = pd.read_csv(paths[2], dtype={"code": str})
    pairs = set()
    for row in outcomes.itertuples(index=False):
        code = str(row.code).zfill(6)
        if pd.notna(row.buy_date_B):
            pairs.add((str(row.buy_date_B), code))
        if pd.notna(row.exit_date_B_plus_1):
            pairs.add((str(row.exit_date_B_plus_1), code))
    bars = load_relevant_bars(paths[1], pairs)
    rows, report = audit(outcomes, bars, top1)
    target.mkdir(parents=True)
    rows_path = target / "t1_unexitable_loss_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path) for path in paths}
    report["artifact_hashes"] = {rows_path.name: _sha256(rows_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--bars", default="data/v3/imported/daily-bars.csv")
    parser.add_argument("--top1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.outcomes, args.bars, args.top1, args.output),
                     ensure_ascii=False, indent=2))
