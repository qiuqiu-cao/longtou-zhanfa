"""Audit payoff asymmetry for frozen daily first-board top-one selections.

Win rate is not expectancy.  This audit keeps the frozen top-one selection,
never replaces an unfilled name, and reports payoff ratio, profit factor,
compounded return and mark-to-close maximum drawdown for each ranker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drawdown(returns: pd.Series) -> tuple[float | None, float | None]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(float)
    if len(values) == 0:
        return None, None
    equity = np.concatenate([[1.0], np.cumprod(1.0 + values)])
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(equity[-1] - 1.0), float(drawdowns.min())


def trade_metrics(part: pd.DataFrame) -> dict:
    selected_sessions = int(part.signal_date.nunique())
    missing = part.fill_status.isin(["MISSING_B_BAR", "INVALID_B_PRICE"])
    queue = part.fill_status.eq("UNVERIFIABLE_LIMIT_QUEUE")
    filled = part[part.fill_status.eq("FILLED_OPEN_PROXY")].copy()
    returns = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
    winners = returns[returns.gt(0)]
    losers = returns[returns.le(0)]
    gross_profit = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(-losers.sum()) if len(losers) else 0.0
    avg_win = float(winners.mean()) if len(winners) else None
    avg_loss = float(losers.mean()) if len(losers) else None
    compounded, max_drawdown = _drawdown(returns)
    big = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
    return {
        "selected_sessions": selected_sessions,
        "missing_execution_evidence_sessions": int(missing.sum()),
        "unverifiable_limit_queue_sessions": int(queue.sum()),
        "filled_trades": int(len(filled)),
        "B_plus_2_known_trades": int(len(returns)),
        "win_count": int(returns.gt(0).sum()),
        "loss_or_flat_count": int(returns.le(0).sum()),
        "win_rate": float(returns.gt(0).mean()) if len(returns) else None,
        "mean_net_return": float(returns.mean()) if len(returns) else None,
        "median_net_return": float(returns.median()) if len(returns) else None,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": (
            float(avg_win / abs(avg_loss))
            if avg_win is not None and avg_loss not in (None, 0) else None),
        "profit_factor": (
            float(gross_profit / gross_loss) if gross_loss > 0 else None),
        "compounded_return_full_capital_proxy": compounded,
        "maximum_drawdown_full_capital_proxy": max_drawdown,
        "best_trade": float(returns.max()) if len(returns) else None,
        "worst_trade": float(returns.min()) if len(returns) else None,
        "B_plus_1_big_loss_rate": float(big.mean()) if len(big) else None,
    }


def _bootstrap(part: pd.DataFrame, repetitions: int = 5000,
               seed: int = 198) -> dict:
    returns = pd.to_numeric(
        part.loc[part.fill_status.eq("FILLED_OPEN_PROXY"), "net_return_B_plus_2"],
        errors="coerce").dropna().to_numpy(float)
    if len(returns) == 0:
        return {"mean_net_return": None, "win_rate": None, "profit_factor": None}
    rng = np.random.default_rng(seed)
    draws = {"mean_net_return": [], "win_rate": [], "profit_factor": []}
    for _ in range(repetitions):
        sample = rng.choice(returns, len(returns), replace=True)
        draws["mean_net_return"].append(float(sample.mean()))
        draws["win_rate"].append(float((sample > 0).mean()))
        profit = sample[sample > 0].sum()
        loss = -sample[sample <= 0].sum()
        if loss > 0:
            draws["profit_factor"].append(float(profit / loss))
    return {
        key: np.quantile(values, [.025, .975]).tolist() if values else None
        for key, values in draws.items()
    }


def _paired(frame: pd.DataFrame, left: str, right: str) -> dict:
    columns = ["signal_date", "fill_status", "net_return_B_plus_2", "code"]
    a = frame[frame.ranker.eq(left)][columns].rename(columns={
        "fill_status": "left_fill", "net_return_B_plus_2": "left_return",
        "code": "left_code"})
    b = frame[frame.ranker.eq(right)][columns].rename(columns={
        "fill_status": "right_fill", "net_return_B_plus_2": "right_return",
        "code": "right_code"})
    joined = a.merge(b, on="signal_date", how="inner", validate="one_to_one")
    common = joined[
        joined.left_fill.eq("FILLED_OPEN_PROXY")
        & joined.right_fill.eq("FILLED_OPEN_PROXY")].copy()
    common["difference"] = (
        pd.to_numeric(common.left_return, errors="coerce")
        - pd.to_numeric(common.right_return, errors="coerce"))
    common = common.dropna(subset=["difference"])
    return {
        "common_filled_sessions": int(len(common)),
        "same_stock_sessions": int(common.left_code.eq(common.right_code).sum()),
        "left_better_sessions": int(common.difference.gt(0).sum()),
        "tie_sessions": int(common.difference.eq(0).sum()),
        "left_worse_sessions": int(common.difference.lt(0).sum()),
        "mean_return_difference": float(common.difference.mean()) if len(common) else None,
    }


def audit(frame: pd.DataFrame, bootstrap_repetitions: int = 5000) -> dict:
    required = {
        "ranker", "signal_date", "code", "fill_status", "net_return_B_plus_2",
        "big_loss_B_plus_1",
    }
    if required - set(frame):
        raise ValueError(f"top-one fields missing: {sorted(required - set(frame))}")
    if frame.duplicated(["ranker", "signal_date"]).any():
        raise ValueError("each ranker must have exactly one frozen selection per session")
    rankers = sorted(frame.ranker.unique())
    metrics = {}
    for ranker in rankers:
        part = frame[frame.ranker.eq(ranker)].sort_values("signal_date")
        metrics[ranker] = trade_metrics(part)
        metrics[ranker]["bootstrap95"] = _bootstrap(
            part, repetitions=bootstrap_repetitions,
            seed=198 + sum(map(ord, ranker)))
    comparisons = {}
    focus = "profit_emotion_probability"
    if focus in rankers:
        for ranker in rankers:
            if ranker != focus:
                comparisons[f"{focus}_minus_{ranker}"] = _paired(frame, focus, ranker)
    focus_metrics = metrics.get(focus, {})
    higher_hit_negative_expectancy = bool(
        focus_metrics.get("win_rate") is not None
        and focus_metrics.get("mean_net_return") is not None
        and focus_metrics["mean_net_return"] <= 0
        and any(
            focus_metrics["win_rate"] > values.get("win_rate", 1)
            for name, values in metrics.items() if name != focus
            and values.get("win_rate") is not None)
    )
    return {
        "version": "v198-first-board-payoff-tail-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "ranker_metrics": metrics,
        "paired_comparisons": comparisons,
        "diagnosis": {
            "emotion_ranker_higher_hit_but_negative_expectancy": (
                higher_hit_negative_expectancy),
            "win_rate_alone_sufficient_for_strategy": False,
        },
        "governance": {
            "unfilled_selection_replaced": False,
            "full_capital_compounding_is_account_backtest": False,
            "profit_factor_above_one_required_before_promotion": True,
            "positive_mean_return_required_before_promotion": True,
            "acceptable_drawdown_required_before_promotion": True,
            "ranker_promotion_allowed": False,
            "entry_authorized": False,
        },
        "interpretation": (
            "A higher hit rate can coexist with negative expectancy when losses are larger "
            "than wins. Identity, win probability, payoff magnitude and tail loss must be "
            "separate model targets."),
        "limitations": [
            "only ten complete evaluation sessions are available",
            "the period has already been inspected and is not a sealed blind test",
            "returns use a daily-open execution proxy and B+2 close valuation",
            "full-capital compounding ignores concurrent positions and is not an account ledger",
            "no threshold or stop rule is tuned on this small sample",
        ],
    }


def run(input_path, output_dir) -> dict:
    source = Path(input_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable payoff-tail audit")
    frame = pd.read_csv(source, dtype={"code": str})
    report = audit(frame)
    target.mkdir(parents=True)
    report["source_hash"] = {str(source.resolve()): _sha256(source)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))
