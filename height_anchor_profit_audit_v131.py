"""Link v130 height-anchor selections to strict board-entry price-path outcomes.

The audit deliberately separates three questions:
1. Was the candidate selected before the future window?
2. Did it become or remain a market-height anchor?
3. If it touched the next-session limit and was not a one-price board, did a
   hypothetical limit-price entry have positive B+1/B+2/B+3 close marks?

Daily bars cannot prove an accepted order or fill.  No unfilled selection is
replaced by a runner-up.  Results therefore remain diagnostic-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "MODEL_ALL_TOP1": "模型全池首选",
    "CURRENT_HEIGHT_BASELINE": "机械当前最高板",
    "MODEL_CHALLENGER_TOP1": "模型次高挑战者首选",
    "SECOND_HEIGHT_BASELINE": "机械次高板",
}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})


def prepare(selections: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    selected = selections.copy()
    labels = outcomes.copy()
    for frame in (selected, labels):
        frame["code"] = frame.code.astype(str).str.zfill(6)
        frame["signal_date"] = pd.to_datetime(frame.signal_date).dt.strftime("%Y-%m-%d")
    required_selection = {"code", "signal_date", "selection_method",
                          "top_within_3_sessions", "new_challenger_top_within_3"}
    required_outcome = {"code", "signal_date", "price_path_state",
                        "entry_one_price_LABEL_ONLY", "b1_price_path_LABEL_ONLY",
                        "b1_net_close_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
                        "b3_net_close_mark_LABEL_ONLY"}
    missing = {
        "selections": sorted(required_selection - set(selected)),
        "outcomes": sorted(required_outcome - set(labels)),
    }
    if any(missing.values()):
        raise ValueError("missing v131 fields: " + json.dumps(missing))
    if selected.duplicated(["selection_method", "signal_date"]).any():
        raise ValueError("more than one selection per method/date")
    if labels.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate outcome key")

    rows = selected.merge(labels, on=["code", "signal_date"], how="left",
                          validate="many_to_one", indicator="outcome_join")
    one_price = _bool_series(rows.entry_one_price_LABEL_ONLY)
    valid_path = rows.price_path_state.eq("PRICE_OPPORTUNITY_PROXY_ONLY")
    touched = pd.to_numeric(rows.b1_net_close_mark_LABEL_ONLY, errors="coerce").notna()
    rows["execution_class"] = "OUTCOME_UNAVAILABLE"
    rows.loc[valid_path & one_price.eq(True), "execution_class"] = "QUEUE_UNVERIFIABLE"
    rows.loc[valid_path & one_price.eq(False) & ~touched,
             "execution_class"] = "NO_NEXT_DAY_BOARD_TOUCH"
    rows.loc[valid_path & one_price.eq(False) & touched,
             "execution_class"] = "LIMIT_PRICE_PATH_PROXY"
    return rows


def _wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def _path_metrics(frame: pd.DataFrame) -> dict:
    filled = frame.execution_class.eq("LIMIT_PRICE_PATH_PROXY")
    result: dict[str, object] = {
        "selections": int(len(frame)),
        "joined": int(frame.outcome_join.eq("both").sum()),
        "price_path_fill_proxies": int(filled.sum()),
        "queue_unverifiable": int(frame.execution_class.eq("QUEUE_UNVERIFIABLE").sum()),
        "no_next_day_board_touch": int(frame.execution_class.eq("NO_NEXT_DAY_BOARD_TOUCH").sum()),
        "outcome_unavailable": int(frame.execution_class.eq("OUTCOME_UNAVAILABLE").sum()),
    }
    for horizon in (1, 2, 3):
        values = pd.to_numeric(
            frame.loc[filled, f"b{horizon}_net_close_mark_LABEL_ONLY"], errors="coerce").dropna()
        positive = int(values.gt(0).sum())
        big_loss = int(values.le(-.07).sum())
        result[f"b{horizon}"] = {
            "known": int(len(values)),
            "mean_net_close_mark": float(values.mean()) if len(values) else None,
            "median_net_close_mark": float(values.median()) if len(values) else None,
            "positive_rate": float(values.gt(0).mean()) if len(values) else None,
            "positive_rate_wilson95": _wilson(positive, len(values)),
            "large_loss_rate": float(values.le(-.07).mean()) if len(values) else None,
            "large_loss_rate_wilson95": _wilson(big_loss, len(values)),
        }
    return result


def _role_profit_split(frame: pd.DataFrame) -> dict:
    target = np.where(frame.selection_method.str.contains("CHALLENGER"),
                      pd.to_numeric(frame.new_challenger_top_within_3, errors="coerce"),
                      pd.to_numeric(frame.top_within_3_sessions, errors="coerce"))
    working = frame.assign(role_success=target)
    return {
        str(int(role)): _path_metrics(part)
        for role, part in working[working.role_success.notna()].groupby("role_success")
    }


def _paired_path_delta(rows: pd.DataFrame, left_method: str, right_method: str,
                       horizon: int, bootstrap_samples: int = 5000) -> dict:
    column = f"b{horizon}_net_close_mark_LABEL_ONLY"
    fields = ["signal_date", "execution_class", column]
    left = rows[rows.selection_method.eq(left_method)][fields].rename(
        columns={"execution_class": "left_execution", column: "left_return"})
    right = rows[rows.selection_method.eq(right_method)][fields].rename(
        columns={"execution_class": "right_execution", column: "right_return"})
    paired = left.merge(right, on="signal_date", how="inner", validate="one_to_one")
    valid = (paired.left_execution.eq("LIMIT_PRICE_PATH_PROXY")
             & paired.right_execution.eq("LIMIT_PRICE_PATH_PROXY"))
    paired = paired.loc[valid].copy()
    paired.left_return = pd.to_numeric(paired.left_return, errors="coerce")
    paired.right_return = pd.to_numeric(paired.right_return, errors="coerce")
    paired = paired.dropna(subset=["left_return", "right_return"])
    difference = (paired.left_return - paired.right_return).to_numpy()
    if not len(difference):
        return {"paired_common_fill_dates": 0, "mean_return_delta": None,
                "bootstrap95": None}
    rng = np.random.default_rng(131 + horizon)
    samples = rng.choice(difference, size=(bootstrap_samples, len(difference)),
                         replace=True).mean(axis=1)
    return {
        "paired_common_fill_dates": int(len(difference)),
        "mean_return_delta": float(difference.mean()),
        "bootstrap95": [float(np.quantile(samples, .025)),
                        float(np.quantile(samples, .975))],
    }


def audit(rows: pd.DataFrame) -> dict:
    by_method = {
        method: {
            "label": METHOD_LABELS.get(method, method),
            **_path_metrics(part),
            "profit_by_role_success": _role_profit_split(part),
        }
        for method, part in rows.groupby("selection_method", sort=True)
    }
    comparisons = {}
    pairs = [
        ("MODEL_ALL_TOP1", "CURRENT_HEIGHT_BASELINE"),
        ("MODEL_CHALLENGER_TOP1", "SECOND_HEIGHT_BASELINE"),
    ]
    for left, right in pairs:
        comparisons[f"{left}_vs_{right}"] = {
            f"b{horizon}": _paired_path_delta(rows, left, right, horizon)
            for horizon in (1, 2, 3)
        }
    return {
        "status": "DIAGNOSTIC_ONLY",
        "methods": by_method,
        "paired_common_fill_comparisons": comparisons,
        "decision": {
            "role_model_is_buy_model": False,
            "deployment_authorized": False,
            "reason": "height-anchor role accuracy and buy-after-board profitability are separate gates",
        },
    }


def _fmt(value: float | None) -> str:
    return "不可得" if value is None else f"{value:.2%}"


def markdown(report: dict) -> str:
    lines = [
        "# 最高标角色与买后溢价审计 V131", "",
        "状态：`DIAGNOSTIC_ONLY`。次日触板且非一字仅构成涨停价价格路径代理，不证明真实成交。", "",
        "| 方法 | 逐日首选 | 可研究成交代理 | 一字排队不可验证 | 未触次日板 | B+1均值/正率/大面 | B+2均值/正率/大面 | B+3均值/正率/大面 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for method, item in report["methods"].items():
        horizon = []
        for key in ("b1", "b2", "b3"):
            h = item[key]
            horizon.append("/".join((_fmt(h["mean_net_close_mark"]),
                                     _fmt(h["positive_rate"]),
                                     _fmt(h["large_loss_rate"]))))
        lines.append(
            f"| {item['label']} | {item['selections']} | {item['price_path_fill_proxies']} | "
            f"{item['queue_unverifiable']} | {item['no_next_day_board_touch']} | "
            f"{horizon[0]} | {horizon[1]} | {horizon[2]} |")
    lines += [
        "", "## 解释边界", "",
        "- 选中后次日未触板：没有打板信号，不把第二名递补为成交。",
        "- 次日一字涨停：只能确认价格路径，不能确认排队成交，排除在收益统计外。",
        "- 只有角色预测与可成交后的2—3日收益、大面率同时过关，才可能进入买入排序。",
        "- 本审计不使用竞价逐笔、撤单或真实委托回报，不能直接生成实盘买点。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable v131 output")
    selections = pd.read_csv(args.selections, dtype={"code": str})
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    rows = prepare(selections, outcomes)
    report = {
        "version": "v131-height-anchor-profit-audit-1",
        "source_selection_rows": int(len(selections)),
        "audit": audit(rows),
        "assumptions": {
            "entry": "next-session rounded upper-limit price after daily high touched it",
            "fill": "price-path proxy only; one-price boards excluded; no runner-up substitution",
            "exit_marks": "B+1/B+2/B+3 close after the common hypothetical entry",
            "actual_order_or_fill_verified": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(args.output / "selection_outcomes_LABEL_ONLY.csv", index=False,
                encoding="utf-8-sig")
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "最高标角色与买后溢价审计.md").write_text(
        markdown(report["audit"]), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
