"""Audit next-session exit timing causality and hindsight timing choice.

A B+1 official open can proxy a sell only for an exit instruction frozen before
that opening result is known.  A rule that observes the open and then claims a
fill at that same open has the same-event timing error already prohibited for
entries.  Selecting ex post between the B+1 open and close is an exit oracle.
This audit quantifies that optimism without promoting either timing rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["signal_date", "code"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _return_metrics(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(values)),
        "mean_net_return": float(values.mean()) if len(values) else None,
        "median_net_return": float(values.median()) if len(values) else None,
        "positive_rate": float(values.gt(0).mean()) if len(values) else None,
        "big_loss_rate": float(values.le(-.07).mean()) if len(values) else None,
    }


def summarize(rows: pd.DataFrame) -> dict:
    executable = rows[
        rows.B_plus_1_exit_status.eq("B_PLUS_1_OPEN_EXIT_PROXY")].copy()
    executable["open_return"] = pd.to_numeric(
        executable.B_plus_1_open_exit_net_return, errors="coerce")
    executable["close_mark_return"] = pd.to_numeric(
        executable.existing_B_plus_1_close_net_return, errors="coerce")
    comparable = executable.dropna(
        subset=["open_return", "close_mark_return"]).copy()
    comparable["hindsight_best_return"] = comparable[
        ["open_return", "close_mark_return"]].max(axis=1)
    comparable["hindsight_worst_return"] = comparable[
        ["open_return", "close_mark_return"]].min(axis=1)
    open_big = comparable.open_return.le(-.07)
    close_big = comparable.close_mark_return.le(-.07)
    open_mean = comparable.open_return.mean() if len(comparable) else np.nan
    close_mean = comparable.close_mark_return.mean() if len(comparable) else np.nan
    oracle_mean = (comparable.hindsight_best_return.mean()
                   if len(comparable) else np.nan)
    return {
        "rows": int(len(rows)),
        "open_exit_proxy_status_rows": int(len(executable)),
        "comparable_open_and_close_rows": int(len(comparable)),
        "registered_pre_open_exit_plan_rows": 0,
        "fixed_precommitted_B_plus_1_open_proxy": _return_metrics(
            comparable.open_return),
        "fixed_B_plus_1_close_mark": _return_metrics(
            comparable.close_mark_return),
        "hindsight_choose_better_open_or_close": _return_metrics(
            comparable.hindsight_best_return),
        "hindsight_choose_worse_open_or_close": _return_metrics(
            comparable.hindsight_worst_return),
        "hindsight_timing_choice_mean_optimism_vs_better_fixed_rule": (
            float(oracle_mean - max(open_mean, close_mean))
            if len(comparable) else None),
        "timing_winner_counts": {
            "open_better": int(
                comparable.open_return.gt(comparable.close_mark_return).sum()),
            "close_better": int(
                comparable.close_mark_return.gt(comparable.open_return).sum()),
            "equal": int(
                comparable.open_return.eq(comparable.close_mark_return).sum()),
        },
        "big_loss_transition_counts": {
            "big_at_open_and_close": int((open_big & close_big).sum()),
            "big_at_open_not_close": int((open_big & ~close_big).sum()),
            "not_big_at_open_big_at_close": int((~open_big & close_big).sum()),
            "not_big_at_either": int((~open_big & ~close_big).sum()),
        },
        "unresolved_exit_status_counts": {
            str(key): int(value) for key, value in rows.loc[
                ~rows.B_plus_1_exit_status.eq("B_PLUS_1_OPEN_EXIT_PROXY"),
                "B_plus_1_exit_status"].value_counts(dropna=False).items()
        },
    }


def audit(rows: pd.DataFrame, frozen_top1: pd.DataFrame,
          readiness: dict) -> dict:
    required = set(KEYS + [
        "name", "B_plus_1_exit_status", "B_plus_1_open_exit_net_return",
        "existing_B_plus_1_close_net_return",
    ])
    if required - set(rows):
        raise ValueError("T+1 audit fields missing")
    if rows.duplicated(KEYS).any():
        raise ValueError("T+1 rows must be unique")
    frozen = frozen_top1[
        frozen_top1.ranker.eq("profit_emotion_probability")][KEYS].copy()
    if frozen.empty or frozen.duplicated("signal_date").any():
        raise ValueError("frozen emotion top-one is invalid")
    top = frozen.merge(rows, on=KEYS, how="left", validate="one_to_one")
    technical = int(readiness.get("technical_complete_sequences", 0) or 0)
    full = int(readiness.get("full_complete_sequences", 0) or 0)
    report = {
        "version": "v202-exit-information-causality-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "all_candidates": summarize(rows),
        "frozen_emotion_top1": summarize(top),
        "exit_information_contract": {
            "B_plus_1_open_known_before_opening_match": False,
            "official_open_can_fill_precommitted_exit_proxy": True,
            "official_open_can_condition_same_open_exit": False,
            "post_open_conditional_exit_requires_new_price": True,
            "B_plus_1_close_mark_is_verified_execution": False,
            "choose_open_or_close_after_seeing_both_allowed": False,
            "unmonitored_user_requires_pre_frozen_instruction": True,
        },
        "intraday_exit_readiness": {
            "technical_complete_sequences": technical,
            "full_complete_sequences": full,
            "post_open_exit_model_ready": bool(technical > 0 and full > 0),
        },
        "official_basis": {
            "sse_rule": (
                "https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/"
                "exchange/c/c_20260424_10816482.shtml"),
            "szse_rule": (
                "https://docs.static.szse.cn/www/lawrules/rule/trade/current/"
                "W020260424690713155663.pdf"),
            "timing_summary": (
                "Opening call auction is 09:15-09:25 and continuous auction "
                "begins at 09:30; an official opening result cannot retroactively "
                "determine an order filled in that same opening match."),
        },
        "governance": {
            "same_open_conditional_exit_fill_allowed": False,
            "hindsight_open_close_timing_oracle_allowed": False,
            "historical_open_proxy_is_broker_confirmed_exit": False,
            "post_open_exit_rule_authorized": False,
            "entry_or_exit_policy_promoted": False,
        },
        "interpretation": (
            "For a user who may not monitor the open, an opening-auction exit must "
            "be frozen before the official open is known. Observing the opening "
            "result and retaining that same opening price is non-causal. Choosing "
            "afterward between open and close is a hindsight timing oracle and may "
            "not be reported as a strategy result."),
        "limitations": [
            "historical rows contain no broker order timestamp or registered exit plan",
            "the close field is a valuation mark, not a verified closing-auction fill",
            "no complete minute sequence exists for a causal 09:30/09:35 exit model",
            "lower-limit queue and partial-fill states remain unresolved without order evidence",
        ],
    }
    return report


def run(rows_path, top1_path, readiness_path, output_path) -> dict:
    paths = [Path(rows_path), Path(top1_path), Path(readiness_path)]
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V202 audit")
    rows = pd.read_csv(paths[0], dtype={"code": str})
    top1 = pd.read_csv(paths[1], dtype={"code": str})
    readiness = json.loads(paths[2].read_text(encoding="utf-8"))
    report = audit(rows, top1, readiness)
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path) for path in paths}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--top1", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.rows, args.top1, args.readiness, args.output),
                     ensure_ascii=False, indent=2))
