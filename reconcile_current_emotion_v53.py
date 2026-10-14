"""Recompute a latest emotion row from reconciled closing/failed evidence.

The historical market table stays untouched. A new immutable overlay is bound
to the reconciled pool hashes for the closing-plan daily review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import leader_system_v3 as v3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_pool(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "code": frame.code.astype(str).str.zfill(6),
        "boards": pd.to_numeric(frame.boards, errors="raise").astype(int),
        "first_seal": frame.first_seal_time,
        "last_seal": frame.last_seal_time,
        "bomb_count": frame.failed_seal_count,
        "turnover": frame.turnover,
    })


def _previous_pool(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    filtered["_code"] = filtered["代码"].astype(str).str.zfill(6)
    mainboard = filtered["_code"].str.startswith(
        ("000", "001", "002", "003", "600", "601", "603", "605")
    )
    names = filtered["名称"].astype(str) if "名称" in filtered else pd.Series("", index=filtered.index)
    filtered = filtered[mainboard & ~names.str.contains(r"ST|退", case=False, regex=True)].copy()
    board_column = "昨日连板数" if "昨日连板数" in filtered else "连板数"
    return pd.DataFrame({
        "code": filtered["_code"],
        "boards": pd.to_numeric(filtered[board_column], errors="raise").astype(int),
    })


def build(market: pd.DataFrame, closing: pd.DataFrame, failed: pd.DataFrame,
          previous_limit_up: pd.DataFrame, limit_down_count: int, signal_date: str):
    dates = market.date.astype(str)
    matched = market[dates.eq(signal_date)]
    history = market[dates.lt(signal_date)].sort_values("date").copy()
    if len(matched) != 1 or history.empty:
        raise ValueError("market table must contain one target row and prior history")
    if set(closing.code.astype(str).str.zfill(6)) & set(failed.code.astype(str).str.zfill(6)):
        raise ValueError("closing and failed pools overlap")
    current_pool = _normal_pool(closing)
    previous_pool = _previous_pool(previous_limit_up)
    old = matched.iloc[0].to_dict()
    row = dict(old)
    ladder = current_pool[current_pool.boards.ge(2)]
    limit_count, failed_count = len(current_pool), len(failed)
    row.update({
        "source": "Reconciled Eastmoney pools + independent public close quotes",
        "source_timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "limit_count": float(limit_count), "consecutive_count": int(len(ladder)),
        "max_board": int(ladder.boards.max()) if len(ladder) else 1,
        "failed_count": float(failed_count), "down_limit_count": float(limit_down_count),
        "seal_rate": limit_count / (limit_count + failed_count),
        "failed_rate": failed_count / (limit_count + failed_count),
        "loss_fields_status": "reconciled_recent_pool",
    })
    prior = history.iloc[-1]
    row["height_change"] = row["max_board"] - int(prior.max_board)
    row["limit_change"] = row["limit_count"] - float(prior.limit_count)
    row["middle_density"] = float(ladder.boards.between(2, 4).mean()) if len(ladder) else 0.0
    row["unique_space"] = float((ladder.boards == row["max_board"]).sum() == 1) if len(ladder) else 0.0
    levels = set(ladder.boards.astype(int))
    row["ladder_gap_count"] = sum(level not in levels for level in range(2, row["max_board"] + 1))
    one_word = []
    for item in ladder.to_dict("records"):
        first, last = v3.parse_clock(item.get("first_seal")), v3.parse_clock(item.get("last_seal"))
        bombs, turnover = v3.safe_float(item.get("bomb_count")), v3.safe_float(item.get("turnover"))
        locked = (not math.isnan(first) and first <= 565.02
                  and not math.isnan(last) and last <= 565.02
                  and (math.isnan(bombs) or bombs == 0)
                  and (math.isnan(turnover) or turnover < 3))
        one_word.append(float(locked))
    row["one_word_share"] = float(np.mean(one_word)) if one_word else math.nan
    row.update(v3.promotion_rates(previous_pool, current_pool))
    prior_space = [v3.safe_float(value) for value in history.space_feedback.tail(2)]
    valid_space = [value for value in prior_space + [row["space_feedback"]] if not math.isnan(value)]
    row["space_feedback_3d"] = float(np.mean(valid_space)) if valid_space else math.nan
    row["breadth_height_divergence"] = float(row["height_change"] > 0 and row["limit_change"] < -20)
    previous_rows = history.to_dict("records")
    row.update(v3.rolling_emotion_context(previous_rows, row))
    row["emotion_heat_score"] = v3.weighted_available([
        (0.12, v3.clamp((row["limit_count"] - 20) / 80)),
        (0.12, v3.clamp((row["max_board"] - 2) / 6)),
        (0.08, v3.clamp((row["height_change"] + 3) / 6)),
        (0.17, row["promotion_overall"]), (0.12, row["promotion_mid"]),
        (0.13, v3.weighted_available([
            (0.55, 1 - v3.clamp(row["failed_rate"] / .30)),
            (0.45, 1 - v3.clamp(row["down_limit_count"] / 10)),
        ])),
        (0.11, v3.clamp(row["middle_density"])), (0.08, row["unique_space"]),
        (0.07, row["space_feedback"]),
    ])
    row.update(v3.emotion_dimensions(row))
    row["emotion_score"] = row["emotion_quality_score"]
    row["emotion_state"] = v3.classify_emotion(row, str(prior.emotion_state))
    v3.attach_state_features(row, row["emotion_state"])
    required = ("failed_count", "down_limit_count", "seal_rate", "yesterday_premium",
                "previous_limit_tail_loss")
    missing = [key for key in required if math.isnan(v3.safe_float(row.get(key)))]
    row["unavailable_fields"] = ",".join(missing)
    completeness = v3.safe_float(row.get("emotion_data_completeness"))
    row["data_status"] = "verified" if completeness >= .90 else "partial" if completeness >= .50 else "sparse"
    combined = pd.concat([history, pd.DataFrame([row])], ignore_index=True, sort=False)
    corrected = v3.attach_online_emotion_forecast(combined).iloc[-1].to_dict()
    changes = {}
    for key in ("limit_count", "consecutive_count", "failed_count", "down_limit_count",
                "seal_rate", "failed_rate", "promotion_overall", "promotion_1to2",
                "promotion_2to3", "promotion_3to4", "max_board", "emotion_state",
                "emotion_score", "emotion_adverse_next_probability"):
        before, after = old.get(key), corrected.get(key)
        if str(before) != str(after):
            changes[key] = {"before": None if pd.isna(before) else before,
                            "after": None if pd.isna(after) else after}
    return pd.DataFrame([corrected]), changes


def run(args):
    paths = [Path(args.market), Path(args.closing), Path(args.failed),
             Path(args.previous_limit_up), Path(args.raw_manifest)]
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable emotion overlay")
    raw_manifest = json.loads(paths[4].read_text(encoding="utf-8"))
    if str(raw_manifest.get("date")) != pd.Timestamp(args.date).strftime("%Y%m%d"):
        raise ValueError("raw manifest date mismatch")
    down = raw_manifest.get("datasets", {}).get("limit_down", {})
    if down.get("status") != "verified":
        raise ValueError("limit-down count is not provider-verified")
    row, changes = build(
        pd.read_csv(paths[0]), pd.read_csv(paths[1], dtype={"code": str}),
        pd.read_csv(paths[2], dtype={"code": str}),
        pd.read_csv(paths[3], dtype={"代码": str}), int(down["rows"]),
        pd.Timestamp(args.date).strftime("%Y-%m-%d"),
    )
    target.mkdir(parents=True, exist_ok=False)
    row_path = target / "market_emotion_reconciled.csv"
    row.to_csv(row_path, index=False, encoding="utf-8-sig")
    report = {
        "signal_date": pd.Timestamp(args.date).strftime("%Y-%m-%d"),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "RECONCILED_EMOTION_OVERLAY",
        "changes_from_raw_market_row": changes,
        "entry_authorized": False,
        "limitation": "The quote scan is complete for the bound current official roster; the emotion forecast remains research-only.",
        "source_hashes": {str(path.resolve()): sha256_file(path) for path in paths + [Path(__file__)]},
        "output_sha256": sha256_file(row_path),
    }
    (target / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--closing", required=True)
    parser.add_argument("--failed", required=True)
    parser.add_argument("--previous-limit-up", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
