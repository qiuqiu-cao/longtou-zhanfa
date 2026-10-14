#!/usr/bin/env python3
"""Combine V8 relay ranks with verified V9 intraday evidence.

This is deliberately a shadow-research interface.  It can identify one daily
observation candidate and apply the open risk/cancel rules, but it cannot emit
ENTER because the V8 large-loss model and joint historical gate were rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import open_confirmation_v4 as opening
from emotion_challenger_v4 import ACTIVE_STATES
from leader_system_v3 import zfill_code


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = ROOT / "results" / "v8" / "relay_model_lab" / "prequential_predictions.csv"
DEFAULT_SNAPSHOTS = ROOT / "data" / "v9" / "derived_open_snapshots.csv"
DEFAULT_OUTPUT = ROOT / "results" / "v9" / "intraday_shadow"
VERSION = "v9-intraday-shadow-1"
PROMOTION = "p__promotion__plus_stock_memory__hist_gb"
FILL = "p__fill__plus_stock_memory__hist_gb"
TWO_MORE = "p__two_more__plus_stock_memory__hist_gb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_snapshots(path: Path, signal: str, trade: str, asof: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    raw = pd.read_csv(path, dtype={"code": "string"})
    if raw.empty:
        return pd.DataFrame()
    checked = opening.validate_snapshots(raw)
    checked = checked[
        checked["signal_date"].eq(signal)
        & checked["trade_date"].eq(trade)
        & (checked["asof"].map(opening.clock_seconds) <= opening.clock_seconds(asof))
    ].sort_values("asof", key=lambda x: x.map(opening.clock_seconds)).drop_duplicates("code", keep="last")
    return checked


def score_shadow(
    predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
    signal_date: str,
    trade_date: str,
    asof: str,
) -> pd.DataFrame:
    signal = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    point = opening.normalize_asof(asof)
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["code"] = frame["code"].map(zfill_code)
    frame = frame[frame["date"].eq(signal)].copy()
    if frame.empty:
        raise ValueError(f"no V8 predictions for signal date {signal}")
    for field in (PROMOTION, FILL, TWO_MORE):
        if field not in frame:
            raise ValueError(f"V8 predictions missing {field}")
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["joint_score"] = 0.65 * frame[TWO_MORE] + 0.25 * frame[PROMOTION] + 0.10 * frame[FILL]
    frame["shadow_scope"] = (
        frame["boards"].between(3, 5)
        & frame["emotion_state"].isin(ACTIVE_STATES)
        & pd.to_numeric(frame["buyable_eod"], errors="coerce").eq(1)
        & pd.to_numeric(frame["one_word_proxy"], errors="coerce").eq(0)
        & frame[[PROMOTION, FILL, TWO_MORE]].notna().all(axis=1)
    )
    frame["shadow_observation"] = False
    eligible = frame[frame["shadow_scope"]]
    if not eligible.empty:
        top_index = eligible.sort_values(["joint_score", "boards", "code"], ascending=[False, False, True]).index[0]
        frame.loc[top_index, "shadow_observation"] = True
    snapshot_map = snapshots.set_index("code").to_dict("index") if not snapshots.empty else {}
    rows = []
    for _, row in frame.sort_values(["shadow_observation", "joint_score"], ascending=[False, False]).iterrows():
        code = row["code"]
        selected_snapshot = pd.Series(snapshot_map[code]) if code in snapshot_map else None
        if not bool(row["shadow_scope"]):
            confirmation = "OUT_OF_RESEARCH_SCOPE"
            reasons = "非3—5板、非活跃情绪、前一日不可买或概率缺失"
        elif not bool(row["shadow_observation"]):
            confirmation = "NOT_DAILY_TOP_SHADOW"
            reasons = "每日只预注册联合排序第一的影子观察"
        else:
            candidate = row.copy()
            candidate["date"] = signal
            candidate["announcement_risk"] = candidate.get("announcement_risk", math.nan)
            evaluated = opening.evaluate_row(
                candidate, selected_snapshot, point, allow_height_confirmed_research=True,
            )
            confirmation = str(evaluated["confirmation_status"])
            reasons = str(evaluated["reasons"])
            if confirmation == "CONFIRM_RESEAL":
                confirmation = "SHADOW_CONFIRM_RESEAL"
        rows.append({
            "signal_date": signal, "trade_date": trade, "asof": point,
            "code": code, "name": row.get("name", ""), "boards": int(row["boards"]),
            "emotion_state": row.get("emotion_state", ""),
            "p_promotion": row[PROMOTION], "p_fill_proxy": row[FILL],
            "p_two_more": row[TWO_MORE], "joint_score": row["joint_score"],
            "shadow_scope": bool(row["shadow_scope"]),
            "shadow_observation": bool(row["shadow_observation"]),
            "open_confirmation": confirmation, "reasons": reasons,
            "large_loss_probability": math.nan,
            "large_loss_probability_status": "UNAVAILABLE_REJECTED_V8_MODEL",
            "decision": "SHADOW_ONLY_NEVER_ENTER",
            "execution_evidence": "CONDITIONAL_PAPER_ONLY_NO_LEVEL2_QUEUE",
        })
    return pd.DataFrame(rows)


def run(
    signal_date: str,
    trade_date: str,
    asof: str,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    snapshots_path: Path = DEFAULT_SNAPSHOTS,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path, dtype={"code": "string"}, low_memory=False)
    signal = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    trade = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    point = opening.normalize_asof(asof)
    snapshots = _load_snapshots(snapshots_path, signal, trade, point)
    scored = score_shadow(predictions, snapshots, signal, trade, point)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{signal}_to_{trade}_{point.replace(':', '')}"
    score_path = output / f"{stem}.csv"
    scored.to_csv(score_path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": VERSION, "status": "RESEARCH_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": signal, "trade_date": trade, "asof": point,
        "rows": int(len(scored)),
        "shadow_observations": int(scored["shadow_observation"].sum()),
        "shadow_confirmed": int(scored["open_confirmation"].eq("SHADOW_CONFIRM_RESEAL").sum()),
        "enter_count": 0, "deployment_allowed": False,
        "deployment_reason": "V8 joint risk gate failed and large-loss probability is unavailable",
        "scores": {"path": str(score_path.resolve()), "sha256": sha256_file(score_path)},
        "inputs": {
            "predictions": {"path": str(predictions_path.resolve()), "sha256": sha256_file(predictions_path)},
            "snapshots": {
                "path": str(snapshots_path.resolve()),
                "sha256": sha256_file(snapshots_path) if snapshots_path.exists() else None,
                "status": "VERIFIED_DERIVED" if not snapshots.empty else "UNAVAILABLE",
            },
        },
    }
    manifest_path = output / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 intraday shadow confirmation")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--asof", required=True, choices=opening.SNAPSHOT_TIMES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(
        args.signal_date, args.trade_date, args.asof,
        args.predictions, args.snapshots, args.output,
    ), ensure_ascii=False, indent=2))
