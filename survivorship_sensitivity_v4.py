#!/usr/bin/env python3
"""Measure emotion sensitivity after point-in-time delisted-stock recovery.

The verified reconstruction uses official SZSE name history and official SSE
announcement/PDF intervals. Metadata-only SSE transitions stay inferred and
cannot enter the verified sample. Frozen V3 labels are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from security_master_v4 import validate_master


ROOT = Path(__file__).resolve().parent
BARS = ROOT / "data" / "v4" / "historical_universe" / "delisted_daily_bars.csv"
MARKET = ROOT / "data" / "v3" / "market_emotion.csv"
ORIGINAL_CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
COMBINED_CANDIDATES = ROOT / "data" / "v4" / "historical_universe" / "leader_candidates_with_verified_delisted.csv"
SECURITY_MASTER = ROOT / "data" / "v4" / "security_master_intervals.csv"
OUTPUT = ROOT / "results" / "v4" / "historical_universe"
VERSION = "v4-survivorship-sensitivity-2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def attach_streaks(bars: pd.DataFrame, eligibility: pd.Series, target: str) -> pd.DataFrame:
    result = bars.copy()
    hit = result["is_limit_10"] & eligibility
    groups = (~hit).groupby(result["code"]).cumsum()
    result[target] = hit.groupby([result["code"], groups]).cumsum().where(hit, 0).astype(int)
    return result


def attach_sse_point_in_time_eligibility(bars: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    result = bars.copy()
    result["sse_status_verified"] = False
    result["sse_verified_eligible"] = False
    sse_master = master[master["exchange"].eq("SSE")].copy()
    sse_master["_start"] = pd.to_datetime(sse_master["valid_from"])
    sse_master["_end"] = pd.to_datetime(sse_master["valid_to"]).fillna(pd.Timestamp("2262-04-11"))
    for code, part in sse_master.groupby("code", sort=False):
        bar_index = result.index[result["code"].eq(code)]
        if len(bar_index) == 0:
            continue
        for _, interval in part.iterrows():
            covered = bar_index[
                (result.loc[bar_index, "date"] >= interval["_start"])
                & (result.loc[bar_index, "date"] <= interval["_end"])
            ]
            if len(covered) == 0:
                continue
            verified = bool(interval["source_verified"])
            result.loc[covered, "sse_status_verified"] = verified
            result.loc[covered, "sse_verified_eligible"] = bool(
                verified and interval["board"] == "MAINBOARD" and interval["listing_status"] == "LISTED"
                and interval["st_status"] == "NORMAL" and interval["price_limit_regime"] == "MAINBOARD_10"
            )
    return result


def prepare_bars(path: Path = BARS, master_path: Path = SECURITY_MASTER) -> pd.DataFrame:
    bars = pd.read_csv(path, dtype={"code": "string"})
    bars["date"] = pd.to_datetime(bars["date"], errors="raise").dt.normalize()
    bars = bars.sort_values(["code", "date"]).reset_index(drop=True)
    for field in ["is_limit_10", "touched_limit_10", "is_down_limit_10"]:
        bars[field] = as_bool(bars[field])
    szse_verified = (
        bars["exchange"].eq("SZSE")
        & bars["status_basis"].eq("VERIFIED_SZSE_DATED_NAME_HISTORY")
        & bars["st_status"].eq("NORMAL")
    )
    master = validate_master(pd.read_csv(master_path, dtype={"code": "string"}))
    bars = attach_sse_point_in_time_eligibility(bars, master)
    bars["verified_eligible"] = szse_verified | bars["sse_verified_eligible"]
    bars["upper_bound_eligible"] = bars["verified_eligible"] | bars["exchange"].eq("SSE")
    bars = attach_streaks(bars, bars["verified_eligible"], "verified_boards")
    bars = attach_streaks(bars, bars["upper_bound_eligible"], "upper_boards")
    return bars


def candidate_rows(bars: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for boards_field, scope, evidence in [
        ("verified_boards", "VERIFIED_POINT_IN_TIME", "official dated SSE/SZSE name and status evidence"),
        ("upper_boards", "SSE_UNRESOLVED_UPPER_BOUND", "10% OHLC pattern; point-in-time SSE interval not verified"),
    ]:
        if scope.startswith("SSE"):
            part = bars[
                bars["exchange"].eq("SSE") & ~bars["sse_status_verified"]
                & bars[boards_field].between(2, 5)
            ].copy()
        else:
            part = bars[bars[boards_field].between(2, 5)].copy()
        next_boards = bars.groupby("code")[boards_field].shift(-1)
        next_date = bars.groupby("code")["date"].shift(-1)
        part["promoted_next_observation"] = (
            next_boards.loc[part.index].to_numpy() == part[boards_field].to_numpy() + 1
        )
        part["next_observation_date"] = next_date.loc[part.index].to_numpy()
        part["boards"] = part[boards_field]
        part["scope"] = scope
        part["evidence_basis"] = evidence
        frames.append(part[[
            "date", "code", "exchange", "name", "final_name", "boards", "close",
            "promoted_next_observation", "next_observation_date", "scope", "evidence_basis",
        ]])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code", "scope"])


def aggregate_scope(bars: pd.DataFrame, eligibility_field: str, boards_field: str, prefix: str) -> pd.DataFrame:
    part = bars[bars[eligibility_field]].copy()
    part["failed"] = part["touched_limit_10"] & ~part["is_limit_10"]
    grouped = part.groupby("date").agg(
        added_limit=("is_limit_10", "sum"),
        added_failed=("failed", "sum"),
        added_down_limit=("is_down_limit_10", "sum"),
        added_consecutive=(boards_field, lambda values: int((values >= 2).sum())),
        added_max_board=(boards_field, "max"),
    )
    return grouped.add_prefix(prefix + "_")


def build_sensitivity(bars: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    lower = aggregate_scope(bars, "verified_eligible", "verified_boards", "verified")
    upper = aggregate_scope(bars, "upper_bound_eligible", "upper_boards", "upper")
    result = market.merge(lower, left_on="date", right_index=True, how="left")
    result = result.merge(upper, left_on="date", right_index=True, how="left")
    addition_columns = [column for column in result.columns if column.startswith(("verified_", "upper_"))]
    result[addition_columns] = result[addition_columns].fillna(0)
    for prefix in ["verified", "upper"]:
        result[f"{prefix}_revised_limit_count"] = result["limit_count"] + result[f"{prefix}_added_limit"]
        result[f"{prefix}_revised_failed_count"] = result["failed_count"] + result[f"{prefix}_added_failed"]
        result[f"{prefix}_revised_down_limit_count"] = result["down_limit_count"] + result[f"{prefix}_added_down_limit"]
        result[f"{prefix}_revised_consecutive_count"] = result["consecutive_count"] + result[f"{prefix}_added_consecutive"]
        result[f"{prefix}_revised_max_board"] = np.maximum(result["max_board"], result[f"{prefix}_added_max_board"])
        denominator = result[f"{prefix}_revised_limit_count"] + result[f"{prefix}_revised_failed_count"]
        result[f"{prefix}_revised_failed_rate"] = result[f"{prefix}_revised_failed_count"] / denominator.replace(0, np.nan)
        result[f"{prefix}_limit_relative_delta"] = result[f"{prefix}_added_limit"] / result["limit_count"].replace(0, np.nan)
    verified_any = result[[
        "verified_added_limit", "verified_added_failed", "verified_added_down_limit"
    ]].sum(axis=1) > 0
    upper_any = result[["upper_added_limit", "upper_added_failed", "upper_added_down_limit"]].sum(axis=1) > 0
    height = result["verified_revised_max_board"] > result["max_board"]
    breadth = result["verified_limit_relative_delta"] >= 0.05
    loss = result["verified_added_down_limit"] >= 2
    result["sensitivity_class"] = np.select(
        [height, breadth | loss, verified_any, upper_any],
        ["VERIFIED_HEIGHT_CHANGE", "VERIFIED_MATERIAL", "VERIFIED_MINOR", "SSE_UPPER_BOUND_ONLY"],
        default="NO_OBSERVED_IMPACT",
    )
    keep = [
        "date", "emotion_state", "limit_count", "consecutive_count", "max_board", "failed_count",
        "down_limit_count", "failed_rate",
    ] + addition_columns + [
        "verified_revised_limit_count", "verified_revised_failed_count", "verified_revised_down_limit_count",
        "verified_revised_consecutive_count", "verified_revised_max_board", "verified_revised_failed_rate",
        "verified_limit_relative_delta", "upper_revised_limit_count", "upper_revised_failed_count",
        "upper_revised_down_limit_count", "upper_revised_consecutive_count", "upper_revised_max_board",
        "upper_revised_failed_rate", "upper_limit_relative_delta", "sensitivity_class",
    ]
    return result[keep]


def report_text(metadata: dict[str, Any], height_rows: pd.DataFrame) -> str:
    lines = [
        "# V4 退市证券遗漏与情绪敏感性审计", "",
        f"- 结论：`{metadata['status']}`",
        f"- 退市代码行情恢复：{metadata['recovered_codes']}/{metadata['missing_codes']}（{metadata['bar_coverage']:.1%}）",
        f"- 时点状态已核验：{metadata['status_verified_codes']}/{metadata['missing_codes']}（{metadata['status_coverage']:.1%}）",
        f"- 全行情区间均已核验代码：{metadata['fully_status_verified_codes']}/{metadata['missing_codes']}；已核验行情行覆盖：{metadata['status_verified_bar_coverage']:.1%}",
        f"- 沪深已核验新增2—5板观察：{metadata['verified_candidate_rows']}条，{metadata['verified_candidate_codes']}个代码，{metadata['verified_candidate_dates']}个日期",
        f"- 已核验情绪宽度受影响：{metadata['verified_affected_dates']}个日期；涨停数低估至少5%：{metadata['verified_breadth_material_dates']}个日期",
        f"- 已核验空间高度变化：{metadata['verified_height_change_dates']}个日期",
        f"- 沪市状态仍未核验的上界候选：{metadata['sse_upper_candidate_rows']}条；不得并入正式样本。",
        f"- 沪市已确认ST/退市风险而剔除的伪10%连板：{metadata['sse_rejected_status_candidate_rows']}条。", "",
        "## 已核验空间高度变化", "", "| 日期 | 原情绪 | 原高度 | 补全高度 | 新增涨停 |", "|---|---|---:|---:|---:|",
    ]
    for row in height_rows.itertuples(index=False):
        lines.append(
            f"| {pd.Timestamp(row.date).strftime('%Y-%m-%d')} | {row.emotion_state} | {int(row.max_board)} | "
            f"{int(row.verified_revised_max_board)} | {int(row.verified_added_limit)} |"
        )
    lines.extend(["", "说明：本报告不重写V3冻结标签。元数据推断区间不进入正式样本；当前结果用于量化原情绪统计对幸存者偏差的敏感程度。", ""])
    return "\n".join(lines)


def run(bars_path: Path = BARS, market_path: Path = MARKET, output: Path = OUTPUT,
        original_candidates_path: Path = ORIGINAL_CANDIDATES,
        security_master_path: Path = SECURITY_MASTER) -> dict[str, Any]:
    bars = prepare_bars(bars_path, security_master_path)
    market = pd.read_csv(market_path)
    sensitivity = build_sensitivity(bars, market)
    candidates = candidate_rows(bars)
    verified_candidates = candidates[candidates["scope"].eq("VERIFIED_POINT_IN_TIME")]
    upper_candidates = candidates[candidates["scope"].eq("SSE_UNRESOLVED_UPPER_BOUND")]
    verified_any = sensitivity[[
        "verified_added_limit", "verified_added_failed", "verified_added_down_limit"
    ]].sum(axis=1) > 0
    height = sensitivity["verified_revised_max_board"] > sensitivity["max_board"]
    output.mkdir(parents=True, exist_ok=True)
    daily_path = output / "emotion_survivorship_sensitivity_daily.csv"
    candidate_path = output / "recovered_board_candidates.csv"
    report_path = output / "emotion_survivorship_sensitivity.md"
    manifest_path = output / "emotion_survivorship_sensitivity.json"
    sensitivity.to_csv(daily_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    original = pd.read_csv(
        original_candidates_path, dtype={"code": "string"},
        usecols=["date", "code", "name", "boards"], low_memory=False,
    )
    recovered = verified_candidates[["date", "code", "name", "boards"]].copy()
    recovered["date"] = pd.to_datetime(recovered["date"]).dt.strftime("%Y-%m-%d")
    original["date"] = pd.to_datetime(original["date"]).dt.strftime("%Y-%m-%d")
    combined_candidates = pd.concat([original, recovered], ignore_index=True)
    if combined_candidates.duplicated(["date", "code"]).any():
        raise ValueError("combined candidate universe contains duplicate date/code")
    combined_candidates["universe_complete"] = False
    COMBINED_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    combined_candidates.to_csv(COMBINED_CANDIDATES, index=False, encoding="utf-8-sig")
    recovered_codes = int(bars["code"].nunique())
    point_status_verified = (
        bars["status_basis"].eq("VERIFIED_SZSE_DATED_NAME_HISTORY")
        | bars["sse_status_verified"]
    )
    verified_codes = int(bars.loc[point_status_verified, "code"].nunique())
    fully_verified_codes = int(point_status_verified.groupby(bars["code"]).all().sum())
    rejected_sse = bars[
        bars["exchange"].eq("SSE") & bars["sse_status_verified"] & ~bars["sse_verified_eligible"]
        & bars["upper_boards"].between(2, 5)
    ]
    metadata = {
        "version": VERSION, "status": "MATERIAL_SURVIVORSHIP_SENSITIVITY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "missing_codes": 84, "recovered_codes": recovered_codes, "bar_coverage": recovered_codes / 84,
        "status_verified_codes": verified_codes, "status_coverage": verified_codes / 84,
        "fully_status_verified_codes": fully_verified_codes,
        "status_verified_bar_rows": int(point_status_verified.sum()),
        "status_verified_bar_coverage": float(point_status_verified.mean()),
        "verified_candidate_rows": int(len(verified_candidates)),
        "verified_candidate_codes": int(verified_candidates["code"].nunique()),
        "verified_candidate_dates": int(verified_candidates["date"].nunique()),
        "original_candidate_rows": int(len(original)),
        "combined_candidate_rows": int(len(combined_candidates)),
        "sse_upper_candidate_rows": int(len(upper_candidates)),
        "sse_upper_candidate_codes": int(upper_candidates["code"].nunique()),
        "sse_rejected_status_candidate_rows": int(len(rejected_sse)),
        "verified_affected_dates": int(verified_any.sum()),
        "verified_breadth_material_dates": int((sensitivity["verified_limit_relative_delta"] >= 0.05).sum()),
        "verified_height_change_dates": int(height.sum()),
        "verified_added_limit_days": int(sensitivity["verified_added_limit"].sum()),
        "verified_added_failed_days": int(sensitivity["verified_added_failed"].sum()),
        "verified_added_down_limit_days": int(sensitivity["verified_added_down_limit"].sum()),
        "max_verified_daily_limit_addition": int(sensitivity["verified_added_limit"].max()),
        "max_verified_limit_relative_delta": float(sensitivity["verified_limit_relative_delta"].max()),
        "deployment_effect": "V3 remains DIAGNOSTIC_ONLY; recovered verified rows may enter a new challenger rebuild only",
        "frozen_v3_modified": False,
        "files": {},
    }
    report_path.write_text(report_text(metadata, sensitivity[height]), encoding="utf-8")
    for key, path in {
        "daily": daily_path, "candidates": candidate_path, "combined_candidates": COMBINED_CANDIDATES,
        "report": report_path,
    }.items():
        metadata["files"][key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    metadata["input_hashes"] = {
        "bars": sha256_file(bars_path), "market": sha256_file(market_path),
        "security_master": sha256_file(security_master_path),
    }
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["manifest"] = str(manifest_path.resolve())
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit emotion sensitivity to recovered delisted securities")
    parser.add_argument("--bars", type=Path, default=BARS)
    parser.add_argument("--market", type=Path, default=MARKET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--original-candidates", type=Path, default=ORIGINAL_CANDIDATES)
    parser.add_argument("--security-master", type=Path, default=SECURITY_MASTER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(
        run(args.bars, args.market, args.output, args.original_candidates, args.security_master),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
