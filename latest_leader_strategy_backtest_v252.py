"""Backtest the latest leader strategy with strict data gating and a fixed proxy.

The V246/V249/V250 documents require auction/VWAP/theme/real-fill evidence.  This
script does not impute those fields.  It reports the strict strategy (CASH when
required evidence is missing) and a separately labelled structural proxy using
only fields present in the point-in-time candidate file.  Promotion is the
identity target; net_return is an inferred daily-OHLC proxy, never a fill claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "v252-latest-leader-strategy-backtest-1"
REQUIRED_LATEST = {
    "auction_gap", "auction_amount_ratio", "expectation_gap",
    "vwap_slope_5m", "vwap_slope_15m", "vwap_reclaimed",
    "theme_support", "theme_middle_support", "theme_first_support",
    "theme_sync_open", "rivals_eliminated", "below_vwap_expanding",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate(s: pd.Series) -> float | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return None if x.empty else float(x.mean())


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [float(max(0, ctr-half)), float(min(1, ctr+half))]


def summarize(rows: pd.DataFrame, label: str) -> dict:
    n = len(rows)
    promo = pd.to_numeric(rows.get("promoted_next_day"), errors="coerce")
    net = pd.to_numeric(rows.get("net_return"), errors="coerce")
    known_net = net.notna()
    return {
        "label": label,
        "rows": int(n),
        "dates": int(rows.date.nunique()) if n else 0,
        "promotion_known": int(promo.notna().sum()),
        "promotion_rate": rate(promo),
        "promotion_wilson_95": wilson(int(promo.sum()), int(promo.notna().sum())),
        "proxy_net_known": int(known_net.sum()),
        "proxy_net_positive_rate": rate((net[known_net] > 0).astype(float)) if known_net.any() else None,
        "proxy_mean_net_return": float(net[known_net].mean()) if known_net.any() else None,
        "proxy_large_loss_rate": rate(rows.loc[known_net, "large_loss"]) if known_net.any() else None,
        "actual_fill_rows": int(rows.execution_status.eq("actual_fill").sum()) if "execution_status" in rows else 0,
        "inferred_touch_rows": int(rows.execution_status.eq("inferred_daily_limit_touch_fill").sum()) if "execution_status" in rows else 0,
    }


def proxy_pool(frame: pd.DataFrame) -> pd.DataFrame:
    # Frozen structural approximation of V246/V249/V250.  No outcome field is
    # used: candidate must be >=3 boards, non-one-word, no known hard warning,
    # and have a resolved identity.  Theme/VWAP gates remain untestable.
    x = frame[
        frame.boards.ge(3)
        & frame.buyable_eod.eq(1)
        & frame.one_word_proxy.eq(0)
        & frame.valid_identity.eq(1)
        & frame.announcement_risk.eq(0)
        & frame.false_acceleration.eq(0)
        & frame.emotion_risk_level.isin(["LOW", "MEDIUM"])
    ].copy()
    # One frozen candidate per day: highest board, then identity and space
    # leadership.  All are EOD fields, so this is a close-time diagnostic.
    return x


def choose_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    x = proxy_pool(frame)
    x = x.sort_values(
        ["date", "boards", "identity_score", "height_share", "is_space_leader", "turnover"],
        ascending=[True, False, False, False, False, False],
    ).drop_duplicates("date", keep="first")
    return x


def choose_proxy_height_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Highest board within exactly the proxy's eligible denominator."""
    x = proxy_pool(frame).sort_values(
        ["date", "boards", "height_share", "is_space_leader", "identity_score"],
        ascending=[True, False, False, False, False],
    ).drop_duplicates("date", keep="first")
    return x


def choose_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[frame.boards.ge(3)].copy()
    return x.sort_values(
        ["date", "boards", "is_space_leader", "height_share", "identity_score"],
        ascending=[True, False, False, False, False],
    ).drop_duplicates("date", keep="first")


def run(data_path: str, out_dir: str) -> dict:
    src = Path(data_path)
    usecols = None
    df = pd.read_csv(src, dtype={"code": str}, low_memory=False, usecols=usecols)
    df["date"] = df.date.astype(str)
    numeric = [
        "boards", "buyable_eod", "one_word_proxy", "valid_identity", "announcement_risk",
        "false_acceleration", "emotion_risk_level", "promoted_next_day", "net_return",
        "large_loss", "identity_score", "height_share", "is_space_leader", "turnover",
    ]
    for c in numeric:
        if c != "emotion_risk_level":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # `unavailable_fields` is the authoritative availability audit in the file.
    unavailable = set()
    for value in df.unavailable_fields.dropna().astype(str):
        unavailable.update(v.strip() for v in value.split(",") if v.strip())
    missing_required = sorted(REQUIRED_LATEST & unavailable)
    strict = pd.DataFrame(columns=df.columns)
    proxy = choose_proxy(df)
    proxy_height = choose_proxy_height_baseline(df)
    baseline = choose_baseline(df)

    all_dates = sorted(df[df.boards.ge(3)].date.unique())
    split = all_dates[max(1, int(len(all_dates)*.7))] if all_dates else None
    for x in (proxy, proxy_height, baseline):
        x["time_segment"] = np.where(x.date < split, "DISCOVERY", "VALIDATION") if split else "ALL"

    def segments(x: pd.DataFrame, name: str) -> list[dict]:
        out = [summarize(x, name + ":ALL")]
        for seg in ("DISCOVERY", "VALIDATION"):
            out.append(summarize(x[x.time_segment.eq(seg)], name + ":" + seg))
        return out

    strict_doc = {
        "strategy": "V246/V249/V250 strict",
        "decision": "CASH",
        "rows": 0,
        "dates": 0,
        "reason": "required auction/VWAP/theme evidence is unavailable; V249 says missing data is CASH",
    }
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "source_cutoff": str(df.date.max()) if len(df) else None,
        "source_rows": int(len(df)),
        "source_dates": int(df.date.nunique()),
        "strict_latest": strict_doc,
        "required_latest_fields": sorted(REQUIRED_LATEST),
        "missing_required_fields_in_unavailable_audit": missing_required,
        "proxy_definition": {
            "boards_min": 3,
            "buyable_eod": 1,
            "one_word_proxy": 0,
            "valid_identity": 1,
            "announcement_risk": 0,
            "false_acceleration": 0,
            "emotion_risk_level": ["LOW", "MEDIUM"],
            "selection": "one per date; boards, identity_score, height_share, space leadership, turnover descending",
            "not_implemented": ["auction confirmation", "09:35 VWAP", "theme ladder", "actual fills", "intraday path"],
        },
        "retrospective_split": {
            "method": "fixed chronological 70/30 by date; no threshold tuning",
            "split_date": split,
            "is_blind_forward_test": False,
        },
        "metrics": (segments(proxy, "LATEST_STRUCTURAL_PROXY")
                    + segments(proxy_height, "PROXY_DENOMINATOR_HEIGHT_BASELINE")
                    + segments(baseline, "SAME_DAY_HIGHEST_BOARD_BASELINE")),
        "comparison": {
            "validation_promotion_difference_proxy_minus_baseline": (
                float(proxy[proxy.time_segment.eq("VALIDATION")].promoted_next_day.mean())
                - float(baseline[baseline.time_segment.eq("VALIDATION")].promoted_next_day.mean())
                if len(proxy[proxy.time_segment.eq("VALIDATION")]) and len(baseline[baseline.time_segment.eq("VALIDATION")]) else None
            ),
            "validation_promotion_difference_proxy_minus_same_denominator_height": (
                float(proxy[proxy.time_segment.eq("VALIDATION")].promoted_next_day.mean())
                - float(proxy_height[proxy_height.time_segment.eq("VALIDATION")].promoted_next_day.mean())
                if len(proxy[proxy.time_segment.eq("VALIDATION")]) and len(proxy_height[proxy_height.time_segment.eq("VALIDATION")]) else None
            ),
            "proxy_uses_inferred_daily_ohlc_not_real_fill": True,
            "actual_fill_rows_proxy": int(proxy.execution_status.eq("actual_fill").sum()),
        },
        "conclusions": {
            "latest_strategy_fully_backtested": False,
            "strict_rule_trade_count": 0,
            "latest_rules_can_be_claimed_as_win_rate": False,
            "promotion_proxy_validated": False,
            "entry_authorized": False,
        },
        "blocking_reasons": [
            "历史数据的 unavailable_fields 对全部样本都包含竞价、VWAP和题材同步字段",
            "没有真实盘口成交路径，net_return 只能作为日线触板/收盘代理",
            "V250 的 A/B/C 龙型标签尚未落成结构化历史字段",
            "晋级率与买后净收益不是同一个目标，不能用晋级代理替代 V249 的 B1/ANTI_NUCLEAR",
        ],
        "source_hashes": {str(src.resolve()): sha(src)},
    }
    target = Path(out_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.mkdir(parents=True)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    proxy.to_csv(target / "latest_structural_proxy_selected.csv", index=False, encoding="utf-8-sig")
    proxy_height.to_csv(target / "proxy_denominator_height_baseline_selected.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(target / "highest_board_baseline_selected.csv", index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.data, ns.output), ensure_ascii=False, indent=2))
