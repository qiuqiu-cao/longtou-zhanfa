"""Audit whether historical data can validate V242 path expectations.

The audit deliberately refuses to reconstruct QUICK_REPAIR/LOCKED/PROLONGED
from zero-filled or missing intraday fields.  It reports only supported
one-price versus non-one-price proxies and retrospective next-open band
sensitivity.  Next-open bands are observations available after the auction,
not close-time ranking features and not same-open executable signals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v245-relay-path-history-audit-1"
REQUIRED = {
    "date", "code", "boards", "promoted_next_day", "promotion_label_status",
    "next_open_gap", "one_word_proxy", "first_seal_minute", "last_seal_minute",
    "bomb_count", "turnover", "execution_status", "net_return", "large_loss",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _summarize(frame: pd.DataFrame, label: str, segment: str,
               boards: int | None = None) -> dict:
    selected = frame if boards is None else frame[frame.boards.eq(boards)]
    filled = selected.net_return.notna()
    return {
        "segment": segment, "slice": label, "boards": boards,
        "rows": int(len(selected)), "date_clusters": int(selected.date.nunique()),
        "promotion_rate": _rate(selected.promoted_next_day),
        "mean_next_open_gap_pct": _mean(selected.next_open_gap_pct),
        "observed_net_return_rows": int(filled.sum()),
        "mean_net_return_observed_only": _mean(selected.loc[filled, "net_return"]),
        "big_loss_rate_observed_only": _rate(selected.loc[filled, "large_loss"]),
        "same_open_execution_claimed": False,
    }


def _cluster_bootstrap_gap_band(frame: pd.DataFrame, segment: str, boards: int,
                                reps: int = 2000, seed: int = 245) -> dict:
    sample = frame[frame.boards.eq(boards) & frame.one_word_proxy.eq(0)].copy()
    old = sample[sample.next_open_gap_pct.between(0, 5)]
    challenger = sample[sample.next_open_gap_pct.between(3, 7)]
    dates = sorted(set(old.date) | set(challenger.date))
    base = {"segment": segment, "boards": boards,
            "comparison": "V242_3_TO_7_MINUS_OLD_0_TO_5_PROMOTION_RATE",
            "old_rows": len(old), "challenger_rows": len(challenger),
            "date_clusters": len(dates), "difference": None,
            "cluster_ci_low": None, "cluster_ci_high": None,
            "interval_excludes_zero": False}
    if not dates or old.empty or challenger.empty:
        return base
    grouped_old = old.groupby("date").promoted_next_day.agg(["sum", "count"])
    grouped_new = challenger.groupby("date").promoted_next_day.agg(["sum", "count"])
    old_sum = np.array([grouped_old.loc[d, "sum"] if d in grouped_old.index else 0 for d in dates])
    old_count = np.array([grouped_old.loc[d, "count"] if d in grouped_old.index else 0 for d in dates])
    new_sum = np.array([grouped_new.loc[d, "sum"] if d in grouped_new.index else 0 for d in dates])
    new_count = np.array([grouped_new.loc[d, "count"] if d in grouped_new.index else 0 for d in dates])
    observed = float(new_sum.sum() / new_count.sum() - old_sum.sum() / old_count.sum())
    rng = np.random.default_rng(seed + boards + len(dates))
    values = []
    for _ in range(reps):
        picks = rng.integers(0, len(dates), len(dates))
        old_n, new_n = old_count[picks].sum(), new_count[picks].sum()
        if old_n and new_n:
            values.append(new_sum[picks].sum() / new_n - old_sum[picks].sum() / old_n)
    low, high = (np.quantile(values, [.025, .975]) if values else (np.nan, np.nan))
    base.update({"difference": observed,
                 "cluster_ci_low": None if np.isnan(low) else float(low),
                 "cluster_ci_high": None if np.isnan(high) else float(high),
                 "interval_excludes_zero": bool(low > 0 or high < 0) if values else False})
    return base


def audit(data: pd.DataFrame, v242: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    missing = sorted(REQUIRED - set(data.columns))
    if missing: raise ValueError(f"historical dataset missing fields: {missing}")
    rows = data.copy(); rows["code"] = rows.code.astype(str).str.zfill(6)
    rows["date"] = rows.date.astype(str)
    rows["boards"] = pd.to_numeric(rows.boards, errors="coerce")
    universe = rows[rows.boards.isin([3, 4])].copy()
    for field in ("promoted_next_day", "next_open_gap", "one_word_proxy",
                  "first_seal_minute", "last_seal_minute", "bomb_count",
                  "turnover", "net_return", "large_loss"):
        universe[field] = pd.to_numeric(universe[field], errors="coerce")
    labeled = universe[
        universe.promoted_next_day.notna() & universe.next_open_gap.notna()
        & universe.promotion_label_status.eq("verified_from_next_daily_limit_close")
    ].copy()
    labeled["next_open_gap_pct"] = labeled.next_open_gap * 100

    path_fields = universe[["first_seal_minute", "last_seal_minute", "bomb_count"]]
    any_intraday = path_fields.notna().all(axis=1)
    duration = universe.last_seal_minute - universe.first_seal_minute
    global_zero_placeholder = bool(
        any_intraday.any()
        and universe.loc[any_intraday, "first_seal_minute"].fillna(0).eq(0).all()
        and universe.loc[any_intraday, "last_seal_minute"].fillna(0).eq(0).all()
        and universe.loc[any_intraday, "bomb_count"].fillna(0).eq(0).all()
    )
    path_usable = any_intraday & (not global_zero_placeholder)
    universe["path_evidence_status"] = np.where(
        universe.one_word_proxy.eq(1), "ONE_PRICE_PROXY_ONLY",
        np.where(path_usable, "INTRADAY_PATH_FIELDS_PRESENT",
                 "NON_ONE_PRICE_PATH_UNRESOLVED"))
    universe["path_duration_minutes"] = duration

    unique_dates = sorted(labeled.date.unique())
    split_index = max(1, int(len(unique_dates) * .7)) if unique_dates else 0
    split_date = unique_dates[split_index] if split_index < len(unique_dates) else None
    if split_date is None:
        labeled["time_segment"] = "DISCOVERY"
    else:
        labeled["time_segment"] = np.where(labeled.date < split_date,
                                            "DISCOVERY", "VALIDATION")

    summaries = []
    for segment in ("DISCOVERY", "VALIDATION", "ALL"):
        sample = labeled if segment == "ALL" else labeled[labeled.time_segment.eq(segment)]
        one_price = sample[sample.one_word_proxy.eq(1)]
        non_one = sample[sample.one_word_proxy.eq(0)]
        bands = {
            "ONE_PRICE_PROXY_IDENTITY_ONLY": one_price,
            "NON_ONE_PRICE_ALL_PATH_UNRESOLVED": non_one,
            "OLD_UNIVERSAL_0_TO_5_LABEL_ONLY": non_one[non_one.next_open_gap_pct.between(0, 5)],
            "V242_LOCKED_3_TO_7_LABEL_ONLY": non_one[non_one.next_open_gap_pct.between(3, 7)],
            "V242_DIVERGENCE_0_TO_5_LABEL_ONLY": non_one[non_one.next_open_gap_pct.between(0, 5)],
            "V242_HIGH_TURNOVER_0_TO_4_LABEL_ONLY": non_one[
                non_one.next_open_gap_pct.between(0, 4) & non_one.turnover.gt(25)],
        }
        for label, frame in bands.items():
            for board in (3, 4):
                summaries.append(_summarize(frame, label, segment, board))
    metrics = pd.DataFrame(summaries)
    comparisons = []
    for segment in ("DISCOVERY", "VALIDATION", "ALL"):
        sample = labeled if segment == "ALL" else labeled[labeled.time_segment.eq(segment)]
        for board in (3, 4):
            comparisons.append(_cluster_bootstrap_gap_band(sample, segment, board))
    comparisons = pd.DataFrame(comparisons)
    coverage = pd.DataFrame([
        {"field_group": "FULL_3_4_BOARD_DENOMINATOR", "rows": len(universe),
         "date_clusters": universe.date.nunique(), "status": "AVAILABLE"},
        {"field_group": "VERIFIED_PROMOTION_AND_NEXT_OPEN", "rows": len(labeled),
         "date_clusters": labeled.date.nunique(), "status": "AVAILABLE_LABEL_ONLY"},
        {"field_group": "ONE_PRICE_PROXY", "rows": int(universe.one_word_proxy.notna().sum()),
         "date_clusters": universe[universe.one_word_proxy.notna()].date.nunique(),
         "status": "AVAILABLE_PROXY"},
        {"field_group": "USABLE_FIRST_LAST_SEAL_AND_FAILED_SEAL_PATH",
         "rows": int(path_usable.sum()),
         "date_clusters": int(universe.loc[path_usable, "date"].nunique()),
         "status": ("AVAILABLE" if path_usable.any()
                    else "UNAVAILABLE_ZERO_PLACEHOLDER_OR_MISSING")},
        {"field_group": "0935_1000_VWAP_SEAL_THEME_STAGE", "rows": 0,
         "date_clusters": 0, "status": "UNAVAILABLE"},
    ])
    one_price_labeled = labeled[labeled.one_word_proxy.eq(1)]
    unbuyable = one_price_labeled.execution_status.eq("unbuyable_one_word_up")
    report = {
        "version": VERSION,
        "status": "PATH_EXPECTATION_VALIDATION_BLOCKED_BY_INTRADAY_COVERAGE",
        "source_cutoff": str(universe.date.max()) if not universe.empty else None,
        "full_3_4_board_rows": int(len(universe)),
        "full_date_clusters": int(universe.date.nunique()),
        "promotion_and_open_labeled_rows": int(len(labeled)),
        "promotion_and_open_date_clusters": int(labeled.date.nunique()),
        "global_zero_intraday_placeholder_detected": global_zero_placeholder,
        "usable_full_path_rows": int(path_usable.sum()),
        "one_price_proxy_rows": int(universe.one_word_proxy.eq(1).sum()),
        "one_price_labeled_promotion_rate": _rate(one_price_labeled.promoted_next_day),
        "one_price_next_session_unbuyable_rate": (
            float(unbuyable.mean()) if len(one_price_labeled) else None),
        "retrospective_split": {
            "method": "FIXED_70_30_BY_DATE_NO_THRESHOLD_TUNING",
            "split_date": split_date,
            "is_blind_forward_test": False,
            "community_examples_may_overlap_history": True,
        },
        "conclusions": {
            "v242_path_thresholds_validated": False,
            "v243_stage_chain_validated": False,
            "only_one_price_vs_non_one_price_proxy_supported": True,
            "next_open_band_is_close_time_feature": False,
            "same_open_execution_allowed": False,
            "historical_band_metrics_are_sensitivity_only": True,
            "rule_upgrade_allowed": False,
        },
        "validation_gap_band_comparisons": comparisons[
            comparisons.segment.eq("VALIDATION")].to_dict("records"),
        "blocking_reasons": [
            "3/4板历史首次封板、最后封板和炸板次数为缺失或零占位，无法区分快速修复、锁板和长争夺",
            "缺少09:35与10:00的VWAP、封板稳定性和题材带动完整分母",
            "次日开盘涨幅是竞价结束后信息，不能作为收盘排序特征或保留同一开盘价成交",
            "社区案例可能与历史区间重叠，70/30切分仍不是独立前向盲测",
        ],
        "required_next_evidence": [
            "至少60个独立信号日的全部3/4板分钟路径",
            "每只候选09:25、09:35、10:00时点证据及完整题材梯队",
            "按10:00后可成交价格重算B+1至B+3收益、大面率和最大回撤",
        ],
        "entry_authorized": False,
    }
    return coverage, metrics, comparisons, report


def run(data_path: str, v242_report_path: str, output_dir: str) -> dict:
    data_file, report_file = Path(data_path), Path(v242_report_path)
    v242 = json.loads(report_file.read_text(encoding="utf-8-sig"))
    if v242.get("version") != "v242-relay-path-expectation-1":
        raise ValueError("V242 report version mismatch")
    data = pd.read_csv(data_file, dtype={"code": str}, low_memory=False)
    coverage, metrics, comparisons, report = audit(data, v242)
    target = Path(output_dir)
    if target.exists(): raise FileExistsError("refusing to overwrite immutable V245 output")
    target.mkdir(parents=True)
    coverage_path, metrics_path = target / "coverage.csv", target / "band_sensitivity.csv"
    comparisons_path = target / "cluster_bootstrap_band_comparisons.csv"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    comparisons.to_csv(comparisons_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(data_file.resolve()): _sha(data_file),
                               str(report_file.resolve()): _sha(report_file)}
    report["artifact_hashes"] = {coverage_path.name: _sha(coverage_path),
                                 metrics_path.name: _sha(metrics_path),
                                 comparisons_path.name: _sha(comparisons_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True); ap.add_argument("--v242-report", required=True)
    ap.add_argument("--output", required=True); ns = ap.parse_args()
    print(json.dumps(run(ns.data, ns.v242_report, ns.output), ensure_ascii=False, indent=2))
