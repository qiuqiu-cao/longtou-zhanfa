"""Compute current provider-aligned severe-abnormal arithmetic diagnostics.

This is not an official regulatory distance.  It assumes no undiscovered
exchange-publication reset and uses provider-unadjusted daily closes without
corporate-action reconciliation.  Those unresolved items are explicit hard
blockers for official distance, probability, stock reranking, and entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

from abnormal_distance_v18 import project_limit_path, window_summary
from severe_abnormal_rule_registry_v145 import verified_rule_for_code


VERSION = "v191-current-unreconciled-abnormal-distance-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _rounded_ten_percent_returns(close: float, steps: int = 5) -> list[float]:
    price = Decimal(str(close))
    returns = []
    for _ in range(steps):
        limit_price = (price * Decimal("1.10")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP)
        returns.append(float((limit_price / price - 1) * 100))
        price = limit_price
    return returns


def _v171_compat(frame: pd.DataFrame, available_at: str) -> pd.DataFrame:
    """Export complete attached evidence while preserving non-verified status."""
    result = frame.copy()
    result["available_at"] = available_at
    result["coverage_complete"] = result[
        "provider_aligned_30_return_coverage"].fillna(False)
    result["indicator_status"] = np.where(
        result["research_indicator_status"].eq("inferred_unreconciled"),
        "inferred", "unavailable")
    result["security_status"] = result["security_eligibility_status"].fillna("unavailable")
    result["scenario_limit_boards_to_100"] = result[
        "flat_benchmark_rounded_limit_first_hit_step"]
    result["active_theme_path_status"] = "unavailable"
    result["competition_identity_status"] = "unavailable"
    result["competition_identity"] = "unavailable"
    result["theme_support_ex_self"] = np.nan
    result["same_height_rivals"] = np.nan
    columns = [
        "signal_date", "code", "available_at", "rule_id", "rule_status",
        "benchmark_code", "benchmark_status", "reset_status",
        "corporate_action_status", "security_status", "coverage_complete",
        "indicator_status", "distance_100_pp", "distance_200_pp",
        "scenario_limit_boards_to_100", "scenario_status",
        "active_theme_path_status", "competition_identity_status",
        "competition_identity", "theme_support_ex_self", "same_height_rivals",
    ]
    return result[columns]


def _prepare_benchmark(frame: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    required = {"date", "close", "data_status"}
    if required - set(frame):
        raise ValueError("benchmark fields missing")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.strftime("%Y-%m-%d")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result[result["date"] <= signal_date].sort_values("date")
    if result.empty or result["date"].duplicated().any() or result["close"].isna().any():
        raise ValueError("invalid benchmark history")
    return result


def _candidate_row(row: pd.Series, stock: pd.DataFrame,
                   benchmark: pd.DataFrame, signal_date: str) -> tuple[dict, list[dict]]:
    code = str(row["code"]).zfill(6)
    rule = verified_rule_for_code(code, signal_date)
    base = {
        "signal_date": signal_date, "code": code, "name": row.get("name"),
        "boards": int(row.get("boards")), "board_group": row.get("board_group"),
        "exchange": rule["exchange"], "rule_id": rule["rule_id"],
        "rule_status": rule["verification_status"],
        "benchmark_code": rule["benchmark_code"],
        "benchmark_status": "provider_observed_not_official_reconciled",
        "reset_status": "unavailable",
        "corporate_action_status": "unavailable",
        "historical_price_limit_status": "unavailable",
        "security_eligibility_status": row.get("eligibility_status", "unavailable"),
        "regulatory_distance_status": "unavailable",
        "research_indicator_status": "unavailable",
        "research_assumption": (
            "provider_closes_no_exchange_reset_no_corporate_action_reconciliation"),
        "ranking_allowed": False, "buy_signal": False,
        "hundred_percent_path_potential": None,
    }
    candidate = stock[stock["code"].astype(str).str.zfill(6).eq(code)].copy()
    if candidate.empty:
        base["unavailable_reason"] = "STOCK_HISTORY_MISSING"
        return base, []
    candidate["date"] = pd.to_datetime(candidate["date"], errors="raise").dt.strftime("%Y-%m-%d")
    candidate["close"] = pd.to_numeric(candidate["close"], errors="coerce")
    candidate = candidate[candidate["date"] <= signal_date].sort_values("date")
    if candidate["date"].duplicated().any() or candidate["close"].isna().any():
        base["unavailable_reason"] = "INVALID_STOCK_HISTORY"
        return base, []
    base["stock_provider"] = str(candidate.iloc[-1].get("provider", ""))
    base["stock_data_status"] = str(candidate.iloc[-1].get("data_status", ""))
    base["price_basis"] = str(candidate.iloc[-1].get("price_basis", ""))
    base["latest_stock_date"] = str(candidate.iloc[-1]["date"])
    base["latest_close"] = float(candidate.iloc[-1]["close"])
    pool_price = pd.to_numeric(pd.Series([row.get("last_price")]), errors="coerce").iloc[0]
    base["pool_last_price"] = None if pd.isna(pool_price) else float(pool_price)
    base["pool_close_consistent"] = bool(
        not pd.isna(pool_price) and abs(float(pool_price) - base["latest_close"]) < .0051)

    calendar = benchmark.tail(31)[["date", "close", "data_status"]].rename(
        columns={"close": "benchmark_close", "data_status": "benchmark_data_status"})
    aligned = calendar.merge(
        candidate[["date", "close"]], on="date", how="left", validate="one_to_one")
    base["benchmark_calendar_rows"] = int(len(calendar))
    base["aligned_close_rows"] = int(aligned["close"].notna().sum())
    base["latest_benchmark_date"] = str(benchmark.iloc[-1]["date"])
    coverage = bool(
        len(aligned) == 31 and aligned["close"].notna().all()
        and str(aligned.iloc[-1]["date"]) == signal_date
        and base["latest_stock_date"] == signal_date
        and base["pool_close_consistent"]
        and aligned["benchmark_data_status"].eq(
            "provider_observed_not_official_reconciled").all()
        and base["price_basis"] == "provider_unadjusted"
    )
    base["provider_aligned_30_return_coverage"] = coverage
    if not coverage:
        base["unavailable_reason"] = "PROVIDER_CALENDAR_OR_CLOSE_COVERAGE_INCOMPLETE"
        return base, []
    aligned["stock_return_pct"] = aligned["close"].pct_change(fill_method=None) * 100
    aligned["benchmark_return_pct"] = aligned["benchmark_close"].pct_change(fill_method=None) * 100
    returns = aligned.iloc[1:].copy()
    outliers = returns["stock_return_pct"].abs().gt(20.5)
    base["raw_return_over_20_5pct_count"] = int(outliers.sum())
    if outliers.any():
        base["unavailable_reason"] = "POSSIBLE_NO_LIMIT_OR_CORPORATE_ACTION_SESSION"
        return base, []
    stock_returns = returns["stock_return_pct"].tolist()
    benchmark_returns = returns["benchmark_return_pct"].tolist()
    ten = window_summary(stock_returns, benchmark_returns, 10)
    thirty = window_summary(stock_returns, benchmark_returns, 30)
    base.update({
        "research_indicator_status": "inferred_unreconciled",
        "unavailable_reason": None,
        "trailing_10d_deviation_pct": ten["trailing_deviation_pct"],
        "maximum_10d_suffix_deviation_pct": ten["maximum_suffix_deviation_pct"],
        "maximum_10d_suffix_sessions": ten["maximum_suffix_sessions"],
        "trailing_30d_deviation_pct": thirty["trailing_deviation_pct"],
        "maximum_30d_suffix_deviation_pct": thirty["maximum_suffix_deviation_pct"],
        "maximum_30d_suffix_sessions": thirty["maximum_suffix_sessions"],
        "distance_100_pp": 100 - ten["maximum_suffix_deviation_pct"],
        "distance_200_pp": 200 - thirty["maximum_suffix_deviation_pct"],
        "research_threshold_breached": bool(
            ten["maximum_suffix_deviation_pct"] >= 100
            or thirty["maximum_suffix_deviation_pct"] >= 200),
    })
    base["normalized_arithmetic_headroom"] = min(
        base["distance_100_pp"] / 100, base["distance_200_pp"] / 200)
    future_stock = _rounded_ten_percent_returns(base["latest_close"], 5)
    scenario = project_limit_path(
        stock_returns, benchmark_returns, future_stock, [0.0] * 5)
    hits = scenario[scenario["scenario_threshold_hit"]]
    base["flat_benchmark_rounded_limit_first_hit_step"] = (
        None if hits.empty else int(hits.iloc[0]["future_session"]))
    base["scenario_status"] = "counterfactual_no_reset_flat_benchmark_not_forecast"
    scenario_rows = []
    for _, scenario_row in scenario.iterrows():
        scenario_rows.append({
            "signal_date": signal_date, "code": code,
            "future_session": int(scenario_row["future_session"]),
            "assumed_stock_return_pct": future_stock[
                int(scenario_row["future_session"]) - 1],
            "assumed_benchmark_return_pct": 0.0,
            **scenario_row.to_dict(),
        })
    return base, scenario_rows


def build(pool: pd.DataFrame, pool_manifest: dict, history: pd.DataFrame,
          input_manifest: dict, benchmarks: dict[str, pd.DataFrame],
          benchmark_manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    signal_date = str(pool_manifest.get("signal_date"))
    required = {
        "signal_date", "code", "name", "boards", "board_group",
        "pool_assignment", "is_mainboard", "last_price", "eligibility_status",
    }
    if required - set(pool):
        raise ValueError("pool fields missing")
    frame = pool.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    closing = frame[
        frame["signal_date"].astype(str).eq(signal_date)
        & frame["pool_assignment"].eq("CLOSING_LIMIT_UP")
        & frame["is_mainboard"].map(_truthy)
    ].copy()
    expected = int(pool_manifest.get("mainboard_closing_limit_rows", -1))
    if len(closing) != expected or closing["code"].duplicated().any():
        raise ValueError("pool does not match bound manifest")
    if (input_manifest.get("signal_date") != signal_date
            or int(input_manifest.get("candidate_count", -1)) != expected):
        raise ValueError("history input manifest is not bound to pool")
    if benchmark_manifest.get("requested_end") != signal_date:
        raise ValueError("benchmark manifest is not bound to signal date")
    prepared_benchmarks = {
        code: _prepare_benchmark(value, signal_date)
        for code, value in benchmarks.items()
    }
    rows, scenarios = [], []
    for _, row in closing.sort_values("code").iterrows():
        rule = verified_rule_for_code(str(row["code"]), signal_date)
        benchmark_code = rule["benchmark_code"]
        if benchmark_code not in prepared_benchmarks:
            raise ValueError(f"benchmark missing: {benchmark_code}")
        result, scenario_rows = _candidate_row(
            row, history, prepared_benchmarks[benchmark_code], signal_date)
        rows.append(result)
        scenarios.extend(scenario_rows)
    result = pd.DataFrame(rows)
    scenario_frame = pd.DataFrame(scenarios)
    inferred = result[result["research_indicator_status"].eq("inferred_unreconciled")]
    closest = inferred.sort_values(
        ["normalized_arithmetic_headroom", "boards"],
        ascending=[True, False]).head(10)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal_date, "candidate_count": int(len(result)),
        "provider_aligned_coverage_count": int(
            result["provider_aligned_30_return_coverage"].fillna(False).sum()),
        "research_distance_count": int(len(inferred)),
        "official_regulatory_distance_count": 0,
        "official_reset_evidence_count": 0,
        "corporate_action_reconciled_count": 0,
        "security_eligibility_complete": bool(
            pool_manifest.get("point_in_time_security_scope_complete") is True),
        "pool_close_consistency_count": int(
            result["pool_close_consistent"].fillna(False).sum()),
        "research_threshold_breach_count": int(
            inferred["research_threshold_breached"].fillna(False).sum()),
        "unreconciled_arithmetic_closest_top10": closest[[
            "code", "name", "boards", "distance_100_pp", "distance_200_pp",
            "flat_benchmark_rounded_limit_first_hit_step",
        ]].replace({np.nan: None}).to_dict("records"),
        "hundred_percent_abnormal_path_potential": None,
        "candidate_reranking_allowed": False, "buy_signal": False,
        "entry_authorized": False,
        "governance": {
            "research_arithmetic_may_be_displayed_with_unreconciled_label": True,
            "research_arithmetic_may_be_called_official_distance": False,
            "closest_distance_may_define_path_potential": False,
            "closest_distance_may_rank_entries": False,
            "scenario_is_forecast": False,
        },
        "interpretation": (
            "All available arithmetic is provider-aligned and assumes no undiscovered reset. "
            "It improves observability but does not establish official regulatory distance, "
            "leader-path potential, hit probability, stock ranking, or a buy signal."),
        "limitations": [
            "exchange-publication reset chronology unavailable",
            "corporate-action and historical no-limit eligibility unreconciled",
            "benchmark and stock closes are provider-observed rather than official examples",
            "flat-benchmark rounded-limit path is a counterfactual scenario, not a forecast",
        ],
    }
    return result, scenario_frame, report


def run(pool_path: str | Path, pool_manifest_path: str | Path,
        history_path: str | Path, input_manifest_path: str | Path,
        benchmark_dir: str | Path, output_dir: str | Path) -> dict:
    paths = [Path(x) for x in (
        pool_path, pool_manifest_path, history_path, input_manifest_path)]
    benchmark_dir, target = Path(benchmark_dir), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V191 distance archive")
    benchmark_paths = {
        code: benchmark_dir / f"{code}.csv" for code in ("sh000002", "sz399107")}
    benchmark_manifest_path = benchmark_dir / "manifest.json"
    result, scenarios, report = build(
        pd.read_csv(paths[0], dtype={"code": str}),
        json.loads(paths[1].read_text(encoding="utf-8")),
        pd.read_csv(paths[2], dtype={"code": str}),
        json.loads(paths[3].read_text(encoding="utf-8")),
        {code: pd.read_csv(path) for code, path in benchmark_paths.items()},
        json.loads(benchmark_manifest_path.read_text(encoding="utf-8")),
    )
    target.mkdir(parents=True, exist_ok=False)
    result_path = target / "candidate_unreconciled_distances.csv"
    scenario_path = target / "counterfactual_limit_paths.csv"
    compatible_path = target / "v171_unreconciled_evidence.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    scenarios.to_csv(scenario_path, index=False, encoding="utf-8-sig")
    _v171_compat(result, str(
        json.loads(paths[3].read_text(encoding="utf-8"))["collected_at"])
    ).to_csv(compatible_path, index=False, encoding="utf-8-sig")
    all_sources = [*paths, *benchmark_paths.values(), benchmark_manifest_path, Path(__file__)]
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path) for path in all_sources}
    report["distance_csv_sha256"] = _sha256(result_path)
    report["scenario_csv_sha256"] = _sha256(scenario_path)
    report["v171_unreconciled_evidence_sha256"] = _sha256(compatible_path)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--history-manifest", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.pool, args.pool_manifest, args.history,
                         args.history_manifest, args.benchmark_dir, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
