"""Audit divergence between broad A-share indexes and limit-up ecology.

The broad tape and the speculative board-relay ecology are separate layers.
This audit uses fixed sign rules, preserves missing emotion sessions on the
complete dual-index calendar, and tests whether "both indexes up while the
median previous-limit return is non-positive" carries stable forward warning.
It is descriptive and cannot directly gate or rank a stock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ADVERSE = {"大分歧", "退潮"}
SUPPORTIVE = {"启动", "升温", "加速", "修复"}
CATEGORIES = [
    "BOTH_UP_SHORT_POSITIVE", "BOTH_UP_SHORT_LOSS",
    "BOTH_DOWN_SHORT_POSITIVE", "BOTH_DOWN_SHORT_LOSS", "MIXED_INDEX",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(index_sh_return: pd.Series, index_sz_return: pd.Series,
             short_return: pd.Series) -> pd.Series:
    sh = pd.to_numeric(index_sh_return, errors="coerce")
    sz = pd.to_numeric(index_sz_return, errors="coerce")
    short = pd.to_numeric(short_return, errors="coerce")
    result = pd.Series(pd.NA, index=sh.index, dtype="string")
    known = sh.notna() & sz.notna() & short.notna()
    both_up = known & sh.gt(0) & sz.gt(0)
    both_down = known & sh.lt(0) & sz.lt(0)
    result.loc[both_up & short.gt(0)] = "BOTH_UP_SHORT_POSITIVE"
    result.loc[both_up & short.le(0)] = "BOTH_UP_SHORT_LOSS"
    result.loc[both_down & short.gt(0)] = "BOTH_DOWN_SHORT_POSITIVE"
    result.loc[both_down & short.le(0)] = "BOTH_DOWN_SHORT_LOSS"
    result.loc[known & ~(both_up | both_down)] = "MIXED_INDEX"
    return result


def build_calendar(emotion: pd.DataFrame, sh: pd.DataFrame,
                   sz: pd.DataFrame) -> pd.DataFrame:
    for frame, label in ((emotion, "emotion"), (sh, "sh"), (sz, "sz")):
        if "date" not in frame or frame.date.astype(str).duplicated().any():
            raise ValueError(f"{label} dates must be unique")
    required = {
        "date", "emotion_state", "previous_limit_median_return",
        "failed_rate", "down_limit_count", "promotion_mid",
    }
    if required - set(emotion):
        raise ValueError("emotion fields missing")
    sh_part = sh[["date", "benchmark_return_pct"]].rename(
        columns={"benchmark_return_pct": "sh_a_return_pct"})
    sz_part = sz[["date", "benchmark_return_pct"]].rename(
        columns={"benchmark_return_pct": "sz_a_return_pct"})
    calendar = sh_part.merge(sz_part, on="date", how="inner",
                             validate="one_to_one").sort_values("date")
    emotion_fields = sorted(required)
    data = calendar.merge(emotion[emotion_fields], on="date", how="left",
                          validate="one_to_one")
    data["emotion_row_available"] = data.emotion_state.notna().astype(int)
    data["divergence_category"] = classify(
        data.sh_a_return_pct, data.sz_a_return_pct,
        data.previous_limit_median_return)

    next_state = data.emotion_state.shift(-1)
    data["next_emotion_state"] = next_state
    data["next_adverse_state"] = next_state.map(
        lambda value: int(value in ADVERSE) if pd.notna(value) else np.nan)
    data["next_supportive_state"] = next_state.map(
        lambda value: int(value in SUPPORTIVE) if pd.notna(value) else np.nan)
    for field in ("previous_limit_median_return", "failed_rate",
                  "down_limit_count", "promotion_mid"):
        data[f"next_{field}"] = pd.to_numeric(data[field], errors="coerce").shift(-1)

    states = [data.emotion_state.shift(-offset) for offset in (1, 2, 3)]
    complete_three = states[0].notna() & states[1].notna() & states[2].notna()
    any_adverse = pd.concat(
        [series.isin(ADVERSE) for series in states], axis=1).any(axis=1)
    data["next_three_any_adverse"] = np.where(
        complete_three, any_adverse.astype(int), np.nan)
    data["iso_week"] = pd.to_datetime(data.date).dt.strftime("%G-W%V")
    return data


def group_summary(frame: pd.DataFrame) -> dict:
    result = {}
    for category in CATEGORIES:
        part = frame[frame.divergence_category.eq(category)]
        item = {
            "sessions": int(len(part)),
            "first_date": str(part.date.min()) if len(part) else None,
            "last_date": str(part.date.max()) if len(part) else None,
        }
        for field in (
            "next_adverse_state", "next_supportive_state",
            "next_three_any_adverse", "next_previous_limit_median_return",
            "next_failed_rate", "next_down_limit_count", "next_promotion_mid",
        ):
            values = pd.to_numeric(part[field], errors="coerce").dropna()
            item[field] = {
                "known": int(len(values)),
                "mean": float(values.mean()) if len(values) else None,
            }
        result[category] = item
    return result


def weekly_cluster_diff(frame: pd.DataFrame, field: str,
                        group_a: str = "BOTH_UP_SHORT_LOSS",
                        group_b: str = "BOTH_UP_SHORT_POSITIVE",
                        iterations: int = 2000, seed: int = 203) -> dict:
    usable = frame[
        frame.divergence_category.isin([group_a, group_b])
        & pd.to_numeric(frame[field], errors="coerce").notna()].copy()
    usable[field] = pd.to_numeric(usable[field], errors="coerce")
    a = usable.loc[usable.divergence_category.eq(group_a), field]
    b = usable.loc[usable.divergence_category.eq(group_b), field]
    result = {
        "group_a": group_a, "group_b": group_b,
        "group_a_n": int(len(a)), "group_b_n": int(len(b)),
        "observed_mean_difference_a_minus_b": (
            float(a.mean() - b.mean()) if len(a) and len(b) else None),
        "weekly_cluster_bootstrap_ci95": None,
    }
    weeks = usable.iso_week.dropna().unique()
    if len(a) < 5 or len(b) < 5 or len(weeks) < 5:
        return result
    rng = np.random.default_rng(seed)
    samples = []
    groups = {week: usable[usable.iso_week.eq(week)] for week in weeks}
    for _ in range(iterations):
        chosen = rng.choice(weeks, size=len(weeks), replace=True)
        sample = pd.concat([groups[week] for week in chosen], ignore_index=True)
        sa = sample.loc[sample.divergence_category.eq(group_a), field]
        sb = sample.loc[sample.divergence_category.eq(group_b), field]
        if len(sa) and len(sb):
            samples.append(float(sa.mean() - sb.mean()))
    if samples:
        result["weekly_cluster_bootstrap_ci95"] = [
            float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]
    return result


def replication_gate(comparisons: dict, fields: list[str],
                     min_group_n: int = 30) -> dict:
    """Require replication, uncertainty separation and adequate group sizes.

    Matching point-estimate signs across discovery and validation are reported
    separately from a passed replication gate.  This prevents a visually
    consistent direction with wide intervals from being promoted to a rule.
    """
    result = {}
    for field in fields:
        discovery = comparisons["discovery_first_70pct"][field]
        validation = comparisons["validation_last_30pct"][field]
        discovery_value = discovery["observed_mean_difference_a_minus_b"]
        validation_value = validation["observed_mean_difference_a_minus_b"]
        sign_consistent = bool(
            discovery_value is not None
            and validation_value is not None
            and np.sign(discovery_value) != 0
            and np.sign(discovery_value) == np.sign(validation_value)
        )

        def _ci_excludes_zero(item: dict) -> bool:
            ci = item.get("weekly_cluster_bootstrap_ci95")
            return bool(ci and (ci[0] > 0 or ci[1] < 0))

        discovery_ci_pass = _ci_excludes_zero(discovery)
        validation_ci_pass = _ci_excludes_zero(validation)
        sample_pass = all(
            item.get(key, 0) >= min_group_n
            for item in (discovery, validation)
            for key in ("group_a_n", "group_b_n")
        )
        result[field] = {
            "point_estimate_sign_consistent": sign_consistent,
            "discovery_ci_excludes_zero": discovery_ci_pass,
            "validation_ci_excludes_zero": validation_ci_pass,
            "minimum_group_n": min_group_n,
            "minimum_group_n_passed": sample_pass,
            "replication_gate_passed": bool(
                sign_consistent and discovery_ci_pass
                and validation_ci_pass and sample_pass),
        }
    return result


def payoff_summary(calendar: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    required = {
        "signal_date", "fill_status", "net_return_B_plus_1",
        "net_return_B_plus_2", "big_loss_B_plus_1",
    }
    if required - set(outcomes):
        raise ValueError("outcome fields missing")
    categories = calendar[["date", "divergence_category"]].rename(
        columns={"date": "signal_date"})
    joined = outcomes.merge(categories, on="signal_date", how="left",
                            validate="many_to_one")
    filled = joined[joined.fill_status.eq("FILLED_OPEN_PROXY")].copy()
    result = {}
    for category in CATEGORIES:
        part = filled[filled.divergence_category.eq(category)]
        b1 = pd.to_numeric(part.net_return_B_plus_1, errors="coerce").dropna()
        b2 = pd.to_numeric(part.net_return_B_plus_2, errors="coerce").dropna()
        big = pd.to_numeric(part.big_loss_B_plus_1, errors="coerce").dropna()
        result[category] = {
            "signal_sessions": int(part.signal_date.nunique()),
            "filled_rows": int(len(part)),
            "B_plus_1_known": int(len(b1)),
            "B_plus_1_mean_net_return": float(b1.mean()) if len(b1) else None,
            "B_plus_1_big_loss_rate": float(big.mean()) if len(big) else None,
            "B_plus_2_known": int(len(b2)),
            "B_plus_2_mean_net_return": float(b2.mean()) if len(b2) else None,
        }
    return result


def audit(emotion: pd.DataFrame, sh: pd.DataFrame, sz: pd.DataFrame,
          manifest: dict, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = build_calendar(emotion, sh, sz)
    known = data[data.divergence_category.notna()].copy()
    split = int(len(known) * .70)
    discovery = known.iloc[:split].copy()
    validation = known.iloc[split:].copy()
    fields = ["next_adverse_state", "next_three_any_adverse",
              "next_previous_limit_median_return", "next_failed_rate",
              "next_down_limit_count", "next_promotion_mid"]
    comparisons = {
        "full": {field: weekly_cluster_diff(known, field) for field in fields},
        "discovery_first_70pct": {
            field: weekly_cluster_diff(discovery, field) for field in fields},
        "validation_last_30pct": {
            field: weekly_cluster_diff(validation, field) for field in fields},
    }
    def _direction(segment, field):
        value = comparisons[segment][field][
            "observed_mean_difference_a_minus_b"]
        return None if value is None else int(np.sign(value))
    stable = {
        field: (
            _direction("discovery_first_70pct", field) != 0
            and _direction("discovery_first_70pct", field)
            == _direction("validation_last_30pct", field))
        for field in fields
    }
    replication = replication_gate(comparisons, fields)
    providers = {
        symbol: manifest.get("indexes", {}).get(symbol, {}).get("provider")
        for symbol in ("sh000002", "sz399107")}
    missing_emotion_dates = data.loc[
        data.emotion_row_available.eq(0), "date"].astype(str).tolist()
    latest = known.iloc[-1]
    report = {
        "version": "v203-index-emotion-divergence-audit-1",
        "status": "DIAGNOSTIC_ONLY",
        "data_quality": {
            "dual_index_calendar_sessions": int(len(data)),
            "emotion_rows_available": int(data.emotion_row_available.sum()),
            "emotion_rows_missing_on_index_calendar": int(
                data.emotion_row_available.eq(0).sum()),
            "missing_emotion_dates": missing_emotion_dates,
            "index_providers": providers,
            "index_status": "DELAYED_PROVIDER_OBSERVED_NOT_OFFICIAL_RECONCILED",
            "emotion_status": "MIXED_VERIFIED_AND_SPARSE_HISTORICAL_ROWS",
            "missing_emotion_rows_imputed": False,
            "sparse_review_dates_used_as_calendar": False,
        },
        "definition": {
            "broad_index_up": "both 上证A股指数 and 深证A股指数 daily returns > 0",
            "short_positive": "same-day previous-limit median return > 0",
            "short_loss": "same-day previous-limit median return <= 0",
            "divergence": "BOTH_UP_SHORT_LOSS",
            "thresholds_selected_on_outcomes": False,
        },
        "latest_context": {
            "date": str(latest.date),
            "sh_a_return_pct": float(latest.sh_a_return_pct),
            "sz_a_return_pct": float(latest.sz_a_return_pct),
            "previous_limit_median_return": float(
                latest.previous_limit_median_return),
            "divergence_category": str(latest.divergence_category),
            "emotion_state": str(latest.emotion_state),
            "broad_index_may_override_short_ecology": False,
        },
        "full_group_summary": group_summary(known),
        "discovery_group_summary": group_summary(discovery),
        "validation_group_summary": group_summary(validation),
        "weekly_clustered_comparisons": comparisons,
        "discovery_validation_point_estimate_sign_consistent": stable,
        "replication_gate": replication,
        "recent_first_board_payoff_context": payoff_summary(data, outcomes),
        "governance": {
            "broad_index_direction_equals_short_term_emotion": False,
            "index_up_may_override_short_loss_effect": False,
            "divergence_may_be_displayed_as_context": True,
            "divergence_direct_emotion_gate_allowed": False,
            "divergence_direct_stock_reranking_allowed": False,
            "divergence_entry_authorized": False,
        },
        "interpretation": (
            "Broad-index direction and board-relay ecology are separate evidence layers. "
            "An index-up day with a non-positive median return among prior limit-up names "
            "must not be called supportive solely because the indexes are green. Any "
            "forward warning remains descriptive until it replicates in isolated data."),
        "limitations": [
            "index series are provider observations and are not officially reconciled",
            "two index-calendar sessions lack emotion rows and remain unavailable",
            "the sign-based divergence is intentionally simple and may omit distribution tails",
            "recent first-board payoff context covers only a short inspected sample",
            "validation dates have already been inspected and are not a sealed final blind test",
        ],
    }
    return data, report


def run(emotion_path, sh_path, sz_path, manifest_path, outcomes_path,
        output_dir) -> dict:
    paths = [Path(emotion_path), Path(sh_path), Path(sz_path),
             Path(manifest_path), Path(outcomes_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V203 audit")
    emotion = pd.read_csv(paths[0], dtype={"date": str})
    sh = pd.read_csv(paths[1], dtype={"date": str})
    sz = pd.read_csv(paths[2], dtype={"date": str})
    manifest = json.loads(paths[3].read_text(encoding="utf-8"))
    outcomes = pd.read_csv(paths[4], dtype={"signal_date": str, "code": str})
    rows, report = audit(emotion, sh, sz, manifest, outcomes)
    target.mkdir(parents=True)
    rows_path = target / "index_emotion_calendar_rows.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): _sha256(path) for path in paths}
    report["artifact_hashes"] = {rows_path.name: _sha256(rows_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion", required=True)
    parser.add_argument("--sh-index", required=True)
    parser.add_argument("--sz-index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.emotion, args.sh_index, args.sz_index, args.manifest,
        args.outcomes, args.output), ensure_ascii=False, indent=2))
