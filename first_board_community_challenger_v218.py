"""Audit community first-board heuristics without changing the relay model.

The community rules are hypotheses, not trading truth.  Every feature is
available at the signal-date close.  Promotion and payoff labels are used only
for later evaluation.  The module keeps promotion identity, executable payoff
and large-loss risk as separate questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "v218-first-board-community-challenger-1"
KEYS = ["signal_date", "code"]
BASE_FEATURES = [
    "first_seal_minute", "last_minus_first_minute", "failed_seal_count",
    "turnover", "amount_to_float_cap", "seal_fund_to_float_cap",
    "one_price_proxy", "thin_control_risk", "industry_first_board_count",
    "industry_all_board_count", "industry_higher_board_count",
    "market_first_board_count", "market_relay_count",
]
COMMUNITY_FEATURES = BASE_FEATURES + [
    "sealed_by_1030", "low_failed_seal", "quick_reseal",
    "turnover_5_15", "turnover_10_20", "liquidity_ratio_3_20",
    "industry_pioneer_rank", "industry_pioneer_rank_pct",
    "industry_breadth_3plus", "industry_ladder_support",
    "supported_pioneer", "volume_ratio_5", "platform_breakout_20",
    "moving_average_alignment_5_10_20", "pre_board_position_60",
    "amount_billion", "float_cap_billion", "community_evidence_count",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_daily_bar_features(rows: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Attach signal-close technical context using only same/prior daily bars."""
    required = {"date", "code", "high", "close", "pre_close", "volume", "amount", "float_cap"}
    if required - set(bars):
        raise ValueError(f"daily-bar fields missing: {sorted(required - set(bars))}")
    frame = bars.copy()
    frame["code"] = frame.code.astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date"]).drop_duplicates(["date", "code"], keep="last")
    for column in required - {"date", "code"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["code", "date"])
    by_code = frame.groupby("code", group_keys=False)
    frame["prior5_volume_mean"] = by_code.volume.transform(
        lambda value: value.shift(1).rolling(5, min_periods=5).mean())
    frame["prior20_high"] = by_code.high.transform(
        lambda value: value.shift(1).rolling(20, min_periods=20).max())
    frame["ma5"] = by_code.close.transform(lambda value: value.rolling(5, min_periods=5).mean())
    frame["ma10"] = by_code.close.transform(lambda value: value.rolling(10, min_periods=10).mean())
    frame["ma20"] = by_code.close.transform(lambda value: value.rolling(20, min_periods=20).mean())
    frame["prior60_low"] = by_code.close.transform(
        lambda value: value.shift(1).rolling(60, min_periods=20).min())
    frame["prior60_high"] = by_code.close.transform(
        lambda value: value.shift(1).rolling(60, min_periods=20).max())
    frame["volume_ratio_5"] = frame.volume / frame.prior5_volume_mean.where(
        frame.prior5_volume_mean.gt(0))
    frame["platform_breakout_20"] = frame.close.gt(frame.prior20_high).where(
        frame.prior20_high.notna()).astype("Int64")
    frame["moving_average_alignment_5_10_20"] = (
        frame.ma5.gt(frame.ma10) & frame.ma10.gt(frame.ma20)
    ).where(frame[["ma5", "ma10", "ma20"]].notna().all(axis=1)).astype("Int64")
    spread = frame.prior60_high - frame.prior60_low
    frame["pre_board_position_60"] = (
        (frame.pre_close - frame.prior60_low) / spread.where(spread.gt(0))
    ).clip(0, 1)
    frame["amount_billion"] = frame.amount / 1e8
    frame["float_cap_billion"] = frame.float_cap / 1e8
    view = frame[["date", "code", "volume_ratio_5", "platform_breakout_20",
                  "moving_average_alignment_5_10_20", "pre_board_position_60",
                  "amount_billion", "float_cap_billion"]].rename(columns={"date": "signal_date"})
    out = rows.copy()
    out["code"] = out.code.astype(str).str.zfill(6)
    return out.merge(view, on=KEYS, how="left", validate="many_to_one")


def load_relevant_bars(path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    codes = set(rows.code.astype(str).str.zfill(6))
    dates = pd.to_datetime(rows.signal_date, errors="coerce")
    lower = (dates.min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    upper = dates.max().strftime("%Y-%m-%d")
    usecols = ["date", "code", "high", "close", "pre_close", "volume", "amount", "float_cap"]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"code": str}, chunksize=250_000):
        code = chunk.code.astype(str).str.zfill(6)
        date = chunk.date.astype(str)
        mask = code.isin(codes) & date.between(lower, upper)
        if mask.any():
            part = chunk.loc[mask].copy()
            part["code"] = code[mask]
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


def derive_features(rows: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["name", "industry_proxy"] + BASE_FEATURES)
    missing = required - set(rows)
    if missing:
        raise ValueError(f"first-board fields missing: {sorted(missing)}")
    out = rows.copy()
    out["code"] = out.code.astype(str).str.zfill(6)
    numeric = set(BASE_FEATURES) | {"promotion_label"}
    for column in numeric & set(out):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["sealed_by_1030"] = out.first_seal_minute.le(75).astype(int)
    out["low_failed_seal"] = out.failed_seal_count.le(1).astype(int)
    out["quick_reseal"] = (
        out.failed_seal_count.le(1) & out.last_minus_first_minute.le(30)
    ).astype(int)
    out["turnover_5_15"] = out.turnover.between(5, 15, inclusive="both").astype(int)
    out["turnover_10_20"] = out.turnover.between(10, 20, inclusive="both").astype(int)
    out["liquidity_ratio_3_20"] = out.amount_to_float_cap.between(
        .03, .20, inclusive="both").astype(int)
    # Rank is based only on same-close seal time within the provider industry.
    group = [out.signal_date.astype(str), out.industry_proxy.fillna("__MISSING__")]
    out["industry_pioneer_rank"] = out.first_seal_minute.groupby(group).rank(
        method="min", ascending=True)
    size = out.groupby([out.signal_date.astype(str), out.industry_proxy.fillna("__MISSING__")]).code.transform("size")
    out["industry_pioneer_rank_pct"] = (out.industry_pioneer_rank / size.clip(lower=1)).clip(0, 1)
    out["industry_breadth_3plus"] = out.industry_first_board_count.ge(3).astype(int)
    out["industry_ladder_support"] = out.industry_higher_board_count.ge(1).astype(int)
    out["supported_pioneer"] = (
        out.industry_pioneer_rank.le(2)
        & (out.industry_breadth_3plus.eq(1) | out.industry_ladder_support.eq(1))
    ).astype(int)
    for column in ["volume_ratio_5", "platform_breakout_20",
                   "moving_average_alignment_5_10_20", "pre_board_position_60",
                   "amount_billion", "float_cap_billion"]:
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["community_volume_1_5_to_3"] = out.volume_ratio_5.between(1.5, 3).astype(int)
    out["community_low_position"] = out.pre_board_position_60.le(.60).astype(int)
    positive = [
        "sealed_by_1030", "low_failed_seal", "quick_reseal", "turnover_5_15",
        "liquidity_ratio_3_20", "industry_breadth_3plus",
        "industry_ladder_support", "supported_pioneer",
        "platform_breakout_20", "moving_average_alignment_5_10_20",
        "community_volume_1_5_to_3", "community_low_position",
    ]
    out["community_evidence_count"] = out[positive].sum(axis=1)
    out["community_evidence_count"] -= 2 * out.one_price_proxy.fillna(0)
    out["community_evidence_count"] -= out.thin_control_risk.fillna(0)
    out["close_quality_gate"] = (
        out.one_price_proxy.eq(0) & out.thin_control_risk.eq(0)
        & out.sealed_by_1030.eq(1) & out.quick_reseal.eq(1)
        & out.turnover.between(5, 20, inclusive="both")
        & out.amount_to_float_cap.between(.03, .20, inclusive="both")
    )
    return out


def attach_outcomes(rows: pd.DataFrame, outcomes: pd.DataFrame | None) -> pd.DataFrame:
    if outcomes is None:
        return rows
    view = outcomes.copy()
    view["code"] = view.code.astype(str).str.zfill(6)
    keep = KEYS + [name for name in [
        "fill_status", "net_return_B_plus_1", "net_return_B_plus_2",
        "big_loss_B_plus_1",
    ] if name in view]
    if view.duplicated(KEYS).any():
        raise ValueError("outcome keys must be unique")
    out = rows.merge(view[keep], on=KEYS, how="left", validate="one_to_one")
    out["executable_B_plus_2_positive"] = np.where(
        out.fill_status.eq("FILLED_OPEN_PROXY")
        & pd.to_numeric(out.net_return_B_plus_2, errors="coerce").notna(),
        pd.to_numeric(out.net_return_B_plus_2, errors="coerce").gt(0).astype(int),
        np.nan,
    )
    return out


def _fit_predict(train: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> np.ndarray:
    y = pd.to_numeric(train.promotion_label, errors="raise").astype(int)
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        LogisticRegression(C=.5, max_iter=3000, random_state=218),
    )
    model.fit(train[features], y, logisticregression__sample_weight=weights)
    return model.predict_proba(score[features])[:, 1]


def prequential(rows: pd.DataFrame, min_train_sessions: int = 12) -> pd.DataFrame:
    dates = sorted(rows.signal_date.astype(str).unique())
    outputs = []
    for index in range(min_train_sessions, len(dates)):
        train_dates, date = dates[:index], dates[index]
        train = rows[rows.signal_date.astype(str).isin(train_dates)].copy()
        score = rows[rows.signal_date.astype(str).eq(date)].copy()
        if score.empty or train.promotion_label.nunique() < 2:
            continue
        score["base_probability"] = _fit_predict(train, score, BASE_FEATURES)
        score["community_probability"] = _fit_predict(train, score, COMMUNITY_FEATURES)
        outputs.append(score)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def daily_top1(rows: pd.DataFrame, score: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return (rows.sort_values(["signal_date", score, "code"], ascending=[True, False, True])
            .groupby("signal_date", as_index=False).first())


def _metric(rows: pd.DataFrame, score: str) -> dict:
    y = pd.to_numeric(rows.promotion_label, errors="coerce")
    p = pd.to_numeric(rows[score], errors="coerce")
    valid = y.notna() & p.notna()
    y, p = y[valid].astype(int), p[valid]
    top = daily_top1(rows.loc[valid], score)
    result = {
        "candidate_n": int(len(y)),
        "sessions": int(top.signal_date.nunique()) if len(top) else 0,
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "top1_promotions": int(top.promotion_label.sum()) if len(top) else 0,
        "top1_promotion_rate": float(top.promotion_label.mean()) if len(top) else None,
    }
    if "net_return_B_plus_2" in top:
        filled = top[top.fill_status.eq("FILLED_OPEN_PROXY")].copy()
        b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
        b1loss = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
        result.update({
            "top1_filled_B_plus_2_n": int(len(b2)),
            "top1_mean_B_plus_2_net_return": float(b2.mean()) if len(b2) else None,
            "top1_B_plus_2_positive_rate": float(b2.gt(0).mean()) if len(b2) else None,
            "top1_B_plus_1_big_loss_rate": float(b1loss.mean()) if len(b1loss) else None,
        })
    return result


def factor_table(rows: pd.DataFrame) -> list[dict]:
    masks = {
        "sealed_by_1030": rows.sealed_by_1030.eq(1),
        "low_failed_seal": rows.low_failed_seal.eq(1),
        "quick_reseal": rows.quick_reseal.eq(1),
        "turnover_5_15": rows.turnover_5_15.eq(1),
        "turnover_10_20": rows.turnover_10_20.eq(1),
        "industry_breadth_3plus": rows.industry_breadth_3plus.eq(1),
        "industry_ladder_support": rows.industry_ladder_support.eq(1),
        "supported_pioneer": rows.supported_pioneer.eq(1),
        "volume_ratio_1_5_to_3": rows.community_volume_1_5_to_3.eq(1),
        "platform_breakout_20": rows.platform_breakout_20.eq(1),
        "moving_average_alignment_5_10_20": rows.moving_average_alignment_5_10_20.eq(1),
        "pre_board_position_le_60pct": rows.community_low_position.eq(1),
        "one_price_proxy": rows.one_price_proxy.eq(1),
    }
    records = []
    for name, mask in masks.items():
        part = rows[mask]
        record = {
            "factor": name, "n": int(len(part)),
            "sessions": int(part.signal_date.nunique()),
            "promotion_rate": float(part.promotion_label.mean()) if len(part) else None,
        }
        if "net_return_B_plus_2" in part:
            filled = part[part.fill_status.eq("FILLED_OPEN_PROXY")]
            b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
            record["filled_B_plus_2_n"] = int(len(b2))
            record["mean_B_plus_2_net_return"] = float(b2.mean()) if len(b2) else None
            record["B_plus_2_positive_rate"] = float(b2.gt(0).mean()) if len(b2) else None
        records.append(record)
    return records


def score_current(history: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    current = derive_features(current)
    current["community_probability_not_calibrated"] = _fit_predict(
        history, current, COMMUNITY_FEATURES)
    current["execution_status"] = np.select(
        [current.one_price_proxy.eq(1), current.thin_control_risk.eq(1)],
        ["WATCH_UNBUYABLE_ONE_PRICE", "WATCH_THIN_CONTROL_RISK"],
        default="WATCH_REQUIRES_NEXT_SESSION_CONFIRMATION",
    )
    current["entry_authorized"] = False
    if "best_theme_lane" not in current:
        current["best_theme_lane"] = "UNAVAILABLE"
        current["best_theme_name"] = pd.NA
        current["best_theme_stage"] = "UNAVAILABLE"
        current["best_theme_rank_within_lane"] = np.nan
        current["all_theme_tags"] = ""
    current["theme_research_gate"] = (
        current.close_quality_gate
        & current.best_theme_lane.isin({
            "VERIFIED_REAL_THEME", "VERIFIED_MEMBERSHIP_ONLY", "OBSERVED_THEME_ROSTER"})
        & pd.to_numeric(current.best_theme_rank_within_lane, errors="coerce").le(3)
        & current.best_theme_stage.ne("ISOLATED_OR_THIN")
    )
    return current.sort_values(
        ["theme_research_gate", "close_quality_gate", "community_probability_not_calibrated",
         "community_evidence_count", "code"],
        ascending=[False, False, False, False, True],
    )


def attach_current_theme_map(current: pd.DataFrame, theme_map: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signal_date", "code", "theme_name", "evidence_lane", "theme_stage",
        "rank_within_lane", "theme_pioneer_rank",
    }
    if required - set(theme_map):
        raise ValueError(f"current theme-map fields missing: {sorted(required - set(theme_map))}")
    view = theme_map.copy()
    view["code"] = view.code.astype(str).str.zfill(6)
    lane_order = {
        "VERIFIED_REAL_THEME": 0, "VERIFIED_MEMBERSHIP_ONLY": 1,
        "OBSERVED_THEME_ROSTER": 2, "INDUSTRY_PROXY": 3,
        "SYMBOLIC_NAME_CLUSTER": 4, "RUMOR_UNVERIFIED": 5,
    }
    view["_lane_order"] = view.evidence_lane.map(lane_order).fillna(99)
    view["rank_within_lane"] = pd.to_numeric(view.rank_within_lane, errors="coerce")
    view["theme_pioneer_rank"] = pd.to_numeric(view.theme_pioneer_rank, errors="coerce")
    tags = (view.sort_values(["signal_date", "code", "_lane_order", "rank_within_lane", "theme_name"])
            .groupby(KEYS).apply(
                lambda part: "|".join(
                    f"{row.evidence_lane}:{row.theme_name}" for row in part.itertuples()),
                include_groups=False).rename("all_theme_tags").reset_index())
    best = (view.sort_values(
        ["signal_date", "code", "_lane_order", "rank_within_lane",
         "theme_pioneer_rank", "theme_name"],
        na_position="last").groupby(KEYS, as_index=False).first())
    best = best.rename(columns={
        "evidence_lane": "best_theme_lane", "theme_name": "best_theme_name",
        "theme_stage": "best_theme_stage",
        "rank_within_lane": "best_theme_rank_within_lane",
        "theme_pioneer_rank": "best_theme_pioneer_rank",
    })
    keep = KEYS + ["best_theme_lane", "best_theme_name", "best_theme_stage",
                   "best_theme_rank_within_lane", "best_theme_pioneer_rank"]
    out = current.copy()
    out["code"] = out.code.astype(str).str.zfill(6)
    return (out.merge(best[keep], on=KEYS, how="left", validate="one_to_one")
            .merge(tags, on=KEYS, how="left", validate="one_to_one"))


def run(observations_path: Path, output: Path, outcomes_path: Path | None = None,
        current_path: Path | None = None, bars_path: Path | None = None,
        current_theme_map_path: Path | None = None,
        min_train_sessions: int = 12) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable v218 output")
    observations = pd.read_csv(observations_path, dtype={"code": str})
    outcomes = pd.read_csv(outcomes_path, dtype={"code": str}) if outcomes_path else None
    current_raw = pd.read_csv(current_path, dtype={"code": str}) if current_path else None
    if current_raw is not None and current_theme_map_path:
        current_raw = attach_current_theme_map(
            current_raw, pd.read_csv(current_theme_map_path, dtype={"code": str}))
    if bars_path:
        all_rows = pd.concat([observations[KEYS], current_raw[KEYS] if current_raw is not None else observations[KEYS].iloc[:0]])
        bars = load_relevant_bars(bars_path, all_rows)
        observations = attach_daily_bar_features(observations, bars)
        if current_raw is not None:
            current_raw = attach_daily_bar_features(current_raw, bars)
    rows = attach_outcomes(derive_features(observations), outcomes)
    predictions = prequential(rows, min_train_sessions)
    base = _metric(predictions, "base_probability") if len(predictions) else {}
    challenger = _metric(predictions, "community_probability") if len(predictions) else {}
    promoted = bool(
        base and challenger
        and challenger.get("auc") is not None and base.get("auc") is not None
        and challenger["auc"] > base["auc"]
        and challenger["brier"] < base["brier"]
        and challenger["top1_promotion_rate"] > base["top1_promotion_rate"]
        and challenger.get("top1_mean_B_plus_2_net_return") is not None
        and base.get("top1_mean_B_plus_2_net_return") is not None
        and challenger["top1_mean_B_plus_2_net_return"] >= base["top1_mean_B_plus_2_net_return"]
        and challenger.get("top1_B_plus_1_big_loss_rate", 1)
            <= base.get("top1_B_plus_1_big_loss_rate", 0)
    )
    output.mkdir(parents=True)
    rows.to_csv(output / "community_feature_rows_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "prequential_predictions.csv", index=False, encoding="utf-8-sig")
    current_top = []
    if current_raw is not None:
        scored_current = score_current(rows, current_raw)
        scored_current.to_csv(output / "current_first_board_watchlist.csv", index=False,
                              encoding="utf-8-sig")
        current_top = scored_current.head(10)[[
            "code", "name", "industry_proxy", "community_probability_not_calibrated",
            "community_evidence_count", "close_quality_gate", "supported_pioneer", "one_price_proxy",
            "best_theme_lane", "best_theme_name", "best_theme_stage",
            "best_theme_rank_within_lane", "theme_research_gate",
            "execution_status", "entry_authorized",
        ]].to_dict("records")
        eligible = scored_current[scored_current.theme_research_gate]
        primary = eligible.iloc[0] if len(eligible) else None
    else:
        primary = None
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "scope": "FIRST_BOARD_ONLY_RELAY_AND_LEADER_MODELS_UNCHANGED",
        "observations": int(len(rows)),
        "sessions": int(rows.signal_date.nunique()),
        "prequential_sessions": int(predictions.signal_date.nunique()) if len(predictions) else 0,
        "base": base,
        "community_challenger": challenger,
        "challenger_promoted": promoted,
        "factor_summary": factor_table(rows),
        "daily_bar_feature_coverage": {
            name: int(pd.to_numeric(rows[name], errors="coerce").notna().sum())
            for name in ["volume_ratio_5", "platform_breakout_20",
                         "moving_average_alignment_5_10_20", "pre_board_position_60",
                         "amount_billion", "float_cap_billion"]
        },
        "current_top10_research_watchlist": current_top,
        "current_primary_close_quality_watch": ({
            "code": str(primary.code), "name": str(primary["name"]),
            "industry_proxy": str(primary.industry_proxy),
            "theme_name": str(primary.best_theme_name),
            "theme_lane": str(primary.best_theme_lane),
            "theme_stage": str(primary.best_theme_stage),
            "community_probability_not_calibrated": float(
                primary.community_probability_not_calibrated),
            "selection_status": "FROZEN_WATCH_NO_REPLACEMENT_IF_OPEN_CONFIRMATION_FAILS",
            "entry_authorized": False,
        } if primary is not None else None),
        "governance": {
            "community_sources_are_hypotheses": True,
            "features_known_by_signal_close": True,
            "promotion_and_payoff_separated": True,
            "one_price_never_treated_as_executable": True,
            "daily_top1_no_replacement": True,
            "entry_authorized": False,
            "minimum_forward_predictions_before_deployment": 60,
            "minimum_forward_promotions_before_deployment": 20,
        },
        "limitations": [
            "industry is a provider proxy, not a verified real-theme roster",
            "platform breakout and five-day relative volume have partial daily-bar coverage; auction confirmation is unavailable at close",
            "short clustered history cannot validate a deployable threshold",
            "current ranking is a research watchlist and requires next-session auction/open confirmation",
        ],
    }
    sources = ([observations_path] + ([outcomes_path] if outcomes_path else [])
               + ([current_path] if current_path else []) + ([bars_path] if bars_path else []))
    if current_theme_map_path:
        sources.append(current_theme_map_path)
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in sources}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--bars", type=Path)
    parser.add_argument("--current-theme-map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-train-sessions", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.output, args.outcomes, args.current,
                         args.bars, args.current_theme_map, args.min_train_sessions),
                     ensure_ascii=False, indent=2))
