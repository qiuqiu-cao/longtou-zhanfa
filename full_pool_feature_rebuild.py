"""Rebuild first-board and relay EOD features from the observed inventory.

This recovers common features for all known board levels without changing the
underlying limitations: current-universe survivorship bias, reconstructed
availability, and supplier-declared security status remain explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_COLUMNS = [
    "limit_count", "consecutive_count", "max_board", "failed_count", "down_limit_count",
    "failed_rate", "seal_rate", "height_change", "limit_change", "middle_density",
    "ladder_gap_count", "unique_space", "one_word_share", "space_feedback_3d",
    "breadth_height_divergence", "yesterday_premium", "previous_limit_tail_loss",
    "failed_tail_loss", "promotion_overall", "promotion_1to2", "promotion_2to3",
    "promotion_3to4", "promotion_4to5", "promotion_mid", "space_feedback",
    "previous_limit_feedback_coverage", "failed_rate_percentile_60d",
    "down_limit_percentile_60d", "promotion_mid_percentile_60d",
    "yesterday_premium_percentile_60d", "tail_loss_percentile_60d", "emotion_heat_score",
    "breadth_health", "relay_appetite", "ladder_health", "loss_effect_health",
    "crowding_risk", "regime_fragility", "false_acceleration", "emotion_data_completeness",
    "emotion_quality_score", "emotion_score", "state_start", "state_warming",
    "state_acceleration", "state_consensus", "state_divergence", "state_retreat",
    "state_repair",
]

EMOTION_STATE_COLUMNS = {
    "state_start": "启动",
    "state_warming": "升温",
    "state_acceleration": "加速",
    "state_consensus": "一致",
    "state_divergence": "大分歧",
    "state_retreat": "退潮",
    "state_repair": "修复",
}


def _truth(series):
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def _reconstruct_emotion_state(market):
    """Prefer the verified one-hot state fields over a corrupted display label.

    Some archived CSVs contain replacement characters in ``emotion_state``.
    The seven numeric state indicators are independently stored and, when they
    form a valid one-hot row, provide a lossless human-readable reconstruction.
    Invalid/incomplete rows retain the raw label but are counted as unresolved.
    """
    columns = list(EMOTION_STATE_COLUMNS)
    values = market[columns].apply(pd.to_numeric, errors="coerce")
    finite = values.notna().all(axis=1)
    row_sum = values.sum(axis=1)
    one_hot = finite & np.isclose(row_sum, 1.0) & values.eq(1.0).sum(axis=1).eq(1)
    derived = market["emotion_state"].astype(object).copy()
    if one_hot.any():
        winner = values.loc[one_hot, columns].idxmax(axis=1)
        derived.loc[one_hot] = winner.map(EMOTION_STATE_COLUMNS)
    return derived, one_hot


def build(observations, daily, market, end_date=None):
    observations = observations.copy()
    daily = daily.copy()
    market = market.copy()
    # Accept both the inventory schema (`signal_date`, `episode_id`) and the
    # rebuilt survivorship schema (`date`, no episode id).  This keeps the
    # strict-cutoff rebuild consumable by the same ranking/backtest pipeline
    # without inventing a missing board identity.
    if "signal_date" not in observations.columns:
        if "date" not in observations.columns:
            raise ValueError("observations missing signal_date/date")
        observations = observations.rename(columns={"date": "signal_date"})
    if "episode_id" not in observations.columns:
        observations["episode_id"] = (
            observations["code"].astype(str).str.zfill(6) + "_" + observations["signal_date"].astype(str)
        )
    if "start_known" not in observations.columns:
        observations["start_known"] = pd.to_numeric(observations.get("boards"), errors="coerce").notna()
    if end_date is not None:
        cutoff = pd.to_datetime(end_date, errors="raise").normalize()
        dates = pd.to_datetime(observations["signal_date"], errors="coerce").dt.normalize()
        if dates.isna().any():
            raise ValueError("observations contain invalid signal_date")
        observations = observations.loc[dates.le(cutoff)].copy()
    observations = observations[["code", "signal_date", "episode_id", "boards", "start_known"]].copy()
    for frame, date_col in ((observations, "signal_date"), (daily, "date")):
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame.duplicated(["code", date_col]).any():
            raise ValueError("duplicate candidate or daily key")
    if market.duplicated("date").any():
        raise ValueError("duplicate market date")
    boards = pd.to_numeric(observations["boards"], errors="coerce")
    known = boards.notna() & boards.ge(1) & boards.mod(1).eq(0)
    observations["boards_numeric"] = boards
    observations["known_board_identity"] = known
    observations["board_group_rebuilt"] = np.where(boards.eq(1), "FIRST_BOARD",
                                                     np.where(known, "CONSECUTIVE_BOARD", "UNKNOWN_LEFT_CENSORED"))
    date_stats = observations.groupby("signal_date", dropna=False).agg(
        candidate_pool_count=("code", "size"), unknown_height_pool_count=("known_board_identity", lambda value: int((~value).sum())),
        known_pool_max_boards=("boards_numeric", "max")).reset_index()
    known_counts = (observations[known].groupby(["signal_date", "boards_numeric"]).size()
                    .rename("same_height_known_count").reset_index())
    merged = observations.merge(date_stats, on="signal_date", how="left", validate="many_to_one")
    merged = merged.merge(known_counts, on=["signal_date", "boards_numeric"], how="left", validate="many_to_one")
    daily_fields = ["date", "code", "open", "high", "low", "close", "pre_close", "amount", "turnover",
                    "float_cap", "st", "new_listing", "status_verified", "source", "universe_complete"]
    missing_daily = set(daily_fields) - set(daily)
    if missing_daily:
        raise ValueError("daily bars missing fields: " + ",".join(sorted(missing_daily)))
    merged = merged.merge(daily[daily_fields], left_on=["signal_date", "code"], right_on=["date", "code"],
                          how="left", validate="one_to_one", indicator="daily_join")
    market_fields = ["date", "source_timestamp", "data_status", "source", "emotion_state", *MARKET_COLUMNS]
    missing_market = set(market_fields) - set(market)
    if missing_market:
        raise ValueError("market emotion missing fields: " + ",".join(sorted(missing_market)))
    market_subset = market[market_fields].copy()
    market_subset["emotion_state"], state_reconstructed = _reconstruct_emotion_state(market)
    market_renamed = market_subset.rename(columns={
        "date": "market_date", "source": "market_source", "data_status": "market_data_status",
        "source_timestamp": "market_source_timestamp", "emotion_state": "market_emotion_state",
        **{name: "market_" + name for name in MARKET_COLUMNS}})
    merged = merged.merge(market_renamed, left_on="signal_date", right_on="market_date",
                          how="left", validate="many_to_one", indicator="market_join")
    numeric = ["open", "high", "low", "close", "pre_close", "amount", "turnover", "float_cap",
               "boards_numeric", "known_pool_max_boards", "same_height_known_count"]
    for column in numeric + ["market_" + name for name in MARKET_COLUMNS]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["boards"] = merged["boards_numeric"]
    merged["boards_for_comparison"] = merged["boards_numeric"]
    merged["height_gap_to_known_top"] = merged["known_pool_max_boards"] - merged["boards_numeric"]
    merged["sole_known_top_proxy"] = np.where(
        merged["known_board_identity"],
        ((merged["boards_numeric"] == merged["known_pool_max_boards"])
         & merged["same_height_known_count"].eq(1)).astype(float), np.nan)
    merged["open_gap_pct"] = (merged["open"] / merged["pre_close"] - 1) * 100
    merged["amplitude_pct"] = (merged["high"] - merged["low"]) / merged["pre_close"] * 100
    merged["amount_to_float_cap"] = merged["amount"] / merged["float_cap"]
    ohlc_complete = merged[["open", "high", "low", "close"]].notna().all(axis=1)
    merged["one_price_day_proxy"] = np.where(
        ohlc_complete,
        ((merged[["open", "high", "low", "close"]].max(axis=1)
          - merged[["open", "high", "low", "close"]].min(axis=1)).abs() < 1e-8).astype(float), np.nan)
    bar_scope = (merged["daily_join"].eq("both") & _truth(merged["status_verified"])
                 & ~_truth(merged["st"]) & ~_truth(merged["new_listing"]))
    market_scope = merged["market_join"].eq("both") & merged["market_data_status"].eq("verified")
    merged["feature_status"] = np.select(
        [~merged["known_board_identity"], ~bar_scope, ~market_scope],
        ["left_censored_identity", "daily_bar_or_security_status_unavailable", "market_context_unavailable"],
        default="historical_reconstruction_not_forward_archive")
    merged["feature_available_not_before"] = merged["signal_date"].astype(str) + "T15:05:00+08:00"
    merged["pool_scope"] = "available_nonst_mainboard_history_not_complete_market"
    merged["board_group"] = merged["board_group_rebuilt"]
    columns = ["code", "signal_date", "episode_id", "boards", "board_group", "start_known",
               "candidate_pool_count", "unknown_height_pool_count", "known_pool_max_boards",
               "boards_for_comparison", "height_gap_to_known_top", "same_height_known_count",
               "sole_known_top_proxy", "open_gap_pct", "amplitude_pct", "amount_to_float_cap",
               "one_price_day_proxy", "turnover", "market_emotion_state",
               *["market_" + name for name in MARKET_COLUMNS],
               "feature_available_not_before", "feature_status", "pool_scope", "daily_join", "market_join"]
    output = merged[columns].copy()
    counts = output["board_group"].value_counts(dropna=False).to_dict()
    ready_rows = output["feature_status"].eq("historical_reconstruction_not_forward_archive")
    report = {"status": "DIAGNOSTIC_ONLY", "input_observations": len(observations), "output_rows": len(output),
              "known_identity_rows": int(known.sum()), "left_censored_rows": int((~known).sum()),
              "feature_complete_reconstructed_rows": int(ready_rows.sum()), "board_group_counts": counts,
              "daily_join_counts": output["daily_join"].value_counts().to_dict(),
              "market_join_counts": output["market_join"].value_counts().to_dict(),
              "first_and_relay_rows_in_same_schema": counts.get("FIRST_BOARD", 0) > 0 and counts.get("CONSECUTIVE_BOARD", 0) > 0,
              "original_point_in_time_capture": False, "historical_universe_complete": False,
              "model_training_authorized": False,
              "emotion_state_one_hot_reconstructed_rows": int(state_reconstructed.sum()),
              "emotion_state_unresolved_rows": int((~state_reconstructed).sum()),
              "emotion_evidence_columns": len(MARKET_COLUMNS),
              "limitations": ["features are reconstructed after the fact, not archived contemporaneously",
                              "current-universe history remains subject to survivorship bias",
                              "security status is supplier-declared and not reaudited here",
                              "left-censored identities are preserved and excluded from known-board features"]}
    return output, report


def run(observations_path, daily_path, market_path, output_dir, end_date=None):
    paths = list(map(Path, (observations_path, daily_path, market_path)))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("choose a new immutable full-pool feature directory")
    observations = pd.read_csv(paths[0], dtype={"code": str})
    daily = pd.read_csv(paths[1], dtype={"code": str})
    market = pd.read_csv(paths[2])
    frame, report = build(observations, daily, market, end_date=end_date)
    report["historical_end_cutoff"] = (pd.to_datetime(end_date).strftime("%Y-%m-%d")
                                        if end_date is not None else None)
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / "full_pool_eod_features.csv"
    frame.to_csv(feature_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths + [Path(__file__)]}
    report["feature_file_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                         allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--daily", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--end-date", help="inclusive historical cutoff for signal dates")
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.daily, args.market, args.output, args.end_date),
                     ensure_ascii=False, indent=2))
