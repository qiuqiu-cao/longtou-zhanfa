"""Exploratory all-limit-up-pool ranking backtest.

The experiment ranks first boards and consecutive boards together and selects
at most one candidate per signal date.  Its label is a daily-bar opportunity
proxy, not an executable trade: the next session must touch the inferred limit
without being a one-price limit, and the B+2 close mark must be positive after
fixed hypothetical costs.  Legacy inputs without B+2 fields fall back
explicitly to the old B+1 mark and expose that horizon in the output.
"""
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

from duration_factor_ablation_v28 import IDENTITY


TARGET = "next_session_non_one_price_limit_touch_and_b2_positive_close_mark_proxy"
SCOPE = "available_nonst_mainboard_history_first_and_consecutive_same_day_rank"
EMOTION = [
    "market_yesterday_premium", "market_previous_limit_tail_loss",
    "market_failed_tail_loss", "market_failed_rate", "market_down_limit_count",
    "market_promotion_overall", "market_promotion_1to2", "market_promotion_2to3",
    "market_promotion_3to4", "market_promotion_4to5", "market_promotion_mid",
    "market_space_feedback", "market_middle_density", "market_ladder_gap_count",
    "market_breadth_health", "market_relay_appetite", "market_ladder_health",
    "market_loss_effect_health", "market_crowding_risk", "market_regime_fragility",
    "market_false_acceleration", "market_emotion_quality_score", "market_emotion_score",
    "market_state_start", "market_state_warming", "market_state_acceleration",
    "market_state_consensus", "market_state_divergence", "market_state_retreat",
    "market_state_repair",
]
FULL = list(dict.fromkeys(IDENTITY + EMOTION + [
    "open_gap_pct", "amplitude_pct", "amount_to_float_cap", "turnover",
    "one_price_day_proxy", "market_limit_count", "market_consecutive_count",
    "market_seal_rate", "market_promotion_overall", "market_middle_density",
    "market_ladder_gap_count",
]))
GROUPS = {
    "height_only": ["boards"],
    "identity": IDENTITY,
    "identity_plus_emotion": list(dict.fromkeys(IDENTITY + EMOTION)),
    "identity_emotion_price_volume": FULL,
}
ROLE_NORMALIZED = "role_normalized_identity_emotion_price_volume"


def _logit(value):
    value = np.clip(np.asarray(value, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def role_normalized_scores(train, validation, columns):
    """Score each role against its own training base rate.

    This is a diagnostic ranking score, not a calibrated probability.  Fitting
    separate FIRST_BOARD/CONSECUTIVE_BOARD models and subtracting each role's
    training log-odds removes the mechanical advantage of the more common or
    higher-base-rate role without using validation labels.
    """
    scores = np.full(len(validation), np.nan, dtype=float)
    for board_group in ("FIRST_BOARD", "CONSECUTIVE_BOARD"):
        train_group = train[train.board_group.eq(board_group)].copy()
        validation_mask = validation.board_group.eq(board_group).to_numpy()
        if not validation_mask.any():
            continue
        if train_group.empty or train_group.opportunity_proxy_label.nunique() != 2:
            raise ValueError("role-normalized training group lacks both target classes: " + board_group)
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=97),
        )
        weights = 1 / train_group.groupby("signal_date").signal_date.transform("size")
        model.fit(train_group[columns], train_group.opportunity_proxy_label,
                  logisticregression__sample_weight=weights)
        probability = model.predict_proba(validation.loc[validation_mask, columns])[:, 1]
        prior = float(train_group.opportunity_proxy_label.mean())
        scores[validation_mask] = _logit(probability) - float(_logit(prior))
    if np.isnan(scores).any():
        raise ValueError("role-normalized score missing a validation role")
    return scores


def _bool(series):
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})


def prepare(features, outcomes):
    """Return resolved, known-identity research rows without filling identity."""
    required_features = {
        "code", "signal_date", "episode_id", "boards", "board_group",
        "feature_status", "feature_available_not_before", *FULL,
    }
    required_outcomes = {
        "code", "signal_date", "exit_date_LABEL_ONLY", "price_path_state",
        "net_close_mark_LABEL_ONLY", "entry_one_price_LABEL_ONLY",
    }
    missing = {
        "features": sorted(required_features - set(features)),
        "outcomes": sorted(required_outcomes - set(outcomes)),
    }
    if any(missing.values()):
        raise ValueError("missing backtest fields: " + json.dumps(missing))
    features, outcomes = features.copy(), outcomes.copy()
    for frame in (features, outcomes):
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate candidate/date key")
    boards = pd.to_numeric(features["boards"], errors="coerce")
    expected = np.where(boards.eq(1), "FIRST_BOARD", "CONSECUTIVE_BOARD")
    known = (boards.notna() & boards.ge(1) & boards.mod(1).eq(0)
             & features["board_group"].eq(expected)
             & features["feature_status"].eq("historical_reconstruction_not_forward_archive"))
    features = features.loc[known].copy()
    for column in FULL:
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    available = pd.to_datetime(features["feature_available_not_before"], utc=True, errors="coerce")
    close = pd.to_datetime(features["signal_date"] + "T15:05:00+08:00", utc=True, errors="coerce")
    if available.isna().any() or (available > close).any():
        raise ValueError("feature availability later than signal-date close")
    merged = features.merge(outcomes, on=["code", "signal_date"], how="inner", validate="one_to_one")
    # Keep older outcome archives usable while exposing the new forward path
    # fields when they are present.
    if "b1_net_close_mark_LABEL_ONLY" not in merged and "net_close_mark_LABEL_ONLY" in merged:
        merged["b1_net_close_mark_LABEL_ONLY"] = merged["net_close_mark_LABEL_ONLY"]
    if "b1_big_loss_mark_LABEL_ONLY" not in merged and "big_loss_mark_LABEL_ONLY" in merged:
        merged["b1_big_loss_mark_LABEL_ONLY"] = merged["big_loss_mark_LABEL_ONLY"]
    one_price = _bool(merged["entry_one_price_LABEL_ONLY"])
    explicit_touch = "next_session_touched_limit_LABEL_ONLY" in merged
    touched_observed = (_bool(merged["next_session_touched_limit_LABEL_ONLY"])
                        if explicit_touch else None)
    touch_column = ("b1_net_close_mark_LABEL_ONLY"
                    if "b1_net_close_mark_LABEL_ONLY" in merged else "net_close_mark_LABEL_ONLY")
    mark_column = ("b2_net_close_mark_LABEL_ONLY"
                   if "b2_net_close_mark_LABEL_ONLY" in merged else "net_close_mark_LABEL_ONLY")
    touch_mark = pd.to_numeric(merged[touch_column], errors="coerce")
    mark = pd.to_numeric(merged[mark_column], errors="coerce")
    base_state = merged["price_path_state"].eq("PRICE_OPPORTUNITY_PROXY_ONLY")
    if explicit_touch:
        # No-touch and one-price sessions are known failures, not missing
        # observations.  A B+2 mark is required only for a touched, non-one-
        # price opportunity whose hypothetical entry can actually be assessed.
        known_entry = base_state & one_price.notna() & touched_observed.notna()
        needs_mark = touched_observed.eq(True) & one_price.eq(False)
        resolved = known_entry & (~needs_mark | mark.notna())
        unknown_target_count = int((~resolved).sum())
    else:
        # Explicitly retain legacy behavior for old immutable outcome archives.
        # These rows cannot support a full-pool no-touch-negative claim.
        resolved_base = base_state & one_price.notna() & touch_mark.notna()
        resolved = resolved_base.copy()
        if mark_column == "b2_net_close_mark_LABEL_ONLY":
            resolved &= mark.notna()
        unknown_target_count = int((~resolved).sum())
    identity_resolved_count = int(len(merged))
    merged = merged.loc[resolved].copy()
    one_price = one_price.loc[resolved].astype(bool)
    mark = mark.loc[resolved]
    touched = (touched_observed.loc[resolved].astype(bool) if explicit_touch
               else touch_mark.loc[resolved].notna())
    merged["opportunity_proxy_label"] = (touched & ~one_price & mark.gt(0)).astype(int)
    merged["next_session_one_price_LABEL_ONLY"] = one_price
    merged["next_session_touched_limit_LABEL_ONLY"] = touched
    merged["b2_close_mark_net_return_proxy"] = mark
    merged["opportunity_target_horizon"] = "B+2" if mark_column == "b2_net_close_mark_LABEL_ONLY" else "B+1_FALLBACK"
    merged.attrs.update({
        "target_horizon": "B+2" if mark_column == "b2_net_close_mark_LABEL_ONLY" else "B+1_FALLBACK",
        "explicit_touch_field_used": explicit_touch,
        "no_touch_rows_retained_as_negative": bool(explicit_touch),
        "identity_resolved_rows": identity_resolved_count,
        "target_unknown_rows_excluded": unknown_target_count,
        "target_known_rows": int(len(merged)),
    })
    return merged


def split(rows, validation_start, validation_end):
    """Use outcome availability plus episode purge to isolate validation."""
    validation = rows[(rows.signal_date >= validation_start)
                      & (rows.signal_date < validation_end)].copy()
    train = rows[(rows.signal_date < validation_start)
                 & rows.exit_date_LABEL_ONLY.notna()
                 & (rows.exit_date_LABEL_ONLY < validation_start)].copy()
    train = train[~train.episode_id.isin(validation.episode_id)].copy()
    if train.empty or validation.empty:
        raise ValueError("empty training or validation partition")
    if train.opportunity_proxy_label.nunique() != 2:
        raise ValueError("training partition lacks both target classes")
    if set(validation.board_group) != {"FIRST_BOARD", "CONSECUTIVE_BOARD"}:
        raise ValueError("validation must contain first and consecutive boards")
    return train, validation


def select_top1(frame, score_column):
    ranked = frame.sort_values(
        ["signal_date", score_column, "boards", "code"],
        ascending=[True, False, False, True], kind="mergesort")
    return ranked.drop_duplicates("signal_date", keep="first").copy()


def _wilson(successes, total, z=1.959963984540054):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - margin), float(centre + margin)]


def _summary_core(frame, score_column, probability_score=True):
    y = frame.opportunity_proxy_label.astype(int)
    successes = int(y.sum())
    path_proxy = {}
    for horizon in (1, 2, 3):
        column = f"b{horizon}_net_close_mark_LABEL_ONLY"
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        known = values.notna()
        path_proxy[f"b{horizon}"] = {
            "known": int(known.sum()),
            "mean_net_close_mark": float(values[known].mean()) if known.any() else None,
            "positive_rate": float(values[known].gt(0).mean()) if known.any() else None,
            "positive_rate_wilson_95": _wilson(int(values[known].gt(0).sum()), int(known.sum())),
            "large_loss_rate": float(values[known].le(-.07).mean()) if known.any() else None,
            "large_loss_rate_wilson_95": _wilson(int(values[known].le(-.07).sum()), int(known.sum())),
        }
    if "b2_net_close_mark_LABEL_ONLY" in frame and "b3_net_close_mark_LABEL_ONLY" in frame:
        b2 = pd.to_numeric(frame["b2_net_close_mark_LABEL_ONLY"], errors="coerce")
        b3 = pd.to_numeric(frame["b3_net_close_mark_LABEL_ONLY"], errors="coerce")
        both = b2.notna() & b3.notna()
        path_proxy["b2_b3_both_positive"] = {
            "known": int(both.sum()),
            "rate": float((b2[both].gt(0) & b3[both].gt(0)).mean()) if both.any() else None,
        }
    return {
        "selected_sessions": int(len(frame)),
        "proxy_successes": successes,
        "proxy_success_rate": float(y.mean()) if len(y) else None,
        "proxy_success_wilson_95": _wilson(successes, len(frame)),
        "selected_score_auc": float(roc_auc_score(y, frame[score_column])) if y.nunique() == 2 else None,
        "selected_score_brier": (float(brier_score_loss(y, frame[score_column]))
                                  if len(y) and probability_score else None),
        "first_board_selections": int(frame.board_group.eq("FIRST_BOARD").sum()),
        "consecutive_board_selections": int(frame.board_group.eq("CONSECUTIVE_BOARD").sum()),
        "next_session_one_price_cases": int(frame.next_session_one_price_LABEL_ONLY.sum()),
        "actual_fills": 0,
        "complete_trade_win_rate": None,
        "path_proxy": path_proxy,
    }


def _breakdown(frame, score_column, probability_score, column, formatter):
    if column not in frame:
        return {}
    result = {}
    for key, part in frame.groupby(column, dropna=False, sort=True):
        if pd.isna(key):
            label = "UNKNOWN"
        elif isinstance(key, (float, np.floating)) and float(key).is_integer():
            label = str(int(key))
        else:
            label = str(key)
        result[label] = formatter(part, score_column, probability_score)
    return result


def summarize(frame, score_column, probability_score=True):
    result = _summary_core(frame, score_column, probability_score)
    result["by_board_group"] = _breakdown(
        frame, score_column, probability_score, "board_group", _summary_core)
    result["by_board_level"] = _breakdown(
        frame, score_column, probability_score, "boards", _summary_core)
    return result


def _candidate_summary_core(frame, score_column, probability_score=True):
    y = frame.opportunity_proxy_label.astype(int)
    return {
        "candidates": int(len(frame)),
        "proxy_positives": int(y.sum()),
        "proxy_base_rate": float(y.mean()) if len(y) else None,
        "candidate_auc": float(roc_auc_score(y, frame[score_column])) if y.nunique() == 2 else None,
        "candidate_brier": (float(brier_score_loss(y, frame[score_column]))
                            if len(y) and probability_score else None),
    }


def candidate_summary(frame, score_column, probability_score=True):
    result = _candidate_summary_core(frame, score_column, probability_score)
    result["by_board_group"] = _breakdown(
        frame, score_column, probability_score, "board_group", _candidate_summary_core)
    result["by_board_level"] = _breakdown(
        frame, score_column, probability_score, "boards", _candidate_summary_core)
    return result


def run(features, outcomes, validation_start="2025-07-28", validation_end="2025-12-26"):
    rows = prepare(features, outcomes)
    train, validation = split(rows, validation_start, validation_end)
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    predictions = validation[["code", "signal_date", "episode_id", "boards", "board_group"]].copy()
    reports = {}
    path_columns = [
        "b1_net_close_mark_LABEL_ONLY", "b2_net_close_mark_LABEL_ONLY",
        "b3_net_close_mark_LABEL_ONLY", "b1_big_loss_mark_LABEL_ONLY",
        "b2_big_loss_mark_LABEL_ONLY", "b3_big_loss_mark_LABEL_ONLY",
    ]
    validation_detail_columns = [
        "code", "signal_date", "opportunity_proxy_label",
        "next_session_one_price_LABEL_ONLY",
        "next_session_touched_limit_LABEL_ONLY",
        "b2_close_mark_net_return_proxy",
    ] + [column for column in path_columns if column in validation]
    model_specs = list(GROUPS.items()) + [(ROLE_NORMALIZED, FULL)]
    for name, columns in model_specs:
        probability_score = name != ROLE_NORMALIZED
        if name == ROLE_NORMALIZED:
            predictions["score_" + name + "_NOT_LIVE"] = role_normalized_scores(
                train, validation, columns)
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=61),
            )
            model.fit(train[columns], train.opportunity_proxy_label,
                      logisticregression__sample_weight=weights)
            predictions["score_" + name + "_NOT_LIVE"] = model.predict_proba(validation[columns])[:, 1]
        score = "score_" + name + "_NOT_LIVE"
        top = select_top1(predictions, score)
        evaluated = top.merge(
            validation[validation_detail_columns],
            on=["code", "signal_date"], validate="one_to_one")
        group_top_reports = {}
        for board_group, group_predictions in predictions.groupby("board_group", sort=True):
            group_top = select_top1(group_predictions, score)
            group_evaluated = group_top.merge(
                validation[validation_detail_columns],
                on=["code", "signal_date"], validate="one_to_one")
            group_top_reports[str(board_group)] = summarize(
                group_evaluated, score, probability_score=probability_score)
        all_evaluated = predictions[["code", "signal_date", "boards", "board_group", score]].merge(
            validation[["code", "signal_date", "opportunity_proxy_label"]],
            on=["code", "signal_date"], validate="one_to_one")
        reports[name] = {
            "score_type": "role_relative_log_odds" if name == ROLE_NORMALIZED else "calibrated_probability_proxy",
            "all_candidates": candidate_summary(all_evaluated, score, probability_score=probability_score),
            "daily_top1": summarize(evaluated, score, probability_score=probability_score),
            "daily_top1_by_board_group": group_top_reports,
        }
    baseline_pool = validation.copy()
    baseline_pool["highest_board_score_NOT_LIVE"] = baseline_pool["boards"].astype(float)
    baseline = select_top1(baseline_pool, "highest_board_score_NOT_LIVE")
    reports["highest_board_baseline"] = {
        "all_candidates": candidate_summary(
            baseline_pool, "highest_board_score_NOT_LIVE", probability_score=False),
        "daily_top1": summarize(baseline, "highest_board_score_NOT_LIVE", probability_score=False),
    }
    august = predictions[(predictions.signal_date >= "2025-08-01")
                         & (predictions.signal_date < "2025-09-01")]
    august_report = {}
    for name, _ in model_specs:
        score = "score_" + name + "_NOT_LIVE"
        top = select_top1(august, score).merge(
            validation[validation_detail_columns],
            on=["code", "signal_date"], validate="one_to_one")
        august_report[name] = summarize(top, score, probability_score=name != ROLE_NORMALIZED)
    return predictions, {
        "status": "EXPLORATORY_ONLY",
        "target": TARGET,
        "selection_scope": SCOPE,
        "validation_start": validation_start,
        "validation_end_exclusive": validation_end,
        "training_rows": int(len(train)),
        "training_sessions": int(train.signal_date.nunique()),
        "validation_candidates": int(len(validation)),
        "validation_sessions": int(validation.signal_date.nunique()),
        "target_coverage": dict(rows.attrs),
        "one_choice_per_day": True,
        "outcome_availability_purge": True,
        "episode_overlap_purged": True,
        "same_day_weight_total": True,
        "groups": reports,
        "august_2025_subgroup": august_report,
        "deployment_authorized": False,
        "limitations": [
            "reused historical validation is not a sealed blind test",
            "current-universe historical reconstruction is not a complete point-in-time market universe",
            "daily OHLC cannot prove order timing, queue position, fill, or a sellable exit",
            "the target is a price-opportunity proxy and is not complete-trade win probability",
            "themes, opening confirmation, failed-board manifests, announcements, and actual fills are absent",
            "results cannot guarantee premium, prevent a nuclear open, or authorize a live recommendation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-start", default="2025-07-28")
    parser.add_argument("--validation-end", default="2025-12-26")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable backtest output")
    features = pd.read_csv(args.features, dtype={"code": str})
    outcomes = pd.read_csv(args.outcomes, dtype={"code": str})
    predictions, report = run(features, outcomes, args.validation_start, args.validation_end)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output / "rank_scores_BEFORE_LABELS.csv", index=False, encoding="utf-8-sig")
    report["source_hashes"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.features, args.outcomes, Path(__file__))
    }
    with (args.output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
