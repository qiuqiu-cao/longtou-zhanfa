"""Role-separated exploratory backtest for the full limit-up pool.

The experiment keeps three roles distinct:
* emotion anchor: deterministic highest observed board;
* leader potential: next-session close-limit promotion proxy;
* current buy opportunity: next-session touch plus positive B+2 close mark.

All labels are daily-bar proxies.  No role is an executable trade probability.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_pool_proxy_backtest import FULL


WINDOWS = [
    ("2025-01-02", "2025-04-01"),
    ("2025-04-01", "2025-07-01"),
    ("2025-07-01", "2025-10-01"),
    ("2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-04-01"),
    ("2026-04-01", "2026-07-01"),
    ("2026-07-01", "2026-09-01"),
]


def _wilson(successes, total, z=1.959963984540054):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - margin), float(centre + margin)]


def _prepare(features, outcomes):
    features = features.copy()
    outcomes = outcomes.copy()
    for frame in (features, outcomes):
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        if frame.duplicated(["code", "signal_date"]).any():
            raise ValueError("duplicate code/signal_date key")
    missing = {"features": sorted(set(FULL + ["code", "signal_date", "episode_id", "boards",
                                                "board_group", "feature_status", "feature_available_not_before"])
                                   - set(features)),
               "outcomes": sorted({"code", "signal_date", "promotion_LABEL_ONLY", "promotion_state",
                                    "price_path_state", "net_close_mark_LABEL_ONLY",
                                    "entry_one_price_LABEL_ONLY", "exit_date_LABEL_ONLY"} - set(outcomes))}
    if any(missing.values()):
        raise ValueError("missing role backtest fields: " + json.dumps(missing))
    rows = features.merge(outcomes, on=["code", "signal_date"], how="inner", validate="one_to_one")
    known = rows["feature_status"].eq("historical_reconstruction_not_forward_archive")
    boards = pd.to_numeric(rows["boards"], errors="coerce")
    known &= boards.notna() & boards.ge(1) & boards.mod(1).eq(0)
    rows = rows.loc[known].copy()
    for column in FULL:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows["leader_target"] = pd.to_numeric(rows["promotion_LABEL_ONLY"], errors="coerce")
    touch_mark = pd.to_numeric(
        rows.get("b1_net_close_mark_LABEL_ONLY", rows["net_close_mark_LABEL_ONLY"]), errors="coerce")
    mark_column = ("b2_net_close_mark_LABEL_ONLY"
                   if "b2_net_close_mark_LABEL_ONLY" in rows else "net_close_mark_LABEL_ONLY")
    mark = pd.to_numeric(rows[mark_column], errors="coerce")
    one_price = rows["entry_one_price_LABEL_ONLY"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    explicit_touch = "next_session_touched_limit_LABEL_ONLY" in rows
    touched = (rows["next_session_touched_limit_LABEL_ONLY"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
        if explicit_touch else None)
    base_state = rows["price_path_state"].eq("PRICE_OPPORTUNITY_PROXY_ONLY")
    if explicit_touch:
        known_entry = base_state & touched.notna() & one_price.notna()
        needs_mark = touched.eq(True) & one_price.eq(False)
        opportunity_known = known_entry & (~needs_mark | mark.notna())
        opportunity_value = touched.eq(True) & one_price.eq(False) & mark.gt(0)
    else:
        opportunity_known = base_state & touch_mark.notna() & mark.notna() & one_price.notna()
        opportunity_value = touch_mark.notna() & mark.gt(0) & one_price.eq(False)
    rows["opportunity_target_horizon"] = "B+2" if mark_column == "b2_net_close_mark_LABEL_ONLY" else "B+1_FALLBACK"
    rows["opportunity_target"] = np.where(
        opportunity_known, opportunity_value.astype(float), np.nan)
    rows.attrs.update({
        "target_horizon": "B+2" if mark_column == "b2_net_close_mark_LABEL_ONLY" else "B+1_FALLBACK",
        "target_rows_before_unknown_filter": int(len(rows)),
        "target_unknown_rows": int((~opportunity_known).sum()),
        "target_known_rows": int(opportunity_known.sum()),
        "explicit_touch_field_used": explicit_touch,
        "no_touch_rows_retained_as_negative": bool(explicit_touch),
    })
    return rows


def _fit(train, validation, target):
    train = train[train[target].notna()].copy()
    if train.empty or train[target].nunique() != 2:
        raise ValueError("role target lacks two training classes: " + target)
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=181),
    )
    model.fit(train[FULL], train[target], logisticregression__sample_weight=weights)
    return model.predict_proba(validation[FULL])[:, 1]


def _metrics(selected, score, target):
    y = pd.to_numeric(selected[target], errors="coerce")
    known = y.notna()
    values = y[known].astype(int)
    score_values = selected.loc[known, score]
    return {
        "selected_sessions": int(len(selected)),
        "known_target": int(known.sum()),
        "positive": int(values.sum()),
        "positive_rate": float(values.mean()) if len(values) else None,
        "positive_rate_wilson_95": _wilson(int(values.sum()), int(len(values))),
        "auc": float(roc_auc_score(values, score_values)) if values.nunique() == 2 else None,
        "first_board_selections": int(selected.board_group.eq("FIRST_BOARD").sum()),
        "consecutive_board_selections": int(selected.board_group.eq("CONSECUTIVE_BOARD").sum()),
    }


def _metrics_by_emotion(selected, score, target):
    if "market_emotion_state" not in selected:
        return {}
    return {
        "UNAVAILABLE" if pd.isna(state) else str(state): _metrics(part, score, target)
        for state, part in selected.groupby("market_emotion_state", dropna=False, sort=True)
    }


def _metrics_by_group(selected, score, target, column):
    if column not in selected:
        return {}
    return {
        "UNAVAILABLE" if pd.isna(group) else str(group): _metrics(part, score, target)
        for group, part in selected.groupby(column, dropna=False, sort=True)
    }


def _path_metrics(selected):
    """Summarize close-mark paths without treating them as executable fills."""
    result = {}
    for horizon in (1, 2, 3):
        column = f"b{horizon}_net_close_mark_LABEL_ONLY"
        if column not in selected:
            continue
        values = pd.to_numeric(selected[column], errors="coerce")
        known = values.notna()
        result[f"b{horizon}"] = {
            "known": int(known.sum()),
            "mean_net_close_mark": float(values[known].mean()) if known.any() else None,
            "positive_rate": float(values[known].gt(0).mean()) if known.any() else None,
            "positive_rate_wilson_95": _wilson(int(values[known].gt(0).sum()), int(known.sum())),
            "large_loss_rate": float(values[known].le(-.07).mean()) if known.any() else None,
            "large_loss_rate_wilson_95": _wilson(int(values[known].le(-.07).sum()), int(known.sum())),
        }
    if all(column in selected for column in
           ("b2_net_close_mark_LABEL_ONLY", "b3_net_close_mark_LABEL_ONLY")):
        b2 = pd.to_numeric(selected["b2_net_close_mark_LABEL_ONLY"], errors="coerce")
        b3 = pd.to_numeric(selected["b3_net_close_mark_LABEL_ONLY"], errors="coerce")
        known = b2.notna() & b3.notna()
        result["b2_b3_both_positive"] = {
            "known": int(known.sum()),
            "rate": float((b2[known].gt(0) & b3[known].gt(0)).mean()) if known.any() else None,
        }
    return result


def _path_metrics_by_emotion(selected):
    if "market_emotion_state" not in selected:
        return {}
    return {
        "UNAVAILABLE" if pd.isna(state) else str(state): _path_metrics(part)
        for state, part in selected.groupby("market_emotion_state", dropna=False, sort=True)
    }


def _path_metrics_by_group(selected, column):
    if column not in selected:
        return {}
    return {
        "UNAVAILABLE" if pd.isna(group) else str(group): _path_metrics(part)
        for group, part in selected.groupby(column, dropna=False, sort=True)
    }


def _select_top(frame, score):
    return frame.sort_values(["signal_date", score, "boards", "code"],
                             ascending=[True, False, False, True], kind="mergesort").drop_duplicates("signal_date")


def run(features, outcomes):
    rows = _prepare(features, outcomes)
    fold_reports = []
    aggregate = {"emotion_anchor": [], "leader_potential": [], "current_buy_opportunity": []}
    for start, end in WINDOWS:
        validation = rows[(rows.signal_date >= start) & (rows.signal_date < end)].copy()
        train = rows[(rows.signal_date < start) & rows.exit_date_LABEL_ONLY.notna()
                     & (rows.exit_date_LABEL_ONLY < start)].copy()
        train = train[~train.episode_id.isin(validation.episode_id)].copy()
        fold = {"start": start, "end_exclusive": end,
                "training_rows": int(len(train)), "validation_rows": int(len(validation))}
        if train.empty or validation.empty or set(validation.board_group) != {"FIRST_BOARD", "CONSECUTIVE_BOARD"}:
            fold["status"] = "INSUFFICIENT_DATA"
            fold_reports.append(fold)
            continue
        validation["leader_score_NOT_LIVE"] = _fit(train, validation, "leader_target")
        validation["opportunity_score_NOT_LIVE"] = _fit(train, validation, "opportunity_target")
        anchor = _select_top(validation, "boards")
        leader = _select_top(validation, "leader_score_NOT_LIVE")
        opportunity = _select_top(validation, "opportunity_score_NOT_LIVE")
        selected = {"emotion_anchor": anchor, "leader_potential": leader,
                    "current_buy_opportunity": opportunity}
        fold["status"] = "OK"
        fold["roles"] = {
            "emotion_anchor": _metrics(anchor, "boards", "leader_target"),
            "leader_potential": _metrics(leader, "leader_score_NOT_LIVE", "leader_target"),
            "current_buy_opportunity": _metrics(opportunity, "opportunity_score_NOT_LIVE", "opportunity_target"),
            "opportunity_selection_on_leader_target": _metrics(opportunity, "opportunity_score_NOT_LIVE", "leader_target"),
        }
        fold["roles_by_board_group"] = {
            "emotion_anchor": _metrics_by_group(anchor, "boards", "leader_target", "board_group"),
            "leader_potential": _metrics_by_group(leader, "leader_score_NOT_LIVE", "leader_target", "board_group"),
            "current_buy_opportunity": _metrics_by_group(
                opportunity, "opportunity_score_NOT_LIVE", "opportunity_target", "board_group"),
        }
        fold["roles_by_emotion_state"] = {
            "leader_potential": _metrics_by_emotion(leader, "leader_score_NOT_LIVE", "leader_target"),
            "current_buy_opportunity": _metrics_by_emotion(
                opportunity, "opportunity_score_NOT_LIVE", "opportunity_target"),
        }
        fold["path_metrics"] = {name: _path_metrics(frame) for name, frame in selected.items()}
        fold["path_metrics_by_emotion_state"] = {
            name: _path_metrics_by_emotion(frame) for name, frame in selected.items()
        }
        fold["path_metrics_by_board_group"] = {
            name: _path_metrics_by_group(frame, "board_group") for name, frame in selected.items()
        }
        fold["overlap"] = {
            "anchor_leader": int(len(set(anchor.code) & set(leader.code))),
            "leader_opportunity": int(len(set(leader.code) & set(opportunity.code))),
            "anchor_opportunity": int(len(set(anchor.code) & set(opportunity.code))),
        }
        for name, frame in selected.items():
            aggregate[name].append(frame)
        fold_reports.append(fold)
    aggregate_report = {}
    for name, frames in aggregate.items():
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        score = {"emotion_anchor": "boards", "leader_potential": "leader_score_NOT_LIVE",
                 "current_buy_opportunity": "opportunity_score_NOT_LIVE"}[name]
        target = {"emotion_anchor": "leader_target", "leader_potential": "leader_target",
                  "current_buy_opportunity": "opportunity_target"}[name]
        aggregate_report[name] = _metrics(frame, score, target)
        aggregate_report[name]["by_emotion_state"] = _metrics_by_emotion(frame, score, target)
        aggregate_report[name]["path_metrics"] = _path_metrics(frame)
        aggregate_report[name]["path_metrics_by_emotion_state"] = _path_metrics_by_emotion(frame)
        aggregate_report[name]["by_board_group"] = _metrics_by_group(frame, score, target, "board_group")
        aggregate_report[name]["path_metrics_by_board_group"] = _path_metrics_by_group(frame, "board_group")
    return {
        "status": "EXPLORATORY_ONLY",
        "role_definitions": {
            "emotion_anchor": "highest observed board; descriptive emotion reference only",
            "leader_potential": "next-session closing-limit promotion proxy",
            "current_buy_opportunity": "next-session touch plus positive B+2 close-mark proxy (B+1 fallback only for legacy inputs)",
        },
        "windows": fold_reports,
        "aggregate_roles": aggregate_report,
        "target_coverage": dict(rows.attrs),
        "deployment_authorized": False,
        "complete_trade_count": 0,
        "limitations": [
            "role targets are daily-bar proxies, not fills or complete-trade probabilities",
            "current historical candidate universe is not proven complete",
            "auction, Level-2, specific theme and broker execution evidence are absent",
            "emotion anchor is descriptive and not a buy signal",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable role backtest")
    result = run(pd.read_csv(args.features, dtype={"code": str}),
                 pd.read_csv(args.outcomes, dtype={"code": str}))
    result["source_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.features, args.outcomes, Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                              encoding="utf-8")
    print(json.dumps({"status": result["status"], "aggregate_roles": result["aggregate_roles"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
