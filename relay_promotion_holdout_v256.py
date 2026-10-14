"""Frozen-specification 2026 holdout audit for every >=2-board candidate.

Targets next-session *closing* board promotion, not a tradable return. Candidate
features are signal-date EOD fields only. No theme, auction, VWAP, next-session
gap, inferred fill, or future terminal height enters a model. 2026 is an
out-of-time diagnostic period, not a pristine blind test: this project has
already inspected portions of its 2026 history in earlier versions.
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

VERSION = "v256-relay-promotion-holdout-2"
TRAIN_END_EXCLUSIVE = "2026-01-01"
TEST_START = "2026-01-05"
IDENTITY = ["boards", "height_gap", "same_height_rivals", "unique_space",
            "height_share", "is_space_leader"]
EMOTION = ["emotion_score", "promotion_2to3", "promotion_3to4",
           "promotion_mid", "previous_limit_tail_loss", "failed_rate",
           "down_limit_rate", "breadth_health", "ladder_health",
           "loss_effect_health", "crowding_risk", "regime_fragility"]
PRICE_VOLUME = ["turnover", "signal_amount", "amplitude", "turnover_vs_prev", "price"]
SPECS = {
    "identity": IDENTITY,
    "identity_emotion": IDENTITY + EMOTION,
    "identity_emotion_price_volume": IDENTITY + EMOTION + PRICE_VOLUME,
}
FORBIDDEN_MODEL_FIELDS = {
    "promoted_next_day", "next_open_gap", "net_return", "large_loss",
    "trade_high", "trade_low", "trade_close", "exit_close", "terminal_board",
    "additional_boards", "execution_status", "buyable_eod", "one_word_proxy",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959963984540054
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0, center-half)), float(min(1, center+half))]


def _summarize(rows: pd.DataFrame, score: str | None = None) -> dict:
    n = len(rows)
    y = rows.promoted_next_day.astype(int)
    result = {
        "selected_days": n, "successes": int(y.sum()),
        "promotion_rate": float(y.mean()) if n else None,
        "wilson95": wilson(int(y.sum()), n),
        "one_word_signal_rows": int(rows.one_word_proxy.eq(1).sum()),
        "inferred_touch_fill_rows": int(rows.execution_status.eq("inferred_daily_limit_touch_fill").sum()),
        "actual_fill_rows": int(rows.execution_status.eq("actual_fill").sum()),
        "known_proxy_net_rows": int(rows.net_return.notna().sum()),
        "mean_proxy_net_observed_only": float(rows.net_return.mean()) if rows.net_return.notna().any() else None,
        "proxy_positive_observed_only": float(rows.loc[rows.net_return.notna(), "net_return"].gt(0).mean())
        if rows.net_return.notna().any() else None,
        "proxy_big_loss_observed_only": float(rows.loc[rows.large_loss.notna(), "large_loss"].mean())
        if rows.large_loss.notna().any() else None,
    }
    if score is not None:
        result["selected_score_brier"] = float(brier_score_loss(y, rows[score])) if n else None
        result["selected_score_auc"] = float(roc_auc_score(y, rows[score])) if y.nunique() == 2 else None
    return result


def _paired(model: pd.DataFrame, baseline: pd.DataFrame, seed: int = 256) -> dict:
    pair = model[["date", "promoted_next_day"]].merge(
        baseline[["date", "promoted_next_day"]], on="date", validate="one_to_one",
        suffixes=("_model", "_baseline"))
    pair = pair.sort_values("date")
    diff = (pair.promoted_next_day_model - pair.promoted_next_day_baseline).to_numpy(dtype=float)
    if len(diff) == 0:
        return {"paired_days": 0, "difference": None, "bootstrap95": None}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(20000)])
    block_length = min(5, len(diff))
    starts = rng.integers(0, len(diff), size=(10000, int(np.ceil(len(diff)/block_length))))
    indices = ((starts[:, :, None]+np.arange(block_length)) % len(diff)).reshape(10000, -1)[:, :len(diff)]
    block_boot = diff[indices].mean(axis=1)
    return {
        "paired_days": int(len(diff)),
        "model_successes": int(pair.promoted_next_day_model.sum()),
        "baseline_successes": int(pair.promoted_next_day_baseline.sum()),
        "difference": float(diff.mean()),
        "bootstrap95": [float(v) for v in np.quantile(boot, [.025, .975])],
        "circular_5_date_block95": [float(v) for v in np.quantile(block_boot, [.025, .975])],
        "model_only_wins": int((diff == 1).sum()),
        "baseline_only_wins": int((diff == -1).sum()),
    }


def _proxy_delta(model: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    def values(frame: pd.DataFrame) -> dict:
        net = pd.to_numeric(frame.net_return, errors="coerce")
        loss = pd.to_numeric(frame.large_loss, errors="coerce")
        return {
            "known_net_rows": int(net.notna().sum()),
            "mean_net_observed_only": float(net.mean()) if net.notna().any() else None,
            "positive_rate_observed_only": float(net[net.notna()].gt(0).mean()) if net.notna().any() else None,
            "big_loss_rate_observed_only": float(loss[loss.notna()].mean()) if loss.notna().any() else None,
        }
    m, b = values(model), values(baseline)
    return {
        "model": m, "baseline": b,
        "mean_net_difference": (m["mean_net_observed_only"] - b["mean_net_observed_only"]
                                if m["mean_net_observed_only"] is not None and b["mean_net_observed_only"] is not None else None),
        "positive_rate_difference": (m["positive_rate_observed_only"] - b["positive_rate_observed_only"]
                                     if m["positive_rate_observed_only"] is not None and b["positive_rate_observed_only"] is not None else None),
        "big_loss_rate_difference": (m["big_loss_rate_observed_only"] - b["big_loss_rate_observed_only"]
                                    if m["big_loss_rate_observed_only"] is not None and b["big_loss_rate_observed_only"] is not None else None),
        "warning": "inferred daily-OHLC proxy; no actual fills",
    }


def _top(rows: pd.DataFrame, score: str | None = None) -> pd.DataFrame:
    if score is None:
        order = ["date", "boards", "code"]
        asc = [True, False, True]
    else:
        order = ["date", score, "boards", "code"]
        asc = [True, False, False, True]
    return rows.sort_values(order, ascending=asc, kind="mergesort") \
        .drop_duplicates("date", keep="first").copy()


def _same_height_score(rows: pd.DataFrame) -> pd.Series:
    """Shared within each date/height: this cannot distinguish equal-height peers."""
    return (-rows.height_gap.fillna(999)
            + 0.10*rows.unique_space.fillna(0)
            - 0.01*rows.same_height_rivals.fillna(999)
            + 0.05*rows.is_space_leader.fillna(0))


def heuristic_top(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    """Untuned price-volume comparators; added after v1, hence exploratory."""
    specs = {
        "low_turnover": (["turnover", "boards"], [True, False]),
        "low_amplitude": (["amplitude", "boards"], [True, False]),
        "one_word_then_low_turnover": (["one_word_proxy", "turnover", "boards"], [False, True, False]),
    }
    columns, ascending = specs[name]
    return rows.sort_values(["date", *columns, "code"],
                           ascending=[True, *ascending, True], na_position="last",
                           kind="mergesort").drop_duplicates("date").copy()


def split_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Purge labels unavailable by model freeze; evaluate only complete dates.

    Unknown outcomes never remove a stock from the prediction universe. Date
    completeness affects evaluation only, not scoring or candidate selection.
    """
    next_date = pd.to_datetime(rows.trade_date, errors="coerce")
    valid = rows.promoted_next_day.isin([0, 1]) & rows.promotion_label_status.eq(
        "verified_from_next_daily_limit_close")
    train = rows[(rows.date < TRAIN_END_EXCLUSIVE) & valid
                 & (next_date < pd.Timestamp(TRAIN_END_EXCLUSIVE))
                 & (next_date > pd.to_datetime(rows.date))].copy()
    test = rows[rows.date >= TEST_START].copy()
    complete = valid.loc[test.index].groupby(test.date).all()
    return train, test, complete


def _fit(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> np.ndarray:
    # Weight trading dates equally to avoid unusually broad days dominating.
    weights = 1/train.groupby("date").date.transform("size")
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1, max_iter=3000, random_state=256),
    )
    pipeline.fit(train[fields], train.promoted_next_day,
                 logisticregression__sample_weight=weights)
    return pipeline.predict_proba(test[fields])[:, 1]


def load_rows(source: Path) -> tuple[pd.DataFrame, dict]:
    if set(sum(SPECS.values(), [])) & FORBIDDEN_MODEL_FIELDS:
        raise AssertionError("future or execution field included as model input")
    rows = pd.read_csv(source, dtype={"code": str}, low_memory=False)
    required = {"date", "trade_date", "code", "boards", "promoted_next_day", "promotion_label_status",
                "one_word_proxy", "source_timestamp", "execution_status", "net_return",
                "large_loss", *set(sum(SPECS.values(), []))}
    if required - set(rows):
        raise ValueError(f"missing fields: {sorted(required-set(rows))}")
    if rows.duplicated(["date", "code"]).any():
        raise ValueError("duplicate candidate/date key")
    rows["date"] = rows.date.astype(str)
    rows["code"] = rows.code.astype(str).str.zfill(6)
    for col in set(sum(SPECS.values(), [])) | {"promoted_next_day", "one_word_proxy", "net_return", "large_loss"}:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows = rows[rows.boards.ge(2)].copy()
    ts = pd.to_datetime(rows.source_timestamp, errors="coerce")
    dates = pd.to_datetime(rows.date, errors="coerce")
    if ts.isna().any() or dates.isna().any() or (ts.dt.date != dates.dt.date).any():
        raise ValueError("source timestamp/date mismatch")
    if (ts.dt.hour < 15).any():
        raise ValueError("signal EOD timestamp before 15:00")
    audit = {
        "source_rows": int(len(rows)), "source_dates": int(rows.date.nunique()),
        "source_range": [str(rows.date.min()), str(rows.date.max())],
        "theme_available_rows": int(rows.theme_status.eq("verified").sum()) if "theme_status" in rows else None,
        "complete_label_dates": int(rows.groupby("date").promoted_next_day.apply(lambda x: x.notna().all()).sum()),
    }
    return rows, audit


def run(data_path: str, output_dir: str) -> dict:
    source = Path(data_path)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("immutable output directory already exists")
    rows, audit = load_rows(source)
    train, test_all, complete = split_rows(rows)
    test = test_all[test_all.date.isin(complete[complete].index)].copy()
    if train.empty or test.empty:
        raise ValueError("empty training or holdout rows")
    all_scored = test_all.copy()
    for name, fields in SPECS.items():
        all_scored["score_"+name] = _fit(train, test_all, fields)
    # Keep identity ranking and trade authorization as separate, auditable channels.
    # This dataset has no verified theme/intraday/actual-fill evidence, so no row
    # may be promoted from a diagnostic rank into a trading permission.
    all_scored["identity_channel"] = "LEADER_IDENTITY_DIAGNOSTIC"
    all_scored["trade_authorization"] = "CASH"
    all_scored["trade_authorization_reason"] = (
        "theme_intraday_actual_fill_evidence_missing"
    )
    all_scored["evaluation_date_complete"] = all_scored.date.map(complete)
    scored = all_scored[all_scored.evaluation_date_complete].copy()

    results = {}
    selections = {}
    for scope_name, predicate in {
        "all_boards": lambda x: x.boards.ge(2),
        "two_board": lambda x: x.boards.eq(2),
        "three_plus": lambda x: x.boards.ge(3),
    }.items():
        for eligibility, eligible in {
            "all_identity": scored,
            "non_one_price": scored[scored.one_word_proxy.eq(0)],
        }.items():
            pool = eligible[predicate(eligible)].copy()
            key = scope_name + ":" + eligibility
            if pool.empty:
                results[key] = {"status": "NO_CANDIDATES"}
                continue
            base = _top(pool)
            pool["score_same_height_identity"] = _same_height_score(pool)
            same_height_base = _top(pool, "score_same_height_identity")
            selections[key+":baseline"] = base
            selections[key+":same_height_identity_baseline"] = same_height_base
            heuristics = {name: heuristic_top(pool, name) for name in
                          ("low_turnover", "low_amplitude", "one_word_then_low_turnover")}
            for hname, hrows in heuristics.items():
                selections[key+":"+hname] = hrows
            item = {
                "pool_rows": int(len(pool)), "pool_dates": int(pool.date.nunique()),
                "baseline": _summarize(base),
                "same_height_identity_baseline": _summarize(same_height_base),
                "same_height_warning": "shared fields within height; not an independent strong comparator",
                "heuristics": {name: _summarize(hrows) for name, hrows in heuristics.items()},
                "models": {},
            }
            for name in SPECS:
                score = "score_"+name
                chosen = _top(pool, score)
                selections[key+":"+name] = chosen
                item["models"][name] = {
                    "summary": _summarize(chosen, score),
                    "paired_vs_same_pool_height": _paired(chosen, base),
                    "paired_vs_same_height_identity": _paired(chosen, same_height_base),
                    "proxy_trade_outcome_vs_same_pool_height": _proxy_delta(chosen, base),
                    "paired_vs_price_volume_heuristics": {hname: _paired(chosen, hrows)
                                                          for hname, hrows in heuristics.items()},
                }
            results[key] = item

    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "target": "next_session_closing_next_board_promotion",
        "cutoffs": {"train_end_exclusive": TRAIN_END_EXCLUSIVE, "test_start": TEST_START},
        "train_rows": int(len(train)), "train_dates": int(train.date.nunique()),
        "max_training_label_date": str(train.trade_date.max()),
        "training_rows_purged_label_unavailable_at_freeze": int(
            ((rows.date < TRAIN_END_EXCLUSIVE) & rows.promoted_next_day.notna()).sum()-len(train)),
        "test_rows": int(len(test)), "test_dates": int(test.date.nunique()),
        "test_days_dropped_for_incomplete_labels": int((~complete).sum()),
        "source_audit": audit, "model_features": SPECS, "results": results,
        "interpretation": {
            "is_pristine_blind_test": False,
            "reason": "2026 archive had already been inspected by earlier project versions",
            "market_emotion_usage": "diagnostic model feature only, not a validated trading gate",
            "trade_win_rate_available": False,
            "actual_fill_count": 0,
            "theme_and_intraday_missing": True,
            "universe_survivorship_risk": True,
            "entry_authorized": False,
            "v2_heuristics_and_block_intervals": "POST_HOC_DIAGNOSTIC_COMPARATORS_NOT_NEW_BLIND_PROOF",
            "promotion_is_eventual_leader": False,
            "decision_contract": {
                "identity_channel": "LEADER_IDENTITY_DIAGNOSTIC",
                "trade_authorization": "CASH",
                "reason": "theme_intraday_actual_fill_evidence_missing",
            },
        },
        "source_hashes": {str(source.resolve()): sha(source), str(Path(__file__).resolve()): sha(Path(__file__))},
    }
    target.mkdir(parents=True)
    prediction_columns = ["date", "trade_date", "code", "name", "boards", "one_word_proxy",
                          *["score_"+name for name in SPECS],
                          "identity_channel", "trade_authorization",
                          "trade_authorization_reason", "promoted_next_day",
                          "evaluation_date_complete"]
    all_scored[prediction_columns].to_csv(target/"all_candidate_predictions_LABEL_ONLY.csv",
                                        index=False, encoding="utf-8-sig")
    for key, chosen in selections.items():
        name = key.replace(":", "__") + ".csv"
        chosen.to_csv(target/name, index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    result = run(ns.data, ns.output)
    print(json.dumps({k: result[k] for k in ("version", "status", "train_rows", "test_rows", "test_dates")},
                     ensure_ascii=False))
