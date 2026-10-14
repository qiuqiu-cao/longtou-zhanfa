"""Walk-forward diagnostic for V253's two-board leader-seed stage.

This is an identity backtest, not a trade backtest.  It uses only signal-date
features reconstructed in ``full_pool_eod_features.csv`` and the next-session
promotion label from the immutable LABEL_ONLY outcome archive.  It deliberately
does not use theme, auction, VWAP or actual-fill fields, because those fields
are unavailable in this historical source.
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

VERSION = "v254-two-board-leader-seed-walkforward-1"

IDENTITY = [
    "height_gap_to_known_top", "same_height_known_count", "sole_known_top_proxy",
    "candidate_pool_count", "unknown_height_pool_count",
]
EMOTION = [
    "market_yesterday_premium", "market_previous_limit_tail_loss",
    "market_failed_tail_loss", "market_failed_rate", "market_down_limit_count",
    "market_promotion_overall", "market_promotion_1to2", "market_promotion_2to3",
    "market_promotion_mid", "market_space_feedback_3d", "market_middle_density",
    "market_ladder_gap_count", "market_breadth_health", "market_relay_appetite",
    "market_ladder_health", "market_loss_effect_health", "market_crowding_risk",
    "market_regime_fragility", "market_false_acceleration", "market_emotion_score",
    "market_state_start", "market_state_warming", "market_state_acceleration",
    "market_state_consensus", "market_state_divergence", "market_state_retreat",
    "market_state_repair",
]
PRICE_VOLUME = ["open_gap_pct", "amplitude_pct", "amount_to_float_cap", "turnover",
                "one_price_day_proxy"]
MODELS = {
    "identity": IDENTITY,
    "identity_emotion": IDENTITY + EMOTION,
    "identity_emotion_price_volume": IDENTITY + EMOTION + PRICE_VOLUME,
    "identity_emotion_price_volume_non_one_price": IDENTITY + EMOTION + [
        "open_gap_pct", "amplitude_pct", "amount_to_float_cap", "turnover"
    ],
}
WINDOWS = [
    ("2024-07-01", "2024-10-01"),
    ("2024-10-01", "2025-01-01"),
    ("2025-01-01", "2025-04-01"),
    ("2025-04-01", "2025-07-01"),
    ("2025-07-01", "2025-10-01"),
    ("2025-10-01", "2026-01-01"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    margin = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [float(max(0, centre-margin)), float(min(1, centre+margin))]


def load_rows(features_path: Path, outcomes_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path, dtype={"code": str}, low_memory=False)
    outcomes = pd.read_csv(outcomes_path, dtype={"code": str}, low_memory=False)
    required_f = {"code", "signal_date", "boards", "board_group", "feature_status",
                  "feature_available_not_before", *set(sum(MODELS.values(), []))}
    required_o = {"code", "signal_date", "promotion_LABEL_ONLY", "promotion_state"}
    missing = {"features": sorted(required_f - set(features)),
               "outcomes": sorted(required_o - set(outcomes))}
    if any(missing.values()):
        raise ValueError(json.dumps(missing, ensure_ascii=False))
    features["code"] = features.code.astype(str).str.zfill(6)
    outcomes["code"] = outcomes.code.astype(str).str.zfill(6)
    rows = features.merge(outcomes[["code", "signal_date", "promotion_LABEL_ONLY",
                                    "promotion_state"]],
                          on=["code", "signal_date"], validate="one_to_one")
    rows["boards"] = pd.to_numeric(rows.boards, errors="coerce")
    rows = rows[rows.boards.eq(2) & rows.board_group.eq("CONSECUTIVE_BOARD")].copy()
    rows["target"] = pd.to_numeric(rows.promotion_LABEL_ONLY, errors="coerce")
    rows = rows[rows.target.notna()].copy()
    rows["signal_date"] = rows.signal_date.astype(str)
    for col in set(sum(MODELS.values(), [])):
        rows[col] = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    available = pd.to_datetime(rows.feature_available_not_before, utc=True, errors="coerce")
    close = pd.to_datetime(rows.signal_date + "T15:05:00+08:00", utc=True, errors="coerce")
    if available.isna().any() or (available > close).any():
        raise ValueError("feature availability is later than signal-date close")
    return rows.sort_values(["signal_date", "code"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, validation: pd.DataFrame, cols: list[str]) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1, max_iter=3000, random_state=254),
    )
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model.fit(train[cols], train.target, logisticregression__sample_weight=weights)
    return model.predict_proba(validation[cols])[:, 1]


def baseline_score(rows: pd.DataFrame) -> pd.Series:
    # A frozen non-learned same-height proxy: lower height gap, sole-top flag,
    # fewer same-height rivals. It is not a probability and uses no outcomes.
    return (-rows.height_gap_to_known_top.fillna(999)
            + 0.10*rows.sole_known_top_proxy.fillna(0)
            - 0.01*rows.same_height_known_count.fillna(999))


def select_top(rows: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return rows.sort_values(["signal_date", score_col, "code"],
                            ascending=[True, False, True], kind="mergesort") \
               .drop_duplicates("signal_date", keep="first").copy()


def summarize(selected: pd.DataFrame, score_col: str, calibrated: bool) -> dict:
    y = selected.target.astype(int)
    return {
        "selected_days": int(len(selected)),
        "successes_2_to_3": int(y.sum()),
        "promotion_rate": float(y.mean()) if len(y) else None,
        "promotion_wilson_95": wilson(int(y.sum()), int(len(y))),
        "candidate_auc": (float(roc_auc_score(selected.target, selected[score_col]))
                          if selected.target.nunique() == 2 else None),
        "candidate_brier": (float(brier_score_loss(selected.target, selected[score_col]))
                            if calibrated else None),
        "one_price_signal_rows": int(selected.one_price_day_proxy.fillna(0).eq(1).sum()),
    }


def paired_delta(left: pd.DataFrame, right: pd.DataFrame, reps: int = 10000) -> dict:
    p = left[["signal_date", "target"]].merge(right[["signal_date", "target"]],
        on="signal_date", suffixes=("_left", "_right"), validate="one_to_one")
    if p.empty:
        return {"paired_days": 0, "difference": None, "bootstrap95": [None, None]}
    d = (p.target_left - p.target_right).astype(float).to_numpy()
    rng = np.random.default_rng(254)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(reps)])
    return {"paired_days": int(len(d)), "left_successes": int(p.target_left.sum()),
            "right_successes": int(p.target_right.sum()), "difference": float(d.mean()),
            "bootstrap95": [float(x) for x in np.quantile(boot, [.025, .975])]}


def run(features_path: str, outcomes_path: str, output_dir: str) -> dict:
    features_file, outcomes_file = Path(features_path), Path(outcomes_path)
    rows = load_rows(features_file, outcomes_file)
    fold_reports, selected = [], {name: [] for name in MODELS}
    selected["same_height_identity_baseline"] = []
    selected["same_height_identity_baseline_non_one_price"] = []
    for start, end in WINDOWS:
        validation = rows[(rows.signal_date >= start) & (rows.signal_date < end)].copy()
        train = rows[rows.signal_date < start].copy()
        fold = {"start": start, "end_exclusive": end,
                "training_rows": int(len(train)), "validation_rows": int(len(validation)),
                "training_days": int(train.signal_date.nunique()),
                "validation_days": int(validation.signal_date.nunique())}
        if train.empty or validation.empty or train.target.nunique() < 2:
            fold["status"] = "INSUFFICIENT_TRAIN_OR_VALIDATION"
            fold_reports.append(fold)
            continue
        fold["status"] = "OK"
        validation["score_same_height_identity_baseline"] = baseline_score(validation)
        base = select_top(validation, "score_same_height_identity_baseline")
        selected["same_height_identity_baseline"].append(base)
        fold["baseline"] = summarize(base, "score_same_height_identity_baseline", False)
        non_one_validation = validation[validation.one_price_day_proxy.fillna(0).eq(0)].copy()
        non_one_validation["score_same_height_identity_baseline_non_one_price"] = baseline_score(non_one_validation)
        non_one_base = select_top(non_one_validation, "score_same_height_identity_baseline_non_one_price")
        selected["same_height_identity_baseline_non_one_price"].append(non_one_base)
        fold["baseline_non_one_price"] = summarize(
            non_one_base, "score_same_height_identity_baseline_non_one_price", False)
        fold["models"] = {}
        non_one_name = "identity_emotion_price_volume_non_one_price"
        for name, cols in MODELS.items():
            if name == non_one_name:
                continue
            col = "score_" + name
            validation[col] = fit_predict(train, validation, cols)
            top = select_top(validation, col)
            selected[name].append(top)
            fold["models"][name] = summarize(top, col, True)
        non_one_cols = MODELS[non_one_name]
        non_one_train = train[train.one_price_day_proxy.fillna(0).eq(0)].copy()
        if len(non_one_validation) and non_one_train.target.nunique() >= 2:
            non_one_validation["score_" + non_one_name] = fit_predict(
                non_one_train, non_one_validation, non_one_cols)
            non_one_top = select_top(non_one_validation, "score_" + non_one_name)
            # Replace the all-candidate selection for this dedicated model;
            # it is evaluated only on the same non-one-price denominator.
            selected[non_one_name].append(non_one_top)
            fold["models"][non_one_name] = summarize(
                non_one_top, "score_" + non_one_name, True)
        else:
            fold["models"][non_one_name] = {"status": "INSUFFICIENT_NON_ONE_PRICE_TRAIN_OR_VALIDATION"}
        fold_reports.append(fold)

    summaries = {}
    for name, parts in selected.items():
        all_selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        is_baseline = name.startswith("same_height_identity_baseline")
        summaries[name] = summarize(all_selected, "score_" + name,
                                    not is_baseline) if len(all_selected) else {}

    comparison = {}
    baseline = pd.concat(selected["same_height_identity_baseline"], ignore_index=True)
    for name in MODELS:
        chosen = pd.concat(selected[name], ignore_index=True)
        comparison[name + "_minus_baseline"] = paired_delta(chosen, baseline)
    non_one_baseline = pd.concat(selected["same_height_identity_baseline_non_one_price"], ignore_index=True)
    if selected["identity_emotion_price_volume_non_one_price"]:
        non_one_chosen = pd.concat(selected["identity_emotion_price_volume_non_one_price"], ignore_index=True)
        comparison["identity_emotion_price_volume_non_one_price_minus_same_denominator_baseline"] = paired_delta(
            non_one_chosen, non_one_baseline)
        # Post-hoc V255 environment slice, inspected after the ungated result.
        # This is a diagnostic market filter, not independent proof or a stock
        # score. A later untouched forward period is required for validation.
        gate_model = non_one_chosen[
            non_one_chosen.market_state_retreat.eq(0)
            & non_one_chosen.market_state_divergence.eq(0)
        ].copy()
        gate_baseline = non_one_baseline[
            non_one_baseline.market_state_retreat.eq(0)
            & non_one_baseline.market_state_divergence.eq(0)
        ].copy()
        market_gate = {
            "rule": "market_state_retreat==0 AND market_state_divergence==0",
            "evaluation_status": "POST_HOC_EXPLORATORY_NOT_PRE_REGISTERED",
            "model": summarize(gate_model, "score_identity_emotion_price_volume_non_one_price", True),
            "same_denominator_baseline": summarize(
                gate_baseline, "score_same_height_identity_baseline_non_one_price", False),
            "paired_model_minus_baseline": paired_delta(gate_model, gate_baseline),
            "purpose": "environment permission diagnostic; not individual probability bonus",
        }
    else:
        market_gate = {"status": "NO_NON_ONE_PRICE_SELECTIONS"}

    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "target": "2_to_3_next_session_promotion_LABEL_ONLY",
        "source_cutoff": str(rows.signal_date.max()), "rows": int(len(rows)),
        "date_clusters": int(rows.signal_date.nunique()),
        "target_rate_all_candidates": float(rows.target.mean()),
        "models": MODELS,
        "walk_forward": {
            "method": "expanding chronological windows; train strictly before validation",
            "windows": fold_reports, "blind_complete_trade_count": 0,
        },
        "aggregate_selected": summaries,
        "paired_comparison_vs_same_height_baseline": comparison,
        "fixed_market_gate": market_gate,
        "data_quality": {
            "feature_status": rows.feature_status.value_counts(dropna=False).to_dict(),
            "feature_archive": "historical_reconstruction_not_forward_archive",
            "promotion_label": "daily limit-up pool LABEL_ONLY; identity target, not fill/profit",
            "theme_fields": "unavailable in this source",
            "auction_vwap_level2": "unavailable",
            "survivorship": "historical universe incomplete",
        },
        "conclusions": {
            "v253_two_board_module_validated": False,
            "leader_identity_improvement_proven": False,
            "trade_win_rate_proven": False,
            "entry_authorized": False,
        },
        "blocking_reasons": [
            "二板历史身份标签没有完整题材与同题材竞争分母",
            "封板时间/封板金额/竞价/VWAP不可用，无法验证完整二板条件",
            "特征为事后重建而非当时归档，且历史股票宇宙不完整",
            "晋级标签不等于买后收益，真实成交数为0",
        ],
        "source_hashes": {str(features_file.resolve()): sha(features_file),
                          str(outcomes_file.resolve()): sha(outcomes_file)},
    }
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.mkdir(parents=True)
    # selected files are written above only after target exists; this block is
    # intentionally moved below in the caller-safe implementation.
    for name, parts in selected.items():
        all_selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(all_selected):
            all_selected.to_csv(target / f"selected_{name}.csv", index=False, encoding="utf-8-sig")
    report["artifact_hashes"] = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.features, ns.outcomes, ns.output), ensure_ascii=False, indent=2))
