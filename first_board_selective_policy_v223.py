"""V223 three-target, time-isolated first-board abstention policy.

Promotion identity, executable B+2 payoff and B+1 large-loss risk are modeled
separately.  The daily promotion top-one is frozen first; a failed gate becomes
cash and is never replaced by the runner-up.  All output is diagnostic.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "v223-first-board-selective-policy-1"
FEATURES = [
    "first_seal_minute", "last_minus_first_minute", "failed_seal_count",
    "turnover", "amount_to_float_cap", "seal_fund_to_float_cap",
    "one_price_proxy", "ultra_fast", "fast_seal", "thin_control_risk",
    "validated_fast_seal", "industry_first_board_count",
    "industry_all_board_count", "industry_higher_board_count",
    "market_first_board_count", "market_relay_count", "emotion_score",
    "failed_rate", "down_limit_count", "promotion_1to2", "relay_appetite",
    "crowding_risk", "regime_fragility", "emotion_adverse_next_probability",
    "emotion_supportive_regime", "sealed_by_1030", "low_failed_seal",
    "quick_reseal", "industry_pioneer_rank_pct", "industry_breadth_3plus",
    "industry_ladder_support", "supported_pioneer", "community_low_position",
    "community_evidence_count", "close_quality_gate",
]


def _truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(train: pd.DataFrame, score: pd.DataFrame, target: str, seed: int) -> np.ndarray:
    y = pd.to_numeric(train[target], errors="coerce").astype(int)
    if y.nunique() < 2:
        return np.repeat(float(y.mean()), len(score))
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=.5, max_iter=3000, class_weight="balanced", random_state=seed),
    )
    pipe.fit(train[FEATURES], y)
    return pipe.predict_proba(score[FEATURES])[:, 1]


def _eligible(rows: pd.DataFrame, score_date: str, target: str) -> pd.DataFrame:
    prior = rows[rows.signal_date.astype(str).lt(score_date)].copy()
    if target == "b2_positive":
        prior = prior[
            prior.fill_status.eq("FILLED_OPEN_PROXY")
            & pd.to_numeric(prior.net_return_B_plus_2, errors="coerce").notna()
            & prior.target_date.astype(str).lt(score_date)
        ]
    elif target == "b1_big_loss":
        prior = prior[
            prior.fill_status.eq("FILLED_OPEN_PROXY")
            & pd.to_numeric(prior.big_loss_B_plus_1, errors="coerce").notna()
            & prior.target_date.astype(str).le(score_date)
        ]
    return prior


def prequential(rows: pd.DataFrame, promotion_predictions: pd.DataFrame,
                min_train_sessions: int = 10) -> pd.DataFrame:
    required = set(FEATURES + ["code", "name", "signal_date", "target_date",
                               "fill_status", "net_return_B_plus_2", "big_loss_B_plus_1"])
    missing = required - set(rows)
    if missing:
        raise ValueError(f"V223 fields missing: {sorted(missing)}")
    promotion = promotion_predictions[["signal_date", "code", "community_probability"]].copy()
    promotion["code"] = promotion.code.astype(str).str.zfill(6)
    outputs = []
    for date in sorted(promotion.signal_date.astype(str).unique()):
        current = rows[rows.signal_date.astype(str).eq(date)].copy()
        if current.empty:
            continue
        profit = _eligible(rows, date, "b2_positive")
        risk = _eligible(rows, date, "b1_big_loss")
        if min(profit.signal_date.nunique(), risk.signal_date.nunique()) < min_train_sessions:
            continue
        profit = profit.assign(b2_positive=pd.to_numeric(profit.net_return_B_plus_2).gt(0).astype(int))
        risk = risk.assign(b1_big_loss=pd.to_numeric(risk.big_loss_B_plus_1).astype(int))
        current["b2_positive_probability"] = _model(profit, current, "b2_positive", 223)
        current["b1_big_loss_probability"] = _model(risk, current, "b1_big_loss", 224)
        current = current.merge(promotion[promotion.signal_date.astype(str).eq(date)],
                                on=["signal_date", "code"], how="inner", validate="one_to_one")
        current["profit_train_sessions"] = profit.signal_date.nunique()
        current["risk_train_sessions"] = risk.signal_date.nunique()
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def apply_policy(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if scored.empty:
        return scored, {"status": "INSUFFICIENT_TIME_ISOLATED_SAMPLE"}
    decisions = []
    for date, day in scored.groupby("signal_date", sort=True):
        day = day.copy()
        # Freeze the identity target first.  This prevents hindsight replacement.
        top = day.sort_values(["community_probability", "code"], ascending=[False, True]).iloc[0].copy()
        top["promotion_rank_percentile"] = day.community_probability.rank(pct=True).loc[top.name]
        top["payoff_rank_percentile"] = day.b2_positive_probability.rank(pct=True).loc[top.name]
        top["risk_rank_percentile"] = day.b1_big_loss_probability.rank(pct=True).loc[top.name]
        accepted = bool(
            top.promotion_rank_percentile >= .80
            and top.payoff_rank_percentile >= .80
            and top.risk_rank_percentile <= .50
            and _truth(top.close_quality_gate)
        )
        top["policy_decision"] = "PAPER_ACCEPT" if accepted else "ABSTAIN_TO_CASH"
        top["replacement_after_abstention"] = False
        top["entry_authorized"] = False
        decisions.append(top)
    frozen = pd.DataFrame(decisions).reset_index(drop=True)

    def metrics(frame: pd.DataFrame) -> dict:
        filled = frame[frame.fill_status.eq("FILLED_OPEN_PROXY")]
        b2 = pd.to_numeric(filled.net_return_B_plus_2, errors="coerce").dropna()
        loss = pd.to_numeric(filled.big_loss_B_plus_1, errors="coerce").dropna()
        # Promotion is an identity outcome for the whole frozen denominator;
        # payoff and loss remain restricted to causally executable rows.
        promotions = pd.to_numeric(frame.get("promotion_label"), errors="coerce").dropna()
        return {
            "decisions": int(len(frame)), "filled": int(len(filled)),
            "promotion_rate": float(promotions.mean()) if len(promotions) else None,
            "B_plus_2_mean_net_return": float(b2.mean()) if len(b2) else None,
            "B_plus_2_win_rate": float(b2.gt(0).mean()) if len(b2) else None,
            "B_plus_1_big_loss_rate": float(loss.mean()) if len(loss) else None,
        }

    accepted = frozen[frozen.policy_decision.eq("PAPER_ACCEPT")]
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "fixed_policy": {
            "freeze_identity": "daily highest prequential promotion score",
            "promotion_percentile_min": .80, "payoff_percentile_min": .80,
            "risk_percentile_max": .50, "close_quality_required": True,
            "replacement_after_abstention": False,
        },
        "forced_top1": metrics(frozen), "selective_policy": metrics(accepted),
        "coverage": float(len(accepted) / len(frozen)) if len(frozen) else None,
        "claims_win_rate_improvement": False,
        "entry_authorized": False,
    }
    return frozen, report


def score_current(rows: pd.DataFrame, historical_scored: pd.DataFrame,
                  current: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return current
    profit = rows[rows.fill_status.eq("FILLED_OPEN_PROXY") &
                  pd.to_numeric(rows.net_return_B_plus_2, errors="coerce").notna()].copy()
    risk = rows[rows.fill_status.eq("FILLED_OPEN_PROXY") &
                pd.to_numeric(rows.big_loss_B_plus_1, errors="coerce").notna()].copy()
    profit["b2_positive"] = pd.to_numeric(profit.net_return_B_plus_2).gt(0).astype(int)
    risk["b1_big_loss"] = pd.to_numeric(risk.big_loss_B_plus_1).astype(int)
    out = current.copy()
    out["b2_positive_probability"] = _model(profit, out, "b2_positive", 223)
    out["b1_big_loss_probability"] = _model(risk, out, "b1_big_loss", 224)
    out["promotion_rank_percentile"] = out.community_probability_not_calibrated.rank(pct=True)
    out["payoff_rank_percentile"] = out.b2_positive_probability.rank(pct=True)
    out["risk_rank_percentile"] = out.b1_big_loss_probability.rank(pct=True)
    theme_gate = out.theme_research_gate.astype(str).str.lower().isin({"true", "1"})
    eligible = out[theme_gate]
    if eligible.empty:
        out["current_policy_decision"] = "ABSTAIN_NO_THEME_ELIGIBLE_CANDIDATE"
        out["entry_authorized"] = False
        return out
    # V220 freezes identity only inside the pre-close theme-research gate.
    top_index = eligible.community_probability_not_calibrated.idxmax()
    out["current_policy_decision"] = "NOT_FROZEN_TOP1"
    accepted = bool(
        out.loc[top_index, "promotion_rank_percentile"] >= .80
        and out.loc[top_index, "payoff_rank_percentile"] >= .80
        and out.loc[top_index, "risk_rank_percentile"] <= .50
        and bool(out.loc[top_index, "close_quality_gate"])
        and bool(theme_gate.loc[top_index])
    )
    out.loc[top_index, "current_policy_decision"] = "PAPER_ACCEPT_REQUIRES_V221" if accepted else "ABSTAIN_TO_CASH"
    out["entry_authorized"] = False
    return out


def run(rows_path: str, promotion_path: str, current_path: str, output_dir: str,
        min_train_sessions: int = 10) -> dict:
    sources = [Path(rows_path), Path(promotion_path), Path(current_path)]
    rows = pd.read_csv(sources[0], dtype={"code": str})
    promotion = pd.read_csv(sources[1], dtype={"code": str})
    current = pd.read_csv(sources[2], dtype={"code": str})
    scored = prequential(rows, promotion, min_train_sessions)
    frozen, report = apply_policy(scored)
    current_scores = score_current(rows, scored, current)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("refusing to overwrite immutable V223 output")
    out.mkdir(parents=True)
    scored.to_csv(out / "prequential_three_target_scores.csv", index=False, encoding="utf-8-sig")
    frozen.to_csv(out / "frozen_top1_selective_decisions.csv", index=False, encoding="utf-8-sig")
    current_scores.to_csv(out / "current_three_target_watchlist.csv", index=False, encoding="utf-8-sig")
    report["current_frozen_top1"] = (
        current_scores[current_scores.current_policy_decision.ne("NOT_FROZEN_TOP1")][
            ["code", "name", "current_policy_decision", "community_probability_not_calibrated",
             "b2_positive_probability", "b1_big_loss_probability"]
        ].to_dict("records") if not current_scores.empty else []
    )
    report["source_hashes"] = {str(p.resolve()): _sha(p) for p in sources}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--promotion", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-train-sessions", type=int, default=10)
    ns = ap.parse_args()
    print(json.dumps(run(ns.rows, ns.promotion, ns.current, ns.output,
                         ns.min_train_sessions), ensure_ascii=False, indent=2))
