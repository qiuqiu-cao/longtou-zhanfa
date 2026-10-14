"""Audit first-board -> second-board promotion with seal-speed quality factors.

The module deliberately treats an early limit-up as ambiguous.  Early sealing can
be genuine demand, but it can also be a thin-liquidity/one-price board whose first
tradable opening is dangerous.  All features are known at the signal-date close;
labels come only from the immediately following trading session.

Outputs are research-only.  A closing limit-up is not assumed to be buyable and
the promotion label is not a complete-trade profit label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
FEATURES_SPEED_ONLY = ["first_seal_minute"]
FEATURES_QUALITY_BASE = [
    "first_seal_minute", "last_minus_first_minute", "failed_seal_count",
    "turnover", "amount_to_float_cap", "seal_fund_to_float_cap",
    "one_price_proxy", "ultra_fast", "fast_seal", "thin_control_risk",
    "validated_fast_seal", "industry_first_board_count",
    "industry_all_board_count", "industry_higher_board_count",
    "market_first_board_count", "market_relay_count",
]
FEATURES_EMOTION = [
    "emotion_score", "failed_rate", "down_limit_count", "promotion_1to2",
    "relay_appetite", "crowding_risk", "regime_fragility",
    "emotion_adverse_next_probability", "emotion_supportive_regime",
]
FEATURES_QUALITY = FEATURES_QUALITY_BASE + FEATURES_EMOTION


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hhmmss_to_minute(value) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).strip().split(".")[0].zfill(6)
    if len(text) != 6 or not text.isdigit():
        return np.nan
    hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:])
    if not (9 <= hour <= 15 and 0 <= minute < 60 and 0 <= second < 60):
        return np.nan
    return (hour * 60 + minute + second / 60) - (9 * 60 + 15)


def _number(row: pd.Series, name: str, default=np.nan) -> float:
    value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else default


def _normalize_pool(frame: pd.DataFrame, signal_date: str) -> pd.DataFrame:
    rename = {
        "代码": "code", "名称": "name", "连板数": "boards", "成交额": "amount",
        "流通市值": "float_cap", "换手率": "turnover", "封板资金": "seal_fund",
        "首次封板时间": "first_seal", "最后封板时间": "last_seal",
        "炸板次数": "failed_seal_count", "所属行业": "industry",
    }
    result = frame.rename(columns=rename).copy()
    # Older archived pools do not contain amount/float-cap/seal-fund.  Preserve
    # those fields as genuinely missing instead of silently treating them as 0;
    # the imputer and its missing indicators then keep data provenance visible.
    required = {"code", "name", "boards", "turnover", "first_seal",
                "last_seal", "failed_seal_count", "industry"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"pool {signal_date} missing columns: {sorted(missing)}")
    for optional in ("amount", "float_cap", "seal_fund"):
        if optional not in result:
            result[optional] = np.nan
    result["code"] = result.code.astype(str).str.zfill(6)
    result["name"] = result.name.astype(str)
    result = result[result.code.str.startswith(MAINBOARD_PREFIXES)].copy()
    result = result[~result.name.str.contains(r"ST|退", case=False, na=False)].copy()
    for column in ("boards", "amount", "float_cap", "turnover", "seal_fund", "failed_seal_count"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["signal_date"] = signal_date
    return result


def discover_pools(august_root: Path, raw_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    paths: dict[str, Path] = {}
    for path in sorted(august_root.glob("zt_pool_*.csv")):
        date = path.stem.rsplit("_", 1)[-1]
        if len(date) == 8 and date.isdigit():
            paths[pd.Timestamp(date).strftime("%Y-%m-%d")] = path
    for directory in sorted(raw_root.iterdir() if raw_root.exists() else []):
        path = directory / "limit_up.csv"
        if directory.is_dir() and len(directory.name) == 8 and directory.name.isdigit() and path.exists():
            paths[pd.Timestamp(directory.name).strftime("%Y-%m-%d")] = path
    pools = {date: _normalize_pool(pd.read_csv(path, dtype={"代码": str}), date)
             for date, path in sorted(paths.items())}
    return pools, {date: str(path.resolve()) for date, path in sorted(paths.items())}


def build_dataset(pools: dict[str, pd.DataFrame], calendar: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    sessions = pd.to_datetime(calendar["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").tolist()
    next_session = {sessions[i]: sessions[i + 1] for i in range(len(sessions) - 1)}
    records, skipped = [], []
    for signal_date, pool in sorted(pools.items()):
        target_date = next_session.get(signal_date)
        if not target_date or target_date not in pools:
            skipped.append({"signal_date": signal_date, "reason": "IMMEDIATE_NEXT_SESSION_POOL_MISSING"})
            continue
        next_pool = pools[target_date].set_index("code", drop=False)
        first = pool[pool.boards.eq(1)].copy()
        industry_first = first.groupby("industry", dropna=False).size().to_dict()
        industry_all = pool.groupby("industry", dropna=False).size().to_dict()
        industry_higher = pool[pool.boards.gt(1)].groupby("industry", dropna=False).size().to_dict()
        market_first_count, market_relay_count = len(first), int(pool.boards.gt(1).sum())
        for _, row in first.iterrows():
            first_minute = hhmmss_to_minute(row.first_seal)
            last_minute = hhmmss_to_minute(row.last_seal)
            turnover = _number(row, "turnover")
            amount = _number(row, "amount")
            float_cap = _number(row, "float_cap")
            seal_fund = _number(row, "seal_fund")
            failed = _number(row, "failed_seal_count", 0.0)
            amount_ratio = amount / float_cap if float_cap and float_cap > 0 else np.nan
            seal_ratio = seal_fund / float_cap if float_cap and float_cap > 0 else np.nan
            ultra_fast = bool(pd.notna(first_minute) and first_minute <= 15)  # <=09:30
            fast = bool(pd.notna(first_minute) and first_minute <= 20)  # <=09:35
            one_price = bool(str(row.first_seal).split(".")[0].zfill(6) == "092500"
                             and str(row.last_seal).split(".")[0].zfill(6) == "092500"
                             and failed == 0)
            thin_control = bool(fast and (one_price or turnover < 3 or amount_ratio < .02))
            validated_fast = bool(
                fast and not one_price and 4 <= turnover <= 18
                and .03 <= amount_ratio <= .20 and failed <= 1
                and pd.notna(seal_ratio) and seal_ratio >= .006
            )
            promoted = False
            if row.code in next_pool.index:
                next_row = next_pool.loc[row.code]
                if isinstance(next_row, pd.DataFrame):
                    next_boards = pd.to_numeric(next_row.boards, errors="coerce").max()
                else:
                    next_boards = _number(next_row, "boards")
                promoted = bool(next_boards >= 2)
            records.append({
                "code": row.code, "name": row["name"], "signal_date": signal_date,
                "target_date": target_date, "industry_proxy": row.industry,
                "boards": 1, "promotion_label": int(promoted),
                "first_seal_minute": first_minute,
                "last_minus_first_minute": max(last_minute - first_minute, 0)
                if pd.notna(first_minute) and pd.notna(last_minute) else np.nan,
                "failed_seal_count": failed, "turnover": turnover,
                "amount_to_float_cap": amount_ratio,
                "seal_fund_to_float_cap": seal_ratio,
                "one_price_proxy": int(one_price), "ultra_fast": int(ultra_fast),
                "fast_seal": int(fast), "thin_control_risk": int(thin_control),
                "validated_fast_seal": int(validated_fast),
                "industry_first_board_count": int(industry_first.get(row.industry, 0)),
                "industry_all_board_count": int(industry_all.get(row.industry, 0)),
                "industry_higher_board_count": int(industry_higher.get(row.industry, 0)),
                "market_first_board_count": market_first_count,
                "market_relay_count": market_relay_count,
                "feature_available_at": signal_date + "T15:05:00+08:00",
            })
    return pd.DataFrame(records), skipped


def enrich_emotion(rows: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Attach only signal-date-close emotion fields; never forward-fill them."""
    available = [name for name in FEATURES_EMOTION if name in market.columns]
    selected = market[["date", *available]].copy()
    selected["date"] = pd.to_datetime(selected.date, errors="coerce").dt.strftime("%Y-%m-%d")
    if "emotion_supportive_regime" not in selected:
        state = market.get("emotion_state", pd.Series(index=market.index, dtype=object))
        selected["emotion_supportive_regime"] = state.isin(["启动", "升温", "加速", "修复"]).astype(int)
    enriched = rows.merge(selected.rename(columns={"date": "signal_date"}),
                          on="signal_date", how="left", validate="many_to_one")
    for name in FEATURES_EMOTION:
        if name not in enriched:
            enriched[name] = np.nan
        enriched[name] = pd.to_numeric(enriched[name], errors="coerce")
    return enriched


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _top1(predictions: pd.DataFrame, score: str) -> dict:
    selected = (predictions.sort_values(["signal_date", score, "code"], ascending=[True, False, True])
                .groupby("signal_date", as_index=False).first())
    return {
        "sessions": int(len(selected)),
        "successes": int(selected.promotion_label.sum()),
        "precision": float(selected.promotion_label.mean()) if len(selected) else None,
        "rows": selected[["signal_date", "code", "name", score, "promotion_label"]].to_dict("records"),
    }


def _fit_predict(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> np.ndarray:
    weights = 1 / train.groupby("signal_date").signal_date.transform("size")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(C=.5, max_iter=3000, random_state=124),
    )
    model.fit(train[features], train.promotion_label,
              logisticregression__sample_weight=weights)
    return model.predict_proba(validation[features])[:, 1]


def prequential_audit(rows: pd.DataFrame, min_train_sessions: int = 12) -> tuple[pd.DataFrame, dict]:
    dates = sorted(rows.signal_date.unique())
    predictions = []
    for index in range(min_train_sessions, len(dates)):
        date = dates[index]
        train = rows[rows.signal_date.isin(dates[:index])].copy()
        validation = rows[rows.signal_date.eq(date)].copy()
        if train.promotion_label.nunique() != 2 or validation.empty:
            continue
        validation["speed_probability"] = _fit_predict(train, validation, FEATURES_SPEED_ONLY)
        validation["base_quality_probability"] = _fit_predict(
            train, validation, FEATURES_QUALITY_BASE)
        validation["emotion_quality_probability"] = _fit_predict(
            train, validation, FEATURES_QUALITY)
        predictions.append(validation)
    if not predictions:
        return pd.DataFrame(), {"status": "INSUFFICIENT_PREQUENTIAL_WINDOWS"}
    scored = pd.concat(predictions, ignore_index=True)
    earliest = scored.copy()
    earliest["earliest_score"] = -earliest.first_seal_minute.fillna(9999)
    quality_metrics = {
        "candidate_count": int(len(scored)),
        "sessions": int(scored.signal_date.nunique()),
        "positive_rate": float(scored.promotion_label.mean()),
        "speed_only_auc": _safe_auc(scored.promotion_label, scored.speed_probability),
        "speed_only_brier": float(brier_score_loss(scored.promotion_label, scored.speed_probability)),
        "base_quality_auc": _safe_auc(scored.promotion_label, scored.base_quality_probability),
        "base_quality_brier": float(brier_score_loss(
            scored.promotion_label, scored.base_quality_probability)),
        "emotion_quality_auc": _safe_auc(
            scored.promotion_label, scored.emotion_quality_probability),
        "emotion_quality_brier": float(brier_score_loss(
            scored.promotion_label, scored.emotion_quality_probability)),
        "earliest_seal_daily_top1": _top1(earliest, "earliest_score"),
        "speed_model_daily_top1": _top1(scored, "speed_probability"),
        "base_quality_daily_top1": _top1(scored, "base_quality_probability"),
        "emotion_quality_daily_top1": _top1(scored, "emotion_quality_probability"),
    }
    return scored, quality_metrics


def factor_summary(rows: pd.DataFrame) -> dict:
    definitions = {
        "ultra_fast_le_0930": rows.ultra_fast.eq(1),
        "fast_le_0935": rows.fast_seal.eq(1),
        "fast_thin_control_risk": rows.thin_control_risk.eq(1),
        "fast_validated_demand": rows.validated_fast_seal.eq(1),
        "not_fast_after_0935": rows.fast_seal.eq(0),
        "one_price_proxy": rows.one_price_proxy.eq(1),
    }
    result = {}
    for name, mask in definitions.items():
        part = rows[mask]
        result[name] = {
            "n": int(len(part)), "sessions": int(part.signal_date.nunique()),
            "promotion_rate": float(part.promotion_label.mean()) if len(part) else None,
        }
    return result


def current_scores(rows: pd.DataFrame, current_pool: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or current_pool.empty or rows.promotion_label.nunique() != 2:
        return pd.DataFrame()
    # The current pool is supplied as unlabeled feature rows by appending it to a
    # temporary two-pool dictionary outside this function.  This helper expects
    # rows with promotion_label missing.
    current = current_pool.copy()
    # The emotion-augmented challenger is intentionally not used for ranking
    # unless it first beats this base-quality model in a larger frozen audit.
    probability = _fit_predict(rows, current, FEATURES_QUALITY_BASE)
    current["research_score_not_probability"] = probability
    return current.sort_values(["research_score_not_probability", "code"], ascending=[False, True])


def run(august_root: Path, raw_root: Path, market_path: Path, output: Path,
        min_train_sessions: int = 12, calendar_path: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable v124 output")
    pools, paths = discover_pools(august_root, raw_root)
    market = pd.read_csv(market_path, dtype={"date": str})
    calendar_source = calendar_path or market_path
    calendar = pd.read_csv(calendar_source, dtype={"date": str})
    dataset, skipped = build_dataset(pools, calendar)
    dataset = enrich_emotion(dataset, market)
    if dataset.empty:
        raise ValueError("no adjacent-session first-board observations")
    scored, evaluation = prequential_audit(dataset, min_train_sessions)

    # Score the latest pool by constructing its close-known features with a dummy
    # next pool, then dropping the unknown label.  No outcome fields are read.
    latest_date = max(pools)
    latest_pool = pools[latest_date]
    first = latest_pool[latest_pool.boards.eq(1)].copy()
    industry_first = first.groupby("industry", dropna=False).size().to_dict()
    industry_all = latest_pool.groupby("industry", dropna=False).size().to_dict()
    industry_higher = latest_pool[latest_pool.boards.gt(1)].groupby("industry", dropna=False).size().to_dict()
    current_records = []
    for _, row in first.iterrows():
        first_minute, last_minute = hhmmss_to_minute(row.first_seal), hhmmss_to_minute(row.last_seal)
        turnover, amount, float_cap = _number(row, "turnover"), _number(row, "amount"), _number(row, "float_cap")
        seal_fund, failed = _number(row, "seal_fund"), _number(row, "failed_seal_count", 0)
        amount_ratio = amount / float_cap if float_cap and float_cap > 0 else np.nan
        seal_ratio = seal_fund / float_cap if float_cap and float_cap > 0 else np.nan
        fast = bool(pd.notna(first_minute) and first_minute <= 20)
        ultra = bool(pd.notna(first_minute) and first_minute <= 15)
        one_price = bool(str(row.first_seal).split(".")[0].zfill(6) == "092500"
                         and str(row.last_seal).split(".")[0].zfill(6) == "092500" and failed == 0)
        thin = bool(fast and (one_price or turnover < 3 or amount_ratio < .02))
        valid = bool(fast and not one_price and 4 <= turnover <= 18 and .03 <= amount_ratio <= .20
                     and failed <= 1 and pd.notna(seal_ratio) and seal_ratio >= .006)
        current_records.append({
            "code": row.code, "name": row["name"], "signal_date": latest_date,
            "first_seal_minute": first_minute,
            "last_minus_first_minute": max(last_minute - first_minute, 0)
            if pd.notna(first_minute) and pd.notna(last_minute) else np.nan,
            "failed_seal_count": failed, "turnover": turnover,
            "amount_to_float_cap": amount_ratio, "seal_fund_to_float_cap": seal_ratio,
            "one_price_proxy": int(one_price), "ultra_fast": int(ultra), "fast_seal": int(fast),
            "thin_control_risk": int(thin), "validated_fast_seal": int(valid),
            "industry_first_board_count": int(industry_first.get(row.industry, 0)),
            "industry_all_board_count": int(industry_all.get(row.industry, 0)),
            "industry_higher_board_count": int(industry_higher.get(row.industry, 0)),
            "market_first_board_count": int(len(first)),
            "market_relay_count": int(latest_pool.boards.gt(1).sum()),
        })
    current_frame = enrich_emotion(pd.DataFrame(current_records), market)
    current = current_scores(dataset, current_frame)
    if not current.empty:
        current["execution_gate"] = np.where(
            current.thin_control_risk.eq(1), "WATCH_UNTRADABLE_OR_CONTROL_RISK",
            "RESEARCH_ONLY_NO_ENTRY_AUTHORIZATION")
        current = current.sort_values(
            ["thin_control_risk", "research_score_not_probability", "code"],
            ascending=[True, False, True])

    output.mkdir(parents=True, exist_ok=False)
    dataset.to_csv(output / "first_board_observations.csv", index=False, encoding="utf-8-sig")
    if not scored.empty:
        scored.to_csv(output / "prequential_predictions.csv", index=False, encoding="utf-8-sig")
    if not current.empty:
        current.to_csv(output / "current_scores_RESEARCH_ONLY.csv", index=False, encoding="utf-8-sig")
    report = {
        "version": "v124-first-board-speed-quality-audit-2",
        "status": "DIAGNOSTIC_ONLY",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "target": "next_session_closing_second_board_promotion",
        "observations": int(len(dataset)), "sessions": int(dataset.signal_date.nunique()),
        "latest_scored_signal_date": latest_date,
        "calendar_sessions": int(calendar.date.nunique()),
        "factor_summary_full_development": factor_summary(dataset),
        "prequential_evaluation": evaluation,
        "current_top10": current.head(10).to_dict("records") if not current.empty else [],
        "governance": {
            "early_seal_is_ambiguous": True,
            "thin_control_risk_never_positive_by_itself": True,
            "industry_fields_are_proxy_not_verified_theme": True,
            "rank_score_named_probability": False,
            "deployment_authorized": False,
            "ranking_champion": "base_quality_without_emotion",
            "emotion_challenger_promoted": False,
            "emotion_challenger_rule": "must improve both prequential AUC and Brier before promotion",
            "minimum_forward_predictions_before_use": 60,
            "minimum_forward_promotions_before_use": 20,
            "trading_calendar_separate_from_observation_dates": bool(calendar_path),
            "missing_immediate_next_pool_is_unavailable_not_negative": True,
        },
        "skipped_pairs": skipped,
        "limitations": [
            "promotion is a closing-board label, not complete-trade profitability",
            "limit-up close does not prove queue access or a fill",
            "industry co-movement is not a verified real-theme roster",
            "available seal-time history is short and may not cover delisted names",
            "thresholds are predeclared hypotheses and must not be tuned on these results",
        ],
        "source_hashes": {**{path: sha256(Path(path)) for path in paths.values()},
                          str(market_path.resolve()): sha256(market_path),
                          str(calendar_source.resolve()): sha256(calendar_source),
                          str(Path(__file__).resolve()): sha256(Path(__file__))},
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--august-root", type=Path, default=Path("data/2026-08"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/v3/raw"))
    parser.add_argument("--market", type=Path, default=Path("data/v3/market_emotion.csv"))
    parser.add_argument("--calendar", type=Path,
                        help="verified trading-session calendar; never use sparse review dates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-sessions", type=int, default=12)
    args = parser.parse_args()
    report = run(args.august_root, args.raw_root, args.market, args.output,
                 args.min_train_sessions, args.calendar)
    print(json.dumps({
        "status": report["status"], "observations": report["observations"],
        "sessions": report["sessions"], "prequential": report["prequential_evaluation"],
        "current_top10": report["current_top10"],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
