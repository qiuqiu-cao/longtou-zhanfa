from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leader_system_v3 import RegularizedLogistic, auc_score, expected_calibration_error


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "v3" / "market_emotion.csv"
DEFAULT_OUTPUT = ROOT / "results" / "v4"
ACTIVE_STATES = {"启动", "升温", "加速", "修复"}

FEATURES = [
    "breadth_health", "relay_appetite", "ladder_health", "loss_effect_health",
    "crowding_risk", "regime_fragility", "false_acceleration", "promotion_mid",
    "failed_rate", "down_limit_count", "previous_limit_tail_loss", "yesterday_premium",
    "height_change", "limit_change", "one_word_share", "space_feedback",
    "space_feedback_3d", "emotion_state_age_scaled", "state_start", "state_warming",
    "state_acceleration", "state_consensus", "state_divergence", "state_retreat",
    "state_repair",
]

TARGETS = {
    "adverse_state_next": "next emotion_state is 大分歧 or 退潮",
    "relay_damage_next": (
        "next loss_effect_health < 0.35 OR "
        "(next promotion_mid < 0.20 AND next down_limit_count >= 5)"
    ),
}

LOCKED_PARAMETERS = {
    "initial_history": 160,
    "retrain_every": 20,
    "l2": 1.5,
    "epochs": 700,
    "learning_rate": 0.03,
    "folds": 3,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"unsupported json value {type(value)!r}")


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("date").reset_index(drop=True).copy()
    next_state = ordered["emotion_state"].shift(-1)
    adverse = next_state.isin({"大分歧", "退潮"}).astype(float)
    adverse[next_state.isna()] = math.nan
    ordered["adverse_state_next"] = adverse

    next_loss = pd.to_numeric(ordered["loss_effect_health"], errors="coerce").shift(-1)
    next_mid = pd.to_numeric(ordered["promotion_mid"], errors="coerce").shift(-1)
    next_down = pd.to_numeric(ordered["down_limit_count"], errors="coerce").shift(-1)
    loss_known = next_loss.notna()
    ladder_known = next_mid.notna() & next_down.notna()
    loss_damage = loss_known & (next_loss < 0.35)
    ladder_damage = ladder_known & (next_mid < 0.20) & (next_down >= 5)
    damage = pd.Series(math.nan, index=ordered.index, dtype=float)
    damage[loss_damage | ladder_damage] = 1.0
    damage[loss_known & ladder_known & ~(loss_damage | ladder_damage)] = 0.0
    ordered["relay_damage_next"] = damage
    return ordered


def causal_predictions(
    frame: pd.DataFrame,
    target: str,
    *,
    initial_history: int = LOCKED_PARAMETERS["initial_history"],
    retrain_every: int = LOCKED_PARAMETERS["retrain_every"],
    l2: float = LOCKED_PARAMETERS["l2"],
    epochs: int = LOCKED_PARAMETERS["epochs"],
    learning_rate: float = LOCKED_PARAMETERS["learning_rate"],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Prequential forecasts using only labels observable by the current close.

    Row t-1's next-day label becomes known at close t.  The model predicting row
    t therefore trains only on rows before t and is kept frozen between scheduled
    retraining dates.
    """
    probabilities = np.full(len(frame), math.nan)
    causal_base = np.full(len(frame), math.nan)
    model: RegularizedLogistic | None = None
    latest_model: dict[str, Any] | None = None
    for position in range(initial_history, len(frame)):
        available = frame.iloc[:position].dropna(subset=[target]).copy()
        if len(available) < initial_history or available[target].nunique() < 2:
            continue
        positive = float(available[target].sum())
        causal_base[position] = (positive + 1.0) / (len(available) + 2.0)
        if model is None or (position - initial_history) % retrain_every == 0:
            model = RegularizedLogistic.fit(
                available,
                target,
                FEATURES,
                l2=l2,
                epochs=epochs,
                lr=learning_rate,
            )
            latest_model = model.to_dict()
        probabilities[position] = float(model.predict(frame.iloc[[position]])[0])
    return probabilities, causal_base, latest_model


def binary_metrics(y: np.ndarray, p: np.ndarray, baseline: np.ndarray | None = None) -> dict[str, Any]:
    valid = np.isfinite(y) & np.isfinite(p)
    yv = y[valid]
    pv = np.clip(p[valid], 1e-6, 1 - 1e-6)
    if not len(yv):
        return {"n": 0}
    rate = float(yv.mean())
    brier = float(np.mean((pv - yv) ** 2))
    constant_brier = float(np.mean((rate - yv) ** 2))
    logloss = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
    result = {
        "n": int(len(yv)),
        "positive_rate": rate,
        "auc": auc_score(yv, pv),
        "brier": brier,
        "brier_fold_constant": constant_brier,
        "brier_skill_vs_fold_constant": 1 - brier / constant_brier if constant_brier > 0 else None,
        "ece_5bin": expected_calibration_error(yv, pv),
        "log_loss": logloss,
    }
    if baseline is not None:
        bv = np.clip(baseline[valid], 1e-6, 1 - 1e-6)
        base_valid = np.isfinite(bv)
        if base_valid.all():
            base_brier = float(np.mean((bv - yv) ** 2))
            base_logloss = float(-np.mean(yv * np.log(bv) + (1 - yv) * np.log(1 - bv)))
            result.update({
                "baseline_brier": base_brier,
                "brier_skill_vs_baseline": 1 - brier / base_brier if base_brier > 0 else None,
                "baseline_log_loss": base_logloss,
                "log_loss_improvement_vs_baseline": base_logloss - logloss,
            })
    cutoff = float(np.quantile(pv, 0.80))
    high = pv >= cutoff
    high_precision = float(yv[high].mean()) if high.any() else math.nan
    result.update({
        "diagnostic_top_quintile_cutoff": cutoff,
        "diagnostic_top_quintile_n": int(high.sum()),
        "diagnostic_top_quintile_precision": high_precision,
        "diagnostic_top_quintile_lift": high_precision / rate if rate > 0 else None,
    })
    return result


def chronological_folds(valid_positions: np.ndarray, folds: int = 3) -> list[np.ndarray]:
    return [part for part in np.array_split(valid_positions, folds) if len(part)]


def evaluate_target(
    frame: pd.DataFrame,
    target: str,
    probabilities: np.ndarray,
    baseline: np.ndarray,
    baseline_name: str,
) -> dict[str, Any]:
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    valid_positions = np.flatnonzero(np.isfinite(y) & np.isfinite(probabilities) & np.isfinite(baseline))

    def fold_metrics(positions_to_split: np.ndarray) -> list[dict[str, Any]]:
        reports = []
        for number, positions in enumerate(
            chronological_folds(positions_to_split, LOCKED_PARAMETERS["folds"]), 1
        ):
            mask = np.zeros(len(frame), dtype=bool)
            mask[positions] = True
            reports.append({
                "fold": number,
                "start_date": str(frame.iloc[positions[0]]["date"]),
                "end_date": str(frame.iloc[positions[-1]]["date"]),
                **binary_metrics(np.where(mask, y, math.nan), probabilities, baseline),
            })
        return reports

    def stable(reports: list[dict[str, Any]]) -> bool:
        return bool(reports) and all(
            report.get("brier_skill_vs_baseline", -math.inf) > 0
            and report.get("log_loss_improvement_vs_baseline", -math.inf) > 0
            and (report.get("auc") or 0) > 0.5
            for report in reports
        )

    fold_reports = fold_metrics(valid_positions)
    overall = binary_metrics(y, probabilities, baseline)
    active_mask = frame["emotion_state"].isin(ACTIVE_STATES).to_numpy(bool)
    active_positions = valid_positions[active_mask[valid_positions]]
    active_folds = fold_metrics(active_positions)
    active_overall = binary_metrics(np.where(active_mask, y, math.nan), probabilities, baseline)
    all_date_stable = stable(fold_reports)
    active_stable = stable(active_folds)
    return {
        "target": target,
        "definition": TARGETS[target],
        "baseline": baseline_name,
        "overall": overall,
        "folds": fold_reports,
        "all_date_stability_gate": all_date_stable,
        "active_state_evaluation": {
            "states": sorted(ACTIVE_STATES),
            "overall": active_overall,
            "folds": active_folds,
            "stability_gate": active_stable,
        },
        "historical_stability_gate": all_date_stable and active_stable,
        "gate_rule": "all-date and active-state views must each pass all 3 chronological folds: Brier skill vs named causal baseline >0, lower log loss, AUC >0.5",
    }


def markdown_report(card: dict[str, Any]) -> str:
    lines = [
        "# 情绪挑战者V4模型卡",
        "",
        f"- 生成时间：{card['generated_at']}",
        f"- 数据区间：{card['data_start']} 至 {card['data_end']}",
        f"- 状态：{card['deployment_status']}",
        f"- 原因：{card['deployment_reason']}",
        "- 纪律：旧数据全部视为开发样本；未经过锁定后的未来验证，不接管V3.8。",
        "",
    ]
    for target, report in card["targets"].items():
        overall = report["overall"]
        lines += [
            f"## {target}",
            "",
            f"定义：{report['definition']}",
            "",
            f"整体样本 {overall.get('n', 0)}，AUC {overall.get('auc')}，"
            f"Brier技能（相对因果基线）{overall.get('brier_skill_vs_baseline')}，"
            f"历史三段稳定闸门：{report['historical_stability_gate']}。",
            f"比较基线：{report['baseline']}。",
            "",
            "| 时间段 | 样本 | AUC | Brier技能 | 对数损失改善 | 高风险20%精确率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for fold in report["folds"]:
            lines.append(
                f"| {fold['start_date']}—{fold['end_date']} | {fold.get('n')} | {fold.get('auc')} | "
                f"{fold.get('brier_skill_vs_baseline')} | {fold.get('log_loss_improvement_vs_baseline')} | "
                f"{fold.get('diagnostic_top_quintile_precision')} |"
            )
        lines.append("")
        active = report["active_state_evaluation"]
        active_overall = active["overall"]
        lines += [
            "### 仅允许开仓的情绪状态",
            "",
            f"启动/升温/加速/修复共 {active_overall.get('n', 0)} 个可评估日，"
            f"AUC {active_overall.get('auc')}，Brier技能 {active_overall.get('brier_skill_vs_baseline')}，"
            f"三段闸门：{active['stability_gate']}。",
            "",
            "| 时间段 | 样本 | AUC | Brier技能 | 对数损失改善 | 高风险20%精确率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for fold in active["folds"]:
            lines.append(
                f"| {fold['start_date']}—{fold['end_date']} | {fold.get('n')} | {fold.get('auc')} | "
                f"{fold.get('brier_skill_vs_baseline')} | {fold.get('log_loss_improvement_vs_baseline')} | "
                f"{fold.get('diagnostic_top_quintile_precision')} |"
            )
        lines.append("")
    latest = card["latest_forecast"]
    lines += [
        "## 最新研究预测",
        "",
        f"- 日期：{latest['date']}",
        f"- 状态逆风概率：{latest['adverse_state_probability']}",
        f"- 接力损伤概率：{latest['relay_damage_probability']}",
        "- 这些概率是开发期研究输出，不是实盘胜率或买卖信号。",
        "",
    ]
    return "\n".join(lines)


def run(input_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = pd.read_csv(input_path)
    required = {"date", "emotion_state", "loss_effect_health", "promotion_mid", "down_limit_count", *FEATURES}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"market emotion input missing columns: {missing}")
    frame = add_targets(raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = frame[["date", "emotion_state", "adverse_state_next", "relay_damage_next"]].copy()
    target_reports: dict[str, Any] = {}
    model_payloads: dict[str, Any] = {}
    for target in TARGETS:
        probability, causal_base, latest_model = causal_predictions(frame, target)
        predictions[f"{target}_probability"] = probability
        predictions[f"{target}_causal_base"] = causal_base
        comparison = (
            pd.to_numeric(frame["emotion_adverse_markov_probability"], errors="coerce").to_numpy(float)
            if target == "adverse_state_next"
            else causal_base
        )
        baseline_name = "CAUSAL_MARKOV" if target == "adverse_state_next" else "EXPANDING_BETA_BASE_RATE"
        target_reports[target] = evaluate_target(
            frame, target, probability, comparison, baseline_name
        )
        model_payloads[target] = latest_model

    latest = predictions.iloc[-1]
    historical_gate = all(report["historical_stability_gate"] for report in target_reports.values())
    card = {
        "version": "4.0-research",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_start": str(frame.iloc[0]["date"]),
        "data_end": str(frame.iloc[-1]["date"]),
        "rows": len(frame),
        "features": FEATURES,
        "targets": target_reports,
        "locked_parameters": LOCKED_PARAMETERS,
        "historical_joint_gate": historical_gate,
        "deployment_status": "RESEARCH_ONLY",
        "deployment_reason": (
            "historical data has already informed model development; require a locked future period and stable gains before promotion"
        ),
        "latest_forecast": {
            "date": str(latest["date"]),
            "adverse_state_probability": latest["adverse_state_next_probability"],
            "adverse_state_markov_baseline": frame.iloc[-1]["emotion_adverse_markov_probability"],
            "relay_damage_probability": latest["relay_damage_next_probability"],
            "relay_damage_base_rate": latest["relay_damage_next_causal_base"],
            "label_status": "RIGHT_CENSORED_FUTURE_UNKNOWN",
        },
        "data_quality": {
            "market_data": "INFERRED_FROM_DAILY_MARKET_POOLS",
            "theme_data": "UNAVAILABLE_IN_THIS_CHALLENGER",
            "level2": "UNAVAILABLE",
            "warning": "daily close features cannot verify auction cancellations, queues, VWAP or intraday sector synchronization",
        },
        "integrity": {
            "input_path": str(input_path.resolve()),
            "input_sha256": file_sha256(input_path),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "existing_v3_freeze_unchanged_required": True,
        },
        "models": model_payloads,
    }
    predictions.to_csv(output_dir / "emotion_challenger_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "emotion_challenger_model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    (output_dir / "emotion_challenger_model_card.md").write_text(markdown_report(card), encoding="utf-8")
    return card


def lock_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": card["version"],
        "development_data_end": card["data_end"],
        "development_input_sha256": card["integrity"]["input_sha256"],
        "script_sha256": card["integrity"]["script_sha256"],
        "features": card["features"],
        "targets": TARGETS,
        "locked_parameters": card["locked_parameters"],
        "first_fresh_date_rule": f"forecast date must be later than {card['data_end']}",
        "status": "LOCKED_RESEARCH_CHALLENGER",
    }


def create_lock(card: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    path = output_dir / "emotion_challenger_lock.json"
    payload = lock_payload(card)
    lock_id = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]
    locked = {
        "lock_id": lock_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key not in {"lock_id", "created_at"}}
        if canonical_json(comparable) != canonical_json(payload):
            raise RuntimeError("existing V4 lock differs; never overwrite a development lock")
        return existing
    path.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    return locked


def verify_lock(card: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    path = output_dir / "emotion_challenger_lock.json"
    if not path.exists():
        return {"status": "MISSING", "errors": ["emotion_challenger_lock.json not found"]}
    locked = json.loads(path.read_text(encoding="utf-8"))
    expected = lock_payload(card)
    errors = []
    for key, value in expected.items():
        if canonical_json(locked.get(key)) != canonical_json(value):
            errors.append(f"{key} mismatch")
    expected_id = hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest()[:16]
    if locked.get("lock_id") != expected_id:
        errors.append("lock_id mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "lock_id": locked.get("lock_id"),
        "development_data_end": locked.get("development_data_end"),
        "errors": errors,
        "assurance": "LOCAL_TAMPER_EVIDENT_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal V4 emotion-risk challenger outside the frozen V3.8 model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock", action="store_true", help="lock code, targets, parameters and development data end")
    parser.add_argument("--verify-lock", action="store_true")
    args = parser.parse_args()
    card = run(args.input, args.output)
    lock = create_lock(card, args.output) if args.lock else None
    verification = verify_lock(card, args.output) if args.verify_lock or args.lock else None
    summary = {
        "deployment_status": card["deployment_status"],
        "historical_joint_gate": card["historical_joint_gate"],
        "latest_forecast": card["latest_forecast"],
        "targets": {
            name: {
                "overall": report["overall"],
                "historical_stability_gate": report["historical_stability_gate"],
            }
            for name, report in card["targets"].items()
        },
        "lock": ({"lock_id": lock["lock_id"], "development_data_end": lock["development_data_end"]}
                 if lock else None),
        "lock_verification": verification,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
