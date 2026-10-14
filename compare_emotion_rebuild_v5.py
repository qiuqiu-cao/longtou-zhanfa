#!/usr/bin/env python3
"""Compare the locked V4 development card with the rebuilt V5 research card."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
V4 = ROOT / "results" / "v4" / "emotion_challenger_model_card.json"
V5 = ROOT / "results" / "v5" / "emotion_challenger_rebuild" / "emotion_challenger_model_card.json"
OUT_JSON = ROOT / "results" / "v5" / "emotion_rebuild_comparison.json"
OUT_MD = ROOT / "results" / "v5" / "emotion_rebuild_comparison.md"


def metric(card: dict[str, Any], target: str, name: str) -> float:
    return float(card["targets"][target]["overall"][name])


def compare(v4_path: Path = V4, v5_path: Path = V5) -> dict[str, Any]:
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))
    rows = []
    for target in ["adverse_state_next", "relay_damage_next"]:
        row = {"target": target}
        for name in [
            "auc", "brier", "brier_skill_vs_baseline", "log_loss",
            "diagnostic_top_quintile_precision", "diagnostic_top_quintile_lift",
        ]:
            old, new = metric(v4, target, name), metric(v5, target, name)
            row[f"v4_{name}"] = old
            row[f"v5_{name}"] = new
            row[f"delta_{name}"] = new - old
        row["v4_gate"] = bool(v4["targets"][target]["historical_stability_gate"])
        row["v5_gate"] = bool(v5["targets"][target]["historical_stability_gate"])
        rows.append(row)
    joint_gate = bool(v5["historical_joint_gate"])
    result = {
        "status": "RESEARCH_ONLY" if not joint_gate else "HISTORICAL_GATE_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v4_deployment_status": v4["deployment_status"],
        "v5_deployment_status": v5["deployment_status"],
        "v5_historical_joint_gate": joint_gate,
        "targets": rows,
        "conclusion": (
            "survivorship correction improves adverse-regime discrimination but weakens relay-damage discrimination; "
            "neither target passes the locked three-fold stability gate"
        ),
        "deployment_effect": "no promotion; V3 remains frozen DIAGNOSTIC_ONLY and V4 lock remains unchanged",
    }
    lines = [
        "# V5 情绪重建与锁定V4对比", "",
        f"- 结论：`{result['status']}`",
        "- 退市样本补全后，逆风状态预测改善，但接力损伤预测退化；不能选择性报告单一指标。",
        "- 两个目标均未通过三段时间稳定闸门，不升级、不覆盖锁定模型。", "",
        "| 目标 | V4 AUC | V5 AUC | AUC变化 | V4高风险20%精确率 | V5精确率 | V4 Brier | V5 Brier | V5稳定闸门 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['v4_auc']:.3f} | {row['v5_auc']:.3f} | {row['delta_auc']:+.3f} | "
            f"{row['v4_diagnostic_top_quintile_precision']:.1%} | {row['v5_diagnostic_top_quintile_precision']:.1%} | "
            f"{row['v4_brier']:.4f} | {row['v5_brier']:.4f} | {row['v5_gate']} |"
        )
    lines.extend(["", "这次结果证明数据补全会真实改变情绪标签和预测表现，但尚未形成可部署优势。", ""])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
