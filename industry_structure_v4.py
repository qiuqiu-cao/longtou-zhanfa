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

from leader_system_v3 import normalize_pool, parse_clock, zfill_code


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "v3" / "raw"
CANDIDATES = ROOT / "data" / "v3" / "leader_candidates.csv"
OUTPUT = ROOT / "results" / "v4" / "structure"
ARCHIVE = ROOT / "data" / "v4"
EVIDENCE_LEVEL = "INDUSTRY_PROXY_NOT_THEME"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pool(day_dir: Path, name: str, signal_date: str) -> pd.DataFrame:
    path = day_dir / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return normalize_pool(pd.read_csv(path, dtype={"代码": str}), signal_date)


def clean_industry(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "未知"} else text


def append_dedup(frame: pd.DataFrame, path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype={"code": str})
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def attach_temporal_context(
    structures: pd.DataFrame,
    archive_path: Path,
    signal_date: str,
) -> pd.DataFrame:
    result = structures.copy()
    result["previous_proxy_date"] = ""
    for column in ("limit_up_change", "consecutive_change", "failed_rate_change", "max_board_change"):
        result[column] = math.nan
    result["industry_proxy_trend"] = "NO_PREVIOUS_ARCHIVE"
    if not archive_path.exists() or result.empty:
        return result
    history = pd.read_csv(archive_path, dtype={"industry": str}, low_memory=False)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    current_date = pd.Timestamp(signal_date)
    prior_dates = history.loc[history["date"] < current_date, "date"].dropna()
    if prior_dates.empty:
        return result
    previous_date = prior_dates.max()
    previous = history[history["date"] == previous_date].set_index("industry")
    for index, row in result.iterrows():
        industry = row["industry"]
        result.at[index, "previous_proxy_date"] = previous_date.strftime("%Y-%m-%d")
        if industry not in previous.index:
            result.at[index, "industry_proxy_trend"] = "NOT_OBSERVED_PREVIOUS_DAY"
            continue
        prior = previous.loc[industry]
        changes = {
            "limit_up_change": float(row["limit_up_count"]) - float(prior["limit_up_count"]),
            "consecutive_change": float(row["consecutive_count"]) - float(prior["consecutive_count"]),
            "failed_rate_change": float(row["failed_rate"]) - float(prior["failed_rate"]),
            "max_board_change": float(row["max_board"]) - float(prior["max_board"]),
        }
        for column, value in changes.items():
            result.at[index, column] = value
        if changes["failed_rate_change"] >= 0.20:
            trend = "PROXY_DIVERGENCE_EXPANDING"
        elif changes["limit_up_change"] >= 2 or changes["consecutive_change"] >= 1:
            trend = "PROXY_EXPANDING"
        elif changes["limit_up_change"] <= -2 or changes["consecutive_change"] <= -1:
            trend = "PROXY_CONTRACTING"
        else:
            trend = "PROXY_STABLE"
        result.at[index, "industry_proxy_trend"] = trend
    return result


def structure_flags(row: pd.Series) -> dict[str, bool]:
    ladder = bool(
        row["limit_up_count"] >= 3
        and row["consecutive_count"] >= 2
        and row["first_board_count"] >= 1
        and row["middle_board_count"] >= 1
        and row["ladder_integrity"] >= 0.50
    )
    diffusion = bool(row["limit_up_count"] >= 4 and row["consecutive_count"] <= 1)
    isolation = bool(row["max_board"] >= 3 and row["consecutive_count"] == 1)
    divergence = bool(row["touch_count"] >= 3 and row["failed_rate"] >= 0.40)
    return {
        "ladder_reinforced_proxy": ladder,
        "first_board_diffusion_proxy": diffusion,
        "height_isolation_proxy": isolation,
        "divergence_pressure_proxy": divergence,
    }


def structure_label(flags: dict[str, bool]) -> str:
    if flags["ladder_reinforced_proxy"] and flags["divergence_pressure_proxy"]:
        return "PROXY_LADDER_WITH_DIVERGENCE"
    if flags["ladder_reinforced_proxy"]:
        return "PROXY_LADDER_REINFORCEMENT"
    if flags["first_board_diffusion_proxy"] and flags["divergence_pressure_proxy"]:
        return "PROXY_DIFFUSION_WITH_DIVERGENCE"
    if flags["first_board_diffusion_proxy"]:
        return "PROXY_FIRST_BOARD_DIFFUSION"
    if flags["height_isolation_proxy"]:
        return "PROXY_HEIGHT_ISOLATION"
    if flags["divergence_pressure_proxy"]:
        return "PROXY_DIVERGENCE"
    return "PROXY_WEAK_OR_UNRESOLVED"


def build_industry_structure(
    signal_date: str,
    limit_up: pd.DataFrame,
    failed: pd.DataFrame,
    strong: pd.DataFrame,
) -> pd.DataFrame:
    for frame in (limit_up, failed, strong):
        if not frame.empty:
            frame["industry"] = frame["industry"].map(clean_industry)
    known_limit = limit_up[limit_up["industry"] != ""].copy()
    known_failed = failed[failed["industry"] != ""].copy()
    known_strong = strong[strong["industry"] != ""].copy()
    industries = sorted(set(known_limit["industry"]) | set(known_failed["industry"]) | set(known_strong["industry"]))
    total_limits = max(1, len(known_limit))
    total_consecutive = max(1, int((pd.to_numeric(known_limit["boards"], errors="coerce") >= 2).sum()))
    rows = []
    for industry in industries:
        limits = known_limit[known_limit["industry"] == industry].copy()
        failures = known_failed[known_failed["industry"] == industry].copy()
        strong_names = known_strong[known_strong["industry"] == industry].copy()
        boards = pd.to_numeric(limits["boards"], errors="coerce").fillna(1).clip(lower=1).astype(int)
        levels = sorted(set(boards.tolist()))
        max_board = int(max(levels)) if levels else 0
        expected_levels = set(range(1, max_board + 1)) if max_board else set()
        gaps = sorted(expected_levels - set(levels))
        first_clock = limits["first_seal"].map(parse_clock) if not limits.empty else pd.Series(dtype=float)
        last_clock = limits["last_seal"].map(parse_clock) if not limits.empty else pd.Series(dtype=float)
        bombs = pd.to_numeric(limits["bomb_count"], errors="coerce").fillna(0)
        one_word = (first_clock == 9 * 60 + 25) & (last_clock == 9 * 60 + 25) & (bombs == 0)
        touch_count = len(limits) + len(failures)
        row = {
            "date": signal_date,
            "industry": industry,
            "evidence_level": EVIDENCE_LEVEL,
            "theme_verified": False,
            "limit_up_count": len(limits),
            "failed_count": len(failures),
            "touch_count": touch_count,
            "seal_rate": len(limits) / touch_count if touch_count else math.nan,
            "failed_rate": len(failures) / touch_count if touch_count else math.nan,
            "strong_count": len(strong_names),
            "first_board_count": int((boards == 1).sum()),
            "consecutive_count": int((boards >= 2).sum()),
            "middle_board_count": int(boards.between(2, 4).sum()),
            "high_board_count": int((boards >= 5).sum()),
            "max_board": max_board,
            "board_levels": "/".join(map(str, levels)),
            "ladder_gap_count": len(gaps),
            "ladder_gaps": "/".join(map(str, gaps)),
            "ladder_integrity": len(levels) / max_board if max_board else math.nan,
            "limit_concentration": len(limits) / total_limits,
            "consecutive_concentration": int((boards >= 2).sum()) / total_consecutive,
            "one_word_proxy_count": int(one_word.sum()),
            "one_word_proxy_share": float(one_word.mean()) if len(one_word) else math.nan,
            "median_turnover": float(pd.to_numeric(limits["turnover"], errors="coerce").median()) if len(limits) else math.nan,
            "limit_names": "、".join(limits.sort_values("boards", ascending=False)["name"].astype(str)),
            "failed_names": "、".join(failures["name"].astype(str)),
            "source": "Eastmoney via AKShare daily market pools",
            "source_timestamp": f"{signal_date} 15:00:00",
            "data_status": "verified_industry_proxy",
            "warning": "industry co-movement is not a verified event theme or catalyst",
        }
        flags = structure_flags(pd.Series(row))
        row.update(flags)
        row["structure_label"] = structure_label(flags)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_industry_evidence(
    signal_date: str,
    candidates: pd.DataFrame,
    limit_up: pd.DataFrame,
    structures: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    pool = limit_up.copy()
    pool["industry"] = pool["industry"].map(clean_industry)
    pool["boards"] = pd.to_numeric(pool["boards"], errors="coerce").fillna(1).astype(int)
    structure_map = structures.set_index("industry").to_dict("index") if not structures.empty else {}
    code_map = pool.set_index("code").to_dict("index") if not pool.empty else {}
    rows = []
    for _, candidate in candidates.iterrows():
        code = zfill_code(candidate["code"])
        stock = code_map.get(code, {})
        industry = clean_industry(stock.get("industry", candidate.get("industry", "")))
        base = structure_map.get(industry)
        if not industry or base is None:
            rows.append({
                "date": signal_date, "code": code, "name": candidate.get("name", ""),
                "boards": candidate.get("boards"), "industry": industry,
                "industry_evidence_status": "unavailable",
                "evidence_level": EVIDENCE_LEVEL, "theme_verified": False,
                "industry_limit_support_ex_self": math.nan,
                "industry_first_support_ex_self": math.nan,
                "industry_middle_support_ex_self": math.nan,
            "structure_label": "UNAVAILABLE",
                "industry_proxy_trend": "UNAVAILABLE",
                "warning": "missing industry proxy; never interpret as no theme",
            })
            continue
        peers = pool[pool["industry"] == industry]
        others = peers[peers["code"] != code]
        rows.append({
            "date": signal_date,
            "code": code,
            "name": candidate.get("name", stock.get("name", "")),
            "boards": candidate.get("boards", stock.get("boards")),
            "industry": industry,
            "industry_evidence_status": "verified_industry_proxy",
            "evidence_level": EVIDENCE_LEVEL,
            "theme_verified": False,
            "industry_limit_support_ex_self": len(others),
            "industry_first_support_ex_self": int((others["boards"] == 1).sum()),
            "industry_middle_support_ex_self": int(others["boards"].between(2, 4).sum()),
            "industry_same_or_higher_ex_self": int((others["boards"] >= int(stock.get("boards", candidate.get("boards", 1)))).sum()),
            "industry_failed_rate": base["failed_rate"],
            "industry_ladder_integrity": base["ladder_integrity"],
            "industry_limit_concentration": base["limit_concentration"],
            "ladder_reinforced_proxy": base["ladder_reinforced_proxy"],
            "first_board_diffusion_proxy": base["first_board_diffusion_proxy"],
            "height_isolation_proxy": base["height_isolation_proxy"],
            "divergence_pressure_proxy": base["divergence_pressure_proxy"],
            "structure_label": base["structure_label"],
            "industry_proxy_trend": base.get("industry_proxy_trend", "NO_PREVIOUS_ARCHIVE"),
            "industry_limit_up_change": base.get("limit_up_change", math.nan),
            "industry_consecutive_change": base.get("consecutive_change", math.nan),
            "industry_failed_rate_change": base.get("failed_rate_change", math.nan),
            "warning": "industry proxy is contextual risk evidence, not a verified theme buy signal",
        })
    return pd.DataFrame(rows)


def make_report(signal_date: str, structures: pd.DataFrame, evidence: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        f"# {signal_date} 行业结构代理报告", "",
        f"- 数据源：{manifest['source']}",
        f"- 采集时间：{manifest['source_timestamp']}",
        f"- 证据等级：`{EVIDENCE_LEVEL}`",
        "- 行业共振只能描述结构，不能证明共同催化、真实题材或资金意图。", "",
        "## 行业结构", "",
        "| 行业 | 涨停/炸板 | 首板/连板 | 最高板 | 梯队完整度 | 相对前日 | 结构标签 |", 
        "|---|---:|---:|---:|---:|---|---|",
    ]
    display = structures[structures["limit_up_count"] > 0].sort_values(
        ["consecutive_count", "limit_up_count", "max_board"], ascending=False
    )
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.industry} | {row.limit_up_count}/{row.failed_count} | "
            f"{row.first_board_count}/{row.consecutive_count} | {row.max_board} | "
            f"{row.ladder_integrity:.2f} | {row.industry_proxy_trend} | {row.structure_label} |"
        )
    lines += ["", "## 连板候选的行业代理", "",
              "| 代码 | 名称 | 板位 | 行业 | 同行业涨停支持(去自身) | 首板支持 | 中位支持 | 相对前日 | 结构 |",
              "|---|---|---:|---|---:|---:|---:|---|---|"]
    for row in evidence.sort_values(["boards", "code"], ascending=[False, True]).itertuples(index=False):
        if row.industry_evidence_status == "unavailable":
            lines.append(f"| {row.code} | {row.name} | {row.boards} | 未知 | — | — | — | UNAVAILABLE | UNAVAILABLE |")
        else:
            lines.append(
                f"| {row.code} | {row.name} | {row.boards} | {row.industry} | "
                f"{row.industry_limit_support_ex_self} | {row.industry_first_support_ex_self} | "
                f"{row.industry_middle_support_ex_self} | {row.industry_proxy_trend} | {row.structure_label} |"
            )
    lines += ["", "该报告不回写已冻结模型，也不把行业代理标记为真实题材。", ""]
    return "\n".join(lines)


def run(signal_date: str, raw_root: Path = RAW, output: Path = OUTPUT, archive: Path = ARCHIVE) -> dict[str, Any]:
    normalized_date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    day_dir = raw_root / pd.Timestamp(signal_date).strftime("%Y%m%d")
    manifest_path = day_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"raw pool manifest missing: {manifest_path}")
    pool_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limit_up = read_pool(day_dir, "limit_up", normalized_date)
    failed = read_pool(day_dir, "failed_board", normalized_date)
    strong = read_pool(day_dir, "strong", normalized_date)
    candidates = pd.read_csv(CANDIDATES, dtype={"code": str, "industry": str}, low_memory=False)
    candidates = candidates[candidates["date"].astype(str) == normalized_date].copy()
    structures = build_industry_structure(normalized_date, limit_up, failed, strong)
    archive_structure_path = archive / "industry_structure_daily.csv"
    structures = attach_temporal_context(structures, archive_structure_path, normalized_date)
    evidence = candidate_industry_evidence(normalized_date, candidates, limit_up, structures)
    output.mkdir(parents=True, exist_ok=True)
    structures_path = output / f"{normalized_date}_industry_structure.csv"
    evidence_path = output / f"{normalized_date}_candidate_industry_evidence.csv"
    structures.to_csv(structures_path, index=False, encoding="utf-8-sig")
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    append_dedup(structures, archive_structure_path, ["date", "industry"])
    append_dedup(evidence, archive / "candidate_industry_evidence.csv", ["date", "code"])
    manifest = {
        "date": normalized_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Eastmoney via AKShare daily market pools",
        "source_timestamp": f"{normalized_date} 15:00:00",
        "pool_manifest_status": pool_manifest.get("datasets", {}),
        "evidence_level": EVIDENCE_LEVEL,
        "theme_verified": False,
        "industry_rows": len(structures),
        "candidate_rows": len(evidence),
        "unknown_candidate_industry_rows": int((evidence.get("industry_evidence_status") == "unavailable").sum()) if len(evidence) else 0,
        "files": {
            "industry_structure": {"path": str(structures_path), "sha256": sha256_file(structures_path)},
            "candidate_evidence": {"path": str(evidence_path), "sha256": sha256_file(evidence_path)},
        },
        "warning": "industry proxy must never be imported as verified themes",
    }
    report_path = output / f"{normalized_date}_industry_structure_report.md"
    report_path.write_text(make_report(normalized_date, structures, evidence, manifest), encoding="utf-8")
    manifest["files"]["report"] = {"path": str(report_path), "sha256": sha256_file(report_path)}
    (output / f"{normalized_date}_industry_structure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time industry structure proxies without claiming verified themes")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    try:
        result = run(args.date)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
