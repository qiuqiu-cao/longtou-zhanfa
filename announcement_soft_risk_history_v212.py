"""Rebuild candidate-level announcement soft-risk evidence without assuming silence.

A zero score is valid only when the candidate belongs to the exact archived
closing-limit pool, the announcement query covered that pool, and the code was
not in the collector error set.  Missing or failed queries remain unavailable.
Hard announcement risks are reported separately and are never diluted into the
soft score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v212-announcement-soft-risk-history-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> int:
    values = re.findall(r"_v(\d+)", path.as_posix())
    return int(values[-1]) if values else 0


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def discover_pools(root: Path) -> dict[str, pd.DataFrame]:
    chosen = {}
    for path in root.rglob("all_closing_and_failed_candidates.csv"):
        try:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except Exception:
            continue
        if frame.empty or "signal_date" not in frame:
            continue
        dates = frame.signal_date.astype(str).str[:10].unique()
        if len(dates) != 1:
            continue
        date = dates[0]
        if date not in chosen or _revision(path) > chosen[date][0]:
            chosen[date] = (_revision(path), frame)
    return {date: item[1] for date, item in chosen.items()}


def discover_context(root: Path) -> dict[str, tuple[dict, pd.DataFrame, Path]]:
    chosen = {}
    for manifest_path in root.rglob("*_manifest.json"):
        manifest = _load(manifest_path)
        meta = manifest.get("announcements")
        if not isinstance(meta, dict) or not manifest.get("signal_date"):
            continue
        file_info = manifest.get("files", {}).get("announcements", {})
        raw_path = file_info.get("path")
        candidates = []
        if raw_path:
            raw = Path(raw_path)
            candidates += [raw if raw.is_absolute() else root / raw]
        candidates += [manifest_path.parent / f"{manifest['signal_date']}_announcements.csv"]
        csv_path = next((path for path in candidates if path.exists()), None)
        if csv_path is None:
            continue
        try:
            frame = pd.read_csv(csv_path, dtype={"code": str})
        except Exception:
            continue
        success = int(meta.get("candidate_codes", 0)) - int(meta.get("failed_codes", 0))
        quality = (success, int(meta.get("rows", 0)), _revision(manifest_path))
        date = str(manifest["signal_date"])[:10]
        if date not in chosen or quality > chosen[date][0]:
            chosen[date] = (quality, manifest, frame, manifest_path)
    return {date: item[1:] for date, item in chosen.items()}


def build(root: Path, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[Path]]:
    if not {"signal_date", "code", "name"} <= set(candidates):
        raise ValueError("candidate input requires signal_date, code and name")
    if candidates.duplicated(["signal_date", "code"]).any():
        raise ValueError("candidate keys must be unique")
    pools, contexts = discover_pools(root), discover_context(root)
    output, sources = [], []
    for date, group in candidates.groupby(candidates.signal_date.astype(str).str[:10]):
        pool = pools.get(date)
        context = contexts.get(date)
        closing_codes = set()
        if pool is not None:
            closing = pool[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
            closing_codes = set(closing.code.astype(str).str.zfill(6))
        if context:
            manifest, notices, manifest_path = context
            sources.append(manifest_path)
            file_path = manifest_path.parent / f"{date}_announcements.csv"
            if file_path.exists():
                sources.append(file_path)
            errors = set(str(code).zfill(6) for code in manifest.get("announcements", {}).get("errors", {}))
            exact_scope = (
                manifest.get("candidate_count") == len(closing_codes)
                and manifest.get("announcements", {}).get("candidate_codes") == len(closing_codes)
            )
            notices = notices.copy()
            if not notices.empty:
                notices["code"] = notices.code.astype(str).str.zfill(6)
        else:
            manifest, notices, errors, exact_scope = {}, pd.DataFrame(), set(), False
        for row in group.to_dict("records"):
            code = str(row["code"]).zfill(6)
            verified = bool(exact_scope and code in closing_codes and code not in errors)
            code_notices = notices[notices.code.eq(code)].copy() if verified and not notices.empty else pd.DataFrame()
            hard = int(pd.to_numeric(code_notices.get("risk_hard", pd.Series(dtype=float)),
                                     errors="coerce").fillna(0).gt(0).sum())
            soft_types = code_notices.get("risk_type", pd.Series(dtype=str)).fillna("").astype(str)
            ordinary = int(soft_types.str.contains("普通异常波动").sum())
            closed = int(soft_types.str.contains("closed|条件成就", case=False).sum())
            # Notice count is a small crowding term. Hard risks remain a
            # separate veto and are not hidden inside this score.
            soft_score = min(1.0, ordinary * .30 + closed * .10
                             + max(0, len(code_notices) - ordinary - closed - hard) * .01) if verified else np.nan
            output.append({
                "signal_date": date, "code": code, "name": row.get("name"),
                "announcement_soft_risk": soft_score,
                "announcement_soft_risk_evidence_status": (
                    "POINT_IN_TIME_VERIFIED" if verified else "UNAVAILABLE"),
                "hard_announcement_risk_count": hard if verified else np.nan,
                "ordinary_abnormal_notice_count": ordinary if verified else np.nan,
                "announcement_query_failed": code in errors,
                "announcement_scope_exact": exact_scope,
                "announcement_available_at": manifest.get("collected_at"),
                "missing_announcement_is_zero": False,
            })
    rows = pd.DataFrame(output)
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "candidate_rows": int(len(rows)), "sessions": int(rows.signal_date.nunique()) if len(rows) else 0,
        "verified_soft_risk_rows": int(rows.announcement_soft_risk.notna().sum()) if len(rows) else 0,
        "hard_risk_rows": int(pd.to_numeric(rows.hard_announcement_risk_count,
                                            errors="coerce").fillna(0).gt(0).sum()) if len(rows) else 0,
        "missing_announcement_is_zero": False,
    }
    return rows, report, sources


def run(root_path, candidates_path, output_dir):
    root, source, target = Path(root_path), Path(candidates_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable announcement history")
    rows, report, evidence = build(root, pd.read_csv(source, dtype={"code": str}))
    target.mkdir(parents=True)
    output = target / "announcement_soft_risk_evidence.csv"
    rows.to_csv(output, index=False, encoding="utf-8-sig")
    paths = {source, Path(__file__), *evidence}
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths if path.exists()}
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.candidates, args.output), ensure_ascii=False, indent=2))
