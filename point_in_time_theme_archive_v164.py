"""Normalize every archived closing-limit candidate into a point-in-time theme archive.

Theme membership, catalyst timing and expectation saturation are different
claims.  A review article that groups stocks after the close may verify a theme
roster for next-session planning, but it does not automatically prove the
underlying catalyst's first-public time or whether the market has already priced
it.  Unknown candidates remain explicit rows and never become zero-strength
themes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


VERSION = "v164-point-in-time-theme-archive-1"
FACTOR_FIELDS = (
    "theme_granularity", "driver_type", "driver_summary",
    "driver_first_available_at", "driver_claim_status", "expectation_status",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _stamp(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    else:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp


def _date_from_frame(frame: pd.DataFrame) -> str | None:
    for column in ("signal_date", "date"):
        if column in frame and not frame.empty:
            values = frame[column].dropna().astype(str).str[:10].unique().tolist()
            if len(values) == 1:
                return values[0]
    return None


def _revision(path: Path) -> int:
    matches = re.findall(r"_v(\d+)", path.as_posix())
    return int(matches[-1]) if matches else 0


def _pool_manifest(pool_path: Path) -> tuple[Path | None, dict]:
    path = pool_path.parent / "manifest.json"
    return (path, _read_json(path)) if path.exists() else (None, {})


def discover_pools(root: Path) -> dict[str, dict]:
    by_date = {}
    paths = root.rglob("all_closing_and_failed_candidates.csv")
    for path in paths:
        try:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        except Exception:
            continue
        date = _date_from_frame(frame)
        if not date:
            continue
        manifest_path, manifest = _pool_manifest(path)
        item = {
            "path": path, "frame": frame, "manifest_path": manifest_path,
            "manifest": manifest, "revision": _revision(path),
        }
        if date not in by_date or item["revision"] > by_date[date]["revision"]:
            by_date[date] = item
    return by_date


def discover_roles(root: Path) -> dict[str, list[dict]]:
    by_date = {}
    for path in root.rglob("candidate_theme_roles.csv"):
        try:
            frame = pd.read_csv(path, dtype={"code": str})
        except Exception:
            continue
        date = _date_from_frame(frame)
        if not date:
            continue
        manifest_candidates = [path.parent / "manifest.json", path.parent / "theme_evidence_manifest.json"]
        manifest_path = next((item for item in manifest_candidates if item.exists()), None)
        manifest = _read_json(manifest_path) if manifest_path else {}
        verified = int(frame.get("theme_status", pd.Series(dtype=str)).astype(str).str.startswith("verified").sum())
        by_date.setdefault(date, []).append({
            "path": path, "frame": frame, "manifest_path": manifest_path,
            "manifest": manifest, "verified": verified, "revision": _revision(path),
        })
    return by_date


def discover_registries(root: Path) -> pd.DataFrame:
    rows = []
    search_roots = [root / "data", root / "results"]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("theme_evidence_registry.csv"):
            try:
                frame = pd.read_csv(path, dtype={"code": str})
            except Exception:
                continue
            if {"signal_date", "code"} <= set(frame):
                frame = frame.copy()
                frame["registry_source_path"] = str(path.resolve())
                rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["signal_date", "code"])
    result = pd.concat(rows, ignore_index=True, sort=False)
    result["code"] = result.code.astype(str).str.zfill(6)
    result["signal_date"] = result.signal_date.astype(str).str[:10]
    return result.drop_duplicates(["signal_date", "code", "evidence_id"], keep="last")


def _calendar_map(calendar: list[str]) -> dict[str, str]:
    sessions = sorted(set(str(item)[:10] for item in calendar))
    return {sessions[index]: sessions[index + 1] for index in range(len(sessions) - 1)}


def _role_lookup(role: pd.DataFrame) -> dict[str, dict]:
    if role.empty or "code" not in role:
        return {}
    work = role.copy()
    work["code"] = work.code.astype(str).str.zfill(6)
    # A primary role table should have one row per candidate.  If legacy data
    # contains duplicates, keep the row with the richest explicit evidence.
    work["_quality"] = sum(
        work.get(column, pd.Series(index=work.index, dtype=object)).notna().astype(int)
        for column in ("theme_group_id", "theme", "primary_evidence_id", "evidence_published_at")
    )
    work = work.sort_values(["code", "_quality"]).drop_duplicates("code", keep="last")
    return {row["code"]: row for row in work.drop(columns="_quality").to_dict("records")}


def _registry_lookup(registry: pd.DataFrame, date: str) -> dict[str, dict]:
    if registry.empty:
        return {}
    work = registry.loc[registry.signal_date.eq(date)].copy()
    if work.empty:
        return {}
    rank = work.get("usable_at_signal", pd.Series(False, index=work.index)).map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"})
    work["_usable"] = rank.astype(int)
    work = work.sort_values(["code", "_usable"]).drop_duplicates("code", keep="last")
    return {row["code"]: row for row in work.drop(columns="_usable").to_dict("records")}


def _closing_pool(item: dict | None, role_frame: pd.DataFrame) -> tuple[pd.DataFrame, str, dict]:
    if item:
        frame = item["frame"].copy()
        if "pool_assignment" in frame:
            frame = frame.loc[frame.pool_assignment.eq("CLOSING_LIMIT_UP")].copy()
        return frame, "archived_closing_and_failed_pool", item.get("manifest", {})
    # Legacy theme files may contain only a relay roster.  Preserve it, but do
    # not mislabel it as a complete closing-limit universe.
    frame = role_frame.copy()
    if "signal_date" not in frame and "date" in frame:
        frame["signal_date"] = frame.date
    frame["pool_assignment"] = "DECLARED_ROLE_ROSTER"
    return frame, "declared_role_roster_only", {}


def assemble(pools: dict[str, dict], roles: dict[str, list[dict]],
             registry: pd.DataFrame, calendar: list[str]):
    dates = sorted(set(pools) | set(roles))
    next_sessions = _calendar_map(calendar)
    archive_rows, daily_rows, selected_sources = [], [], []
    for date in dates:
        revisions = roles.get(date, [])
        selected_role = max(revisions, key=lambda item: (item["verified"], item["revision"])) if revisions else None
        role_frame = selected_role["frame"].copy() if selected_role else pd.DataFrame()
        pool, pool_scope, pool_manifest = _closing_pool(pools.get(date), role_frame)
        if pool.empty or "code" not in pool:
            continue
        pool["code"] = pool.code.astype(str).str.zfill(6)
        pool = pool.drop_duplicates("code", keep="last")
        role_by_code = _role_lookup(role_frame)
        registry_by_code = _registry_lookup(registry, date)
        next_session = next_sessions.get(date)
        cutoff = _stamp(f"{next_session}T09:15:00+08:00") if next_session else pd.NaT
        full_market = pool_manifest.get("full_market_completeness") == "verified_full_market"
        complete_roster = pool_manifest.get("complete_within_archived_roster") is True
        verified_count = pit_count = catalyst_count = expectation_count = eligible_count = 0
        for candidate in pool.to_dict("records"):
            code = str(candidate["code"]).zfill(6)
            role = role_by_code.get(code, {})
            evidence = registry_by_code.get(code, {})
            theme_status = str(role.get("theme_status") or "unavailable")
            theme_verified = theme_status.startswith("verified")
            evidence_at = _stamp(
                role.get("evidence_published_at") or evidence.get("effective_published_at")
                or evidence.get("source_published_at"))
            point_in_time = bool(theme_verified and pd.notna(evidence_at) and pd.notna(cutoff)
                                 and evidence_at <= cutoff)
            theme_granularity = evidence.get("theme_granularity")
            driver_summary = evidence.get("driver")
            driver_first = evidence.get("effective_published_at") or evidence.get("source_published_at")
            # A membership record may contain a driver description, but it does
            # not by itself verify driver type, first-public timing or expectation saturation.
            driver_claim_status = (
                "membership_source_mentions_driver_not_first_publication_verified"
                if driver_summary else "unavailable"
            )
            factors = {
                "theme_granularity": theme_granularity,
                "driver_type": None,
                "driver_summary": driver_summary,
                "driver_first_available_at": driver_first,
                "driver_claim_status": driver_claim_status,
                "expectation_status": None,
            }
            factor_complete = all(value is not None and str(value).strip() for value in factors.values())
            model_eligible = bool(full_market and complete_roster and point_in_time and factor_complete)
            verified_count += int(theme_verified)
            pit_count += int(point_in_time)
            catalyst_count += int(bool(driver_summary and driver_first))
            expectation_count += int(bool(factors["expectation_status"]))
            eligible_count += int(model_eligible)
            archive_rows.append({
                "signal_date": date, "next_session": next_session, "code": code,
                "name": candidate.get("name"), "boards": candidate.get("boards"),
                "board_group": candidate.get("board_group"),
                "pool_assignment": candidate.get("pool_assignment", "CLOSING_LIMIT_UP"),
                "pool_scope": pool_scope,
                "full_market_completeness": pool_manifest.get(
                    "full_market_completeness", "not_proven_declared_roster_only"),
                "complete_within_archived_roster": complete_roster,
                "theme_status": theme_status,
                "theme_group_id": role.get("theme_group_id"),
                "theme": role.get("theme"), "theme_role": role.get("theme_role"),
                "theme_support_ex_self": role.get("theme_support_ex_self"),
                "same_height_rivals": role.get("same_height_rivals"),
                "theme_ladder_verified": role.get("theme_ladder_verified"),
                "primary_evidence_id": role.get("primary_evidence_id"),
                "source_tier": role.get("source_tier"),
                "theme_evidence_status": role.get("evidence_status"),
                "theme_evidence_available_at": evidence_at.isoformat() if pd.notna(evidence_at) else None,
                "next_session_preopen_cutoff": cutoff.isoformat() if pd.notna(cutoff) else None,
                "point_in_time_available_for_next_session": point_in_time,
                **factors,
                "factor_schema_complete": factor_complete,
                "theme_factor_model_eligible": model_eligible,
                "missing_is_zero": False,
            })
        closing_count = int(len(pool))
        daily_rows.append({
            "signal_date": date, "next_session": next_session,
            "closing_candidate_count": closing_count,
            "verified_theme_candidate_count": verified_count,
            "theme_coverage": verified_count / closing_count,
            "point_in_time_theme_count": pit_count,
            "catalyst_partial_count": catalyst_count,
            "expectation_status_count": expectation_count,
            "model_eligible_count": eligible_count,
            "pool_scope": pool_scope,
            "full_market_complete": full_market,
            "complete_within_archived_roster": complete_roster,
            "role_revision_count": len(revisions),
            "selected_role_path": str(selected_role["path"].resolve()) if selected_role else None,
            "issues": ";".join(filter(None, [
                None if full_market else "FULL_MARKET_COMPLETENESS_NOT_PROVEN",
                None if verified_count == closing_count else "THEME_COVERAGE_INCOMPLETE",
                None if catalyst_count == closing_count else "CATALYST_FIRST_TIME_OR_TYPE_INCOMPLETE",
                None if expectation_count == closing_count else "EXPECTATION_STATUS_INCOMPLETE",
            ])),
        })
        if selected_role:
            selected_sources.append(selected_role)
        if pools.get(date):
            selected_sources.append(pools[date])
    archive = pd.DataFrame(archive_rows)
    daily = pd.DataFrame(daily_rows)
    summary = {
        "version": VERSION,
        "status": "READY_FOR_TIME_ISOLATED_THEME_FACTOR_MODEL"
        if len(daily) >= 250 and daily.model_eligible_count.eq(daily.closing_candidate_count).all()
        else "DIAGNOSTIC_ONLY",
        "distinct_sessions": int(len(daily)),
        "candidate_rows": int(len(archive)),
        "verified_theme_rows": int(archive.theme_status.astype(str).str.startswith("verified").sum())
        if len(archive) else 0,
        "point_in_time_theme_rows": int(archive.point_in_time_available_for_next_session.sum())
        if len(archive) else 0,
        "model_eligible_rows": int(archive.theme_factor_model_eligible.sum()) if len(archive) else 0,
        "full_market_complete_sessions": int(daily.full_market_complete.sum()) if len(daily) else 0,
        "minimum_model_sessions": 250,
        "remaining_model_sessions": max(0, 250 - int(
            daily.loc[daily.model_eligible_count.eq(daily.closing_candidate_count)].shape[0]
        )),
        "missing_values_treated_as_zero": False,
        "policy": {
            "theme_membership_is_not_catalyst_timing": True,
            "review_publication_is_not_driver_first_publication": True,
            "unknown_theme_is_not_weak_theme": True,
            "expectation_status_required_for_model": True,
        },
    }
    return archive, daily, summary, selected_sources


def run(root_path, calendar_path, output_dir):
    root, calendar_source, target = Path(root_path), Path(calendar_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable theme archive")
    calendar = pd.read_csv(calendar_source).date.astype(str).tolist()
    pools, roles, registry = discover_pools(root), discover_roles(root), discover_registries(root)
    archive, daily, summary, sources = assemble(pools, roles, registry, calendar)
    target.mkdir(parents=True)
    archive_path, daily_path = target / "candidate_theme_archive.csv", target / "daily_coverage.csv"
    archive.to_csv(archive_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    paths = {calendar_source, Path(__file__)}
    for item in sources:
        paths.add(item["path"])
        if item.get("manifest_path"):
            paths.add(item["manifest_path"])
    summary["source_hashes"] = {str(path.resolve()): _sha256(path) for path in paths if path.exists()}
    summary["outputs"] = {
        "archive": {"path": str(archive_path.resolve()), "sha256": _sha256(archive_path)},
        "daily": {"path": str(daily_path.resolve()), "sha256": _sha256(daily_path)},
    }
    (target / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--calendar", default="data/v3/market_emotion.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.calendar, args.output), ensure_ascii=False, indent=2))
