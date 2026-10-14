"""Preserve every first-board theme lane and expose hidden symbolic identities.

The legacy V218 view collapses a stock to one ``best_theme``.  That is useful
for a primary narrative but can hide a simultaneous symbolic-name ladder such
as 华字辈.  V229 keeps verified/observed themes, industry proxies and symbolic
clusters separate, produces a symbolic pure pioneer and a dual-identity
challenger, and maps them to the symbolic height anchor.  It never treats a
name cluster as a business theme, calibrated probability or buy signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v229-first-board-multilane-identity-1"
BUSINESS_LANES = {
    "VERIFIED_REAL_THEME", "VERIFIED_MEMBERSHIP_ONLY", "OBSERVED_THEME_ROSTER",
}
SUPPORTED_STAGES = {"FULL_LADDER", "RELAY_WITH_SUPPORT"}
SYMBOLIC_LADDER = "SYMBOLIC_LADDER_CANDIDATE"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _truth(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _number(value, default=np.nan):
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else float(parsed)


def _cluster_meta(symbolic: dict) -> dict[str, dict]:
    result = {}
    for cluster in symbolic.get("clusters", []):
        members = cluster.get("members", [])
        max_board = int(cluster.get("max_board", 0))
        anchors = [row for row in members if int(row.get("boards", 0)) == max_board]
        relays = sorted(
            [row for row in members if int(row.get("boards", 0)) >= 2],
            key=lambda row: (-int(row.get("boards", 0)), str(row.get("code", ""))),
        )
        previous = cluster.get("previous_session_comparison") or {}
        result[str(cluster["cluster_id"])] = {
            "token": cluster.get("token"),
            "classification": cluster.get("classification"),
            "member_count": int(cluster.get("member_count", 0)),
            "member_count_change": previous.get("member_count_change"),
            "max_board": max_board,
            "max_board_change": previous.get("max_board_change"),
            "failed_board_member_count": int(cluster.get("failed_board_member_count", 0)),
            "ladder_contiguous": bool(cluster.get("ladder_contiguous", False)),
            "anchor_codes": [str(row.get("code", "")).zfill(6) for row in anchors],
            "anchor_names": [str(row.get("name", "")) for row in anchors],
            "relay_members": [{
                "code": str(row.get("code", "")).zfill(6),
                "name": str(row.get("name", "")), "boards": int(row.get("boards", 0)),
            } for row in relays],
        }
    return result


def _validate(watch: pd.DataFrame, theme_map: pd.DataFrame, symbolic: dict) -> str:
    watch_required = {
        "signal_date", "code", "name", "close_quality_gate", "one_price_proxy",
        "thin_control_risk", "first_seal_minute", "community_evidence_count",
        "community_probability_not_calibrated", "theme_research_gate",
    }
    theme_required = {
        "signal_date", "code", "theme_id", "theme_name", "evidence_lane",
        "theme_stage", "rank_within_lane", "theme_pioneer_rank",
    }
    if watch_required - set(watch):
        raise ValueError(f"watchlist fields missing: {sorted(watch_required - set(watch))}")
    if theme_required - set(theme_map):
        raise ValueError(f"theme-map fields missing: {sorted(theme_required - set(theme_map))}")
    dates = set(watch.signal_date.astype(str)) | set(theme_map.signal_date.astype(str))
    dates.add(str(symbolic.get("signal_date", "")))
    if len(dates) != 1:
        raise ValueError(f"signal dates differ across V229 inputs: {sorted(dates)}")
    if watch.duplicated(["signal_date", "code"]).any():
        raise ValueError("watchlist keys must be unique")
    if theme_map.duplicated(["signal_date", "code", "theme_id"]).any():
        raise ValueError("theme-map keys must be unique")
    if symbolic.get("version") != "v217-symbolic-name-cluster-1":
        raise ValueError("symbolic report version mismatch")
    return dates.pop()


def build(watch: pd.DataFrame, theme_map: pd.DataFrame,
          symbolic: dict, driver_claims: pd.DataFrame | None = None,
          official_driver_times: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    signal_date = _validate(watch, theme_map, symbolic)
    watch = watch.copy()
    theme_map = theme_map.copy()
    watch["code"] = watch.code.astype(str).str.zfill(6)
    theme_map["code"] = theme_map.code.astype(str).str.zfill(6)
    for column in ["close_quality_gate", "one_price_proxy", "thin_control_risk",
                   "theme_research_gate"]:
        watch[column] = watch[column].map(_truth)
    for column in ["first_seal_minute", "community_evidence_count",
                   "community_probability_not_calibrated"]:
        watch[column] = pd.to_numeric(watch[column], errors="coerce")
    theme_map["rank_within_lane"] = pd.to_numeric(theme_map.rank_within_lane, errors="coerce")
    theme_map["theme_pioneer_rank"] = pd.to_numeric(theme_map.theme_pioneer_rank, errors="coerce")
    claim_by_code = {}
    if driver_claims is not None:
        required_claims = {
            "signal_date", "code", "driver_claim", "driver_claim_status",
            "driver_claim_category", "negative_or_limited_exposure",
            "recency_language_status", "catalyst_first_public_time_status",
        }
        if required_claims - set(driver_claims):
            raise ValueError(f"driver-claim fields missing: {sorted(required_claims-set(driver_claims))}")
        claims = driver_claims.copy()
        claims["code"] = claims.code.astype(str).str.zfill(6)
        if set(claims.signal_date.astype(str)) != {signal_date}:
            raise ValueError("driver-claim signal date mismatch")
        if claims.code.duplicated().any():
            raise ValueError("driver-claim codes must be unique")
        claim_by_code = {row["code"]: row for row in claims.to_dict("records")}
    official_by_code = {}
    if official_driver_times is not None:
        required_official = {
            "signal_date", "code", "official_driver_evidence_status",
            "first_public_time_status", "official_driver_age_lower_bound_days",
            "signal_date_novelty_status", "driver_freshness_readthrough",
            "lifecycle_status", "driver_is_new_on_signal_date",
        }
        if required_official - set(official_driver_times):
            raise ValueError(
                f"official-driver-time fields missing: {sorted(required_official-set(official_driver_times))}")
        official = official_driver_times.copy()
        official["code"] = official.code.astype(str).str.zfill(6)
        if set(official.signal_date.astype(str)) != {signal_date}:
            raise ValueError("official-driver-time signal date mismatch")
        if official.code.duplicated().any():
            raise ValueError("official-driver-time codes must be unique")
        official_by_code = {row["code"]: row for row in official.to_dict("records")}
    joined = theme_map.merge(
        watch[["signal_date", "code", "close_quality_gate", "one_price_proxy",
               "thin_control_risk", "first_seal_minute"]],
        on=["signal_date", "code"], how="inner", validate="many_to_one")
    joined["buyable_close_quality"] = (
        joined.close_quality_gate & ~joined.one_price_proxy & ~joined.thin_control_risk)
    symbolic_rows = joined[joined.evidence_lane.eq("SYMBOLIC_NAME_CLUSTER")].copy()
    symbolic_rows["symbolic_buyable_pioneer_rank"] = np.nan
    if len(symbolic_rows):
        eligible = symbolic_rows[symbolic_rows.buyable_close_quality].copy()
        ranks = eligible.groupby(["signal_date", "theme_id"]).first_seal_minute.rank(
            method="min", ascending=True)
        symbolic_rows.loc[eligible.index, "symbolic_buyable_pioneer_rank"] = ranks

    meta = _cluster_meta(symbolic)
    symbolic_by_code = {
        code: part.sort_values(["symbolic_buyable_pioneer_rank", "theme_pioneer_rank"],
                               na_position="last")
        for code, part in symbolic_rows.groupby("code")
    }
    business = joined[joined.evidence_lane.isin(BUSINESS_LANES)].copy()
    business["observed_structure_support"] = (
        business.theme_stage.isin(SUPPORTED_STAGES)
        & business.rank_within_lane.le(3))
    business_order = {
        "VERIFIED_REAL_THEME": 0, "VERIFIED_MEMBERSHIP_ONLY": 1,
        "OBSERVED_THEME_ROSTER": 2,
    }
    business["_lane_order"] = business.evidence_lane.map(business_order)
    business_by_code = {
        code: part.sort_values(["_lane_order", "rank_within_lane", "theme_pioneer_rank"])
        for code, part in business.groupby("code")
    }

    rows = []
    for source in watch.to_dict("records"):
        code = str(source["code"]).zfill(6)
        business_part = business_by_code.get(code, pd.DataFrame())
        symbolic_part = symbolic_by_code.get(code, pd.DataFrame())
        supported = (business_part[business_part.observed_structure_support]
                     if len(business_part) else pd.DataFrame())
        best_business = business_part.iloc[0] if len(business_part) else None
        best_symbolic = symbolic_part.iloc[0] if len(symbolic_part) else None
        best_meta = meta.get(str(best_symbolic.theme_id), {}) if best_symbolic is not None else {}
        claim = claim_by_code.get(code, {})
        official_driver = official_by_code.get(code, {})
        close_buyable = bool(source["close_quality_gate"] and not source["one_price_proxy"]
                             and not source["thin_control_risk"])
        ladder_identity = bool(best_meta.get("classification") == SYMBOLIC_LADDER)
        dual = bool(close_buyable and ladder_identity and len(supported))
        pure_pioneer = bool(
            close_buyable and ladder_identity and best_symbolic is not None
            and _number(best_symbolic.symbolic_buyable_pioneer_rank) == 1
        )
        if dual:
            role = "SYMBOLIC_DUAL_IDENTITY_CHALLENGER"
            bucket = 0
        elif pure_pioneer:
            role = "SYMBOLIC_PURE_BUYABLE_PIONEER"
            bucket = 1
        elif close_buyable and ladder_identity:
            role = "SYMBOLIC_BUYABLE_FOLLOWER"
            bucket = 2
        elif best_symbolic is not None:
            role = "SYMBOLIC_OBSERVATION_ONLY"
            bucket = 3
        else:
            role = "NO_SYMBOLIC_IDENTITY"
            bucket = 4
        contradictions = []
        if best_symbolic is not None and not close_buyable:
            contradictions.append("CLOSE_QUALITY_OR_BUYABILITY_FAILED")
        if best_symbolic is not None and not len(supported):
            contradictions.append("NO_TOP3_SUPPORTED_BUSINESS_NARRATIVE")
        if best_meta and best_meta.get("member_count_change") is None:
            contradictions.append("NO_COMPARABLE_PRIOR_SYMBOLIC_SESSION")
        if best_symbolic is not None:
            contradictions.append("SYMBOLIC_CLUSTER_NOT_REAL_THEME")
        if claim:
            if official_driver.get("official_driver_evidence_status") != "OFFICIAL_CORE_CLAIM_MATCHED":
                contradictions.append("SECONDARY_DRIVER_CLAIM_NOT_VERIFIED_FACT")
                contradictions.append("CATALYST_FIRST_PUBLIC_TIME_UNAVAILABLE")
            else:
                if not _truth(official_driver.get("driver_is_new_on_signal_date", False)):
                    contradictions.append("OFFICIAL_DRIVER_PREEXISTED_SIGNAL_DATE")
                if official_driver.get("driver_freshness_readthrough") == (
                        "LEGACY_OR_RECYCLED_DRIVER_NOT_NEW"):
                    contradictions.append("LEGACY_OR_RECYCLED_DRIVER")
                if official_driver.get("driver_freshness_readthrough") == (
                        "ONGOING_ACTION_BUT_NOT_NEW_ON_SIGNAL_DATE"):
                    contradictions.append("ONGOING_ACTION_NOT_NEW_CATALYST")
            if _truth(claim.get("negative_or_limited_exposure")):
                contradictions.append("NEGATIVE_OR_LIMITED_EXPOSURE_CLAIM")
            if claim.get("recency_language_status") == "STATIC_OR_UNDATED_DESCRIPTION":
                contradictions.append("STATIC_OR_UNDATED_DRIVER_DESCRIPTION")
        rows.append({
            "signal_date": signal_date, "code": code, "name": source.get("name"),
            "close_quality_gate": bool(source["close_quality_gate"]),
            "first_seal_minute": _number(source.get("first_seal_minute")),
            "community_evidence_count": _number(source.get("community_evidence_count")),
            "community_probability_NOT_CALIBRATED": _number(
                source.get("community_probability_not_calibrated")),
            "legacy_theme_research_gate": bool(source.get("theme_research_gate")),
            "primary_business_lane": (str(best_business.evidence_lane)
                                      if best_business is not None else "UNAVAILABLE"),
            "primary_business_theme": (str(best_business.theme_name)
                                       if best_business is not None else "UNAVAILABLE"),
            "primary_business_stage": (str(best_business.theme_stage)
                                       if best_business is not None else "UNAVAILABLE"),
            "primary_business_rank_within_lane": (
                _number(best_business.rank_within_lane) if best_business is not None else np.nan),
            "observed_or_verified_structure_support": bool(len(supported)),
            "verified_real_theme_support": bool(
                len(supported) and supported.evidence_lane.eq("VERIFIED_REAL_THEME").any()),
            "all_theme_lanes": "|".join(
                f"{row.evidence_lane}:{row.theme_name}" for row in
                joined[joined.code.eq(code)].sort_values(
                    ["evidence_lane", "rank_within_lane"]).itertuples()),
            "symbolic_cluster_id": (str(best_symbolic.theme_id)
                                    if best_symbolic is not None else ""),
            "symbolic_theme_name": (str(best_symbolic.theme_name)
                                    if best_symbolic is not None else ""),
            "symbolic_classification": str(best_meta.get("classification", "")),
            "symbolic_raw_pioneer_rank": (
                _number(best_symbolic.theme_pioneer_rank) if best_symbolic is not None else np.nan),
            "symbolic_buyable_pioneer_rank": (
                _number(best_symbolic.symbolic_buyable_pioneer_rank)
                if best_symbolic is not None else np.nan),
            "symbolic_member_count": best_meta.get("member_count"),
            "symbolic_member_count_change": best_meta.get("member_count_change"),
            "symbolic_max_board": best_meta.get("max_board"),
            "symbolic_anchor_codes": "|".join(best_meta.get("anchor_codes", [])),
            "symbolic_anchor_names": "|".join(best_meta.get("anchor_names", [])),
            "symbolic_relay_ladder": "|".join(
                f"{row['name']}({row['boards']}板)"
                for row in best_meta.get("relay_members", [])),
            "secondary_driver_claim": claim.get("driver_claim", ""),
            "secondary_driver_claim_status": claim.get("driver_claim_status", "UNAVAILABLE"),
            "secondary_driver_claim_category": claim.get("driver_claim_category", "UNAVAILABLE"),
            "secondary_driver_negative_or_limited": _truth(
                claim.get("negative_or_limited_exposure", False)),
            "secondary_driver_recency_status": claim.get(
                "recency_language_status", "UNAVAILABLE"),
            "catalyst_first_public_time_status": claim.get(
                "catalyst_first_public_time_status", "UNAVAILABLE"),
            "official_driver_evidence_status": official_driver.get(
                "official_driver_evidence_status", "NO_OFFICIAL_DRIVER_EVIDENCE"),
            "official_first_public_time_status": official_driver.get(
                "first_public_time_status", "UNAVAILABLE"),
            "official_driver_age_lower_bound_days": _number(
                official_driver.get("official_driver_age_lower_bound_days")),
            "signal_date_driver_novelty_status": official_driver.get(
                "signal_date_novelty_status", "UNAVAILABLE"),
            "driver_freshness_readthrough": official_driver.get(
                "driver_freshness_readthrough", "UNAVAILABLE"),
            "official_driver_lifecycle_status": official_driver.get(
                "lifecycle_status", "UNAVAILABLE"),
            "multilane_role": role, "selection_bucket": bucket,
            "contradictions": "|".join(contradictions),
            "symbolic_is_real_theme": False, "official_primary": False,
            "entry_authorized": False,
        })
    candidates = pd.DataFrame(rows)
    buyable_symbolic = candidates[
        candidates.multilane_role.isin({
            "SYMBOLIC_DUAL_IDENTITY_CHALLENGER", "SYMBOLIC_PURE_BUYABLE_PIONEER",
            "SYMBOLIC_BUYABLE_FOLLOWER"})].copy()
    pure = (buyable_symbolic.sort_values(
        ["symbolic_buyable_pioneer_rank", "first_seal_minute", "code"]
    ).iloc[0] if len(buyable_symbolic) else None)
    duals = candidates[candidates.multilane_role.eq(
        "SYMBOLIC_DUAL_IDENTITY_CHALLENGER")].copy()
    dual = (duals.sort_values(
        ["symbolic_buyable_pioneer_rank", "first_seal_minute",
         "community_evidence_count", "code"],
        ascending=[True, True, False, True]).iloc[0] if len(duals) else None)
    narrative = candidates[candidates.legacy_theme_research_gate].copy()
    narrative = (narrative.sort_values(
        ["community_probability_NOT_CALIBRATED", "community_evidence_count", "code"],
        ascending=[False, False, True]).iloc[0] if len(narrative) else None)

    mappings = []
    for row in candidates[candidates.symbolic_cluster_id.ne("")].itertuples():
        cluster = meta.get(str(row.symbolic_cluster_id), {})
        for relay in cluster.get("relay_members", []):
            anchor_claim = claim_by_code.get(relay["code"], {})
            mappings.append({
                "signal_date": signal_date, "cluster_id": row.symbolic_cluster_id,
                "symbolic_theme_name": row.symbolic_theme_name,
                "high_anchor_code": relay["code"], "high_anchor_name": relay["name"],
                "high_anchor_boards": relay["boards"],
                "is_cluster_height_anchor": relay["boards"] == cluster.get("max_board"),
                "high_anchor_secondary_driver_category": anchor_claim.get(
                    "driver_claim_category", "UNAVAILABLE"),
                "high_anchor_negative_or_limited_claim": _truth(
                    anchor_claim.get("negative_or_limited_exposure", False)),
                "low_code": row.code, "low_name": row.name,
                "low_role": row.multilane_role,
                "mapping_quality": (
                    "DUAL_IDENTITY_POTENTIAL" if row.multilane_role == "SYMBOLIC_DUAL_IDENTITY_CHALLENGER"
                    else "PURE_SYMBOLIC_POTENTIAL" if row.multilane_role == "SYMBOLIC_PURE_BUYABLE_PIONEER"
                    else "OBSERVATION_ONLY"),
                "mapping_is_real_theme": False, "entry_authorized": False,
            })
    mapping_frame = pd.DataFrame(mappings)

    def identity(row):
        return None if row is None else {
            "code": str(row.code), "name": str(row["name"]),
            "business_theme": str(row.primary_business_theme),
            "symbolic_theme": str(row.symbolic_theme_name),
            "multilane_role": str(row.multilane_role),
            "secondary_driver_claim": str(row.secondary_driver_claim),
            "secondary_driver_claim_category": str(row.secondary_driver_claim_category),
            "catalyst_first_public_time_status": str(row.catalyst_first_public_time_status),
            "official_driver_evidence_status": str(row.official_driver_evidence_status),
            "official_first_public_time_status": str(row.official_first_public_time_status),
            "official_driver_age_lower_bound_days": (
                None if pd.isna(row.official_driver_age_lower_bound_days)
                else int(row.official_driver_age_lower_bound_days)),
            "driver_freshness_readthrough": str(row.driver_freshness_readthrough),
        }

    report = {
        "version": VERSION, "status": "CASE_DERIVED_CHALLENGER_DIAGNOSTIC_ONLY",
        "signal_date": signal_date, "first_board_candidates": int(len(candidates)),
        "symbolic_first_board_candidates": int(candidates.symbolic_cluster_id.ne("").sum()),
        "narrative_lane_candidate": identity(narrative),
        "symbolic_pure_buyable_pioneer": identity(pure),
        "symbolic_dual_identity_candidate": identity(dual),
        "official_first_board_primary": None, "winner_claimed": False,
        "governance": {
            "single_best_theme_can_hide_other_lanes": True,
            "all_theme_lanes_preserved": True,
            "symbolic_cluster_is_real_theme": False,
            "cross_lane_score_sum_allowed": False,
            "symbolic_candidate_replaces_official_first_board": False,
            "minimum_forward_date_clusters_before_promotion": 60,
            "entry_authorized": False,
        },
        "data_quality": {
            "symbolic_name_match_status": symbolic.get("data_quality", {}).get("name_match_status"),
            "symbolic_availability_status": symbolic.get("data_quality", {}).get("availability_status"),
            "full_market_completeness": symbolic.get("data_quality", {}).get("full_market_completeness"),
            "secondary_driver_claim_overlay": "ATTACHED" if driver_claims is not None else "UNAVAILABLE",
            "official_driver_time_overlay": (
                "PARTIAL_ATTACHED" if official_driver_times is not None else "UNAVAILABLE"),
        },
        "limitations": [
            "observed theme rosters are secondary narratives unless separately verified",
            "hash-matched secondary driver text is preserved but is not promoted to a company fact or verified catalyst",
            "symbolic name ladders can reverse abruptly and never prove a shared business catalyst",
            "community probabilities are uncalibrated and are used only to preserve the legacy narrative candidate",
            "auction/open confirmation and executable price remain unavailable at the close",
        ],
        "entry_authorized": False,
    }
    return candidates, mapping_frame, report


def markdown(candidates: pd.DataFrame, report: dict) -> str:
    def show(item):
        return "无" if not item else f"{item['name']}({item['code']})"
    lines = [
        "# 首板多通道身份复核 V229", "",
        f"信号日：{report['signal_date']}；状态：`{report['status']}`；正式首选：无；买入授权：否。", "",
        f"- 原叙事通道候选：{show(report['narrative_lane_candidate'])}",
        f"- 符号通道可买先锋：{show(report['symbolic_pure_buyable_pioneer'])}",
        f"- 符号＋叙事双身份候选：{show(report['symbolic_dual_identity_candidate'])}", "",
        "符号通道与产业/二手叙事通道不做总分相加。名称聚集只代表市场身份，不能替代业务、催化或次日确认。", "",
        "|代码|名称|产业/叙事主身份|符号身份|符号可买先锋序|多通道角色|高位映射|反对证据|", 
        "|---|---|---|---|---:|---|---|---|",
    ]
    view = candidates[candidates.symbolic_cluster_id.ne("")].sort_values(
        ["selection_bucket", "symbolic_buyable_pioneer_rank", "first_seal_minute"],
        na_position="last")
    for row in view.itertuples():
        rank = "不可用" if pd.isna(row.symbolic_buyable_pioneer_rank) else f"{row.symbolic_buyable_pioneer_rank:.0f}"
        lines.append(
            f"|{row.code}|{row.name}|{row.primary_business_lane}:{row.primary_business_theme}|"
            f"{row.symbolic_theme_name}|{rank}|{row.multilane_role}|"
            f"{row.symbolic_relay_ladder or '无'}|{row.contradictions or '无'}|")
    lines += ["", "## 验收边界", "",
              "至少累计60个不同交易日前向样本，并分别比较晋级、可成交B+2收益和B+1大亏。"
              "在此之前只作为首板漏检校正层，不替换正式规则。", ""]
    return "\n".join(lines)


def run(watchlist_path: str, theme_map_path: str, symbolic_report_path: str,
        output_dir: str, driver_claims_path: str | None = None,
        official_driver_times_path: str | None = None) -> dict:
    paths = [Path(watchlist_path), Path(theme_map_path), Path(symbolic_report_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V229 output")
    watch = pd.read_csv(paths[0], dtype={"code": str})
    theme_map = pd.read_csv(paths[1], dtype={"code": str})
    symbolic = json.loads(paths[2].read_text(encoding="utf-8-sig"))
    driver_claims = None
    if driver_claims_path:
        claim_path = Path(driver_claims_path)
        paths.append(claim_path)
        driver_claims = pd.read_csv(claim_path, dtype={"code": str})
    official_driver_times = None
    if official_driver_times_path:
        official_path = Path(official_driver_times_path)
        paths.append(official_path)
        official_driver_times = pd.read_csv(official_path, dtype={"code": str})
    candidates, mappings, report = build(
        watch, theme_map, symbolic, driver_claims, official_driver_times)
    target.mkdir(parents=True)
    candidate_path = target / "multilane_first_board_candidates.csv"
    mapping_path = target / "symbolic_high_to_low_mappings.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    mappings.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha(path) for path in paths}
    report["artifact_hashes"] = {
        candidate_path.name: _sha(candidate_path), mapping_path.name: _sha(mapping_path),
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "report.md").write_text(markdown(candidates, report), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watchlist", required=True)
    ap.add_argument("--theme-map", required=True)
    ap.add_argument("--symbolic-report", required=True)
    ap.add_argument("--driver-claims")
    ap.add_argument("--official-driver-times")
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.watchlist, ns.theme_map, ns.symbolic_report, ns.output,
                         ns.driver_claims, ns.official_driver_times),
                     ensure_ascii=False, indent=2))
