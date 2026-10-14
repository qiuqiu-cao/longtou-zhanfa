"""V225 evidence-layered same-height leader PK challenger.

The challenger fixes four descriptive defects in the legacy height ranking:
title-only risk notices are not treated as proven substantive hard risks,
reopening is classified by repair duration rather than count alone, symbolic
market themes are kept separate from real business themes, and every stock at
the same height is compared pairwise.  It is case-derived and diagnostic only;
it cannot select an official champion or authorise an entry before forward
validation and next-session confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v225-same-height-leader-pk-1"
SUBSTANTIVE_HARD_TERMS = (
    "立案", "调查", "行政处罚", "退市风险", "终止上市", "重大亏损",
    "债务逾期", "资金占用", "违规担保", "控制权不稳定",
)
NOTICE_ID_RE = re.compile(r"(AN\d+)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _clock_minutes(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).split(".")[0].strip().zfill(6)
    if len(text) != 6 or not text.isdigit():
        return None
    hour, minute, second = int(text[:2]), int(text[2:4]), int(text[4:])
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 60 + minute + second / 60


def classify_board_path(first_time, last_time, failed_count) -> dict:
    first, last, failed = _clock_minutes(first_time), _clock_minutes(last_time), _number(failed_count)
    if first is None or failed is None:
        return {"status": "UNAVAILABLE", "span_minutes": None, "score": 0.0}
    one_price_proxy = str(first_time).split(".")[0].zfill(6) == "092500" and failed == 0
    if one_price_proxy:
        return {"status": "ONE_PRICE_BUYABILITY_UNPROVEN", "span_minutes": 0.0, "score": -4.0}
    if failed == 0:
        return {"status": "LOCKED_NO_REOPEN", "span_minutes": 0.0, "score": 2.0}
    if last is None or last < first:
        return {"status": "REPAIR_DURATION_UNAVAILABLE", "span_minutes": None, "score": 0.0}
    span = last - first
    if span <= 30:
        return {"status": "QUICK_REPAIR", "span_minutes": round(span, 4), "score": 2.0}
    return {"status": "PROLONGED_CONTEST", "span_minutes": round(span, 4), "score": -3.0}


def classify_notice_risk(notices: pd.DataFrame, exact_query_scope: bool = False) -> dict:
    if notices.empty:
        return {
            "tier": ("NO_ACTIVE_RISK_IN_COVERED_INDEX" if exact_query_scope else "UNAVAILABLE"),
            "score": 0.0, "hard_veto": False,
            "evidence_complete": exact_query_scope, "titles": [],
        }
    active = notices[
        notices.risk_window_status.fillna("").astype(str).str.startswith("active_within_")
    ].copy()
    if active.empty:
        return {"tier": "NO_ACTIVE_RISK_IN_COVERED_INDEX", "score": 0.0,
                "hard_veto": False, "evidence_complete": True, "titles": []}
    titles = active.title.fillna("").astype(str).tolist()
    if "fulltext_review_status" in active:
        point_in_time = active.get(
            "point_in_time_eligible", pd.Series(False, index=active.index)
        ).map(_truthy)
        reviewed = active.fulltext_review_status.fillna("").astype(str).eq(
            "POINT_IN_TIME_REVIEWED") & point_in_time
        if reviewed.any():
            eligible = active[reviewed].copy()
            eligible["_risk_score"] = pd.to_numeric(
                eligible.get("fulltext_risk_score"), errors="coerce").fillna(0.0)
            worst = eligible.sort_values("_risk_score").iloc[0]
            content_class = str(worst.get("content_risk_class", ""))
            hard_veto = _truthy(worst.get("fulltext_hard_veto", False))
            tiers = {
                "SUBSTANTIVE_HARD_RISK": (
                    "FULLTEXT_AUTHENTICATED_SUBSTANTIVE_HARD_RISK" if hard_veto
                    else "FULLTEXT_MIRROR_SUBSTANTIVE_RISK_NO_HARD_VETO"),
                "HOT_THEME_CLARIFICATION_NEGATIVE": "FULLTEXT_HOT_THEME_CLARIFICATION_NEGATIVE",
                "ROUTINE_MARKET_AND_PERFORMANCE_RISK": "FULLTEXT_ROUTINE_MARKET_AND_PERFORMANCE_RISK",
                "ROUTINE_MARKET_RISK": "FULLTEXT_ROUTINE_MARKET_RISK",
                "NO_CLASSIFIED_SUBSTANTIVE_RISK": "FULLTEXT_NO_CLASSIFIED_SUBSTANTIVE_RISK",
            }
            return {
                "tier": tiers.get(content_class, "FULLTEXT_REVIEWED_OTHER"),
                "score": float(worst["_risk_score"]), "hard_veto": hard_veto,
                "evidence_complete": True, "titles": titles,
            }
        retrospective = active.fulltext_review_status.fillna("").astype(str).eq(
            "RETROSPECTIVE_REVIEW_NOT_POINT_IN_TIME")
        if retrospective.any() and pd.to_numeric(
                active.risk_hard, errors="coerce").fillna(0).gt(0).any():
            return {
                "tier": "RETROSPECTIVE_FULLTEXT_NOT_APPLIED_TITLE_ONLY",
                "score": -4.0, "hard_veto": False,
                "evidence_complete": False, "titles": titles,
            }
    full_status = active.get("full_text_status", pd.Series("UNAVAILABLE", index=active.index))
    full_text = active.get("full_text", pd.Series("", index=active.index)).fillna("").astype(str)
    full_verified = full_status.astype(str).str.upper().isin(
        {"VERIFIED", "POINT_IN_TIME_VERIFIED"})
    substantive = pd.Series(False, index=active.index)
    for term in SUBSTANTIVE_HARD_TERMS:
        substantive |= full_text.str.contains(term, regex=False)
    if (full_verified & substantive).any():
        return {"tier": "FULLTEXT_VERIFIED_SUBSTANTIVE_HARD_RISK", "score": -20.0,
                "hard_veto": True, "evidence_complete": True, "titles": titles}
    if pd.to_numeric(active.risk_hard, errors="coerce").fillna(0).gt(0).any():
        return {"tier": "TITLE_ONLY_RISK_NOTICE_REQUIRES_FULLTEXT", "score": -4.0,
                "hard_veto": False, "evidence_complete": False, "titles": titles}
    if active.risk_type.fillna("").astype(str).str.contains("普通异常波动").any():
        return {"tier": "ORDINARY_ABNORMAL_VOLATILITY_NOTICE", "score": -1.0,
                "hard_veto": False, "evidence_complete": True, "titles": titles}
    return {"tier": "ACTIVE_NOTICE_OTHER", "score": -1.0, "hard_veto": False,
            "evidence_complete": False, "titles": titles}


def _raw_map(raw: pd.DataFrame) -> dict[str, dict]:
    aliases = {
        "code": ("代码", "code"), "first": ("首次封板时间", "first_seal_time"),
        "last": ("最后封板时间", "last_seal_time"),
        "failed": ("炸板次数", "failed_seal_count"),
    }
    columns = {}
    for key, choices in aliases.items():
        columns[key] = next((value for value in choices if value in raw), None)
    if any(columns[key] is None for key in aliases):
        raise ValueError(f"raw limit-up fields missing: {columns}")
    result = {}
    for _, row in raw.iterrows():
        code = str(row[columns["code"]]).split(".")[0].zfill(6)
        result[code] = {
            "first_seal_time_raw": row[columns["first"]],
            "last_seal_time_raw": row[columns["last"]],
            "failed_seal_count_raw": row[columns["failed"]],
        }
    return result


def build(review: dict, raw: pd.DataFrame, announcements: pd.DataFrame,
          announcement_manifest: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidates = review.get("relay_candidates", [])
    if not candidates:
        raise ValueError("review contains no relay candidates")
    announcement_meta = (announcement_manifest or {}).get("announcements", {})
    exact_announcement_scope = bool(
        str((announcement_manifest or {}).get("signal_date")) == str(review.get("signal_date"))
        and int((announcement_manifest or {}).get("candidate_count", -1)) == len(candidates)
        and int(announcement_meta.get("candidate_codes", -1)) == len(candidates)
        and int(announcement_meta.get("failed_codes", -1)) == 0
        and str(announcement_meta.get("status", "")).lower() == "verified_index_date_only"
    )
    raw_by_code = _raw_map(raw)
    announcements = announcements.copy()
    announcements["code"] = announcements.code.astype(str).str.zfill(6)
    rows = []
    for item in candidates:
        code = str(item["code"]).zfill(6)
        raw_item = raw_by_code.get(code, {})
        path = classify_board_path(
            raw_item.get("first_seal_time_raw", item.get("first_seal_time")),
            raw_item.get("last_seal_time_raw"),
            raw_item.get("failed_seal_count_raw", item.get("failed_seal_count")),
        )
        notice = classify_notice_risk(
            announcements[announcements.code.eq(code)], exact_announcement_scope)
        theme = item.get("theme_evidence") or {}
        theme_verified_score = 6.0 if theme.get("status") == "VERIFIED" else 0.0
        ladder_score = 4.0 if theme.get("ladder_verified") is True else 0.0
        support = _number(theme.get("support_ex_self"))
        breadth_score = min(max(support or 0.0, 0.0), 3.0)
        role_score = 3.0 if item.get("theme_role") in {"THEME_HEIGHT_LEADER", "SAME_HEIGHT_CORE"} else 0.0
        first = _clock_minutes(raw_item.get("first_seal_time_raw", item.get("first_seal_time")))
        initiative_score = 2.0 if first is not None and first <= 600 else (1.0 if first is not None and first <= 630 else 0.0)
        if path["status"] == "ONE_PRICE_BUYABILITY_UNPROVEN":
            initiative_score = 0.0
        turnover = _number(item.get("turnover"))
        turnover_score = 2.0 if turnover is not None and 4 <= turnover <= 20 else (-4.0 if turnover is not None and turnover > 30 else 0.0)
        symbolic_score = 0.0
        symbolic_tags = []
        for cluster in item.get("symbolic_name_clusters", []):
            classification = cluster.get("classification")
            if classification == "SYMBOLIC_LADDER_CANDIDATE":
                symbolic_score += 4.0
            elif classification == "SYMBOLIC_DIFFUSION_CANDIDATE":
                symbolic_score += 1.0
            symbolic_tags.append(cluster.get("token"))
        symbolic_score = min(symbolic_score, 5.0)
        factors = {
            "verified_real_theme": theme_verified_score,
            "observed_ladder": ladder_score,
            "support_breadth_capped": breadth_score,
            "theme_height_role": role_score,
            "board_path": path["score"],
            "first_seal_initiative": initiative_score,
            "turnover_quality": turnover_score,
            "symbolic_market_identity": symbolic_score,
            "announcement_risk": notice["score"],
        }
        missing = []
        if path["status"] in {"UNAVAILABLE", "REPAIR_DURATION_UNAVAILABLE"}:
            missing.append("BOARD_PATH")
        if notice["tier"] == "UNAVAILABLE":
            missing.append("ANNOUNCEMENT_SCOPE")
        if theme.get("status") != "VERIFIED":
            missing.append("VERIFIED_REAL_THEME")
        score = float(sum(factors.values()))
        rows.append({
            "signal_date": review.get("signal_date"), "code": code,
            "name": item.get("name"), "boards": int(item["boards"]),
            "height_group_id": f"{review.get('signal_date')}:H{int(item['boards'])}",
            "research_score_NOT_PROBABILITY": score,
            "board_path_status": path["status"], "repair_span_minutes": path["span_minutes"],
            "failed_seal_count": item.get("failed_seal_count"),
            "first_seal_time": raw_item.get("first_seal_time_raw", item.get("first_seal_time")),
            "last_seal_time": raw_item.get("last_seal_time_raw"),
            "announcement_risk_tier": notice["tier"],
            "announcement_hard_veto": notice["hard_veto"],
            "announcement_titles": "|".join(notice["titles"]),
            "symbolic_tags": "|".join(tag for tag in symbolic_tags if tag),
            "symbolic_is_real_theme": False,
            "missing_evidence": "|".join(missing),
            "factor_breakdown_json": json.dumps(factors, ensure_ascii=False, sort_keys=True),
            "entry_authorized": False,
            **{f"factor_{key}": value for key, value in factors.items()},
        })
    frame = pd.DataFrame(rows)
    frame["rank_within_height_descriptive"] = frame.groupby("height_group_id")[
        "research_score_NOT_PROBABILITY"].rank(method="min", ascending=False)

    pair_rows = []
    for group_id, group in frame.groupby("height_group_id", sort=True):
        if len(group) < 2:
            continue
        for (_, left), (_, right) in itertools.combinations(group.sort_values("code").iterrows(), 2):
            factor_columns = [c for c in frame if c.startswith("factor_") and c != "factor_breakdown_json"]
            left_wins, right_wins, ties = [], [], []
            differences = {}
            for factor in factor_columns:
                delta = float(left[factor] - right[factor])
                differences[factor + "_left_minus_right"] = delta
                (left_wins if delta > 0 else right_wins if delta < 0 else ties).append(factor.removeprefix("factor_"))
            pair_rows.append({
                "signal_date": left.signal_date, "height_group_id": group_id,
                "boards": int(left.boards), "left_code": left.code, "left_name": left["name"],
                "right_code": right.code, "right_name": right["name"],
                "score_difference_left_minus_right": float(
                    left.research_score_NOT_PROBABILITY - right.research_score_NOT_PROBABILITY),
                "left_advantages": "|".join(left_wins), "right_advantages": "|".join(right_wins),
                "ties": "|".join(ties), "official_winner_claimed": False,
                "entry_authorized": False, **differences,
            })
    pairs = pd.DataFrame(pair_rows)
    leaders = []
    for group_id, group in frame.groupby("height_group_id", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(["research_score_NOT_PROBABILITY", "code"], ascending=[False, True])
        leaders.append({
            "height_group_id": group_id, "boards": int(ordered.iloc[0].boards),
            "descriptive_leader_code": ordered.iloc[0].code,
            "descriptive_leader_name": ordered.iloc[0]["name"],
            "score_margin": float(ordered.iloc[0].research_score_NOT_PROBABILITY - ordered.iloc[1].research_score_NOT_PROBABILITY),
            "official_champion": None, "winner_claimed": False,
        })
    report = {
        "version": VERSION, "status": "CASE_DERIVED_CHALLENGER_DIAGNOSTIC_ONLY",
        "signal_date": review.get("signal_date"), "candidate_rows": int(len(frame)),
        "same_height_pair_rows": int(len(pairs)), "group_descriptive_leaders": leaders,
        "policy": {
            "announcement_query_scope_exact": exact_announcement_scope,
            "title_only_risk_notice_is_substantive_hard_veto": False,
            "fulltext_verified_substantive_risk_can_veto": True,
            "raw_failed_seal_count_monotonic_penalty": False,
            "quick_repair_and_locked_board_same_path_score": True,
            "symbolic_identity_kept_separate_from_real_theme": True,
            "symbolic_identity_score_cap": 5,
            "official_champion_claimed": False, "entry_authorized": False,
            "minimum_forward_date_clusters_before_promotion": 60,
        },
        "limitations": [
            "weights are predeclared case-derived hypotheses, not fitted probabilities",
            "current announcement archive has title/index evidence but no verified full text",
            "seal times are provider snapshots, not Level-2 order or queue evidence",
            "next-session auction and opening confirmation remain mandatory",
        ],
    }
    return frame, pairs, report


def markdown(rows: pd.DataFrame, pairs: pd.DataFrame, report: dict) -> str:
    lines = [
        "# 同高度龙头PK V225", "",
        f"状态：`{report['status']}`；信号日：{report['signal_date']}；正式冠军：无；买入授权：否。", "",
        "符号题材只表示市场辨识度，不能替代真实业务题材。公告只有标题/索引时，不得升级成已核验实质性硬风险。", "",
        "## 各高度描述性领先者", "",
        "|高度|描述性领先者|领先分差|正式冠军|", "|---:|---|---:|---|",
    ]
    for item in report["group_descriptive_leaders"]:
        lines.append(
            f"|{item['boards']}板|{item['descriptive_leader_name']}({item['descriptive_leader_code']})|"
            f"{item['score_margin']:.2f}|无，需前向验证|"
        )
    lines += ["", "## 候选评分拆解", "",
              "|代码|名称|板位|研究分|封板路径|修复分钟|公告风险层|符号标签|缺失证据|",
              "|---|---|---:|---:|---|---:|---|---|---|"]
    for _, row in rows.sort_values(["boards", "research_score_NOT_PROBABILITY"], ascending=[False, False]).iterrows():
        span = "不可得" if pd.isna(row.repair_span_minutes) else f"{row.repair_span_minutes:.2f}"
        lines.append(
            f"|{row.code}|{row['name']}|{row.boards}|{row.research_score_NOT_PROBABILITY:.2f}|"
            f"{row.board_path_status}|{span}|{row.announcement_risk_tier}|"
            f"{row.symbolic_tags if pd.notna(row.symbolic_tags) else '无'}|"
            f"{row.missing_evidence if row.missing_evidence else '无'}|"
        )
    lines += ["", "## 同高度逐对比较", "",
              "|高度|左侧|右侧|左减右|左侧优势|右侧优势|正式赢家|",
              "|---:|---|---|---:|---|---|---|"]
    for _, row in pairs.sort_values(["boards", "score_difference_left_minus_right"], ascending=[False, False]).iterrows():
        lines.append(
            f"|{row.boards}|{row.left_name}({row.left_code})|{row.right_name}({row.right_code})|"
            f"{row.score_difference_left_minus_right:.2f}|{row.left_advantages or '无'}|"
            f"{row.right_advantages or '无'}|无|"
        )
    lines += ["", "## 验收边界", "",
              "至少积累60个不同交易日簇；按交易日聚类比较晋级率、可成交B+2收益和B+1大亏率。"
              "三项不能同时改善时，不得替换原正式规则。竞价、开盘承接和真实可成交性仍是次日必检条件。", ""]
    return "\n".join(lines)


def _attach_fulltext_review(announcements: pd.DataFrame, review_path: Path) -> pd.DataFrame:
    reviewed = pd.read_csv(review_path, dtype={"code": str, "notice_id": str})
    reviewed["code"] = reviewed.code.astype(str).str.zfill(6)
    result = announcements.copy()
    result["notice_id"] = result.url.fillna("").astype(str).map(
        lambda value: (NOTICE_ID_RE.search(value).group(1) if NOTICE_ID_RE.search(value) else ""))
    fields = [
        "code", "notice_id", "content_risk_class", "fulltext_review_status",
        "point_in_time_eligible", "fulltext_risk_score", "fulltext_hard_veto",
        "source_authenticated_for_hard_veto", "text_sha256",
    ]
    missing = [field for field in fields if field not in reviewed]
    if missing:
        raise ValueError(f"V228 review fields missing: {missing}")
    return result.merge(reviewed[fields], on=["code", "notice_id"], how="left",
                        validate="many_to_one")


def run(review_path: str, raw_path: str, announcement_path: str,
        announcement_manifest_path: str, output_dir: str,
        fulltext_review_path: str | None = None,
        fulltext_review_report_path: str | None = None) -> dict:
    paths = [Path(review_path), Path(raw_path), Path(announcement_path),
             Path(announcement_manifest_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V225 PK output")
    review = json.loads(paths[0].read_text(encoding="utf-8-sig"))
    raw = pd.read_csv(paths[1], dtype=str)
    announcements = pd.read_csv(paths[2], dtype={"code": str})
    announcement_manifest = json.loads(paths[3].read_text(encoding="utf-8-sig"))
    expected_hash = (announcement_manifest.get("files", {}).get("announcements", {})
                     .get("sha256"))
    if not expected_hash or _sha(paths[2]) != expected_hash:
        raise ValueError("announcement CSV does not match its manifest hash")
    if bool(fulltext_review_path) != bool(fulltext_review_report_path):
        raise ValueError("V228 review CSV and report must be supplied together")
    if fulltext_review_path:
        review_file, review_report_file = Path(fulltext_review_path), Path(fulltext_review_report_path)
        review_report = json.loads(review_report_file.read_text(encoding="utf-8-sig"))
        if review_report.get("review_sha256") != _sha(review_file):
            raise ValueError("V228 review CSV hash mismatch")
        if review_report.get("announcement_index_sha256") != _sha(paths[2]):
            raise ValueError("V228 review does not belong to this announcement index")
        announcements = _attach_fulltext_review(announcements, review_file)
        paths.extend([review_file, review_report_file])
    rows, pairs, report = build(review, raw, announcements, announcement_manifest)
    report["fulltext_review_attached"] = bool(fulltext_review_path)
    if fulltext_review_path:
        report["policy"]["late_fulltext_backdated"] = False
        report["policy"]["mirror_alone_can_authorize_hard_veto"] = False
        report["limitations"] = [
            ("full-text review is attached, but evidence received after the decision cutoff "
             "remains retrospective and is not applied")
            if item == "current announcement archive has title/index evidence but no verified full text"
            else item
            for item in report["limitations"]
        ]
    target.mkdir(parents=True)
    rows_path, pairs_path = target / "same_height_scores.csv", target / "same_height_pairs.csv"
    rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha(path) for path in paths}
    report["artifact_hashes"] = {rows_path.name: _sha(rows_path), pairs_path.name: _sha(pairs_path)}
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "report.md").write_text(markdown(rows, pairs, report), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", required=True)
    ap.add_argument("--raw-limit-up", required=True)
    ap.add_argument("--announcements", required=True)
    ap.add_argument("--announcement-manifest", required=True)
    ap.add_argument("--fulltext-review")
    ap.add_argument("--fulltext-review-report")
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()
    print(json.dumps(run(ns.review, ns.raw_limit_up, ns.announcements,
                         ns.announcement_manifest, ns.output,
                         ns.fulltext_review, ns.fulltext_review_report),
                     ensure_ascii=False, indent=2))
