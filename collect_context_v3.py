"""Point-in-time theme and announcement archive for the frozen V3 leader model.

This companion collector intentionally lives outside ``leader_system_v3.py`` so
the frozen prediction code does not change.  It records current concept
membership and public announcement indexes with timestamps and explicit quality
labels.  Current membership is never backfilled as if it were historical truth.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "v3"
IMPORTED = DATA / "imported"
RESULTS = ROOT / "results" / "v3" / "context"

GENERIC_CONCEPT_TOKENS = (
    "融资融券", "沪股通", "深股通", "标准普尔", "富时罗素", "MSCI", "证金持股",
    "机构重仓", "基金重仓", "昨日涨停", "昨日连板", "预亏预减", "转债标的",
    "破净股", "低价股", "高股息", "国企改革", "参股银行", "参股券商",
)


def code6(value: Any) -> str:
    return str(value).split(".")[0].strip().zfill(6)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(call: Callable[[], Any], attempts: int = 3, base_delay: float = 0.8) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # Provider instability is evidence, not silence.
            last = exc
            if attempt + 1 < attempts:
                time.sleep(base_delay * (attempt + 1))
    assert last is not None
    raise last


def classify_announcement(title: Any, category: Any = "") -> tuple[str, int, list[str]]:
    title_text = str(title or "").strip()
    category_text = str(category or "").strip()
    text = f"{title_text} {category_text}".strip()
    matches: list[str] = []
    # An incentive award merely satisfying vesting/unlock conditions is not
    # evidence that the shares will become tradable on the next session.  The
    # actual listing/circulation notice remains a hard supply-risk event.
    unlock_conditions_tokens = (
        "解除限售期解除限售条件成就", "解除限售条件成就",
    )
    actual_unlock_tokens = (
        "限售股份上市流通", "限售股上市流通", "解除限售股份上市流通",
        "解除限售股票上市流通", "上市流通提示性公告",
    )
    unlock_conditions_only = (
        any(token in text for token in unlock_conditions_tokens)
        and not any(token in text for token in actual_unlock_tokens)
    )
    reduction_closed_tokens = (
        "减持计划完成", "减持计划实施完成", "减持实施完毕", "减持结果",
        "减持计划实施完毕", "减持股份计划实施完毕", "减持股份结果",
        "减持股份计划期限届满", "减持股份计划实施期限届满",
        "减持计划期限届满", "减持计划时间届满",
        "减持期限届满", "减持时间届满", "减持计划提前终止", "终止减持计划",
    )
    # Remove only explicitly completed reduction phrases. A combined title
    # can still contain a NEW plan or an unrelated hard risk. Category is
    # retained: ambiguous broad reduction categories require manual review.
    closed = False
    for token in sorted(reduction_closed_tokens, key=len, reverse=True):
        if token in title_text:
            closed = True
            title_text = title_text.replace(token, " ")
    # A broad provider category such as "股东增减持" must not turn an
    # explicitly completed title back into a new active reduction.  A combined
    # title that also says "拟减持" or announces a new plan remains hard.
    text = f"{title_text} {'' if closed else category_text}".strip()
    active_reduction_tokens = (
        "拟减持", "减持预披露", "减持计划预披露",
        "新增减持计划", "减持股份计划公告", "减持计划公告",
    )
    reduction_closed_without_new_plan = (
        closed and not any(token in text for token in active_reduction_tokens)
    )
    # Reply/special-explanation wording is not evidence that a risk is closed.
    rules = [
        ("风险提示_退市或重大违法", ("退市风险", "终止上市", "重大违法", "立案调查"), 1),
        ("风险提示_监管", ("监管函", "监管工作函", "监管警示", "问询函", "警示函", "行政处罚", "纪律处分", "停牌核查"), 1),
        ("风险提示_风险公告", ("股票交易风险提示公告", "交易风险提示公告", "风险提示性公告"), 1),
        ("风险提示_严重异常波动", ("严重异常波动",), 1),
        # Ordinary abnormal-move notices are context/crowding evidence.  They
        # do not mechanically veto a strong stock; explicit risk notices,
        # serious abnormal moves and regulatory actions remain hard above.
        ("减持", ("减持计划", "拟减持", "减持股份", "集中竞价减持"), 1),
        ("解禁", ("解除限售", "限售股份上市流通", "限售股上市流通"), 1),
        ("业绩雷", ("预亏", "续亏", "首亏", "业绩预减", "亏损", "商誉减值"), 1),
        ("监管背景_普通异常波动", ("交易异常波动", "股票交易异常"), 0),
    ]
    for label, tokens, hard in rules:
        if label == "减持" and reduction_closed_without_new_plan:
            continue
        if label == "解禁" and unlock_conditions_only:
            matches.append("unlock_conditions_met_pending_listing")
            return "解禁条件成就_待上市流通公告", 0, matches
        if any(token in text for token in tokens):
            matches.append(label)
            return label, hard, matches
    if closed:
        return "historical_corporate_action_closed", 0, ["explicit_reduction_closed_title_only"]
    return "其他公告", 0, matches


def apply_risk_window(label: str, hard: int, title: Any, age_days: int) -> tuple[str, int, str]:
    if not hard:
        return label, 0, "not_hard_or_closed"
    windows = {
        "风险提示_风险公告": 5,
        "风险提示_严重异常波动": 5,
        "风险提示_监管": 15,
        "风险提示_退市或重大违法": 45,
        "减持": 45,
        "解禁": 15,
        "业绩雷": 45,
    }
    horizon = windows.get(label, 0)
    if age_days < 0:
        return "future_notice_not_yet_effective", 0, "future_dated"
    if age_days > horizon:
        return "historical_notice_expired", 0, f"expired_after_{horizon}d"
    return label, 1, f"active_within_{horizon}d"


def is_generic_concept(name: Any) -> bool:
    text = str(name).strip()
    return not text or any(token.lower() in text.lower() for token in GENERIC_CONCEPT_TOKENS)


def choose_theme(evidence: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "name", "theme", "theme_status", "source",
               "source_timestamp", "same_day_limit_up_support", "same_day_ladder_support"]
    if evidence.empty:
        return pd.DataFrame(columns=columns)
    usable = evidence[(~evidence["generic_concept"]) & (evidence["membership_status"] == "verified_current_snapshot")].copy()
    usable = usable[(usable["same_day_limit_up_support"] >= 2) | (usable["same_day_ladder_support"] >= 2)]
    if usable.empty:
        return pd.DataFrame(columns=columns)
    usable["_name_length"] = usable["concept"].astype(str).str.len()
    usable = usable.sort_values(
        ["code", "same_day_ladder_support", "same_day_limit_up_support", "_name_length", "concept"],
        ascending=[True, False, False, True, True])
    selected = usable.drop_duplicates("code").rename(columns={"concept": "theme"})
    selected["theme_status"] = "current_membership_with_same_day_limit_support"
    return selected.reindex(columns=columns)


def select_candidate_scope(all_candidates: pd.DataFrame, requested_date: str | None) -> tuple[str, pd.DataFrame]:
    """Accept legacy ``date`` or full-pool ``signal_date`` and exclude failed boards."""
    date_column = "date" if "date" in all_candidates else "signal_date" if "signal_date" in all_candidates else None
    if date_column is None or not {"code", "name"}.issubset(all_candidates.columns):
        raise ValueError("candidate input requires date/signal_date, code and name")
    signal_date = pd.Timestamp(requested_date or all_candidates[date_column].max()).strftime("%Y-%m-%d")
    candidates = all_candidates[all_candidates[date_column].astype(str) == signal_date].copy()
    if "pool_assignment" in candidates:
        candidates = candidates[candidates.pool_assignment.eq("CLOSING_LIMIT_UP")]
    if "is_mainboard" in candidates:
        main = candidates.is_mainboard.astype(str).str.lower().isin({"true", "1", "1.0"})
        candidates = candidates[main]
    candidates["code"] = candidates["code"].map(code6)
    if candidates.code.duplicated().any():
        raise ValueError("duplicate codes in closing-limit candidate scope")
    return signal_date, candidates


def append_dedup(frame: pd.DataFrame, path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path, dtype={"code": str})
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def collect_announcements(ak: Any, candidates: pd.DataFrame, signal_date: str,
                          lookback_days: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = pd.Timestamp(signal_date)
    begin = end - pd.Timedelta(days=lookback_days)
    collected_at = datetime.now().isoformat(timespec="seconds")
    errors: dict[str, str] = {}

    def fetch(row: dict[str, Any]) -> pd.DataFrame:
        code = code6(row["code"])
        frame = retry(lambda: ak.stock_individual_notice_report(
            security=code, symbol="全部", begin_date=begin.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d")), attempts=2)
        if frame.empty:
            return frame
        frame = frame.rename(columns={"代码": "code", "名称": "name", "公告标题": "title",
                                      "公告类型": "announcement_category", "公告日期": "announcement_date",
                                      "网址": "url"})
        frame["code"] = frame["code"].map(code6)
        return frame

    pieces: list[pd.DataFrame] = []
    rows = candidates[["code", "name"]].drop_duplicates("code").to_dict("records")
    with futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 5))) as executor:
        jobs = {executor.submit(fetch, row): row for row in rows}
        for job in futures.as_completed(jobs):
            row = jobs[job]
            try:
                piece = job.result()
                if not piece.empty:
                    pieces.append(piece)
            except Exception as exc:
                errors[code6(row["code"])] = str(exc)[:300]
    if not pieces:
        empty = pd.DataFrame(columns=["code", "name", "published_at", "risk_type", "title",
                                      "announcement_category", "url", "risk_hard", "published_at_status",
                                      "source", "source_timestamp", "data_status"])
        status = "unavailable" if errors else "verified_no_rows"
        return empty, {"status": status, "rows": 0, "errors": errors}
    result = pd.concat(pieces, ignore_index=True).drop_duplicates(["code", "title", "announcement_date"])
    dates = pd.to_datetime(result["announcement_date"], errors="coerce")
    classified = result.apply(lambda row: classify_announcement(row["title"], row["announcement_category"]), axis=1)
    windowed = [apply_risk_window(value[0], value[1], result.iloc[index]["title"],
                                  int((end.normalize() - dates.iloc[index].normalize()).days)
                                  if pd.notna(dates.iloc[index]) else 9999)
                for index, value in enumerate(classified)]
    result["risk_type"] = [value[0] for value in windowed]
    result["risk_hard"] = [value[1] for value in windowed]
    result["risk_window_status"] = [value[2] for value in windowed]
    # The provider exposes only a date.  23:59 conservatively prevents same-day
    # announcements from leaking into a 15:00 close decision.
    result["published_at"] = dates.dt.strftime("%Y-%m-%d") + " 23:59:59"
    result["published_at_status"] = "date_only_conservative_end_of_day"
    result["source"] = "Eastmoney announcement index via AKShare"
    result["source_timestamp"] = collected_at
    result["data_status"] = "verified_index_date_only"
    keep = ["code", "name", "published_at", "risk_type", "title", "announcement_category",
            "url", "risk_hard", "risk_window_status", "published_at_status", "source", "source_timestamp", "data_status"]
    return result[keep].sort_values(["code", "published_at", "title"]), {
        "status": "partial" if errors else "verified_index_date_only", "rows": int(len(result)),
        "candidate_codes": len(rows), "failed_codes": len(errors), "errors": errors,
    }


def collect_concepts(ak: Any, candidates: pd.DataFrame, signal_date: str,
                     limit_up_codes: set[str], workers: int,
                     concept_limit: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    collected_at = datetime.now().isoformat(timespec="seconds")
    try:
        names = retry(ak.stock_board_concept_name_em, attempts=3)
    except Exception as exc:
        return pd.DataFrame(), {"status": "unavailable", "rows": 0, "error": str(exc)[:500]}
    name_column = "板块名称" if "板块名称" in names else "板块名称" if len(names.columns) == 0 else names.columns[1]
    code_column = "板块代码" if "板块代码" in names else names.columns[0]
    boards = names[[code_column, name_column]].dropna().drop_duplicates().to_dict("records")
    if concept_limit:
        boards = boards[:concept_limit]
    candidate_map = {code6(row.code): str(row.name) for row in candidates[["code", "name"]].itertuples(index=False)}
    candidate_codes = set(candidate_map)
    ladder_codes = set(candidate_codes)
    errors: dict[str, str] = {}
    evidence_rows: list[dict[str, Any]] = []

    def fetch(board: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
        name = str(board[name_column])
        members = retry(lambda: ak.stock_board_concept_cons_em(symbol=name), attempts=2)
        return board, members

    with futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        jobs = {executor.submit(fetch, board): board for board in boards}
        for job in futures.as_completed(jobs):
            board = jobs[job]
            concept = str(board[name_column])
            try:
                _, members = job.result()
                if members.empty:
                    continue
                member_code_column = "代码" if "代码" in members else members.columns[1]
                member_codes = set(members[member_code_column].map(code6))
                matched = member_codes & candidate_codes
                if not matched:
                    continue
                limit_support = len(member_codes & limit_up_codes)
                ladder_support = len(member_codes & ladder_codes)
                for code in matched:
                    evidence_rows.append({
                        "date": signal_date, "code": code, "name": candidate_map[code],
                        "concept": concept, "concept_code": str(board[code_column]),
                        "same_day_limit_up_support": limit_support,
                        "same_day_ladder_support": ladder_support,
                        "generic_concept": is_generic_concept(concept),
                        "membership_status": "verified_current_snapshot",
                        "source": "Eastmoney current concept constituents via AKShare",
                        "source_timestamp": collected_at,
                    })
            except Exception as exc:
                errors[concept] = str(exc)[:220]
    evidence = pd.DataFrame(evidence_rows)
    return evidence, {
        "status": "partial" if errors else "verified_current_snapshot",
        "concepts_requested": len(boards), "concepts_failed": len(errors),
        "candidate_concept_rows": len(evidence), "errors": errors,
        "warning": "current membership snapshot is valid only from collection time forward; never backfill history",
    }


def make_report(signal_date: str, themes: pd.DataFrame, announcements: pd.DataFrame,
                manifest: dict[str, Any]) -> str:
    lines = [f"# {signal_date} 题材与公告证据报告", "", "## 数据质量", "",
             f"- 概念成员：{manifest['concepts']['status']}",
             f"- 公告索引：{manifest['announcements']['status']}",
             "- 概念为采集时点的当前成员快照，不得回填成历史成分。",
             "- 公告接口只有日期没有分钟，统一按23:59:59处理，避免泄漏到当日15:00决策。", "",
             "## 选中题材", ""]
    if themes.empty:
        concept_status = manifest["concepts"]["status"]
        if concept_status in {"unavailable", "skipped"}:
            lines.append("概念成员证据不可用或本次主动跳过，无法判断是否存在满足同日共振的细分题材；不得解释为无题材。")
        else:
            lines.append("在本次已取得的概念成员证据中，无满足同日涨停/连板共振的可验证细分概念。")
    else:
        lines += ["| 代码 | 名称 | 题材 | 涨停支持 | 连板支持 |", "|---|---|---|---:|---:|"]
        for row in themes.itertuples(index=False):
            lines.append(f"| {row.code} | {row.name} | {row.theme} | {row.same_day_limit_up_support} | {row.same_day_ladder_support} |")
    risks = announcements[pd.to_numeric(announcements["risk_hard"], errors="coerce") == 1] if not announcements.empty else announcements
    lines += ["", "## 风险公告", ""]
    if risks.empty:
        lines.append("回看窗口内未在公共公告索引识别到预定义风险类型；不等于官方公告核验完整。")
    else:
        lines += ["| 代码 | 名称 | 日期 | 风险类型 | 标题 |", "|---|---|---|---|---|"]
        for row in risks.sort_values(["published_at", "code"], ascending=[False, True]).itertuples(index=False):
            lines.append(f"| {row.code} | {row.name} | {str(row.published_at)[:10]} | {row.risk_type} | {row.title} |")
    lines += ["", "该报告只提供带时间戳的输入证据，不改变已经封存的同日模型快照。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive point-in-time themes and announcement risks")
    parser.add_argument("--date", help="signal date; defaults to latest candidate date")
    parser.add_argument("--candidates", type=Path, default=DATA / "leader_candidates.csv",
                        help="candidate CSV; defaults to the V3 archive")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--concept-limit", type=int)
    parser.add_argument("--skip-concepts", action="store_true")
    parser.add_argument("--skip-announcements", action="store_true",
                        help="write a separate concept-retry package without refetching notices")
    parser.add_argument("--commit-announcements", action="store_true")
    parser.add_argument("--commit-themes", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULTS,
                        help="separate evidence directory; existing artifacts are never overwritten")
    args = parser.parse_args()

    import akshare as ak

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.candidates
    if not candidate_path.exists():
        raise SystemExit("run leader_system_v3.py build first")
    all_candidates = pd.read_csv(candidate_path, dtype={"code": str}, low_memory=False)
    signal_date, candidates = select_candidate_scope(all_candidates, args.date)
    if candidates.empty:
        raise SystemExit(f"no candidates for {signal_date}")
    raw_limit_path = DATA / "raw" / signal_date.replace("-", "") / "limit_up.csv"
    limit_up_codes = set(candidates["code"])
    if raw_limit_path.exists():
        raw = pd.read_csv(raw_limit_path, dtype={"代码": str})
        # Keep reconciled closing candidates (for example, a provider-pool
        # misclassification corrected by an independent close quote) while
        # also retaining non-candidate limit-ups as same-day theme support.
        limit_up_codes |= set(raw["代码"].map(code6))

    if args.skip_announcements:
        announcements = pd.DataFrame(columns=[
            "code", "name", "published_at", "title", "announcement_category",
            "risk_type", "risk_hard", "matched_rules", "url", "source",
            "source_timestamp", "data_status",
        ])
        announcement_meta = {"status": "skipped_for_concept_retry", "rows": 0}
    else:
        announcements, announcement_meta = collect_announcements(
            ak, candidates, signal_date, args.lookback_days, args.workers)
    if args.skip_concepts:
        concepts, concept_meta = pd.DataFrame(), {"status": "skipped", "rows": 0}
    else:
        concepts, concept_meta = collect_concepts(
            ak, candidates, signal_date, limit_up_codes, args.workers, args.concept_limit)
    themes = choose_theme(concepts)

    prefix = args.output_dir / signal_date
    announcements_path = prefix.with_name(prefix.name + "_announcements.csv")
    concepts_path = prefix.with_name(prefix.name + "_concept_evidence.csv")
    themes_path = prefix.with_name(prefix.name + "_themes.csv")
    paths_to_create = [announcements_path, concepts_path, themes_path,
                       prefix.with_name(prefix.name + "_manifest.json"),
                       prefix.with_name(prefix.name + "_context_report.md")]
    existing = [str(path) for path in paths_to_create if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite context evidence: " + ",".join(existing))
    announcements.to_csv(announcements_path, index=False, encoding="utf-8-sig")
    concepts.to_csv(concepts_path, index=False, encoding="utf-8-sig")
    themes.to_csv(themes_path, index=False, encoding="utf-8-sig")
    if args.commit_announcements and not announcements.empty:
        append_dedup(announcements, IMPORTED / "announcements.csv", ["code", "published_at", "title"])
    if args.commit_themes and not themes.empty:
        append_dedup(themes, IMPORTED / "themes.csv", ["date", "code"])

    manifest = {
        "signal_date": signal_date, "collected_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_count": len(candidates), "announcements": announcement_meta,
        "concepts": concept_meta, "selected_themes": len(themes),
        "committed_announcements": bool(args.commit_announcements),
        "committed_themes": bool(args.commit_themes),
        "files": {
            "announcements": {"path": str(announcements_path), "sha256": sha256_file(announcements_path)},
            "concept_evidence": {"path": str(concepts_path), "sha256": sha256_file(concepts_path)},
            "themes": {"path": str(themes_path), "sha256": sha256_file(themes_path)},
        },
    }
    manifest_path = prefix.with_name(prefix.name + "_manifest.json")
    report_path = prefix.with_name(prefix.name + "_context_report.md")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(make_report(signal_date, themes, announcements, manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
