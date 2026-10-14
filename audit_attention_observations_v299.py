"""Rebuild primary raw observations, time-gated views and paired rank changes."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import tomllib

from attention_archive_dispatch_v299 import reconstruct
from attention_asof_v299 import view_at
from attention_observations_v299 import ENDPOINTS, ROOT, sha, write
from next_session_expectation_contract_v271 import load_registered
from point_context_reconciled_v295 import reconstruct as baseline_reconstruct


def verify_hashes(mapping, substitutions=None):
    for path, wanted in mapping.items():
        if sha((substitutions or {}).get(path, path)) != wanted:
            raise ValueError("protected source changed: " + path)


def audit(archive, baseline, original, tests, old_schedule, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("acceptance outputs are immutable")
    data = reconstruct(archive)
    previous = baseline_reconstruct(baseline)
    registered = load_registered(original)
    if sha(original) != "bfbaeb295637451e396850ee583fcafa3eb0ab6a99fb1b154e4c46a8c23aa40c" or registered["plan"]["target_date"] != "2026-10-08":
        raise ValueError("original forward prediction changed")
    original_codes = {c["code"] for c in registered["plan"]["candidates"]}
    codes = {r["code"] for r in data["observations"]}
    if len(codes) != len(data["observations"]) or len(codes) != 48 or not original_codes.issubset(codes):
        raise ValueError("original stock denominator incomplete")
    old_cutoff = registered["plan"].get("knowledge_cutoff", registered["plan"]["evidence_cutoff"])
    old_view = view_at(archive, old_cutoff)
    if any(r[k]["status"] != "UNKNOWN" for r in old_view["observations"] for k in ENDPOINTS):
        raise ValueError("later attention data backfilled into original cutoff")
    recorded = datetime.now(timezone.utc).isoformat()
    current = view_at(archive, recorded)
    for r in current["observations"]:
        if any(r[k]["status"] != "UNKNOWN" for k in ["forum_heat", "discussion_change", "market_agreement", "bull_bear_extremes"]):
            raise ValueError("provider observations masquerade as forum consensus")
    before = {r["code"]: r["attention"] for r in previous["candidates"]}
    paired = []
    for row in data["observations"]:
        early, late = before[row["code"]], row["latest"]
        if early is None or late is None or early["provider_universe_count"] != late["provider_universe_count"] or early["calculated_at"] > late["calculated_at"]:
            raise ValueError("paired rank source clocks or provider universe inconsistent")
        paired.append({"code": row["code"], "earlier_rank": early["rank"], "later_rank": late["rank"],
            "earlier_calculated_at": early["calculated_at"], "later_calculated_at": late["calculated_at"],
            "earlier_available_at": early["available_at"], "later_available_at": late["available_at"],
            "rank_change_later_minus_earlier": late["rank"]-early["rank"]})
    profiles = [x for r in data["observations"] if r["profiles"] for x in r["profiles"]["rows"]]
    if any(r["verified_user_count"] is not None for r in profiles):
        raise ValueError("unverified user count promoted to a fact")
    manifest = json.loads((Path(archive)/"manifest.json").read_text(encoding="utf-8"))
    parent = Path(manifest["parent"])
    initial = reconstruct(parent)
    parent_manifest = json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
    initial_errors = [{"code": r["code"], "errors": r["errors"]} for r in initial["observations"] if r["errors"]]
    if len(parent_manifest["entries"]) != 240 or len(manifest["entries"]) != 1 or len(initial_errors) != 1:
        raise ValueError("initial source failure or one-attempt supplement disappeared")
    test_record = json.loads(Path(tests).read_text(encoding="utf-8"))
    verify_hashes(test_record["source_hashes"])
    if not test_record["passed"] or test_record["errors"] or test_record["failures"]:
        raise ValueError("actual related checks not passed")
    forum_probe = ROOT/"data/v299/forum_source_probe_20261002_v1/report.json"
    probe = json.loads(forum_probe.read_text(encoding="utf-8"))
    verify_hashes(probe["source_hashes"])
    for entry in probe["requests"]:
        if sha(entry["raw_path"]) != entry["raw_sha256"] or entry.get("http_status") != 403 or entry["status"] != "UNAVAILABLE":
            raise ValueError("actual forum rejection evidence changed")
    app_path = Path.home()/".codex/automations/automation/automation.toml"
    old_acceptance = json.loads((ROOT/"results/v298/hypothesis_lifecycle_acceptance_20261002_v1/report.json").read_text(encoding="utf-8"))
    verify_hashes(old_acceptance["source_hashes"], {str(app_path): str(Path(old_schedule))})
    app = tomllib.loads(app_path.read_text(encoding="utf-8")); old_app = tomllib.loads(Path(old_schedule).read_text(encoding="utf-8"))
    for key in ["id", "kind", "name", "status", "rrule", "target_thread_id", "created_at"]:
        if app[key] != old_app[key]:
            raise ValueError("existing schedule settings changed")
    for text in ["attention_observations_v299.py collect", "attention_asof_v299.py", "关注度轨迹与共识数据边界V299.md", "hypothesis_bound_daily_runner_v298.py freeze"]:
        if text not in app["prompt"]:
            raise ValueError("new source collection missing from active daily task")
    counts = {"intraday_rows": sum(len(r["intraday"]["rows"]) for r in data["observations"] if r["intraday"]),
        "profile_rows": len(profiles), "keyword_rows": sum(len(r["keywords"]["rows"]) for r in data["observations"] if r["keywords"]),
        "related_rows": sum(len(r["related"]["rows"]) for r in data["observations"] if r["related"]),
        "provider_zero_count_nonzero_share_rows": sum(r["zero_count_with_nonzero_shares"] for r in profiles)}
    output.mkdir(parents=True, exist_ok=False)
    write(output/"old_cutoff_view_NOT_FORECAST.json", old_view)
    write(output/"current_cutoff_view_NOT_FORECAST.json", current)
    write(output/"paired_rank_changes_NOT_PRICE_OR_DISCUSSION.json", {"recorded_at": recorded, "rows": paired,
        "paired_codes": len(paired), "earlier_source": str(Path(baseline).resolve()), "later_source": str(Path(archive).resolve()),
        "uniform_sampling_window": False, "rank_changes_are_discussion_changes": False, "trade_authorized": False})
    lines = ["# 当前关注度观测附页（不属于原预测）", "", "生成时间："+recorded+"；完整新档案形成："+data["recorded_at"]+"。",
        "范围：原9月30日供应商主板按名称非ST涨停池48只，包含原11只候选；非当日行情或官方全市场认证。",
        "旧预测截止前本次五类观测全部不可用，不能回填。", "",
        "| 股票 | 较早排名 | 较晚排名 | 后减前 | 较早计算时间 | 较晚计算时间 |", "|---|---:|---:|---:|---|---|"]
    for r in paired:
        lines.append(f"| {r['code']} | {r['earlier_rank']} | {r['later_rank']} | {r['rank_change_later_minus_earlier']} | {r['earlier_calculated_at']} | {r['later_calculated_at']} |")
    usable = {k: sum(r[k]["status"] != "UNKNOWN" for r in current["observations"]) for k in ENDPOINTS}
    lines += ["", "排名降低代表供应商排名改善；不代表看多、资金流入或交易胜率。两个来源的逐股计算时刻不同，没有统一采样窗口。",
        "当前截止下可用观测数量："+json.dumps(usable, ensure_ascii=False)+"；新鲜度在读取时重检，失效不删原值。",
        "粉丝接口的用户数0与非零占比共存，真实人数保持未知。关键词hitCount分母和关联rate含义未认证。",
        "股吧帖子读取被拒绝；讨论变化、一致性、看多/看空极端化与共识→资金→价格反馈仍未验证。",
        "没有新增真实前向结果、交易授权、概率加分或核心升级；稳定60%未验证。"]
    (output/"review_NOT_FORECAST.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    source_paths = [Path(__file__), ROOT/"attention_observations_v299.py", ROOT/"attention_asof_v299.py", ROOT/"attention_repair_v299.py", ROOT/"attention_archive_dispatch_v299.py",
        ROOT/"关注度轨迹与共识数据边界V299.md", Path(archive)/"manifest.json", Path(archive)/"observations.json", parent/"manifest.json", parent/"observations.json",
        Path(baseline)/"context.json", Path(original), Path(tests), forum_probe, Path(old_schedule), app_path,
        output/"old_cutoff_view_NOT_FORECAST.json", output/"current_cutoff_view_NOT_FORECAST.json", output/"paired_rank_changes_NOT_PRICE_OR_DISCUSSION.json", output/"review_NOT_FORECAST.md"]
    report = {"version": "v299-attention-observation-acceptance-1", "recorded_at": recorded,
        "status": "PRIMARY_RAW_ATTENTION_OBSERVATIONS_TIME_GATES_AND_ORIGINAL_PRESERVATION_VERIFIED",
        "codes": len(codes), "original_candidates_included": len(original_codes), "initial_queries": 240, "supplement_queries": 1,
        **counts, "initial_failures_preserved": initial_errors, "remaining_missing_fields": sum(len(r["errors"]) for r in data["observations"]),
        "old_cutoff_available_fields": 0, "current_cutoff_available_fields": usable,
        "paired_codes": len(paired), "rank_improved": sum(r["rank_change_later_minus_earlier"] < 0 for r in paired),
        "rank_worsened": sum(r["rank_change_later_minus_earlier"] > 0 for r in paired), "rank_unchanged": sum(r["rank_change_later_minus_earlier"] == 0 for r in paired),
        "forum_consensus_verified": False, "historical_original_availability_verified": False, "original_forecast_bytes_unchanged": True,
        "related_tests_observed_passed": test_record["tests_run"], "schedule_active": app["status"] == "ACTIVE", "future_schedule_completed": False,
        "forward_samples_added": 0, "core_model_upgrade": False, "trade_authorized": False, "stable_60_verified": False, "goal_complete": False,
        "source_hashes": {str(p.resolve()): sha(p) for p in source_paths}}
    write(output/"report.json", report)
    return {k: report[k] for k in ["status", "codes", "original_candidates_included", "intraday_rows", "profile_rows", "keyword_rows", "related_rows", "provider_zero_count_nonzero_share_rows", "related_tests_observed_passed", "old_cutoff_available_fields", "stable_60_verified"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "baseline", "original", "tests", "old-schedule", "output"]:
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.baseline, args.original, args.tests, args.old_schedule, args.output), ensure_ascii=True))
