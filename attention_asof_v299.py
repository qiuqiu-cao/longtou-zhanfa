"""Cutoff-aware diagnostic view; never a replacement frozen forecast."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

from attention_observations_v299 import ENDPOINTS, ZONE, sha, write
from attention_archive_dispatch_v299 import reconstruct
from daily_research_contract_v292 import stamp


def unknown(reason):
    return {"status": "UNKNOWN", "value": None, "reason": reason}


def project(observations, cutoff):
    cut = stamp(cutoff)
    archive_late = stamp(observations["recorded_at"]) > cut
    result = []
    for row in observations["observations"]:
        view = {"code": row["code"], "forum_heat": unknown("未取得可核验帖子及完整讨论分母"),
            "discussion_change": unknown("排名变化和关键词hitCount不能替代讨论量变化"),
            "market_agreement": unknown("用户构成及关联列表不能证明全市场看多一致性"),
            "bull_bear_extremes": unknown("未取得同口径看多/看空样本及反例")}
        for kind in ENDPOINTS:
            value = row.get(kind)
            reason = "完整档案在知识截止后才形成，不倒填其中的较早计算时间" if archive_late else row.get("errors", {}).get(kind, "来源不可用")
            view[kind] = unknown(reason)
            if archive_late or value is None:
                continue
            if stamp(value["available_at"]) > cut:
                view[kind] = unknown("原始响应实际取得时间晚于截止")
                continue
            if kind == "latest":
                clocks, ttl = [value["calculated_at"]], 1200
            elif kind == "intraday":
                clocks, ttl = [value["rows"][-1]["calculated_at"]], 1200
                if stamp(clocks[0]).astimezone(ZONE).date() != cut.astimezone(ZONE).date():
                    view[kind] = unknown("轨迹属于另一自然日，不能当作截止日的日内变化")
                    continue
            elif kind == "profiles":
                clocks, ttl = [value["latest"]["calculated_at"]], 5400
            else:
                clocks, ttl = [r["calculated_at"] for r in value["rows"]], 5400
            ages = [(cut-stamp(t)).total_seconds() for t in clocks]
            if any(age < 0 or age > ttl for age in ages):
                view[kind] = unknown("计算时间晚于截止或已超过预设新鲜度；原值仍保留在档案中")
                continue
            if kind == "profiles":
                projection = value["latest"]
            elif kind == "intraday":
                projection = {k: value[k] for k in ["first_rank", "last_rank", "rank_improvement_first_minus_last", "window_seconds", "missing_grid_times", "identity_limitation", "same_clock_latest_checked"]}
            else:
                projection = value.get("rows", {k: v for k, v in value.items() if k not in {"raw_path", "raw_sha256"}})
            view[kind] = {"status": "inferred" if kind == "intraday" else "PROVIDER_OBSERVATION",
                "value": projection, "available_at": value["available_at"],
                "raw_path": value["raw_path"], "raw_sha256": value["raw_sha256"],
                "forum_consensus_verified": False, "probability_authorized": False}
        result.append(view)
    return {"version": "v299-cutoff-attention-view-1", "knowledge_cutoff": cutoff,
        "archive_recorded_at": observations["recorded_at"], "scope": observations["scope"], "observations": result,
        "all_original_codes_preserved": True, "forum_consensus_verified": False,
        "historical_rows_are_live_features": False, "original_forecast_modified": False,
        "trade_authorized": False, "probability_adjustment_authorized": False}


def view_at(archive, cutoff):
    if stamp(cutoff) > datetime.now(timezone.utc):
        raise ValueError("future cutoff is not an observed state")
    result = project(reconstruct(archive), cutoff)
    result["source_manifest"] = str((Path(archive)/"manifest.json").resolve())
    result["source_manifest_sha256"] = sha(Path(archive)/"manifest.json")
    result["recorded_at"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = view_at(args.archive, args.cutoff)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(json.dumps({"status": "DIAGNOSTIC_VIEW_NOT_FORECAST", "codes": len(result["observations"]),
        "available_fields": {k: sum(r[k]["status"] != "UNKNOWN" for r in result["observations"]) for k in ENDPOINTS},
        "forum_consensus_verified": False}, ensure_ascii=True))


if __name__ == "__main__":
    main()
