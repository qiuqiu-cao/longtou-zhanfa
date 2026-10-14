"""Recount current forecast evidence without fitting or changing a forecast.

Historical CSV outcomes remain diagnostic. A passing process test, old proxy,
one-word promotion or a mostly-negative classifier cannot prove the full goal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from next_session_expectation_contract_v271 import load_registered, now, stamp

ROOT = Path(__file__).resolve().parent
MARKET = ROOT / "results/v274/monthly_refit_20261002_v1"
RELAY = ROOT / "results/v279/daily_path_paired_20261002_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def proportion(k, n):
    if not n:
        return dict(successes=0, n=0, accuracy=None, wilson95=None)
    p, z = k / n, 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return dict(successes=int(k), n=int(n), accuracy=p, wilson95=[center-half, center+half])


def picked(frame):
    # Same announced rank/tie-break, independently implemented from the CSV.
    return frame.sort_values(["date", "probability", "boards", "code"],
        ascending=[True, False, False, True], kind="stable").drop_duplicates("date")


def relay_metrics(frame):
    full_dates = frame.groupby("date").label_valid.all()
    complete = frame[frame.date.isin(full_dates[full_dates].index)].copy()
    if not complete.target.isin([0, 1]).all():
        raise ValueError("complete relay dates contain missing/nonbinary labels")
    nonword = complete[complete.daily_true_one_word.eq(0)].copy()
    top = picked(nonword)
    positive = nonword[nonword.predicted.eq(1)]
    successes = int(positive.target.sum())
    return dict(total_dates=int(frame.date.nunique()), complete_dates=int(complete.date.nunique()),
        unavailable_dates=sorted(full_dates[~full_dates].index.tolist()),
        nonword_rows=len(nonword), actual_promotions=int(nonword.target.sum()),
        unfiltered_rank_top1=proportion(int(top.target.sum()), len(top)),
        positive_precision=proportion(successes, len(positive)),
        positive_recall=successes/int(nonword.target.sum()) if nonword.target.sum() else None,
        binary_accuracy=proportion(int(nonword.predicted.eq(nonword.target).sum()),len(nonword)),
        all_failure_binary_baseline=1-float(nonword.target.mean()) if len(nonword) else None,
        top1_is_a_ranking_metric_not_an_automatically_positive_call=True)


def run(output):
    target = Path(output)
    if target.exists():
        raise FileExistsError("refusing to overwrite goal evidence audit")
    market_report, relay_report = read(MARKET / "report.json"), read(RELAY / "report.json")
    checked = {}
    for report in (market_report, relay_report):
        for path, wanted in report["source_hashes"].items():
            actual = sha(ROOT / path)
            if actual != wanted:
                raise ValueError("research source hash mismatch: " + path)
            checked[path] = actual
    market = pd.read_csv(MARKET / "market_predictions_LABEL_ONLY.csv", low_memory=False)
    relay = pd.read_csv(RELAY / "CONTROL_predictions_LABEL_ONLY.csv", dtype={"code":str}, low_memory=False)
    if market.date.duplicated().any() or relay.duplicated(["date", "code"]).any():
        raise ValueError("duplicate date/candidate in performance evidence")
    market = market[market.label_valid.eq(True) & market.target.isin(["IMPROVE", "CONTINUE", "WORSEN"])]
    market_rate = proportion(int(market.predicted.eq(market.target).sum()), len(market))
    if market_rate["successes"] != market_report["market"]["combined"]["all"]["correct"]:
        raise ValueError("market recount differs from stored report")
    classes = {}
    for label in ("IMPROVE", "CONTINUE", "WORSEN"):
        truth, called = market.target.eq(label), market.predicted.eq(label)
        correct = int((truth & called).sum())
        classes[label] = dict(actual=int(truth.sum()), called=int(called.sum()), correct=correct,
            recall=correct/int(truth.sum()) if truth.any() else None)
    relay_by_period = {}
    for fold in ("2025H2", "2026YTD", "SEPTEMBER"):
        metrics = relay_metrics(relay[relay.fold.eq(fold)])
        original = relay_report["arms"]["CONTROL"][fold]["ohlc_non_one_word"]
        if any(metrics["unfiltered_rank_top1"][key] != original[key] for key in ("n", "successes")):
            raise ValueError("true nonword recount differs from stored report")
        relay_by_period[fold] = metrics
    relay_by_period["combined_pre_september"] = relay_metrics(relay[relay.fold.ne("SEPTEMBER")])
    registrations = []
    current = now()
    for path in sorted((ROOT / "data/v271/expectation_registry").glob("*.json")):
        frozen = load_registered(path)
        close = stamp(frozen["plan"]["target_date"] + "T15:00:00+08:00")
        registrations.append(dict(path=str(path), sha256=sha(path), registration_hash=frozen["sha256"],
            forecast_status=frozen["forecast_status"], target_date=frozen["plan"]["target_date"],
            exact_target_close_has_occurred=current >= close))
    # Inspection is of this explicit contract registry, not every old project's
    # forward registry. Future dates cannot be interpreted as settled outcomes.
    requirements = [
        dict(requirement="稳定复盘流程", status="IMPLEMENTED_WITH_PROCESS_EVIDENCE",
             evidence="V284 daily output and 151 process tests; these do not measure accuracy"),
        dict(requirement="下一交易日市场预期稳定超过60%", status="UNPROVEN",
             evidence="Historical 180/285; reused dates, interval below 60%, weak CONTINUE recall, no completed current forward date"),
        dict(requirement="连板下一板预测稳定超过60%", status="NOT_ACHIEVED",
             evidence="True nonword ranking and positive-call precision are below 60%; word-inclusive promotion is separate"),
        dict(requirement="市场行情和短线情绪预期", status="IMPLEMENTED_FORECAST_AWAITING_OUTCOME",
             evidence="V280 original market scenario and direction; indices remain separate targets"),
        dict(requirement="超预期、符合预期、不及预期及预案", status="CONTRACT_IMPLEMENTED_REAL_CONFIRMATION_PENDING",
             evidence="V271 AUCTION/OPEN0935/OPEN1000/CLOSE; real target-day auction and minute observations unavailable"),
        dict(requirement="龙头身份判断及预测优化", status="RULES_IMPLEMENTED_ADVANTAGE_UNPROVEN",
             evidence="V273 evidence transitions; theme/absorption facts and independent leader-identity outcomes missing"),
        dict(requirement="股吧经验与A股玄学", status="RESEARCHED_ADVANTAGE_UNPROVEN",
             evidence="V273 original-source hypothesis bank; symbol attention is not verified industry evidence or predictive uplift"),
    ]
    sources = [MARKET / "report.json", MARKET / "market_predictions_LABEL_ONLY.csv",
        RELAY / "report.json", RELAY / "CONTROL_predictions_LABEL_ONLY.csv",
        ROOT / "results/v284/integration_acceptance_20261002_v1/report.json",
        ROOT / "leader_identity_evidence_v273.py", ROOT / "股吧经验与玄学研究V273.md",
        ROOT / "next_session_expectation_contract_v271.py", Path(__file__)]
    result = dict(version="v285-objective-evidence-recount-1", recorded_at=current.isoformat(),
        status="FULL_GOAL_NOT_ACHIEVED", goal_verified=False, accuracy_claim_allowed=False,
        unchanged_models=True, unchanged_frozen_forecasts=True, market_direction=market_rate,
        market_by_class=classes, true_nonword_control=relay_by_period,
        current_contract_registrations=registrations, requirements=requirements,
        historical_evidence_is_diagnostic=True, source_hashes={str(p):sha(p) for p in sources},
        verified_research_source_hashes=checked)
    target.mkdir(parents=True, exist_ok=False)
    (target / "report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    lines = ["# 隔日复盘目标证据核对 V285", "", f"核对时间：{current.astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}。",
        "完整目标未完成。本次从逐条预测重算，未拟合模型或修改冻结答案。", "",
        "| 对照时期 | 真实非一字排名首选晋级 | 正向判断精确率 | 实际晋级召回 | 完整日期/全部日期 |",
        "|---|---:|---:|---:|---:|"]
    for fold, metrics in relay_by_period.items():
        top, precision = metrics["unfiltered_rank_top1"], metrics["positive_precision"]
        lines.append(f"| {fold} | {top['successes']}/{top['n']}（{top['accuracy']:.2%}） | "
            f"{precision['successes']}/{precision['n']}（{precision['accuracy']:.2%}） | "
            f"{metrics['positive_recall']:.2%} | {metrics['complete_dates']}/{metrics['total_dates']} |")
    lines += ["", "排名首选指标逐日强制选最高研究分数，可能仍被二分类判失败；不等于每天都有正向首选或可交易主选。",
        "128/285为V279按真实非一字验证重新选择月度模型的对照；138/285为V274旧一字代理选择模型的结果。模型选择口径也不同，不能将差异全部归因于一字定义。", "",
        f"市场方向：{market_rate['successes']}/{market_rate['n']}（{market_rate['accuracy']:.2%}）；"
        f"延续类识别{classes['CONTINUE']['correct']}/{classes['CONTINUE']['actual']}。", "",
        "| 用户目标 | 当前核对结果 | 证据及缺口 |", "|---|---|---|"]
    lines += [f"| {r['requirement']} | {r['status']} | {r['evidence']} |" for r in requirements]
    lines += ["", "当前V271契约登记逐份核验内容哈希。10月8日尚未发生，不能产生其真实前向结算证据。",
        "下一步需要真实的新交易日、完整候选与按原规则结算的独立记录；反复调整已查看的同一段历史不补足这项证据。"]
    (target / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"status":result["status"], "market":market_rate,
            "true_nonword":relay_by_period["combined_pre_september"], "registrations":registrations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=True))
