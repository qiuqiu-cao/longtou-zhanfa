"""Eleven-section report from the original forecast, never a replacement plan."""
from __future__ import annotations

from daily_research_contract_v292 import ANCHORS, EVENTS, METRICS, VERSION

NAMES = {"RISK_OFF": "偏弱/风险释放（旧三分类）", "RISK_ON": "偏强（旧三分类）", "REPAIR": "修复（旧三分类）",
    "WORSEN": "恶化", "IMPROVE": "改善", "CONTINUE": "延续", "PROMOTE": "晋级", "FAIL": "断板", "UNCERTAIN": "未判断",
    "promotion": "晋级", "break": "断板", "reseal": "炸板回封", "high_open_fade": "高开冲高回落",
    "weak_to_strong": "弱转强", "strong_to_weak": "强转弱", "rebound": "反包",
    "index_return": "指数涨跌幅", "turnover_amount": "成交额", "limit_up_count": "涨停数", "limit_down_count": "跌停数",
    "failed_rate": "炸板率", "max_boards": "最高板", "promotion_rate": "晋级率", "previous_limit_premium": "昨日涨停溢价",
    "high_stock_feedback": "高位反馈", "theme_strength": "题材强度",
    "emotion": "情绪", "highest": "最高板", "main_leader": "主线龙头", "main_middle": "主线中军", "middle_risk": "中位风险", "new_cycle": "低位新周期"}


def text(value):
    if value is None or value == []:
        return "无法确认 / 原记录未提供"
    if isinstance(value, list):
        return "；".join(text(x) for x in value)
    if isinstance(value, dict):
        if value.get("status") == "UNKNOWN":
            return "无法确认："+value["reason"]
        if "value" in value:
            return text(value["value"])+"（"+str(value.get("status", ""))+"）"
        return "；".join(str(k)+"="+text(v) for k, v in value.items())
    return str(value).replace("|", "\\|").replace("\n", " ")


def probability(item):
    if not item or item.get("probability") is None:
        return "未校准，留空"
    lo, hi = item.get("probability_interval", [item["probability"], item["probability"]])
    return f"{lo:.1%}—{hi:.1%}（点值{item['probability']:.1%}）"


def rate(value):
    return "无法计算" if value is None else f"{value:.2%}"


def render(registered, run_report, settlement=None):
    p = registered["plan"]
    ext = p.get("daily_research")
    market = p["market"]
    lines = [f"# {p['signal_date']} 隔日研究日报 → {p['target_date']}", "",
        f"生成视图：{run_report['recorded_at']}；原预测登记：{registered['recorded_at']}。",
        f"行情截止：{p['evidence_cutoff']}；知识截止：{p.get('knowledge_cutoff', p['evidence_cutoff'])}。",
        f"原件哈希：`{registered['sha256']}`；状态：`{registered['forecast_status']}`。",
        "此日报是原件的展示与验证视图，视图生成时间不能代替原预测时间。", "",
        "旧三分类、短线生态方向、细分情绪周期与指数驱动分别评价。" if not ext else "阶段与驱动分别记录；回封、弱转强等事件可以重叠，概率不相加。", "",
        "【一、今日市场复盘】", ""]
    if ext:
        lines.append("情绪/周期："+text(ext.get("today", {})))
    else:
        lines.append("旧原件支持："+text(market.get("supporting_evidence")))
        lines.append("旧原件反证："+text(market.get("opposing_evidence")))
        lines.append("完整今日八项指标未存入原预测，不能用后补材料倒填。")
    lines += ["", "【二、明日市场基础预期】", "",
        "市场："+NAMES.get(market.get("primary"), text(market.get("primary")))+"；短线生态："+NAMES.get(market.get("direction"), text(market.get("direction"))),
        "市场概率："+probability(market)]
    if ext:
        for key, label in [("market_phase", "阶段"), ("emotion", "情绪"), ("driver", "驱动")]:
            lines.append(label+"："+text(ext[key].get("call"))+"；"+probability(ext[key]))
    else:
        lines.append("细分阶段、情绪标签、指数主导/情绪主导：原记录未预测。")
    lines += ["", "| 指标 | 昨日基准 | 次日范围 | 单位与范围 |", "|---|---|---|---|"]
    for metric in METRICS:
        item = ext["metrics"][metric] if ext else None
        band = item.get("band") if item else None
        lines.append("| "+NAMES[metric]+" | "+text(item.get("baseline") if item else None)+" | "+
            (f"{band['low']}～{band['high']}" if band else "原件未提供 / 无法确认")+" | "+
            (text(band["unit"])+" / "+text(band["scope"]) if band else "无法确认")+" |")
    if not ext:
        lines.append("旧原件另有固定质量变化区间（部分基准），保留其原范围，不当成完整市场指标。")
    lines += ["", "【三、超预期条件】", ""]
    lines.append(text(ext["plans"]["A"]["conditions"]) if ext else "原件市场推翻条件："+text(market.get("invalidation")))
    lines += ["", "【四、不及预期条件】", ""]
    if ext:
        lines.append(text(ext["plans"]["C"]["conditions"]))
    else:
        lines.append("按旧原件各指标的低于预期与硬否决逐项观察；不存在额外事前综合条件。")
    lines.append("跌停数、炸板率按越低越有利比较；边界值属于符合预期。")
    lines += ["", "【五、主线题材】", ""]
    if ext and ext["themes"]:
        for theme in ext["themes"]:
            lines += ["- "+text(theme["theme_id"])+"："+text(theme["forecast"]["call"])+"；"+probability(theme["forecast"]),
                "  梯队关系："+text(theme["relationships"])]
    else:
        lines.append("题材时点归属、完整梯队与次日题材状态预测缺失，无法确认。")
    lines += ["", "【六、连板晋级预测】", "", "原完整候选池全部保留。晋级成绩、真实龙头身份、成交与收益分别验证。", "",
        "| 股票 | 高度 | 原身份 | 晋级判断 | 晋级概率 | 核心依据与风险 |", "|---|---:|---|---|---|---|"]
    for candidate in p["candidates"]:
        item = next((r for r in ext["candidates"] if r["code"] == candidate["code"]), None) if ext else None
        prob = item["events"]["promotion"] if item else None
        lines.append("| "+candidate["code"]+" "+text(candidate.get("name"))+" | "+str(candidate["boards"])+" | "+text(candidate["identity"])+" | "+
            NAMES[candidate["promotion_call"]]+" | "+probability(prob)+" | "+text(candidate.get("identity_evidence"))+"；风险："+text(candidate.get("identity_invalidation"))+" |")
        if item:
            lines.append("\n"+candidate["code"]+" 事件："+"；".join(NAMES[event]+"="+text(item["events"][event]["call"])+" / "+probability(item["events"][event]) for event in EVENTS))
    if not ext:
        lines.append("\n其他五种分时事件没有原预测或校准证据，不能由30分钟K线猜成真实回封。")
    lines += ["", "【七、龙头候选】", ""]
    for i in range(3):
        leader = ext["leaders"][i] if ext else None
        if not leader or not leader.get("code"):
            lines.append(f"第{i+1}候选：无法确认。"+(text(leader.get("reason")) if leader else "原件没有三名独立龙头身份预测。"))
        else:
            lines += [f"第{i+1}候选：{leader['code']} / {leader['role']} / {leader['identity_status']}。",
                "确认："+text(leader["confirm"])+"；失效："+text(leader["invalidate"]),
                "正常："+text(leader["normal"])+"；偏强："+text(leader["strong"])+"；偏弱："+text(leader["weak"]),
                "持有条件："+text(leader["hold"])+"；观察条件："+text(leader["watch"])]
    lines.append("高度与晋级只用于空间锚和晋级检验，不能替代带动性、承接和题材梯队证据。")
    lines += ["", "【八、关键锚定】", ""]
    for key in ANCHORS:
        anchor = ext["anchors"][key] if ext else None
        code = anchor.get("code") if anchor else None
        if not ext and key == "highest":
            code = "、".join(c["code"] for c in p["candidates"] if c["boards"] == max((r["boards"] for r in p["candidates"]), default=0))
        lines.append(NAMES[key]+"："+text(code)+("；观察："+text(anchor.get("observe")) if anchor else "；其他锚定证据未冻结。"))
    lines += ["", "【九、三套预案】", ""]
    if ext:
        for key, label in [("A", "强"), ("B", "正常"), ("C", "弱")]:
            case = ext["plans"][key]
            lines.append(label+"：条件="+text(case["conditions"])+"；观察="+text(case["observe"])+"；动作="+text(case["action"])+"；取消="+text(case["cancel"]))
    else:
        lines.append("旧原件未按A/B/C三套完整结构冻结。保留其原竞价与盘中条件，不补造预案。")
        for band in p.get("bands", []):
            lines.append("- "+text(band["stage"])+" "+text(band["scope"])+" / "+text(band["metric"])+f"：{band['low']}～{band['high']} "+text(band["unit"])+"；依据="+text(band["basis"]))
    lines += ["", "【十、昨日模型验证】", ""]
    if settlement:
        made = [r for r in settlement["rows"] if r["forecast_made"]]
        correct = sum(r["correct"] is True for r in made)
        wrong = sum(r["correct"] is False for r in made)
        lines.append(f"本原件总预测 {len(made)}；正确 {correct}；错误 {wrong}；待验证 {len(made)-correct-wrong}。不同层目标不合并成一个准确率。")
        for error in settlement["errors"]:
            lines.append("- "+error["layer"]+" / "+error["key"]+"：原="+text(error["original"])+"；实际="+text(error["actual"])+"；"+error["explanation"])
    else:
        lines.append("目标交易日尚未结束或结果未核验，正确/错误均待验证。")
    summary = run_report["statistics"]
    if "windows" in summary:
        for window, label in [("20", "最近20交易日"), ("60", "最近60交易日"), ("all", "全部前向记录")]:
            s = summary["windows"][window]
            lines.append(f"\n{label}：前向登记{s['registered_forward_dates']}日；未结算{len(s['unsettled_registration_hashes'])}日；未来待结算{s['future_pending']}日。")
            for layer in ["market_primary", "market_direction", "market_phase", "emotion", "theme", "leader", "promotion"]:
                metrics = s["layers"].get(layer)
                if metrics:
                    lines.append("- "+layer+"：准确率 "+rate(metrics["accuracy"])+f"；评分{metrics['scored']}/{metrics['total_frozen_rows']}；缺结果{metrics['unknown_outcomes']}；未预测{metrics['unmade_forecasts']}。")
                    if "precision" in metrics:
                        lines.append("  Precision="+rate(metrics["precision"])+"；全已知正例Recall="+rate(metrics["all_known_positive_recall"])+"；F1="+rate(metrics["f1"])+"；Brier="+text(metrics["brier"]))
                else:
                    lines.append("- "+layer+"：暂无已完成前向样本，无法计算。")
    else:
        lines.append("统计暂停："+text(summary.get("reason")))
    lines += ["", "【十一、模型修正】", "", "本轮只完善记录与验证，不因单日案例升级核心预测模型。稳定60%仍需真实样本外证据。"]
    if ext:
        lines.append("共识观测："+text(ext["consensus"]))
        lines.append("玄学假设："+text(ext["hypotheses"]))
    else:
        lines.append("论坛共识、热度与玄学回测未包含于原件。后续材料只能追加为研究，不能提高旧预测概率。")
    lines.append("没有实际成交证据时，回撤与盈亏比无法确认；不从晋级结果推算交易胜率。")
    return "\n".join(lines)+"\n"
