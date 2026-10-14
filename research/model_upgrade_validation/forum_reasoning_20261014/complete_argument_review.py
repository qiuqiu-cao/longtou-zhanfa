"""Review all saved arguments, without rewriting an old direction or clock."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import stock_judgement_review as o
import forum_reasoning_review as f

OUT = Path(__file__).resolve().parent
ACTUAL = OUT/"actual_full15"
facts_path = ACTUAL/"facts_with_forum_reasoning.json"
facts = o.read(facts_path)
context = f.checked(facts)
draft = o.read(ACTUAL/"author_draft_PENDING.json")

# These are today's interpretations of already checked historical text.
# No author is assigned an explicit Oct15 forecast that they never made.
interpretations = {
 "opinion_615005c65e583807a9d6975a": (
  "作者把三羊、九阳、百洋、圣阳串成名称链，并用调侃语气评价。",
  "以名称与高低位情绪连接寻找交易叙事；调侃不等于作者看多。",
  "核查链条启动节点与完整同位表现，持续弱反馈时削弱情绪补涨解释。",
  "HYPOTHESIS","VERIFIED","DISCUSSION_PRESENCE","该名称链确在已核原文出现；仅确认讨论存在。",
  "ACCEPT","纳入百洋角色候选分析，同时保留农业/鱼粉独立解释。","ROLE_ONLY",
  ["名称链未证明资金迁移、唯一父锚或次日晋级。","当日正文覆盖缺失；三羊/圣阳节点未核。"],
  ["完整名称关联池持续负反馈或独立行业解释更充分时，降低该关系的判断作用。"]),
 "opinion_3881102d5c2dfd4f09cbdd21": (
  "作者认为农业板块负反馈下仍走强可体现主动性，并提醒弱者隔日失败。",
  "把个股对同方向弱势的相对抵抗作为区分主动/被动的假设。",
  "在完整同题材和同位中核对：是否板块走弱而本股延续；若一起转弱，主动性解释不成立。",
  "HYPOTHESIS","UNKNOWN","ROLE_HYPOTHESIS","原文逻辑存在，但资金主动性未认证；本批仅有可核收盘路径，缺原同刻完整资金/分时。",
  "ACCEPT","采纳相对表现检查，可在核验相对延续后给晋级倾向提供支持；不能只因板块弱就自动看多。","SUPPORT_PROMOTE",
  ["板块整体风险也可能最终拖累个股；逆势收盘可能受一字或供给约束影响。","同位其他候选可能更强，不能只挑成功样本解释主动性。"],
  ["与同方向股票一起走弱或失去同位相对延续时，取消这条正向支持。"]),
 "opinion_afb25256f83d606e24a62a4c": (
  "作者询问时代/九阳是否装糖、七板意义及百洋/远程Y线能否独立。",
  "用问题引导检查高标对抗、名称联系和低位独立性，没有给出明确答案。",
  "分别核对高标分歧与百洋是否持续相对独立，而非把问题当作肯定晋级。",
  "QUESTION","UNKNOWN","ROLE_HYPOTHESIS","这是已核作者问句，资金意图和独立性仍未知。",
  "PENDING","作为待核问题保留，不给作者虚构明确预测。","UNCERTAIN",
  ["高标与低位可以共同晋级、共同失败或分化；问句无法区分这些结果。"],
  ["只有完整实际路径与竞争核验后才更新该问题的答案。"]),
 "opinion_5a91f9088710f3769a037d7d": (
  "评论者认为时代/九阳存在装糖、七板意义有限，并把百洋Y线联系九阳布局。",
  "由高标分时对抗与名称符号推断主力意图和低位连接。",
  "把高标分歧反馈、低位延续和名称联系分别核验，避免由符号直接认定资金从父锚流入。",
  "HYPOTHESIS","UNKNOWN","FUNDS_CAUSALITY","评论归属与自身时间已核；主力意图和资金迁移无独立证明。",
  "PENDING","保留与原作者问句不同的评论观点，不把评论继承为作者结论。","UNCERTAIN",
  ["评论可能事后解释；相同名称也可能有独立催化。"],
  ["与原行情路径矛盾或独立解释更充分时否决资金归因。"]),
 "opinion_5e39671456abb7359dd06f49": (
  "作者称慈文传媒伴飞、封单有一定强度，要求爆量检验后弱转强确认。",
  "把静态强势与经过真实分歧后的确认分开，预期是有条件的。",
  "实际发生换手分歧后检查回封和完整同位相对表现；放量本身不等于承接完成。",
  "HYPOTHESIS","UNKNOWN","MARKET_OBSERVATION","原文条件存在，原同刻封单强度和爆量检验序列未独立核实。",
  "ACCEPT","纳入后续预期与取消条件；未发生检验时不转为无条件晋级判断。","NEUTRAL",
  ["放量可能是兑现，封单也可能撤单；一字阶段不能确认打开后的承接。","其他传媒同位可一起晋级或分化。"],
  ["发生分歧后不能回封、相对同位走弱或传媒反馈恶化时，取消弱转强预期。"]),
 "opinion_ff0ac3c2b547de1d6f4990ba": (
  "作者根据当日硬度把新经典称为传媒新补涨龙。",
  "由当日表现提出角色身份假设，而非已核业务或因果事实。",
  "核具体父锚、启动节点、传媒扩散及完整同位反馈，检验该角色能否持续。",
  "HYPOTHESIS","UNKNOWN","ROLE_HYPOTHESIS","作者角色称谓确有原文，具体父锚和独立角色真值未知。",
  "PENDING","保留角色候选线索，不能把作者称龙直接转成必晋级。","ROLE_ONLY",
  ["一天硬度不证明带动能力；同位慈文与百洋仍是完整比较对象。"],
  ["缺题材扩散或同位相对延续时削弱补涨核心假设。"]),
 "opinion_24737d19b06598046ac385ef": (
  "作者讨论化工及善水跟随新华，倾向先按兵不动，并认为化工主板经常输。",
  "从跟随性和行业历史印象提出谨慎预期；善水是直接关系主体。",
  "逐股比较化工与其他同位，核真实跟随关系及扩散；行业谨慎不等于醋化必失败。",
  "OPINION","UNKNOWN","MARKET_OBSERVATION","行业经常输没有完整分母佐证；善水—新华不能自动转写为醋化—新华。",
  "ACCEPT","采纳对后排/跟随风险的检查，否决用行业印象直接判醋化断板。","NEUTRAL",
  ["醋化可能有独立化工催化或相对延续；同段点名不等于同一父子关系。"],
  ["完整个股核验显示独立强势时，取消由行业印象导出的弱预期。"]),
 "opinion_83bf69c0ffa8a102e1b5769c": (
  "9月30日消费段点名阳光，提九阳/均瑶身位优势，也提示缩量承接不明、后排谨慎。",
  "高标发酵可能带动低位，但高标承接不足也可能削弱后排。",
  "把父锚反馈与阳光当前重新起板路径共同核验；不能把旧跟风经历当作当前无条件强势。",
  "HYPOTHESIS","UNKNOWN","ROLE_HYPOTHESIS","仅为9月30日背景；14日当前关系正文未取得，资金因果仍未知。",
  "ACCEPT","同时采纳补涨连接与后排风险，用于阳光角色和确认条件。","ROLE_ONLY",
  ["均瑶是替代锚，父子身份不唯一；旧帖不证明当前作者仍持相同预期。"],
  ["父锚/消费反馈持续弱化且阳光失去完整二板相对表现时，削弱补涨预期。"]),
 "opinion_c411f771d7bcb085bfb42037": (
  "9月30日化工段用亚/业Y等拼音联系九阳、均瑶，并提出先观察后确认。",
  "名称符号是作者的分组线索，但该段直接主体是其他化工股。",
  "核该摘录是否直接点名阳光；不能由其他化工的Y联想替代本股连接证据。",
  "OPINION","VERIFIED","DISCUSSION_PRESENCE","原文存在该Y讨论，但本段没有直接点名阳光。",
  "REJECT","否决把这段当作阳光直接父锚或晋级依据；保留符号背景范围。","NEUTRAL",
  ["同一字母可覆盖大量无关股票，不能单独区分成功和失败。"],
  ["取得本股直接关联来源后另行追加，不从当前未覆盖转正。"]),
 "opinion_983be17090052372fbb8bbfa": (
  "9月30日分别讨论均瑶Y配套、九阳身位及新华相关分支的竞争。",
  "用多个锚、身位与分支竞争解释强弱，单一父锚未必足够。",
  "给阳光保留九阳/均瑶替代解释，核当前节点后再决定其作用。",
  "HYPOTHESIS","UNKNOWN","ROLE_HYPOTHESIS","仅有旧分组背景，均瑶本批同口径市场时序未知。",
  "ACCEPT","采纳多父锚和完整竞争检查，保留当前背景未知。","ROLE_ONLY",
  ["摘录主要讨论其他股票，不能直接认证阳光的当前唯一父锚。"],
  ["直接本股来源和时序出现相反证据时否决对应关系，不机械保留全部父锚。"]),
}

bycode = {r["code"]:r for r in context["rows"]}
assert sum(len(r["opinions"]) for r in context["rows"]) == len(interpretations) == 10
for row in draft["rows"]:
    a = row["forum_reasoning_assessment"]
    covered = bool(a["opinions"])
    a.update(reviewed=True,agreement="PARTIAL" if covered else "UNDETERMINED",
        agreement_reason="采纳可检验的逻辑与条件；作者角色和资金断言仍由独立证据判断。" if covered else
        "限定原文未覆盖本股，不能声称与不存在的作者观点一致或相反。",
        synthesis="结合完整同位、父锚时序与不同解释逐观点审阅；本次仅追加已见旧讨论的逻辑，不生成或改写原方向。" if covered else
        "未覆盖保持未知。新日期仍可根据行情、事件及完整竞争形成独立方向，不把论坛未点名当没有催化。")
    source_by_id = {p["opinion_id"]:p for p in bycode[row["code"]]["opinions"]}
    for opinion in a["opinions"]:
        claim,logic,expected,form,status,scope,verification,decision,reason,effect,against,cancel = interpretations[opinion["opinion_id"]]
        source = source_by_id[opinion["opinion_id"]]
        opinion.update(author_claim=claim,author_reasoning=logic,testable_expectation=expected,
            author_claim_form=form,author_call="UNSPECIFIED",author_expectation_target_date=None,
            verification={"status":status,"scope":scope,"reason":verification,"evidence_refs":[source["text_path"]]},
            decision=decision,reason=reason,counterevidence=against,independent_effect=effect,cancel_conditions=cancel)
draft.update(author_review_completed=True,status="AUTHOR_COMPLETED_DIAGNOSTIC_FORUM_REASONING_REVIEW")
f.validate(context,draft,facts["source_bindings"])
draft_path = ACTUAL/"author_argument_completed.json"
o.write_new(draft_path,draft)
completed = f.review_context(facts_path,draft_path,ACTUAL/"completed_argument_review.json")
assert all(r["promotion_judgement"] is None for r in completed["rows"])

lines = ["# 股吧逐观点推理与独立预期", "", "完整15只、10去重观点；已见旧模型与结果，仅追加观点审阅，不倒填新预测。", "",
    "| 股票 | 作者观点与逻辑 | 独立采纳及预期影响 | 反证与取消条件 |", "|---|---|---|---|"]
for row in completed["rows"]:
    assessment = row["forum_reasoning_assessment"]
    if not assessment["opinions"]:
        lines.append(f"| {row['name']} {row['code']} | 限定来源未覆盖 | 仍可依据独立行情/事件判断；本次未新增方向 | 未覆盖不看空或转正 |")
    for p in assessment["opinions"]:
        texts = [row["name"]+" "+row["code"],p["author_claim"]+"；"+p["author_reasoning"],p["decision"]+" / "+p["independent_effect"]+"："+p["reason"]+"；可检验预期："+p["testable_expectation"],"反证："+"；".join(p["counterevidence"])+"；取消："+"；".join(p["cancel_conditions"])]
        lines.append("| "+" | ".join(t.replace("|","/").replace("\n"," ") for t in texts)+" |")
(OUT/"股吧逐观点推理与独立预期.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
o.write_new(OUT/"execution_receipt.json",{"completed_at":o.now().isoformat(),"full_pool":15,"deduplicated_opinions":10,
    "checked_current_day_discussion_count":0,"uncovered_stocks":10,
    "new_forecasts":0,"new_fits":0,"new_formal_forward_samples":0,
    "old_directions_changed":False,"author_explicit_oct15_calls_created":False,
    "source_bindings":facts["source_bindings"] | o.bind(__file__) | o.bind(draft_path) | o.bind(ACTUAL/"completed_argument_review.json")})
print({"reviewed_full_pool":15,"reviewed_arguments":10,"new_forecasts":0,"new_fits":0})
