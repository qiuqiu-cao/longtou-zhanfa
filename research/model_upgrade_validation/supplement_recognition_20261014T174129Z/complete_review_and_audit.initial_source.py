"""Actual full-pool source review, then read-only old score/conflict comparison."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import stock_judgement_review as o
import supplement_relation_review as relation
import supplement_role_review as workflow

OUT = Path(__file__).resolve().parent
ACTUAL = OUT/"actual_full15_final"
facts_path = ACTUAL/"facts_with_supplement.json"
facts = o.read(facts_path)
draft = o.read(ACTUAL/"author_supplement_PENDING.json")
old15_path = ROOT/"results"/"sunshine_jiuyang_relation_20261014T152814Z"/"15日完整15只角色补充输出.json"
old10_path = ROOT/"results"/"deep_review_20261014T115540Z"/"原10只判断与反例.csv"
guard = o.read(OUT/"gate_agent"/"read_only_gate_findings.json")["source_bindings"]
o.check_sources(guard)
unchanged_before = guard | o.bind(old15_path) | o.bind(old10_path)

decisions = {
 "baiyang_jiuyang": ("ACCEPT","采纳九阳周期下的名称/情绪补涨候选假设。名称链原文、作者Y疑问及另列评论均存在，首板时九阳5板，九阳13日断连续板时百洋3板并延续至14日4板。父锚出当前池不应导致关系丢失。",
    ["鱼粉/农业等独立叙事仍可能解释上涨；作者疑问不等于资金因果。","14日当日直接关系正文未核；实际一字不能证明打开后的承接或可成交。"],
    ["名称/消费关联候选持续负反馈且独立行业解释更充分时削弱关系归因。","完整同位及同锚比较失去延续，实际打开后无法维持相对承接时取消强势角色假设。"]),
 "sunshine_jiuyang": ("ACCEPT","采纳低置信度的旧跟风重新起板候选假设。9月30日消费段点名阳光且提九阳/均瑶发酵；本轮13日首板与九阳断板节点重合，14日二板。保留旧经历而不称13日首次启动。",
    ["依据是9月30日分组背景，14日关系正文不可读；未确认本轮唯一父锚。","均瑶是另一个消费/Y背景锚；百洋同线高度领先，不能自动把它当新父龙。","14日真实非一字及已存报告开板不能单独证明吸收完成。"],
    ["完整二板及同锚候选比较持续落后、消费扩散失败时削弱或取消。","实际回封/成交证据与预期不符时取消可交易假设；当前尚未确认可交易。"]),
 "sunshine_junyao_alternative": ("PENDING","保留均瑶替代锚。原消费段和Y背景存在，但本批没有同口径均瑶日线和断板节点，不能定为唯一父龙，也不能排除。",["当前父锚时序未知。"],["补齐同口径父锚节点后再采纳或否决。"]),
 "xinjingdian_sector_role": ("PENDING","作者确称新经典为传媒新补涨龙，采纳为待检验角色线索；原文不足以绑定具体父锚和资金因果，当前完整4板竞争与同题材竞争都保留。",["作者角色称谓没有独立真值。"],["具体父锚、题材扩散和相对反馈缺乏时保持未知。"]),
 "ciwen_sector_role": ("PENDING","原文伴飞观点可作传媒背景，不能据此自动赋予补涨龙身份；具体父锚仍未知，保留与新经典、百洋共同晋级、共同失败或分化。",["伴飞不等于补涨核心；目前具体父锚和启动链资料不全。"],["有直接逐股关系及完整时序证据后再独立判别。"]),
 "cuhua_sector_context": ("REJECT","否决把该化工段转写为醋化—新华的具体补涨父子关系。正文直接连接善水与新华，醋化只是同段点名；化工/乙腈线索继续保留，不因关系未核否定题材或晋级。",["原文主体为善水，不能偷换成醋化。"],["取得醋化自身直接关系和完整节点来源后可另行追加复核。"]),
}

draft["author_review_completed"] = True
draft["status"] = "AUTHOR_COMPLETED_DIAGNOSTIC_SUPPLEMENT_REVIEW"
for row in draft["rows"]:
    fact = next(x for x in facts["rows"] if x["code"] == row["code"])
    covered = bool(fact["supplement_relation_context"]["relations"])
    forum_covered = bool(fact["forum_preread_context"]["forum_evidence"])
    row["forum_preread_assessment"].update(reviewed=True,reason=
        "已核12日原文背景与评论分栏，未当14日当前共识。" if forum_covered else
        "原限定两帖未覆盖本股；不代表没有催化。关系层另核已存9月30日背景，仍不冒称当日正文。" if row["code"] == "001318" else
        "限定已读来源未覆盖本股，题材保持未知；未覆盖不能作为断板理由。")
    a = row["supplement_assessment"]
    a.update(reviewed=True,reason="逐条审阅关系、替代锚与完整竞争，角色和晋级方向分开。" if covered else
        "本批没有本股直接父锚关系证据，补涨身份未知；保留全部同位，不以材料缺失转正或看空。")
    for decision in a["relations"]:
        state,reason,opposition,cancel = decisions[decision["relation_id"]]
        decision.update(decision=state,reason=reason,counterevidence=opposition,cancel_conditions=cancel)
draft_path = ACTUAL/"author_supplement_completed.json"
o.write_new(draft_path,draft)
completed = workflow.review_context(facts_path,draft_path,ACTUAL/"completed_supplement_review.json")

# Only after saving the independent relation assessment expose the old scores.
# The author had already seen them on the original date: this is not blind.
old15 = o.read(old15_path)
with old10_path.open(encoding="utf-8-sig",newline="") as stream:
    old10 = list(csv.DictReader(stream))
bycode = o.unique_rows(facts["rows"])
assert set(bycode) == {x["code"] for x in old15["rows"]} and len(old10) == 10
rows15 = []
for old in old15["rows"]:
    fact = bycode[old["code"]]
    assert old["complete_same_height_peers"] == fact["same_height_peers"]
    rows15.append({k:old[k] for k in ("code","name","boards","raw_score","score_display","rank","predicted_positive","original_independent_direction","actual_one_word_daily","resume_calendar_qualification_gap")})
    rows15[-1].update({"same_height_peers":fact["same_height_peers"],
        "relation_ids":[e["relation_id"] for e in fact["supplement_relation_context"]["relations"]],
        "recognition_states":[e["recognition_state"] for e in fact["supplement_relation_context"]["relations"]],
        "target_label":None,"new_direction":None,"old_model_or_direction_changed":False})
calls,independent,labels = {},{},{}
for row in old10:
    code = row["code"]
    calls[code] = row["original_model_positive"] == "True"
    labels[code] = row["supplemental_promoted"] == "True"
    independent[code] = True if row["original_independent_call"] == "PROMOTE" else False if row["original_independent_call"] == "FAIL" else None
audit = {"formed_at":o.now().isoformat(),"review_saved_before_this_comparison":str((ACTUAL/"completed_supplement_review.json").resolve()),
    "actual_model_exposure":"ALREADY_SEEN_NOT_BLIND","new_fits":0,"new_formal_forward_labels":0,
    "full13_to14_old10":old10,"old10_BASE_metrics":o.metrics(calls,labels),
    "old10_independent_metrics":o.metrics(independent,labels),"full14_to15_old15_with_relation_conflicts":rows15,
    "original_predictions_backfill_allowed":False,"original_role_truth_changed":False,
    "source_bindings":unchanged_before | facts["source_bindings"] | o.bind(draft_path) | o.bind(__file__)}
o.check_sources(unchanged_before)
o.write_new(OUT/"full_pool_conflict_audit.json",audit)
o.write_new(OUT/"initial_failure_receipt.json",{"failure":"First normalization omitted the existing sunshine chronology markdown hash; source guard rejected before relation output or registration.",
    "failed_bank":str((OUT/"qualified_relation_bank.json").resolve()),"partial_forum_only":str((OUT/"actual_full15").resolve()),
    "repair":"Appended missing existing source binding in qualified_relation_bank_v2; no source recollection or fitting.",
    "old_versions":"actual_full15_v2/v3 and exact module source preimages retained; final validates current source.","new_fits":0})
print({"reviewed_pool":len(completed["rows"]),"old10_BASE_accuracy":audit["old10_BASE_metrics"]["Accuracy"],
    "old10_BASE_recall":audit["old10_BASE_metrics"]["complete_positive_Recall"],"old15_scores_unchanged":True,"new_fits":0})
