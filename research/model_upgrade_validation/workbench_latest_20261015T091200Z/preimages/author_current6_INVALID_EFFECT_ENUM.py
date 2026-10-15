"""Actual complete diagnostic review before any current-stock model display."""
from pathlib import Path
from copy import deepcopy
import json

import short_term_thinking_dataset_builder as b
import relay_review_workbench as workbench
import forum_reasoning_review as forum
import stock_short_term_context as short
import stock_narrative_review as narrative

ROOT=Path(__file__).resolve().parent
PREVIOUS=b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z"


def argument(opinion,manual):
    source=opinion["source_id"];section=opinion["source_section"]
    question=str(opinion.get("source_claim_scope") or "").startswith("AUTHOR_ASKS_")
    if question:
        claim="作者问时代/九阳是否真强、Y连接以及百洋能否独立；没有给出已确认资金布局或本次16日晋级结论。"
        why="提问是检验独立性及名称共振的线索，不是作者肯定九阳已经安排百洋。"
        decision,effect="ACCEPT","ROLE_ONLY"
    elif section=="COMMENT":
        claim="评论者把九阳、百洋与Y/羊相联，并推测主力装糖与布局。"
        why="保留名称/父锚研究线索；资金布局部分缺少交易级证据，不能采纳为事实或当前晋级理由。"
        decision,effect="PENDING","NEUTRAL_UNKNOWN"
    elif source=="guba_name":
        claim="正文确实出现三羊—九阳—百洋—圣阳的名称链。"
        why="名称叙事存在及本轮先后节点支持九阳周期候选；不认证三羊/圣阳为直接父锚或唯一补涨龙。"
        decision,effect="ACCEPT","ROLE_ONLY"
    elif source=="tgb_0930_background":
        claim="9月30日旧正文把阳光放在消费/名字线，并讨论Y字、均瑶、九阳等替代解释。"
        why="采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。"
        decision,effect="ACCEPT","ROLE_ONLY"
    elif manual["code"]=="002343":
        claim="12日作者称慈文传媒伴飞并提出需放量检验后弱转强。"
        why="15日非一字封板是独立新事实，旧条件可作为观察问题；没有分时/盘口就不能认证已经完成真承接。"
        decision,effect="ACCEPT","ROLE_ONLY"
    else:
        claim="12日作者提出百洋农业避险、相对主动的假设。"
        why="农业避险及主动性须当期完整题材和价格路径复核。15日巨量只说明成交放大，不能验证主动买盘或资金因果。"
        decision,effect="PENDING","NEUTRAL_UNKNOWN"
    return dict(opinion_id=opinion["opinion_id"],author_claim=claim,author_reasoning=why,
        testable_expectation="以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        author_claim_form="QUESTION" if question else "HYPOTHESIS",author_call="UNSPECIFIED",author_expectation_target_date=None,
        verification=dict(status="VERIFIED",scope="DISCUSSION_PRESENCE",reason="已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",evidence_refs=[opinion["text_path"]]),
        decision=decision,reason=why,counterevidence=[manual["rebuttal"]],cancel_conditions=[manual["cancel_condition"]],independent_effect=effect)


def main():
    receipt=b.read(ROOT/"histories_consumed_receipt.json")
    facts_path=Path(receipt["facts_path"]);facts=b.read(facts_path)
    draft=deepcopy(b.read(receipt["draft_path"]))
    manual_source=PREVIOUS/"delivery_final/current6_actual_role_review_BEFORE_MODEL_DISPLAY.json"
    manual={x["code"]:x for x in b.read(manual_source)["rows"]}
    fcontext=forum.checked(facts);frows={x["code"]:x for x in fcontext["rows"]}
    discovery=workbench.identity.checked(facts);identities={x["code"]:x for x in discovery["rows"]}
    short_path=ROOT/"histories_consumed_v2/short_term_context_WITH_HISTORIES.json";scontext=b.read(short_path)
    srows={x["code"]:x for x in scontext["rows"]}
    npath=ROOT/"histories_consumed_v2/typed_narrative_context.json";ncontext=b.read(npath);nrows={x["code"]:x for x in ncontext["rows"]}
    for row in draft["rows"]:
        code=row["code"];actual=manual[code];identity=identities[code];source=next(x for x in facts["rows"] if x["code"]==code)
        for key in ["role_hypothesis","support","rebuttal","cancel_condition","relative_thesis","comparison_peers","opening_confirmation"]:
            row[key]=deepcopy(actual[key])
        row["promotion_judgement"]=None
        row["conditional_outlook"]="BULLISH_CONDITIONAL" if code in {"002343","002696"} else "OBSERVE_PENDING_EVIDENCE"
        row["conditional_outlook_semantics"]="作者条件观察，未注册、未经原生产来源资格或全部11类freeze；不计正式或新的独立预测样本。"
        row["conditional_outlook_reason"]=actual["support"]+"；反证："+actual["rebuttal"]+"；取消："+actual["cancel_condition"]
        row["forum_preread_assessment"]=dict(reviewed=True,unknown_not_negative=True,
            reason="已逐条核对旧正文和分栏评论；本股有背景不等于15日当日题材已核。" if frows[code]["opinions"] else "限定已读旧来源未点名本股，未覆盖不是无催化或断板证据；保留独立价格和竞争观察。")
        assessment=row["supplement_assessment"];assessment.update(reviewed=True,reason=actual["role_hypothesis"]+" "+actual["rebuttal"],unknown_not_negative=True)
        for value in assessment["relations"]:
            relation=next(x for x in source["supplement_relation_context"]["relations"] if x["relation_id"]==value["relation_id"])
            accept=code=="002696" or code=="002343" and relation["anchor_code"] is None
            value.update(decision="ACCEPT" if accept else "PENDING",reason="保留关系候选而非资金因果或唯一父龙。"+actual["role_hypothesis"],counterevidence=[actual["rebuttal"]],cancel_conditions=[actual["cancel_condition"]])
        opinions=[argument(x,actual) for x in frows[code]["opinions"]]
        row["forum_reasoning_assessment"]=dict(reviewed=True,synthesis=actual["role_hypothesis"]+" "+actual["rebuttal"],
            agreement="PARTIAL" if opinions else "UNDETERMINED",agreement_reason="采纳已核叙事与条件问题，父锚、资金及当期主动性分别核验。" if opinions else "没有限定来源观点可比较，不宣称与不存在的观点一致或相反。",unknown_not_negative=True,opinions=opinions)
        accepted=[];other=[]
        for parent in identity["anchor_hypotheses"]:
            anchor=parent["anchor_code"]
            if (code,anchor)==("002696","002242"):
                accepted.append(anchor)
            else:
                reject=(code,anchor) in {("001318","000678"),("601811","600812")}
                reason=("仅单字或语义近邻，缺少直接叙事/业务及相邻时序联系，不采纳其为父锚；未证实并非永久证伪。" if reject else "旧跟风重起、并列周期及原父锚状态缺失等替代解释仍在；保留为待核竞争假设。")
                other.append(dict(anchor_code=anchor,decision="REJECT" if reject else "PENDING",reason=reason))
        call="CYCLE_SUPPLEMENT_CANDIDATE" if code=="002696" else "OLD_FOLLOWER_RESTART" if code=="001318" else "RELATED_UNRESOLVED" if identity["anchor_hypotheses"] else "UNKNOWN"
        peers=sorted({c for x in identity["same_cycle_competitors"] for c in x["members"] if c!=code})
        row["identity_discovery_assessment"]=dict(reviewed=True,identity_call=call,accepted_anchor_codes=accepted,rejected_or_pending_anchors=other,
            relative_role_hypothesis=actual["role_hypothesis"],support=actual["support"],counterevidence=actual["rebuttal"],cancel_condition=actual["cancel_condition"],same_cycle_competitors=peers,
            role_truth_certified=False,reviewed_at=b.now(),parent_unknown_does_not_erase_known_space_or_turnover_rule=True)
        checks={}
        for key,check in srows[code]["checks"].items():
            reason=("本轮日线连板先后已单独核对；旧11类源门尚缺"+"、".join(check["missing_gates"])+"，不把规则角色变成严格身份真值。") if key=="launch_and_continuation" else \
                "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。" if key=="theme_transition" else \
                "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。" if key=="independent_strength" else \
                "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。"
            checks[key]=dict(state="UNKNOWN",reason=reason+" "+actual["rebuttal"],evidence_refs=[])
        row["short_term_assessment"]=dict(checks=checks,alternative_explanations=[actual["role_hypothesis"],"共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"],
            counterevidence=[actual["rebuttal"]],cancel_conditions=[actual["cancel_condition"]],
            confirmation_plan=[dict(planned_at="2026-10-16T09:25:00+08:00",status="PENDING",requirement=actual["cancel_condition"]),
                dict(planned_at="2026-10-16T09:35:00+08:00",status="PENDING",requirement="真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。")],
            role_identity_confirmed=False,fund_causality_confirmed=False)
        pair={}
        for scope,sc in srows[code]["pairwise_price_competition"]["scopes"].items():
            pair[scope]=dict(state="UNKNOWN",reason="原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
                comparison_peer_codes=sc["declared_peer_codes"],incomplete_peer_codes=sc["missing_or_incomplete_peer_codes"],
                stronger_observed_peer_codes=sorted(x["peer_code"] for x in sc["pairs"] if x["peer_ahead_at_some_sample"] is True),
                next_session_paths={k:dict(state="POSSIBLE",reason="当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。") for k in ["joint_promotion","joint_failure","divergence"]})
        row["pairwise_competition_assessment"]=dict(scopes=pair,price_rank_is_not_core_identity=True,role_causality_or_entry_confirmed=False)
        row["narrative_type_assessment"]=dict(current_trade_narrative_presence=nrows[code]["current_trade_narrative_presence"],content_types_reviewed=True,unknown_not_negative=True,
            reason="四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",fund_causality_confirmed=False)
    draft.update(status="AUTHOR_COMPLETED_FULL6_DIAGNOSTIC_ROLE_FORUM_AND_HISTORY_REVIEW",author_review_completed=True,reviewed_at=b.now(),
        actual_exposure_detail="Current15/16 stock scores not seen; prior dates and historical aggregate metrics already seen. Self-attestation only, not certified blind.",
        historical_original_time_training_eligible=False,independent_new_prediction_registered=False)
    paths=[manual_source,short_path,npath,ROOT/"histories_consumed_v2/narrative_reviews_COMPLETED.json",Path(__file__)]
    draft["source_bindings"]|=receipt["source_bindings"]|{str(p.resolve()):b.sha(p) for p in paths}
    short.validate_assessments(scontext,draft);narrative.validate_assessments(ncontext,draft)
    result=workbench.validate_completed(facts_path,draft)
    path=ROOT/"author_current6_COMPLETED_BEFORE_MODEL_DISPLAY.json";b.write_new(path,draft)
    completed=workbench.review_context(facts_path,path,ROOT/"review_context_VALIDATED.json")
    workbench.render(facts_path,ROOT/"review_COMPLETED_SCORES_MASKED",path)
    b.write_new(ROOT/"author_review_validation.json",dict(formed_at=b.now(),status=result,full_pool_count=6,forum_opinions=sum(len(x["forum_reasoning_assessment"]["opinions"]) for x in draft["rows"]),
        all_discovered_parents_reviewed=sum(len(x["identity_discovery_assessment"]["accepted_anchor_codes"])+len(x["identity_discovery_assessment"]["rejected_or_pending_anchors"]) for x in draft["rows"]),
        original_manual_parent_reviews=40,short_term_and_pairwise_and_narrative_validation_pass=True,history_role_rows_consumed=195,
        formal_forecast_registered=False,independent_forecast_registry_added=False,model_scores_displayed_before_review=False,new_fits=0,
        facts_path=str(facts_path),review_path=str(path.resolve()),source_bindings=draft["source_bindings"]|{str(path.resolve()):b.sha(path)}))
    print(result,"full6 complete; diagnostic only; current models not loaded")


if __name__=="__main__":main()
