"""Actual author scenarios after model exposure; no new model or fake odds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import joblib
import pandas as pd
import short_term_thinking_dataset_builder as b
from relay_multiday_scenarios import EVENTS,evaluate
import preserved_bindings as preservation

ROOT=Path(__file__).resolve().parent
CURRENT=b.ROOT/"results/daily_review_20261015T093600Z"
ENGINE=b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z/attempt_reclassification_correction/corrected_engine_context_THROUGH_OCT15_RESEARCH_NOT_FITTED.joblib"


def horizon(cumulative,path):
    return dict(cumulative_up_call=cumulative,every_day_up_call="NO",consecutive_limit_ups_call="NO",path=path,
        uncertainty="QUALITATIVE_AUTHOR_SCENARIO_NO_VALIDATED_MULTIDAY_HEAD_OR_CALIBRATION")


def main():
    earlier=b.read(CURRENT/"independent_forward_views_BEFORE_MODEL.json");b.verify(earlier["source_bindings"])
    rows={x["code"]:x for x in earlier["rows"]}
    planned=[
        dict(code="002343",rank=1,tier="重点",next="YES",touch="YES",
            rationale="5板中相对百洋振幅更窄，传媒/出版归档仍有2板与首板同伴，保留并列高度股共同晋级路径。AIGC和短剧盈利不作为已证实催化收益。",
            next_path="偏向换手后争取6板；盘中触板容易受到高位兑现影响，只有收盘仍封住才兑现涨停预测。",
            horizons={"2":horizon("YES","16日尝试延续，19日更可能出现分歧；若传媒中位和首板未塌，预计19日收盘仍高于15日，但不预计两天都涨。"),
                "3":horizon("YES","20日前偏向震荡中延续，预计累计仍偏正；需要出版/传媒梯队继续提供同伴，单股孤立加速会推翻判断。"),
                "5":horizon("UNKNOWN","22日前更偏向高位震荡或分歧修复，累计方向不作单边正向判断；不预测五天逐日上涨或五连板。")}),
        dict(code="002696",rank=2,tier="重点",next="YES",touch="YES",
            rationale="保留九阳周期提前启动补涨假设和自身5板辨识度；15日首次一价后大换手仍封住。22倍原生量和10%左右振幅也可能是末段兑现，排序弱于慈文。",
            next_path="偏向分歧后再争6板；预期有触板机会，但若回封失败则收盘涨停假设失败，不把触板当成功。",
            horizons={"2":horizon("YES","16日若完成换手延续，19日即使回落仍可能保住相对15日的累计上涨；不预计连续两个涨停。"),
                "3":horizon("UNKNOWN","20日前高位分歧概率路径更突出，需阳光及低位同周期候选承接；累计方向保留未知，不能从九阳关系直接推出继续涨。"),
                "5":horizon("NO","22日前基础情景偏向高度消耗后的震荡回撤，不押期末仍高于15日；若新主线扩散且放量后持续缩量稳价，此偏弱判断被推翻。")}),
        dict(code="605189",rank=3,tier="重点",next="YES",touch="YES",
            rationale="与新华文轩同为2板，但本轮位置较低，存在已归档公司业绩预告。归母与扣非差额大、未经审计及放量6.8倍均保留反证，不把染织与其他化工股当同业务。",
            next_path="偏向业绩事件带动的2进3；即使盘中触板，也须独立检验收盘是否封住，不能因同位新华走弱直接断言它成功。",
            horizons={"2":horizon("YES","16日若延续3板，19日预计进入分歧，基础情景仍偏累计上涨；需要扣非业绩叙事未被新公告否定。"),
                "3":horizon("YES","20日前偏向事件驱动后震荡偏强，累计仍倾向高于15日；不押逐日上涨，冲高放量滞涨将否定延续路径。"),
                "5":horizon("UNKNOWN","22日前偏向事件脉冲后横盘/回撤，累计方向不作正向承诺；持续主业盈利和完整纺织扩散未核。")}),
        dict(code="001318",rank=4,tier="观察",next="UNKNOWN",touch="UNKNOWN",
            rationale="3板与冰山竞争，缩量和6.33%换手可能锁筹也可能未经抛压检验；九阳相关旧跟风重起与新的补涨身份分开。未达到本批独立偏晋级标准。",
            next_path="先看百洋与本股相对主动性，当前不押4板；不得因模型排序靠前自动改为正向。",
            horizons={"2":horizon("UNKNOWN","19日前可能随百洋分歧修复，但缺乳业首板扩散，累计方向暂不押上。"),"3":horizon("UNKNOWN","20日前按旧周期反复活跃理解，倾向震荡；放量不能独立证明新核心。"),"5":horizon("NO","22日前若仍只有名称链、没有消费梯队，基础情景偏回落；若新主线明确扩散则撤销此偏弱判断。")}),
        dict(code="601811",rank=5,tier="观察",next="UNKNOWN",touch="YES",
            rationale="连续2板但已有14天10板经历，不能当普通低位2进3。日内振幅大、14:49封板，出版同伴提供观察理由，但高位反复兑现风险更大。",
            next_path="可能冲板或反复触板，收盘能否封住保持未知；需要慈文和出版同伴同时确认。",
            horizons={"2":horizon("NO","19日前基础情景偏高位宽幅震荡，不押期末高于15日；若低位出版扩散加强且本股稳住均价则失效。"),"3":horizon("NO","20日前累计不偏多，长期反复涨停后的拥挤和尾盘封板是主要反证。"),"5":horizon("NO","22日前偏回撤或横盘，除非出现可核新事件与持续梯队扩散，不按连续高位主升预测。")}),
        dict(code="000530",rank=6,tier="偏不晋级",next="NO",touch="UNKNOWN",
            rationale="与阳光同为3板，15日41.05%换手和13.61倍原生量显示筹码交换剧烈，相关旧冰雪叙事中的哈空/龙江负反馈仍在。放量既可能承接也可能兑现，保留本股独立活口反例。",
            next_path="基础情景偏3进4失败；盘中若再触板不能自动作为收盘成功，若真实回封及相关分支扩散确认则取消偏弱判断。",
            horizons={"2":horizon("NO","19日前偏分歧回撤，不押累计上涨；只有实际承接与完整相关同伴转强才能推翻。"),"3":horizon("NO","20日前偏震荡下行；不把跌幅小于同伴当作必然主升。"),"5":horizon("NO","22日前基础情景不看期末高于15日，若形成新的合格主线且持续换手延续则另记修正，不回写本次。")})]
    bundle=joblib.load(ENGINE);b.verify(bundle["sources"]);sessions=bundle["sessions"];i=sessions.index("2026-10-15")
    for item in planned:
        old=rows[item["code"]]
        item.update(name=old["name"],boards=old["boards"],next_close_limit_up_call=item.pop("next"),next_intraday_touch_call=item.pop("touch"),
            support=old["support"],counterevidence=old["rebuttal"],cancel_condition=old["cancel_condition"],
            complete_same_height_competitors=[r["code"] for r in earlier["rows"] if r["boards"]==old["boards"] and r["code"]!=item["code"]],
            alternatives="共同晋级、共同失败、相互分化均保留；角色、名称、单项放量不认证资金因果或可买性。",
            probability=None,probability_status="未验证",execution="条件观察；09:25/09:35须有实际证据，仅能在观察之后安排可执行入场，当前无订单或成交")
    urls=["https://pdf.dfcfw.com/pdf/H2_AN202610141830470870_1.pdf","https://pdf.dfcfw.com/pdf/H2_AN202610141830462441_1.pdf","https://pdf.dfcfw.com/pdf/H2_AN202610131830411651_1.pdf"]
    out=ROOT/"current_forecast";out.mkdir(exist_ok=False)
    b.write_new(out/"bounded_public_recheck_failures.json",dict(recorded_at=b.now(),urls=urls,maximum_attempts_per_url=1,
        tool_result="URL is not accessible via web tool for all3; no new readable primary text",qualified_new_public_texts=0,
        existing_bound_issuer_PDFs_and_texts_reused=True,current_additional_announcements_not_certified=True))
    sourcefiles=[Path(__file__),ENGINE,CURRENT/"independent_forward_views_BEFORE_MODEL.json",CURRENT/"prospective_fact_bundle.json",CURRENT/"notice_fulltext/manifest.json",b.ROOT/"AGENTS.md",b.ROOT/"templates/复盘_涨停优先与多日持续上涨.md",ROOT/"workbench/完整81只_全部证据与原审阅.json",b.ROOT/"relay_multiday_scenarios.py"]
    record=dict(formed_at=b.now(),status="PROSPECTIVE_QUALITATIVE_SCENARIOS_AFTER_MODEL_EXPOSURE_NOT_FORMAL_FREEZE",signal_date="2026-10-15",target_date=sessions[i+1],
        horizon_targets={str(h):sessions[i+h] for h in [2,3,5]},rows=planned,top3=[x["code"] for x in planned[:3]],watch=[x["code"] for x in planned[3:5]],
        event_definitions=EVENTS,model_exposure="MODEL_OUTPUT_ALREADY_SEEN",independent_blind_claim=False,author_judgement_formed_now=True,
        evidence_cutoff="Bound15-day closing archive and already saved issuer PDFs; fresh3-URL tool recheck unreadable",
        data_scope="ARCHIVED_ALTERNATIVE_PROVIDER_CURRENT_POOL_NOT_EXCHANGE_FULL_MARKET_OR_LIVE_QUEUE",numeric_probability_calibrated=False,
        original_calls_scores_or_registries_changed=False,new_formal_predictions=0,new_fits=0,new_inferences=0,actual_orders=0,
        source_bindings=preservation.resolve({str(p.resolve()):b.sha(p) for p in sourcefiles}))
    b.write_new(out/"15日收盘_16日优先与2_3_5日情景_已看模型.json",record)
    result=evaluate(record,bundle["bars"],sessions,b.now())
    assert len(result["rows"])==66 and result["new_mature_labels"]==0
    b.write_new(out/"current66_events_PENDING_NOT_ZERO_OR_MATURED.json",result)
    lines=["# 15日收盘：16日优先候选与未来2/3/5日路径","", "相对优先：**慈文传媒、百洋股份、富春染织**；阳光乳业、新华文轩列观察，冰山冷热偏不晋级。概率未验证，不使用模型分数代替涨停概率。", "", "这份情景判断现在形成，已看过本日模型，是有曝光声明的前瞻情景追加，不是盲审或正式freeze，不覆盖先前任何判断。依据项目已归档15日完整6只、首板/炸板背景和已存公司文件；新补核3个原PDF地址均工具不可读，不称已核当前新增公告。", "", "|优先|股票|16日收盘涨停|16日触板区别|2日累计至19日|3日累计至20日|5日累计至22日|", "|---:|---|---|---|---|---|---|"]
    label={"YES":"偏是","NO":"偏否","UNKNOWN":"有理由的未知"}
    for item in planned:
        lines.append(f"|{item['rank']}|{item['name']} {item['code']}（{item['tier']}）|{label[item['next_close_limit_up_call']]}|{label[item['next_intraday_touch_call']]}；触板与封住分开|{label[item['horizons']['2']['cumulative_up_call']]}|{label[item['horizons']['3']['cumulative_up_call']]}|{label[item['horizons']['5']['cumulative_up_call']]}|")
    lines += ["", "累计上涨指窗口末收盘高于15日；逐日上涨要求每天收盘高于前日；连续涨停另算。当前不把任何候选的2/3/5日逐日上涨或连续涨停作为基础情景，不能将单日判断外推成五天都涨。", ""]
    for item in planned:
        lines += [f"## {item['rank']} {item['name']} {item['code']}","",item["rationale"],"",item["next_path"],""]
        for h in [2,3,5]:lines.append(f"- 未来{h}日：{item['horizons'][str(h)]['path']}")
        lines += ["", "主要反证："+item["counterevidence"],"", "取消条件："+item["cancel_condition"],"", "完整同位："+"、".join(item["complete_same_height_competitors"])+"。"+item["alternatives"],"",item["execution"],""]
    lines += ["## 到期评价", "", "16日检验收盘涨停与盘中触板；19、20、22日分别检验累计上涨、逐日上涨、连续涨停。完整6只×11事件=66项目前全部待窗口成熟，未知不填0。`relay_multiday_scenarios.py`已实现同一官方日历评价，记录原情景不改写、不提前成熟。实际成交、账户收益和最大回撤仍未知。"]
    (out/"16日优先候选与2_3_5日独立情景.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    b.write_new(ROOT/"forecast_completion.json",dict(formed_at=b.now(),full_pool=6,top3=record["top3"],watch=record["watch"],events=66,mature=0,
        model_exposure_disclosed=True,probabilities_not_invented=True,new_fits=0,new_formal_predictions=0,
        source_bindings=record["source_bindings"]|{str(p.resolve()):b.sha(p) for p in out.glob("*")}))
    print(dict(top3=record["top3"],events=66,mature=0,fits=0))


if __name__=="__main__":main()
