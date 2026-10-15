"""Append actual archived conditions and evidence to every one of81 rows."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import html
import json

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/"lagged_price_relation_20261015T103000Z"


def text(value):
    if isinstance(value,(dict,list)):return json.dumps(b.clean(value),ensure_ascii=False)
    return "未知" if value is None or pd.isna(value) else str(value)


def main():
    before=b.read(ROOT/"execution_start.json")["protected_before"];b.verify(before)
    source=PREVIOUS/"delivery_final/完整81只对照_追加价格联动研究与独立观点.csv"
    original=pd.read_csv(source,dtype={"code":str},low_memory=False)
    artifact=joblib.load(ROOT/"context/review_scope_context.joblib");scope=artifact["rows"]
    added=[x for x in scope if x not in original or x in ["date","code"]]
    table=original.merge(scope[added],on=["date","code"],validate="one_to_one")
    records={};bindings=before|{str(source.resolve()):b.sha(source)}
    registry=b.ROOT/"data/independent_stock_judgement/registry.jsonl"
    bindings[str(registry)]=b.sha(registry)
    for line in registry.read_text(encoding="utf-8-sig").splitlines():
        entry=json.loads(line)
        if b.sha(entry["path"])!=entry["sha256"]:raise ValueError("archived independent record changed")
        obj=b.read(entry["path"]);b.verify(obj["source_bindings"])
        bindings[str(Path(entry["path"]).resolve())]=entry["sha256"]
        for row in obj["rows"]:records[(obj["signal_date"],row["code"])]=(obj,row,entry["path"])
    more=[b.ROOT/"results/daily_review_20261014T1118Z/15日完整15只条件判断.json",
        b.ROOT/"research/model_upgrade_validation/workbench_latest_20261015T091200Z/author_current6_COMPLETED_BEFORE_MODEL_DISPLAY.json"]
    for path in more:
        obj=b.read(path);b.verify(obj.get("source_bindings",{}));bindings[str(path.resolve())]=b.sha(path)
        for row in obj["rows"]:records[(obj["signal_date"],row["code"])]=(obj,row,str(path.resolve()))
    parent_groups={key:part.to_dict("records") for key,part in artifact["parents"].groupby(["date","code"])}
    previous_members=artifact["transitions"]["members"]
    allmembers={key:part.to_dict("records") for key,part in previous_members.groupby(["date","target_height"])}
    additions=[]
    for stock in table.to_dict("records"):
        key=(stock["date"],stock["code"]);record=records.get(key)
        peers=table[table.date.eq(stock["date"])&table.boards.eq(stock["boards"])&table.code.ne(stock["code"])][["code","name","boards"]].to_dict("records")
        previous=allmembers.get((stock["date"],int(stock["boards"])),[])
        parents=parent_groups.get(key,[])
        if record:
            obj,row,path=record
            opening=row.get("opening_confirmation") or {k:row[k] for k in ["09:25","09:35","10:00"] if k in row}
            author=dict(archive_status="ACTUAL_ARCHIVED_RECORD",source_path=path,source_sha256=b.sha(path),
                formed_at=obj.get("formed_at"),review_completed_at=obj.get("reviewed_at"),
                exposure=obj.get("model_exposure"),blinding_independently_verified=False,
                original_direction=row.get("promotion_judgement",row.get("prediction_judgement")),
                role_hypothesis=row.get("role_hypothesis",row.get("role_hypothesis_not_certified")),support=row.get("support"),
                counterevidence=row.get("rebuttal"),cancel_condition=row.get("cancel_condition"),opening_confirmation=opening or None,
                source_references=row.get("fact_references",list(obj.get("source_bindings",{}))),
                forum_opinions=row.get("forum_reasoning_assessment",{}).get("opinions"),
                catalyst_assessment=row.get("catalyst_assessment"),independent_direction_not_autofilled=True)
        else:
            author=dict(archive_status="UNKNOWN_NO_ORIGINAL_INDEPENDENT_RECORD",source_path=None,source_sha256=None,
                formed_at=None,review_completed_at=None,exposure="NO_RECORD_NOT_A_BLINDNESS_CLAIM",blinding_independently_verified=False,
                original_direction=None,role_hypothesis=None,support=None,counterevidence=None,cancel_condition=None,opening_confirmation=None,
                source_references=[],forum_opinions=None,catalyst_assessment=None,independent_direction_not_autofilled=True)
        theme=dict(active_theme_identity="UNKNOWN_NO_COMPLETE_TYPED_THEME_MEMBERSHIP",
            name_diffusion_proxy=stock.get("symbolic_roles"),name_proxy_members=stock.get("symbolic_primary_members"),
            dated_classification_available=stock.get("symbolic_industry_evidence_available"),
            original_author_catalyst=author["catalyst_assessment"],original_forum_opinion_dispositions=author["forum_opinions"],
            not_covered_is_not_no_catalyst=True)
        actual=stock.get("latest_source_outcome")
        if pd.isna(actual):actual=stock.get("source_supplement_target")
        if pd.isna(actual):actual=stock.get("target")
        additions.append(dict(archived_author_record=author,full_current_same_height_competitors=peers,
            complete_prior_cohort_members=previous,all_reported_active_anchor_evidence=parents,theme_evidence_layers=theme,
            outcome_for_display=actual,execution_confirmation_status="UNKNOWN_NO_SAME_TIME_TRIGGER_OR_BROKER_EVIDENCE",
            new_B_direction_review=("PROMOTE" if stock["lagged_price_relation_research_score"]>.5 else "FAIL") if stock["channel_review"]=="B_NONONE_PRICE_RESEARCH_PROXY" else None))
    for column in additions[0]:table[column]=[x[column] for x in additions]
    pd.testing.assert_frame_equal(table[list(original)],original)
    output=ROOT/"workbench";output.mkdir(exist_ok=False)
    raw=table.copy()
    for col in additions[0]:
        if raw[col].map(lambda x:isinstance(x,(dict,list))).any():raw[col]=raw[col].map(text)
    raw.to_csv(output/"完整81只_原判断条件来源与校正口径.csv",index=False,encoding="utf-8-sig")
    b.write_new(output/"完整81只_全部证据与原审阅.json",dict(formed_at=b.now(),rows=table.to_dict("records"),
        old_fields_untouched=True,new_fits=0,old_dates_diagnostic_only=True,current15_target16_pending=True))
    body=[]
    for stock in table.to_dict("records"):
        au=stock["archived_author_record"]
        cols=[stock["date"],stock["code"],stock["name"],stock["boards"],stock["review_cycle_state"],stock["review_primary_role"],stock["original_independent_judgement"],stock.get("additional_15_author_call"),
            stock["BASE"] if pd.notna(stock["BASE"]) else stock.get("current15_after_review_BASE"),stock["lagged_price_relation_research_score"],stock["outcome_for_display"]]
        for i in [8,9]:cols[i]="未知" if pd.isna(cols[i]) else f"{100*cols[i]:.2f}"
        detail=[]
        items=[("当前形态／资格",dict(channel=stock["channel_review"],path=stock["current_price_path_status"],scope="10%价格路径归档代理；交易所资格未认证")),
            ("完整前池与触板分母",{k:stock[k] for k in ["full_cohort_status","full_cohort_previous_count","full_cohort_promoted_count","full_cohort_failed_no_touch_count","full_cohort_unknown_count","full_cohort_promotion_rate","touch_failure_attempt_count","touch_failure_rate"]}),
            ("全部当前同位竞争者",stock["full_current_same_height_competitors"]),("完整前池成员（含未触板失败与未知）",stock["complete_prior_cohort_members"]),
            ("全部已观察活跃父锚与替代解释",stock["all_reported_active_anchor_evidence"]),("题材／行业／名称／股吧分层",stock["theme_evidence_layers"]),
            ("原作者证据、反证、确认、取消与来源",au),("后补15日另一作者条件",dict(direction=stock.get("additional_15_author_call"),cancel=stock.get("additional_15_author_cancel"),support=stock.get("additional_15_author_support"),counterevidence=stock.get("additional_15_author_counterevidence"))),
            ("当前校正角色的适用与取消",dict(support=stock["support"],counterevidence=stock["counterevidence"],cancel=stock["cancel_condition"],scope=stock["active_anchor_discovery_scope"])),
            ("执行与结果资格",dict(confirmation=stock["execution_confirmation_status"],result=stock["outcome_for_display"],qualification=stock.get("latest_source_status"),new_B_direction=stock["new_B_direction_review"],formal_prediction_added=False))]
        for title,value in items:detail.append('<h4>'+html.escape(title)+'</h4><pre>'+html.escape(text(value))+'</pre>')
        row=''.join('<td>'+html.escape(text(v))+'</td>' for v in cols)
        body.append('<tr><td colspan="12"><details><summary><span class="line">'+row+'</span></summary><div class="evidence">'+''.join(detail)+'</div></details></td></tr>')
    # Each stock is one expandable row; all details remain in the document and JSON.
    # Summary is a CSS grid instead of invalid nested table cells.
    body=[x.replace('<td>','<span>').replace('</td>','</span>') if False else x for x in body]
    body=[x.replace('<summary><span class="line"><td>','<summary><span class="line"><span>').replace('</td><td>','</span><span>').replace('</td></span></summary>','</span></span></summary>') for x in body]
    head=["信号日","代码","股票","板数","校正节点","结构候选","原独立记录","15日追加观点","BASE研究分","价格联动研究分","补充结果"]
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整81只研究工作台</title><style>body{font:14px/1.6 system-ui,sans-serif;margin:28px;color:#163047;background:#f4f7fa}h1{font-size:26px}input{padding:10px;width:350px}table{width:100%;border-collapse:collapse;background:white}td{padding:0;border-bottom:1px solid #dde5ec}summary{padding:12px;cursor:pointer}.line,.head{display:grid;grid-template-columns:90px 70px 95px 40px 150px 230px 230px 100px 80px 100px 70px;gap:10px;align-items:start;min-width:1480px}.head{padding:12px;background:#163047;color:white}.evidence{padding:16px 28px;background:#eef3f7}pre{white-space:pre-wrap;word-break:break-all;margin:5px 0 20px}h4{margin-bottom:4px}.scroll{overflow:auto}small{display:block;max-width:1100px}</style><h1>完整81只研究工作台</h1><p>7个信号日，81只次全部保留。展开逐股可见完整前池、竞争者、全部锚、题材分层、原作者条件及来源。</p><small>原冻结字段和方向未改。新语义层不重新训练或改分。名称共现、价格路径和角色候选不是独立身份真值；未覆盖不是无催化。9/30与10/8共21行没有原独立记录，保持未知。分数未校准，固定研究方案失败、未部署。15日六只结果等待16日，实际执行未知。</small><p><input id="q" placeholder="按日期、代码、股票或证据文字筛选"> <span id="n">81 / 81</span></p><div class="scroll"><div class="head">'''+''.join('<span>'+html.escape(x)+'</span>' for x in head)+'''</div><table id="rows"><tbody>'''+''.join(body)+'''</tbody></table></div><script>const q=document.getElementById('q'),rs=[...document.querySelectorAll('#rows tbody tr')];q.oninput=()=>{let n=0;rs.forEach(r=>{const hit=r.textContent.toLowerCase().includes(q.value.toLowerCase());r.hidden=!hit;n+=hit});document.getElementById('n').textContent=n+' / 81'};</script></html>'''
    target=output/"完整81只_可展开证据工作台.html";target.write_text(page,encoding="utf-8")
    assert page.count("<details>")==81 and "NaN" not in page and "�" not in page
    author_count=sum(x["archived_author_record"]["archive_status"]=="ACTUAL_ARCHIVED_RECORD" for x in additions)
    assert author_count==60 and table.outcome_for_display.notna().sum()==74
    manifest=dict(formed_at=b.now(),rows=81,original_all_columns_exact=True,actual_archived_author_conditions=author_count,
        unknown_original_author_records=21,original_review_sources=[str(x) for x in more]+[json.loads(line)["path"] for line in registry.read_text(encoding="utf-8-sig").splitlines()],
        supplementary_outcomes_known=74,supplementary_outcomes_unknown=7,new_formal_forecasts=0,new_author_calls_invented=0,new_fits=0,
        source_bindings=bindings|{str(p.resolve()):b.sha(p) for p in [Path(__file__),ROOT/"context/review_scope_context.joblib",*list(output.glob("*"))]})
    b.write_new(ROOT/"workbench_completion.json",manifest)
    print({k:manifest[k] for k in ["rows","actual_archived_author_conditions","unknown_original_author_records","supplementary_outcomes_known","new_fits"]})


if __name__=="__main__":main()
