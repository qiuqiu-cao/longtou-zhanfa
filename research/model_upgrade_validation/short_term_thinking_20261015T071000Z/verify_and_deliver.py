"""Read-only model replay, independent counts, and human-readable delivery.

No fitting. The two post-fit corrections preserve all old outputs: an October
ROLL252 alias is removed, and the already saved October14 failed pool is added.
"""
from pathlib import Path
import ast
import gc
import html
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import short_term_thinking_dataset_builder as b
import short_term_thinking_research as r

RUN=Path(__file__).resolve().parent
OUT=RUN/"delivery"
RAW=RUN/"execution_after_binning_repair"
DATA=RUN/"scope_correction"


def independent_binary(frame):
    f=frame[frame.target.notna()&frame.label_valid.astype(bool)&frame.score.notna()]
    y=f.target.to_numpy(dtype=float)==1
    p=f.score.to_numpy(dtype=float)>.5
    tp=int(np.count_nonzero(y&p));fp=int(np.count_nonzero(~y&p));fn=int(np.count_nonzero(y&~p));tn=int(np.count_nonzero(~y&~p))
    return dict(candidates=len(frame),scored=len(f),tp=tp,fp=fp,fn=fn,tn=tn,
        Accuracy=(tp+tn)/len(f) if len(f) else None,Recall=tp/(tp+fn) if tp+fn else None,
        Precision=tp/(tp+fp) if tp+fp else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def pct(value):
    return "未知" if value is None or pd.isna(value) else f"{value*100:.2f}%"


def check_sources():
    protocol=b.read(RUN/"protocol_BEFORE_ANY_FIT.json")
    amendment=b.read(RUN/"EMPTY_BINNING_ENGINEERING_AMENDMENT_BEFORE_MODELS.json")
    redirects={str((ROOT/"short_term_thinking_dataset_builder.py").resolve()):RUN/"source_preimages/short_term_thinking_dataset_builder.executed_32_models.py",
        str((ROOT/"short_term_thinking_current.py").resolve()):RUN/"source_preimages/short_term_thinking_current.executed_before_failed_pool_fix.py",
        str((ROOT/"short_term_thinking_research.py").resolve()):RUN/"source_preimages/short_term_thinking_research.before_empty_column_repair.py"}
    for path,digest in protocol["sources"].items():
        if b.sha(redirects.get(path,path))!=digest:
            raise ValueError("original frozen source is not preserved: "+path)
    for path,digest in amendment["sources"].items():
        p=RUN/"source_preimages/short_term_thinking_research.executed_32_models.py" if path==str((ROOT/"short_term_thinking_research.py").resolve()) else path
        if b.sha(p)!=digest:
            raise ValueError("executed model program or repair evidence changed")
    receipt=b.read(DATA/"coverage_and_role_distribution.json")
    b.verify(receipt["source_bindings"])
    b.verify(b.read(DATA/"current_extension_receipt.json")["source_bindings"])
    start=b.read(RUN/"execution_start.json")
    preserved={path:b.sha(path)==digest for path,digest in start["protected_before"].items()
        if Path(path).name not in ("AGENTS.md","relay_review_workbench.py","README.md")}
    if not all(preserved.values()):
        raise ValueError("protected original engines or ledgers changed")
    return dict(original_frozen_bindings_checked=len(protocol["sources"]),executed_program_bindings_checked=len(amendment["sources"]),
        protected_originals=preserved,post_fit_program_preimages_preserved=True)


def main():
    OUT.mkdir(exist_ok=True)
    checks=check_sources()
    original=joblib.load(RUN/"historical_final/dataset.joblib")
    bundle=joblib.load(DATA/"dataset.joblib")
    current=joblib.load(DATA/"current_extension.joblib")
    # The failed-pool completeness fix leaves EVERY historical training feature,
    # operational role, original label and candidate identical.
    pd.testing.assert_frame_equal(original["features"],bundle["features"])
    pd.testing.assert_frame_equal(original["prepared"]["relay"],bundle["prepared"]["relay"])
    checks["historical9245_features_and_original_labels_exactly_unchanged"]=True
    oldcurrent=joblib.load(RUN/"current_extension.joblib")["full"]
    newcurrent=current["full"]
    cols=list(oldcurrent.columns)
    pd.testing.assert_frame_equal(oldcurrent[oldcurrent.date.lt("2026-10-14")].reset_index(drop=True),newcurrent[newcurrent.date.lt("2026-10-14")][cols].reset_index(drop=True))
    checks["october_before14_all_inputs_and_targets_unchanged"]=True
    checks["october14_failed_source_rows"]=int((joblib.load(DATA/"engine_context.joblib")["attempts"].query("signal_date == '2026-10-14'").pool_assignment=="FAILED_BOARD_OBSERVATION").sum())
    del original;gc.collect()
    full=r.merged(bundle)
    predictions=pd.read_csv(RAW/"complete_paired_predictions.csv",dtype={"code":str},low_memory=False)
    # October ROLL252 was a BASE alias, not a fitted rolling comparator. Keep it
    # in the original artifact and exclude it from the corrected delivery.
    alias=predictions.arm.eq("ROLL252")&predictions.date.ge("2026-10-01")
    checks["october_mislabeled_roll_alias_rows_excluded"]=int(alias.sum())
    predictions=predictions[~alias].copy()
    model_checks=[]
    replay_current={}
    for month in pd.period_range("2025-07","2026-10",freq="M"):
        expected=str(month)
        test=newcurrent if expected=="2026-10" else full[full.date.str[:7].eq(expected)]
        for arm in r.ARMS:
            model_path=RAW/(expected+"_"+arm+"_model.joblib")
            saved=joblib.load(model_path)
            train=r.training_subset(full,str(month.start_time.date()),arm)
            receipt=b.read(RAW/(expected+"_"+arm+"_fit_STARTED.json"))
            if r.history._frame_digest(train,r.FEATURES)!=receipt["training_digest"]:
                raise ValueError("monthly training members or mature targets changed")
            if train.target_date.max()>=saved["cutoff"] or train.date.ge(saved["cutoff"]).any():
                raise ValueError("future training outcome")
            with threadpool_limits(limits=2):
                scores=saved["model"].predict_proba(test[r.FEATURES])[:,list(saved["model"].classes_).index(1)]
            old=pd.read_csv(RAW/(expected+"_"+arm+"_predictions_BEFORE_EVALUATION.csv"),dtype={"code":str})
            old=old.set_index(["date","code"])
            raw=np.asarray([old.loc[(row.date,row.code),"b_score"] for row in test.itertuples()])
            compare=test.date.lt("2026-10-14").to_numpy()
            np.testing.assert_allclose(scores[compare],raw[compare],atol=1e-12,rtol=0)
            if expected=="2026-10":
                replay_current[arm]=scores
                changed=test.date.eq("2026-10-14").to_numpy()
                model_checks.append(dict(month=expected,arm=arm,rows=len(test),training_rows=len(train),
                    corrected_current14_score_rows=int(np.sum(np.abs(scores[changed]-raw[changed])>1e-12)),
                    max_pre14_score_drift=float(np.max(np.abs(scores[compare]-raw[compare]))),model_sha256=b.sha(model_path)))
            else:
                model_checks.append(dict(month=expected,arm=arm,rows=len(test),training_rows=len(train),max_score_drift=float(np.max(np.abs(scores-raw))),model_sha256=b.sha(model_path)))
    # Only this new research view's October14 rows change, using the very same
    # already-trained models. Old forecasts, predictions and fit receipts remain.
    for arm in r.ARMS:
        for j,row in enumerate(newcurrent.itertuples()):
            if row.date!="2026-10-14":continue
            mask=predictions.arm.eq(arm)&predictions.date.eq(row.date)&predictions.code.eq(row.code)
            predictions.loc[mask,"b_score"]=replay_current[arm][j]
            predictions.loc[mask,"score"]=row.base_score if row.daily_one_word==1 else replay_current[arm][j]
            predictions.loc[mask,"primary_role"]=row.primary_role
    if predictions.duplicated(["date","code","arm"]).any():raise ValueError("duplicate full cohort")
    # Independent full-pool outcome and immutable BASE-score check.
    historical_targets=full.set_index(["date","code"])
    for arm in ("BASE","ROLL252")+r.ARMS:
        x=predictions[predictions.arm.eq(arm)&predictions.date.lt("2026-10-01")]
        assert len(x)==3707
        for row in x.itertuples():
            src=historical_targets.loc[(row.date,row.code)]
            assert (pd.isna(row.target) and pd.isna(src.target)) or row.target==src.target
            assert bool(row.label_valid)==bool(src.label_valid)
    base=predictions[predictions.arm.eq("BASE")].set_index(["date","code"]).score
    for arm in r.ARMS:
        z=predictions[predictions.arm.eq(arm)&predictions.daily_one_word.eq(1)]
        np.testing.assert_allclose(z.score.to_numpy(),[base.loc[(x.date,x.code)] for x in z.itertuples()],atol=1e-15,rtol=0)
    predictions.to_csv(OUT/"complete_paired_predictions_VALIDATED.csv",index=False,encoding="utf-8-sig")
    evaluation=r.evaluate(predictions,{**bundle,"bars":current["bars"]},OUT)
    evaluation.update(actual_new_fits=32,failed_preoptimization_fit_calls=1,total_fit_invocations=33,models_refit_for_corrections=0,
        post_fit_input_correction="October14 complete archived failed-board pool, no known labels that day; pre14 predictions bit-identical within1e-12",
        october_roll252="No saved October monthly baseline. The erroneous BASE alias is excluded, not counted as ROLL252.")
    # Independently count confusion matrices and relative ranks, not just call
    # the reporting implementation a second time.
    metrics_rows=[]
    for fold,report in evaluation["comparisons"].items():
        lo,hi={"2025H2":("2025-07-01","2025-12-31"),"2026JanAug":("2026-01-01","2026-08-31"),"2026Sep":("2026-09-01","2026-09-30"),"2026Oct":("2026-10-01","2026-10-31")}[fold]
        for arm,m in report["metrics"].items():
            part=predictions[predictions.arm.eq(arm)&predictions.date.between(lo,hi)]
            nonword=part[part.daily_one_word.eq(0)]
            counts=independent_binary(nonword)
            for k,v in counts.items():
                target=m["observed_nonword"].get(k)
                assert target==v or (target is not None and v is not None and abs(target-v)<1e-14),(fold,arm,k,target,v)
            # Top1/Top2 must be selected from the full signal pool before reading
            # labels; explicitly retain selected unresolved outcomes.
            for k in (1,2):
                selected=nonword.sort_values(["date","score","code"],ascending=[True,False,True]).groupby("date").head(k)
                valid=selected[selected.label_valid.astype(bool)&selected.target.notna()]
                metric=m["top"+str(k)]
                assert len(selected)==metric["selected"] and len(valid)==metric["known"]
                assert abs(valid.target.mean()-metric["candidate_hit_rate"])<1e-14
            multi=nonword.groupby(["date","boards"]).filter(lambda g:len(g)>1)
            m["multiple_nonword_same_height_top1"]=r.top_metrics(multi,1,True)
            m["multiple_nonword_same_height_top2"]=r.top_metrics(multi,2,True)
            metrics_rows.append(dict(fold=fold,arm=arm,**counts,Top1=m["top1"]["candidate_hit_rate"],Top1_known=m["top1"]["known"],Top1_selected=m["top1"]["selected"],
                Top2=m["top2"]["candidate_hit_rate"],height_Top1=m["height_baseline_top1"]["candidate_hit_rate"],same_height_Top1=m["same_height_top1"]["candidate_hit_rate"]))
    b.write_new(OUT/"model_validation_results_VALIDATED.json",evaluation)
    pd.DataFrame(metrics_rows).to_csv(OUT/"nonword_metrics_VALIDATED.csv",index=False,encoding="utf-8-sig")
    # Conditional promotion of rule roles is measurable, identity accuracy is not.
    role_rows=[]
    role_full=pd.concat([full,newcurrent],ignore_index=True)
    for fold,lo,hi in [("2025H2","2025-07-01","2025-12-31"),("2026JanAug","2026-01-01","2026-08-31"),("2026Sep","2026-09-01","2026-09-30"),("2026Oct","2026-10-01","2026-10-31")]:
        f=role_full[role_full.date.between(lo,hi)]
        for role in b.ROLES:
            z=f[f["role_"+role].eq(1)];known=z[z.target.notna()&z.label_valid.astype(bool)]
            role_rows.append(dict(fold=fold,operational_role=role,candidates=len(z),known_outcomes=len(known),promoted=int(known.target.sum()),
                conditional_promotion_rate=float(known.target.mean()) if len(known) else None,independent_role_gold_n=0,identity_accuracy=None))
    pd.DataFrame(role_rows).to_csv(OUT/"role_conditional_promotion_NOT_identity_accuracy.csv",index=False,encoding="utf-8-sig")
    recent=pd.read_csv(DATA/"20260930_to_20261014_all_roles.csv",dtype={"code":str,"parent_anchor_code":str})
    old14=pd.read_csv(ROOT/"results/score_update_20261014T1153Z/完整15只评分与独立判断.csv",dtype={"代码":str})
    old14_map=old14.set_index("代码")["评分前独立判断"].to_dict()
    for index,row in recent[recent.date.eq("2026-10-14")].iterrows():recent.loc[index,"original_independent_judgement"]=old14_map[row.code]+"（原补充审阅，已见模型）"
    wide=predictions.pivot(index=["date","code"],columns="arm",values="score").reset_index()
    recent=recent.merge(wide,on=["date","code"],validate="one_to_one")
    labels=predictions[predictions.arm.eq("BASE")][["date","code","target","label_valid"]]
    recent=recent.merge(labels,on=["date","code"],validate="one_to_one")
    for arm in ("BASE",)+r.ARMS:
        recent[arm+"_rank_nonword"]=recent[recent.daily_one_word.eq(0)].groupby("date")[arm].rank(ascending=False,method="min")
    recent.to_csv(OUT/"短线思维周期演进与个股龙头角色总表.csv",index=False,encoding="utf-8-sig")
    names=dict(zip(role_full.code,role_full.name))
    role_zh={"SPACE_DRAGON":"空间锚","CYCLE_CROSS_DRAGON":"穿越/破局候选","SUPPLEMENT_DRAGON":"补涨候选","USURP_DRAGON":"卡位候选","TURNOVER_DRAGON":"换手核心候选","THEME_LEADER":"行业主线代理","THEME_SEED":"行业先锋代理","RELAY_CONTENDER":"接力竞争者"}
    def role_names(text):
        values=ast.literal_eval(text) if isinstance(text,str) else text
        return "、".join(role_zh.get(v,v) for v in values) or "接力竞争者"
    lines=["# 短线思维周期演进与个股龙头角色总表","","信号日2026-09-30至10-14，六个完整归档日共75只次；15日当前池请求失败，无旧池替代。规则标签为候选；原独立判断、BASE和新研究模型分栏，不是新事前预测。","",
        "父锚列为固定2—6交易日滞后筛选内的最佳时序候选，未通过补涨规则不表示存在父子关系。完整父锚竞争及反证见原始链接CSV，不能把表中任意匹配者认定为父龙。",""]
    for day,f in recent.groupby("date"):
        lines += ["## "+day,f"周期：{f.cycle_regime.iloc[0]}；完整{len(f)}只。","",
            "| 股票 | 板位/形态 | 全部规则角色 | 父锚时序候选/状态 | 原独立方向 | BASE评分 | 双通道研究评分 | 实际晋级 |",
            "|---|---|---|---|---|---:|---:|---|"]
        for row in f.sort_values(["boards","code"],ascending=[False,True]).itertuples():
            code=str(row.parent_anchor_code) if pd.notna(row.parent_anchor_code) else ""
            parent=(names.get(code,code)+" / "+row.parent_anchor_state) if code else "无匹配/证据不足"
            shape="一字" if row.daily_one_word==1 else "非一字" if row.daily_one_word==0 else "形态待核"
            truth="晋级" if row.target==1 else "未晋级" if row.target==0 and row.label_valid else "待核/未到期"
            score=f"{row.ST_DUAL*100:.1f}" + ("（A监测沿BASE）" if row.daily_one_word==1 else "")
            independent=row.original_independent_judgement
            if independent=="NO_ARCHIVED_INDEPENDENT_CALL":independent="本登记无独立判断"
            lines.append(f"| {row.name} {row.code} | {row.boards}/{shape} | {role_names(row.roles)} | {parent} | {independent} | {row.BASE*100:.1f} | {score} | {truth} |")
    text="\n".join(lines)+"\n"
    (OUT/"短线思维周期演进与个股龙头角色总表.md").write_text(text,encoding="utf-8")
    # Compact self-contained table for search/filter, with no external scripts.
    headings=["日期","股票","板位","周期","规则角色","父锚候选","独立判断","BASE","双通道研究分","结果"]
    body=[]
    for row in recent.itertuples():
        vals=[row.date,row.name+" "+row.code,str(row.boards),row.cycle_regime,role_names(row.roles),str(row.parent_anchor_code) if pd.notna(row.parent_anchor_code) else "无匹配",row.original_independent_judgement,f"{row.BASE*100:.1f}",f"{row.ST_DUAL*100:.1f}","晋级" if row.target==1 else "未晋级" if row.target==0 and row.label_valid else "待核/未到期"]
        body.append("<tr>"+"".join("<td>"+html.escape(x)+"</td>" for x in vals)+"</tr>")
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>短线思维周期与角色总表</title><style>body{font:15px/1.6 system-ui;color:#17313c;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd8dd;text-align:left}th{position:sticky;top:0;background:#e2eef3}input{font:inherit;padding:8px;width:360px}tr:hover{background:#f1f7f7}.note{max-width:1000px}</style><h1>短线思维周期与角色总表</h1><p class="note">75只次完整保留。规则候选非独立龙头真值；分数未校准，一字沿BASE作情绪监测；本页为已见历史回放。15日完整新池请求失败，最新完整归档为14日。</p><input id="filter" placeholder="按日期、股票、角色搜索"><p id="count">75 / 75</p><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headings)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table><script>document.getElementById("filter").oninput=function(){let n=0;document.querySelectorAll("tbody tr").forEach(r=>{let show=r.textContent.toLowerCase().includes(this.value.toLowerCase());r.hidden=!show;if(show)n++;});document.getElementById("count").textContent=n+" / 75"}</script></html>'
    (OUT/"短线思维周期演进与个股龙头角色总表.html").write_text(page,encoding="utf-8")
    coverage=b.read(DATA/"coverage_and_role_distribution.json")
    summary=["# 全周期短线思维重建与模型验证交付","",
        "已完成9245条因果规则重建、两种方案各16个月的32次训练、同池验证和工作台接入。双通道方案9月非一字Recall从15%到50%，但误报增加、收益证据不足；两个方案均未通过本批整体研究标准，未替换正式模型。","",
        "## 全历史覆盖与角色","",
        "9245只次、684个含连板候选的信号日全部保留。主角色规则决策100%，未满足六类条件者明确为接力竞争者。规则决策覆盖、可观察事实覆盖、独立身份准确率分别统计。","",
        "| 原训练窗 | 行数 | 规则决策 | 日线形态可观察 | 完整本轮路径 | 炸板淘汰率可观察 | 历史题材/行业 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in ("2025H2","2026JanAug","2026Sep"):
        x=coverage["windows"][fold+"_training"]
        summary.append(f"| {fold} | {x['rows']} | {pct(x['rule_decision_coverage'])} | {pct(x['daily_price_identity_coverage'])} | {pct(x['full_streak_price_coverage'])} | {pct(x['pk_elimination_coverage'])} | {pct(x['theme_industry_proxy_coverage'])} |")
    summary += ["","全字段真实非空100%没有达成：历史行业/题材仍无来源，部分日线有缺失或价格冲突。不会用0、旧一字代理、当前分类或晚到帖子填补这些事实。父锚不再是0输入，但时序候选匹配不等于已确认补涨关系。","",
        "主角色分布："+"；".join(role_zh.get(k,k)+str(v)+"条" for k,v in coverage["primary_role_distribution"].items())+"。",
        "多角色分布："+"；".join(role_zh.get(k,k)+str(v)+"条" for k,v in coverage["multi_role_distribution"].items())+"。角色可重叠，不强制同位二选一。",
        f"V215原58772条全部保留原件，新模块精确价格/分类检查保留{coverage['exact_v215_audit']['exact_kept']}条、排除3126条，并按相邻官方交易日重算板位。仍是当前股票池回溯，未消除幸存者偏差或历史非ST资格缺口。","",
        "## 固定模型比较","",
        "训练窗先取原完整252个已成熟信号日，再区分真实日线非一字训练与全池训练。移除旧signal_one_word代理，加入因果动态关系，使用固定浅层直方图梯度提升；训练期按日及类别平衡，未搜索参数、阈值或权重。分数未经校准。A通道的一字情绪锚沿用原BASE；B通道独立研究。","",
        "| 区间 | 方案 | 非一字Recall | Precision | F1 | Top-1 | Top-2 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for x in metrics_rows:
        summary.append(f"| {x['fold']} | {x['arm']} | {pct(x['Recall'])} | {pct(x['Precision'])} | {pct(x['F1'])} | {pct(x['Top1'])} | {pct(x['Top2'])} |")
    sep=evaluation["comparisons"]["2026Sep"]["metrics"]
    summary += ["",f"9月真实非一字143条，107条目标可核、40个正例。BASE检出6个、误报4个；双通道检出20个、误报22个，减少14个漏报同时增加18个误报。B专用方案检出27个、误报38个。不能只报告Recall改善。",
        "9月Top-1：双通道8/13=61.54%，BASE7/13=53.85%，高度基线6/13=46.15%；完整选择16天，3天所选目标仍未知。Top-2为17/26对15/26，完整选择32条，6条目标未知。收益与标签缺口均保留。",
        "同板位Top-1/Top-2及多候选同位组另见验证JSON；单成员组不作为竞争排序提升的证明。按原先固定的五个有观察信号日块、两挑战校正区间，双通道9月Recall差值区间为"+str(evaluation['comparisons']['2026Sep']['intervals']['ST_DUAL']['nonword_recall'])+"；Top-1差值区间跨0。历史被反复研究过，以上不是新盲测或因果证明。","",
        "## 收益代理与角色有效性","",
        "B是次日入场日；仅模拟非涨停开盘且0—7%高开的开盘价场景，入/出场各0.1%滑点、0.3%往返费，B+1/B+2收盘退出。一字下跌卖出受阻保留未平仓标记并计入风险尾部，未成交/缺行情不填零收益。此代理不验证9:25/9:35承接、队列或回封，也不是实盘成交。","",
        "| 9月方案 | Top-1选择 | B+1可核 | B+1跌幅≥7%风险 | B+2可核 | B+2净收益均值 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in ("BASE",)+r.ARMS:
        e=sep[arm]["execution_proxy"]
        summary.append(f"| {arm} | {e['selected']} | {e['b1']['matured']} | {pct(e['b1']['risk_loss_ge7pct_including_blocked'])} | {e['b2']['matured']} | {pct(e['b2']['mean_net'])} |")
    summary += ["","以上只有4—5个B+2可核场景，且各方案选择不同；未证明期望收益或抗大面改善。2025H2和1—8月部分收益代理退化也完整保留于JSON，不能用9月小样本覆盖。实际/纸面成交新增均0。",
        "六类角色提供全部规则候选及跨期条件晋级率（role_conditional_promotion_NOT_identity_accuracy.csv），不是用自己的规则标签验证自己。独立六角色真值为0，身份识别准确率不可计算；历史题材先锋为0尤其不能声称已解决全部角色识别。","",
        "## 9—10月全池与案例","",
        "[完整75只次逐日总表](短线思维周期演进与个股龙头角色总表.md)及同名CSV/HTML，覆盖9月30日、10月8/9/12/13/14日。10月15日一次原源请求SSL失败已留档，不能用14日15只冒充15日新池。",
        "百洋13日被同一规则引擎识别为九阳首次断板窗口的补涨候选，14日保留该切换记忆；也与新经典、慈文并列空间锚。名称/价格关系只是候选印证，未认证资金从父锚迁移。其他原漏判案例的每一天、一字与非一字阶段、两方案分数及排名均在总表保留。",
        "六大历史案例复读用于校验实战算子与反证：AI的分支回流不抹去旧核心损失，华为/消费的高度不替代群体健康，低空的触板不等于收盘，9·24的拥挤不等于可成交，DeepSeek的技术动作不等于订单。没有把六段赢家故事做成晋级标签或历史题材真值。实际论坛读取、评论时钟和观点取舍仍由原工作台前置层校验，新模型不继承大佬多空结论。","",
        "## 工程接入与本批修正","",
        "relay_review_workbench.py prepare新增--thinking-bundle，完整传history_datasets、mainline_history、narrative_reviews及父锚证据，先保存遮蔽模型分数的角色事实。thinking-score要求可核原独立审阅，单独显示研究分，保持原freeze/compare/execution门。新日期可使用scope_correction/engine_context.joblib保留完整近期市场/父锚历史；缺当前炸板源时不会填0。",
        "本批实际完成32个模型及预测保存，另有1次在优化前失败的分箱调用：训练全空列触发安装库缺陷。失败、原协议和源码保留；零已完成模型时追加工程修复记录，仅对训练全空列保持常数，未根据效果改参数。随后本固定比较一次完成，原BASE重拟合0。",
        "复核还纠正了14日替代炸板池未传入、以及10月将BASE别名显示为ROLL252的错误。均保留原产物，用独立修正版交付；9245条历史特征与全部训练标签完全一致，14日前分数逐模型复现，32个模型均未重训，14日尚无目标结果不能借此追结果调参。",
        "原正式预测、旧分数/排名、失败比较、登记和账本哈希保留。新正式预测0、新真实成熟0、新订单/成交0。两个方案未通过完整证据标准，不部署，不按本结果再换权重或阈值。",
        "唯一下一步：沿现有新日期流程保留完整首板/炸板/父锚价格链与独立角色审阅，在不改本批模型的情况下前向检验误报、Top排序和可成交性；恢复15日完整新源后再补当天全池。整体预测与身份/收益提升目标尚未完成。",""]
    (OUT/"全周期短线思维重建与模型验证交付.md").write_text("\n".join(summary),encoding="utf-8")
    checks.update(formed_at=b.now(),model_replays=model_checks,total_completed_models=32,failed_preoptimization_calls=1,new_fits_in_verifier=0,
        complete_paired_rows=len(predictions),complete_original_historical_candidates=9245,current_candidates=64,recent_report_rows=len(recent),
        final_report_path=str((OUT/"全周期短线思维重建与模型验证交付.md").resolve()),final_report_sha256=b.sha(OUT/"全周期短线思维重建与模型验证交付.md"),
        latest_complete_archived_signal_date="2026-10-14",latest15_unavailable=True,old_forecasts_edited=False)
    b.write_new(OUT/"independent_verification.json",checks)
    print(json.dumps({k:checks[k] for k in ("total_completed_models","failed_preoptimization_calls","complete_paired_rows","historical9245_features_and_original_labels_exactly_unchanged","october14_failed_source_rows","recent_report_rows")},ensure_ascii=False))


if __name__=="__main__":main()
