"""Deliver the fixed positive-filter experiment without additional fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import relay_positive_filter_research as r
import short_term_thinking_dataset_builder as b


def main():
    result=b.read(r.ROOT/'evaluation/fixed_study_result.json')
    r.verify(result['source_bindings'])
    table=pd.read_csv(r.ROOT/'evaluation/complete_same_pool_classification.csv')
    meta=pd.read_csv(r.ROOT/'execution/all32_training_only_selections.csv')
    top=pd.read_csv(r.ROOT/'evaluation/positive_recommendation_Top3_FULL_SLOTS.csv')
    filtered=pd.read_csv(r.ROOT/'evaluation/all_filtered_candidates_GAINS_LOSSES_UNKNOWN.csv',dtype={'code':str})
    summary=[]
    for period in ['2025H2','2026JanAug','2026Sep','HISTORICAL_POOLED','2026Oct']:
        for arm in ['BASE','ST_DUAL',r.BASE_ARM]+r.ARMS:
            row=table[(table.period==period)&(table.arm==arm)&(table.scope=='ORIGINAL_NONWORD')].iloc[0]
            name={'BASE':'正式BASE','ST_DUAL':'原双通道',r.BASE_ARM:'冻结SOURCE基准',
                  r.ARMS[0]:'二级分数对照',r.ARMS[1]:'二级个股信息过滤'}[arm]
            summary.append(f"|{period}|{name}|{int(row['tp'])}/{int(row['fp'])}|{row.Precision:.2%}|{row.Recall:.2%}|{row.F1:.2%}|")
    decisions=[]
    for arm in r.ARMS:
        info=result['arms'][arm]
        failures=[key for key,passed in info['guards'].items() if not passed]
        counts=meta[meta.arm.eq(arm)].selected_threshold.value_counts().sort_index().to_dict()
        effects=filtered[filtered.arm.eq(arm)].filter_effect.value_counts().to_dict()
        decisions.append(f"- `{arm}`：{'通过固定历史筛选' if info['fixed_screen_pass'] else '未通过固定历史筛选'}；"
            f"16月阈值次数：{counts}；过滤效果（含10月补充）：{effects}；未过门：{failures}。")
    rec=[]
    for arm in [r.BASE_ARM]+r.ARMS:
        row=top[(top.period=='HISTORICAL_POOLED')&top.arm.eq(arm)].iloc[0]
        rec.append(f"- `{arm}`：正向推荐Top3已知命中{int(row['hits'])}/{int(row.full_slots)}完整槽，"
                   f"选出{int(row.selected)}，空槽{int(row.empty_slots)}，结果未知{int(row.unknown_selected)}。")
    audit_files=[]
    for sub in ['training_audit','independent_verification','lessons']:
        audit_files.extend(p for p in (r.ROOT/sub).rglob('*') if p.is_file())
    if not any('independent_verification' in str(p) and p.suffix=='.json' for p in audit_files):
        raise ValueError('independent effect verification must be delivered before final receipt')
    if not any('training_audit' in str(p) and p.suffix=='.json' for p in audit_files):
        raise ValueError('training qualification audit required')
    tests=r.ROOT/'preflight_tests.xml'
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    if any(int(s.get('errors',0))+int(s.get('failures',0)) for s in suites):
        raise ValueError('targeted checks must pass')
    test_count=sum(int(s.get('tests',0)) for s in suites)
    delivery=r.ROOT/'delivery';delivery.mkdir(exist_ok=False)
    pass_any=any(x['fixed_screen_pass'] for x in result['arms'].values())
    historical=table[(table.period=='HISTORICAL_POOLED') & (table.scope=='ORIGINAL_NONWORD')].set_index('arm')
    baseline=historical.loc[r.BASE_ARM]; candidate=historical.loc[r.ARMS[1]]
    selection=b.read(r.ROOT/'execution/2025-11_SELECTION_BEFORE_FINAL_FIT_AND_OUTER.json')['choices'][r.ARMS[1]]
    validation_base=selection['statistics']['0.0'];validation_new=selection['statistics']['0.3']
    text=f'''# 历轮经验总结与二级误报过滤：实际交付

本批先保存历轮经验及来源，再冻结一个新假设，实际完成{result['new_second_stage_fits']}次二级Logistic拟合，第一阶段SOURCE/BASE/双通道重拟合0次。{'至少一个方案通过本批固定历史筛选，仍只可影子验证。' if pass_any else '两项二级过滤方案均未通过固定历史筛选，不部署，本批不再调阈值或字段。'}

主候选合计由{int(baseline['tp'])}TP/{int(baseline['fp'])}FP变为{int(candidate['tp'])}TP/{int(candidate['fp'])}FP：只少{int(baseline['fp']-candidate['fp'])}个误报，同时多漏{int(baseline['tp']-candidate['tp'])}个晋级，Precision由{baseline.Precision:.2%}降至{candidate.Precision:.2%}，完整已知Accuracy由{baseline.full_known_accuracy:.2%}降至{candidate.full_known_accuracy:.2%}。分数单变量对照16个月全部选择不滤，结果与原基准相同。

## 已吸收的历轮经验

资料资格恢复、推断输入补齐和重新学习权重分开归因；不把少报造成的0误报算优化，不把训练内拟合、模块覆盖、LLM一致性或单月Top1上升当稳定预测增益。旧失败与原验收条件保持，新允许的Recall最多下降5个百分点仅用于本批。完整试验索引、真实fit/复用/失败次数及不同分母在`../lessons/`，未将旧全池3707、原非一字2980和补充81只混为同一榜单。

## 本批如何训练

冻结SOURCE的16个最终头和48个内层模型。每月只用自己的3块时间OOF；前两块的已知、已核非一字原正向记录训练二级模型，标签严格成熟于第三块首日。第三块仅选阈值；选择留档后，用月初前成熟的全部3块同条件记录重拟合二级模型，再对外月推断。

两臂为原分logit单变量对照和额外9个个股/同位字段；固定等行权重、C=1、L2、lbfgs最多2000轮，不另平衡类别。填补/标准化只fit当前元训练部分。同位分位直接来自完整原池，未在正向子集重新排序。

每个日期完整OOF由原保存模型推断，已有缓存分数重放核对；本批实际没有新增原B缓存补算行，不把完整性防护称新增资料。阈值0显式不滤，其余固定0.1至0.9；只在完整B统计中选择Precision严格改善、FP至少少10%、Recall下降不超过5pp且F1不降的阈值，否则不滤。过滤者在研究方向记FAIL，原始方向/分数/排序保留；原UNKNOWN不改，过滤掉的TP仍计漏判。

全部13头共49101条输出，旧11头41547条原列精确保留；新两头各3777条。新候选没有取得生产权限。

## 同池实际结果

历史主比较原非一字2980条：2932已知、48未知、871个已知晋级；9月143条：107已知、40正例、36未知。10月只是部分成熟补充，15日6只在两臂共12条记录中保持结果未知，不读取16日新实际结果。

|时段|方案|TP/FP|Precision|Recall|F1|
|---|---|---:|---:|---:|---:|
{chr(10).join(summary)}

{chr(10).join(decisions)}

合计通过门：Precision至少提高3pp、FP至少少10%、F1不降；每历史段Recall最多下降5pp、FP不增、F1不降；正向推荐Top3的合计命中不降。原观察Top3的守恒是工程核对，不是新排序效果。

## 推荐槽与过滤代价

{chr(10).join(rec)}

每个被过滤者的获益、损失、结果未知都在`../evaluation/all_filtered_candidates_GAINS_LOSSES_UNKNOWN.csv`。推荐Top3先从完整正向集合按原排序选择，再看标签；每原日期固定3槽，不先删除未知结果，也不因没有推荐而缩分母。

## 这轮新增的经验

多特征过滤15个月未找到合格阈值；只有2025年11月选择0.3。该月训练期第三块从{validation_base['tp']}TP/{validation_base['fp']}FP变为{validation_new['tp']}TP/{validation_new['fp']}FP，表现为删6个FP、未丢TP；真正推到后续11月却删2个FP并误删3个TP。训练期资料中的过滤关系没有稳定延续，不能把第三块选阈表现当外月效果。

这没有排除所有二级模型，也不能把退化唯一归因于final重拟合或市场变化；本批同时存在第一阶段训练长度和二级final尺度迁移，因果贡献未拆解。能确认的是：在本批固定字段、算法、选择政策和原资料下，未得到稳定的Precision增益；已有信息再组合并未补出新的真实题材或盘口证据。

## 核验与适用边界

{test_count}项针对检查通过，另有独立结果复算及训练资格/保存模型重放。拟合前和完成收据分别记录真实时钟；全部已完成模型、阈值选择、完整OOF、预测先于评价保存。5信号日循环块、2000次配对抽样区间只作探索，两新臂校正不覆盖此前全部顺序试验。

第三块是选择材料，不是无偏效果验证；全3块final重拟合改变分数尺度，OOF第一阶段62/125/188日模型与外部252日模型也有迁移风险。历史已多次见过，来源晚取得及幸存者范围等限制仍保留，没有新的盲测、角色身份认证或真实成交收益结论。

旧19份测试XML缺件沿显式本地恢复清单处理，模型/代码/数据哈希例外0。没有输出IP、恢复含IP错误页、使用网络API、改正式模型/旧预测/账本/定时任务或平台目标。后续若研究未过门则停止该批，不继续换阈值追分；若历史通过，仍沿原前向验收门做影子验证。
'''
    report=delivery/'历轮经验总结与二级误报过滤交付.md'
    report.write_text(text,encoding='utf-8')
    bindings=result['source_bindings'] | {str(p.resolve()):b.sha(p) for p in audit_files+[tests,Path(__file__),report,
        r.ROOT/'protocol_BEFORE_ANY_FIT.json',r.ROOT/'evaluation/fixed_study_result.json']}
    recovery_result=r.verify(bindings)
    receipt=dict(formed_at=b.now(),research_second_stage_fits=result['new_second_stage_fits'],
        first_stage_refits=0,targeted_tests_passed=test_count,
        isolated_synthetic_unit_model_fits=1,history_summary_formed_before_protocol=True,
        all_old41547_rows_preserved=True,all_outputs49101=True,
        original_models_or_formal_predictions_changed=False,new_official_predictions=0,
        deployment=False,network_requests=0,IP_values_output=False,old_error_page_restored=False,
        acceptance=result['arms'],source_recovery=recovery_result,source_bindings=bindings)
    b.write_new(delivery/'study_delivery_verified_NO_PRODUCTION_CHANGE.json',receipt)
    print(dict(report=str(report.resolve()),second_stage_fits=result['new_second_stage_fits'],
               accepted={arm:v['fixed_screen_pass'] for arm,v in result['arms'].items()}))


if __name__=='__main__':main()
