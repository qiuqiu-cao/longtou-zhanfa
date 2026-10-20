"""Read-only discrimination diagnostics and evidence-backed delivery."""
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import short_term_thinking_dataset_builder as b
import relay_temporal_score_calibration as research


def run():
    delivery=ROOT/'delivery'
    if delivery.exists():raise FileExistsError('immutable new delivery only')
    result_path=ROOT/'execution/calibration_fixed_result.json'
    result=b.read(result_path);b.verify(result['source_bindings'])
    independent_path=ROOT/'independent_verification/temporal_calibration_independent_receipt.json'
    independent=b.read(independent_path)
    numeric_path=ROOT/'training_audit/executed_numeric_qualification/independent_numeric_qualification_receipt.json'
    numeric=b.read(numeric_path)
    for data in [independent,numeric]:
        sources=data.get('source_bindings',data.get('sources',{})) | data.get('output_bindings',{})
        if sources:b.verify(sources)
    if not numeric['numerical_qualification_pass'] or not independent['original_26439_all_columns_exact']:
        raise ValueError('independent numerical or preservation audit failed')
    if not independent['all16_calibrators_positive_slope_and_converged'] or not independent['original_day_rankings_and_Topk_all_identical']:
        raise ValueError('independent mapping or ranking audit failed')
    data_path=ROOT/'execution/all30216_predictions_BEFORE_EVALUATION.csv'
    frame=pd.read_csv(data_path,dtype={'code':str},float_precision='round_trip',low_memory=False)
    if len(frame)!=30216 or result['total_new_fits']!=64 or result['fixed_screen_pass']:
        raise ValueError('complete fixed failed challenge expected')
    records=[]
    for period in research.PERIODS:
        for arm in [research.BASE_ARM,research.ARM]:
            part=frame[frame.date.between(period['start'],period['end'])&frame.arm.eq(arm)&frame.daily_one_word.eq(0)]
            valid=part.target.isin([0,1])&part.label_valid.fillna(False).astype(bool)&part.b_score.notna()
            known=part.loc[valid]
            records.append(dict(period=period['id'],arm=arm,complete_nonword=len(part),known=len(known),
                positive=int(known.target.eq(1).sum()),ROC_AUC=roc_auc_score(known.target,known.b_score),
                average_precision=average_precision_score(known.target,known.b_score),
                min_internal_score=known.b_score.min(),max_internal_score=known.b_score.max(),
                event_probability_validated=False,not_blind=True))
    diagnostics=pd.DataFrame(records)
    counts=pd.read_csv(ROOT/'execution/full_same_scope_classification_metrics.csv')
    sep=counts[counts.period.eq('2026Sep')&counts.scope.eq('ORIGINAL_NONWORD')].set_index('arm')
    old,new=sep.loc[research.BASE_ARM],sep.loc[research.ARM]
    if (old.tp,old.fp,new.tp,new.fp)!=(22,24,0,0):raise ValueError('fixed outcome differs')
    delivery.mkdir()
    diagnostics.to_csv(delivery/'raw_and_mapped_score_discrimination_NOT_PROBABILITY.csv',index=False,encoding='utf-8-sig')
    root=ROOT.resolve().as_posix();previous=research.PREVIOUS.resolve().as_posix()
    text=f'''# 误报原因、进一步优化实测与结论

本轮按用户“为什么误报这么多能不能再优化”的新手动指令完成：完整误报审查，以及一个新的时间顺序OOF校准固定挑战。原平台目标上次读取状态为usageLimited，本轮没有创建、恢复或改变平台目标，也没有修改原预测或模型。

9月原非一字池143条，已知107条，其中40正例、67负例，另36条未知。最新源特征研究头已知正向46条中22正确、24误报，另13条正向结果仍未知。24/46=52.17%是成熟正向信号中的错误占比；24/67=35.82%是标准误报率，两者分母不同。

## 为什么误报这么多

1. **分数尺度未按实际晋级率校准。** 原HGB训练先按日分配权重，再让正负类总质量各50%，最后仍以raw score>0.5判晋级。这个训练分不是自然市场下的事件概率。这是一项已核机制，不能据此断言24个误报全部由它造成。
2. **正误股票的特征重叠明显。** 9月18个误报与真阳性同日出现，9个还与真阳性同日同板位；0.5—0.6区间有18个误报、16个真阳。直接删除这个区间会同时删掉大量正确判断，不能据已见结果选择新阈值。
3. **资料修复没有补齐个股区分证据。** 已恢复当前价型不等于确认完整连板路径、题材主线、父锚身份或真实盘口。92字段中39列在所有多候选日内相同（包含常量），其余个股字段确有差异，但现有区分力有限；不能把规则角色当独立身份真值。

24个旧误报在上一轮单字段重训后一个都没有被消除，只多抓1个真阳性。完整逐股及分层证据见[误报机制审查]({root}/fp_mechanism_audit/误报机制_完整三段归档诊断.md)与[独立对照]({root}/comparison_audit/误报独立对照与归因.md)。同位比较中也存在多股共同晋级和全体失败，不能机械二选一。

## 这次实际优化了什么，结果如何

每月保持原252个成熟信号日、原92字段和原最终头。新增3个顺序验证块，各块训练仅用更早且目标已成熟的样本（62/125/188日）；只用这些未参与对应模型训练的原非一字输出学习一个Platt映射。校准每日质量1，未再次平衡正负类；原排序分完全保留。该数据分离原则依据[scikit-learn校准文档](https://scikit-learn.org/stable/modules/calibration.html)。本实现使用固定Logistic(logit(raw))，没有阈值或参数搜索。

实际新增48个内层HGB拟合和16个校准器拟合，共64次，约42.23秒；16个旧最终头原样复用。16个校准器斜率均为正、迭代仅5—7次；48块成熟边界、权重和保存模型核验通过。技术运行成功不代表预测改善。

| 9月同一107条已知非一字 | 抓到晋级 | 误报 | Recall | Precision |
|---|---:|---:|---:|---:|
| 最新源特征研究头 | 22 | 24 | 55% | 47.83% |
| 旧误报成本1.5研究头 | 8 | 5 | 20% | 61.54% |
| 本次时间OOF校准、固定>0.5 | 0 | 0 | 0% | 不适用：无正向信号 |

本次0误报来自关闭全部非一字正向信号，40个真晋级全部漏掉；2025H2和2026年1—8月也没有发出非一字正向。Precision是无定义的0/0，不能报100%或宣称误报问题解决。固定误报门虽通过，但Precision增益、Recall及F1门失败，整体淘汰，不部署、不换阈值再试。

9月原分对107个已知结果的ROC AUC为0.6287；同月单调校准不增加排序信息。这里的AUC是历史区分指标，不是胜率或涨停概率。跨月份使用不同映射可改变跨月混合AUC，因此没有把全期排序声称为数学不变；日内Top1/2/3使用原ranking_score逐股保持一致。

## 保留下来的改进与仍未解决的部分

本轮保留可复用的严格时间OOF校准研究入口、完整逐股误报归因和独立核验；没有把失败映射接管原评分或判断。完整输出为8组×同一3777条观察，共30216条，旧7组26439条的原始字段逐值保留。原[7方案评分页]({previous}/scorecard/final_verified26439_research_view/complete_scorecard.html)保持原件，本批失败研究在独立目录留档。

降低误报仍有研究空间，但接下来需要检验能够区分同日同板位候选的新有效信息，例如既存归档中可核对的完整竞争、首板助攻、题材及承接证据；仅把相同输入反复重训、改阈值或提高误报惩罚，没有通过本批效果验证。先做来源和区分能力检查，再对新假设固定一次比较，不能把未知自动当否决条件。

本批8项新增针对测试通过，保存模型、时间边界与全量旧输出另有独立核验。旧BASE、全部正式评分/账本/原失败与定时任务保留；没有新增正式预测、真实成熟或成交。归档10月15日6只的原目标字段（含缺失）保持，官方下一目标日按已绑定日历独立附列为10月16日；本批未读取该日结果用于调试。

历史结果已多次见过，资料取得时钟有晚到，仍有幸存者与题材/盘口缺口；这不是新盲测，也不证明角色识别或真实扣费收益改善。校准映射虽已拟合，事件概率依然未验证。
'''
    report=delivery/'误报原因与进一步优化实测.md';report.write_text(text,encoding='utf-8')
    sources=result['source_bindings'].copy()
    for folder in [ROOT/'fp_mechanism_audit',ROOT/'comparison_audit',ROOT/'training_audit',ROOT/'independent_verification',delivery]:
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:sources[str(path.resolve())]=b.sha(path)
    for path in [Path(__file__),result_path,ROOT/'protocol_BEFORE_ANY_FIT.json',ROOT/'preflight_tests.xml']:
        sources[str(path.resolve())]=b.sha(path)
    b.verify(sources)
    b.write_new(delivery/'manual_request_verified_fixed_challenge_failed.json',dict(verified_at=b.now(),
        user_manual_request=True,platform_goal_status_last_observed='usageLimited',platform_goal_changed=False,
        root_new_fits=64,inner_HGB_fits=48,Platt_fits=16,old_final_model_refits=0,
        complete_paired_outputs=30216,original_7_head_outputs_preserved=26439,
        September_original_nonword=143,known=107,positive=40,unknown=36,
        September_before_TP=22,September_before_FP=24,September_after_TP=0,September_after_FP=0,
        calibrated_Recall=0.,calibrated_Precision=None,all_40_positives_missed=True,
        fixed_acceptance_pass=False,deployed=False,event_probability_validated=False,
        new_formal_forecasts=0,new_mature_outcomes=0,actual_orders=0,
        targeted_new_tests_passed=8,old_prediction_files_overwritten=False,
        source_bindings=sources))
    print(dict(complete_outputs=30216,source_bindings_verified=len(sources),screen_pass=False,new_fits=64))


if __name__=='__main__':run()
