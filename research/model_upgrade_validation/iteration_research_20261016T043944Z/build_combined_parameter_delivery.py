"""Seal capacity, threshold and saved-prefix evidence without more fitting."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO))
import short_term_thinking_dataset_builder as b

PARAM=REPO/'research/model_upgrade_validation/parameter_research_20261016T032229Z'
OLD='SOURCE_PATCHED_FEATURE_CLASSIFIER'
CAP='PARAM_CAPACITY_OOF_FIXED_HALF'
THR='PARAM_CAPACITY_OOF_SELECTED_THRESHOLD'
ROUND='PARAM_BASE_OOF_SELECTED_BOOSTING_PREFIX'


def run():
    delivery=ROOT/'delivery'
    if delivery.exists():raise FileExistsError('immutable delivery already exists')
    receipts=[
        PARAM/'protocol_BEFORE_ANY_FIT.json',PARAM/'execution/parameter_fixed_result.json',
        PARAM/'independent_verification/parameter_study_independent_receipt.json',
        PARAM/'independent_verification/completion_receipt_timestamp_semantics.json',
        PARAM/'training_audit/executed_numeric_qualification/independent_parameter_numeric_qualification_receipt.json',
        PARAM/'training_diagnostics/saved_training_diagnostic_receipt.json',
        ROOT/'protocol_BEFORE_ANY_EFFECT.json',ROOT/'execution/iteration_fixed_result.json',
        ROOT/'verification/primary_agent_second_implementation_verification.json',
    ]
    bindings={}
    for path in receipts:
        data=b.read(path);sources=data.get('source_bindings',{})|data.get('output_bindings',{})
        b.verify(sources)
        for key,value in sources.items():
            if key in bindings and bindings[key]!=value:raise ValueError('conflicting source versions: '+key)
            bindings[key]=value
        bindings[str(path.resolve())]=b.sha(path)
    capacity=b.read(PARAM/'execution/parameter_fixed_result.json')
    iteration=b.read(ROOT/'execution/iteration_fixed_result.json')
    verification=b.read(ROOT/'verification/primary_agent_second_implementation_verification.json')
    numeric=b.read(PARAM/'training_audit/executed_numeric_qualification/independent_parameter_numeric_qualification_receipt.json')
    assert capacity['inner_new_fits']==96 and capacity['selected_final_new_fits']==11 and capacity['total_new_fits']==107
    assert all(not item['fixed_screen_pass'] for item in capacity['arms'].values())
    assert iteration['new_fits']==0 and not iteration['fixed_screen_pass']
    assert verification['passed'] and verification['original37770_all_columns_unchanged']
    assert numeric['independent_numeric_qualification_pass']
    tests_path=ROOT/'preflight_tests.xml';tree=ET.parse(tests_path).getroot()
    suites=list(tree.iter('testsuite'));tests=sum(int(x.get('tests',0)) for x in suites)
    assert tests==13 and not any(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites)
    cap_metrics=pd.read_csv(PARAM/'execution/same_pool_full_classification_metrics.csv')
    round_metrics=pd.read_csv(ROOT/'execution/same_pool_classification_metrics.csv')
    combined=pd.concat([cap_metrics,round_metrics[round_metrics.arm.eq(ROUND)]],ignore_index=True)
    cap_ranks=pd.read_csv(PARAM/'execution/full_slot_Top1Top2Top3.csv')
    round_ranks=pd.read_csv(ROOT/'execution/full_slot_Top1Top2Top3.csv')
    ranks=pd.concat([cap_ranks,round_ranks[round_ranks.arm.eq(ROUND)]],ignore_index=True)
    pooled=combined[combined.period.eq('HISTORICAL_POOLED')&combined.scope.eq('ORIGINAL_NONWORD')].set_index('arm')
    sep=combined[combined.period.eq('2026Sep')&combined.scope.eq('ORIGINAL_NONWORD')].set_index('arm')
    select=pd.read_csv(PARAM/'execution/all16_training_only_selections.csv')
    prefixes=pd.read_csv(ROOT/'execution/all16_training_only_prefix_selections.csv')
    assert select.selected_capacity.value_counts().to_dict()=={'COMPACT':7,'BASE':5,'EXPANDED':4}
    assert prefixes.selected_rounds.value_counts().to_dict()=={120:13,60:3}
    titles={OLD:'原来源特征研究基准',CAP:'容量选择＋固定0.5',THR:'容量选择＋训练期选阈值',ROUND:'原容量＋训练期选30/60/120轮'}
    def percent(x):return f'{100*x:.2f}%'
    def table(period):
        lines=[]
        for arm,title in titles.items():
            row=period.loc[arm]
            lines.append(f'| {title} | {int(row.tp)} | {int(row.fp)} | {percent(row.Recall)} | {percent(row.Precision)} | {percent(row.F1)} |')
        return '\n'.join(lines)
    rank_lines=[]
    for arm,title in titles.items():
        rows=ranks[ranks.period.eq('HISTORICAL_POOLED')&ranks.arm.eq(arm)].set_index('k')
        rank_lines.append(f'| {title} | {int(rows.loc[1,"hits"])}/302 | {int(rows.loc[2,"hits"])}/604 | {int(rows.loc[3,"hits"])}/906 |')
    root=ROOT.resolve().as_posix();param=PARAM.resolve().as_posix()
    report=f'''# 参数研究与继续验证：完整交付

本次实际研究了树容量组合、决策阈值和训练轮数。容量比较新增107次拟合（96内层＋11所选最终头），训练轮数通过已有模型的30/60/120轮前缀复用完成，新增拟合0。三项固定策略均未通过各自事前效果门，未替换旧模型。

结论是：参数会改变结果，当前比较找到了局部取舍，尚未找到稳定降低误报的设置。不能据此排除所有参数，也不能宣称这两个容量组合已经覆盖全部可能的好配置。

## 如何防止看结果选参数

每月使用原252个已成熟信号日、原9315条完整数据构造和92字段。原训练窗分为4段，每段63个信号日；后三段依次作内层验证，各段模型只用其首日之前且目标已成熟的62/125/188日训练。只在这些OOF验证输出中选配置，选择结果先落盘再推断外部月份；原48内层缓存与原最终头按SHA精确复用。

只核原冻结非一字身份，不用后来引擎身份缩分母。每个候选使用相同标签和资料资格；未知来源的正例保留为漏判。选择目标F0.5偏重Precision，同时Recall相对原基准最多下降2.5个百分点、F1最多下降1个百分点。容量与阈值两步的约束均对原基准，防止损失叠加；精确分式比较，同分保留基准。

外部历史月份没有作为代码选参输入，但这些历史已经多次见过，因此仍是事后历史研究，不是新盲测。资料真实取得时钟及幸存者等原限制保留，没有读取10月16日结果。

## 树容量与阈值实际比较

| 参数 | 原基准 | 收紧COMPACT | 放宽EXPANDED |
|---|---:|---:|---:|
| 最大深度 | 3 | 2 | 4 |
| 最多叶子 | 7 | 3 | 15 |
| 每叶最少训练样本 | 25 | 50 | 12 |
| L2 | 3 | 6 | 1.5 |

学习率0.06、120轮、随机种子、原逐日及类别权重都保持。四个参数同时变化，只能评价容量组合，不能把变化单独归因为深度或L2。模型参数对树大小、正则及训练轮数的含义参见[scikit-learn文档](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)。

16个月训练期选择为COMPACT7月、EXPANDED4月、原基准5月；只对选中的11个新配置拟合完整252日模型。阈值候选事先限定为0.4/0.45/0.5/0.55/0.6；训练期实际选择0.45有10个月、0.4有6个月，没有按外月结果调整。

2025H2＋2026年1—8月＋9月原非一字完整2980条，2932条结果已知、871正例，48未知。完整同池结果如下：

| 策略 | 抓到晋级TP | 误报FP | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
{table(pooled)}

容量选择、固定0.5只多抓1个真阳、少1个误报，F1约提高0.14个百分点，Precision只提高约0.19个百分点；未达到合并误报至少降10%和Precision至少增3个百分点，且1—8月误报增加。训练期选阈值虽多抓157个真阳，也多326个误报，未实现本次控制误报目标。

9月原非一字143条、107已知、40正例、36未知：

| 策略 | 抓到晋级TP | 误报FP | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
{table(sep)}

9月训练期容量选择保留原模型，所以容量策略22/24完全不变；阈值选择0.45后为26/31。该月65%的Recall不是总体更准的证明，Precision下降、误报增加。10月只作补充，不参与历史门；阈值0.4策略在37条已知原非一字中9TP/22FP，原基准8TP/6FP，误报增加也保留。

## 扩容为何没带来稳定改善

保存模型结构已确认容量变化实际生效：BASE/COMPACT/EXPANDED平均每棵树叶数6.80/3.00/13.19，平均使用字段数50.77/25.04/61.52。扩容不是参数没写进去。

EXPANDED的48个内层块训练内原始分LogLoss全部下降，但39个对应OOF块变差。描述均值：BASE训练/OOF为0.56255/0.64180，EXPANDED为0.49876/0.64808。这提供本配置拟合历史噪声的线索；训练与后时段的差异也含市场变化和训练样本长度变化，不能当纯因果过拟合测量。48块相互重叠，也不是48个独立检验。

COMPACT的OOF平均Precision略高、Recall略低，没有一致优势。有些角色/周期字段在这些训练块没有可学习变化，扩大树也不能创造缺失的真实题材、父锚或盘口证据。[完整结构与训练/OOF诊断]({param}/training_diagnostics/参数容量_训练与OOF诊断.md)保留全部字段覆盖、树结构和反例；没有用这些事后诊断再修改候选配置。

## 继续研究训练轮数

在上述容量差距证据后，另冻结一个单变量研究：保持原容量、权重及0.5不变，只用原48个内层和16个完整训练模型的30/60/120轮前缀。采用公共[staged_predict_proba](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier.staged_predict_proba)读取保存模型中间输出，不重训、不截改模型文件。

16个月中3个月选择60轮（2025-12、2026-01、2026-05），13个月保留120轮，没有月份选择30轮。后续历史合计TP225→222、FP303→301、F1 32.17%→31.85%；少2个误报但少抓3个真阳，Precision也略降。9月和10月均仍选120轮，旧结果不变。该研究未通过固定门。

## 排序的取舍

先从完整池选候选再查结果。Top1有4个未知槽、Top2有8个未知槽、Top3有12个未知槽；下表分母含未知槽，并没有把未知当失败标签。所有空缺、并列及完整分母保留。

| 策略 | Top1已知命中/完整槽 | Top2已知命中/完整槽 | Top3已知命中/完整槽 |
|---|---:|---:|---:|
{chr(10).join(rank_lines)}

容量策略Top1改善9次但Top3少5次；只改变阈值不会改变同模型排序。轮数策略Top1/2/3各多3/5/2个命中，但绝对事件分类变差。局部排序改善不认证误报、实际可买性或净收益改善。

## 核验、原件及剩余范围

容量批次由子代理独立重算16月精确选参、全部96新内层、48原缓存和16所选最终模型；旧8组30216条原列、BASE原值精确保留，完整10组37770条。全部107个新增HGB实际120轮、无早停。原协议、所有模型、训练期分数、选参及外月结果完整保存。

轮数批次新增第11组完整输出41547条，旧37770条原列不变。子代理触及用量限制，本轮核验由主代理另写的第二套只读实现执行：使用staged_decision_function＋expit重放64个保存模型、精确分式重算选择和全池指标；不是新的独立子代理审阅。最大分数重放差8.33e-17，模型SHA在前后核对一致。13项相关针对检查通过，没有据测试数宣称预测提升。

容量批次月度COMPLETED收据的formed_at沿用了选择形成时间，不能作为实际完成时钟；原件保留，追加语义说明将实际完成时钟记UNKNOWN，mtime仅作辅助。该元数据问题未改变模型和结果。轮数批次完成收据分别保存selection_formed_at与completed_at。

所有新分数仍未校准为可信事件概率。原10月15日每策略6只的结果仍未知、原目标字段含缺失照存，未把16日新结果用于研究；没有新增正式预测、成熟样本、交易或定时任务。旧7方案评分页和正式账本保持原件，本次失败研究只另存比较表。

**尚未检验**学习率与更长训练轮数的联合变化、更多单变量参数取值；本轮不据已见外月结果补充候选继续追分。继续研究应先明确新的训练期假设与范围，再另行冻结；不能把当前几个失败组合写成所有参数都无效，也不能在缺少证据时认定误报必然仅由参数造成。

完整结果：[11组41547条CSV]({root}/execution/all41547_predictions_BEFORE_EVALUATION.csv)；[容量独立收据]({param}/independent_verification/parameter_study_independent_receipt.json)；[轮数第二实现核验]({root}/verification/primary_agent_second_implementation_verification.json)。
'''
    delivery.mkdir()
    combined.to_csv(delivery/'complete_parameter_strategy_metrics_WITH_UNKNOWN.csv',index=False,encoding='utf-8-sig')
    ranks.to_csv(delivery/'complete_parameter_strategy_Top1Top2Top3.csv',index=False,encoding='utf-8-sig')
    report_path=delivery/'参数研究与继续验证完整交付.md';report_path.write_text(report,encoding='utf-8')
    for folder in [PARAM/'training_audit',PARAM/'training_diagnostics',PARAM/'independent_verification',ROOT/'verification',delivery]:
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:bindings[str(path.resolve())]=b.sha(path)
    for path in [Path(__file__),tests_path,ROOT/'execution/iteration_fixed_result.json',PARAM/'execution/parameter_fixed_result.json']:
        bindings[str(path.resolve())]=b.sha(path)
    b.verify(bindings)
    b.write_new(delivery/'combined_parameter_delivery_verified_NO_UPGRADE.json',dict(verified_at=b.now(),
        user_authorized_parameter_research_and_continuation=True,
        actual_capacity_inner_fits=96,actual_capacity_selected_final_fits=11,total_new_fits=107,
        prefix_research_new_fits=0,old_model_refits=0,
        capacity_fixed_half_accepted=False,capacity_selected_threshold_accepted=False,selected_prefix_accepted=False,
        complete_model_outputs=41547,candidates_per_arm=3777,arms=11,
        targeted_checks_passed=13,capacity_independent_subagent_verification=True,
        prefix_independent_subagent_verification=False,prefix_primary_agent_second_implementation_verified=True,
        probability_validated=False,new_formal_forecasts=0,new_mature_results=0,actual_trades=0,
        production_replaced=False,schedules_changed=False,platform_goal_changed=False,
        platform_goal_last_known_status='usageLimited',monthly_capacity_completion_clock='UNKNOWN',
        source_bindings=bindings))
    print(dict(new_fits=107,prefix_new_fits=0,source_bindings_verified=len(bindings),all_three_screens=False,report=str(report_path.resolve())))


if __name__=='__main__':run()
