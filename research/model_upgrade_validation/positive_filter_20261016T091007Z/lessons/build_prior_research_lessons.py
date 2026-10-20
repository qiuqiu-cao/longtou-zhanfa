"""Summarize immutable prior experiments. No model fitting or network calls."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
from relay_local_recovery import verify_sources

B=REPO/'research/model_upgrade_validation'
UNKNOWN='unknown'
OLD_SCOPE='原固定全池3707=1584/1941/182；已知3640=1584/1923/133；未知67=0/18/49。早两段真实一字身份不全，proxy0不是非一字。'
NEW_SCOPE='原冻结B三段2980=1227/1610/143；已知2932=1227/1598/107；正例871=422/409/40；未知48=0/12/36。每月完整252成熟信号日。'
SEP='9月B143/已知107/正40/负67/未知36；先完整池选后查结果。'
GATE_OLD='原各段全池Accuracy至少+3pp及各自Recall/Precision/F1/覆盖/真实非一字首选保护；历史通过仍须原120官方日升级门。见原协议精确文本，不以本批5pp追溯改判。'
SOURCES=set()
ROWS=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def attach(paths):
    out=[]
    for name in paths:
        p=B/name
        if not p.is_file():raise FileNotFoundError(p)
        SOURCES.add(p);out.append(str(p.resolve()))
    return out

def add(id,name,fit,success,saved,refit,reuse,scope,identity,gate,failure,lesson,paths,metrics='',fit_note='',gate_keys=None):
    evidence=attach(paths)
    exact={}
    for p in evidence:
        f=Path(p)
        if 'protocol' in f.name and f.suffix=='.json':
            d=load(f)
            keys=gate_keys or ['historical_screen','historical_gate','fixed_screen','screen','fixed_acceptance','acceptance','acceptance_corrected','acceptance_cost','production_upgrade']
            exact[str(f)]={k:d[k] for k in keys if k in d}
    ROWS.append(dict(study_id=id,study=name,actual_fit_invocations=fit,successful_new_estimator_fits=success,
        persistent_model_artifacts_or_bundles=saved,old_production_BASE_refits=refit,reused_final_or_prior_models=reuse,
        fit_count_evidence_and_limits=fit_note,sample_denominator=scope,source_and_identity_scope=identity,
        original_acceptance_gate=gate,original_gate_exact_JSON=json.dumps(exact,ensure_ascii=False),
        original_verdict='未通过/未认证；原结论保留',failure_or_qualification_limit=failure,
        reusable_lesson=lesson,selected_original_result_with_scope=metrics,evidence_files=' | '.join(evidence)))

def main():
    HERE.mkdir(parents=True,exist_ok=True)
    for name in ['历轮研究总表_各自口径与原验收门.csv','历轮经验总结_实施前.md','lessons_source_bindings.json']:
        if (HERE/name).exists():raise FileExistsError('immutable lessons already generated')
    add('BASE_ORIGIN','原BASE与冻结三臂建立',UNKNOWN,UNKNOWN,'6冻结模型（市场3/个股3），来源为比较缓存',UNKNOWN,'9月比较缓存复用，无再次冻结拟合',OLD_SCOPE,
        '29字段Logistic、原inverse-day；低换手signal_one_word为代理；原场景概率未校准。',
        '原真实非一字首选主门、完整池保护及120日前向；基准不是本批新挑战。',
        '原非一字漏判明显，历史Accuracy不能等于当日真实晋级概率；初始全过程fit次数无足够收据统一计数，记unknown。',
        '原模型保留；原始分数×100只作显示；同身份同标签同池对照，不把市场三分类与个股晋级Accuracy混合。',
        ['fixed_20261007T112017Z/固定模型研究与升级实施.md','fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json','fixed_20261007T112017Z/models/relay/BASE/manifest.json'],
        '原个股Accuracy三段64.71/71.97/61.65%；9月真实B6TP/4FP Recall15%，主首选7/13。',
        '文档确认最终6冻结模型复用缓存；不把目录STARTED或重放次数当fit总数。')
    add('R1_CLASS_BALANCE','旧29字段损失类别平衡R1',3,3,'0可部署权重；内存roundtrip',0,'原BASE输出复用',OLD_SCOPE,
        '仅训练损失按原日权重平衡两类，成员/29字段/Logistic C=.25/严格.5不变。',GATE_OLD,
        '各段Accuracy未+3pp；1—8月多92TP同时多190FP，9月多18TP同时多24FP。',
        '多报正向带来Recall和FP一起增长；只看Recall不足，C与权重质量尺度要冻结。',
        ['relay_balance_20261009T044001Z/固定比较与误判复核.md','relay_balance_20261009T044001Z/protocol_BEFORE_EFFECTS.json','relay_balance_20261009T044001Z/report.json'],
        'Accuracy BASE→R1 64.71→64.96/71.97→66.87/61.65→57.14%。')
    add('R2_FOREST','旧29字段固定随机森林R2',3,3,'0可部署权重；内存roundtrip',0,'原BASE输出复用',OLD_SCOPE,
        '同原成员/日权重，仅固定160树depth5/minleaf15森林，无类别重权。',GATE_OLD,
        'Accuracy各段未达+3pp，后两段Recall/F1下降，真实B首选未升。',
        '换非线性算法不自动补齐题材或同位区分，不能把评分计算解释当因果。',
        ['relay_error_20261009T045526Z/两只漏判与模型改进核验.md','relay_error_20261009T045526Z/protocol_BEFORE_EFFECTS.json','relay_error_20261009T045526Z/comparison.json'],
        'R2 Accuracy65.91/73.63/61.65%；9月真实B首选7/13。')
    add('R3_TRANSITION','旧一字代理转换手时序R3',3,3,'0可部署权重',0,'原BASE输出复用',OLD_SCOPE,
        '新增精确官方前日proxy及转换手proxy，仍是代理；缺前日或板位不连贯NaN。',GATE_OLD,
        'Accuracy不升/下降，Precision下降；9月首选8/13不能覆盖全池退化。',
        '代理转换不是实际一字身份，更不等于角色死亡；相邻官方日与未知必须保留。',
        ['relay_phase_20261009T095400Z/兼容修复与转换手特征检验.md','relay_phase_20261009T095400Z/protocol_BEFORE_EFFECTS.json','relay_phase_20261009T095400Z/report.json'],
        'Accuracy64.71/71.66/60.90%；三段代理变化字段覆盖652/638/64。')
    add('E1_E2_CONTEXT','情绪环境及线性交互E1/E2',6,6,'6研究估计器',0,'原BASE输出复用',OLD_SCOPE,
        '原29+7连续情绪及14交互；三个训练窗4062/3243/3059，同252日与原权重。',GATE_OLD,
        '无段Accuracy+3pp，9月两方案P/R/F1下降。',
        '增加共享市场背景不能证明能区分同日股票；来源阶段缺失不由结果填。',
        ['relay_context_20261011T151204Z/情绪环境重评与固定比较.md','relay_context_20261011T151204Z/protocol_BEFORE_FITS.json','relay_context_20261011T151204Z/completion_receipt.json'],
        '9月BASE19TP/15FP，E1 16/14、E2 12/12；此为全池55正分母，不是40正B。')
    add('H1_H2_EXPERT','旧梯度提升与proxy分支专家H1/H2',12,12,'6模型包（H1每段1；H2每段含全组+2分支）',0,'原BASE输出复用',OLD_SCOPE,
        'H1三次+H2九次底层boosters；旧proxy0/1不认证真实一字，分支权重来自完整原组。',GATE_OLD,
        '9月真实B Recall15→2.5/0%，全池亦退化；各段未达门。',
        '模型包数不等于底层fit数；分支资格、样本权重和真实身份不得偷换。',
        ['relay_expert_20261011T162713Z/非一字评分结构检验与升级结论.md','relay_expert_20261011T162713Z/protocol_BEFORE_FITS.json','relay_expert_20261011T162713Z/completion_receipt.json'],
        'H1/H2 9月全池Accuracy60.90%；H2首选8/13不抵Recall0%。')
    add('P1_DROP_PROXY','删除旧一字代理P1',3,3,'0可部署权重；内存roundtrip',0,'原BASE输出复用',OLD_SCOPE,
        '仅删除signal_one_word，28字段与原成员/每日权重/Logistic C/.5不变。',GATE_OLD,
        '早两段未过Accuracy+3pp且Precision下降；9月局部改善不启用。',
        '可疑代理依赖并不证明删除即稳定改善；不用未来新高分修补旧漏判。',
        ['jiuyang_shidai_20261012T072934Z/重训与九阳时代经验结算.md','jiuyang_shidai_20261012T072934Z/retrain/protocol_BEFORE_FITS.json','jiuyang_shidai_20261012T072934Z/retrain/report.json'],
        '9月B6TP→9TP Recall15→22.5%；全池Accuracy66.17%。')
    add('SOURCE_SHAPE_QUALIFICATION','原代理语义与OHLC来源资格审查',0,0,0,0,'仅原始来源重算，不推断或训练',
        '完整9245；9199可观察、46冲突未知；训练形态4034/3233/3048。',
        '2878沿原raw资格，6321只有标准化不复权声明；严格原时点0；不是独立角色/真实承接。',
        '仅来源资格审查，无预测改善门；后来单独冻结价型挑战。',
        '可观察覆盖不等于当时已取得或训练角色金标准，缺真实HTTP/时钟不能补造。',
        '分别记事实覆盖、价格一致、PIT、角色真值和训练许可。',
        ['early_signal_ohlc_20261013T1303Z/更早信号日价型资格与固定假设.md','early_signal_ohlc_20261013T1303Z/completion_receipt.json','signal_ohlc_20261013T1257Z/completion_receipt.json'])
    add('PROXY_EVIDENCE_UNKNOWN','551条旧代理不足证据记未知',3,3,'0可部署权重',0,'原BASE输出复用',OLD_SCOPE,
        '仅551条旧signal_one_word设NaN，其他成员/字段/标签/252/C/.5相同。',GATE_OLD,
        'Accuracy增益0/.31/0pp；Precision下降，整体淘汰。',
        '语义修正是合理数据工作，但效果仍需同池比较。',
        ['prediction_delivery_20261013T123946Z/预测能力固定比较交付.md','prediction_delivery_20261013T123946Z/completion_receipt.json','relay_semantics_20261013T115031Z/protocol_PROXY_EVIDENCE_BEFORE_ANY_FIT.json'],
        '9月B TP6→7、FP4→5，P60→58.33%。')
    add('SIGNAL_PRICE_SHAPE','信号日OHLC价型替换固定尝试',2,2,'0持久部署权重；仅首段预测可验',0,'BASE复用；9月未fit',OLD_SCOPE,
        '只有当前信号OHLC价型替旧proxy，46矛盾NaN；来源诊断资格不等于PIT。',GATE_OLD+'另价型非单一价格组Recall/F1不退。',
        '首段+0.51pp不足且非一字Recall下降；第二段fit后NaN比较报错，未提前写预测导致结果丢失；9月停。',
        'attempt/完成fit/可验预测分开；预测先落盘再校验；NaN双方缺失不能误判标签变动。',
        ['price_shape_compare_20261013T132007Z/信号日价型固定尝试与失败修复.md','price_shape_compare_20261013T132007Z/completion_receipt.json','early_signal_ohlc_20261013T1303Z/protocol_SIGNAL_PRICE_SHAPE_BEFORE_ANY_FIT.json'],
        '首段价型非单一组Recall5.21→3.08%；第二段指标unknown；9月未拟合。',
        '2次fit确已完成，不等于三段完成；后续仅技术修复，未重fit。')
    add('RECENCY','月度ROLL252与HL63/HL126衰减',45,45,'本批持久joblib模型0；45份预测/checkpoint',0,'新公平ROLL252拟合15，不是原生产BASE重fit',OLD_SCOPE,
        '原9245/9178已知；29字段月初前成熟同252成员；衰减后权重总量252。',
        '原每段Accuracy+3pp及Recall/P/F1/覆盖/proxy0和9月真实B/配对块区间门。',
        '两衰减9月Recall及Accuracy退步，均失败；滚动基准略升Accuracy伴随Recall/F1下降。',
        '时间适应不是最近数据越重越好；同月不能读月内结果，正则尺度固定。',
        ['recency_walkforward_20261013T134351Z/历史数据与情绪变化固定比较.md','recency_walkforward_20261013T134351Z/protocol_BEFORE_ANY_FIT.json','recency_walkforward_20261013T134351Z/report.json'],
        '9月B Recall ROLL15/HL63 5/HL126 10%；45=基准15+挑战30。',
        '原源码保存预测，fit roundtrip在内存；不从文档“保存模型”推导45可重用持久权重。')
    add('PHASE_PROXY','7阶段+7代理交互响应',15,15,'本批持久joblib0；15预测/checkpoint',0,'ROLL252只读预测复用，不重fit',OLD_SCOPE,
        '29+14字段，阶段为原算法输出晚到附层，9月原UNKNOWN不回写。',
        '原每段Accuracy+3pp及P/R/F1/覆盖/真实B与共同区间门。',
        '前两段P/R/F1退化；9月完全不变。',
        '阶段频率变化与逐股可分性不同；算法阶段不是独立角色/情绪真值。',
        ['phase_response_20261013T1649Z/历史阶段响应固定比较与结果.md','phase_response_20261013T1649Z/protocol_BEFORE_ANY_FIT.json','phase_response_20261013T1649Z/report.json'],
        'Accuracy65.59→65.21/72.49→71.76/61.65→61.65%；B6TP4FP不变。')
    add('ST_B_DUAL','全周期ST_B_ONLY与ST_DUAL',33,32,'32持久模型',0,'原BASE及ROLL输出复用',NEW_SCOPE,
        '9245/684日全规则决策，历史题材/行业0；OHLC真实形态与旧proxy分开；当前池幸存者偏差。',
        '新非一字Recall/F1/Top1/Top2/Precision限退5pp及执行≥10/50%覆盖、收益/风险/角色真值；已取消旧全池Accuracy+3pp否决。',
        'Recall上升但误报多；跨期及执行证据门未达，收益B+2仅4—5场景、独立角色真值0。',
        '规则100%有输出不等于身份准确；B专用更积极并不等于Precision改善；真实执行另层。',
        ['short_term_thinking_20261015T071000Z/delivery/全周期短线思维重建与模型验证交付_定稿.md','short_term_thinking_20261015T071000Z/protocol_BEFORE_ANY_FIT.json','short_term_thinking_20261015T071000Z/completion_final.json'],
        '9月ST_B_ONLY27TP38FP，DUAL20TP22FP；原BASE6TP4FP。',
        '33次fit调用=1次优化前全空列分箱失败+32成功保存；修复时0已完成模型，不按效果重试。')
    add('SYMBOLIC','17字段名称/字辈扩散',16,16,'16持久模型',0,'ST_DUAL16月及BASE复用',NEW_SCOPE,
        '历史日期名称/原分类代理，9月名称超主表边界降来源等级；独立题材及角色真值仍缺。',
        '各历史段B Recall/P/F1及Top1/2不降；9月P≥55%、R≥50%、P或Top1严格增。',
        '1—8月及9月Recall/P/F1退化，9月Top1局部增加不抵Top2下降。',
        '名字关系只作弱代理，不能被重新包装成真实题材或资金因果。',
        ['historical_symbolic_context_20261015T081200Z/delivery_final/历史短线补层与10月15日续接交付.md','historical_symbolic_context_20261015T081200Z/protocol_BEFORE_ANY_FIT.json','historical_symbolic_context_20261015T081200Z/completion_final.json'],
        '9月19TP25FP，R47.5/P43.18；Top1 9/13，Top2 16/26。')
    add('EXACT_FAILED_ATTEMPTS','精确炸板分类恢复362条',16,16,'16持久模型',0,'旧BASE/ST_DUAL复用',NEW_SCOPE,
        'V21558772原行，合格55646→56008，362条触及精确价后收低一分恢复；684候选上下文变。',
        '同字辈批B/排名保护及9月P/R门，原协议screen保持。',
        '早段微改善，1—8月和9月退化；资料纠错不等于预测升级。',
        '保留数据更正与失败模型；Decimal精确价、相邻交易日、失败尝试分母均须同码。',
        ['exact_attempt_response_20261015T100800Z/delivery_final/更正炸板输入固定比较与完整交付.md','exact_attempt_response_20261015T100800Z/protocol_BEFORE_ANY_FIT.json','exact_attempt_response_20261015T100800Z/completion_final.json'],
        '9月19TP23FP，对ST_DUAL20TP22FP；Top1 8/13、Top2 16/26。',
        '目录额外experiment_STARTED不是额外fit；每月1次共16。')
    add('LAGGED_PRICE','滞后20日价格联动12字段',16,16,'16持久模型',0,'原BASE/DUAL/更正炸板等复用',NEW_SCOPE,
        'T-25至T-6，20日≥12共同观测，1465910配对；正相关平方连续权，不能认证主题。',
        '对更正炸板各段B Recall/P/F1和Top1/2不降；9月R≥50/P≥55；执行身份另验。',
        '9月FP少但Recall降、Top2降；前两段Recall/F1低于参考。',
        '统计联动是可复用独立来源，不是父子身份；相邻收盘不能跨缺日，少FP必须同看少TP。',
        ['lagged_price_relation_20261015T103000Z/delivery_final/历史价格联动证据与固定比较交付.md','lagged_price_relation_20261015T103000Z/protocol_BEFORE_ANY_FIT.json','lagged_price_relation_20261015T103000Z/completion_final.json'],
        '9月17TP12FP，R42.5/P58.62；DUAL20/22；当前6有2股不足12观测。')
    add('CAUSAL_SCOPE','完整池/父锚/路径语义更正与回放',0,0,0,0,'原全部保存模型仅重放',
        '完整9315=9245历史+70十月；完整近期81、74补充结果/7未知另表。',
        '绑定原全前池范围、严格连续路径、来源unknown；原冻结身份与最新引擎身份分栏。',
        '工程来源与历史/当前API同码校验，不自称效果升级。',
        '早期上下文语义不完整；修复仍不产生角色金标准或新晋级标签。',
        '先把事实与来源正确传播，再训练；已见模型/结果不能倒填盲审。',
        ['causal_scope_alignment_20261015T121000Z/delivery_final/全周期短线研究_修复比较与验收.md','causal_scope_alignment_20261015T121000Z/completion_verified.json'])
    add('CORRECTED92','92字段CORRECTED_CAUSAL重训',16,16,'16持久模型',0,'原BASE/ST_DUAL/LAG复用',NEW_SCOPE+' 外月完整3777/头。',
        '完整252日训练，旧proxy/两证据不足换手列不入；规则身份不当金标准；35条9月当前价型资格未知。',
        '原pooled F1≥DUAL、各段Recall损失≤5pp、pooled Top1不低。此5pp是该批原门，非本批追改。',
        '跨期F1/Recall/Top1门失败；9月方向弃判导致15TP19FP，原raw21/24另诊断不回写。',
        '区分raw score、资格路由和学习权重；不能把18条身份口径差异缩分母。',
        ['expectation_345_20261015T163226Z/delivery/第3至5项优化_独立评测重训与误报验收.md','expectation_345_20261015T163226Z/protocol_BEFORE_ANY_FIT.json','expectation_345_20261015T163226Z/completion_345_verified.json'],
        '历史原资格218TP298FP，R25.03/F131.43%；9月15TP19FP R37.5。')
    add('COST1P5','92字段负类成本1.5',16,16,'16持久模型',0,'更正头16及BASE等复用',NEW_SCOPE,
        '只在相同成员/字段下改变训练负类损失质量1.5，严格.5；不认证概率。',
        '原pooled FP至少降20%、各段FP不高、各段Recall最多降10pp、P升、F1最多降2pp、TP-1.5FP提高。',
        'FP298→73伴随TP218→58，R25.03→6.66%，F1明显退化。',
        '成本惩罚可关闭大多数正向，但不是学会区分；不能用少FP掩盖漏报。',
        ['expectation_345_20261015T163226Z/delivery/第3至5项优化_独立评测重训与误报验收.md','expectation_345_20261015T163226Z/protocol_BEFORE_ANY_FIT.json','expectation_345_20261015T163226Z/completion_345_verified.json'],
        '9月原资格5TP4FP，补来源资格后8TP5FP另层，不混归训练。')
    add('LLM_REPEAT','3个Codex审阅上下文独立输出',0,0,'243份人工形成逐股审阅，无权重fit',0,'3次×81份，非3种模型',
        '完整81；历史75、补充已知74/正28/未知1；当前6结果未知。',
        '上下文暴露旧分数/历史结果及后续信号可暗示前结果；historical_blind=false。',
        '实际完整输出/事实引用/曝光和一致性核验；效应未认证，不自动融合部署。',
        '一致64/81非准确率；多数P76%但覆盖67.57%、全已知正确率58.11%低于BASE72.97%。',
        '单日封闭材料避免未来间接泄漏；保留弃判/无记录/未知区别，不挑最佳审阅者。',
        ['expectation_345_20261015T163226Z/delivery/第3至5项优化_独立评测重训与误报验收.md','expectation_345_20261015T163226Z/LLM_evaluation/LLM_repeat_verification_and_results.json','expectation_345_20261015T163226Z/LLM_repeated_review_protocol.md'],
        '3×81=243；64全票一致、17分歧；多数TP19FP6弃判24，仅暴露历史诊断。')
    add('PAIRWISE','左/右/差276字段成对排序',16,16,'16持久排序模型',0,'原5头18885行复用',NEW_SCOPE,
        '同原B身份；left92/right92/diff92保留共享环境；仅混合正负日配对，日权重1。',
        'pooled Top3全槽+2pp、Top1不低、各段Top3已知命中降≤2pp、覆盖降≤5pp。',
        'Top1多6但Top3少15，1—8月Top3退化；无绝对事件分类头。',
        '相对偏好不是绝对涨停概率，不能偏好>.5转PROMOTE；单成员/未知/并列保留。',
        ['ranking_optimization_20261015T173045Z/delivery/重新调试_新旧模型与评分系统完整交付.md','ranking_optimization_20261015T173045Z/protocol_BEFORE_ANY_FIT.json','ranking_optimization_20261015T173045Z/fixed_comparison/ranking_validation_results.json'],
        '历史Top1 DUAL130→136/302；Top3 370→355/906；TP/FP/R/P/F1不适用。')
    add('SOURCE_RECOVERY47','当前价型47条已存原文资格补全',0,0,0,0,'旧保存分数不重fit',NEW_SCOPE,
        '原244资格缺口恢复47、197仍未知/不合格；9月35当前价型恢复；原时钟晚到。',
        '来源身份/日期/字段指针/哈希/相邻官方日/正成交资格门，非预测升级门。',
        '资料恢复同时解锁正确与错误方向；不能把当前价型资格扩张为完整路径或历史PIT。',
        '资格变化6TP5FP与训练1TP分开；原生量单位只验正成交，不混入量比。',
        ['ranking_optimization_20261015T173045Z/delivery/重新调试_新旧模型与评分系统完整交付.md','ranking_optimization_20261015T173045Z/source_repair/verified_22662_final/source_qualification_repair_receipt.json','ranking_optimization_20261015T173045Z/source_repair/verified_22662_final/final_semantic_verification.json'],
        '9月CORRECTED15TP19FP→21TP24FP，R37.5→52.5；35中21已知14未知。')
    add('SOURCE_FEATURE47','单字段causal_price_shape补值重学',9,9,'9新模型+7精确原模型复用',0,7,NEW_SCOPE,
        '仅47原NaN→核实价型，其余91字段/元数据/标签/成员完全相同；A路由对齐同资料原头。',
        'pooled B F1至少+1pp、各段Recall和P损失≤2.5pp、pooled Top3全槽不降。',
        'F1只+0.12pp且Top3少2；原24FP一个没消除，只多1TP。原9一字路由差异附层纠正无重fit。',
        '同来源资格对照才可归因重训；原始失败与实际可调用推断包装都保留。',
        ['ranking_optimization_20261015T173045Z/source_feature_training/来源特征固定比较_最终交付与可调用入口.md','ranking_optimization_20261015T173045Z/source_feature_training/protocol_BEFORE_ANY_FIT.json','ranking_optimization_20261015T173045Z/source_feature_training/same_source_route_alignment/same_source_aligned_fixed_result.json'],
        '历史同资格224TP303FP→225TP303FP，F132.05→32.17%；9月21/24→22/24。')
    add('PLATT_FIXED_HALF','时间OOF Platt固定.5',64,64,'48inner HGB+16校准映射',0,'16最终SOURCE头精确复用',NEW_SCOPE,
        '每月同252成熟窗，3×63后验块模型训练62/125/188日；原B已知OOF校准、日权重1无类平衡。',
        'pooled P+3pp、FP≤.9倍、F1不低；每历史段FP不高、Recall最多降2.5pp；Top-k逐股不变。',
        '所有原B正向关闭，各段0TP/0FP、Recall0、P未定义、F10，整体失败。',
        '校准不增加同日区分；0误报不等于有用；保留可复用严格时序缓存而不复跑失败映射。',
        ['false_positive_20261016T030307Z/delivery/误报原因与进一步优化实测.md','false_positive_20261016T030307Z/protocol_BEFORE_ANY_FIT.json','false_positive_20261016T030307Z/execution/calibration_fixed_result.json'],
        '48inner+16Logistic，16映射正斜率/n_iter5—7；9月最大映射分0.468648，固定.5无正向。')
    add('CAPACITY_THRESHOLD','固定容量2组合＋五阈值训练期选择',107,107,'96inner+11选中final；5final复用',0,'48原inner缓存、5选BASE final',NEW_SCOPE,
        '仅depth/leaf/minleaf/L2四参数组合变化；OOF选择精确Fraction；阈值护栏对原BASE防损失叠加。',
        'pooled P+3pp、FP至少-10%、F1不低；各段Recall最多-2.5pp/F1不低/FP不高；pooled Top3不降。',
        '固定.5只少1FP、多1TP，1—8月FP增；选低阈值虽增Recall却FP翻倍；Top3退化。',
        '扩大容量改善训练、39/48块OOF LogLoss变差；不能据局部结果再改候选；两策略同raw不是两批模型。',
        ['iteration_research_20261016T043944Z/delivery/参数研究与继续验证完整交付.md','parameter_research_20261016T032229Z/protocol_BEFORE_ANY_FIT.json','parameter_research_20261016T032229Z/execution/parameter_fixed_result.json','parameter_research_20261016T032229Z/training_diagnostics/参数容量_训练与OOF诊断.md'],
        '历史225/303→226/302或382/629；9月22/24→22/24或26/31；Top3 364→359/906。')
    add('PREFIX_ITERATION','保存原容量30/60/120轮前缀选择',0,0,'0新模型；复用48inner+16final=64模型前缀',0,'64原模型仅staged推断',NEW_SCOPE,
        '单改使用轮数，保持原容量/字段/权重/.5；OOF精确选择先落盘，再推断外月。',
        'pooled P+3pp、FP至少-10%、F1不低；各段Recall最多-2.5pp/F1不低/FP不高；pooled Top3不降。',
        '少2FP同时少3TP，P/F1退化；Top-k局部改善不能覆盖分类门失败。',
        '读取已保存模型前缀不是重新fit；训练轮数有取舍，不等于所有参数均无效。',
        ['iteration_research_20261016T043944Z/delivery/参数研究与继续验证完整交付.md','iteration_research_20261016T043944Z/protocol_BEFORE_ANY_EFFECT.json','iteration_research_20261016T043944Z/execution/iteration_fixed_result.json','iteration_research_20261016T043944Z/verification/primary_agent_second_implementation_verification.json'],
        '3月选60轮、13月120轮、0月30轮；历史222TP301FP F131.85；Top3 366/906。')

    recovery=REPO/'research/local_repair_20261016T051940Z/recovery_manifest.json'
    combined=B/'iteration_research_20261016T043944Z/delivery/combined_parameter_delivery_verified_NO_UPGRADE.json'
    verification=verify_sources(load(combined)['source_bindings'],recovery)
    assert verification['exact_original_sources_verified']==2533 and verification['historical_test_records_unavailable']==19 and verification['model_data_code_hash_exceptions']==0
    SOURCES.update([Path(__file__),REPO/'relay_local_recovery.py',recovery,combined,REPO/'AGENTS.md'])
    for name in ['relay_temporal_score_calibration.py','relay_parameter_research.py','expectation_345_features.py','relay_source_feature_research.py','leader_model_stack_audit.py','relay_promotion_regime_gate_v258.py']:
        SOURCES.add(REPO/name)
    # Keep exact legacy model/data/code hashes; only the explicit19 missing test
    # XML exceptions may pass the unchanged recovery adapter.
    bound=load(combined)['source_bindings']
    for p in SOURCES:
        expected=bound.get(str(p.resolve()))
        if expected is not None and sha(p)!=expected:raise ValueError('prior exact-bound evidence drift: '+str(p))
    csv_path=HERE/'历轮研究总表_各自口径与原验收门.csv'
    with csv_path.open('x',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(ROWS[0]));writer.writeheader();writer.writerows(ROWS)
    (HERE/'all_study_records_with_exact_original_gates.json').write_text(json.dumps(ROWS,ensure_ascii=False,indent=2),encoding='utf-8')
    # The narrative is supplied separately in this same authorized directory.
    scope=dict(formed_at_UTC=datetime.now(timezone.utc).isoformat(),study_rows=len(ROWS),summary_new_fits=0,no_network=True,
        no_oct16_market_results_read=True,old_predictions_models_and_goal_unchanged=True,
        no_original_gate_reinterpreted_with_new5pp=True,not_cross_scope_leaderboard=True,
        current_artifact_count_is_not_original_fit_count=True,unknown_fit_counts_marked_unknown=True,
        original_source_verification=verification,original_bytes_of19_test_XML_not_recovered=True,
        models_data_code_hash_exceptions=0,whole_legacy_chains_not_all_reexecuted='Summary binds source reports/protocols; exact latest2552-binding chain is separately verified via unchanged explicit recovery. No claim that every earlier dependency chain was reexecuted.',
        source_bindings={str(p.resolve()):sha(p) for p in sorted(SOURCES,key=str)},
        protected_runtime_binding_receipt=str(combined.resolve()),protected_runtime_bindings=bound,
        output_bindings={str(p.resolve()):sha(p) for p in [csv_path,HERE/'all_study_records_with_exact_original_gates.json']})
    (HERE/'lessons_source_bindings.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(formed_at=scope['formed_at_UTC'],rows=len(ROWS),summary_new_fits=0,source_files=len(SOURCES),verification=verification),ensure_ascii=False))

if __name__=='__main__':main()
