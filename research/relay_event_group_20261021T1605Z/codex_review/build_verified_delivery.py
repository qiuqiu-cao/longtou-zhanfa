from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'validation_final'
ROOT=HERE.parents[2]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
r=read(OUT/'RESULT.json')
a=read(OUT/'CODEX_INDEPENDENT_PREDICTION_AUDIT.json')
models=read(OUT/'CODEX_SAVED_MODEL_AUDIT.json')
assert r['new_metrics']['nonword_top1']['hits']==a['new_metrics']['nonword_top1']['hits']==137
assert r['gates']==a['gates'] and models['actual_fits']==3
control=pd.read_csv(ROOT/'research/relay_joint_research_20261021T1142Z/final_prevalidation/CONTROL_FIXED3_3707.csv',dtype={'code':str},low_memory=False)
new=pd.read_csv(OUT/'ALL_PREDICTIONS_BEFORE_EVALUATION.csv',dtype={'code':str},low_memory=False)
features=pd.read_csv(HERE.parent/'qualified_final/event_group_features_9245.csv',dtype={'code':str})
def select(d):
    return d.loc[d.daily_one_word.eq(0)].sort_values(['date','score','code'],ascending=[True,False,True]).groupby('date',sort=True).head(1)
columns=['date','code','boards','target','score']
c=select(control)[columns].rename(columns={k:'control_'+k for k in columns if k!='date'})
n=select(new)[columns].rename(columns={k:'new_'+k for k in columns if k!='date'})
choices=c.merge(n,on='date',validate='one_to_one')
assert len(choices)==302
choices['changed']=choices.control_code.ne(choices.new_code)
choices['outcome_relation']='SAME_KNOWN_TARGET'
choices.loc[choices.control_target.eq(0)&choices.new_target.eq(1),'outcome_relation']='CORRECTED_PREVIOUS_MISS'
choices.loc[choices.control_target.eq(1)&choices.new_target.eq(0),'outcome_relation']='LOST_PREVIOUS_HIT'
choices.loc[choices.control_target.isna()|choices.new_target.isna(),'outcome_relation']='UNKNOWN_RETAINED'
added=[x for x in features.columns if x.startswith('event_')]
for prefix in ('control','new'):
    attached=features[['date','code']+added].rename(columns={'code':prefix+'_code',**{x:prefix+'_'+x for x in added}})
    choices=choices.merge(attached,on=['date',prefix+'_code'],validate='one_to_one')
choices.to_csv(OUT/'ALL302_NONWORD_TOP1_CHOICES_AND_CONTEXT.csv',index=False,encoding='utf-8-sig')
changed=choices.loc[choices.changed]
changed.to_csv(OUT/'ALL_CHANGED_NONWORD_TOP1_DATES.csv',index=False,encoding='utf-8-sig')
counts={str(k):int(v) for k,v in changed.outcome_relation.value_counts().items()}
lines=[
'# 连板事件固定群体：Codex核验结果', '',
'实际Gemini已接受准确冻结协议并完成3次拟合；Codex独立重算全部预测、重新加载3个模型复推，3707条分数最大误差小于1e-12，原绑定源未变。首板继续暂停。', '',
'本轮新增7项事件群体输入：是否匹配、固定规模、是否早期成员、未续封率、重封修复率、锚当前状态及候选成交额份额变化。历史名称只是关系代理，未认证完整真实题材、龙头身份或资金因果。', '',
'| 完整日期指标 | 同算法对照 | 新方案 |',
'|---|---:|---:|',
'| 非一字每天Top1 | 134/302（44.37%） | 137/302（45.36%） |',
'| 全池每天Top1 | 212/302（70.20%） | 211/302（69.87%） |',
'| 非一字Top2命中槽位 | 250/604 | 247/604 |',
'| 非一字Top3命中槽位 | 350/906 | 356/906 |',
'| 非一字Top3至少命中一只的日期数 | 225/302 | 225/302 |', '',
'非一字Top1净增3天、约0.99个百分点；5日块配对95%区间为[-0.99,+2.98]个百分点，包含零，未证明稳定提升。完整302日保留，所有候选未知结果67只次；Top1选中4条未知另列，未从日期分母剔除，也未当成已证实失败。不是盈利胜率。', '',
'| 分段非一字Top1 | 对照 | 新方案 |',
'|---|---:|---:|',
'| 2025年下半年 | 59/126 | 60/126 |',
'| 2026年1—8月 | 67/160 | 69/160 |',
'| 2026年9月 | 8/16（3未知） | 8/16（3未知） |', '',
'整体未通过冻结验收：未到140天、净增未到6天、区间下界未转正；Top3日期覆盖225低于233门；9月Recall50%→45%、F1 48.78%→45.57%，超过原保护限。各段Top1均无倒退仍不足以判升级。', '',
f'全部302日原选择/新选择/7项上下文已保存；其中{len(changed)}日换股，结果分类为{json.dumps(counts,ensure_ascii=False)}。完整变化表包含改善、损失、同结果与未知，不只展示成功例子。这些是配对误差描述，不能据此断言某字段或某资金导致成功。', '',
'处置：本臂封存，不重复拟合、改词典/窗口/阈值/权重救结果；不进入月度，不替换生产BASE，不改旧预测、账本、交易和定时任务。历史已见，不是新盲测。', '',
'下一研究应先取得与简单同名不同的、可按历史日期构造的真实题材关联或封板承接新证据，并对同周期全部成功/失败成员进行核验，再与实际Gemini讨论冻结。不能把这轮的137天当成新生产基准，也不能直接在这302日调参。', '',
'依据：RESULT.json、CODEX_INDEPENDENT_PREDICTION_AUDIT.json、CODEX_SAVED_MODEL_AUDIT.json、冻结协议、三模型及提前保存预测。', '',
f'核验交付时间：{datetime.now(timezone.utc).isoformat()}。']
with (OUT/'CODEX_VERIFIED_RESULT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
with (OUT/'CODEX_DELIVERY_RECEIPT.json').open('x',encoding='utf-8') as f:
    json.dump(dict(at_utc=datetime.now(timezone.utc).isoformat(),actual_fits=3,codex_audit_fits=0,
        nonword_top1_new=137,nonword_top1_control=134,dates=302,unknown_candidates=67,
        changed_dates=len(changed),changed_outcomes=counts,all_gates_passed=False,
        monthly_fits_authorized=False,production_deployed=False,first_board='PAUSED'),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(changed_dates=len(changed),changed_outcomes=counts,result='NOT_ACCEPTED_FOR_UPGRADE'),ensure_ascii=True))
