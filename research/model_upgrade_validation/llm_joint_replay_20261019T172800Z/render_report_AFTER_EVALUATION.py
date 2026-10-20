from run_joint_replay import *
r=read(HERE/'results_AFTER_JOINT_FREEZE.json')
rows=pd.read_csv(HERE/'complete_122_revealed_all_arms.csv',dtype={'code':str,'date':str,'target_date':str})
labels={'codex_independent':'Codex独立','gemini_independent':'Gemini 3.8独立','discussed_joint':'双方讨论后共同决策','archived_ROLL252':'既存ROLL252数值参考','always_no':'恒判不晋级参考'}
def pct(v):return '未知' if v is None else f'{100*v:.2f}%'
def table(label):
    m=r['events'][label]['arms']
    out=['|决策方案|判断正确/已判断|已判断准确率|正确/全部已知结果（弃判计未答对）|判断覆盖率|预测涨停命中率Precision|实际涨停检出率Recall|F1|','|---|---|---|---|---|---|---|---|']
    for a,name in labels.items():
        v=m[a];correct=v['TP']+v['TN']
        out.append(f"|{name}|{correct}/{v['judged_known']}|{pct(v['accuracy_judged'])}|{correct}/{v['labels_known']} = {pct(v['correct_fraction_full_known'])}|{pct(v['coverage_known'])}|{pct(v['precision_known'])}|{pct(v['recall_full_known'])}|{pct(v['F1_full_known'])}|")
    return out
lines=['# Codex与真实Gemini共同决策：固定历史回放首批结果','',f"生成时间：{now()}。完整固定样本{r['days']}个信号日、{r['rows']}只次；没有新拟合、没有部署或旧账本回写。",'',
'本批是两个真实语言模型先独立判断，再在看不到答案时讨论并冻结共同结论，最后揭晓历史结果。不是BASE与ST_DUAL的数值组合。Gemini为Antigravity界面所示Gemini 3.8 Flash High；Codex为本对话实际助手。','',
'样本来自2025年下半年、2026年1—8月、2026年9月，每段按事前固定日期哈希各选4天，保留所选日期完整原归档连板池。未按历史成绩、标签或池大小挑选。日期、股票代码及简称、未来结果与数值模型分数在审阅包中遮蔽；双方独立稿和完整共同稿在揭晓前绑定哈希。','',
'这是已经研究过的历史域上的遮蔽回放，不是独立认证盲测或真实前向测试。历史题材、论坛与公告正文不足，本批实际检验价量、情绪与同位竞争判断；不能外推为完整叙事策略的长期准确率。既存来源晚取得、供应商范围及幸存者偏差保留。','',
'## 主要事件：下一相邻官方交易日收盘涨停，精确价格交叉核验','',
'原归档晋级标签须有效且目标日等于官方相邻交易日；并要求下一日不复权OHLC有效、前收与信号池价相同、按普通主板10%及分位四舍五入计算的收盘涨停一致。价格或标签缺失/冲突保留UNKNOWN，不删除原行。历史特殊状态没有被本批重新全面认证，所以这些是所列普通10%范围的交叉核验结果。',
'普通主板10%及最小价位四舍五入的规则参见[深交所交易规则（2026年修订）](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)；历史特殊状态不能用现行规则倒填。','']
lines+=table('strict_close')
v=r['events']['strict_close']['arms']['discussed_joint']
lines+=['',f"完整122只次中，精确价格交叉核验有结果{v['labels_known']}、结果未知{v['labels_unknown']}。未知结果不记正确；弃判在完整已知结果分母中记为未答对，也单列已判断准确率。Precision只在有结果的正向预测中计算；Recall保留弃判的真实正例为漏检，F1采用该Recall。",'',
'## 次级对照：原归档晋级标签','',
'下表保留原供应商历史标签的完整结算，与上面的价格交叉核验口径分栏。它不能替代严格价格核验，也不与其混合计算。','']
lines+=table('archived_promotion')
lines+=['','## 相对Top3命中与完整名次','',
'Top3是每日完整池内相对候选，不自动代表每只均作YES；盘中触板与收盘涨停分别计数。结果未知的名额仍在完整名额分母中。','',
'|方案|精确收盘Top1（命中/全部名额；未知）|精确收盘Top3|原归档晋级Top3|盘中触板Top3|','|---|---|---|---|---|']
for a in ['codex_independent','gemini_independent','discussed_joint','archived_ROLL252','height_rank_Top3']:
    name=labels.get(a,'仅高度排序参考')
    def cell(event,k):
        v=r['events'][event]['top'][a][k]
        return f"{v['hits']}/{v['slots']}；未知{v['unknown_slots']}"
    lines.append(f"|{name}|{cell('strict_close','Top1')}|{cell('strict_close','Top3')}|{cell('archived_promotion','Top3')}|{cell('touch','Top3')}|")
lines+=['','## 讨论到底纠正了什么、又引入了什么','',
'下列只比较双方均已判断且结果已知的相同行；把原弃判补为判断另外列出，不冒称纠错。','',
'|结果口径与对照|讨论纠错|讨论引入新错误|共同已判断样本数|共同决策准确率/对方准确率|补答原弃判（正确/补答）|','|---|---|---|---|---|---|']
for event in ['strict_close','archived_promotion']:
    for a in ['codex_independent','gemini_independent']:
        p=r['paired'][event+':joint_vs_'+a];resolved=p['resolved_abstentions']
        lines.append(f"|{event} 对 {labels[a]}|{len(p['corrections'])}|{len(p['induced_errors'])}|{p['common_judged_known']}|{pct(p['joint_accuracy_common'])}/{pct(p['other_accuracy_common'])}|{sum(q['correct'] for q in resolved)}/{len(resolved)}|")
lines+=['','同样日簇重复抽样10000次、固定种子20261019，按全部已知结果正确率比较的95%百分位区间如下。这只是本批小样本的条件不确定性，不能消除历史已见、来源或跨日相关性的偏差。','',
'|口径|共同相对对照|正确率差|95%区间|','|---|---|---|---|']
for b in r['bootstrap']:
    interval=b['percentile_95_interval']
    ci='未知' if interval is None else f"[{100*interval[0]:.2f}, {100*interval[1]:.2f}] pp"
    lines.append(f"|{b['event']}|{labels[b['right']]}|{pct(b['delta_full_known_correct'])}（百分点差）|{ci}|")
lines+=['','## 三个历史阶段','',
'|阶段|股票只次|结果口径|Codex正确/完整已知|Gemini正确/完整已知|共同正确/完整已知|','|---|---|---|---|---|---|']
for period,events in r['by_period'].items():
    for event,m in events.items():
        def cell(a):v=m[a];return f"{v['TP']+v['TN']}/{v['labels_known']} ({pct(v['correct_fraction_full_known'])})"
        lines.append(f"|{period}|{m['discussed_joint']['rows']}|{event}|{cell('codex_independent')}|{cell('gemini_independent')}|{cell('discussed_joint')}|")
lines+=['','## 信号日真实非一价组','']
for event in ['strict_close','archived_promotion']:
    lines+=['',event,'','|方案|样本/已知结果|TP/FP/漏检（含弃判正例）|Precision|Recall|F1|','|---|---|---|---|---|---|']
    for a in ['codex_independent','gemini_independent','discussed_joint','archived_ROLL252']:
        v=r['events'][event]['non_one_price'][a]
        lines.append(f"|{labels[a]}|{v['rows']}/{v['labels_known']}|{v['TP']}/{v['FP']}/{v['FN_judged']+v['abstain_positive']}|{pct(v['precision_known'])}|{pct(v['recall_full_known'])}|{pct(v['F1_full_known'])}|")
lines+=['','## 完整12日与共同Top3','',
'|案例|信号日→目标日|完整池|共同相对Top3（原方向）|','|---|---|---|---|']
for case,g in rows.groupby('case',sort=True):
    top=g.loc[g.discussed_joint_rank.gt(0)].sort_values('discussed_joint_rank')
    desc='；'.join(f"{q.code} {q['name']}({q.discussed_joint})" for _,q in top.iterrows())
    lines.append(f"|{case}|{g.iloc[0].date}→{g.iloc[0].target_date}|{len(g)}|{desc}|")
lines+=['','## 来源资格状态','']
for status,count in r['price_status_counts'].items():lines.append(f'- {status}: {count}只次。')
lines+=['','原独立稿、讨论过程、共同冻结、122只次全部结果及原始未知均在同目录保存。`complete_122_revealed_all_arms.csv`不删除误报/漏判/弃判；`results_AFTER_JOINT_FREEZE.json`含逐日矩阵、三阶段、非一价组及配对差值。','',
'股票预测成功不等于能买到或赚钱。本批没有生成真实/纸面成交记录，没有用次日开盘价补造收益，也没有把任何模型分数换写为概率。','']
path=HERE/'双模型共同决策历史回测_首批结果.md'
with path.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
print(path)
