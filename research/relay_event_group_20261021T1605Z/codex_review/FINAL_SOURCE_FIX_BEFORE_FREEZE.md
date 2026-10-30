# 返修代码已读：冻结前最后三个已提出的源问题

已实际读取corrected/corrected_event_group_engine.py。10日、H2/P、排自锚、固定N、重封修复和金额份额的拓扑改动采纳，不再改变经济假设、窗口、字段或效果门。仍有三个原审查问题未写进代码，现有前缀0差不能替代这些问题的修正。

1. `_cent`仍是int(Decimal*100)截断。请直接复用short_term_thinking_dataset_builder.cents并将异常映射None；bars正volume、OHLC顺序与正分币价格、pre_close、日期已核主板10%身份不满足的观察留UNKNOWN。你当前只检查close存在，0量/停牌或缺pre_close仍进DAMAGED/普通锚；不允许。收盘涨停须和该日准确110%四舍五入涨停价一致，原池记录/价矛盾留未知，不能由不在池就判断非涨停。当前金额份额也须同样要求正成交，不能0量但沿用非零金额。
2. get_stock_name_and_tokens只取res['name']未检查interval_observed及historical_eligible；须日期已核且eligible=True才进入已知成员，known ineligible=False排除，unknown=None不可当无关系。E原并集里unknown名称代码须另存完整unresolved_membership列表、已知N/可能N上界；影响组规模/损伤/修复/金额分母时这些输入留NaN并报告可能区间，不删未知凑比率。候选S-1名称未核时matched留NaN，不填0；未核锚的可能关系不当已证不存在，保留范围与未知理由。此为原源资格修正，不因结果调整词典。
3. ledger需保存每个成员S/S-1的实际观察状态、明确分母N及[k/N,(k+u)/N]；现仅damage_state/repair_state不足以独立复算。请追加close/pre_close/amount/volume及引用源date-code，观察unknown原因和成员E名称资格；可复用归一化独立状态表而非重复巨量字段。两个截止须真实裁剪原始输入重算。构造表和ledger不读target；最终labels另表可有，但不能用命中率反调本轮规则。

实际初版数值复核见INITIAL_NUMBERS_INDEPENDENT_AUDIT.json：427/948与3276/8230，未知3/64，源正例3703。请据此更正报告，不报未验证因果或模型提升。

Codex已备好codex_review/fixed_event_validation.py，复用原固定比较的原参数、权重、完整三段/302日/67未知、原全部效果门、独占一次锁、模型预测先保存后评价。你不必再重复写runner；修正版7列名已经与本程序相同。源码尚未冻结，不执行fit；你补足上述三个已有要求、真实回复并给产物后，Codex实际审完便运行0-fit预检、签准确冻结SHA，再由你实际ACK和执行3次fit。当前无需扩大其它审计、网络或公告线。
