# 题材与同身位竞争：输入接入说明
> 2026-10-07用户调整：“没必要继续弄真实题材时点”“原有的基础已经够了”。本专项后续采集及准备已停止，下面的入口、字段和固定C1/C2仅保留为历史能力说明，不触发自动执行，也不是现有行情回测的前置要求。当前优先任务见research_queue.json；使用已有市场行情与原结果做误差诊断。生产价格、日历与原预测结算核验仍按原合同执行。
未来信号日研究采集使用 `capture_signal_theme_inputs.py`；既有 `prepare_theme_competition_inputs.py` 保留为已形成V293档案的历史接入入口。二者复用现有V150题材角色及V114竞争关系计算，不改生产模型或V298冻结入口。

## 已完成的输入能力

- 保存供应商声明范围内的全部收盘候选，以及全部2板以上候选；题材未知不删行。
- 供应商分类只作描述，保留原分类、重叠标签、原始取得时间和缺失。技术标签不自动成为业务题材。
- 未知题材的同题材竞争数、角色和成员状态保留空值；全供应商范围的同板位代码另列，不能替代同题材竞争。
- 具体题材证据须来自单独审阅的原文、实际取得时间和哈希。现有V150只支持一只股票一个已登记主题材；全部供应商重叠标签另存，不自动挑选主归属。
- 题材角色和竞争身份描述当前声明范围内的结构，不确认真实龙头，也不代表下一日晋级。
- 证券状态、独立涨停价格和交易所全市场完整性不明时保留未知。此入口不会生成这些项目的verified状态，V151训练门槛继续阻止通过。
- 没有模型原始分数时留空；不从板位、名次或晋级判断补分。
- 派生档案的可用时间包含实际生成时间。研究截止固定为信号日16:30（Asia/Shanghai），不改变原生产调度或旧登记契约。

## 当期运行（信号日研究入口）

已确认原链路存在时间冲突：V293结算采集必须等目标日收盘，不能作为信号日16:30前研究输入。信号日入口因此单独采集当日四池、独立不复权日线及明确价格字段；不请求下一交易日行情，不伪造V293档案。

必须在真实信号交易日15:00—16:30（Asia/Shanghai）运行，档案实际形成也须不晚于16:30。没有日期/时钟回填参数。休市、未来日期和迟到启动会在网络请求及创建输出目录前拒绝。

10月8日的手动运行示例（输出目录须为全新路径；现在尚未执行）：

```powershell
& 'D:\python\python.exe' .\capture_signal_theme_inputs.py `
  --signal '2026-10-08' `
  --calendar '.\data\v279\exchange_calendar_audit_20261002_v1\calendar.json' `
  --protocol '.\research\theme_same_height\protocol.json' `
  --prior-roster '.\research\prior_roster_capture_20261002T211328999548Z\prior_roster.csv' `
  --prior-roster-sha256 'b10187daa9160485271b0c7711518cff0019d81af9f1535cb66774dd4c6d4b57' `
  --output '<全新信号日研究目录>'
```

原名单与各当日池代码取并集请求。`prior_roster_observations.csv`完整保留原48只，缺席新涨停池不等于断板；价格缺失或冲突保持未知。48只只是原供应商主板名称过滤名单，不能声称完整题材成员或交易所全A。其9月30日底稿在19:18取得，此处只作10月8日以前已存在的原组名单，不能补回9月30日16:30研究截止。

每项来源只请求一次，超时(5,8)秒；403及其他失败留档，不循环补采。响应日期不符、目标之后的日线或晚到响应拒绝使用。原字节、请求参数、真实取得时间、失败和哈希保留。最终`formation_receipt.json`绑定manifest哈希；后续读取须校验该回执、所有绑定哈希及全档实际文件mtime，不仅相信提前记下的时间。跨截止归档不可计时点合格输入。

证券风险状态、官方全市场完整性、业务题材和模型原分数仍有缺口；所有新产物`v151_ready=false`、无交易授权、无晋级真值。明确价格只来自供应商；腾讯限价位置仍属推断，不自动认证交易所限价。没有原文则只输出未审阅草稿，不生成AUTHOR_REVIEWED。

既有历史入口仍可接入已经形成的V293/V295档案，但晚于原截止的档案不计前向样本。原生产冻结和到期结算继续使用V298/V293，不能将新的研究归档送入旧结算桥接冒充兼容档案。本说明未创建或更改定时任务。

## 审阅题材原文后接入

另外准备符合 `templates/specific_theme_evidence_v150.json` 的实际证据，以及来源绑定JSON，再同时传入 `--theme-evidence` 和 `--theme-bindings`。不修改上述历史运行产物。

绑定顶层字段为signal_date、recorded_at、reviewed_at、review_status、reviewer、review_notes、evidence_sha256和groups。只有实际审阅原文后，review_status才可填写REVIEWED_SOURCE_TEXT。日期和哈希须对应提交的V150证据文件。

groups以实际theme_group_id为键。每组分别绑定membership、driver、assessment三个来源。每个来源含绝对raw_path、raw_sha256、captured_at以及能在UTF-8原文中找到的excerpt；membership和driver另含与V150一致的url、published_at及publication_time_basis=REVIEWED_EXACT_TIMESTAMP。membership摘录必须明确包含该组全部代码。出版时间、取得时间、审阅时间、档案形成时间分别核验；禁止倒填。

原文定位和哈希检查不能自动证明业务归属或催化事实。来源层级、事实判断、发布时间仍须实际审阅。空模板、供应商概念分类和未经核验叙事不能代替这项工作。首轮正式比较前还要登记精确算法、特征、训练与验证边界；本工具不授权拟合或部署。

## 本次实际检查

现有9月30日底稿：48只全部保留，其中11只连板候选与原登记一致。已核验的当时题材仍为0；同题材竞争字段均为空。实际档案形成于10月3日，属于历史接入检查，不能成为9月30日事前题材证据，也不能改写原预测。

9项定向合成测试通过；最初失败保留在 `input_preparation_validation.jsonl`。实际输入包的来源和产物哈希、完整候选以及V151拒绝未核验证券的行为均已检查。新增真实前向样本为0；没有准确率改善结论。


## 三类增量观察字段（2026-10-07补充，研究与采集规范）

用户授权补充真实题材支持、封板路径质量及开盘相对竞争强度。本次完成字段定义、研究队列及模板接入；没有新抓行情、回测、自动计算器、模型权重或交易授权。现模型原分数和前5排序继续使用。未知字段不能填0，空模板不能标已采集或AUTHOR_REVIEWED。

### 1. 真实题材支持与竞争压力：原方案保留，专项已停止
沿用既有V150/V114/V151和comparison_specification.json的固定C1/C2，不新增挑战方案或调整六项定义。same_theme_peer_count、same_theme_board_peer_count、higher_theme_peer_count及梯队/容量核心事件均依原契约、原声明范围、已核验细分成员计算，剔除候选自身。成员或范围不足时全部留未知，不能用行业、媒体概念、名字、排名或热度代替。当前5名媒体叙事不证明完整细分题材，已核验具体题材仍为0。

采用真实信号日15:00事实、16:30知识截止（Asia/Shanghai）；原文、取得、审阅和派生档案均须在截止前形成并绑定哈希。实际采集继续capture_signal_theme_inputs.py；题材证据及绑定按已有--theme-evidence/--theme-bindings送入。来源与证券/范围/价格/原模型分数门槛齐备后，才依原固定协议进入训练或比较。

### 2. 封板路径质量：候选字段，未登记数值实验

复用V275字段first_seal、last_seal、bomb_count、reported_first_seconds、reported_last_seconds、reported_seal_status、source_path、source_sha256、captured_at，标为供应商报告事实，与逐笔核验分开。

- first_seal_percentile_same_height_ex_self：冻结信号日原同板组并去自身。设全部同行数N；全部首封时刻有效且来源语义一致时，(更早首封同行数+0.5×同时刻同行数)/N。0—1无量纲，越大表示越晚，不是得分或晋级概率。N=0或任一所需时间缺失时未知，保留全部同行和缺失代码。
- failed_seal_count_reported：只使用原响应明确报告的非负整数开板/炸板次数，0只能表示来源明确报告0。历史重建的0占位不能作真值，不把次数越多直接设为越弱。
- last_seal_time_reported：保留原报告最后封板时刻和采集时点，不由一字状态或当日涨停布尔倒推。
- open_duration_trading_minutes、seal_stable_duration_trading_minutes：只有完整、明确语义的开板/封板事件序列和收盘状态才可形成；当前没有足够序列，默认未知。首封到末封间隔不是实际开板时长或回封时长，30分钟K线也不能补造精确事件。

固定到信号日收盘事实，实际取得/规范化/派生截止最晚16:30。captured_at未知或迟到只允许事后诊断；供应商报告合法时钟不自动成为盘口/队列验证。独立不复权收盘和明确涨停价仍按原程序核对，价格差一分钱不能算封板。当前仅部分原始记录可用，完整训练输入仍待补；本次未实现新分位/时长自动计算器，也未改原C1/C2字段。

### 3. 09:35原竞争组相对强度：只用于次日确认研究

前一晚以candidate_theme_intake.csv的provider_same_height_codes冻结组及原组哈希，按原信号日板位组、去自身。该组只覆盖供应商声明名单，不是完整题材或官方全A。目标日仍保留断板、跌停、停牌和缺失股票，不能用目标日新梯队或幸存者重选组。

- own_return_0935_pct：100×(09:35价格/独立原信号日收盘−1)。价格与昨收须同代码、同复权口径，观察日期须精确为原目标交易日。
- peer_return_0935_median_pct：全部冻结同行的同口径09:35涨幅中位数，不含自身。
- relative_strength_0935_pp：own_return_0935_pct−peer_return_0935_median_pct，单位为百分点，不是买入阈值。
- 同时保存original_peer_codes、original_peer_count、observed_peer_count、missing_peer_codes和peer_coverage。自身缺价格、无同行或任一冻结同行缺价格时，主强度差未知；部分中位数最多另作缺口说明，不能充当完整组强度。

盘后目标日未发生时统一写待发生。observed_at=目标日09:35与available_at/derived_at真实时间分开；09:35结束的K线若09:36才实际取得，只能在09:36之后追加确认，不伪造09:35可用。更晚资料不得进入预先固定的更早决策截止。收盘后补抓只作回放，不增加真实前向记录，不更新原盘后分数、主选、概率或原预测准确率。

已有分钟模板和采集接口只可按各自字段/单位契约使用：V237的minute/share/CNY与opening_minute_measurements.py的bar_end/60秒/end/shares/yuan不能直接互传；V243示例available_at=10:00:30不能当09:35合格记录。本次只定义原组强度字段，完整原组自动采集与计算仍未实现，进入独立待办。

### 统一来源、状态与评价

每行保留signal_date、target_date、code、原板位、original_roster_path/sha256、raw_path/sha256、字段原名与单位、observed_at、available_at、derived_at、knowledge_cutoff、来源等级、状态与缺失原因。保留完整候选和同行表，不从unknown推断失败。状态区分FIELD_DEFINED、SOURCE_PENDING、OBSERVED_RESEARCH_ONLY、WAIT_TARGET_OBSERVATION与EVIDENCE_UNKNOWN；字段定义完成不等于采集或预测改善。

题材支持继续原固定比较。封板路径和09:35强度分别等待真实输入并另批预登记；每批仍一个假设、固定基准、至多两个挑战，不把三类拼成主观综合分。缺失信息对照、完整候选覆盖、同日期配对、各板位Accuracy/Precision/Recall/F1及不确定性均需报告；概率未校准时Brier未知，成交不足时不报收益。固定样本外证据支持前，不改正式评分、前5名或买卖规则。刚完成的Wyckoff代理未见独立增量，继续保留暂不接入结论，不在旧窗口重调。


## 三类观察指标程序链路已接入（2026-10-07本批）

按用户“一起弄，不要一步步问”授权，已完成research_indicator_pipeline.py统一研究入口、seal_quality.py与open_strength.py，不再仅是字段规范。run留存完整原池和原登记子集，collect-signal复用实际收盘采集，collect-open保留原组逐码一次原始响应。现行模型、原预测和前5评分保持。封板同组分位/原报告开板次数可重建；精确时长缺完整事件序列未知。09:35计算器已实现，但当前Sina时间/复权/单位语义缺认证，原响应不自动变成有效强度。

59项针对检查与14子检查通过，一次修正后的现有档案接入保留48只、原登记11只；46行分位和48行开板次数仅旧档诊断，盘后原截止可用0，09:35全部待发生。当前真实题材已核验细分仍0，固定C1/C2未解锁。新结算前向0，准确率提升未评价，不能将程序测试当改善。入口和后续已绑定命令见research/indicator_observations/README.md；可阅读结果见research/indicator_observations/implementation_20261007T080458Z/complete_archive_run/交付说明.md。本批不启用或修改平台目标及定时任务。
