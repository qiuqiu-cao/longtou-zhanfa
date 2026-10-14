# A股主板连板龙头：强度评估与高度预测

> **开盘重点观察卡（2026-10-03用户要求）：** 每次复盘和预测交付现模型前3—5只重点股票，默认5只，附模型晋级评分（原内部分数×100，未校准）、原判断、一字/可成交性、开盘确认及取消条件。模型首选、非一字正向首选和交易主选分开；评分仅用于观察排序，不修改核心模型或原登记。固定规则见[AGENTS.md](AGENTS.md)。

> **完整原竞争组与题材支持准备：** 原供应商48只与原登记11只关系已核对，信号日独立采集入口已补齐；19项定向检查通过，保留失败、未知和实际截止时间。深交所主板名单可用，上交所及证券风险/题材时点证据仍有缺口。新增真实前向样本0，未改善或升级模型。见[完整原竞争组与题材支持采集条件](research/theme_same_height/完整原竞争组与题材支持采集条件.md)。

> **V301官方成交额研究：** 611次有限日请求保留原超时和一次补采，302个日期中299日两所分类齐全；3个历史训练日仍缺。发现深交所响应“亿元”与第三方封装“元”标签不符，按原表头解析；科创板CDR未拆分、主板合计包含ST，明确区别原候选范围。181个信号日、180个相同目标日的成交额比较中，变化量方法区间更窄、损失更低，后段也保持改善；31项检查和完整重建通过。见[官方成交额与分指标区间检验V301.md](官方成交额与分指标区间检验V301.md)。仍为历史诊断，不升级核心规则、改10月8日原预测或宣称稳定60%；生产入口仍为V298。

> **V300市场区间比较：** 保留2026年1—9月181个官方信号日，7项要求、8个数值渠道、两种固定方法共2896行；3项缺数据仍列未知。官方次日连接、昨日全部2板及以上晋级率重算、区间宽度与越界损失均独立检查。变化量方法同日配对后7渠道损失变大，最高板后段改善区间跨零；13项检查及7份结果完整复算通过，不升级规则或改10月8日原预测。见[市场预期区间固定比较V300.md](市场预期区间固定比较V300.md)。生产入口仍为V298，稳定60%未验证。

> **V299关注度观测：** 原供应商范围48只股票保存6864条日内排名、17520条粉丝构成、149条关键词及补采后480条关联记录。用户数0与非零占比共存，真实人数未知；排名、hitCount及关联rate不冒充讨论量、看多比例、资金流或涨跌幅。240次主查询及一次补采均保留，107项本轮检查通过。原截止前所有新观测不可用，未改10月8日预测，未证实稳定60%。见[关注度轨迹与共识数据边界V299.md](关注度轨迹与共识数据边界V299.md)。生产入口仍为V298，共识模块仍有缺口。

> **V298经验假设逐项留档：** 新日期模板与冻结入口要求15项经验假设齐全，核对已完成研究验收、统计来源、文件哈希及实际可得时间；改等级、删项或引用晚于截止的资料均拒绝。旧预测与十一段日报不倒填，当前研究另存附页。246项本轮检查通过，原10月8日11只预测仍等待，原9只历史回放仍6对3错，0项升级核心规则。说明见[经验假设逐项留档与冻结核验V298.md](经验假设逐项留档与冻结核验V298.md)。每日入口改为V298，稳定60%仍未验证。

> **V297经验规律固定检验：** 保留3707条候选、302个日期，15项假设中7项完成固定代理检验、8项缺历史时点输入。节前仅4个后段日期且前后方向相反，周五调整区间跨零，代码末位8没有明确优势；高位组相关性不能证明高标压制或模型增量。295项检查及完整复算通过，0项进入核心规则，原10月8日预测未改，稳定60%未验证。见[经验规律固定检验与分级V297.md](经验规律固定检验与分级V297.md)。每日生产入口仍为V296。

> **V296冻结前来源核验：** 新计划登记和读取时重建V295原始背景来源，拒绝被改动的分类、排名、时间、哈希、绑定及截止后生成的档案。原10月8日11只预测保持等待，原9只历史回放仍6对3错；285项检查通过，不升级模型或宣称稳定60%。每日入口改为V296，原账本与原件保留。说明见[背景字段冻结前重建核验V296.md](背景字段冻结前重建核验V296.md)。

> **V295时点背景基线：** 为原9月30日48只供应商范围涨停股保存10月2日当前分类与关注度，包含原11只候选；1013条分类、48条带计算时间的当前排名及17568条日期型历史排名。纠正内部编号误作股票代码造成的22条拒绝，第一版原件保留。新数据在原冻结截止前全部不可用，不回填历史；272项检查通过。历史当时题材和共识真值仍缺，稳定60%未验证。说明见[题材与关注度时点基线V295.md](题材与关注度时点基线V295.md)。

> **V294长周期路径检验：** 固定训练、此前校准及3—9月检验，三组均保留1656条候选。真实非一字首选原特征52/138、加价格路径49/138、再加成交量46/138；前期概率校准未改善后续Brier，不升级核心规则。官方147日中缺5个整日、4日标签不完整，均单列；特征、预测与评分重建通过，253项检查通过。原10月8日预测未变，稳定60%未验证。见[长周期分钟路径与概率校准检验V294.md](长周期分钟路径与概率校准检验V294.md)。

> **V293公开结果核验：** 已接通原始响应、日期与总条数、精确收盘价、独立日线和连续板位的逐股核验入口。实际发现请求9月29日却返回30日，四份错日期响应被拒绝；独立历史回放保留原9只候选，6对3错，新增前向样本为0。243项测试通过，尚未验证稳定60%。原10月8日预测保持等待；自动任务沿用工作日19:30并切换到新入口。说明见[公开结果核验与自动评分V293.md](公开结果核验与自动评分V293.md)。

> **V292每日研究与验证：** 接通11段日报、全候选分层评分、最近20/60实际交易日统计和不可覆盖的结算修订链。真实10月8日原预测仍为等待收盘，新增栏目不倒填；226项相关测试通过，概率与题材/分时标签缺口继续明确保留。已登记工作日19:30盘后任务，休市日跳过。使用说明见[每日研究与自动验证V292.md](每日研究与自动验证V292.md)。

> **V290长周期数据与标签口径：** 已为原2025年10月至2026年9月2888条候选、236个信号日采集30分钟路径及独立日线；4个超时代码另行补齐，全部原候选保留。发现10条信号日收盘封板必要条件冲突、6条原正向标签存在信号/目标日冲突；原标签与预测未改。全量来源及补采重建、202项测试通过，尚未升级模型或证实稳定60%。详见[长周期分时补齐与收盘口径核验V290.md](长周期分时补齐与收盘口径核验V290.md)。

> **V289扩展八月检验：** 新增原8月303条候选的178个代码5分钟历史及独立日线，双时期完整标签日期由13扩展到33。真实非一字首选合计：对照18/33、末段涨停价区间数17/33、全天占比16/33、回落较浅18/33；9月局部改善未保持，不升级默认规则。全部来源、特征、排名与结果重建通过，182项测试通过。详见[八九月分钟路径扩展检验V289.md](八九月分钟路径扩展检验V289.md)。

> **V288分钟路径检验：** 在原13个完整标签日期比较固定规则，真实非一字首选：既有对照7/13、末段涨停价区间数5/13、全天涨停价区间占比5/13、回落较浅8/13。后者只多命中1日，配对改善区间包含负值，不升级默认模型或改原预测；177项测试及完整特征、排名、结果重建通过。详见[分钟路径增量检验V288.md](分钟路径增量检验V288.md)。

> **V287完整九月分钟路径：** 原120个代码全部取得5分钟记录，共122,744条，覆盖182条候选、16个信号日。汇总价格182条一致，格点176条完整，成交量仅97条匹配；未通过者保留，防止改变候选构成。独立原始重建核验通过，尚未训练新模型或修改原预测。详见[九月完整候选分钟路径补齐V287.md](九月完整候选分钟路径补齐V287.md)。

> **V286分钟证据补齐：** 新浪来源取得原11只候选11,253条近期分钟记录，信号日汇总价格均与已有日线一致；6只满足预设分钟格点，9只满足成交量尺度核对，2只量差和所有缺分钟均保留。原10月8日预测不改，新增数据不能倒填为原时点已知，稳定60%仍未验证。详见[近期分钟数据补齐与核验V286.md](近期分钟数据补齐与核验V286.md)。

> **V285完整目标核对：** 更正操作卡将旧一字代理138/285写为非一字成绩的问题。逐条重算V279真实非一字排名首选为128/285（44.91%），正向精确率59/130（45.38%），召回7.09%；模型选择口径也有变化。市场180/285仍有延续识别与独立前向证据缺口，完整60%目标未完成。详见[隔日预测验收口径校正V285.md](隔日预测验收口径校正V285.md)。

> **V284每日接入：** 官方公告纠错已接入每日复盘输出，核对全候选池、板位、取得时间、原人工记录与PDF页码，拒绝附页文字篡改和倒填时间。151项测试通过，原10月8日预测保留；稳定60%尚未验证。详见[公告纠错接入每日复盘V284.md](公告纠错接入每日复盘V284.md)及[长假证据附页](results/v284/holiday_review_appendix_20261002_v2/review.md)。

> **V283官方证据补充：** 全11只候选取得56条窗口内官方公告索引，23份PDF均归档，9只完成所选公告全文核验。新增业务叙事否认、真实减持、待审批收购和质押的分项记录，并确认旧关键词扫描会把条件/否定语句误判为立案。新增证据另存，不改10月8日冻结答案、不自动调整概率。详见[长假公告核验与题材纠错V283.md](长假公告核验与题材纠错V283.md)。

> **V281—V282候选辨别检验：** 非一字专门训练和同日配对排序都未优于既有对照，不升级默认模型。专门训练二分类约71%，但未战胜全失败基线，晋级召回只有4.33%；不得据此宣称达到60%。完整日期与覆盖率见[非一字训练与排序目标检验V282.md](非一字训练与排序目标检验V282.md)。129项流程与模型契约测试通过，10月8日事前原件保持不变。

> **V280事前落实：** 10月2日真实登记10月8日预案，行情截止与知识截止分开；全11只连板候选、含一字首选、非一字首选和交易主选分别冻结。非一字无正向判断时留空，不把排名强改成晋级。117项流程测试通过，准确率尚待结算。详见[隔日事前留档与长假预案V280.md](隔日事前留档与长假预案V280.md)。

> **V279增量检验：** 为9063条候选补齐日线路径线索并核对真实一字；新增组9月真实非一字首选9/13，但2026年1—8月54/159低于对照61/159，未获稳定增量，不升级默认排序。同时独立核验669个历史交易日与已知下一交易日。详见[龙头日线线索增量检验V279.md](龙头日线线索增量检验V279.md)。

> **V278九月检验：** 新增139个标的不复权日线，217条连板收盘价与原池一致。修正一字定义、换手参考日期和昨日反馈名单后，正式9月回放为市场6/8、身份首选10/13、真实非一字7/13；低覆盖、小样本、无“延续”真值，不能宣布稳定60%达标。详见[九月新时期检验与口径修正V278.md](九月新时期检验与口径修正V278.md)。

> **V277日常落实：** 已将隔日计划留档状态接入日复盘输出，新增原报告哈希、全连板分母、板位与截止时间核对；缺少原计划明确记为未登记。日常步骤见[隔日复盘固定操作卡V277.md](隔日复盘固定操作卡V277.md)。25项契约与绑定测试通过，未据此升级准确率声明。

> **V274—V276 进一步检验（2026-10-02）：** 月度更新后，短线生态方向在2025H2/2026测试段为64.29%/62.26%，身份首选含一字为69.84%/71.07%；非一字为56.35%/42.14%，且市场“延续”召回偏低，完整目标仍未通过。新增原始封板证据归档与只用此前预测的置信度校准；详见 [月度更新与封板证据检验V276.md](月度更新与封板证据检验V276.md)。

> **V271—V273 隔日预期与预案优化（2026-10-02）：** 新增按训练→验证→测试顺序选择模型/阈值的市场与连板实验、量化超预期区间、时点校验与不可覆盖的计划入口、龙头身份证据转换和股吧玄学假设登记。默认复盘按 [隔日预期与预案稳定性升级V273.md](隔日预期与预案稳定性升级V273.md) 补充执行；60%以上目标仍待验证，历史身份命中与非一字/可交易结果分别统计。实验结果在 `results/v271/next_session_nested_20261002_v1/report.json` 和 `results/v272/market_expectation_20261002_v1/report.json`。

> **最新龙头识别规则：V253 二板定龙头优化。** 首板只建候选池，二板生成“龙头种子”，三板及真实分歧后才确认龙头身份；最高板只自动获得空间/情绪锚身份。详见 [二板定龙头优化V253.md](二板定龙头优化V253.md)。

> **整合执行规则：V255。** 市场情绪只负责风险闸门，V253负责二板龙头种子，V236负责三板以上身份，V246/V249负责次日抗核和成交标签；四层不再用一个总分混合。详见 [龙头战法整合升级V255.md](龙头战法整合升级V255.md)。

> **V267 周期与基本面清洁度升级。** 在题材和身份层之前增加旧周期/新周期转换层，并把“基本面干净”定义为风险过滤器：旧周期修复、补涨窗口、新周期试错和新周期确认分开标记；基本面 `DIRTY` 或 `UNKNOWN` 不得自动进入主要接力首选。输入契约同时检查字段覆盖率和时点，列存在但为空不能通过联合研究闸门。详见 [周期与基本面清洁度升级V267.md](周期与基本面清洁度升级V267.md) 和 [cycle_fundamental_gate_v267.py](cycle_fundamental_gate_v267.py)。

> **V270 新复盘提示词。** 将市场行情、情绪、老周期、老龙头、补涨窗口、新周期、题材、基本面清洁度、打板层级和次日执行条件串成一条决策链。详见 [每日龙头复盘提示词V270.md](每日龙头复盘提示词V270.md)。

> **V254 二板回测反馈：** 排除一字/单价代理后，二板模型验证晋级率约35.00%，同分母高度基线约30.56%，配对区间跨0；暂不宣称已证明龙头识别提升。详见 `results/v254/two_board_leader_walkforward_20260926_v3/report.json`。

> **V256 时间外诊断：** 2026时间外二板非一字模型晋级率约40.51%，三板以上约37.66%；二板身份提升未改善日线收益代理，仍为 `DIAGNOSTIC_ONLY`。详见 `results/v256/relay_promotion_holdout_20260926_v5/report.json`。
> **V256 迭代结论：** 身份/情绪字段单独没有带来选股增量，价量字段提高晋级排序但恶化收益代理；以后固定分开输出“晋级身份排序”和“交易授权状态”，高晋级概率不再自动等于买入许可。
> **V257/V260 多窗反馈：** 525个完整日上，无条件价量模型的二板提升只在后两段明显；V258曾用代码顺序基线替换分歧日候选，已撤回。V260改为保留模型排序、只降低分歧日身份置信度；非分歧日二板晋级提升约13.48个百分点，仍属 `DIAGNOSTIC_ONLY`。
> 详细产物：`results/v257/relay_promotion_walkforward_20260926_v3/report.json`、`results/v258/relay_promotion_regime_gate_20260926_v2/report.json`。
> **V259 日历审计修正：** 2026-09-01 的数据把下一交易日错误跳过到 09-04；正确的 09-02 候选池缺失，因此 603721/002084 的晋级标签必须标为 `UNAVAILABLE`，撤回“次日未晋级”结论。详见 `results/v259/historical_replay_calendar_audit_20260927_v2/report.json`。

> **V262 早期龙头 TopK 三段滚动回测：** 固定执行“训练→验证→盲段”时间切分，分别评估首板 A（下一板晋级）、二板 B（同身位未来空间代理）和三板 C（未来市场高度代理）的 Top1/Top3/Top5。盲段合并结果：A 模型 Top1/3/5 精确率 **5.16%/4.21%/4.24%**，B 为 **11.92%/9.86%/9.49%**，C 为 **11.06%/10.94%/10.42%**；相对高度规则，增量主要集中在 A 和 B 的 Top1，C 仅有边际改善。该结果是重建日线样本的身份诊断，不是实盘胜率；题材、竞价、封板路径、VWAP、Level-2 和真实成交仍缺失，B 的“题材龙头”标签不可用。详见 `results/v262/early_leader_topk_walkforward_20260927_v1/report.json`。

> **V263 收益与风险代理补充：** 在同一盲段 Top1 选择上，A/B/C 模型的 1日收盘收益代理分别为 **1.76%/1.96%/2.19%**，5日分别降至 **0.86%/0.52%/0.84%**；二板模型的 1日、3日相对高度基线改善，但 5日不胜随机控制。5日最大不利波动均值约为 A **-8.61%**、B **-8.86%**、C **-10.35%**，说明龙头身份排序不能替代持仓期限和止损规则。收益、Sharpe、回撤和连续亏损均为未经成交验证的收盘价路径代理；原始行情的 `status_verified` 全部为 False，状态保持 `DIAGNOSTIC_ONLY`。详见 `results/v263/early_leader_topk_walkforward_20260927_v5/report.json`。

> V263 的年化 Sharpe 只用于同口径诊断，1/3/5 日窗口存在重叠，且未计入排队、滑点、涨跌停无法成交和真实退出，因此不得当作可交易策略 Sharpe。

> **V264 策略契约审计：** 对 TopK、未来高度、收益代理、环境分层和失败分类逐项验收；题材龙头、竞价/盘口、真实成交、纯净盲测和无生存偏差均未通过，因此自动决策为 `CASH`，且不允许把 V263 升级为生产策略。详见 `results/v264/strategy_contract_audit_20260927_v3/report.json`。

> **V265 输入字段契约：** 已固定基础、竞价、盘中封板、题材、情绪和执行字段的覆盖审计。当前历史池允许身份复盘，但竞价/盘中/题材字段缺失，条件接力复盘和交易授权均为 `false`；补齐数据后可直接复验，不再临时改规则。详见 `results/v265/leader_input_contract_20260927_v1/report.json`。
> **V266 日复盘桥接：** 将日复盘、V263 身份诊断与 V265 字段闸门统一；过期行情只保留历史诊断，缺竞价/盘中/题材/执行字段时保持 `CASH`。2026-09-21 复盘相对 2026-09-27 为 `STALE / DIAGNOSTIC_ONLY / CASH`，华瓷股份是情绪锚、世联行是挑战者，均不等于买入许可。详见 `results/v266/strategy_review_bridge_20260927_v1/report.json`。

> **日复盘管线示例：** 2026-09-21 复盘识别为“加速”，涨停 93 家、连板 21 家、炸板率约 19.83%；最高板华瓷股份只保留情绪锚身份，世联行是观察级高度竞争者，因题材、竞价和执行证据不完整，主选与交易授权均为 `CASH`。详见 [盘后复盘 V7](<D:/Documents/ChatGPT/龙头战法/results/daily_review_20260921_v7/review.md>)。

> **当前交易模式：V246 次日抗核按钮追高模式。** 允许追高，但必须先判断买入后下一个交易日是否具备承接，以及高标是否可能形成市场负反馈；`ANTI_NUCLEAR_PASS`通过后才允许进入追高预案。详见 [交易模式V246_次日抗核按钮.md](交易模式V246_次日抗核按钮.md)。

龙头分类采用V247：周期龙、节点龙、穿越龙、主线龙、换手龙、卡位龙、补涨龙、伴随龙和情绪锚分开判定。最高板只代表空间/情绪锚，不自动代表周期龙或可交易龙。详见 [龙头本质与分类V247.md](龙头本质与分类V247.md)。

经验规则已整理为V248，增加周期节点、穿越/卡位/补涨的专属确认、术语误用对照和股吧经验前向验证约束。详见 [龙头战法经验沉淀V248.md](龙头战法经验沉淀V248.md)。

次日胜率提升按V249执行：把晋级率与买后胜率分开，新增 `FILLABLE`、`ANTI_NUCLEAR`、`B1_NET_POSITIVE`、`B2_SUSTAIN` 四个标签，并用 `TRADE/WATCH/CASH` 控制是否出手。详见 [次日胜率提升方案V249.md](次日胜率提升方案V249.md)。

股吧经验的高胜率候选模式见V250：优先观察破局节点弱转强、主线核心首次良性分歧、新周期独立题材节点龙；固定股价、市值、竞价涨幅只作待验证阈值。详见 [股吧高胜率选股模式V250.md](股吧高胜率选股模式V250.md)。

项目级快捷触发词已设置：以后用户说“复盘”，默认执行详细复盘协议V251，覆盖上一个交易日的市场与情绪、主线、找龙辨龙、晋级与抗核风险、下一交易日情景、候选和执行条件。详见 [复盘触发协议V251.md](复盘触发协议V251.md)。

本项目研究沪深主板、非 ST、非退市整理的全部当日收盘涨停股票，首板与真实连续板分别建池、分别评价；盘中触板但收盘未封另列炸板观察池。它不研究趋势股、反包板、创业板/科创板/北交所，也不把无法成交的一字板当成可实现买点。

> **2026-09-17有效性更正：** 旧机会标签曾排除“次日未触板”候选，构成未来条件筛选。V152已把未触板样本恢复为负样本；依赖旧标签的选股胜率、角色胜率和情绪叠加结果全部停止用于概率提升声明。纠错后首板模型相对基线的提升区间跨零，连板模型未战胜“组内最高板”基线，故当前仍为 `DIAGNOSTIC_ONLY`。情绪V144只允许预警次日大分歧/退潮，不能直接给个股加分。详见 [模型有效性状态V154.md](模型有效性状态V154.md)。

V155把情绪拆成五个可结算的次日损伤分量，只有跌停扩散和中位晋级坍塌通过时序外闸门，2/5不足以生成综合损伤概率。V156因此只用它们解释风险传导，主情景仍由V144承担，并固定输出支持条件、失效条件及允许的候选类型。

V157新增跨题材卡位：首板按新周期/低位资金竞赛建因子，连板按高度/梯队资金竞赛建因子；题材缺失会令整日跨题材训练失效，不能被当成弱题材。当前样本42/48只有题材证据、全市场完整性未证明，因此仅可描述。V158/V194把社区、交易所规则、研究文献和内部反例检验整理为66条含代理变量与反例的假设；除弱转强、PK、累计换手、尾盘板、拥挤和竞价不可撤阶段外，新增龙虎榜事后偏差、资金流口径、高开剩余赔率、炸板修复路径、金额标准化、题材标签去重、时段季节性、反包身份、容量仓位分离、最高板身份与可买性分离、开盘强度与收益权衡、情绪与高开交互、弱转强事后分母偏差、首板供给基数陷阱、晋级身份与买后赔率分离、严重异动证据层级以及快板结构与控盘意图分离，当前0条获准直接进入生产模型。

V166首先检验累计换手：6,213条二至五板候选、2025年四个滚动季度和20交易日隔离带下，晋级模型的AUC小幅增加但Brier差异聚类区间跨零；条件触板且非一字的B+2模型没有改善，倒U形也未在身份和交易目标上同时成立。因此累计换手只保留为诊断字段，不进入正式选股分数。

V167在补齐日线并纠正执行分母后重检炸板路径：13个时序日、577条首板中有575条执行状态可观察。30分钟内修复组相对长时间争夺组，一进二率高约6.56个百分点、B+2平均净收益高约0.91个百分点、B+1大面率低约4.57个百分点，但三项聚类区间仍跨零，完整执行日仅11个。因此“快速修复优于长时间争夺”仍只作风险解释，原始炸板次数和路径分类都不得直接提高正式分数。

V168修正题材来源置信度：复盘网页明确列出股票与题材，只能证明该网页存在这一市场叙事；若页面本身为二手复盘、AI整理、待核验，不能升级为公司业务事实、催化首次公开时间或买入证据。9月16日80只主板收盘涨停均被结构化观察，其中52只形成14个至少两只股票的二手题材组，另外多题材歧义和孤立标签保持未归组；0只获得正向题材因子，0只获准据此建立高切低映射。

V217补上名称/符号题材观测层。系统在全部主板收盘涨停和炸板观察池上扫描一至二字名称片段，剔除“股份、科技、集团、控股”等机械公司描述，并用冻结门槛检查成员占比、连板数量、板位层次和跨行业扩散。名称匹配只标记为归档池内事实；“字辈行情”固定为推断性符号聚集，不能冒充产业题材、催化或业务归属，也不能直接重排候选、提高概率或授权买入。该规则是在9月18日“华字辈”遗漏后设计，历史重放属于事后诊断，必须经过后续前向样本验证。

V160在补齐截至9月16日的17,106条候选代码日线后重检“上板越快越强”：577条时序首板中575条具备执行状态。9:35前封板组一进二率23.60%，非快板14.96%；但可观察成交率82.95%对95.69%，B+1大面率18.31%对12.11%。快板B+2正收益率较高，但平均收益差和大面差的聚类区间仍跨零，完整执行日仅11个；最早封板每日第一仍仅1/13晋级。因此封板速度只保留为注意力、价格发现和执行风险线索，既不单独加分，也不作为硬否决。

V194进一步把快板拆成薄成交/一字代理、换手确认代理、其他快板和非快板。薄成交/一字代理组一进二率24.14%，但可观察成交率仅78.57%，非快板分别为14.96%和95.69%；薄成交组相对非快板的成交率劣势在交易日聚类区间中仍为负。日线涨停池无法识别队列所有者、真实撤单或“主力控盘”，所以“控盘”从模型字段中降级为不可获得，只保留可验证的薄流动性、一字、排队不可验证、换手和买后收益。

V161增加100%/200%严重异常波动数据闸门。现有217个阈值扫描事件只有2个具备公告佐证、0个可训练标签；两个基准指数日期窗口齐备，但仍是第三方行情未完成官方对账，公告重置链、复权处理和价格限制状态也不完整。因此只允许展示基于正式规则的“反事实距离路径”，不允许输出命中概率、用于候选重排或宣称接近翻倍必然触发异动。

V162把未来三日情绪拆成风险主导、进攻主导、修复转进攻和混合轮动四条路径，并使用三段时间隔离和三日隔离带验证。后段136日中，挑战模型Brier为0.684、差于Markov的0.659，修复转进攻仅6例且召回为0。V163进一步加入涨跌停、晋级、溢价、尾部损失的1日/3日变化、前5日偏离和状态持续时间；360个滚动样本外日中Brier恶化至0.745，只在6折中的1折胜过截面模型和Markov。未来2—3日因此只能输出主情景、备选情景、风险情景和失效条件，不得输出已验证概率或用于加仓。

V164统一归档现有9个交易日、403只收盘候选的题材证据，125只具有下一交易日前可用的题材归属，278只明确保留为未知。单日题材覆盖从0%到87.5%不等，但主题归属不能替代催化首次可得时间、驱动类型和预期透支状态；当前0行、0日满足完整模型口径，全市场完整性也未证明。因此题材档案只能用于复盘解释，不能进入概率排序。

V165把125条已知题材归属合并为32个“催化与预期”核验任务，其中高优先级14个、中优先级14个、低优先级4个。每个任务必须补齐驱动类型、催化首次公开时间及依据、是否仍有新增信息、盘前/盘后预期状态和可得时间；题材复盘文章的发布时间不得代替催化首次公开时间，传闻不得作为硬事实或买入理由。目前已对9月8日种植农业、9月8日文化出版和9月10日大消费建立3条二手来源重建，时间顺序均有效，但首次公开时间和预期状态仍属推断，因此模型合格行为仍为0。

## 核心结论

龙头不是单只股票的静态属性，而是“市场情绪—题材梯队—个股身份—换手确认”共同形成的状态。历史经验中，二板确认题材参与，三板附近常发生身份切换；真正可交易的龙头通常至少经历一次放量分歧和回封。到了高位，策略从“寻找晋级”切换为“管理兑现风险”。

预测采用离散生存模型。历史终局高度按真实连续晋级链生成，数据尾部尚未走完的样本标记为右删失；任何终局日期跨越训练边界的样本都会从该折训练中清除。对当前已处于 `k` 板的股票，逐级估计：

`P(至少到 h 板) = Π P(j -> j+1 | 当时可知特征), j=k...h-1`

因此结果包括：下一板晋级概率、至少到各高度的概率、最可能终止高度及 80% 区间，而不是武断地断言“必到几板”。

## 只做连板龙头的筛选闸门

1. 股票范围：代码以 60/00 开头；排除 ST、*ST、退市、新股无涨跌幅期。
2. 状态范围：收盘时至少 2 连板；反包、N 天 M 板不计入连续板。
3. 周期闸门：退潮期原则上不接力；修复/升温只做换手核心；高潮期缩仓；分歧期只看唯一活口。
4. 身份闸门：空间板、主线题材先锋、换手核心、唯一活口优先；后排和孤立题材降级。
5. 可买性闸门：连续一字板只作情绪锚；必须等竞价、开盘承接或真实换手回封。
6. 公告闸门：风险提示、严重异动、减持、监管、业绩雷出现时降低概率或直接剔除。

## 100 分强度体系

| 模块 | 分值 | 关键变量 |
|---|---:|---|
| 情绪周期 | 15 | 涨跌停数、炸板率、昨日涨停溢价 |
| 高度与空间连续性 | 15 | 当前板数、市场高度是否拓展 |
| 晋级与亏钱效应 | 15 | 1进2/2进3/中位晋级率、断板反馈 |
| 题材与梯队 | 12 | 同题材首板/中位板数量、是否主线 |
| 板块结构修正 | -13～+5 | 内部分歧、同高度竞争、高标反馈 |
| 龙头身份 | 10 | 空间板、先锋、换手核心、唯一活口 |
| 量价换手 | 8 | 换手适度、成交额、是否爆量派发 |
| 竞价与开盘 | 8 | 高开幅度、竞价换手、承接和回封 |
| 资金行为 | 7 | 买方合力、卖方集中度、资金连续性 |
| 公告风险 | 5 | 异动、减持、监管、业绩风险 |
| 可买性 | 5 | 一字/缩量程度、实际成交可能性 |

评分只用于排序：85+ 为核心观察，75–84 为确认后小仓候选，65–74 只作情绪锚，低于 65 忽略。

### 板块数量不等于板块强度

板块涨停多只能证明热度，不能证明次日可接力。模型按以下顺序判断：

1. 高标反馈：昨日板块最高标是加速、换手晋级，还是冲高回落/大面。
2. 叙事纯度：高、中、低位是否围绕同一驱动，而不是把化药、中药、创新药等宽泛归为一个板块。
3. 梯队连续性：先锋、中位、首板是否同时存在，还是高标孤立、首板随机扩散。
4. 竞争压力：同高度候选越多，次日互相卡位越严重；板块只能留下一个核心时提高分歧惩罚。
5. 资金结构：买方是否多席位共同确认、卖方是否集中兑现、同一资金是否隔日持续。只有合力，没有所谓“某席位必涨”。

板块处于“高标做顶 + 多分支扩散 + 席位兑现 + 同高度竞争”时，即便当天有多只涨停，也定义为大分歧预期。

## 游资行为如何进入模型

龙虎榜滞后且席位与自然人不是稳定的一一对应关系，因此不根据网传名号机械加分，只提取行为：

- 核心合力型：多个活跃席位围绕主线唯一核心共同买入，且买额相对成交额不过度失真；提高 `capital_consensus`。
- 容量趋势型：偏好高辨识度、流动性充足的容量核心，通常影响题材持续性而非次日必连板；提高板块确认，不直接提高个股可买性。
- 高位接力型：在情绪上升或分歧转一致时参与空间板；若次日没有增量资金，容易变成最后一棒，因此必须同时检查 `capital_continuity`。
- 快进快出型：首板或二板迅速介入、次日兑现；它是溢价信号也是抛压来源，不做无条件正向加分。
- 卡位点火型：资金从旧龙切向同板块低位新核心，表明旧龙衰退；新股的身份上升，但旧龙的 `anchor_health` 必须下降。
- 量化共振型：封板速度更快但撤单和日内反馈也更剧烈；没有逐笔数据时标记为未知，不凭龙虎榜标签臆测。

真正值得学习的是主线选择、情绪阶段、合力形成和失败退出，不是第二天照抄知名席位。

## 文件与运行

- `relay_model.py`：无第三方依赖的评分、离散生存预测和逻辑回归训练器。
- `current_snapshot_2026-08-28.csv`：最近交易日的诊断快照。
- `historical_sample_template.csv`：历史训练数据字段模板。
- `backtest_august_2026.py`：2026年8月逐日最高连板回测，含数据缓存、成交限制和费用。
- `短线龙头战法_平台共识与八月回测.md`：平台观点降噪、实用规则及完整月度结果。
- `每日龙头复盘模板.md`：把股吧复盘中有效的反馈链转成每日可填写的预期、证伪、执行与验收表。
- `leader_system_v2.py`：第二版系统；情绪温度、四类龙头身份、分高度生存概率、贝叶斯收缩和滚动样本外验证。
- `第二版系统说明与验证报告.md`：第二版改造结果、样本外指标、当前角色和仍需解决的数据限制。
- `leader_system_v3.py`：第三版主程序；情绪引擎3.4使用过去60日因果分位数识别相对恶化，并在在线状态转移基线上加入扩展窗口逆风模型挑战者；概率引擎3.8采用训练/校准/阈值验证/时间留出四段隔离，新增右删失终局高度生存预测，以及“晋级精确率×大面率×可成交频率”的联合阈值前沿。
- `collect_context_v3.py`：冻结模型外的时点化题材/公告证据采集器；公告只按索引日期使用并施加风险有效期，概念接口失败时保留 `unavailable/skipped`，绝不把缺失解释为没有题材。
- `announcement_forward_v14.py`：把盘后公告档案与已经注册的V10次日候选绑定；只允许在信号日15:05后、交易日09:15前生成，不接受决策时点之后采集的数据。硬风险只能否决/降级，不能产生买入信号；原始输入、候选风险行和六板以上情绪锚覆盖均以SHA-256冻结。
- `emotion_challenger_v4.py`：不修改V3.8的双目标情绪挑战者；分别预测次日大分歧/退潮与次日接力损伤，采用160日初始历史、每20日重训和三段因果检验，旧数据只算开发样本。
- `emotion_forward_v4.py`：锁后情绪预测的预注册、次交易日自动结算、哈希链校验和前向监控；只接受当日15:05后、晚于开发截止日的最新市场行。
- `industry_structure_v4.py`：利用逐日涨停/炸板/强势池构建行业梯队、扩散、孤立、分歧及相对前日变化；永久标记为 `INDUSTRY_PROXY_NOT_THEME`，不能替代真实题材或催化。
- `theme_evidence_v4.py`：真实题材/催化的时点化注册、不可覆盖归档、同日成员梯队、候选角色及09:35/10:00成员同步率计算；社区观点和行业标签不能获得验证资格。
- `symbolic_name_theme_v217.py`：对全量涨停/炸板池执行名称字符与词组聚类，输出符号高度锚、连板支撑、首板扩散及前日变化；永久保持 `DIAGNOSTIC_ONLY`，不替代真实题材和次日竞价确认。
- `open_confirmation_v4.py`：严格验证09:20、09:24:50、09:25、09:30、09:35、10:00快照，输出 `CONFIRM_RESEAL/WATCH/CANCEL/DATA_LIMITED/INELIGIBLE`；禁止日线开盘和行业代理冒充真实开盘确认。
- `session_forward_v4.py`：在次交易日9:15前冻结候选和六个采集时点，盘中快照按唯一时点追加到哈希链；迟到、漏采和覆盖不足永久保留，不能事后补成训练样本。
- `execution_contract_v4.py`：V4.2将按时完整的 `CONFIRM_RESEAL` 转为不可覆盖的纸面执行契约；覆盖已核验首板与全部连续板，不再用板位机械否决，高板只缩小仓位。按T+1隔夜压力反推仓位，分别记录B+1/B+2/B+3估值；卖出受阻可在后续交易日继续登记，直到首次可成交退出，未解决亏损不会从分母消失。
- `security_master_v4.py`：导入按日期生效的上市/ST/退市/涨跌幅制度区间，审计候选是否真正属于当时的非ST主板；缺失区间不能按正常股票处理。
- `collect_security_master_v4.py`：通过AKShare传输适配器归档深交所当前主板名单与7,469条带日期简称变更，校验历史链尾与当前简称一致后重建ST/正常区间；只提升已观察深市代码的状态证据，不把当前名单冒充完整历史母池。
- `historical_universe_v4.py`：用沪深交易所终止上市清单找回当前股票池遗漏的退市代码，并以可续跑分片恢复未复权OHLCV；行情、证券状态和推断上界分开标记。
- `sse_name_history_v4.py`：归档上交所42只退市样本的10,025条公告索引及113份状态相关PDF；只有旧简称、新简称和生效日同时明确时才提升为已核验事件。
- `collect_sse_security_master_v4.py`：把上交所正式公告PDF、公告元数据和终止上市清单合成为时点区间；元数据单独观察到的切换固定为 `inferred`，不能通过资格闸门。
- `survivorship_sensitivity_v4.py`：合并沪深已核验时点状态计算幸存者偏差；未核验沪市区间继续只列上界，量化对涨跌停宽度、连板梯队和空间高度的影响，不改写冻结V3。
- `collect_delisted_security_master_v4.py`：把深市退市股票的交易所更名链和终止上市日期转换成统一、不可重叠的证券资格区间。
- `survivorship_rebuild_v5.py`：在独立V5目录用已核验退市日线重建全市场情绪与候选，不覆盖V3/V4；`compare_emotion_rebuild_v5.py`按同一锁定规则比较新旧情绪预测。
- `risk_rule_audit_v4.py`：独立审计固定阈值的大面拒绝规则；只用2—5板可成交代理，按三个连续时间段检查方向与覆盖，未通过的经验不得接入冻结模型。
- `forum_hypotheses.csv`：把股吧经验转成可量化、可证伪的研究假设，不把热度和游资名号当买点。
- `daily_decision_contract.py`：审计每日复盘是否同时覆盖首板/连板、三种角色、五类情绪证据、真实细分题材、唯一条件首选及确认后可执行价格；不完整时强制空仓。
- `emotion_reference_builder.py` 与 `emotion_evidence_pack.py`：用带固定分母、缺失数和可用时间的长表指标建立因果历史分位，分别生成赚钱效应、断板亏钱效应、高中低位、题材分歧和修复质量证据；不生成概率或买入指令。
- `risk_budget_audit.py`、`exit_plan_audit.py` 和 `trade_account_ledger.py`：审计事前仓位/风险上限、冻结退出规则、真实或纸面成交、T+1、未平仓净值与完整交易统计。
- `execution_funnel_audit.py`：分别统计会话清单候选、决策行、重点候选、触发、下单、等待、未成交、部分/完整成交及空仓日；入口漏斗明确标记退出证据未接入时未平仓数为未知，不把触价或条件触发当成成交。
- `complete_trade_dataset.py`：只为入场、风险、退出、漏斗和账户全部合规的已平仓交易生成扣费后盈利及-7%大面标签；未选、未成交和未平仓保留但不标注为失败。
- `complete_trade_split_audit.py`：对完整交易按信号日建立训练/校准/验证/盲测四段切分，留交易日隔离带并清除平仓日跨界的未来标签。
- `policy_conditional_model.py` 与 `policy_threshold_validation.py`：只用训练/校准/验证段训练并校准完整交易盈利/大面模型，在预注册网格中检验70%盈利率、10%大面率及每周3—5信号；不读盲测指标，不放行部署。
- `policy_trade_blind_vault.py`、`policy_trade_blind_score.py`、`policy_trade_blind_evaluate.py`：将完整交易盲测特征与盈利/大面/净收益标签分库，冻结支持域、阈值和预测后再开库；独立检查100笔、70%盈利、10%大面、正收益、利润因子及同口径基准。
- `full_pool_rank_dataset.py`：逐日纳入所有已核验主板非ST收盘首板和连板，生成不冒充成交收益的次日机会代理标签；母池、时点状态或标签缺失即降级。
- `full_pool_rank_split_audit.py`、`full_pool_rank_model.py`：按整日候选分组并为B+2标签留两交易日隔离带，用同日等权的逻辑回归/梯度提升比较每日第一名命中率、排序位置和校准误差。
- `full_pool_rank_blind_vault.py`、`full_pool_rank_blind_score.py`、`full_pool_rank_blind_evaluate.py`：把开发集、盲测特征和盲测标签拆成独立哈希工件，先冻结全部预测再开标签库，与预冻结“只买最高板”基准比较。
- `full_pool_proxy_walkforward_v17.py`：固定时间窗滚动样本外机会代理回测，隔离未来标签和跨窗口 episode，并保留首板/连板及 B+1/B+2/B+3 分组结果。
- `full_pool_role_backtest_v18.py`：把情绪锚、成龙潜力和当前买入机会拆成独立目标，在同一滚动窗口中报告角色重合与各自代理表现。
- `leader_model_stack_audit.py`：强制把“全池机会排序”和“冻结政策内的完整交易盈利/大面校准”分为两层，检查样本外、适用范围、支持域和组件哈希后才可接入每日决策。
- `system_acceptance_audit.py`：将数据完整性、独立盲测完整交易和冻结前向验证合并为失败关闭的系统上线闸门。
- `public_live_capture_v1.py`：无需截图的公开 Level-1 行情快照层；自动抓取全部昨日首板/连板候选的价格、开高低、昨收、累计量额和提供方时间戳，并可计算带覆盖率的盘中广度代理。公开行情缺少 Level-2 排单、真实撤单和成交证据，因此本层只允许 `WATCH/CANCEL/CASH`，永不单独授权 `ENTER`。V176进一步规定三时点匹配额只能称净衰减/净增长代理，不能冒充真实撤单或补单。
- `public_minute_capture_v1.py`：在09:35/10:00自动抓取全部候选的公开1分钟OHLCV，形成VWAP斜率、均价位置、首次回落量比、触板/炸板分钟数、回封分钟和跌破VWAP放量代理；所有字段明确标记为分钟推断，不能冒充逐笔或Level-2。
- `public_live_monitor_v1.py`：开盘前启动后自动等待并采集六个注册时点，并在09:35/10:00追加分钟承接代理；错过的时点不会事后补造，采集事件写入哈希链日志。它只做数据监控，不下单。
- `V3系统说明与验证报告.md`：V3架构、当前诚实验证结果及70%上线闸门。
- `V3数据字典.md` 与 `templates/`：免费数据积累和券商/终端CSV导入规范。

第二版运行：

```powershell
python leader_system_v2.py fetch     # 尝试构建长期公开归档并缓存
python leader_system_v2.py train     # 时间切分训练、盲测和滚动验证
python leader_system_v2.py current   # 输出当前角色、概率、置信度和可买性
```

程序设有部署闸门：历史天数、验证阈值证据、盲测样本、校准误差或风险标签不达标时，状态自动保持 `DIAGNOSTIC_ONLY`。诊断阈值与部署阈值分离；部署阈值为空时系统自动弃权，不得把诊断概率当成实盘胜率。

最终盲测区间只能查看一次。若在查看结果后修改模型并再次使用同一日期，系统自动标记为 `REUSED_DEVELOPMENT_HOLDOUT`，保存窗口/标签/代码SHA-256指纹并永久禁止该区间通过部署闸门；必须冻结代码后等待新的未来日期。

第三版运行：

```powershell
python leader_system_v3.py collect --date 20260828
python leader_system_v3.py collect --history-start 2024-01-01 --history-end 2026-08-28 --workers 5
python leader_system_v3.py build
python leader_system_v3.py train
python leader_system_v3.py backtest
python leader_system_v3.py audit
python leader_system_v3.py score --date 2026-08-28 --asof 09:35
python leader_system_v3.py freeze
python leader_system_v3.py score --date 2026-08-31 --asof 09:35 --record-forward
python leader_system_v3.py forward-verify
python collect_context_v3.py --date 2026-08-31 --lookback-days 45 --workers 4 --commit-announcements
python emotion_challenger_v4.py --verify-lock
python emotion_forward_v4.py record
python emotion_forward_v4.py settle
python emotion_forward_v4.py verify
python emotion_forward_v4.py monitor
python industry_structure_v4.py --date 2026-08-31
python theme_evidence_v4.py validate templates\theme_evidence_template_v4.csv
python theme_evidence_v4.py import D:\your_data\theme_evidence.csv
python theme_evidence_v4.py build --signal-date 2026-09-01 --commit-roster
python theme_evidence_v4.py open-sync D:\your_data\theme_member_open.csv --roster results\v4\themes\2026-09-01\verified_theme_roster.csv --roles results\v4\themes\2026-09-01\candidate_theme_roles.csv
python theme_forward_v11.py bind --plan results\v10\intraday_forward\SESSION_ID\plan.json --roster results\v11\themes\SIGNAL_DATE\verified_theme_roster.csv --roles results\v11\themes\SIGNAL_DATE\candidate_theme_roles.csv
python theme_forward_v11.py verify results\v11\theme_forward\SESSION_ID\theme_binding.json
python open_confirmation_v4.py validate templates\open_confirmation_template_v4.csv
python open_confirmation_v4.py import D:\your_data\open_snapshots.csv
python open_confirmation_v4.py evaluate --signal-date 2026-08-31 --trade-date 2026-09-01 --asof 09:35
python session_forward_v4.py prepare --signal-date 2026-08-31 --trade-date 2026-09-01
python session_forward_v4.py record --session-id YOUR_SESSION_ID --asof 09:25 --file D:\your_data\snapshot_0925.csv
python session_forward_v4.py assess --session-id YOUR_SESSION_ID
python session_forward_v4.py verify
python session_forward_v4.py monitor
python execution_contract_v4.py generate --session-id YOUR_SESSION_ID --asof 09:35 --code 600000 --portfolio templates\portfolio_state_template_v4.csv
python execution_contract_v4.py fill --contract-id YOUR_CONTRACT_ID --status VERIFIED_PAPER_FILL --fill-time 09:36 --fill-price 10.50 --fill-quantity 3000 --source 券商纸面记录
python execution_contract_v4.py exit --contract-id YOUR_CONTRACT_ID --session-offset 1 --t1-close 10.20 --actual-exit-price 10.20 --review-classification NO_ERROR
python execution_contract_v4.py verify
python execution_contract_v4.py monitor
python security_master_v4.py validate templates\security_master_template_v4.csv
python collect_security_master_v4.py --asof 2026-08-31
python historical_universe_v4.py --start 2023-11-28 --end 2026-08-31 --workers 4
python collect_delisted_security_master_v4.py --coverage-start 2023-11-28 --asof 2026-08-31
python sse_name_history_v4.py index --start 2022-11-28 --end 2026-08-31 --workers 4
python sse_name_history_v4.py download --workers 3
python sse_name_history_v4.py extract
python collect_sse_security_master_v4.py
python survivorship_sensitivity_v4.py
python survivorship_rebuild_v5.py
python emotion_challenger_v4.py --input data\v5\survivorship_rebuild\market_emotion.csv --output results\v5\emotion_challenger_rebuild
python compare_emotion_rebuild_v5.py
python sse_current_master_v5.py all --start 2023-11-28 --end 2026-08-31 --workers 4
python security_master_v4.py import data\v5\security_master_sse_current_evidenced.csv
python survivorship_rebuild_v6.py
python emotion_challenger_v4.py --input data\v6\point_in_time_rebuild\market_emotion.csv --output results\v6\emotion_challenger
python sse_breadth_master_v7.py all --start 2023-11-28 --end 2026-08-31 --workers 4
python security_master_v4.py import data\v7\security_master_sse_breadth_evidenced.csv
python survivorship_rebuild_v6.py --output data\v7\point_in_time_rebuild --result results\v7\point_in_time_rebuild_manifest.json
python emotion_challenger_v4.py --input data\v7\point_in_time_rebuild\market_emotion.csv --output results\v7\emotion_challenger
python emotion_model_lab_v7.py
python emotion_model_lab_v7.py --lock
python emotion_model_lab_v7.py --verify-lock
python emotion_forward_v7.py verify
python emotion_forward_v7.py monitor
python relay_model_lab_v8.py
python relay_joint_policy_v8.py
python relay_shadow_model_v10.py verify
python intraday_forward_v10.py verify
python intraday_forward_v10.py monitor
python intraday_readiness_v10.py
python risk_gate_lock_v12.py verify
python forward_risk_monitor_v13.py
python announcement_forward_v14.py bind --plan results\v10\intraday_forward\SESSION_ID\plan.json --manifest results\v3\context\SIGNAL_DATE_manifest.json --announcements results\v3\context\SIGNAL_DATE_announcements.csv --candidates data\v11\point_in_time_rebuild\leader_candidates.csv
python announcement_forward_v14.py verify
python security_master_v4.py import D:\your_data\security_master_intervals.csv
python security_master_v4.py audit
python risk_rule_audit_v4.py
python collect_missing_batches_v48.py --requests data\v48\jan_feb_missing_requests.json --output-root data\v48\first_board_minutes --batch-size 20 --max-batches 1
python build_minute_request_pack_v49.py --requests data\v48\jan_feb_missing_requests.json --cohort data\v48\jan_feb_full_pool_locked\cohort.csv --archive-root data\v35 --archive-root data\v40 --archive-root data\v46 --archive-root data\v48\first_board_minutes --output data\v49\minute_request_pack_20260906
python import_minute_csv_v47.py --input D:\your_data\historical_5min.csv --output data\v49\imported_5min_batch1 --source 同花顺导出 --interval 5 --timestamp-side end --volume-unit shares --amount-unit yuan --adjustment unadjusted --requests data\v49\minute_request_pack_20260906\remaining_requests.json

# 2026-09-07 续采后：完整分钟样本977/2243，剩余首板1266条；公开源恢复时可继续续跑
python collect_missing_batches_v48.py --requests data\v49\minute_request_pack_20260907\remaining_requests.json --output-root data\v49\first_board_minutes_recovery --batch-size 20 --max-batches 1
python historical_reseal_replay_v48.py --cohort data\v48\jan_feb_full_pool_locked\cohort.csv --archive-root data\v35 --archive-root data\v40 --archive-root data\v46 --archive-root data\v48\first_board_minutes --archive-root data\v49\first_board_minutes_recovery --daily data\v3\imported\daily-bars.csv --calendar data\v3\market_emotion.csv --features data\v16\full_pool_features_inventory_cutoff_20260831_v3\full_pool_eod_features.csv --output results\v48\jan_feb_full_pool_coverage_YYYYMMDD_vN
python full_pool_proxy_backtest.py --features data\full_pool_features_20260905_v2\full_pool_eod_features.csv --outcomes results\full_pool_outcomes_20260905\outcomes_LABEL_ONLY.csv --output results\full_pool_proxy_backtest_20260905_v3
python full_pool_feature_rebuild.py --observations results\first_board_inventory_20260905\candidate_observations.csv --daily data\v11\point_in_time_rebuild\point_in_time_current_daily_bars.csv --market data\v11\point_in_time_rebuild\market_emotion.csv --output data\full_pool_features_20260905_v2
python full_pool_selective_policy_experiment.py --features data\full_pool_features_20260905_v2\full_pool_eod_features.csv --outcomes results\full_pool_outcomes_20260905\outcomes_LABEL_ONLY.csv --output results\full_pool_selective_policy_20260905_v4
python security_master_v4.py audit --candidates results\first_board_inventory_20260905\candidate_observations.csv --master data\v11\security_master_intervals.csv --output results\v16\full_pool_security_audit_20260906
python security_master_v4.py audit --candidates results\first_board_inventory_20260905\candidate_observations.csv --master data\v11\security_master_intervals.csv --output results\v16\full_pool_security_audit_cutoff_20260831 --end-date 2026-08-31
python system_acceptance_audit.py --output results\current_acceptance_report_20260906_v21.json
python survivorship_rebuild_v6.py --current-bars data\v3\imported\daily-bars.csv --recovered-bars data\v4\historical_universe\delisted_daily_bars.csv --master data\v11\security_master_intervals.csv --base-market data\v5\survivorship_rebuild\market_emotion.csv --output data\v16\point_in_time_rebuild_20260906 --result results\v16\point_in_time_rebuild_manifest_20260906.json
python survivorship_rebuild_v6.py --current-bars data\v3\imported\daily-bars.csv --recovered-bars data\v4\historical_universe\delisted_daily_bars.csv --master data\v11\security_master_intervals.csv --base-market data\v5\survivorship_rebuild\market_emotion.csv --output data\v16\point_in_time_rebuild_cutoff_20260831_v2 --result results\v16\build_manifest_cutoff_20260831_v2.json --end-date 2026-08-31
python full_pool_feature_rebuild.py --observations results\first_board_inventory_20260905\candidate_observations.csv --daily data\v16\point_in_time_rebuild_cutoff_20260831_v2\point_in_time_current_daily_bars.csv --market data\v16\point_in_time_rebuild_cutoff_20260831_v2\market_emotion.csv --output data\v16\full_pool_features_inventory_cutoff_20260831_v3 --end-date 2026-08-31
python full_pool_outcomes.py --observations results\first_board_inventory_20260905\candidate_observations.csv --daily data\v16\point_in_time_rebuild_cutoff_20260831_v2\point_in_time_current_daily_bars.csv --calendar data\v16\point_in_time_rebuild_cutoff_20260831_v2\market_emotion.csv --output data\v16\full_pool_outcomes_inventory_cutoff_20260831_v3 --outcome-end-exclusive 2026-09-01
python full_pool_proxy_backtest.py --features data\v16\full_pool_features_inventory_cutoff_20260831_v3\full_pool_eod_features.csv --outcomes data\v16\full_pool_outcomes_inventory_cutoff_20260831_v3\outcomes_LABEL_ONLY.csv --output results\v16\full_pool_proxy_backtest_inventory_cutoff_20260831_v15 --validation-start 2025-07-28 --validation-end 2025-12-26
python full_pool_proxy_walkforward_v17.py --features data\v16\full_pool_features_inventory_cutoff_20260831_v3\full_pool_eod_features.csv --outcomes data\v16\full_pool_outcomes_inventory_cutoff_20260831_v3\outcomes_LABEL_ONLY.csv --output results\v17\full_pool_proxy_walkforward_cutoff_20260831_v6
python full_pool_role_backtest_v18.py --features data\v16\full_pool_features_inventory_cutoff_20260831_v3\full_pool_eod_features.csv --outcomes data\v16\full_pool_outcomes_inventory_cutoff_20260831_v3\outcomes_LABEL_ONLY.csv --output results\v18\full_pool_role_backtest_cutoff_20260831_v11
python current_full_pool_snapshot.py --limit-up data\v3\raw\20260904\limit_up.csv --failed-board data\v3\raw\20260904\failed_board.csv --manifest data\v3\raw\20260904\manifest.json --security-master data\v49\security_szse_asof_20260904\security_master_szse_verified.csv --security-master data\v49\security_sse_all_asof_20260904_v7\security_master_sse_current_evidenced.csv --signal-date 2026-09-04 --output results\current_full_pool_20260904_v7
python current_daily_review.py --snapshot results\current_full_pool_20260904_v7\all_closing_and_failed_candidates.csv --snapshot-manifest results\current_full_pool_20260904_v7\manifest.json --market data\v3\market_emotion.csv --selective-policy results\full_pool_selective_policy_20260905_v4\report.json --announcements results\v3\context\full_pool_20260904_v5\2026-09-04_announcements.csv --previous-limit data\v3\raw\20260904\previous_limit.csv --asof 2026-09-07T11:55:00+08:00 --output results\daily_review_20260904_v10
python public_live_preflight_v17.py --trade-date 2026-09-07 --output results\v17\public_live_preflight\2026-09-07_v1
python public_live_capture_v1.py --trade-date 2026-09-07 --asof 09:25
python public_live_capture_v1.py --trade-date 2026-09-07 --asof 09:35 --no-breadth
python public_minute_capture_v1.py --trade-date 2026-09-07 --asof 09:35
python public_live_monitor_v1.py --trade-date 2026-09-07
python public_forward_audit_v16.py
python broker_execution_import_v16.py --decision results\v16\public_live_monitor\交易日\运行ID\decisions\0935\decision.json --orders D:\导出\当日委托.csv --fills D:\导出\当日成交.csv --captured-at 2026-09-07T15:10:00+08:00
python broker_exit_import_v16.py --entry-report results\v16\broker_execution\买入日\导入ID\report.json --exit-orders D:\导出\卖出委托.csv --exit-fills D:\导出\卖出成交.csv --exit-trade-date 2026-09-08 --exit-plan D:\计划\冻结退出计划.json --exit-confirmations D:\证据\退出触发.json --sessions data\v3\market_emotion.csv --captured-at 2026-09-08T15:10:00+08:00
python trade_path_metrics_v16.py --entry-report results\v16\broker_execution\买入日\导入ID\report.json --marks D:\导出\每日标记.csv --sessions data\v3\market_emotion.csv --exit-plan D:\计划\冻结退出计划.json --asof 2026-09-10T15:05:00+08:00
python trade_path_aggregate_v16.py --reports results\v16\trade_path_metrics --asof 2026-09-10T15:05:00+08:00
python trade_report_aggregate_v16.py --reports results\v16\broker_exit --asof 2026-09-10T15:15:00+08:00
python live_decision_bridge_v1.py --review results\daily_review_20260904_v6\review.json --quote-manifest results\v16\public_live\2026-09-07\0935\采集ID\manifest.json --minute-manifest results\v16\public_minute\2026-09-07\0935\采集ID\manifest.json --output results\v16\live_decision_20260907_0935_v1
```

`current_daily_review.py` 的盘后报告会同时列出首板、连板、情绪锚，以及“成龙潜力/当前买入机会”尚未可判定的原因。每只三板及以上高位候选都生成映射记录；竞争核心身份尚未核验时也不消失，而是带阻断项保留。同细分题材、低位先锋/核心、非一字代理且无硬风险时才给正式研究映射；二手题材只能列出至多三个“叙事备选”，明确不是补涨候选。V169进一步把题材证据拆成业务支线归属、当日主动驱动和预期状态，V170把首板与连板跨题材竞争拆成两条赛道，V171把严重异常规则距离与路径潜力分开，V174固定输出六种角色及分组主候选或空仓。`--previous-limit` 保存平均、中位、尾部和逐板位晋级；`--height-anchor-evidence`、`--opening-gap-evidence`、`--opening-gap-emotion-evidence`与`--index-emotion-divergence-evidence`分别绑定执行和指数—短线生态诊断，但都不能自行授权排序或入场。竞价、开盘、实际成交和退出证据缺失时保持 `UNAVAILABLE`。最新9月16日重放见 `results/daily_review_20260916_v32/`：指数层与接力生态层分开，仍因阈值、校准概率、竞争身份和次日开盘证据不足而保持现金。

V172对“板位断层就是稀缺龙头”的常见说法做了单独检验：708条最高板观察中，两板以上断层的次日晋级确实高于无断层组，但用中位晋级和尾部亏损划分的断层质量并不稳定，条件B+2代理也没有通过日期聚类区间。历史证券资格又不完整，因此断层只保留为身份与情绪背景，不能作为正向排序因子。报告见 `results/v172/ladder_gap_tradeoff_20260917_v1/report.json`。

V173进一步审计“一字后首次开板就是健康分歧”。在504个交易日、5,103条二至五板非一字连板中，一字后首次换手板的次日晋级高于普通换手连板，但B+2条件代理优势跨零；连续一字更久也没有证明更安全。前70%日期确定换手阈值后，后30%日期的高换手首次开板组反而弱于中换手组。系统因此只把首次开板视为身份与延迟价格发现并存的状态，仍要求分时承接、题材反哺和可成交证据。报告见 `results/v173/first_open_price_discovery_20260917_v1/report.json`。

V175继续审计“高标变弱就去低位”。在461个高度锚交易日中，174日最高板次日未晋级；若只看“是否至少有一只首板或低位连板晋级”，174日全部显示有去向，说明这种常见复盘口径几乎是恒真条件。改用首板和低位连板各自晋级率、固定10个百分点主导差后，低位连板占优116日、两赛道接近45日、首板占优12日。低位连板占优与较低的次日不利情绪状态相关，但收益反馈和炸板率差异不稳定，且赛道晋级率是次日实现标签，不能泄漏进前一晚模型。它只进入情景验收，不授权自动高切低或个股加分。报告见 `results/v175/anchor_capital_destination_20260917_v2/report.json`。

V176补齐竞价流和题材传播顺序的前向采集合同。9:20至9:24:50的匹配额变化只保存为Level-1净衰减/净增长，真实撤单、补单、队列所有者和“骗单/控盘”意图继续标记不可获得；题材分钟层分别记录低位先触板、高标先触板、同一分钟触板，以及低位触板是否落在高标开板至回封之间。即使顺序成立也不宣称因果，因为一分钟数据不能排除共同催化和秒级顺序。当前没有真实前向券商竞价、分钟题材全名册或Level-2样本，状态为 `NO_FORWARD_OBSERVATIONS`，只完成采集准备，不进入排序和买点。合同见 `results/v176/intraday_flow_causality_readiness_20260917_v1.json`，题材名册模板见 `templates/theme_intraday_roster_v176.csv`。

V177补上“平均赚钱是否由少数强股贡献”的盲点。503个严格次日移位的历史交易日中，昨日涨停平均反馈为正但中位反馈不为正的窄赚钱状态有80日；其下一交易日的昨日涨停中位反馈比广泛赚钱状态低约0.75个百分点，bootstrap区间未跨零，但次日综合不利情绪状态率差仍跨零，后30%日期又只有15个窄赚钱样本。因此每日复盘强制输出均值、中位、两者差值和赚钱分布；窄赚钱只能作为环境风险解释，不能给个股加分或直接授权买入。报告见 `results/v177/profit_concentration_emotion_20260917_v1/report.json`。

V178补上“涨停扩张是否伴随更强炸板压力”。在248个严格相邻交易日的涨停扩张样本中，脆弱扩张61日、干净扩张187日；前者次日大分歧/退潮率高约16.70个百分点，中位晋级率低约6.96个百分点。最近30%日期的风险方向仍一致，但前70%发现段的风险率差区间跨零，因此未通过跨时间稳定闸门。每日复盘现在展示 `CLEAN_EXPANSION/FRAGILE_EXPANSION`，但该标签不能直接改变情绪许可、个股排序、仓位或入场。报告见 `results/v178/breadth_quality_emotion_20260917_v2/report.json`。

V179单独审计“一字板越多，次日越容易兑现”。503个严格相邻交易日中，前70%开发段高一字占比组的次日不利状态率反而低约7.50个百分点，后30%验证段则高约4.94个百分点；两段方向相反且区间均跨零，四分位也没有单调风险关系。正式一致状态仅11日，无法支撑概率判断。因此一字占比只描述拥挤、延迟价格发现和市场可买性；个股一字无法成交仍被拒绝，但不能据此推导整个市场次日必分歧。报告见 `results/v179/consensus_crowding_tradeoff_20260917_v1/report.json`。

V180把涨停扩张质量延伸到B+1至B+3市场路径。脆弱扩张未来两日、三日至少一次进入大分歧/退潮的历史比例分别为77.05%和90.16%，干净扩张分别为63.10%和79.68%。最近30%日期差异区间未跨零，但前70%发现段均跨零，所以系统不输出两三日校准概率，只显示历史风险情景；更不能把市场路径直接转换成某只候选的三日持有规则。报告见 `results/v180/breadth_quality_path_20260917_v1/report.json`。

V181重新审计“最高板就是最值得买”。714条最高板观察中，唯一最高板代理成功率20.86%，并列最高板20.00%；按时间切分后的150个验证日里，板位基线与加入唯一性、情绪和量价的挑战模型每日首选成功率都为18.0%，挑战模型AUC更低，Brier与对数损失更差。标签已把次日未触板和次日一字不可买保留为负例，但仍没有真实竞价成交和完整退出，且验证期并非封存盲测。因此最高板继续优先做情绪锚审查，却不能自动成为连板主候选、概率首选或买点。报告见 `results/v181/height_anchor_buyability_20260917_v1/report.json`。

V182继续检验“开得越高越值得追”。25,826条首板与一至五板候选的后段151日中，5%—7%开盘档的可买B+2正收益代理为24.46%，高于7%档为22.67%，但差异区间跨零；高于7%档有31.91%是一字不可买，可触板路径B+1大面代理约39.47%，5%—7%档约34.17%，后段风险差仍轻微跨零。因此开盘涨幅不是单调正向因子：高于7%保留默认风险标记，0%—5%与5%—7%也必须继续核验真实承接、回封、题材同步和价格上限。正式开盘结果只能在集合竞价撮合后获得，9:30进入连续竞价；它禁止进入盘后模型，也不能在看见结果后冒充已按同一开盘价成交。纠正版报告见 `results/v182/opening_gap_payoff_tradeoff_20260917_v3/report.json`。

V183继续检验情绪是否可以覆盖高开规则。5%—7%相对高于7%的综合可买B+2代理在支持性环境前后段均为正，但区间跨零；加速/一致组和分歧/退潮组的方向前后翻转，支持性与逆风环境的差中差同样未稳定。结论是情绪环境只能决定确认强度，不能跳过实际承接、回封、题材同步和价格上限；环境好不能自动豁免高于7%的风险，环境差也不能仅凭高开机械取消。报告见 `results/v183/opening_gap_emotion_interaction_20260917_v1/report.json`。

V184补上“低开弱转强”的事后分母陷阱。验证段151日里，全部低开候选进入可买B+2正收益路径的比例只有3.60%；若事后只留下后来触板且非一字的股票，条件比例变成38.10%，放大约10.57倍。后来触板在09:30不可知，不能作为开盘特征。低开既不自动买，也不自动剔除，只能等待真实收回昨收与VWAP、修复量能、题材同步和竞争者淘汰。报告见 `results/v184/weak_to_strong_hindsight_audit_20260917_v1/report.json`。

V185补上“首板家数多就说明一进二好做”的基数陷阱。验证段151日里，每天至少一只首板晋级的比例高达99.34%，因此这个条件几乎恒真，不能证明选股有效。开发段冻结的不超过22只/高于49只供给组，在验证段的单票晋级率约25.54%/18.78%，高供给没有正向优势，但差异区间仍跨零，不能机械反推首板少就必然更好。首板数量只描述市场广度，必须继续核验真实题材梯队、主动核心、单票可成交性和买后收益。报告见 `results/v185/first_board_supply_base_rate_audit_20260917_v1/report.json`。

V186把“晋级身份”和“买后持续收益”做了纠正全分母对照。验证段151日中，次日晋级率约23.79%，可触板、非一字且B+2为正的路径约10.23%；已晋级股票仍有约62.88%没有形成可买B+2正收益，已晋级且B+2、B+3同时为正的比例仅31.36%。首板和连板方向一致。因此晋级概率只能决定观察优先级，不能冒充完整交易胜率、当前买点或2—3日持股依据。报告见 `results/v186/promotion_payoff_separation_audit_20260917_v1/report.json`。

上交所公告解析为 `REVIEW_REQUIRED` 时，证券状态链默认整体降级为 `inferred`。只有人工复核台账同时匹配股票代码、公告 ID 与官方文本 SHA-256，并记录原文证据、复核时间和复核人后，才能把“未发生状态转换”或“更正确认既有事件”从未决公告中排除；哈希不匹配会直接失败，不能用当前名称倒推历史资格。

### 无需截图的盘中公开行情

`public_live_capture_v1.py` 默认自动发现最近一次标准化的全部收盘涨停池，并在每次运行时生成新的不可覆盖归档：

- 候选价格层：覆盖昨日全部首板和连板，保存公开行情的提供方时间、接收时间、行情年龄、开高低、昨收、累计量额、10%涨停价代理及是否处于涨停价代理。
- 情绪广度层：默认使用 `data/v3/imported/current-universe.csv` 抓取主板非ST名称池，报告候选覆盖率、新鲜覆盖率、上涨/下跌家数及涨跌停数量代理；若只求低延迟可加 `--no-breadth`。
- 延迟控制：每个时点先单独抓取全部首板/连板候选并冻结其接收时间，再抓其余市场广度；候选新鲜度不再受约3000只全市场请求耗时拖累。
- 失败关闭：周末、非采集窗口、行情停留在上一交易日、候选缺失或接口失败时，统一输出 `CASH_PUBLIC_DATA_INCOMPLETE_OR_NOT_ON_TIME`。
- 执行边界：即使全部公开价格及时，也只输出 `CONDITIONAL_PUBLIC_PRICE_FRESH`。真实竞价撤单、Level-2封单队列、换手回封顺序和成交仍不可验证，不能据此把观察候选冒充为已确认买点。

建议在 09:20、09:24:50、09:25、09:30、09:35 和 10:00 分别运行一次；每次归档位于 `results/v16/public_live/交易日/时点/采集ID/`。这解决了截图中的价格与时间戳采集，但要获得自动 `ENTER`，仍需接入券商或终端导出的可核验逐笔/分钟与成交数据。

如果不想手工运行六次，可在交易日 09:20 前启动 `public_live_monitor_v1.py`。它会自动发现最新盘后复盘，校验它与候选池都属于交易日前最近一个已归档交易日，再等待六个时点；09:25、09:35、10:00同时采集市场广度并自动生成三阶段决策报告。错过窗口、旧候选、接口失败、候选池不一致和过期行情都会写入事件账本并失败关闭。只需采集、不生成决策时显式使用 `--capture-only`。

监控V16.4会把候选、盘后复盘、交易日历、报价清单、分钟清单及三份决策的SHA-256写进事件链。可使用 `python public_live_monitor_v1.py --verify-manifest 运行目录\manifest.json`复核；该证明属于本地防篡改证据，不是第三方可信时间戳。

`public_forward_audit_v16.py`会累计审计所有实时运行，不允许从同一日期的多次运行中挑最好的一次：重复日期直接失去前向资格，失败、迟到和空仓全部保留。公开层的 `WATCH/CANCEL/CASH` 只计为决策观察，绝不冒充成交或盈利交易；只有未来接入可核验成交与退出后，完整交易盈利率、平均净收益、大面率和回撤才开始形成有效分母。

`broker_execution_import_v16.py`用于导入同花顺或券商的当日委托与成交CSV，并绑定某个事前生成的实时决策。程序核对委托编号、成交编号、交易日、下单与成交时间、代码、方向、委托/成交数量、100股单位、冻结主选和最高可接受价格。真实成交若发生在 `WATCH/CANCEL/CASH` 下、早于确认、不是冻结主选或价格超限，会保留为 `USER_EXECUTION_OFF_PLAN_NOT_STRATEGY_SAMPLE`，不得进入策略胜率。部分成交、未成交、费用缺失也分别披露；只有买入证据仍不是完整交易，必须继续取得真实退出记录。

`broker_exit_import_v16.py`继续绑定真实卖出委托/成交、事前冻结退出计划、退出触发证据和交易日历。它拒绝买入日卖出，禁止累计卖出超过买入，未成交/部分卖出保持未平仓。只有入场合规、退出审计通过、全部卖出且买卖费用完整时才生成 `strategy_complete_trade_eligible=true` 与策略净收益；否则可以报告用户实际已实现盈亏，但不进入战法完整交易分母。

`trade_path_metrics_v16.py`在完整成交记录上计算 B+1、B+2、B+3 的扣费后收盘估值、两日持续为正、MFE/MAE 和跌停/停牌等不可卖状态；它要求价格来源、观察时点、可用时点和未复权口径均已核验。`trade_path_aggregate_v16.py`把多个不可变路径报告汇总为期限正收益率、未解决样本、不可卖比例和首板/连板分组计数。两者都明确标记为 `PATH_VALUATION_ONLY`，不会把收盘估值或盘中最高价冒充真实卖出盈利率；真实完整交易盈利率仍只来自买卖成交与费用完整的退出报告。

`trade_report_aggregate_v16.py`只把 `strategy_complete_trade_eligible=true` 的完整生命周期纳入完整交易盈利率、平均净收益、盈亏比、利润因子和大面率；未成交、部分成交、未平仓、费用缺失及计划外成交保留在未解决分母中。它不凭交易收益序列推算账户最大回撤，账户回撤必须继续使用带资金、仓位和未平仓盯市的账户账本。

纸面执行契约已升级为 `v4-execution-contract-2`。首板只有在盘后身份已核验，或竞争身份属于题材先锋、换手核心、唯一活口、容量核心时才能进入开盘确认；随机首板仍被拒绝。6板以上不再因高度直接失去确认资格，但仓位上限降至2%，8板以上降至1%，并继续受情绪、题材、公告、可成交性和T+1压力预算约束。退出观察窗口为B+1至B+3；若跌停、停牌或无流动性导致卖出受阻，可以登记新的更晚 `session_offset`，但在真正成交前净收益始终保持未解决。

聚合报告还会把未平仓、部分卖出、费用不完整和计划外/不合规记录分别计数；这些状态不会被合并成“亏损交易”，也不会进入完整交易盈利率分母。

09:35和10:00还会自动调用 `public_minute_capture_v1.py`，把“承接良好、跌破放量、回封”转换为可复核的1分钟代理。分钟内先开板再回封等更细事件仍可能被压缩，因此这些字段只能用于观察、消融和前向纸面验证，不能证明涨停价排队成交。

`live_decision_bridge_v1.py` 将盘后完整首板/连板复盘、公开报价和分钟代理按交易日、时点、候选池与文件哈希绑定，生成统一的首板清单、连板清单、情绪摘要、主选证据和禁止理由。只要盘后没有冻结主选、行情陈旧、候选缺失或风险条件触发，结果就是 `CASH/CANCEL`；公开行情链即使完整也只能输出 `WATCH`，不会越权输出 `ENTER`。

第三版当前仍为 `DIAGNOSTIC_ONLY`。只有历史盲测、风险标签和冻结后的60个前向纸面信号全部达到门槛，才允许升级状态。

### V195—V196：交易日历纠错与收益目标分离

V195发现此前首板研究把“已经采集复盘的日期”误当成交易所交易日历。2026-09-01之后实际还有09-02和09-03两个交易日，但稀疏复盘序列把下一日错误指向09-04，导致晋级与B+1/B+2/B+3标签向后错移。现在交易日历由上交所官方会期查询与80只候选逐日行情截面交叉核验；缺失即时下一交易日涨停池时，标签记为不可获得，禁止用更晚日期补位。纠正证据见 `results/v195/trading_calendar_integrity_20260917_v4/report.json`，首板样本由1,175条重建为1,116条，09-01的59条错误后继样本被排除。

V196不再以“一进二成功”冒充“买后有溢价”，而是直接建立“次日开盘代理可成交且B+2扣费后为正”的目标，并另估B+1大面风险。一字排队不可验证计为该可执行目标的负例；缺失B日行情和B+2尚未结算保持不可获得。时间隔离的10个完整评价日里，加入情绪因子的每日第一为4/10个B+2正收益，但B+2平均净收益仍约-3.77%，B+1大面率30%；风险调整排序更差，仅2/10。样本短且已被查看，所有概率、当前排名和阈值继续锁定为研究用途，不授权买入。报告见 `results/v196/first_board_profit_ranker_20260917_v2/report.json`。

V197进一步检验“复盘中能找到题材”是否就是题材强。10个完整收益评价日、456条首板中，仅4日56条具有次日前可得的题材名单，模型合格题材行为仍为0，全市场题材完整日为0。名单覆盖组的一进二率、可执行B+2正收益率和平均B+2收益均未优于未知组，日期聚类区间全部跨零。因此题材资料覆盖只表示注意力和证据可得性；业务归属、当日主动催化、预期饱和、板块梯队和买后赔率必须分开，未知题材不得填成弱题材。报告见 `results/v197/theme_profit_coverage_20260917_v1/report.json`。

V198补齐“胜率与期望收益分离”。情绪收益排序的每日第一虽然把B+2正收益率从基础收益排序的30%提高到40%，但平均盈利约7.35%、平均亏损约-11.19%，盈亏比仅0.66，利润因子约0.44，平均B+2净收益约-3.77%。按每次全仓顺序复利仅作压力代理时，累计约-36.41%、最大回撤约-47.81%；它不是账户回测，却清楚说明几次大亏足以吞掉多次盈利。系统现在要求胜率、平均收益、盈亏比、利润因子、大面率和回撤同时过闸，70%挑战目标也不能替代正期望。报告见 `results/v198/first_board_payoff_tail_20260917_v1/report.json`。

V199把“大面风险”改成收益排序后的独立拒绝层：收益模型先冻结每日第一，风险模型只能决定保留或空仓，被否决后不补选第二名。507条可观察成交风险样本的大面基础率约11.05%，最优时间隔离模型AUC仅0.576；描述性的最高风险20%大面率反而只有8.82%，没有形成稳定风险浓缩。固定阈值在只保留2—3笔时虽暂时没有大面，但这些交易B+2仍全部亏损；把收益概率直接乘以风险补集，则每日第一平均B+2由-3.77%恶化至-7.12%，大面率由30%升至40%。因此阈值、模型、概率配仓和入场全部不获授权，下一步必须在冻结策略后的新时段验证风险—覆盖率与正期望。报告见 `results/v199/first_board_big_loss_veto_20260917_v1/report.json`。

V200补上“开盘信息与成交价不能同事件穿越”。集合竞价匹配后的正式开盘涨幅只能用于后续9:30/9:35确认；如果先观察正式开盘结果，再仍按同一集合竞价开盘价记成交，就是不可执行的同事件泄漏。507条开盘代理成交样本中，5%—7%和高于7%开盘组的描述性B+1大面率约25.81%和25.00%，但冻结每日第一的三次大面同时出现在-3.78%、-2.10%和+6.54%开盘，不能把单一开盘区间当安全答案。当前完整盘中序列为0，所以9:30/9:35模型必须等待真实分钟证据并按新的成交价格重算收益，不能继承开盘成交代理。报告见 `results/v200/opening_information_causality_20260917_v3/report.json`。

V201补上“T+1下买入日止损不可执行”。上交所、深交所现行交易规则均规定投资者买入证券在交收前不得卖出，普通主板A股不属于规则列举的当日回转品种。1048条首板开盘代理成交中，买入日浮亏曾触及-4%/-5%/-7%的比例约42.65%/30.25%/13.65%，但这些阈值在买入日的可执行退出数均为0；B+1另有4条一字跌停明确不可卖、5条跌停开盘但排队成交不可核验。故“盘中跌破止损价”只能是风险报警，不能在回测中记作成交；真正的尾部控制必须前置为不买、降仓和组合风险预算，B+1退出也必须服从真实可成交状态。报告见 `results/v201/t1_unexitable_loss_audit_20260917_v2/report.json`。

V202继续补上“退出信息与成交价不能同事件穿越”。1017条B+1开盘与收盘均可比较样本中，固定开盘退出代理平均约-1.11%，固定收盘估值约-1.25%；若事后逐笔挑开盘和收盘中较优者，平均会变成+0.47%，相对较优固定规则虚增约1.58个百分点。开盘更好528笔、收盘更好467笔，不存在无需条件验证的永远最优退出时点。对于早盘不打开电脑的使用方式，集合竞价退出必须在正式开盘产生前冻结并以真实成交回填；若先观察开盘再决定卖出，只能使用9:30后的新成交价。当前完整分钟退出序列仍为0，盘中条件只能作为条件计划，不能冒充已执行。报告见 `results/v202/exit_information_causality_audit_20260917_v1.json`。

V203新增“指数层不等于涨停接力生态层”。681个双指数交易日中，指数同涨但昨日涨停中位不正的47日，次日大分歧/退潮率约42.55%，同步改善组约35.27%；然而发现段和验证段按周聚类区间都跨零，验证背离组只有14日，所有复制闸门均失败，近期首板收益也没有验证简单背离组更差。因此指数翻红不能覆盖昨日涨停中位亏损、炸板和断板负反馈，但该背离目前也不能直接否决、重排个股或授权入场。最新审计见 `results/v203/index_emotion_divergence_20260917_v3/report.json`，已接入 `results/daily_review_20260916_v32/`。

V48新增严格的五分钟因果回封重放：已完成K线只能产生信号，成交机会只能由下一根K线标注，一字排队始终保持不可验证；同时分别计算B+1、B+2、B+3扣费后收盘估值。现有2024年1—2月连板开发样本在每日一只固定基线下只有8个价格机会代理，B+2/B+3同时为正2/8，B+3大面代理6/8；10:00仍封住的替代基线也只有3个价格机会代理，不能支持“回封即安全”。全体首板与连板冻结池为2,243条，当前执行证据覆盖767条（连续板522/522、首板245/1,721），正常模式因缺失1,476条首板而固定输出 `BLOCKED_INCOMPLETE_CANDIDATE_UNIVERSE`，不会在残缺池强行选第一名。完整零成交K线独立标为不可交易，不冒充价格缺失或成交。详见 `results/v48/回封与持股窗口阶段报告.md`；状态仍为 `DIAGNOSTIC_ONLY`。

在覆盖起点形成的10个连续完整交易日中，系统另冻结337条首板与连板同池开发样本（首板238、连板99）。最早回封基线8个候选日里有6次选到首板，仅3次出现下一根K线的价格机会；这3次B+2、B+3均为负且大面代理均为3/3。样本极小，不能定阈值，但已经否定“全市场谁最早回封就买谁”这一单因子方案；盘后身份、题材竞争和情绪许可必须先于回封确认。

当前目标口径已经改为“全部首板与连板同池排序、每日最多一个条件候选、确认失败即空仓”，详见 `results/龙头战法目标V31.md`。现有全池探索回测在101个验证日里强制每日选一只仅命中23次价格机会代理，2025年8月为3/21；这不是实际交易胜率，但已经证明不能把“每天必须买一个”作为系统目标。

2026年9月6日已重新审计包含首板的全池 37,325 条观察：全部获得时间点证券状态覆盖（37,325/37,325，含首板与各连板层级），因此 `point_in_time_status_complete` 已通过；但历史候选母池仍未证明完整，`candidate_universe_complete=false`，所以整个部署闸门仍保持 `DIAGNOSTIC_ONLY`。V23验收同时绑定特征重建清单、严格截止窗口的全池结果、B+2目标覆盖、滚动 walk-forward、角色回测和V48盘中执行覆盖；V48当前覆盖767/2,243，完整性闸门失败，但“残缺池必须0选择”的防绕过闸门通过。研究证据区块明确 `complete_trade_count=0`，不会越权成为上线条件。最新完整验收见 `results/current_acceptance_report_20260906_v23.json`。

同日增加严格历史截止日回测：截至2026-08-31，重建得到669个市场日、9,063条连板研究候选，候选状态和市场宽度状态均为完整覆盖，且不会把近期缓存重新拼回历史窗口；但这只解决“当时是否具备交易资格”的时间点覆盖，不等于历史股票母池完整，`universe_complete=false` 仍然保留。对应证据为 `results/v16/point_in_time_rebuild_manifest_cutoff_20260831_v2.json` 和 `results/v16/full_pool_security_audit_cutoff_20260831/security_eligibility_audit.json`，系统状态仍为 `DIAGNOSTIC_ONLY`。

在同一截止窗口上，已把首板与连板库存合并进入探索性排序回测：103个验证日、5,156条候选；综合排序日选中连板 103 次、首板 0 次，说明当前“全池只选一只”的排序天然偏向高板位。为避免掩盖组间差异，报告另做组内独立选择：连板身份+情绪+量价约23/103，首板约15/103；这些都是日线机会代理，不是实盘胜率。2025年8月综合策略为3/21。严格截止后的特征文件无未来日期混入，详见 `results/v16/full_pool_proxy_backtest_inventory_cutoff_20260831_v6/report.json`，状态仍为 `EXPLORATORY_ONLY`。

新增角色内标准化实验：首板和连板分别训练，使用相对各自训练基线的对数优势评分再合并排序。它把综合选择改为首板24次、连板79次，但代理成功率降至19/103（18.4%），8月为2/21；因此只证明“减少板位偏置”与“提高代理表现”存在冲突，暂不接入主策略。详见 `results/v16/full_pool_proxy_backtest_inventory_cutoff_20260831_v7/report.json`。

回测标签现已补齐买入后的 B+1、B+2、B+3 收盘估值和 B+2/B+3 同时为正比例，均保留“日线机会代理、不是成交盈利”的标记。全池标签 manifest 同时报告覆盖分母：B+1/B+2/B+3 可计算样本分别为12,801/12,762/12,735，正收益率约43.1%/37.6%/37.2%，B+2 与 B+3 同时为正约31.3%。策略选中样本另行报告，不能把全池基准当作实际买入后的溢价率。V10起“当前买入机会”明确以次日触及代理+B+2收盘净估值为主要标签；没有B+2字段的旧输入才回退B+1，并在报告标记 `B+1_FALLBACK`。V11进一步把B+2缺失样本从标签分母剔除，V12新增目标覆盖审计，V13把正收益和大面风险区间统一纳入全池路径摘要，V14进一步把缺失的次日触及证据也从严格标签分母剔除，V15使用重建并核验的一位情绪状态字段重新固定回测输入，避免乱码显示字段影响情绪分组。详见 `results/v16/full_pool_outcomes_inventory_cutoff_20260831_v3/manifest.json` 和 `results/v16/full_pool_proxy_backtest_inventory_cutoff_20260831_v15/report.json`。

已增加7个固定季度/近季度窗口的滚动样本外回测：每个窗口只使用开始日前可用标签，清除跨窗口 episode，且不使用验证集调参。V6按B+2延续标签重跑，并同时剔除B+2未知与次日触及证据未知样本；综合身份+情绪+量价模型代理成功率约35.8%，角色内标准化约36.5%，后者选择首板123次、连板277次。由于有效分母随B+2可用性变化，这些数字不能与旧B+1口径直接比较；结果仍用于发现稳定性和板位偏置，不构成完整交易盈利率或实盘授权。详见 `results/v17/full_pool_proxy_walkforward_cutoff_20260831_v6/report.json`。

角色拆分回测进一步显示：情绪锚按最高板定义，成龙潜力用“次日收盘晋级”代理，当前买入机会用“次日触及且B+2收盘估值为正”代理；旧数据缺少B+2时才回退B+1并显式标记。V11使用核验后的一位情绪状态字段，并把缺失的次日触及证据同样视为未知并排除；V9—V11新增晋级、路径正收益率和大面代理率的Wilson 95%区间，避免小样本被误读；当前买入机会仍只有146个已知样本，代理正收益39.7%，其中首板38个、连板108个。报告按 `market_emotion_state` 和首板/连板分组拆分路径收益，缺失路径保持未知，不填成0。该结果只用于建立角色分工，不能把晋级代理当成买入胜率。详见 `results/v18/full_pool_role_backtest_cutoff_20260831_v11/report.json`。
当前买入机会的情绪分组路径提示：启动样本 B+1 平均约 -8.03%、B+3 大面代理约66.7%（仅6个已知样本）；大分歧 B+3 大面约52.5%（40个）；退潮约50.0%（28个）；升温 B+3 平均约 -6.42%（19个）。这些是日线收盘估值代理，样本量和成交证据均不足，暂只能强化环境过滤，不能转成固定概率。

2026年9月4日首次按新口径生成完整当前观察池：38只主板收盘涨停中含31只首板、6只二板、1只五板，另列44只主板炸板。深市19只涨停有官方证券状态证据；沪市官方查询超时，19只保持未验证。五维情绪显示赚钱效应、断板亏钱效应、高度反馈和修复质量均逆风，真实细分题材又缺时点证据，故每日复盘输出 `CASH`、仓位0，而不是沿用旧CLI只看7只连板后强选。

公告风险采用双时点约束：只有公告的保守生效时刻和采集时刻都不晚于绑定时刻，才能进入次日硬风险判断。只有日期的同日公告统一按23:59:59处理，因此不会泄漏进当日15:00盘后特征，但可以在次日开盘前生效。明确的股票交易风险提示/异常波动公告按5日硬风险窗口处理；减持计划完成、实施完毕、期限届满或提前终止归入已结束历史事项，不得误判为新的减持压力。已经错过09:15的历史日期只能做事后审计，禁止伪装成前向绑定。

前向验证不再允许手工事后创造信号。`freeze`先封存模型、代码、冻结时数据末日及SHA-256；`score --record-forward`只接受冻结后、且为本地最新交易日的评分；评分原文和候选进入逐事件哈希链；`paper-log`只能为已经预注册的日期/代码补录成交结果，不能改变当时决策，也不能覆盖旧结果。`forward-verify`同时检查事件链、冻结清单和评分快照。该证据属于 `LOCAL_TAMPER_EVIDENT_ONLY`，可以发现本地后改，但不是第三方可信时间戳。

V3.8验证期的联合诊断阈值为0.45：11笔日线代理成交中晋级9笔（81.8%），大面1笔（9.1%），但只有约1.08笔/活跃周且不足最低31笔证据，因此没有部署阈值。相同规则在已复用的开发留出期为12笔代理成交、晋级75.0%、大面33.3%，三段滚动测试也都未通过联合闸门。系统据此自动空仓，而不是用验证期小样本宣称达到70%。

按新的 B+2 延续标签重算选择性策略后，校准窗口仍为 `NO_SELECTIVE_PROXY_THRESHOLD`，默认决策为 `CASH`；最低样本阈值下代理精确率仍未达到70%，没有任何固定阈值同时满足70%精确率、至少30笔和每活跃周3—5笔。详见 `results/v16/full_pool_selective_policy_20260906_v6/report.json`。

终局高度模型在1497条开发留出样本上，众数高度完全命中率69.8%、误差不超过一板88.3%、众数MAE 0.496（只预测停在当前板的基线为0.510）；“至少再走两板”的Brier技能分为+0.042。名义80%区间实际覆盖97.1%、平均宽2.35板，说明区间偏保守，且四、五板平均终局高度存在高估；结果只作诊断，不作为持股承诺。

2026年8月31日成为首个冻结后预注册快照：冻结ID `61ef3e85ff0edfe1`，模型/代码/数据边界及快照哈希链校验通过。当天88只涨停、30只炸板、11只跌停，18只连板观察输出0个 `ENTER`、0个 `WATCH`、4个情绪锚和14个拒绝。情绪为升温，梯队健康0.88但亏钱效应健康仅0.46，昨日涨停股尾部亏损恶化；次日大分歧/退潮Markov概率30.7%，有65次历史转移证据。当前只能作为冻结后的影子观测，不能计为已执行ENTER样本。

同日外部上下文归档覆盖18/18只候选、215条公告索引。异常波动提示只在5日窗口内作为硬风险，减持/监管/解禁/业绩等分别使用预设窗口；减持完成、提前终止、问询回复和过期事项自动降为历史证据。当前留下7条有效异常波动事件，涉及6只候选。提供方只有公告日期、没有分钟，统一按当日23:59:59入库，因此不会反向改变当天15:00前的封存判断。概念成员接口当时不可用，题材覆盖仍记为缺失，不记为0。

情绪V4研究挑战者已锁定为 `2580844ffb6f7a49`。508次因果预测中，状态逆风AUC约0.570、相对Markov的Brier技能约+0.006，但允许开仓阶段存在连续时间块失效；接力损伤AUC约0.564、相对扩展基准Brier技能约+0.011，在允许开仓的273日上三段校准为正但AUC仅约0.542。两者联合历史闸门未通过，状态保持 `RESEARCH_ONLY`，只允许使用2026年8月31日之后的新日期进行锁后验证。

V4前向账本不允许补录开发期日期：预测必须使用上海时区当天最新市场行且在15:05后登记，完整保存特征快照、模型权重哈希、输入文件哈希和两个概率；次交易日数据出现后才追加结算事件。达到至少60个已结算日期、其中30个允许开仓状态，并且两个目标在全日期和允许开仓日期上均稳定优于各自因果基线，才可标记为 `FORWARD_CANDIDATE`；这仍不等于交易部署。

真实题材仍不可用时，行业结构代理只回答“同行业是否有涨停扩散、连板梯队与炸板压力”，不回答“是否同一催化”。8月31日代理结果显示：家居用品为高位配首板扩散但炸板压力扩大；消费电子、证券、通信设备和塑料的高位候选仍是行业高位孤立；房地产开发是首板扩散与高炸板分歧并存；种植业和农化制品的分歧压力较前一交易日扩大。该结果不进入正向买入概率。

真实题材V4不接受“概念名称相同”作为充分证据。每个成员必须绑定决策前可知的交易所/公司公告、政策、监管、官方媒体或券商终端来源，并保存来源发布时间精度、成员依据、题材组ID和证据ID；只有同日至少两只已验证成员涨停，候选又有至少一只去自身支持时，才标记 `verified_same_day_group`。盘中同步率必须从冻结名册中至少2只其他成员计算，名册覆盖率不得低于67%。当前8月31日没有可用的时点化真实题材证据，因此18只候选全部保持 `unavailable`，系统没有用行业代理回填。

开盘确认V4把竞价筛选与换手回封分开：09:25只能排除竞价匹配额衰减、高开无量等风险，09:35后才可能验证VWAP斜率、首次回落量能、均价收复、真实题材同步和回封。题材同步不再接受自填依据，必须与题材同步注册表的日期、代码、证据ID、题材组ID、成员覆盖率和数值完全一致。两次冲板不封、跌破VWAP且卖量扩大、真实题材同步低于35%、高开超过5%但竞价成交占比不足2%均触发 `CANCEL`。即使完整链条通过也只标记纸面 `CONFIRM_RESEAL`，不等于真实成交或自动下单。

盘中前向证据不能等收盘后再挑图复盘。`session_forward_v4.py`要求信号日15:05后、次交易日9:15前冻结候选、V3冻结ID、信号事件哈希、候选文件哈希、六个采集时点和每个时点必填字段；09:20、09:24:50、09:25、09:30、09:35、10:00各只能登记一次。竞价时点超过2分钟、开盘确认时点超过5分钟即永久标记迟到；缺少预注册候选则标记覆盖不完整。盘后评价只有09:25和09:35均按时完整，且确有预注册执行候选时才具备训练资格。8月31日首个计划ID为 `dd902f04b1d29f0fad2c`，绑定V3冻结ID `61ef3e85ff0edfe1`；当日0个 `ENTER/WATCH`，因此次日状态预注册为 `NO_EXECUTION_CANDIDATES`，18只仅保留影子观察范围。

执行契约不使用“买入后跌5%止损”这种在A股T+1下无法兑现的假设。每笔纸面交易默认只承担账户权益0.5%的压力风险，以12%隔夜压力损失反推仓位，因此正常二、三板的风险上限约4.17%；四板最高4%、五板最高3%，高开超过7%最高2%，并同时受30%接力总暴露、15%同题材暴露和最多2个接力持仓限制。成交必须为100股整数手，实际成交价×股数不得超过冻结名义金额。买入日不能卖出、禁止补仓；T+1固定记录收盘盯市收益，跌停卖不出时 `net_return` 保持未解决。当前没有 `CONFIRM_RESEAL`，所以执行契约账本为0笔，这是正确空仓而不是漏记。

历史行情“已核验”不等于历史证券资格已核验。现有日线由2026年仍在当前股票池中的3050只股票回溯构建，不能覆盖后来退市且已从当前名单消失的股票。深市第一批官方状态证据已归档：1,494只当前主板股票中1,483只有带日期更名链，链尾与当前简称零冲突；据此生成1,850个时点区间，其中200个ST/退市风险区间、66个新股无涨跌幅区间。重审8,913条候选后，4,442条（49.84%）获得 `VERIFIED_ELIGIBLE`，其余4,471条沪市观察仍未核验。这个提升只证明“已观察深市代码在该日的状态”，不证明历史候选母池完整；`candidate_universe_complete=false`，因此状态仍为 `SURVIVORSHIP_BIAS_UNRESOLVED`，历史部署闸门继续失败。

退市遗漏V4进一步从沪深官方清单识别出窗口内原母池缺失的84只主板股票，已恢复84/84只、21,562行腾讯未复权日线；腾讯第六字段只按成交量使用，成交额保持缺失。深市42只全部有交易所带日期更名链，生成124个退市证券资格区间；沪市42只仍缺结构化时点ST证据。仅深市已核验下界就补回46条2—5板观察（16个代码、40个交易日），令191个交易日的涨跌停统计发生变化，20天涨停数原先低估至少5%，3天空间高度被低估。2024-04-26由3板修正为至少5板，2024-04-29由4板修正为6板情绪锚，2024-05-27由2板修正为3板。合并候选8,959条中4,488条通过统一资格审计（50.09%）；沪市95条潜在2—5板只作为上界，不进入正式样本。结论为 `MATERIAL_SURVIVORSHIP_SENSITIVITY`，而不是模型升级。

历史日线执行代理把“盘后入选”和“次日成交”分开：次日未触板不交易，全天一字涨停买不到，非一字触板按涨停价作为代理入场，再下一交易日收盘退出；全天一字跌停时净收益保持未解决。该口径没有逐笔和排队证据，结果始终标记为 `INFERRED_DAILY_OHLC`。当前大面统计模型因样本外无技能已被自动拒绝，系统回退透明风险规则。

透明规则也必须接受样本外纪律。V4固定审计了大分歧/退潮、假加速、高脆弱度、亏钱效应弱、昨日涨停尾部重伤、综合高风险、五板、连续一字、7%以上高开和“高开叠加逆风”10条规则，在2023-11-29至2026-08-26的3703笔2—5板日线成交代理上按三个连续时间段检验。没有一条通过稳定闸门。五板全样本大面率46.2%、连续一字47.6%，高于未标记样本，但五板在最近时间段反向，连续一字首段风险比仅1.03；其余多数情绪规则全样本也未呈正向风险差。因此它们只能保留为解释、观察或开盘确认条件，不能冒充已验证的大面概率。情绪层当前最可靠的职责是控制环境许可、仓位和确认强度，不是单独预测某只股票一定吃面。

官方日线开盘价只进入9:30后的增强层，不能进入盘后模型。它显著提高了晋级排序，但当前诊断组合仍亏损，且缺少VWAP、回落量能和板块同步，因此固定为 `DAILY_OPEN_ONLY`、无部署资格；这也是系统不把高AUC直接等同于好交易的约束。

V187新增“修复日次日延续”审计：只用修复日收盘时已经可知的昨日涨停中位溢价、中位晋级率、炸板率和跌停数形成四项质量分，次日状态只作标签。131个完整修复日按时间切分后，宽修复在最近验证段的复退潮率较低，但开发段效应弱且两段置信区间均跨零。因此该分数只进入每日复盘的历史主情景，不生成校准概率，不改变个股排序、仓位或入场许可。当前9月16日为3/4项的宽修复，V27复盘仍因执行和题材证据不足输出 `CASH`。

V188进一步把修复传导拆成首板、中位和高度三层。开发段中位确认方向较好，后段却不再优于未确认组；三层同步在验证段只有3日。9月16日只确认低位和高度，中位未确认，是2/3层的部分传导。V28每日复盘已明确显示这一反路径，但层级分数不能直接升级情绪、给个股加分、提高仓位或授权入场。

V189新增高切低生态窗口：只有高度风险增加、低位一进二较强，并且昨日涨停中位、炸板率、跌停数和中位晋级至少三项健康时，才把市场标记为值得寻找补涨的环境。该结构在最近段方向较好，但开发段完全没有优势；它又缺少真实细分题材和个股补涨身份，所以只能形成情景，不能自动生成高切低候选。9月16日高度仍是正反馈，窗口未激活，V29复盘继续输出 `CASH`。

V190补充首板与连板的赛道注意力竞争。503个两类候选同时存在的交易日显示：连板占比高并未稳定压制首板B+2路径；首板供给高对连板在最近段方向不利，但开发段不复制。系统因此继续把首板定义为新周期/低位注意力赛道，把连板定义为高度/辨识度赛道，禁止共用卡位总分。首板数、连板数和占比只描述拥挤，不能替代真实细分题材、竞争身份、竞价承接、可成交性或个股排序。报告见 `results/v190/lane_attention_competition_audit_20260917_v1/report.json`。

V191补齐当前全池的严重异常算术可观察性。9月16日80只主板收盘涨停全部取得95个提供方日线，80/80末日收盘与冻结池完全一致；上证、深证分别使用上证A股指数与深证A股指数，80/80均形成10日、30日复合偏离研究值及“基准不动、连续10%涨停”的反事实路径。但交易所公告重置时点、除权处理、历史无涨跌幅限制状态尚未官方对账，所以正式监管距离仍为0/80。当前未核验算术上，闽东电力距10日100%阈值约18.49个百分点、澳弘电子约37.65个百分点；这只说明提供方数据下的算术位置，不是官方距离、100%路径潜力、命中概率或买入理由。权威指针见 `results/v191/current_abnormal_distance_20260916_authority.json`，每日复盘V30继续输出 `CASH`。

V192/V193补上第一段官方重置证据链。上交所主板严重异常波动日度表从2026年7月6日新规生效日至9月16日共53个交易日全部查询成功，当前32只沪市候选均未出现于严重异动记录，因此其“该期间无官方重置事件”得到有界核验；澳弘电子的重置闸门由不可获得提升为已核验。48只深市候选在巨潮官方公司公告索引中均未检出“严重异常波动”标题，但公司公告索引并非深交所交换层的穷尽负面证明，所以闽东电力等仍保持重置不可获得。除权、历史无涨跌幅限制资格和官方基准对账仍未完成，正式监管距离仍为0/80，V171路径角色仍为空，V31每日复盘仍为 `CASH`。当前权威指针见 `results/v193/current_abnormal_distance_20260916_authority.json`。

运行当前评估：

```powershell
python relay_model.py score current_snapshot_2026-08-28.csv
```

有足够历史样本后训练（每一行是某只股票某日从 k 板晋级 k+1 板的一次观察）：

```powershell
python relay_model.py train historical_samples.csv fitted_model.json
python relay_model.py score current_snapshot_2026-08-28.csv --model fitted_model.json
```

## 回测纪律

- 按时间切分：训练、独立概率校准、阈值验证、完全盲测，边界各留一个交易日隔离带；滚动验证也重复同一嵌套结构，禁止随机打散或在测试期挑阈值。
- 每个特征必须在决策时已知；次日最高/最低价只能作为标签或执行评估，不能进入打分。
- 终局高度标签允许使用未来结果，但只能作为目标；每个滚动训练边界必须清除尚未终止或终止日期跨界的晋级链，防止长标签穿越。
- 把涨停买不到、次日跌停卖不掉、停牌、100 股一手、佣金、印花税和滑点纳入回测。
- 分开报告信号命中率与真实可成交收益；至少报告样本数、Brier 分数、校准曲线、AUC、最大回撤、平均盈亏、利润因子。
- 不足 500 次“板级晋级观察”时，概率只作诊断；每个高度样本不足 100 次时合并高度桶或采用正则化。

## 盘中执行

盘后模型只能生成候选，不能直接下结论。次日优先 0%–5% 高开且板块同步增强的换手核心；5%–7% 必须快速承接；7% 以上默认兑现风险。一字板不追。跌破分时均价且卖量扩大、两次冲板不封、同题材核心破板或出现风险公告，立即取消。连板交易失败后不补仓。

这是一套研究与纪律工具，不构成收益承诺或个股买卖建议。

## V8情绪与接力联合闸门

V8把环境情绪与2—5板个股接力直接联合。次日晋级和“至少再走两板”模型的因果滚动AUC分别约0.680和0.681；大面模型因时间与分板位不稳定被拒绝。3—5板、每日最多一只的校准前沿可达83.3%成交后晋级精确率和3.05个信号/活跃周，但同期大面率仍为42.9%。没有阈值同时通过70%晋级、10%大面、最低31笔成交和活跃周3—5个信号的联合闸门，验证段及封存盲测保持关闭，系统继续 `DIAGNOSTIC_ONLY`。详见 `results/v8/emotion_relay_understanding_report.md`。

## V9盘中情绪证据层

V9不接受手工填写的“撤单率”、VWAP斜率或回封判断，而是从9:20/9:24:50/9:25原始匹配额和1分钟OHLCVA本地重算；其中兼容字段 `auction_cancel_ratio` 的严格语义只是匹配额净衰减代理，不代表真实撤单总额。所有快照都必须具有来源、时间和可核验状态；未来分钟、重复键、非主板/ST、未验证题材同步均会被拒绝或降级。影子接口每日只预注册一只3—5板排序第一的观察对象，但固定为 `SHADOW_ONLY_NEVER_ENTER`。当前0/120个完整盘中序列、1/100个影子观察、0/60个已结算确认，状态为 `DATA_COLLECTION_REQUIRED`。详见 `results/v9/intraday_evidence_upgrade_report.md`。

## V10冻结影子模型与前向账本

V10冻结了晋级、可成交代理和“至少再走两板”三个稳定组件，冻结ID为 `9738e2c165075798`；大面模型明确排除。影子证据现在必须经过盘前 `PLAN`、盘中 `CONFIRMATION`和T+1 `SETTLEMENT`三段哈希链，空仓日、迟到快照、取消和未成交也全部保留。V9的一个历史演练CSV因没有V10盘前预注册，不再计入合法影子样本；当前V10前向数为0，账本校验 `PASS`，继续 `FORWARD_SHADOW_ONLY`。详见 `results/v10/forward_shadow_lock_report.md`。

## V205—V214 同高度、双目标与五因子升级

正式候选接口现在要求同一交易日、同一板位的股票形成精确完整的竞争组；封板快和零开板只能作为形态观察，不能单独命名冠军。首板模型拆成两个独立目标：可成交代理下B+2净收益为正，以及B+1净亏至少7%的大面风险；二者分别按结果实际可获得时间滚动训练。当前12个可评分交易日中，B+2基线AUC约0.517，B+1大面基线AUC约0.554，均未完成部署校准。

五项新增证据的历史覆盖为：换手承接1116条、题材纯度6条、板块带动88条、公告软风险178条；同高度竞争者淘汰另有127条归档范围内描述记录，但因历史全市场完整性未获证明，正式可用仍为0。12325条两两比较只来自12个交易日，不能冒充12325个独立样本，验证统计必须按交易日聚类。真实题材驱动目前只有13条可核验记录，100%严重异常波动仍只有2条官方正例、0条可训练正例和0条可靠负例。因此最新验收报告 `results/current_acceptance_report_20260920_v214.json` 与证据注册表 `results/v154/model_evidence_registry_20260920_v54.json` 均继续保持 `DIAGNOSTIC_ONLY`。

## V215—V216 历史炸板高度与竞争淘汰覆盖

V215从约202万条日线按真实相邻市场交易日重建封板与炸板尝试：只有某股上一市场交易日确实收于N连板，今日触板未封，才记为N+1板失败尝试；缺失交易日会断链，禁止跨日制造连板。当前得到58,772条观察，其中38,205条收盘涨停、20,567条炸板、4,635条二板及以上失败尝试。重建涨停与既有37,325条库存匹配37,295条，召回99.92%。

V216据此把同日同高度的“失败尝试数÷（失败尝试数+封板存活数）”接入竞争者淘汰证据，使1,116条首板观察全部获得描述性数值，较此前127条归档范围记录明显扩大。但底层仍是当前股票名单回溯，全部点时ST状态未独立核验，退市证券也不完整，所以正式点时可用仍为0；模型不会把这1,116条误标为已验证因子。双目标与同高度验证按相同10日最小训练口径重跑后，基线指标不变，五因子增强模型仍因完整证据为0而不出概率、不选冠军。
## V218—V219：首板与题材筛选旁路升级

在不改动龙头与连板框架的前提下，新增 `first_board_community_challenger_v218.py` 和 `theme_screening_matrix_v219.py`。前者把社区一进二经验转成盘后可审计字段，并继续分开评估晋级、可成交B+2收益和B+1大面；后者把真实题材、二手题材名单、行业代理和名称字辈分通道建模，禁止把行业或“华字辈”冒充真实题材。

当前27日、1,281条首板观察的滚动审计中，社区+日线挑战模型虽改善AUC和B+1大面率，但日度Top1晋级仍为2/15，Top1 B+2均值由-2.57%恶化至-3.35%，因此不替换旧模型。完整边界、字段和产物见 `首板与题材筛选升级V219.md`。

V220把V218首板质量闸门与V219题材通道接入不可回填的前向影子账本。2026-09-21首次冻结精华制药作为研究观察；它来自二手复盘的医药完整梯队，未获得主动催化强验证，研究分数不视为概率，次日确认失败不补选，且始终 `entry_authorized=false`。
# 首板补充 V221—V224（2026-09-22）

在 V218—V220 基础上新增四个互相隔离的首板模块，龙头与连板框架未改：

- `first_board_open_confirmation_v221.py`：冻结候选的 09:25 / 09:35 次日确认，09:35 后只允许使用 09:36 因果价格机会。
- `first_board_theme_catalyst_v222.py`：成员关系与催化的点时核验，观察名册不能冒充真实题材催化。
- `first_board_selective_policy_v223.py`：晋级、B+2 收益、B+1 大亏三目标独立建模；不通过即空仓且不替补。
- `first_board_forward_settlement_v224.py`：晋级、确认、B+1—B+3 分阶段结算与哈希链账本。
- `first_board_daily_supplement_v224.py`：把 V221—V224 串成每日不可覆盖的数据包。

详细规则、当前缺数状态和输入模板见 `首板方案补充V224.md`。当前结论仍为诊断/研究用途，`entry_authorized=false`。

# 同高度龙头PK挑战层 V225（2026-09-22）

`same_height_leader_pk_v225.py`修正标题级风险公告固定重罚、炸板次数单调处罚、符号题材完全不参与辨识度比较及同高度股票缺少逐对解释的问题。它把公告全文、快速修复路径、符号市场身份和题材支撑拆开，并输出同板位两两对决。当前仍是案例派生挑战规则，正式冠军为空，至少需要60个前向交易日簇才能申请替换原规则。详见`同高度龙头PK升级V225.md`。

`announcement_fulltext_archive_v227.py`对所有有效硬风险标题下载并哈希归档公告正文；`announcement_fulltext_review_v228.py`把实质性法律风险、热点业务澄清、常规市场风险和业绩风险分层，同时严格比较正文实际取得时间与决策截止时点。镜像正文不能单独形成认证硬否决，截止点后取得的正文只作事后诊断，禁止倒灌。

`same_height_pk_forward_v226.py`冻结V225代码、报告、全组候选和描述性领先者，并用哈希链记录快照与后续结算。当前有效注册表为`data/v226/same_height_pk_forward_v5`，冻结号`2134651d6cb56031`，校验状态`PASS`。

`same_height_pk_daily_v226.py`是每日入口：基础复盘与V227/V228正文复核完成后，一次生成同高度PK报告并写入V226前向账本。

## V268：盘前公告与高标优先升级（2026-09-28）

东方中科复盘确认，收盘候选形成后至次日开盘前缺少一次公告全文与业务澄清复核，导致代理产品、非自研、预计净利润大幅减少及仲裁不确定性未进入最终选择器。V268将信号日15:05—交易日09:15的点时公告绑定设为冻结主选的必经契约：官方全文核验或官方索引核验无公告才可放行，重大业绩影响、法律/履约/控制权不确定、代理/非自研/不生产等题材否定、全文未核验和错绑均直接 `CASH`。高标优先继续保留，低位只作满足独立证据和开盘确认后的研究候选，主选失败不自动切换备用。详见`盘前公告与高标优先升级V268.md`。

## V269：老龙与老周期情绪反馈层（2026-09-29）

复盘新增老龙池和老周期龙头池。老龙即使属于不同板块，也必须记录断板/反抽方式、次日反馈、同属性扩散、中位股补跌和对新题材的情绪传导。股吧常见的“老龙断板出补涨、老龙跌停出新周期”只登记为待验证假设，必须用断板后1—3个交易日的真实结果确认；老龙反抽只记为情绪修复，不能自动升格为二波、新主线或买点。详见`复盘触发协议V251.md`和`每日龙头复盘模板.md`。

## V270：统一龙头打板复盘提示词（2026-09-30）

将市场行情、情绪、老周期、老龙头、补涨窗口、新周期、真实题材、基本面清洁度、龙头竞争和具体打板层级串成一条复盘链。最终只允许输出“打首板、打二板、打三板及以上、只观察、空仓”中的一个主结论，并给出竞价、09:35、10:00、取消和退出条件。详见`每日龙头复盘提示词V270.md`。

## V229—V230：首板多通道身份与华字辈前向对照（2026-09-22）

审计确认V217并非没有发现华字辈，而是V218把每只首板压缩成一个`best_theme`后，二手题材名单遮蔽了同时存在的名称符号身份。`first_board_multilane_identity_v229.py`现已并列保留真实/二手题材、行业代理和名称符号，单列原叙事候选、符号通道可买先锋及符号＋叙事双身份候选，并把低位首板映射到同字辈全部二板以上梯队。名称聚集永远不冒充产业题材，也不与其他证据通道强行相加。

2026-09-21描述性重算为：原叙事候选精华制药、华字辈可买先锋延华智能、华字辈双身份候选新华百货；正式首板主候选仍为空。`secondary_driver_claim_archive_v231.py`恢复了哈希一致的93只股票逐股解释，`official_driver_first_time_v232.py`再以官方原文核验候选催化时效：精华制药的指引依据至少已公开1378天，延华智能的AI医疗场景至少已公开395天，新华百货回购方案已公开31天但9月16日仍有官方进展。旧闻、持续事项和当日新催化因此不再混为一谈，且年龄不直接变成排序分。

`official_announcement_recency_v233.py`进一步扫描全部主板解释：85/85只上交所或巨潮官方公司公告索引查询成功，归档120日窗口2352条元数据，13只取得类别相容的官方标题并进入全文核验队列。标题匹配不证明事实，窗口内没有匹配也不证明没有政策、新闻或互动催化；只有日期的信号日公告按15:00不可用处理。

`official_driver_adjudication_v234.py`已完成首批11只高影响解释的官方全文裁决：7只为截止前全文支持，2只为负面/有限暴露澄清，诚邦股份的“定增1.29亿元”与官方“不超过1亿元”冲突，华远控股的9月21日日期级公告不得倒灌为收盘前催化。官方全文匹配不等于买入信号，未覆盖个股继续标为未知。

`first_board_multilane_forward_v230.py`已把三种角色写入不可篡改前向注册表`data/v230/first_board_multilane_forward_v4`，冻结号`aa6f9697877aebe4`，校验`PASS`。至少完成60个前向交易日簇并同时改善晋级、可成交B+2收益和B+1大亏前，不替换正式规则。详见`首板多通道身份升级V230.md`和`首板催化时效核验V232.md`。

## V235—V236：连板高度锚与可接力核心分离（2026-09-22）

`relay_leader_role_gate_v235.py`补上V225之后的跨高度角色层：最高板继续作为空间/情绪锚，收盘可接力挑战者只在最高板和次高板中选择，一字不可交易、长时间争夺、爆量或实质硬风险不进入收盘主挑战。当前描述性结果为：华瓷股份是5板情绪锚但一字可买性未证明，内蒙新华是4板分歧修复挑战者；正式市场核心和连板主候选仍为空。

`relay_leader_role_forward_v236.py`已把两个角色与全部连板池写入`data/v236/relay_leader_role_forward_v1`，冻结号`5fa8569337c8200c`，完整性`PASS`。`relay_leader_daily_v236.py`为V225—V236的每日单命令入口。至少60个前向交易日簇完成之前，该层不替换原正式规则。详见`连板龙头优化V236.md`。

`relay_leader_open_confirmation_v237.py`只对V236冻结挑战者做次日09:25—09:35确认，并把高标断板后的良性核心切换与退潮传染分开。真实产业题材与名称符号梯队使用不同结论，最早因果机会价格为09:36。当前因缺9月22日快照而记录`DATA_UNAVAILABLE`。V238已冻结该确认规则，注册表`data/v238/relay_leader_open_forward_v1`，冻结号`6d5aeae0066d2805`，完整性`PASS`。

## V239—V241：3/4板跨题材卡位与不及预期赢家（2026-09-22）

`relay_cross_theme_expectation_v239.py`不再只看同高度冠军，而是保留全部3板、4板身份分母，盘后冻结“主预期→同高度替补→低一板替补→无赢家/退潮”的可证伪顺序。2026-09-21的描述性顺序为内蒙新华→世联行→华软科技→金健米业；南华生物、华纺股份和龙头股份仍在完整分母，但不具备收盘可交易替补资格。现有题材归属是二手/推断证据，不是真实业务已核验。

`relay_cross_theme_confirmation_v240.py`用全部候选的09:25竞价与09:35前Level-1/分钟代理判断谁达到预期；主预期失败后，下一顺位只能成为复盘中的条件赢家，不能自动替换成盘中交易。任何序列外股票晋级都记为模型漏判；所有预定选择失败时固定输出退潮/空仓。当前因7只完整次日数据未导入而保持`DATA_UNAVAILABLE_INCOMPLETE_3_4_BOARD_DENOMINATOR`。

`relay_cross_theme_forward_v241.py`冻结全分母、预期树和确认结果，当前有效注册表`data/v241/relay_cross_theme_forward_v3`，冻结号`f333f4da7eb1410d`，完整性`PASS`。v1—v2因后续补强“身份赢家也必须来源核验”和“同高度同时晋级不得伪造唯一赢家”而由哈希闸门自动失效并保留审计痕迹。至少完成60个独立日期簇前，不宣称提高更高连板胜率，也不替换现有正式规则。

## V242—V244：股吧经验的路径化预期升级（2026-09-22）

`relay_path_expectation_v242.py`把社区常说的强更强、弱转强、一字后首次价格发现和高换手分歧延续转成不同的盘前预期，不再给所有3/4板统一套用0%—5%区间。`relay_stage_competition_v243.py`将强度拆成09:25竞价、09:35开盘承接、10:00攻击/封板/题材带动和收盘晋级四段；只要4板仍确认，3板不能因更早封板就被写成跨高度胜出。高标与中位批量负反馈下的晋级者只记退潮幸存者，多个最高结果板位同时确认时保留并列赢家。

社区来源仅生成H081—H085待验证假设，数值区间不直接形成概率和买点。V244前向注册表为`data/v244/relay_path_forward_v1`，冻结号`924fd8b4c77f4219`，完整性`PASS`。当前缺9月22日完整分钟分母，V243保持`DATA_UNAVAILABLE`。

V245随后审计截至8月31日的2740条3/4板历史候选：2732条有次日晋级和开盘标签，但高位首次/最后封板与炸板字段全部为缺失或零占位，完整路径可用行为0，不能验证强更强、快速修复或10:00卡位链。非一字代理后30%段中3%—7%相对0%—5%的晋级差在3板、4板分别为+10.51和+7.92个百分点，但交易日聚类区间都跨零，且可观察净收益仍为负；一字代理晋级率约71.55%，同时次日仍一字不可买约37.10%。因此股吧预期只保留为前向挑战规则，不升级正式模型。详见`results/v245/relay_path_history_audit_20260922_v2`。


## 威科夫量价代理固定检验（2026-10-07）

用户要求先检查效果再决定添加。本批仅检验预登记的首板前努力—结果失配代理，保持原29项/3059训练与9月182候选。原非一字排名7/13；加入代理和仅缺失标志对照均8/13，代理没有额外命中，全候选正确数82/133降到81/133。训练风险事件5、测试0，可用字段155/182，事件效果信息不足；不能据此判整套威科夫无效。决定暂不接入正式评分、开盘排序或交易触发。新增真实前向样本0，未调阈值重试，原题材支持研究继续。说明和完整指标见 `research/wyckoff_validation/effort_result_20261007T073941767779Z/威科夫固定检验结果.md`、同目录 `report.json` 与评分前协议。


## 三类增量指标准备（2026-10-07）

已按用户要求补入本地完整目标提示词、每日模板、现有采集说明及research_queue.json：第一优先真实题材支持与竞争压力沿原固定C1/C2；封板路径及目标日09:35原组相对强度独立待办。规范明确公式/单位、去自身/全原组分母、时间与哈希绑定、缺失策略。当前仅字段和流程规范，后二项完整自动计算器未实现，未采新行情/回测，正式模型与前5排序保持；来源不足不写0、不填概率。字段详见 `research/theme_same_height/input_preparation.md`“三类增量观察字段”。


## 三类观察指标程序链路已接入（2026-10-07本批）

按用户“一起弄，不要一步步问”授权，已完成research_indicator_pipeline.py统一研究入口、seal_quality.py与open_strength.py，不再仅是字段规范。run留存完整原池和原登记子集，collect-signal复用实际收盘采集，collect-open保留原组逐码一次原始响应。现行模型、原预测和前5评分保持。封板同组分位/原报告开板次数可重建；精确时长缺完整事件序列未知。09:35计算器已实现，但当前Sina时间/复权/单位语义缺认证，原响应不自动变成有效强度。

59项针对检查与14子检查通过，一次修正后的现有档案接入保留48只、原登记11只；46行分位和48行开板次数仅旧档诊断，盘后原截止可用0，09:35全部待发生。当前真实题材已核验细分仍0，固定C1/C2未解锁。新结算前向0，准确率提升未评价，不能将程序测试当改善。入口和后续已绑定命令见research/indicator_observations/README.md；可阅读结果见research/indicator_observations/implementation_20261007T080458Z/complete_archive_run/交付说明.md。本批不启用或修改平台目标及定时任务。

## 情绪题材案例与环境记忆（2026-10-07）

已完成[情绪题材周期案例与条件反射](research/情绪题材周期案例与条件反射.md)，选择AI、华为链、低空、9·24政策行情、消费食品、DeepSeek六类代表对照，整理原产业/公司证据、同日强弱反馈、后续反证和六行环境记忆卡。每日模板的人工阶段表达对齐完整提示词现有八阶段，阶段/相对变化/主导关系分列；不重映射原代码或训练标签。模板、目标提示词和经验库均已引用，原模型与排序不变。

本批仅历史查证与观察训练，原完整题材竞争组未知；原始来源及失败见research/emotion_theme_cases_20261007T083315Z/source_requests.jsonl。新增真实前向0，未产生Accuracy/Precision/Recall/F1/Brier新比较，不宣称提高准确率。唯一下一步仍是原完整竞争组与题材时点证据采集；10月8日收盘原11只结算优先。未恢复平台目标或修改定时任务。

## 历史行情联合预测固定检验（2026-10-07）

按用户要求完成一次[行情预测数值比较](research/market_environment_validation/conditional_levels_20261007T090923Z/历史行情预测固定检验.md)：复用V300官方日期网格8渠道，以次日原涨停池收益中位数为主目标，LEVEL中位基准对比固定滚动Ridge当日信息及增加相邻日变化两个方案。这是新的数值研究诊断，不是首次市场回归或击败现生产模型的证据。

3—9月147个原信号日，主目标124日可评分。方向80/124=64.52%→79/124=63.71%（两挑战），MAE1.6142→1.6195/1.7091个百分点；基准64.52%等于全正预测命中率，平衡准确率50%，不能代表整个系统或稳定60%。两个主目标门均未通过；辅助涨停家数/最高板/晋级率局部改善不升级。4项时点/缺失合成测试与独立逐行评分核验通过；后续入口哈希拒绝检查另1项通过，无数字重跑。完整181日和未知保留，新真实前向0，原模型与原11只预测未改，不按结果调参。唯一下一步仍按原题材与完整原竞争组时点证据队列，10月8日收盘结算优先。

## 当前研究方向调整（2026-10-07，覆盖上文旧研究下一步）

用户明确“没必要继续弄真实题材时点”“原有的基础已经够了”。已停止THEME_SAME_HEIGHT_PIT_01的后续专项采集与准备，队列优先级改为已有行情和原结果的市场预测误差诊断。已有基础直接用于固定历史比较，不再自动安排补题材、时点档案或新数据源作为前置任务。旧题材协议及已完成回测原件保留；AGENTS.md、完整目标提示词和采集说明同步调整。生产价格、日历与原11只预测结算仍按原合同执行。本次只改研究安排，未重训、未新增样本、准确率没有变化。

## 四种方向方法按序补充（2026-10-07）

已实现并依次完成[直接方向、相似日、误报代价与时间窗固定检验](research/market_direction_validation/sequence_20261007T095212Z/按序改进历史检验.md)。三批、四挑战的参数和后续基准均在看新结果前固定；使用现有8字段，没有新增采集或恢复题材时点分支。3—9月147个原日中124日可评分：原Ridge63.71%、LR12062.10%、KNN64.52%（全部判正）、非正代价加倍56.45%、LR25266.13%。代价方案减少误报但损失正确率；252窗点估计改善4.03个百分点但区间跨零，四挑战均未通过全部门槛。

入口为market_direction_research.py，原评分源码及全部181日原预测/未知保留。44项因果与评分合成检查、3项共享协议检查（含12项字段变更子例）通过，另有逐行独立核验。审查补强共享冻结校验并明确配对子集统计口径，原结果未重跑。正式模型、开盘排序、9月30日原11只预测均未改，真实前向新增0。这是单一短线反馈方向的历史诊断，不代表整个系统稳定60%。唯一研究下一步为用已保存120/252窗预测核对共同误判非正日的结构，不调参重训；10月8日原结算仍优先。

## 市场与个股固定研究及升级链路（2026-10-07，覆盖上一条研究下一步）

按用户批准方案完成[原模型复现、市场与个股固定比较](research/model_upgrade_validation/fixed_20261007T112017Z/固定模型研究与升级实施.md)，入口为[model_upgrade_research.py](model_upgrade_research.py)，每日来源绑定及验收由[model_upgrade_shadow.py](model_upgrade_shadow.py)执行。原市场16判断、个股182判断类别完全复现，误差远低于1e-8；全部来源、依赖与六个估计器已保存。市场先比较、冻结，再比较和冻结个股；冻结不重新拟合。全部指标独立重算。

市场C2在2026年1—8月63.13%→65.63%，2025下半年退步，9月6/8不变且无延续标签。个股真实OHLC非一字Top1三臂均7/13，旧低换手代理8/13不能算真实改善；9月全候选82/133均不变，挑战原始Brier略退。当前没有升级证据，生产仍BASE，新增真实前向0。完整全部候选、未知及三段指标留档，不以以前66.13%二分类升级。

已实现120官方日固定影子门及通过后分模块直接生效，60日固定非重叠回退；[model_upgrade_apply.py](model_upgrade_apply.py)只写生效后新的未审阅计划，仍需实际作者审阅与V298冻结。命令及未来数据边界见[研究说明](research/model_upgrade_validation/README.md)。原实际使用源码、模型、结果均保留；空接力组处理和冻结完成门修复另留真实时间与哈希，无重拟合。全年官方日历复用现有V298验收扩展，2027只按官方公告后续扩展。定时任务及平台目标状态未改。

下一步为10月8日收盘原11只按原口径结算，取得新底稿后登记两个模块首个合格影子信号；题材时点专项继续停止，不因未来等待再调参。

升级后的作者审阅计划用[model_upgrade_freeze.py](model_upgrade_freeze.py)在原V298 freeze前再核所有来源与受改字段，删除应用绑定不能绕过。首个真实前向前另存production guard。120/60成熟节点按真实官方日期时钟认证，末日缺记录照保留覆盖；60日回退不添加原计划未规定的样本最低门，指标未知保持未知。

## 保留现模型，增加判断核对（2026-10-07最新授权）

用户明确“那就不替换，在原有基础上改进”，覆盖上一条验收后自动切换的授权。已追加持久hold及真实时间、源代码和前像绑定，120日验收仍可运行但不会替换，直接切换及挑战应用被拒绝；影子研究继续，生产仍原BASE，平台目标与定时任务未改。

已完成[现模型诊断附页](research/model_upgrade_validation/fixed_20261007T112017Z/base_diagnostics_20261007T123953Z/保留现模型的诊断改进.md)，独立入口为 `current_model_diagnostics.py`。显示原46/29项信号事实、市场三类分数、字段缺口及完整组首选资格；历史漏判与信号日事实分开。307个官方信号日、3707条原候选保留，分数/原判断/排序和分段指标独立核对一致。准确率未变，新增前向0，无新拟合。辅助程序7项及新增5项针对检查、政策11项和3项受授权变化影响的旧检查通过，不重跑全套。

AGENTS、完整目标提示词、每日模板和原研究队列已同步。唯一下一步仍为10月8日真实收盘后的原11只结算，然后用有效新底稿接入BASE附页；不恢复题材时点专项，不应用挑战模型。真实当期附页接入尚待新来源，已完成合成绑定检查。


2026-10-09已完成[周期分层、G1影子与风险提示](research/model_upgrade_validation/regime_theme_g1_20261008T183729Z/交付说明.md)：原分数、判断及正式前5保持，真实前向仍1日11只、Accuracy63.64%，新增成熟样本0；G1独立固定120日验证尚待首个合格信号。新日期调用与V298绑定见[使用说明](research/model_upgrade_validation/周期分层与G1使用说明.md)。

2026-10-09已正式核验[时代万恒、紫竹高科两只漏判与R2检验](research/model_upgrade_validation/relay_error_20261009T045526Z/两只漏判与模型改进核验.md)：原完整10只Accuracy70%、完整Recall50%；累计真实前向2日21只Accuracy66.67%，不是稳定率。逐股原评分已精确分解；R2固定非线性方案因Recall退化等未通过，未替换模型。新增只读原信号/当前目标来源联合索引并经原V293/V298结算，4份错日期重取保留且拒绝。旧三臂影子仍因周期/G1改变V298保护哈希而被拒，下一步修复明确兼容后补结算，不能当已完成。

2026-10-11已继续完成[兼容修复与转换手特征检验](research/model_upgrade_validation/relay_phase_20261009T095400Z/兼容修复与转换手特征检验.md)：原三臂两模块补结算，原10只结果、评分及排序保持；正式显式模块推理也已接入同一来源兼容证据。旧保护和失败保留，仅恢复BASE研究链。R3前日一字代理/转换手增量1—8月Accuracy71.97%→71.66%、9月61.65%→60.90%，淘汰；9月非一字首选7/13→8/13不足以抵消整体退化。29项相关核验通过，无新拟合或独立前向样本，生产模型未变。调用及下一新信号日步骤见研究README。


## 2026-10-12凌晨：评分与情绪环境重评

已完成[原评分诊断与E1/E2固定比较](research/model_upgrade_validation/relay_context_20261011T151204Z/情绪环境重评与固定比较.md)。原BASE精确复现，两挑战各段只拟合一次、均未通过，不升级、不重跑。新登记正文改为模型倾向表述；反馈中位数措辞错误追加更正，原预测和报告不覆盖。完整候选解释、分状态/板位/一字指标、固定协议及研究入口见研究README。真实前向仍2日21只14正确，本轮新增0，非一字漏判为后续重点；18项针对测试通过不代表准确率改善。


2026-10-12按升级要求继续完成[H1/H2非一字评分结构比较](research/model_upgrade_validation/relay_expert_20261011T162713Z/非一字评分结构检验与升级结论.md)：新结构已实现并训练，均未过升级门，未上线；原21只指标不变，本批新增0。完整同池回放仍漏判两只、强化园林误报，13项针对检查通过不表示改善。后续先核现有非一字输入/训练语义，再依固定门研究。
