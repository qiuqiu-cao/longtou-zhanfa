# 固定市场与个股模型研究入口

## 当前统一复盘入口（2026-10-14）

按用户“不足都解决／继续”完成[本轮整改与验收](remediation_20261014/不足整改与验收.md)。下面的统一入口适用于下一次合格新日期，旧历史入口和旧待修说明保留为当时状态。原CLI丢失历史/叙事参数的问题在新入口已修；原受保护源码及冻结哈希未改。

```powershell
python relay_review_workbench.py prepare --features <原归档> --theme-report <完整题材> --mainline-report <独立主线> --compatibility <原兼容政策> --bank <已核原文bank> --calendar <官方日历> --pool <完整遮分候选> --relation-bank <父锚关系bank> --output-dir <新目录> --history-dataset <历史1> --mainline-history <历史主线1> --narrative-reviews <叙事审阅> --model-exposure MODEL_NOT_REVIEWED_SELF_ATTESTED
python relay_review_workbench.py compact --facts <完整事实> --draft <作者稿> --output <共享新稿>
python relay_review_workbench.py expand --compact <作者完成共享稿> --output <完整新稿>
python relay_review_workbench.py freeze --facts <完整事实> --judgements <完整新稿> --output <新事前登记>
python relay_review_workbench.py compare --record <原登记> --baseline <原同池BASE> --outcome-archive <原目标来源> --output <新比较>
python relay_review_workbench.py execution --facts <完整事实> --review <作者完成稿> --original-predictions <同日完整条件判断rows> --calendar <官方日历> --asof <真实带时区截止时间> --output-dir <新执行报告> --evidence <可选已绑定真实或纸面来源bank>
python relay_review_workbench.py validation-status --output-dir <新状态报告>
python relay_review_workbench.py qualify-information --output-dir <新本地资料资格报告>
```

history-dataset/mainline-history可重复；没有材料时不传，未知保持。未看模型必须据实声明，上述MODEL_NOT_REVIEWED_SELF_ATTESTED不自动认证盲测；默认已看模型。旧已见日期仅attach/review-context附页，不能用compact自动改成新预测。共享理由展开继续原完整校验，freeze仍运行原角色、价量竞争、催化、预期计划、时点和登记门。

compare自动生成观点结果反馈，保存原比较后反馈失败仍保留比较及具体失败收据。execution复用原执行/账户模块，开盘观察不算成交、纸面不算实际；缺成交/费用/退出不报收益或空仓。新增两个只读状态命令不拟合、抓取、登记新影子或切换模型，保持原120日验收门。

当前源码的完整15只回放、9245历史资格、75源与正式/独立账本重验在[final_delivery](remediation_20261014/final_delivery/verification.json)，111相关测试通过。正式原3日39/38/28正确与独立两批29/28/11正确分别统计；独立已判断Accuracy50%、同池BASE78.57%。新预测/成熟/拟合/下单均0，非一字漏判、盘口历史覆盖和盈利能力未通过效果验收，不能报全部预测问题已解决或概率已校准。

额外根交叉审查加强了采集字段与同一股票行/返回日期的绑定、真实收件与执行时钟、纸面来源不能挪用为实际成交以及卖出不能借其他计划入场。当前采集scaffold全15字段仍UNKNOWN；通过结构不是训练授权。初版、v2、delivery_final等原输出及源码前像保留，最终当前实跑以`remediation_20261014/final_delivery/`为准。说明见[统一入口](remediation_20261014/workbench/工程入口与共享理由交付.md)、[数据资格](remediation_20261014/data_prediction/数据覆盖修复与预测资格结论.md)、[执行收益](remediation_20261014/execution/执行收益闭环交付.md)。

2026-10-12完成[实际重训与九阳、时代经验结算](jiuyang_shidai_20261012T072934Z/重训与九阳时代经验结算.md)。入口`relay_proxy_ablation_research.py`的freeze已在拟合前完成，compare已实际执行三个固定段各一次；P1_DROP_PROXY淘汰，不能再次运行compare或改参数。它只去掉旧低换手代理`signal_one_word`，不添加今天取得的催化/论坛字段，原训练成员与剩余28字段/算法/阈值不变。完整结果、分层、输入绑定与独立重算在该目录retrain下；失败挑战没有保存部署权重，未开启新120日窗口。

原完整11只10月9→12独立观察已结算：独立Accuracy50%（5/10已判断）、Recall25%、预测覆盖90.91%；BASE72.73%（8/11）、Recall50%、覆盖100%。11只新成熟独立观察不混入原正式2日21只。后续用下列催化扩展入口生成新日期卡片时，竞争分析同时保留共同晋级、共同失败、相互卡位路径；逐日核对真实一字/代理，区分核心延续与后排失败，区分行业/实际叙事以及晋级/承接/可买性。3篇原帖及正反经验存`jiuyang_shidai_20261012T072934Z/forum_sources`，今日取得不倒填旧预测或训练。新日期这些检查仍须完整同池前向结算，不能从案例直接追加数值分数。

2026-10-12继续接通独立11类角色链路，完整调用及真实档案核验见[角色链路接通与验收状态](leader_roles_completion_20261012T092145Z/角色链路接通与验收状态.md)。`leader_role_engine.py augment/attach-outcomes/combine`追加富证据、单独绑定到期晋级标签及合并不同日期；`leader_role_research.py settle --horizon 1 --outcome-archive`沿原来源适配器结算C2；0/3/5分别结算当时身份与未来角色。`review`同时重建结算、检查角色门及原/完整候选C2门，二者都通过才标新日期应用资格，不自动改生产。当前73只角色候选、11只原晋级结果接入核验成功；角色标签待独立盲审，训练0、真实前向新增0、总验收未通过。原第一批协议与回测保留，源代码变化后不得绕过它的哈希保护重跑；本次另存下一次使用前的源码/协议收据，没有重复数值比较。

2026-10-12完成[催化识别与独立判断入口扩展](catalyst_context_20261012T022210Z/催化识别与判断模块补强.md)。新日期使用以下入口，把已核逐股叙事、公司业务、事件状态、业绩影响和未知覆盖接入原遮蔽分数事实卡；原独立控制器及全部数值模型不改。

```powershell
D:\python\python.exe stock_catalyst_context.py build --features <原来源绑定特征目录> --bank <当期已核原文片段.json> --compatibility <原兼容策略.json> --output <唯一催化上下文.json>
D:\python\python.exe stock_catalyst_context.py prepare --features <当期特征目录> --theme-report <完整theme/report.json> --mainline-report <独立mainline/report.json> --compatibility <原兼容策略.json> --bank <当期已核原文片段.json> --output-dir <唯一新事实卡目录>
D:\python\python.exe stock_catalyst_context.py freeze --facts <上述目录/facts_with_catalysts.json> --judgements <完整独立判断.json> --output <唯一事前记录.json>
D:\python\python.exe stock_catalyst_context.py compare --record <原扩展记录.json> --baseline <同池原BASE视图.json> --outcome-archive <原V293精确目标结果档案> --output <唯一新增比较.json>
```

build可用于明确标注实际时钟的功能诊断，不能代替预测登记；prepare/freeze仍检查信号收盘后至目标09:15竞价前，盘中不补竞价前方向。旧11只用原`stock_judgement_review.py compare`，不能用新事实改写原预测。新日期仍须当期来源重建、真实作者审阅及原V298冻结；本入口不改变三臂/G1/T1窗口或生产前5。

已核片段格式实例为本目录`catalyst_context_20261012T022210Z/checked_catalyst_bank.json`。每份源绑定URL、实际取得、页面公开日期/精度、原响应/文本与原收据哈希；Windows文本摘要差异须显式绑定原更正，失败HTTP不能冒充成功。公司业务/业绩只读原公告，事件真伪只读官方原通报；论坛仅证明限定作者叙事出现。正文语义仍由核对者审阅，程序核原文offset、股票锚点、来源资格与时间，不声称全市场自动发现。日期、股票和语义必须按新日期重新核对，不照抄本例方向。

独立判断每行追加`catalyst_assessment`：`narrative_presence`、`earnings_impact`须与事实卡一致，`event_and_business_reviewed=true`，`negative_reason_codes`数组。偏断板至少一个实际独立反证类别PRICE_PATH_WEAKNESS/PEER_COMPETITION_LOSS/CYCLE_PRESSURE/VERIFIED_EVENT_COUNTEREVIDENCE；来源未覆盖、无催化、原模型负向不得当负向理由。未知不自动转正。`turnover_comparison_claim=PREVIOUS_SESSION`在原参考日期未知时拒绝；新卡明确“上次连板池换手比”，真实一字与低换手代理分开。

本批只是可调用识别/判断层升级，新字段因历史完整来源及时点训练门未通过，数值拟合0。已看过的11只回放不计独立预测改善，源证据晚于原截止不可回填。原数字模型择优与120日效果门保持。

2026-10-12新增独立个股判断入口，落实已核股吧经验中的角色、同身位竞争及分歧承接比较。完整实例见[股吧经验与独立个股判断](../../results/weekend_review_20261011T142844Z/independent_judgement_20261011/股吧经验与独立个股判断.md)（从本目录跳转使用仓库根目录下results）。本工具不训练模型、不改原分数或前5，也不代替原生产冻结。

```powershell
D:\python\python.exe stock_judgement_review.py prepare --features <当前原来源绑定特征目录> --theme-report <当前完整theme/report.json> --mainline-report <当前独立mainline/report.json> --compatibility <原已批准compatibility_BEFORE_USE.json> --output <唯一新遮蔽分数事实卡.json>
D:\python\python.exe stock_judgement_review.py freeze --facts <上述事实卡.json> --judgements <完整逐股独立判断.json> --output <唯一事前独立记录.json>
D:\python\python.exe stock_judgement_review.py compare --record <原独立记录.json> --baseline <同日期同特征已保存BASE视图.json> --outcome-archive <原V293准确到期档案> --output <唯一新增同池比较.json>
```

prepare只输出全部原2板以上候选的白名单事实与竞争对象，不输出模型评分、判断或原排序。判断JSON顶层须填写`model_exposure`（`MODEL_OUTPUT_ALREADY_SEEN`或`MODEL_NOT_REVIEWED_SELF_ATTESTED`）及完整`rows`；每行字段为`code`、`promotion_judgement`（PROMOTE/FAIL/UNKNOWN）、`role_hypothesis`、`relative_thesis`、`support`、`rebuttal`、`cancel_condition`、`missing`数组、`comparison_peers`代码数组、`fact_references`已绑定绝对路径数组、`opening_confirmation`的09:25/09:35/10:00字符串。完整填写实例为上述附页中的`judgements_before_auction.json`，但以后必须依据当日事实重新独立判断，不能照抄股票方向。

freeze检查真实当前时间在信号收盘后、目标日09:15竞价前；全候选、同身位对手与来源均不得遗漏或改写。独立登记账本拒绝同信号/目标重复登记，比较重新核对原记录哈希。当前compare支持来源绑定的BASE合并视图：顶层`signal_date`、`target_date`、`generated_at`、`knowledge_cutoff`、`feature_sha256`、`source_hashes`，以及`relay.candidates`原完整代码和`predicted_positive`布尔；当前实例如`results/weekend_review_20261011T142844Z/model/base_predictions.json`。其他生产或三臂原生结构须先明确适配，不能手填预测分数或真值来满足入口。原始特征与目标真值仍按原兼容政策重建，非任意JSON标签输入。

原模型可在独立记录前或后形成，但均须在真实目标竞价前；不能把信号日之后形成的记录说成信号日预测。比较保留未知标签、主观弃判、完整正例Recall及覆盖，按板位与真实一字分层；一日不自动部署，也不迁移原三臂/G1/T1。首批11只已经看过BASE，明确不是盲测；目标10月12日未成熟，新增成熟原前向0。

2026-10-09新增周期分层、G1结构影子筛选及分歧提示，完整调用流程见[周期分层与G1使用说明](周期分层与G1使用说明.md)。新增`current_model_diagnostics.py regime`及theme的`--g1-protocol`，`theme_g1_validation.py`提供独立登记、追加结算、120日验收和通过后的新日期名单应用入口。正式名单当前仍为原前5，原数值模型及三臂实验不改；旧冻结日期只生成独立附页。

2026-10-08本轮新增显式生产推理入口。生产与固定比较分开：生产每个模块只加载已验证的显式模型；研究仍保留原BASE/C1/C2。当前没有已证实的替代赢家，BASE仅为在用基准，原分数、排序及0.5晋级阈值保持。无引用冗余可以删除，固定证据、原预测和必要回退原件按原路径封存；本轮15份历史权重均存在证据引用，没有可安全删除的权重。

```powershell
D:\python\python.exe model_champion_runtime.py predict --protocol <原固定协议.json> --module market --features <已核验当期特征目录> --output <已存在当期目录中的唯一新预测.json>
D:\python\python.exe model_champion_runtime.py predict --protocol <原固定协议.json> --module relay --features <已核验当期特征目录> --output <已存在当期目录中的另一唯一新预测.json>
```

如果有生效挑战，另传`--registration <同模块同信号日原生影子登记.json>`；程序重新核对原验收、授权、来源及时点，并执行新增全候选Accuracy至少改善3个百分点、完整Recall/F1/覆盖不降及完整同池要求。按Accuracy择优，相同优先保留在用模型，再比较Recall、F1、字段数和固定ID。旧原生C1优先结果若与新增择优规则冲突，拒绝应用并明确需要追加政策适配；本入口不改变旧协议、自动切换、回退或登记，也不能代替真实作者审阅与原V298冻结。

已完成[M1延续类平衡固定比较](market_continue_20261008T115045Z/固定比较说明.md)：2026年1—8月Accuracy63.13%降至53.13%，虽延续类Recall上升，改善/恶化类明显退化；挑战淘汰，未保存部署权重，不重跑或换参数。历史已看过，不算盲测或新真实前向。原固定两模块今天已真实影子登记：个股10只完整、窗口启动；市场partial，只有登记而没有合格窗口起点，目标10月9日待发生。

2026-10-08晚间最新授权已覆盖下文永久保留BASE的限制：按完整同口径预测效果择优，允许固定证据支持后的淘汰、切换与回退。原hold与原验收协议保留；追加production_authorization_20261008.json和implementation_policy_AUTHORIZATION_20261008.json表达最新授权，model_upgrade_policy.py重验授权链。授权通过只解除hold，原120日来源绑定验收、作者审阅、冻结及回退门仍执行，本轮未切换任何模型。

每次有新日期有效底稿时，先运行以下题材梯队附页，再作主线与逐股分析。当前分类、行业代理、首板、连板、炸板和未知分别列出；原模型call冲突只提示复核，不替代独立判断。这个程序改进不等于数值预测改善。

```powershell
D:\python\python.exe current_model_diagnostics.py theme --pool-archive <当期原V293目录> --context-archive <同池V295原始或reconciled目录> --signal-date <当期信号日> --base-predictions <当期原BASE输出.json> --output <当期报告中的唯一题材附页目录>
```

独立知识截止默认采用真实运行时间，全部源档和分类必须此前可得；分类24小时，晚于原模型截止的分类不回写原预测。源归档不得含晚于信号日的目标交易日，正文叙事与主动题材/龙头身份仍需独立核验。纯函数保留完整首板和炸板成员，不读次日标签，不加分数或股票排序。

2026-10-09凌晨补齐主线分析入口：事实梯队生成后、独立主线结论及逐股分析前运行以下命令。先核对当期已经保存的原作者正文并记录题材、立场、精确片段和offset；政策与摘录形成时间均不晚于真实知识截止。程序核对原HTML、取得/发布时间、正文、来源清单和哈希，排除泛标签，并用连板数、首板数、连续梯队的支配关系区分主线、分支、竞争和伴随方向；不添加股票分数或买入条件。

```powershell
D:\python\python.exe market_mainline_analysis.py --theme-report <上述theme生成的report.json> --narratives <当期已核原文摘录.json> --policy <先于应用固定的识别政策.json> --output <当期唯一主线附页目录>
```

本次可核对实例见[独立主线识别附页](mainline_analysis_20261008T122231Z/主线识别补强.md)。电池父主题6连板＋3首板、固态焦点4连板＋2首板、航运伴随、银行在涨停池范围之外分别保留。38收盘／30炸板全68只保留；30只炸板分类仍未知。本入口是完整梯队结构推断，正文候选仍来自有限作者摘录；未摘录题材记限定范围未核查，不代表市场没有讨论。尚未实现全市场自动新闻发现或历史预测效果验证；本例已经看过，不算独立盲测。冻结日期只生成独立附页，不写回旧计划。

2026-10-07最新授权：用户要求“那就不替换，在原有基础上改进”。生产保留原BASE，已有挑战研究及固定验收可以继续；下面旧说明中的“验收后直接切换”已停止授权。`production_hold_20261007.json` 与 `implementation_policy_HOLD.json` 追加记录这次变更，不改评分前协议、模型、原结果或原验收。影子CLI在验收通过后仍只交付研究结果，`promote` 和挑战应用受到持久hold限制；作者审阅不能解除hold。

新增独立原模型辅助入口 `current_model_diagnostics.py`，先输出既有BASE误差与输入缺口附页，后续新信号只读取完整来源绑定特征和对应BASE登记。它展示原始分数、原判断、全候选原排序和首选资格，不产生新的模型分数或概率。原第十段前5与第十一段原结算口径继续保留，辅助提示不能抬高晋级判断或替代交易主选。

历史诊断仅为已成熟标签的误差说明。市场延续类、二板漏判、真实非一字排名首选、正向首选分开统计；未知标签与未知一字身份保留。所有历史档案和生成时间均在盘后，不声称当时可用信息或新增真实前向。生产仍先处理10月8日原11只结算。

已完成[保留现模型的诊断改进](fixed_20261007T112017Z/base_diagnostics_20261007T123953Z/保留现模型的诊断改进.md)：307个官方日期网格、3707条原候选仅读取一次。独立核对原类别、分数、排序及分段指标，原准确率不变，新增真实前向0。该历史输出不重跑。

当取得真实新交易日完整来源并完成两个模块的原生影子登记后，用以下入口生成BASE独立附页。两模块截止时间分别保留，共同背景不晚于两者较早截止，附页生成时间单独记录实际时钟。登记来源、模型、全候选和原始分数先重验；当前此路径仅完成合成核验，真实当期接入待有效新底稿。

```powershell
D:\python\python.exe current_model_diagnostics.py signal --market-registration <当期市场影子原登记.json> --relay-registration <当期个股影子原登记.json> --feature-archive <同一原特征目录> --output <当期现有报告目录中的唯一独立附页目录>
```

该命令只读两模块的BASE输出，不运行模型或添加观察卡权重。原十一段报告与前5观察卡按当前BASE继续制作；模型内部负向判断、未发生开盘观察、来源缺口与原失败仍保持。

本研究按2026-10-07用户批准的方案执行：先复现原V278，再完成市场三分类比较、个股晋级比较，随后两模块独立影子验证。此前66.13%反馈二分类、威科夫、题材时点研究不作为这次升级依据，不恢复题材专项采集。

本次协议目录：`fixed_20261007T112017Z`。`protocol_BEFORE_EFFECTS.json`先于效果计算；`implementation_BEFORE_FITS.json`与`provenance_BEFORE_FITS.json`在首次拟合前绑定代码、完整来源与依赖。原始结果和失败只留档，不能覆盖或调参重试。

## 固定比较与冻结

```powershell
$taskProtocolPath = 'D:\Documents\ChatGPT\龙头战法\research\model_upgrade_validation\fixed_20261007T112017Z\protocol_BEFORE_EFFECTS.json'
D:\python\python.exe model_upgrade_research.py reproduce --protocol $taskProtocolPath
D:\python\python.exe model_upgrade_research.py research --protocol $taskProtocolPath --module market
D:\python\python.exe model_upgrade_research.py freeze --protocol $taskProtocolPath --module market
D:\python\python.exe model_upgrade_research.py research --protocol $taskProtocolPath --module relay
D:\python\python.exe model_upgrade_research.py freeze --protocol $taskProtocolPath --module relay
```

这些命令在本次完成后不可重跑。冻结复用9月比较保存的估计器，不重新拟合。三段分别为2025下半年、2026年1—8月、2026年9月，全部训练标签早于对应段起点。基准与挑战相同训练成员；未来训练集、参数、阈值冻结在标签不晚于2026-08-31的版本。

## 当期影子预测与来源绑定结算

先用项目既有V293流程取得真实新收盘日的唯一行情档案。`--archive`指该档案，`--history-archive`按需列出首个信号日之后相邻日的既有档案。缺日保持未知；不取旧池充当新行情。当前指数输入若没有独立当期来源仍为NaN，遵循原训练期插补并明确披露，不能复制历史指数。

```powershell
D:\python\python.exe model_upgrade_research.py build-features --protocol $taskProtocolPath --archive <当期V293目录> --calendar <核验官方日历> --history-archive <前日V293目录> --output <唯一新特征目录>
D:\python\python.exe model_upgrade_shadow.py predict --protocol $taskProtocolPath --module market --features <新特征目录> --models <协议目录\models> --output <协议目录\shadow\market\日期\registration.json>
D:\python\python.exe model_upgrade_shadow.py predict --protocol $taskProtocolPath --module relay --features <新特征目录> --models <协议目录\models> --output <协议目录\shadow\relay\日期\registration.json>
D:\python\python.exe model_upgrade_shadow.py settle --registration <原影子登记.json> --archive <精确signal与target的到期V293目录>
```

预测必须在真实当前上海交易日收盘后登记，来源完整档案及三份模型已形成且不晚于真实截止。首个合格日锁定模块120日窗口；之后缺失、无候选、未知一字或弃判均留在官方日期网格。全候选评分保留，真实OHLC一字身份和旧低换手代理分开，不以代理通过正式首选验收。

结算只能追加。返回日期不符、价格差一分钱、明确涨停价或独立不复权来源不足均不能确认为晋级；未知不计失败。原9月30日登记与10月8日11只结算继续走原V293/V298生产链，影子账本不会改写它。

## 固定节点验收、生效与回退

```powershell
D:\python\python.exe model_upgrade_shadow.py bind-calendar --protocol $taskProtocolPath --calendar <新增官方核验日历.json>
D:\python\python.exe model_upgrade_shadow.py review --protocol $taskProtocolPath --module market --node 120 --output <shadow\唯一验收.json>
D:\python\python.exe model_upgrade_shadow.py review --protocol $taskProtocolPath --module relay --node 120 --output <shadow\唯一验收.json>
```

日历扩展保留原前缀、来源及哈希；不足覆盖120日及最后目标时等待官方扩展，不猜未来节假日。节点验收重建原登记和到期来源。固定120日成熟且全部门槛通过时，验收CLI直接切换该模块独立active指针；同模块两挑战通过优先C1。未通过维持BASE，不因历史胜负选赢家。

市场要求108个配对标签、每类20个，三分类Accuracy改善至少3个百分点，三类召回和宏指标不下降并通过两项朴素基线。个股要求60个完整配对候选日、至少90%覆盖，真实非一字排名首选改善至少3个百分点；全候选Accuracy、完整正例Recall、原始分数Brier、正向首选覆盖不恶化并通过同池最高板基线。四挑战共同校正区间固定5交易日块、2000次、种子20261007、分位0.00625/0.99375，下界须大于零；三个连续40日段不得负改善。

```powershell
D:\python\python.exe model_upgrade_apply.py --plan <当期未审阅全候选计划.json> --registration <当期该模块影子登记.json> --active <协议目录\shadow\active_pointer.json> --output <唯一新未审阅计划.json>
D:\python\python.exe model_upgrade_freeze.py --plan <实际作者审阅后的新计划.json> --calendar <显式核验官方日历.json> --output <唯一冻结核验收据.json> --registry <原V271登记目录>
D:\python\python.exe model_upgrade_shadow.py review --protocol $taskProtocolPath --module market --node 60 --output <shadow\唯一上线监测.json>
```

新模块只写生效后新日期的未审阅计划，市场和个股可分别应用；仍需真实作者审阅及`hypothesis_bound_daily_runner_v298.py freeze/tick`来源、背景和假设核验。不能改旧计划、旧概率、身份、失败、作者审阅或结算。新股票排序固定原始分数降序、代码升序；负向排名首选不能成为正向首选或交易主选。观察卡显示100×原始分数，保留一位小数，继续披露模型、文件和哈希；不称校准概率或盈利胜率。

含升级应用的新计划必须经`model_upgrade_freeze.py`，它在原V298冻结前重验生产guard、最新生效状态、来源、分数与排序，拒绝删除应用绑定绕过重验；原V298文件及全部背景/假设校验不变。wrapper的`--calendar`用于明确核官方日期和计划内日历；`--output`仅为唯一核验收据。原V298的freeze实际参数为`--plan/--registry`，wrapper使用无shell子进程传这些原参数，不能错误地向原freeze传它不支持的`--calendar/--output`。

上线后按固定非重叠60个成熟官方信号日监测。主指标落后旧模型至少5个百分点或任一关键召回下降至少10个百分点，自动回退该模块；来源、日期、语义、哈希或标签边界冲突立即停用新模块。旧模块无有效输入仍输出未知，不自动调参恢复。

本研究不修改定时任务或平台目标状态，不下单、不发送消息、不发布。30分钟预算按单模块固定研究批次记录；程序测试和采集条数不能代替预测改善。历史诊断不是新真实前向样本，60%仍是研究目标。


2026-10-09午间完成[个股R1训练类别平衡固定检验](relay_balance_20261009T044001Z/固定比较与误判复核.md)。三段各一次拟合，原成员、29字段、插补、252窗、C及0.5规则不变；Accuracy门未通过而Recall/F1上升，挑战淘汰、无生产权重、不重跑权重搜索。该批完成时目标尚未收盘，新增成熟前向0。

2026-10-09收盘已完成[时代万恒、紫竹高科漏判及R2核验](relay_error_20261009T045526Z/两只漏判与模型改进核验.md)。解释入口为`relay_error_analysis.py explain --base <原BASE输出> --history <原保存个股预测> --mainline <绑定同一BASE的独立主线附页> --output <唯一新解释.json>`。针对原固定协议中的V298/G1保护文件变化，独立只读解释须显式传`--reader-policy research/model_upgrade_validation/relay_error_20261009T045526Z/reader_compatibility_BEFORE_REPLAY.json`，同时核对原件、当前已留档扩展和全部实际特征依赖；该策略不是生产三臂保护的通用豁免。全候选原分数须复现，不能把logit解释当作评分加减项。

请求旧日返回新日时，使用`archived_signal_outcome_link.py --prior-archive <真实信号日保存V293档案> --target-archive <当期新V293档案> --registered <原登记> --output <唯一只读联合索引>`。只引用原始响应，原取得时间不动，父档案与被拒响应完整保留；原V293解析器重建通过后再交原桥接与V298 tick。没有合格原档案不得补造。原10只已结算；旧三臂保护哈希冲突尚待追加兼容修复，未追加影子结算。

## 2026-10-11继续完成：显式来源兼容与R3淘汰

上述兼容待修为原状态，已完成[原两模块补结算、正式推理适配及R3核验](relay_phase_20261009T095400Z/兼容修复与转换手特征检验.md)。原模型、控制器、冻结和分数没有变动；未来原三臂入口改为明确传入兼容证据的下列调用，所有原门仍执行。直接旧CLI仍按旧保护拒绝，不能删哈希或全局替换验证函数。

```powershell
$taskProtocolPath = 'D:\Documents\ChatGPT\龙头战法\research\model_upgrade_validation\fixed_20261007T112017Z\protocol_BEFORE_EFFECTS.json'
$taskCompatibilityPath = 'D:\Documents\ChatGPT\龙头战法\research\model_upgrade_validation\shadow_compat_20261009T093237Z\compatibility_BEFORE_USE.json'
$taskChampionReceipt = 'D:\Documents\ChatGPT\龙头战法\research\model_upgrade_validation\shadow_compat_20261009T093237Z\champion_implementation_BEFORE_USE.json'
D:\python\python.exe model_upgrade_compat.py --compatibility $taskCompatibilityPath build-features --protocol $taskProtocolPath --archive <当期有效V293目录> --calendar <已核官方日历> --history-archive <已有前日目录> --output <唯一新特征目录>
D:\python\python.exe model_upgrade_compat.py --compatibility $taskCompatibilityPath predict --protocol $taskProtocolPath --module relay --features <当期新特征目录> --models <原协议models目录> --output <原shadow内唯一新日期登记>
D:\python\python.exe model_upgrade_compat.py --compatibility $taskCompatibilityPath settle --registration <原登记> --archive <精确原signal和target的绑定来源>
D:\python\python.exe model_upgrade_compat.py --compatibility $taskCompatibilityPath review --protocol $taskProtocolPath --module relay --node 120 --output <原shadow内唯一验收>
D:\python\python.exe model_champion_compat.py --compatibility $taskCompatibilityPath --implementation $taskChampionReceipt --protocol $taskProtocolPath --module relay --features <当期新特征目录> --output <已有生产输出目录内唯一JSON>
```

市场模块仅把`--module`改为`market`，仍保留partial、未知和合格窗口门。新生产计划仍须真实作者审阅及原V298 freeze/tick；这里的模型调用不能代替冻结，也不能把旧日期材料当今天。模型通过固定验收才允许挑战，不能把兼容恢复当升级。显式推理适配已离线核对10只原分数/排序，真实新信号日计划待发生。

日历扩展已沿原链绑定到2026-12-31，不重复绑定。未来有新的官方有效扩展时，原`bind_calendar`同样在独立兼容命名空间内执行：

```powershell
D:\python\python.exe -c "from model_upgrade_compat import runtime; runtime(r'$taskCompatibilityPath')[1].bind_calendar(r'$taskProtocolPath', r'<已另存的有效官方扩展日历>')"
```

这仍调用原日历前缀与来源核验，不改原计划日历；不足120日覆盖时等待真实官方扩展，不猜节假日。

R3固定三个诊断段各一次已完成，整体Accuracy未提高，已淘汰，无部署权重；不得重跑`relay_turnover_phase_research.py`拟合。原三臂和G1窗口继续自己的协议；研究回放、补结算及测试不增加真实前向样本。下一有效新信号日先完整theme及独立主线，再原评分解释、同期原/G1观察附页与固定影子登记。


## 2026-10-12凌晨：情绪环境E1/E2重评完成，两挑战淘汰

交付见[情绪环境重评与固定比较](relay_context_20261011T151204Z/情绪环境重评与固定比较.md)。独立入口为`relay_context_research.py`及其来源绑定影子模块`relay_context_shadow.py`；原三臂/G1/T1均不修改。原协议是`relay_context_20261011T151204Z/protocol_BEFORE_FITS.json`，输入、程序、参数、成员与截止已冻结；6次拟合全部完成，不得再执行compare寻找更好的结果。BASE未重拟合。

E1为原29字段加7项环境量，E2再加14项中心化原同板位分位交互；精确官方前日，非verified来源保留NaN，原插补、252窗、每日权重、C0.25及0.5判断。历史Accuracy（BASE/E1/E2）：2025H2 64.71/64.96/64.84%；2026JanAug 71.97/72.65/72.18%；9月61.65/60.15/58.65%。两挑战均未通过每段+3个百分点及Recall等保护门；predict入口明确拒绝失败方案，未启动新影子窗口、未部署研究权重。现有估计器仅用于复现，清单标记研究用途。

完整误差文件`complete_error_diagnostics.json`与`all_candidate_explanation_table.csv`覆盖成熟原21只及下一目标尚未发生的11只，并有3707条历史分层。9月30日解释是从原保存输入当前回放，判断、一字身份与原登记逐股核对，不倒填。真实前向仍2日21只14正确；非一字13只的5个正例全部漏判，样本很小，不能按此即调阈值。

```powershell
$taskContextProtocol = 'research/model_upgrade_validation/relay_context_20261011T151204Z/protocol_BEFORE_FITS.json'
D:\python\python.exe relay_context_research.py diagnose --protocol $taskContextProtocol --forward results/weekend_review_20261011T142844Z/regime/report.json --explanation research/model_upgrade_validation/relay_context_20261011T151204Z/september30_saved_logit_replay.json --explanation research/model_upgrade_validation/relay_error_20261009T045526Z/original_logit_explanations.json --explanation results/weekend_review_20261011T142844Z/model/logit_explanations.json --output <唯一新诊断JSON>
```

`freeze`用于预登记，`compare`本批已锁定一次，禁止重跑。`predict --protocol --features --previous-features --calendar --compatibility --output`只允许全部历史门通过的方案；实际当前上海信号日收盘、完整核验资料、真实截止和相同冻结模型均必需。`settle --protocol --registration --archive --compatibility`复用明确兼容的原来源绑定桥，仅追加新的结算链；不接受手填结果。`review --protocol --calendar --output`复核独立固定120日、完整覆盖、同池配对、股票数加权Accuracy差值、5官方交易日块2000次种子20261007和0.00625/0.99375区间、3×40不退化、Precision/Recall/F1及真实非一字/正向覆盖。日历不足等待官方扩展，未知和无候选保留；因本批无历史胜者，这些未来接口保持未启用。通过后的显式生产接入须再绑定真实验收与原V298新计划门，不能靠接口存在宣布模型已升级。

`monitor --protocol --activation <真实通过验收收据> --first-signal <实际生效后首信号日> --node <60的倍数> --calendar --output`输出对应非重叠60日回退决定；Accuracy落后5个百分点或完整Recall下降10个百分点要求对应个股模块回退，不能自动调参恢复。目前没有新模块生效，无回退事件。原BASE也缺有效输入时输出未知。

新登记报告的判断表由`model_prediction_presentation.py`显示“模型偏晋级／模型偏断板”，原机器调用、数值、排序和旧日期渲染保留。`median_wording_correction.json`追加更正旧报告将反馈中位数写成均值的措辞；原文不覆盖。18项针对测试与独立矩阵/配对差值重算通过，原批中断和续批30分钟预算留档。唯一下一步仍为下一有效收盘的完整theme、独立主线、原评分解释和原固定三臂/G1留档，重点核对非一字漏判；不扩大采集、不恢复题材时点专项、不改平台目标或定时任务。


## 2026-10-12：H1/H2非一字评分结构固定比较，未通过升级

按最新“改，然后升级模型”授权，实现[全组与代理分支梯度提升](relay_expert_20261011T162713Z/非一字评分结构检验与升级结论.md)。入口`relay_expert_research.py`，模型类`relay_expert_model.py`可保存加载；独立`relay_expert_shadow.py`绑定原固定前向门，克隆独立命名空间，不修改E1/E2入口、原三臂/G1窗口或原程序保护。

原29字段、原252标签日、完整候选、全组每日权重及0.5阈值不变。H1整体评分；H2完整组加proxy=0/1分支，未知代理走全组，代理不替代真实一字。三段各臂拟合一次，参数与协议在效果前冻结，BASE未重训。Accuracy（BASE/H1/H2）为2025H2 64.71/65.28/65.59%，1—8月71.97/72.13/70.72%，9月61.65/60.90/60.90%；9月真实非一字Recall15%→2.5%/0%。全部历史门未通过，两挑战淘汰，禁止重跑compare或改参数找胜者，未登记新的120日影子。

```powershell
$taskExpertProtocol = 'research/model_upgrade_validation/relay_expert_20261011T162713Z/protocol_BEFORE_FITS.json'
D:\python\python.exe relay_expert_research.py review --protocol $taskExpertProtocol --calendar data/v298/exchange_calendar_extension_20261003_v2/calendar.json --output <唯一未来核验JSON>
```

`freeze`用于效果前冻结，本批已完成；`compare`已锁定一次，不再次运行。`predict --protocol --features --previous-features --calendar --compatibility --output`只允许全部历史门通过的挑战；`settle --protocol --registration --archive --compatibility`从原V293来源桥重建并追加结果；`review --protocol --calendar --output`沿独立120日、60配对、90%覆盖、Accuracy+3个百分点、5日块2000次共同校正区间、3×40与Recall/Precision/F1/非一字/正向覆盖门。同Accuracy并列H1优先，未知及无候选留在日期网格，日历不足等官方扩展。`monitor`沿同一新模块的真实验收收据及生效后的每60成熟信号日，参数同原独立入口。由于两臂未通过，这些影子/上线分支没有启动。

`original_oct8_full_cohort_diagnostic.json`保存完整原10只的新模型回放，时间晚于原预测，不当真实前向；时代万恒、紫竹高科仍漏判，园林股份误报更强。13项针对检查及独立矩阵/覆盖/配对差值重算通过；研究权重保留复现，清单非生产，不能以代码存在或测试通过宣称升级。原真实前向仍2日21只14正确、新增0。唯一研究下一步核查已有非一字训练/当期字段语义、来源覆盖与阶段差异，再另批预登记；新交易日生产原完整题材/主线/评分冲突和三臂/G1留档继续，不扩采、不恢复题材时点、不改任务状态或调度。


## 历史角色标签的时点隔离与匿名高度审阅（2026-10-12）

历史研究复用2023—2026归档，不等待120日才开发；正式效果及生产门仍独立保留。`audit`显示完整年度来源/标签覆盖。已有完整比较不可改参重跑。

```powershell
D:/python/python.exe -X utf8 leader_role_engine.py slice-dates --dataset <父角色数据集> --start <官方交易日> --end <官方交易日> --output <唯一新数据集>
D:/python/python.exe -X utf8 leader_role_research.py prepare-day-reviews --dataset <角色数据集> --output-dir <新的逐日目录>
D:/python/python.exe -X utf8 leader_role_research.py prepare-anonymous-height --dataset <只含pool和price的角色数据集> --output-dir <新的匿名目录>
D:/python/python.exe -X utf8 leader_role_research.py import-reviews --dataset <角色数据集> --packet <匿名目录/anonymous_review_facts.json> --anonymous-mapping <匿名目录/operator_mapping.json> --reviews <审阅A.json> <审阅B.json> --output <唯一新标签>
D:/python/python.exe -X utf8 leader_role_research.py aggregate-labels --dataset <合并角色数据集> --origin-datasets <原每日数据集...> --origin-labels <对应原标签...> --output <唯一新标签汇总>
```

具名原事实包一份仅一个信号日。不得把后续日期的同股板位在先前样本审阅前一起展示；多日原包只作后台完整性档案，不能认证盲审已知值。匿名V4隐藏真实日期、股票名/代码、价格和跨日资产连接，展示互不关联的各日完整原板位；只允许HEIGHT参考，其余类型缺或遮蔽资料保持未知。需要身份、叙事、周期及反馈的类型不能用此匿名高度包代替。

匿名映射为后台核对专用，不展示给审阅者；来源、原候选及映射逐次重建并哈希校验。实际两份独立收据必须声明真实审核时间、当前case对应模型/未来信息是否遮蔽、具体上下文暴露及是否可识别当前case。不因匿名自动认证合格，也不把不相关时期的披露抹掉。已知标签回查原收据、对应意见与证据，意见冲突、识别到暴露或无法盘点暴露的case保持未知。

这里的独立盲审标签资格指当前历史诊断的事实判断来源，不是原时点取得或120日前向验收已通过。历史资料/标签真实形成较晚保留实际时间，STRICT轨仍拒绝晚到/重建输入。合并原标签不改原审核/可得时间；严格计划冻结时保留预备训练标签及全部原未知。


2026-10-12按用户要求接入[短线思维与九阳逐日推理](short_term_thinking_20261012T153000Z/短线思维接入与九阳拆解.md)。新增`stock_short_term_context.py`，原`leader_role_review.py prepare/freeze/compare`关联启动/延续、主题切换、独立强度候选与分歧竞争四项事实审阅。只读来源事实，不加数值分、不训练；旧已冻结卡不补预测。当前18只四项完整事实门尚未通过，新增前向0。

```powershell
D:\python\python.exe leader_role_review.py prepare --features <当期原特征目录> --theme-report <当期完整theme.json> --mainline-report <当期mainline.json> --compatibility <原兼容政策> --bank <当期核对的催化银行> --calendar <官方已扩展日历> --output-dir <唯一新目录> --history-dataset <已有合格跨日角色dataset.json> --mainline-history <已有同范围前日mainline.json>
D:\python\python.exe leader_role_review.py freeze --facts <新目录/facts_with_roles.json> --judgements <完整逐股实际判断.json> --output <唯一新事前记录.json>
D:\python\python.exe stock_short_term_context.py build --dataset <原角色dataset.json> --catalyst-context <原催化context.json> --history-dataset <原合格跨日dataset.json> --mainline-report <原mainline.json> --track DIAGNOSTIC --output <唯一当前诊断.json>
```

`--history-dataset`、`--mainline-history`均可重复，缺资料可省略并保留未知。build不登记预测；prepare/freeze仍受原竞价前时窗和完整候选约束。原角色11项与催化评估仍须全部填写；新每行`short_term_assessment`字段示例见上述目录`short_term_review_template_PENDING.json`（仅模板，不能视为AUTHOR_REVIEWED）。其中四项`checks`为`launch_and_continuation/theme_transition/independent_strength/absorption_and_competition`，每项`state`为SUPPORTED/CONTRADICTED/UNKNOWN，另填`reason/evidence_refs`；填写`alternative_explanations/counterevidence/cancel_conditions`数组、`confirmation_plan`的带时区`planned_at`、PENDING与观察要求，`role_identity_confirmed/fund_causality_confirmed`均false。SUPPORTED只对应附页声明的事实范围；明确具体主题的叙事才可支持主题参与，报板位点名不说明启动催化。当前不自动识别个股叙事退出时间，市场主线、成员归属与叙事切换分栏。


## 2026-10-13凌晨：既有五分钟路径接入，换手身份仍未验收

交付见[历史五分钟路径与换手身份核验](leader_role_sampled_path_20261012T153639Z/历史五分钟路径与换手身份核验.md)。`leader_role_engine.py sampled-history --dataset --report --reported-seals --output-dir`从已有原始响应复建全部182条9月路径，保留原9245候选与BASE；正式产物为该目录的`actual_sampled_history_CAPTURE_BOUND/history_with_sampled_paths.json`。初始缺捕获清单绑定的投影保留，当前入口拒绝将它当合格新来源。

29条出现采样收盘的涨停→回落→恢复序列；66条供应商报过炸板但采样收盘没有回落，九阳9月30日是具体反例。48根五分钟OHLC、首末封板间隔、开板持续时间、真实回封和承接分栏；网格不全时未观察到的序列保持未知。27条有原清单哈希、155条未知，均保留现有v275原始字节哈希与实际重建时间。新增7字段与旧冻结95字段分开，旧三次高度拟合和零增益结论不重跑。80项相关核验通过，本批新增身份真值0、拟合0、真实前向0；统一验收未通过。


## 2026-10-13凌晨：事前预期与证伪接入

用户希望项目短线思维进一步精进，本批[不足与优先级](short_term_refinement_20261012T161258Z/短线思维不足与精进优先级.md)核对原18只卡片、既有11只结果及五分钟边界。新增`stock_expectation_review.py`接入`leader_role_review.py freeze/confirm`，原调用入口不另造登记。`prepare`绑定预期解释源码；新freeze每个`short_term_assessment.confirmation_plan`需`checkpoint_id/baseline_kind=AUTHOR_PREDECLARED/planned_at/observation_window/status=PENDING/requirement/paths`，paths含normal/stronger/weaker，各为DEFINED+具体条件或UNKNOWN+原因（conditions空）。完整待审模板见本目录`short_term_refinement_20261012T161258Z/expectation_schema_template_PENDING.json`；模板不是新预测，原10月13日卡不得补填预期。

DEFINED条件含`condition_id/metric/scope/scope_id/comparison_codes/operator/expected/unit`；scope为STOCK/THEME/PEERS/MARKET，比较成员必须等于事实卡对应合格完整范围。强/弱需个股及外部条件，不能用高开单项确认。actual confirm仍调用`leader_role_review.py confirm --record <原已登记记录> --confirmation <实际观察JSON> --output <唯一追加附页>`；原登记路径/哈希、原事实/来源均重验，逐行按code及原checkpoint_id引用观察proof。proof含condition_id、source_path/source_sha256、确切JSON pointer、raw_value；原源cell含metric/scope/scope_id/comparison_codes/unit/value/observed_at/available_at，时间必须真实可得且处于原窗口。无condition_id只作为未评分观察，不推强弱。缺条件UNKNOWN，多路径PASS为AMBIGUOUS，部分观察保留完整分母、覆盖及未观察名单；legacy不能补造预期。窗口先后观察不自动认证同刻盘口或承接，源字段一致也不认证资金因果。

短线层另补当期cycle末观察、pressure_pre_event_core及窄反证：每项CONTRADICTED须counter_claims准确匹配对应counterevidence_details.target_claim，不能从成员归属未变否定市场/叙事切换。现有原数字/研究层不由本批改写。18+15+88=121项相关测试及5项实际源重建/时钟/原登记检查通过；新增真实前向0、拟合0、Accuracy改善未验证。原失败和未知均保留，详见本批final_verification.json。同一工作区角色引擎及研究层另次修改单独收据，不绕过旧保护；新的当前诊断绑定实际源码并重建通过。


## 2026-10-13凌晨：同期竞争路径与第二轮独立审阅

交付见[同期竞争路径与独立角色审阅](leader_role_peer_paths_20261012T161008Z/同期竞争路径与独立角色审阅.md)。`leader_role_engine.py peer-history --dataset --output-dir`保留9245原候选，复建209股票日路径、122条完整当前收盘同板位比较及1条完整旧对手比较，其余26条旧竞争缺口保留。现收盘池范围是存活者集合，不用作完整旧竞争或全市场证明。新增14个事实字段与原冻结95字段分开，本批拟合0。

`leader_role_research.py prepare-anonymous-peers --dataset --output-dir`提供V5不连通当日匿名原始OHLC包；`import-reviews --dataset --packet --reviews <A> <B> --anonymous-mapping --output`绑定实际收据，原全9245×11格保留，未审候选仍未知。两位代理对182×11格一致：高度19正163负，其他10类各182未知；没有复杂角色已知标签。实际导入/加载核验通过，旧独立高度档案、3次拟合、零增益、BASE和原失败均保留。88项相关核验通过，未通过统一验收、未部署、新增真实前向0。下一步现有690日市场原始记录连续性核验，不扩采、不恢复已停题材时点专项。


## 2026-10-13：叙事分型与审美证据接入

本批[叙事分型与短线审美接入](narrative_aesthetics_20261012T163846Z/叙事分型与短线审美接入.md)新增`stock_narrative_review.py`。原`leader_role_review.py prepare`增加可选`--narrative-reviews <实际分型审阅.json>`，原完整遮蔽分数卡增加reviewed_narrative。未提供审阅时明确原摘录未分型，不凭NARRATIVE字段自动确认交易题材。新freeze需每股narrative_type_assessment的current_trade_narrative_presence与卡片一致，content_types_reviewed/unknown_not_negative为true、reason写实际来源解释、fund_causality_confirmed为false；旧compare允许原卡没有新字段，不回写旧预测。

审阅bank格式实例为本目录`narrative_aesthetics_20261012T163846Z/typed_narrative_reviews_AFTER_ORIGINAL_FREEZE.json`：version=stock-narrative-review-1、原bank_path/sha256、实际formed_at/reviewer、完整reviews。每条原NARRATIVE都以claim_index/claim_sha256绑定，填写code/content_type/reason/reviewed_at及精确source_span(offset,text)。内容类型为STOCK_MENTION/PRICE_BOARD_SUMMARY/EXPLICIT_TRADE_NARRATIVE/COMPANY_MARKET_DISCUSSION/UNKNOWN；明确交易或公司热点关系另填topic_id/topic_label/topic_anchor/topic_level和relationship_review。父主题PARENT、焦点FOCUS、名称叙事NAME_NARRATIVE分开，股票与主题关系须实际核原文；未知不得猜topic_id，同一ID不得换定义，实际审阅晚于原截止不回填。

本批完整18只及19原摘录保留，16旧NARRATIVE分为10明确交易关系、5板数摘要、1公司热点说明；8只在信号日发表的传媒段落有原文联系，其他10只当前叙事未知。此为有限来源的语义分型，不是全市场主题成员、资金因果、业务或核心认证；初始19字段/权重/排序不由本层改动，也不把新叙事补入旧训练。137项相关测试和8项实际接入核验通过，新增真实前向0/拟合0/预测增益未验证。唯一下一步按新未冻结日期事实准备和事前判断检验，原10月13日冻结保持。


## 2026-10-13：价格路径审美描述与反例

[审美价量比较与反例](aesthetic_path_review_20261013T034806Z/审美价量比较与反例.md)新增 `relay_aesthetic_context.py`，只描述已合格价格投影、比较范围和待核问题；不生成晋级分数、核心身份或买点。已接入 `stock_short_term_context.py` 的完整候选事实卡，新日期沿原prepare/freeze入口；旧冻结只生成实际时间附页，不补写预测。

```powershell
& 'D:/python/python.exe' relay_aesthetic_context.py --dataset research/model_upgrade_validation/leader_role_peer_paths_20261012T161008Z/actual_peer_history/history_with_peer_paths.json --track DIAGNOSTIC --output <唯一新文件>
```

STRICT继续要求原字段实际可得性与同口径核验；本批历史182例是DIAGNOSTIC，原9245身份、133已知/49未知标签保留。采样恢复29例中的20已知有14例断板，当前幸存者持续相对领先30例中的25已知有11例断板；仅作反例诊断，不能作为概率或策略验收。完整原对手仅1例，不宣称角色识别已修好。当前18只诊断附页均缺合格分时，不能拿9月路径填10月12日。初版/最终数值与分类一致且原件保护通过。


## 2026-10-13：宽题材资格修正与来源置信等级

[宽题材资格修正与全类型识别缺口](leader_role_theme_qualification_20261013T035922Z/宽题材资格修正与全类型识别缺口.md)修正V75把所有原帖主题当specific的问题。`leader_role_engine.py analyse --dataset --track DIAGNOSTIC/STRICT --output`保留宽父主题/混合范围的原关联，相关角色UNKNOWN；未经审定的主题粒度未知。24个原始主题定义、B/C来源等级及原哈希从已绑定嵌套关系档案恢复，明确点名不等于业务或资金驱动核验。完整9245候选历史轨THEME/CYCLE由23更正为10、NODE由1更正为0；STRICT全角色0。101项核验通过，本批拟合0/新增真实前向0/整体验收未通过，原固定高度三次拟合零增益保留。

002790的归档名称为瑞尔特，之前两份研究报告及聊天写作晋拓股份的名称依据撤回；原件保留，追加名称更正及23例原档交叉核对，代码/竞争/行情未改变。此前市场标签session30401已实际退出0并完成加载核验，101695格已知0，不再重启同一导入。两位审阅者原资格声明和分歧保留，旧独立高度标签另存。唯一下一步核实现有V168/V169焦点成员及保存业务/事件证据，不扩采或重跑失败挑战。


## 2026-10-13：自身强度的声明范围与压力时点修复

[范围与压力时点修复](independent_strength_scope_20261013T041020Z/自身强度的范围与压力时点修复.md)沿原stock_short_term_context/leader_role_review入口，不新增评分。independent_strength facts增加declared_cohort_observation/history/changes、qualification_gap_dates、pressure_window_alignment；同名范围还须实际成员一致，缺资格不作成员退出。共同上涨领先、下跌中的相对抗跌、个股上涨与声明同伴平均涨幅非正分别描述；平均非正不表示所有同伴走弱，有限原文名单不认证完整主题。

SUPPORTED仍只为明确声明范围的候选解释。事件后观察北京时间须匹配信号日，前后与事件处于本次相邻官方交易日窗口且排序有效，压力后竞争覆盖原声明成员并集；旧核心保留不能支持当前窗口，日级匹配不认证同刻盘口。当前23条仅2条完整局部收益、0条对齐独立候选；当前18只重建/范围篡改/删候选核验通过，原预测不动。170项相关检查通过，新增成熟0/拟合0/准确率增益未验证。源码变化与未完成/失败会话单独留档；最终资料正常原路径重建，不用迁移快照绕过保护。


## 2026-10-13：归档叙事与业务成员关系接入

[来源置信关系接入与具体漏判诊断](leader_role_confidence_20261013T042602Z/来源置信关系接入与具体漏判诊断.md)提供 `leader_role_engine.py confidence-roster --manifest --calendar --output [--membership-manifest]`，支持原V168+V169来源重建；`confidence-history --dataset --rosters <一个或多个原重建档案> --output-dir <唯一目录>`追加THEME_NARRATIVE/BUSINESS_MEMBERSHIP_CLAIM关系，不导入自动角色标签为真值。9月16日100只、21日128只全候选保留，165题材标签、27多标签及63未知；原公告/正文没有实际绑定，不认证业务、驱动、核心或实际取得时间。两个具体实际源档案路径见本批目录。

原9245候选增加30题材关联及1业务成员声明，全部原组/BASE和新关联候选的数值特征/角色调用相同。完整严格轨0，历史资格未提升；112项相关核验及实际错哈希/日期/已废旧收据/伪造角色标签拒绝通过。新CLI和历史源接入实际完成，旧固定高度3次拟合零增益保留，本批拟合0/新前向0/整体验收未通过。下一步固定验收指标接口审查，不重跑旧比较。


## 2026-10-13：完整验收指标与准确率增益核验

[验收指标修复与准确率增益核验](leader_role_acceptance_metrics_20261013T045124Z/验收指标修复与准确率增益核验.md)提供 `leader_role_research.py rank-audit --report <原保存report.json> --protocol <原协议> --output <唯一新文件>`，仅重算保存预测，不读取／拟合模型、不重跑旧挑战。完整TopK保留整日和竞争者缺分数，条件有分数命中另列；全类型核心指标以各类独立正例为真值，不以HEIGHT认证总龙。独立审阅cell可加competition_position=LEADING/COMPETING/REAR/UNKNOWN，实际两位合格收据同判断并绑定事实后才用作后排标签，旧审阅不补造；BASE/C1后排误报、未知及覆盖保留。

正式review缺日历或120日未成熟明确不通过，未结算scope完整保留；失效前末正例与后续恢复分开、未检出损失纳入覆盖。2000固定抽样中全类型F1不可定义时不丢弃该抽样求有条件区间。实际保存7414预测的最终无拟合核验及101695原标签加载通过，118项相关检查通过；原晋级三段Accuracy增益仍0，严格Accuracy未知，不算0准确率。新版目标包括提高准确率，目前未证明提升，本批拟合0／新前向0／未部署／整体验收未通过。旧协议、旧指标和失败原件不覆盖，新metric_policy实际冻结并核验；主引擎本批未改。


## 2026-10-13：逐对竞争与均值反例

[逐对竞争接入与均值反例](pairwise_aesthetic_20261013T044951Z/逐对竞争接入与均值反例.md)新增relay_pairwise_comparison.py，接入stock_short_term_context._build及新日期审阅。复用已有182/209路径；122条当前完整比较、1条原对手完整比较。30条持续领先均值中16条曾落后个别对手，14已知6晋级8断板，2未知；不是新负向规则或概率。

```powershell
& 'D:/python/python.exe' relay_pairwise_comparison.py --dataset research/model_upgrade_validation/leader_role_peer_paths_20261012T161008Z/actual_peer_history/history_with_peer_paths.json --track DIAGNOSTIC --output <唯一新文件>
```

新日期draft每行pairwise_competition_assessment含scopes.current/prior，state=REVIEWED/PARTIAL_REVIEWED/UNKNOWN、reason、comparison_peer_codes、incomplete_peer_codes、stronger_observed_peer_codes，均对应完整原事实；next_session_paths分别joint_promotion/joint_failure/divergence，state=POSSIBLE/CONDITIONAL/UNKNOWN及reason。price_rank_is_not_core_identity=true、role_causality_or_entry_confirmed=false。示例为pairwise_review_schema_TEMPLATE_DELIVERY_ONLY.json，待真实作者审阅，未冻结。没有价格路径仍保留合格已知同板位成员，不以空名单消失。

最终交付文件带DELIVERY后缀，82项相关核验及完整源重建/当前18只/篡改拒绝通过；初版价格/分类相同，增加缺路径成员显式保留。原9245及133已知/49未知保留，当前18只合格逐对价格源0、已知当前有方向成员关系184条。原预测未改，拟合0/新成熟0/增益未验证；历史派生DIAGNOSTIC不能冒充原时点STRICT输入。


## 2026-10-13：新prepare同时生成完整作者待填草稿

[独立审阅入口贯通](review_entry_20261013T052032Z/独立审阅入口贯通与完整草稿.md)补齐leader_role_review.py prepare输出author_draft_PENDING.json；原完整事实/角色/叙事/短线/逐对与事前预期字段一次给出。方向、模型暴露、角色状态与作者理由未填，不自动判断或排序。真实作者完成后status=AUTHOR_COMPLETED_ROLE_REVIEW、author_review_completed=true，沿原freeze；标记不认证盲测或预测效果。

新增 `leader_role_review.py draft --facts <完整合格事实> --output <新待填JSON>`，要求实际盘后至目标竞价前及原来源重验。显式--diagnostic只给旧日期当前诊断，不补confirmation_plan、不能新登记。空模板/诊断模板在冻结前拒绝，新增入口不替代原生产入口或更新定时任务。实际18只源接入、184关系、两个真实拒绝及102项相关/6子核验通过，成熟0/拟合0/增益未验证；新日期真实prepare仍需有效新底稿与作者实际审阅。


## 2026-10-13：全类型暖启动资格与C2标签映射缺口

[全类型训练准备资格与真实来源缺口](leader_role_readiness_20261013T051911Z/全类型训练准备资格与真实来源缺口.md)实际核定当前9245候选：历史最近252合格标签日仅高度392正2665负，其他10类0；严格全角色0。原独立标签仅在必要组完全相同时核对，不补审或改标签，不重新训练已结束高度。尚未接入的原V289303五分钟／V2902888三十分钟候选全部属于原集合；356／2150原响应及旧审计哈希通过，277／2862合格路径原谓词保留，10条一分钱收盘高价矛盾不转失败真值或身份认证。资料实际10月2日形成，不倒填前向，也不重跑旧V294路径数值消融。

另发现C2把训练目标从BASE评测预测表读取：2025H2五日隔离后原来源4027目标在BASE表全部漏读；1—8月潜在3281中漏1782。原BASE实际训练集合与C2隔离集合分开核对，正式证据为本批EXACT_ORIGINAL_AND_GAP。唯一下一步独立绑定原promotion训练真值与实际成熟时间；不填造BASE分数、不把标签放进身份特征、不重拟合旧比较。无新测试（程序未改）、拟合0／新前向0／Accuracy未提升／统一验收未过。


## 2026-10-13：观察时刻与实战演练闭环

[观察时刻修复与实战接入](expectation_time_20261013T054040Z/观察时刻修复与实战演练接入.md)按新目标把能力用于实际演练。新计划observation_time_rule=AT_CHECKPOINT（默认，源observed_at必须等planned_at）或DECLARED_TIMES（每条件evaluation_at事前固定且在窗口内）；新freeze必填，确认不得回写。旧合同无规则只保留LEGACY_UNSPECIFIED_WINDOW解释，不改旧登记。

leader_role_review.py practice-collect --record --output用原collector并带独立登记全部代码，实际目标收盘前拒绝；practice-compare --record --baseline --outcome-archive --output按原角色/催化/基础适配器同池比较。完整18只与BASE日期/输入/代码已核，实战固定应用协议留档；13→14增强计划待有效新底稿和真实作者审阅，未新登记。109相关+6子核验非预测提升，成熟0/拟合0，原正式窗口及升级门保持。


## 2026-10-13：原18只到期同池结算

交付见[原18只结算与识别误差](../../results/outcome_review_20261013T073137Z/原18只结算与识别误差.md)。`leader_role_review.py practice-collect/practice-compare`实际沿旧催化/基础适配器核原18只，17已核、新华未知。原12日错日返回拒用，原绑定12日真实池与新13日来源另存V293组合档案；V293桥/V298 tick与固定三臂完成原口径结算，真值相同。原独立50%（6/12）与BASE82.35%（14/17），完整Recall40/60%，判断覆盖72.22/100%，label覆盖94.44%；不是单日升级证据。正式累计3日39登记38已核28正确，Accuracy73.68%，本批新增17；旧独立11只不混并。原预测/评分/知识截止、失败和固定窗均保留，Brier未校准不可算。源码未改不重跑测试。完整错因、原理由、弃判和分层见同目录JSON/CSV。下一生产13→14先当期完整事实/主线及真实增强审阅；未核停牌/盘口不补造。


## 2026-10-13：原晋级训练真值独立绑定与C2接入

[原晋级训练真值接入与C2修复](leader_role_training_truth_20261013T054938Z/原晋级训练真值接入与C2修复.md)新增leader_role_training_truth.py，`--dataset --receipt --output-dir`实际沿原prepared源将完整9245条晋级标签单独附入，9178已知/67未知；原事实、29字段、adapter和BASE评测不改。C2训练与plan-strict改从该层读取，成熟时间和STRICT实际源/投影时效核对、五日隔离/252窗和全11头OOF要求不变。各潜在成员4027/3281/3073，旧BASE表漏4027/1782/0；STRICT旧日期0，不能倒填时效。127相关核验及完整实际源加载/窗口复算通过，原保存分数缺其余10头时接纳0；不把保存评测分数当OOF。固定旧三次拟合/零增益/失败保留，本批拟合0、前向0、未部署/全类型验收未过。原13:49批与续批/外部中断/优先生产墙钟如实留档，不重跑已结束数值比较。下一未冻结14日真实增强审阅与准确率检验，角色标签/OOF条件未满足不启动C2。


## 2026-10-13：增强独立审阅首次用于新日期完整10只

[完整事实与14日独立预案](../../results/daily_review_20261013T074936Z/完整事实与14日独立预案.md)实际使用leader_role_review prepare/freeze，10只6晋级/2断板/2未知已原registry登记，先于新BASE输出。固定AT_CHECKPOINT、所有同板位和三路径，11类/叙事/短线/逐对缺口完整留档，未知不单独负向；原角色价量回放和pair状态拒绝保留并纠正。原63/16源和V295分类复建通过，mainline UNKNOWN因当前盘后原作者文本未核。新3臂13→14已登记，G1事实/风险附页不等于验收。原模型前5和10只完整冲突表保存，新10独立目标未成熟；正式V29815项作者审阅/冻结待续接，不能混称完成。16.4分钟超15后停止，拟合0/新前向0/准确率提升未验证。下一步原正式作者审阅及冻结，不重复独立登记、三臂或旧已核结算。


## 2026-10-13晚间：原V298冻结已核与G1排序缺口

[正式登记核验与G1缺口](../../results/daily_review_20261013T074936Z/formal_freeze_20261013T1118Z/正式登记核验与G1缺口.md)核原13→14 47f5FORWARD_REGISTERED、原10分数/方向/rank及全15假设、源哈希与tick通过，WAIT_TARGET_CLOSE，重复freeze拒绝。G1register因canonicalBASE/G1缺rank拒绝，原正式rank齐全；保存失败和精确更正，不改13材料或倒填截止。下一未冻结日期必要修复BASE投影rank传递。原10logit复现，分类61/63和晚间搜索提示/HTTP失败补正单列；拟合0/新成熟0/增益未验证。


## 2026-10-13：显式BASE完整排名投影

[原模型排名投影修复与G1完整核验](base_g1_projection_20261013T113705Z/原模型排名投影修复与G1完整核验.md)新增`model_prediction_view.py --market-output <原市场JSON> --relay-output <原个股JSON> --compatibility <原兼容策略> --implementation <champion原收据> --output <唯一新JSON>`。复用原源/模型/日历/特征投影保护，`rank`来自原完整`forecast.ranking`，并列score降序/code升序，UNKNOWN仍rank未知。下一未冻结日期原独立判断留档后运行原模块，再本投影，输出传给`current_model_diagnostics.py theme --base-predictions ... --g1-protocol ...`；继续原V298审阅及G1来源/时点门，不手工漏掉排名。

完整原13日10只实际G1回放与原正式板位/score/call/rank相同，7相关+7子例/5错误CLI拒绝通过；新视图实际晚间形成不补原13G1登记，旧失败及冻结/分数哈希保留。新成熟0/拟合0/Accuracy未提升；复杂角色标签/OOF仍不合格。下一研究为现有非一字字段语义、缺失/阶段差异核查，不能重跑已结束挑战或把功能成功当模型增益。


## 2026-10-13：完整选择理由与论坛表达接入

[选择逻辑与论坛经验接入](choice_reasoning_20261013T113350Z/选择逻辑与论坛经验接入.md)新增只读 `relay_choice_reasoning.py`。下一未冻结日期先完成完整事实、主线和独立作者留档，再生成原BASE输出与原分解，最后调用本入口；不修改原预测或冻结文件。已冻结13→14实际10只只做晚间附页。

```powershell
D:/python/python.exe relay_choice_reasoning.py --base <原BASE视图JSON> --record <已登记独立完整记录JSON> --forum-lessons <已核原作者正文经验JSON> --output-dir <唯一新目录> --reader-policy research/model_upgrade_validation/relay_error_20261009T045526Z/reader_compatibility_BEFORE_REPLAY.json
```

已有原完整分解可改传 `--explanations <原解释JSON>`；若已有正式登记，可传 `--registered <原正式登记JSON>`，只保留该登记的交易主选。省略 `--explanations` 时内部调用原 `relay_error_analysis.explain`，并逐项核对原模型、输入及来源，不新增拟合。正文经验样例为 `choice_reasoning_20261013T113350Z/checked_reasoning_expression_lessons.json`，只提供作者推理和表达检查，不认证行情、催化或交易规则。新日期不得把这份经验的较晚取得时间倒填。

输出原完整排名、模型贡献及完整同位逐项差值，独立支持/反证和原09:25/09:35/10:00条件分别展示；真实一字与旧低换手代理分栏。实际10只、40条有方向同位比较、735来源绑定核对及10项相关测试通过，非新增预测或准确率提升。原13→14分数/方向/排名、正式登记和未知不改；下一步14日实际确认及完整结算。


## 2026-10-13：原29字段语义与训练范围接入

[非一字输入核查与解释修复](nonword_input_semantics_20261013T115819Z/非一字输入核查与解释修复.md)新增只读 `relay_field_context.py`，在原完整选择解释之后调用：

```powershell
D:/python/python.exe relay_field_context.py --base <原BASE视图JSON> --choice <relay_choice_reasoning完整report.json> --prepared-receipt research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json --output-dir <唯一新目录>
```

当前入口使用该原准备资料及模型清单，重验完整代码、原训练摘要/成员、输入、原贡献和原分数；派生说明同时给出训练分布、真实一字/代理来源、完整上一日反馈范围。新模型或新准备资料须有实际来源和相同字段资格，不能沿用无效旧收据。数值范围与分位均只描述，不做新阈值、打分或未知方向推断。当前10只实际调用/7核验及完整1218行统计复算通过；原39反馈缺新华1只，不能填实际尾损或确认停牌。原14登记/分数/生产代码不改，新成熟0/拟合0/增益未验证。旧首版源有实际前像，最终来源资格收据单列，不覆盖旧产物。


## 2026-10-13：非一字语义与代理证据预登记

[非一字漏判的字段语义与固定新假设](relay_semantics_20261013T115031Z/非一字漏判的字段语义与固定新假设.md)覆盖9245原候选、3707原BASE及正式39。27真实非一字8正漏7；旧551代理证据不足，单一`protocol_PROXY_EVIDENCE_BEFORE_ANY_FIT.json`已预登记，拟合0。29字段中只使旧不充分proxy为NaN，knownFalse保留，不把缺封时6950全转未知；原成员/标签/权重/参数/0.5及原BASE保存。

注意本批context曾在12:01完整18/10 CLI及6相关测试通过，但共享`relay_field_context.py`现为另一实现，收尾hash冲突已拒绝，不能按旧命令宣称当前通过。下一步先恢复本批到独立名称及原前像/源码绑定，再必要核验后另批一次固定比较；不覆盖当前文件、重跑旧失败、替换身份真值或迁移窗口。统计/协议保留，原源哈希未改，真实新成熟0/拟合0/整体验收未过。


## 2026-10-13：代理证据挑战实际拟合与淘汰

[预测能力固定比较交付](prediction_delivery_20261013T123946Z/预测能力固定比较交付.md)复用独立relay_turnover_reference_context.py精确前像恢复，实际完成已预登记PROXY_EVIDENCE_UNKNOWN。新入口relay_proxy_evidence_research.py只把旧551不充分代理记NaN，原其余字段/成员/标签/252窗/权重/参数与0.5保持，三个段各拟合一次。Accuracy BASE/挑战为64.71/64.71%、71.97/72.28%、61.65/61.65%；9月真实非一字Recall15%→17.5%，误报4→5、Precision60%→58.33%，整体淘汰、无部署权重、不改规则或参数重跑。

六项针对核验、7414行完整CSV/分层独立复算和源哈希一致；一次性共享锁及3实际fit收据、重复compare拒绝保留。旧预登记/失败/原来源冲突均保留，本批恢复与固定比较已完成，不再记该挑战待拟合。原14日10只预测/分数/正式冻结和39登记38已核28正确均不改，新成熟0；实际拟合与局部Recall变化不构成通过验收的预测增益。下一步14日完整10只实际确认及来源绑定结算。

## 2026-10-13：信号日OHLC独立身份附层

[信号日OHLC身份补层与原漏判结算](signal_ohlc_20261013T1257Z/信号日OHLC身份补层与原漏判结算.md)新增只读 `relay_signal_ohlc_context.py`，使用四份原V278/V289/V290日线档案，逐份原响应/日期/原请求候选/哈希重建；不读取未来标签确定身份、不扩大源范围或拟合。

```powershell
D:/python/python.exe -X utf8 relay_signal_ohlc_context.py --receipt research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json --calendar data/v298/exchange_calendar_extension_20261003_v2/calendar.json --output-dir <唯一新诊断目录>
```

完整9245条原成员保留，2878条信号日OHLC身份已知、6367未知（含10条一分钱高价矛盾）；原三个训练窗已知0/753/2678。历史取得时间较晚，STRICT原时点合格0，不接生产数值。84条原proxy差异不自动改分；原3707条BASE全部保存预测与指标不变，新身份分层见独立复算JSON/CSV。8项相关核验通过非准确率改善。唯一研究下一步核查已存V16更早信号日不复权日线及来源链，不重跑原V279已结束挑战；14日原10只确认/结算继续原入口。


## 2026-10-13：更早价型变量与单一固定假设

[更早信号日价型资格与固定假设](early_signal_ohlc_20261013T1303Z/更早信号日价型资格与固定假设.md)按原V279清单SHA匹配V16完整9063条信号价，与原响应附层构成完整9245条临时诊断。9199价型可观察、46矛盾未知；严格原时点仍0。最终DELIVERY中的provisional_one_price_session只为价型变量，原始身份/角色标签资格不补造。

独立protocol_SIGNAL_PRICE_SHAPE_BEFORE_ANY_FIT.json预登记单一SIGNAL_PRICE_SHAPE_REPLACEMENT，下一新30分钟批次实现并只拟合原三个段各一次；其他28字段/原成员/标签/252窗/每日权重/Logistic参数和严格0.5保持，BASE不重拟合。各段Accuracy+3pp与Recall/Precision/F1/覆盖及非单一价格门预先固定；即使诊断通过，原120日前向与严格来源/角色/OOF仍保留。本批拟合0、新成熟0，不重跑原V27915字段或代理551NaN挑战，不改14日原预测。


## 2026-10-13：信号日价型固定尝试已退出

[信号日价型固定尝试与失败修复](price_shape_compare_20261013T132007Z/信号日价型固定尝试与失败修复.md)沿唯一预登记新增隔离relay_price_shape_research.py prepare/compare。本次已实际拟合2次，首段1584只Accuracy64.71→65.21%未过3pp、非单一价格Recall5.21→3.08%；第二段18条双方NaN目标的比较误报发生在拟合后，未保存预测不可认证，不重拟合。9月未拟合且本次尝试不再补，不能称完整三段已结算。第一段固定门已失败，尝试退出、无部署权重。

共享SIGNAL_PRICE_SHAPE_ATTEMPT_STARTED保留，重复compare实际拟合前拒绝。源/测试前像与原fit收据不改；当前代码仅修NaN目标比较、拟合前预检和未来预测优先保存。8项相关检查、3707条输入目标无拟合核验和首段3168配对行独立复算通过均非预测改善。新成熟0，原14日10只及39/38/28不变。下一步原14日固定观察确认与完整结算，无新证据不重开数值方案。


## 2026-10-13：历史研究不等待未来；月度近期加权固定比较

[历史数据与情绪变化固定比较](recency_walkforward_20261013T134351Z/历史数据与情绪变化固定比较.md)按最新用户纠正，实际复用原2023—2026数据进行15个月按时间推进的训练比较。新隔离relay_recency_walkforward.py prepare/compare原29字段和每月同252日成员；ROLL252每月原权重基准、HL63/HL126预先固定时间衰减，总权重252保持。完整45实际fit（公平滚动基准15、挑战30），原生产BASE重拟合0；三段Accuracy65.59/65.59/65.40%、72.49/72.59/72.49%、61.65/59.40/60.90%，近期方案全失败不重跑。目标与知识时点只到该月之前，67未知保持全原分母；旧历史不称新盲测。

完整11121行/15窗/原冻结边界/六份共同校正块区间独立重算及8项针对核验通过，非准确率提升。当前全尝试锁定，compare重复应拒绝；原模型/14日预测、角色/OOF/120日生产门不改。历史下一步用已保存逐月全池、同板位及原情绪阶段对照定位条件变化，distinct证据后另批预登记，不能把未来验收门扩大为禁止历史研究。


## 2026-10-13：同板位情绪误差与信号阶段附层

[同板位情绪误差与固定下一假设](conditional_error_20261013T163637Z/同板位情绪误差与固定下一假设.md)新增只读relay_conditional_context.py --plan <事前诊断计划> --output-dir <唯一新诊断目录>，完整3707原月度基准及15训练窗的63/189同板位/阶段/proxy条件率分解，未知/小组/不重合质量完整保留。source_phase_context附原市场算法阶段：9063与原relay一致，182九月signal-day补层；不改原relayUNKNOWN或29字段，不当角色真值或原时点来源。

最终WITH_SOURCE_PHASE产物来源已核，216差值、680错误组和5626条件记录独立复算，8针对核验通过非预测提升。phase_response_20261013T1649Z/protocol_BEFORE_ANY_FIT.json预登记单一阶段代理响应（原29+14固定阶段项），下一新30分钟批次15挑战fit、原ROLL252不重拟合；原各段Accuracy+3pp、完整Recall/Precision/F1/覆盖/非一字与共同区间门保持，不重跑旧失败。此批fit0/新成熟0，旧14日原预测保持。


## 2026-10-13：固定阶段代理响应已拟合并淘汰

[历史阶段响应固定比较与结果](phase_response_20261013T1649Z/历史阶段响应固定比较与结果.md)已完成relay_phase_response_research.py prepare/compare唯一试验，15个月各实际挑战fit一次、复用ROLL252基准未重拟合。原29+14固定阶段项、原同252日成员/标签/每日权重/Logistic参数与严格0.5保持；三个段Accuracy基准/挑战65.59/65.21%、72.49/71.76%、61.65/61.65%，9月真实非一字Recall仍15%。整体淘汰，compare共享根锁保持，不改阶段字典/参数重跑。

15fit/预测先保存/检查点/保存加载齐全；7414完整配对行、3707候选和67未知、全部基准字段分数、日期成员权重及分层/三份固定区间独立复算与8项针对检查通过非预测提升。182同信号市场阶段只作当前诊断附层，原relayUNKNOWN/冻结不改。新成熟0/无部署权重；历史下一步先核同板位/来源阶段/proxy单元的逐股价量及完整竞争者资料的来源和区分能力，不根据失败连续换算法，原14日结算独立继续。


2026-10-13按用户“模型难道就升级不了了吗”完成[历史个股与竞争者证据审查](research/model_upgrade_validation/peer_discrimination_20261013T170021Z/历史个股与竞争者证据审查.md)。原9245条/2455原同板位池一致；15原月初前252日窗分189/63，固定12字段，1080汇总/54672条件单元全保留，独立排序搜索复算通过，非Accuracy改善或15独立验证。旧2113封板时钟/炸板值均日线派生0，不认证真实盘口；相对强度为涨幅/10、连续数为旧低换手代理。市值及同身位市值差方向关联一致但未证原29字段以外增益，不选阈值/换算法。原9月市值缺失；既存V275源182/182市值为正，板位/价格/金额/换手单位匹配与原池源SHA核验通过，另存无目标附层。跨期日线流通股数推导与供应商市值口径/原时点资格仍未合格，不倒填或改原字段。此批拟合0/新成熟0/无部署，原14日10只与正式39/38/28哈希保持；平台读取paused未改，提升未验收。唯一历史下一步已存流通股本/市值跨期口径与完整覆盖及新增信息审查，证据合格后才另批固定挑战；旧已结束试验不重跑，原14日生产独立继续。


## 2026-10-14：深入复盘与完整同位输出

[当日深入复盘与模型输出经验](../../results/deep_review_20261014T115540Z/10月14日深入复盘与模型输出经验.md)复用原完整10只补充来源核对、54主板收盘/炸板、11跌停原行和15完整候选；正式V293来源门未过不追加正式标签。BASE/独立Accuracy80/62.5%、Recall50/75%，独立覆盖80%；真实非一字2正例模型全漏，独立抓百洋但3误报。原理由、醋化/龙版弃判及原7冰雪完整保留。

新增relay_cohort_output.py，已接入现有relay_choice_reasoning.py project/render/build，同位完整排序与原正负方向、非一字相对/正向首选及支持缺口分开；同位三路径保留，不拟合ranker。沿原CLI参数正常使用，源绑定增加该模块。原10真实源/模型/逐项贡献重建，旧rows/分数/方向/rank相同；旧源码前像与choice_output_source_compatibility.json精确绑定，不宣称当前源码等于旧源码。当前14→15完整15只替代来源补充分数也已实际接入，星帅尔/路畅复牌缺口与正式来源未过仍显示。5新增+10相关核验及3707保存预测/原准备资料994组独立重算通过，非Accuracy提升；正式新成熟0/拟合0。下一未冻结日期独立审阅后调用有效模型及该输出，到期全池检验；旧失败保持。


## 2026-10-14：九阳—百洋情绪补涨关系补正

[补涨关系补正](../../results/deep_review_20261014T115540Z/jiuyang_baiyang_relation_correction_20261014T121615Z/九阳与百洋补涨关系补正.md)按用户线索有限核2篇股吧/雪球原文，另1页正文不可得排除；确认限定羊/阳/洋与九阳/百洋叙事，作者正文与评论分栏。原时序9日九阳5/百洋1、12日6/2、13日九阳断板/百洋3、14日九阳未续板/百洋4一价。当前解释补为九阳周期低位情绪补涨候选，资金因果/唯一父龙/角色认证/可买性不自动确认。新原文均本次取得不倒填，原偏晋级百洋的理由未曾识别该关系据实保留。原当前补充分数、作者方向和完整池未改；角色附页已实际挂接15日百洋输出，拟合0/正式新成熟0。


## 2026-10-14：股吧预读先于独立个股判断

[执行说明](../../results/forum_preread_workflow_20261014T122229Z/执行说明.md)按用户要求新增stock_forum_preread.py，消费与stock_catalyst_context兼容的已核bank和完整候选，优先读已知作者当前正文/相关讨论，输出完整题材预读卡，再实际核公告/行情、独立主线/逐股与角色判断，留档后展示模型。正文/评论归属和时间分别记录，旧文只作背景，未覆盖保留未知。

```powershell
D:/python/python.exe -X utf8 stock_forum_preread.py --pool <完整当期候选JSON> --bank <已核当期bankJSON> --calendar data/v298/exchange_calendar_extension_20261003_v2/calendar.json --output-dir <唯一新目录> --model-exposure MODEL_NOT_REVIEWED_SELF_ATTESTED
```

已看模型时传MODEL_OUTPUT_ALREADY_SEEN，不认证盲测；同一bank继续供leader_role_review.prepare及typed narrative审阅，不替代原freeze/正式来源/11类/OOF门。本批完整15实际调用与6相关核验通过，2篇旧正文/评论形成4背景/11未覆盖、当日正文0，失败/初版摘录及源码前像保留。原预测/分数/登记不变，拟合0/正式新成熟0。


## 2026-10-14：信号日曲率试验与客观审查

[曲率试验与项目客观审查](signal_curvature_20261014T164603Z/曲率试验与项目客观审查.md)记录唯一SIGNAL_TURNOVER_AMPLITUDE_CURVATURE。relay_curvature_research.py prepare/compare以原固定训练资料追加两个平方项；现有协议及一次锁已经尝试，禁止再次比较或换目录重跑。实际挑战拟合2，BASE0；第一段Accuracy+0.19pp且Precision下降，第二段已存预测只读恢复Accuracy+0.05pp，配对区间均跨0，整体淘汰。

原NaN目标日期比较失败/预登记/源码精确前像保留；工程修复没有重拟合或继续9月。独立audit_saved_results.py核两段原保存预测及完整3707/7414行，9月182挑战保留NOT_FITTED_UNKNOWN；它是只读收尾，不是新盲测、9月效果或生产验收。新成熟0，原分数/登记/定时不改。源覆盖审查和同池全断板基准、全部74方向改变及完整指标在同目录。


## 2026-10-14：市值与真实封板来源、日VWAP覆盖修复

[本轮研究结论](cap_followup_20261014T170949Z/继续研究：市值与封板证据结论.md)包含全9245市值冗余与来源核对、V275217及其26竞价邻近秒值的资格区别，原始全池和所有未知保留。新增只读命令：

```powershell
python relay_source_clock_context.py --archive data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv --output <新的只读附层目录>
```

输出结构时间与旧契约接受状态分栏，不变更旧归档、实际一字、训练或分数。已实际应用全1046行及4针对核验。

随后依据旧V294共同1913数值一致而新增973覆盖的独立证据，relay_vwap_source_repair_research.py在固定vwap_source_repair_20261014T172716Z根一次冻结并完成3fit：Accuracy差−0.19/−0.42/−0.75pp，9月真实非一字Recall15%不变，全3707/7414及67未知保留；方案淘汰、一次锁已消费，compare不得重跑。源使用政策显式为当前知识截止事后诊断、九月×100单位推断，非原严格来源通过或新盲测。独立复核脚本和全部20方向改变见新数值目录，3预处理核验非预测提升；源审查0fit/数值3/BASE0/真实新增0，无生产改动。

## 2026-10-14：跨日补涨关系与实际审阅入口

[实际完整15只及边界](supplement_recognition_20261014T174129Z/补涨识别接入与预测改进结论.md)。原leader_role_review.prepare没有调用预读；新日期使用新的加法式入口，原代码/冻结兼容保留：

```powershell
python supplement_role_review.py prepare --features <合格原特征归档目录> --theme-report <完整题材事实> --mainline-report <独立主线> --compatibility <原兼容策略> --pool <同日期完整遮分连板池> --bank <原格式已核催化论坛bank> --relation-bank <同日期父锚关系bank> --calendar <官方calendar.json> --output-dir <新的目录> --model-exposure MODEL_NOT_REVIEWED_SELF_ATTESTED
python supplement_role_review.py freeze --facts <facts_with_supplement.json> --judgements <实际完成的author_draft.json> --output <新登记文件>
python supplement_role_review.py compare --record <该登记文件> --baseline <原同池模型输出> --outcome-archive <原兼容结果归档> --output <新结算文件>
```

真实已看模型必须传MODEL_OUTPUT_ALREADY_SEEN。prepare顺序为完整池→stock_forum_preread→supplement_relation_review→原leader_role_review；模板不填晋级方向或角色。每股新增forum_preread_assessment及supplement_assessment；全部同位、同父锚候选、每条relation_id及ACCEPT/REJECT/PENDING理由必须审阅，采纳还要写反证/取消条件。缺资料不自动看空或转正。新freeze随后执行原11角色、题材、催化、独立登记和时点门，不将跨行业假设当正式SUPPLEMENT标签。

关系bank规范为supplement-relation-bank-1：sources/claims精确正文偏移与SHA、真实取得/复核时间及作者/评论归属；anchors与预测池分开且有代码名称来源；edges支持多个父锚或父锚未知；temporal_facts的OHLC/板位来源分开，原不复权raw必须一致；显式STREAK_BREAK/PRIOR_FIRST_OBSERVED_BOARD事件及官方calendar、完整source_bindings。无当前父锚或不同时序的替代锚保持未知，NAME_MATCH_ONLY不能升为支持候选。多个父锚周期未限定来源范围则不任选有利节点。既存实例qualified_relation_bank_v2.json可供结构参考，不能用于其他日期。

已见旧日期、正式特征来源尚不合格时可实际补诊断：

```powershell
python supplement_role_review.py context --pool <完整遮分池> --bank <原催化bank> --relation-bank <同日期关系bank> --calendar <官方calendar.json> --output-dir <新诊断目录> --model-exposure MODEL_OUTPUT_ALREADY_SEEN
python supplement_role_review.py review-context --facts <facts_with_supplement.json> --judgements <逐股完成的诊断稿> --output <只读完成审阅文件>
```

context明确forecast_registration_allowed=false，review-context拒绝新增晋级方向，不能冻结或重写旧预测。本次actual_full15_release实跑15只、6关系并全部完成审阅；原两帖预读4背景，关系层另有阳光9月30日背景，总5股/10未知；当日关系正文0。22新增+原17相关测试通过非预测提升。full_pool_conflict_audit_release.json保持原10只成熟结算与15只新观察的全部分母、旧分数/方向及弃判。原SUPPLEMENT/C2真值门未过；prediction_validation_protocol.json唯一直接观察时序挑战尚未拟合，先完整历史输入资格，不按例手调分。原来源失败、v2/v3/first-final输出与源码精确前像保留。

## 2026-10-14：逐观点股吧推理进入独立预测

用户进一步要求每次给预测与预期时，对股吧讨论个股的具体内容做独立分析。旧预读只有摘录与“已读”标记，补涨逐关系门未覆盖一般个股论证。新增forum_reasoning_review.py，不改旧核心或模型权重；新日期改用以下入口：

```powershell
python forum_reasoning_review.py prepare --features <合格原特征归档目录> --theme-report <完整题材事实> --mainline-report <独立主线> --compatibility <原兼容策略> --pool <完整遮分连板池> --bank <原已核催化论坛bank> --relation-bank <同日期父锚关系bank> --calendar <官方calendar.json> --output-dir <新的目录> --model-exposure MODEL_NOT_REVIEWED_SELF_ATTESTED
python forum_reasoning_review.py freeze --facts <reasoning/facts_with_forum_reasoning.json> --judgements <实际完成的作者稿> --output <新独立登记>
python forum_reasoning_review.py compare --record <该登记> --baseline <原同池模型输出> --outcome-archive <原兼容目标结果> --output <新结算>
```

prepare先运行原supplement完整准备，再从同一事实卡预读及关系层取已核正文/评论，按source_id/section/offset/excerpt生成稳定opinion_id并去重。author_draft_PENDING.json新增forum_reasoning_assessment：每股reviewed/synthesis/agreement+reason；每观点作者命题、推理、可检验预期、作者实际call与原目标日、verification状态/范围/源引用、ACCEPT/REJECT/PENDING、反证、独立影响及取消条件。promotion_judgement仍由作者依完整市场和竞争独立填写，不自动赋值。UNKNOWN假设可明确采纳，不需等监督角色标签；PENDING观点不能自动提供确定方向，但不会禁止该股凭其他证据形成预测。

freeze拒绝只填“已读”、漏观点/股票、把问句改成明确预测、旧明天移成当前日期、无来源认证事实、用行情认证公司业务或资金因果；随后继续原supplement/角色/催化/登记门。正文、评论时间与来源最晚available时刻保留，真实已看模型必须MODEL_OUTPUT_ALREADY_SEEN。compare沿原完整同池成熟结果结算，独立结论先于新日期模型展示；不强迫先看BASE再填推理。

已见旧日期附层：

```powershell
python forum_reasoning_review.py attach --facts <原facts_with_supplement.json> --draft <原逐股诊断完成稿> --output-dir <新附层>
python forum_reasoning_review.py review-context --facts <facts_with_forum_reasoning.json> --judgements <逐观点完成稿> --output <完成审阅>
```

本次forum_reasoning_20261014/actual_full15已实际附层完整15只、10去重观点并全部完成，10未覆盖股票保留，当日正文0、已见模型不称盲审、原方向不改。 [实际逐观点推理](forum_reasoning_20261014/股吧逐观点推理与独立预期.md)及[执行边界](forum_reasoning_20261014/执行边界.md)说明百洋相对表现、慈文换手后确认、醋化行业谨慎等如何分别影响独立预期；14新增针对测试通过非预测改善。新方向0/fit0/正式新成熟0，原监督角色/C2门仅约束相应训练，不禁止直接独立判断。新日期完整预测/确认/结算持续检验哪些作者推理被正确采纳、哪些导致误判，不靠后验改口提高准确率。

## 2026-10-14：项目不足与逐观点结果附页

[本次客观审查](project_audit_20261014/项目不足与优先改进.md)：非一字漏判、少日期前向、盘口/题材历史覆盖、收益与成交验收仍是主要不足。新入口CLI未完整暴露原history-dataset/mainline-history/narrative-reviews；Python API仍可传history_datasets/mainline_history/narrative_reviews，本轮参数缺口未修，调用时不得把缺传当无历史。审阅层多而重复，统一入口/共享证据和条件引用是后续工程重点，旧冻结源码保护不应删除。

新增只读反馈：

```powershell
python forum_reasoning_outcome_context.py project --record <原正式登记的独立论坛判断> --comparison <原同池compare输出> --output-dir <新反馈目录>
python forum_reasoning_outcome_context.py pending --facts <诊断facts_with_forum_reasoning.json> --review <完成的观点审阅> --output-dir <新待结算目录>
```

project核原登记、exact事实/日期/完整池，读取原比较绑定的唯一BASE和目标来源归档，经原兼容build_outcomes只读重建真值，核方向、指标和分层后显示每观点方向兼容与误报/漏判/弃判。角色或中性观点不以晋级认证；没有独立同刻确认的条件与取消效果均UNKNOWN。原结果/预测不改、不请求数据、不拟合；原comparison未重复列出的BASE source_hashes在本页另验证并追加，不删除旧校验。

实际project_audit_20261014/pending_full15已跑15股/10观点；新方向、标签、指标仍待核，误差NOT_EVALUATED而非15只弃判。15项合成契约测试通过非预测增益，真实project新到期结算尚未执行；0fit/0新正式成熟，完整旧分数/来源保持。不用旧10结果配新15观点，不把方向兼容当论坛论点真值或因果收益。
