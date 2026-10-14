# 隔日复盘目标证据核对 V285

核对时间：2026-10-02T17:47:27.296655+08:00。
完整目标未完成。本次从逐条预测重算，未拟合模型或修改冻结答案。

| 对照时期 | 真实非一字排名首选晋级 | 正向判断精确率 | 实际晋级召回 | 完整日期/全部日期 |
|---|---:|---:|---:|---:|
| 2025H2 | 67/126（53.17%） | 29/56（51.79%） | 6.87% | 126/126 |
| 2026YTD | 61/159（38.36%） | 30/74（40.54%） | 7.32% | 159/160 |
| SEPTEMBER | 7/13（53.85%） | 6/10（60.00%） | 15.00% | 13/16 |
| combined_pre_september | 128/285（44.91%） | 59/130（45.38%） | 7.09% | 285/286 |

排名首选指标逐日强制选最高研究分数，可能仍被二分类判失败；不等于每天都有正向首选或可交易主选。
128/285为V279按真实非一字验证重新选择月度模型的对照；138/285为V274旧一字代理选择模型的结果。模型选择口径也不同，不能将差异全部归因于一字定义。

市场方向：180/285（63.16%）；延续类识别4/57。

| 用户目标 | 当前核对结果 | 证据及缺口 |
|---|---|---|
| 稳定复盘流程 | IMPLEMENTED_WITH_PROCESS_EVIDENCE | V284 daily output and 151 process tests; these do not measure accuracy |
| 下一交易日市场预期稳定超过60% | UNPROVEN | Historical 180/285; reused dates, interval below 60%, weak CONTINUE recall, no completed current forward date |
| 连板下一板预测稳定超过60% | NOT_ACHIEVED | True nonword ranking and positive-call precision are below 60%; word-inclusive promotion is separate |
| 市场行情和短线情绪预期 | IMPLEMENTED_FORECAST_AWAITING_OUTCOME | V280 original market scenario and direction; indices remain separate targets |
| 超预期、符合预期、不及预期及预案 | CONTRACT_IMPLEMENTED_REAL_CONFIRMATION_PENDING | V271 AUCTION/OPEN0935/OPEN1000/CLOSE; real target-day auction and minute observations unavailable |
| 龙头身份判断及预测优化 | RULES_IMPLEMENTED_ADVANTAGE_UNPROVEN | V273 evidence transitions; theme/absorption facts and independent leader-identity outcomes missing |
| 股吧经验与A股玄学 | RESEARCHED_ADVANTAGE_UNPROVEN | V273 original-source hypothesis bank; symbol attention is not verified industry evidence or predictive uplift |

当前V271契约登记逐份核验内容哈希。10月8日尚未发生，不能产生其真实前向结算证据。
下一步需要真实的新交易日、完整候选与按原规则结算的独立记录；反复调整已查看的同一段历史不补足这项证据。
