# GitHub经验与实现筛选（本批只审阅，不升级模型）

实际形成时间：2026-10-02T20:25:01.909112+00:00；北京时间对应2026-10-03。

筛查8个候选仓库，7个可读取，1个API返回404停止。已保留全部请求原文、真实获取时间、返回URL、状态和SHA-256，并绑定实际commit。GitHub源码只证明实现可审查，未测试任何行情接口、未执行外部代码，未新增真实前向样本，未发现可直接认证本项目准确率提高的证据。

本批值得写入的是6项事实记录/未来可检验命题。它们多数与本地既有研究方向有关，应去重补证，不能新增权重、改模型前5或把外部阈值搬进生产。

## 根代理去重后优先采用的细节

本地 V248/V273 已有周期、弱转强、卡位/补涨、分歧承接，AGENTS 已有来源/未知/全候选/前向约束。下列通用经验不作为新因子：分板位晋级反馈（G01）、高位/中位/首板反馈（G03）、梯队完整度（G04）。它们仅补源码、公式与反例。

真正可补充的3项细节是：G02 按题材统计支持先剔除本股并保留宽标签分母；G05 同日/前交易日统计字段映射，NULL 与0分开；G06 触板与封板事件语义及分辨率/代理来源。另将今日题材覆盖昨日、接口失败写0并覆盖旧值列入具体不可搬实现，不新增生产阈值。

## G01 按原登记板位、市场环境和一字/非一字分别记录近期晋级反馈，而不是一句连板越高越好

**处理：** 可写入研究资料库；统计口径仅作未来固定检验候选，现行指标不更换

**原逻辑与适用边界：** plank 近 30 自然日遍历每日板池，计算 1→2 到 5→6 每日晋级比再等权平均；easy-stock 昨日 2+ 板为晋级分母，逐个匹配今日板位增加。 主板非 ST 原全体 2+ 候选；一字与非一字分层；不混首板胜率/交易收益

**可以记录：** original_roster_code、signal_streak、is_one_word、n_registered_by_tier、n_scored_by_tier、n_promoted_by_tier、coverage_by_tier、lagged_regime。

**时点：** 在信号日 T 盘后只使用已结算的 T-1→T；T→目标日结算后才能增加。近期20/60日沿用官方真实交易日和原登记覆盖分母。

**验证方式：** 后续独立预登记比较基准与仅加滞后分层反馈的挑战方案；报告按日期配对 Top1 差值、完整候选 Precision/Recall/F1 和覆盖。当前固定 C1/C2 不增方案。

**反例：** 高板样本少且一字买不到，日平均会被 1/1 高度样本放大；缺某交易日数据仍接前一天会把跨缺日误认为连板；仓库只维护昨日一到五板；昨日六板后续未保留到 next yesterdaySix，七板可退回首板类别。。

**不搬入：** 30 自然日窗口；日晋级率等权平均替代原固定 pooled 口径；三板定龙头的概率主张。

**证据：** B_CODE_INSPECTED_NOT_EMPIRICALLY_VALIDATED；本项目准确率增量未知。概念并非新因子；新增可审查的公式和错误清单，不把熟悉经验包装为独立发现。

来源：[固定commit源码](https://github.com/SingleDogCherry/plank/blob/e9450a938aea96d090df978d12082cea7e7b21d3/src/main/java/com/mistra/plank/job/AnalyzeProcessor.java#L74-L224)，[固定commit源码](https://github.com/jundizhou/easy-stock/blob/37a8c1bb3e2e9cabeb11f94dbef50a8b955f1948/backend/internal/httpapi/market_emotion.go#L505-L794)

## G02 验证题材支持时先剔除候选自身，再看其他首板、中位板、高板；宽泛公共标签单列

**处理：** 优先写入复盘事实字段候选和证据核验；与已授权同身位/题材方案去重

**原逻辑与适用边界：** easy-stock peerCount=item.count−1，连板数和高度能量剔除本股，peerMaxStreak 在本股板位扣一后找其他最高板；另对宽泛题材按概念成员占全域比例衰减。B6 以当日涨停家数最多的概念作代表题材。 已在信号日取得、真实形成且有日期来源的事实题材；同板位竞争另列；不能用当前分类倒填历史。

**可以记录：** asof_theme_source、theme_evidence_timestamp、peer_first_board_count、peer_mid_board_count、peer_higher_count、same_height_peer_count、peer_max_height_excluding_self、theme_member_scope、theme_missingness。

**时点：** T 收盘后在原知识截止前取得并留档 T 题材和全候选；开盘后来证据只写追加确认，不改旧题材。

**验证方式：** 此内容与当前 C2 的主题/同身位特征重合，作为来源与反例补充；新增宽标签或第一板支持字段须另批新假设、固定比较，不改当前协议。

**反例：** 只有本股一个涨停也被叫题材龙头；国企、融资融券、昨日涨停等公共标签把无关联股连接；一股多标签挑次日胜出的主线形成 hindsight。

**不搬入：** peerCount/peerBoard/energy 人工权重 2/7/2 等；题材成员比例 ≥0.2 时乘0.78、≥0.1乘0.88；最高板布尔直接命名为真实龙头。

**证据：** B_CODE_INSPECTED_NOT_EMPIRICALLY_VALIDATED；本项目准确率增量未知。剔除本股的具体实现、宽标签处理与题材置信度不等于概率，是当前主题工作可补充的核验细节。

来源：[固定commit源码](https://github.com/jundizhou/easy-stock/blob/37a8c1bb3e2e9cabeb11f94dbef50a8b955f1948/backend/internal/httpapi/concept_attribution.go#L132-L190)，[固定commit源码](https://github.com/quantskills/skill-b6-limitup-pool/blob/54568d9aeec73057e9f9ecd6f35870bc5b42d97a/%E5%BC%80%E5%8F%91%E4%BA%A7%E7%89%A9/scripts/build.py#L455-L495)

## G03 市场强不强分开看昨日高位、中位和首板反馈，尤其记录高位亏损是否与涨停扩散背离

**处理：** 值得写入未来假设库；先连续原始量记录，再用固定检验判断价值

**原逻辑与适用边界：** easy-stock 按昨日占用板位由高到低，前三个不同板位列 high，余下2+板 mid，1板 low；分别记录样本/有行情/晋级/负收益/明显负反馈；高度坍缩=max(昨高−今高,0)。 主板原候选的昨日反馈；它是盘面事实/待检验环境交互，不是某股的实际晋级概率。

**可以记录：** previous_roster_high_levels、high_sample_count、high_quote_count、high_mean_return、high_negative_count、high_advance_count、mid_negative_count、first_board_feedback、height_change、limit_up_breadth、quote_coverage。

**时点：** T 盘后已结算 T-1→T 可作为 T 的环境输入；目标日09:25/09:35即时反馈只能作为新时间点确认，不能送入T盘后预测。

**验证方式：** 未来独立试验只检验滞后高位反馈与非一字 Top1 的交互；固定分期分环境结果与全体指标，不挑高位杀跌日后报告。

**反例：** 昨日高位新旧周期交错，市场涨停总数扩散但昨日高位仍亏损；高位行情缺失导致均值只剩晋级存活者；原仓库缺行情股票不入return均值，完整样本数必须另外保留。。

**不搬入：** 前三个占用板位自动当真正情绪核心；涨跌幅 ≤−3 为 weak、≤−6 severe 的固定阈值；高风险分55、下跌比0.75等条件直接变成用户模型门槛。

**证据：** B_CODE_INSPECTED_NOT_EMPIRICALLY_VALIDATED；本项目准确率增量未知。同一时间轴区分 breadth 与昨日高位反馈，避免把全市场热度直接当龙头接力环境。

来源：[固定commit源码](https://github.com/jundizhou/easy-stock/blob/37a8c1bb3e2e9cabeb11f94dbef50a8b955f1948/backend/internal/httpapi/market_emotion.go#L505-L794)

## G04 最高连板数之外记录梯队缺口；孤立高度与完整梯队应当分别解释

**处理：** 值得作复盘描述字段和未来假设；当前已有梯队假设先去重

**原逻辑与适用边界：** easy-stock ladderContinuity=占用的2+不同板位数/(最高高度−1)；leaderGap=最高两只股票板位差；KPL 代码把题材里 TDType=0 反包股单列，不计入连续梯队。 同一主板非 ST 范围下的全市场和已核验事实题材梯队分别算；不是证券交易所全 A 认证。

**可以记录：** mainboard_max_streak、occupied_continuous_tiers、missing_tiers、highest_two_stock_gap、theme_occupied_tiers、continuous_vs_n_days_m_boards、rebound_reseal_status、roster_scope_verified。

**时点：** T 收盘后的源绑定完整板池；所有失败/未知/源日期不符必须保留。

**验证方式：** 仅在原协议外未来批次预登记结构因素，不与已有 C2 主题梯队特征重复计分；分高位存活/孤立/多并列描述时保持全候选与目标日一致。

**反例：** 最高高度大所以占用比例小，但旧龙独立存活而新题材刚起步；二板三板齐全可能是多个无关联题材拼出的假完整梯队；连续2板与5天3板混淆。。

**不搬入：** 结构评分权重；孤立最高板必然失败/唯一活口必然晋级；反包Tips里的几天几板直接当连续连板。

**证据：** B_CODE_INSPECTED_NOT_EMPIRICALLY_VALIDATED；本项目准确率增量未知。完整梯队的具体分母、跨题材拼接反例及连续/反包分离。

来源：[固定commit源码](https://github.com/jundizhou/easy-stock/blob/37a8c1bb3e2e9cabeb11f94dbef50a8b955f1948/backend/internal/httpapi/market_emotion.go#L505-L794)，[固定commit源码](https://github.com/yueshushu0207/kaipanla/blob/2c41d2b6d153b3f1cb86351a21f851d491322f32/kaipanla_crawler.py#L718-L870)

## G05 竞价强弱用有真实目标日时间和分母的相对量记录，盘后预测与开盘确认分开

**处理：** 可写入未来09:25观察记录字段；阈值与可用性未验证，不动盘后评分

**原逻辑与适用边界：** eltdx 在09:25前拒绝短线指标；target/previous/stats_date 对齐，同日统计取 prev_*，前交易日统计取该日 amount/seal/open 字段；竞价成交额/昨成交额，竞价量/昨竞价量，封单/自由流通市值或昨封单；分母0/缺失返回None。 目标日09:25后带实时时间记录，不是T盘后已有输入；五档快照不能证明L2排队或撤单序列。

**可以记录：** captured_at_real、provider_quote_time、target_trade_date、previous_trade_date、stats_date、alignment_status、auction_amount、previous_amount、auction_volume、previous_auction_volume、seal_amount、ratio_validity、five_level_snapshot_time、executable_proof_status。

**时点：** 目标日09:25完成后；09:35追加独立状态；真正数据及留档形成时间必须不晚于对应确认时点。

**验证方式：** 先影子记录采集覆盖/日期对齐，不优化买点；后续若研究开盘提升应另立09:25决策任务与标签，不能与盘后准确率混为一个指标。

**反例：** 服务器休市旧日期、统计缓存已错日；竞价量相对昨封单高可能是兑现而非吸筹；五档买一封单大不代表买入排队可成交。。

**不搬入：** 0.0051价格容差；按代码/名字推固定涨跌幅补涨停价；按封单额排序替代已验收模型前5；竞价量/近5日每分钟均量可直接算真实强弱概率。

**证据：** B_CODE_INSPECTED_NOT_EMPIRICALLY_VALIDATED；本项目准确率增量未知。同日/前日统计字段映射和None语义可直接补充未来来源核验细节。

来源：[固定commit源码](https://github.com/electkismet/eltdx/blob/d19ac86f2bae7565660a89baf94a3fcc531fac93/docs/helpers/%E8%BF%9E%E6%9D%BF%E5%A4%A9%E6%A2%AF.md#L77-L109)，[固定commit源码](https://github.com/electkismet/eltdx/blob/d19ac86f2bae7565660a89baf94a3fcc531fac93/src/eltdx/helpers/shortline.py#L435-L465)

## G06 把触板、持续封板、开板、回封和可成交分别留成事件；缺分钟只能保持日线代理身份

**处理：** 值得写入未来采集字段和反例；仓库实现存在明显口径缺陷，不直接采用

**原逻辑与适用边界：** KPL 报告分涨停股、炸板、连板梯队和大幅回撤；B6 分钟布尔上升/下降沿统计回封/炸板，并保留 minute/daily_proxy 来源类型；但布尔是 minute.high≥upper，表示触板而非真实封板。 未来分钟/Tick来源实测后记录路径；缺失不记成功、不记0次开板、不编可成交收益；日线OHLC本身不能还原全天开封次数。

**可以记录：** event_timestamp、event_kind、touch_time、seal_confirm_time、open_board_time、reseal_time、event_resolution、source_kind、unknown_intraminute_events、coverage、limit_price_explicit、fill_or_queue_evidence。

**时点：** 当日实时报价与历史路径都标真实取得时间；T盘后只能使用在截止前完整取得的T路径，未来目标日路径不得回填。

**验证方式：** 先记录能否区分touch与seal，数据源/分钟分辨率是否足够；不把代理事件当标签重跑已有失败模型。若能形成真实路径再独立预登记固定比较。

**反例：** 一分钟内high触板后收低，high布尔却标封死；一分钟内多次炸封计为一次或零次；最后触板发生15:00不能证明此前一直封住。。

**不搬入：** 分钟 high触板当 sealed；最后触板分钟当最终回封时间；日线无明显开板就 blow_up_count=0 当事实；固定涨幅/浮点容差兜底实际涨停价。

**证据：** B_CODE_INSPECTED_NOT_EMPIRICALLY_VALIDATED；本项目准确率增量未知。事件语义和分辨率影响经验可验证性；触板/封死混淆是应写入来源审查的具体问题。

来源：[固定commit源码](https://github.com/HJ2916/kpl-dashboard-public/blob/b2ca64099bbe6e8dbea7b353dbca82f0f8aa6591/src/gen_limit_up_report.py#L32-L186)，[固定commit源码](https://github.com/quantskills/skill-b6-limitup-pool/blob/54568d9aeec73057e9f9ecd6f35870bc5b42d97a/%E5%BC%80%E5%8F%91%E4%BA%A7%E7%89%A9/scripts/build.py#L374-L451)

## 不能直接采用的源码细节

- alignConsecutiveStockThemes 默认以当前日权威题材为authority、昨日股为target，可把今日主炒线索和确认后的连续晋级反写昨日；展示一致性可用，历史模型输入不可用。 [对应源码](https://github.com/jundizhou/easy-stock/blob/37a8c1bb3e2e9cabeb11f94dbef50a8b955f1948/backend/internal/httpapi/concept_attribution.go#L251-L302)
- minute.high触及板价当sealed，触板路径不能作真实封板事件。 [对应源码](https://github.com/quantskills/skill-b6-limitup-pool/blob/54568d9aeec73057e9f9ecd6f35870bc5b42d97a/%E5%BC%80%E5%8F%91%E4%BA%A7%E7%89%A9/scripts/build.py#L374-L451)
- in_date未知被保留，且源码把API成员date重命名in_date；字段语义未独立核验不能认证PIT。 [对应源码](https://github.com/quantskills/skill-b6-limitup-pool/blob/54568d9aeec73057e9f9ecd6f35870bc5b42d97a/%E5%BC%80%E5%8F%91%E4%BA%A7%E7%89%A9/scripts/build.py#L455-L495)
- 板块强度请求失败和无返回均写0.0，后续INSERT OR REPLACE 会将缺失伪装成强度为0；不用于本项目事实/历史回填。 [对应源码](https://github.com/HJ2916/kpl-dashboard-public/blob/b2ca64099bbe6e8dbea7b353dbca82f0f8aa6591/scripts/kpl_data_collector.py#L151-L161)
- 代码作者注释Mistra/Github MistraR、README说明源于其他仓库；SingleDogCherry为可读发布源，原作者与完整改动链未重建，不声称其本人验证经验。 [对应源码](https://github.com/SingleDogCherry/plank/blob/e9450a938aea96d090df978d12082cea7e7b21d3/src/main/java/com/mistra/plank/job/AnalyzeProcessor.java#L74-L224)
- 仓库把请求日期回显成date、失败可返空池，未见本次真实API响应核验；不能据此认定历史请求日期有效或空池等于真实无连板。 [对应源码](https://github.com/yueshushu0207/kaipanla/blob/2c41d2b6d153b3f1cb86351a21f851d491322f32/kaipanla_crawler.py#L718-L870)

## 全部候选

|仓库|实际commit|本批判定|
|---|---|---|
|[SingleDogCherry/plank](https://github.com/SingleDogCherry/plank)|e9450a938aea96d090df978d12082cea7e7b21d3|ADAPTED_CODE_ORIGIN_NOT_FULLY_RESOLVED_NO_WINRATE_CLAIM|
|[electkismet/eltdx](https://github.com/electkismet/eltdx)|d19ac86f2bae7565660a89baf94a3fcc531fac93|SOURCE_IMPLEMENTATION_REFERENCE_ONLY|
|[HJ2916/kpl-dashboard-public](https://github.com/HJ2916/kpl-dashboard-public)|b2ca64099bbe6e8dbea7b353dbca82f0f8aa6591|SOURCE_IMPLEMENTATION_REFERENCE_ONLY|
|[renjiegod/doyoutrade](https://github.com/renjiegod/doyoutrade)|None|UNAVAILABLE_404_STOP|
|[yueshushu0207/kaipanla](https://github.com/yueshushu0207/kaipanla)|2c41d2b6d153b3f1cb86351a21f851d491322f32|SOURCE_IMPLEMENTATION_REFERENCE_ONLY|
|[jundizhou/easy-stock](https://github.com/jundizhou/easy-stock)|37a8c1bb3e2e9cabeb11f94dbef50a8b955f1948|SOURCE_IMPLEMENTATION_REFERENCE_ONLY|
|[quantskills/skill-b6-limitup-pool](https://github.com/quantskills/skill-b6-limitup-pool)|54568d9aeec73057e9f9ecd6f35870bc5b42d97a|SOURCE_IMPLEMENTATION_REFERENCE_ONLY|
|[LeeJUNhao791/ljh-a-stock-data](https://github.com/LeeJUNhao791/ljh-a-stock-data)|f90d67853b8108f13d286e1df20b357e2c5198a9|README_ROUTE_REFERENCE_ONLY_NO_RUNTIME_ORIGIN_VALIDATION|

## 留档与结束

核验62个公开响应/解码源码文件哈希，全部匹配；20份实际源码文件已留档，重点统计/归因/行情对齐/事件逻辑已实际审阅，20次初次失败（19个raw host reset/timeout及1个404）保留；19个raw host缺口各只使用一次已成功metadata的git-blob API补采，不再重试。

当前模型、原预测、原评分、AGENTS、目标提示词和已有固定挑战方案均未修改。真实前向样本新增0，Accuracy/Precision/Recall/F1/Brier本批变化不可计算。

唯一下一步：与本地原规则去重，优先补充题材剔除本股、昨日高位负反馈以及竞价日期/分母核验字段；能够被真实时点证据支持后才进入下一次独立预登记的固定比较。
