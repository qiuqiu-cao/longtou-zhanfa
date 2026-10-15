# 完整个股审阅工作台

信号日 2026-10-15 → 2026-10-16；完整 6 只。
诊断附页：True；真实模型暴露：MODEL_NOT_REVIEWED_SELF_ATTESTED。
页面隐藏全部模型分数与原模型方向；作者审阅与未知均须实际填写。
原BASE分数未经校准；后续显示原分数时不能解释为晋级概率或可信胜率，不变换旧数值。

| 个股 | 板位 | 同位 | 讨论覆盖 | 父锚 |
|---|---:|---|---|---|
| 冰山冷热 000530 | 3 | 阳光乳业 001318 | NOT_COVERED_IN_CHECKED_SOURCES | 未知 |
| 阳光乳业 001318 | 3 | 冰山冷热 000530 | NOT_COVERED_IN_CHECKED_SOURCES | 九阳股份：OLD_FOLLOWER_RESTART_CANDIDATE、均瑶健康：RELATION_BACKGROUND_TIMING_UNRESOLVED |
| 慈文传媒 002343 | 5 | 百洋股份 002696 | BACKGROUND_ONLY | 父锚未知：SECTOR_DISCUSSION_PARENT_UNKNOWN |
| 百洋股份 002696 | 5 | 慈文传媒 002343 | BACKGROUND_ONLY | 九阳股份：EARLY_START_CONTINUATION_CANDIDATE |
| 新华文轩 601811 | 2 | 富春染织 605189 | NOT_COVERED_IN_CHECKED_SOURCES | 未知 |
| 富春染织 605189 | 2 | 新华文轩 601811 | NOT_COVERED_IN_CHECKED_SOURCES | 未知 |

## 冰山冷热 000530（3板）

同位：阳光乳业 001318
同父锚候选：无
日线价量：{"boards": 3, "height_gap": 2, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.6, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.41049599999999997, "signal_amount": 1697317801.0, "amplitude": 5.238095238095238, "turnover_vs_prev": 13.611061374714016, "price": 6.93, "price_same_height_percentile": 0.5, "signal_amount_same_height_percentile": 1.0, "turnover_same_height_percentile": 1.0, "turnover_vs_prev_same_height_percentile": 1.0, "signal_one_word": 0.0, "log_signal_amount": 21.252315078499546, "amount_rank_competition": 0.34657359027997264, "turnover_rank_competition": 0.34657359027997264}
真实一字身份：0.0
因果短线角色（规则候选）：TURNOVER_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002696／ANCHOR_HEAVY_TURNOVER
同位淘汰率：0.0；已观察分歧换手板数：2.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：UNKNOWN
本轮连续板核对：True
父锚假设：未知
同周期与价量对比：[]
历史名称／字辈扩散代理：冰山冷热／未知
扩展规则角色：TURNOVER_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

限定来源未覆盖；未点名不等于无催化，不自动看空或转正。

独立方向／采纳决定／反证／取消条件：待作者审阅。

## 阳光乳业 001318（3板）

同位：冰山冷热 000530
同父锚候选：百洋股份 002696
日线价量：{"boards": 3, "height_gap": 2, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.6, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.06332900000000001, "signal_amount": 300750817.0, "amplitude": 4.342190537913156, "turnover_vs_prev": 0.4446543044311663, "price": 16.97, "price_same_height_percentile": 1.0, "signal_amount_same_height_percentile": 0.5, "turnover_same_height_percentile": 0.5, "turnover_vs_prev_same_height_percentile": 0.5, "signal_one_word": 0.0, "log_signal_amount": 19.521792632682363, "amount_rank_competition": 0.0, "turnover_rank_competition": 0.0}
真实一字身份：0.0
因果短线角色（规则候选）：SUPPLEMENT_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002696／ANCHOR_HEAVY_TURNOVER
同位淘汰率：0.0；已观察分歧换手板数：3.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：OLD_FOLLOWER_RESTART
本轮连续板核对：True
父锚假设：襄阳轴承：RELATED_UNRESOLVED；九阳股份：OLD_FOLLOWER_RESTART；均瑶健康：RELATED_UNRESOLVED
同周期与价量对比：[{"code": "002696", "name": "百洋股份", "boards": 5, "signal_day": {"date": "2026-10-15", "status": "OBSERVED", "previous_official_session": "2026-10-14", "open": 1071, "close": 1073, "high": 1073, "low": 975, "volume_native": 747713.0, "one_price_session": false, "intraday_return": 0.0018674136321195078, "gap_vs_previous_official_close": 0.09846153846153838, "range_vs_previous_official_close": 0.10051282051282051, "volume_ratio_vs_previous_official_session": 22.465312622059308}, "price_path_is_role_truth": false}]
历史名称／字辈扩散代理：阳光乳业／未知
扩展规则角色：SUPPLEMENT_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

父锚假设：九阳股份；GROUP_ASSOCIATION
节点：{"state": "OLD_FOLLOWER_RESTART_AT_ANCHOR_BREAK", "current_streak_start": "2026-10-13", "current_streak_dates": ["2026-10-13", "2026-10-14", "2026-10-15"], "complete_streak_checked": true, "prior_first_observed_board": "2026-09-30", "prior_observation_is_lifetime_first": false, "anchor_first_observed_streak_break": "2026-10-13", "anchor_boards_at_current_start": null, "child_boards_at_break": 1, "missing": [], "timing_is_causal_proof": false}
替代解释：均瑶健康消费/Y背景锚；百洋4板名称线领先候选，并非已经证实阳光的新父龙；乳品/消费独立修复，尚未绑定全部上涨原因
削弱／取消条件：补涨关系当日正文和资金因果仍未核；次日相对二板全池弱化且消费无扩散；盘口回封稳定性与成交条件未核

父锚假设：均瑶健康；GROUP_ASSOCIATION
节点：{"state": "UNKNOWN", "current_streak_start": "2026-10-13", "current_streak_dates": ["2026-10-13", "2026-10-14", "2026-10-15"], "complete_streak_checked": true, "prior_first_observed_board": "2026-09-30", "prior_observation_is_lifetime_first": false, "anchor_first_observed_streak_break": null, "anchor_boards_at_current_start": null, "child_boards_at_break": null, "missing": ["explicit parent streak break; missing board is not a break"], "timing_is_causal_proof": false}
替代解释：
削弱／取消条件：需同口径父锚日线与节点验证，未核前不得排除或定为唯一父龙

讨论 opinion_83bf69c0ffa8a102e1b5769c：延边刺客／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-09-30T16:39:00+08:00；主帖：2026-09-30T16:39:00+08:00
取得：2026-10-14T15:34:22.933405+00:00；核阅后可用：2026-10-15T09:31:30.166860+00:00
> L51: 消费<br>L52: cite18†古越龙山 ，cite19†恒丰纸业 ，cite20†天虹股份 ，cite21†阳光乳业 L53: cite22†金徽酒 L54: 首封时间<br>L55: 1cite23†房地产 配酒，挺合理的。但是房地产连贯性不敢恭维，消费自然同理可得结果。<br>L56: 2九阳均瑶发酵做身位优势。<br>L57: 3日内认可度其实也一般，尤其九阳均瑶目前处于缩量过多状态，实际承接还不明，后排没理由冒险，只能当气氛组看待。<br>
来源：https://m.tgb.cn/a/2vvUuPqu2Tl

讨论 opinion_c411f771d7bcb085bfb42037：延边刺客／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-09-30T16:39:00+08:00；主帖：2026-09-30T16:39:00+08:00
取得：2026-10-14T15:34:22.933405+00:00；核阅后可用：2026-10-15T09:31:30.166860+00:00
> L59: 化工<br>L60: cite24†新亚制程 ，cite25†建业股份 ，cite26†兴业股份 L61: 首封时间<br>L62: 1亚Y，业Y，是共同点。地域角度也是。<br>L63: 2第二个字是Y，同理可得九阳均瑶发酵拼音暗线。<br>L64: 3先当气氛组，脱离后再确认让不让。<br>
来源：https://m.tgb.cn/a/2vvUuPqu2Tl

讨论 opinion_983be17090052372fbb8bbfa：延边刺客／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-09-30T16:39:00+08:00；主帖：2026-09-30T16:39:00+08:00
取得：2026-10-14T15:34:22.933405+00:00；核阅后可用：2026-10-15T09:31:30.166860+00:00
> L109: 二板晋级<br>L110: 紫竹高科、有发酵，不多。而且大家对它的期望是固态，而不是汽配上海啥的边缘属性。<br>L111: cite62†传艺科技 、固态陪跑。<br>L112: cite63†亚邦股份 、Y局配套，但是主板补善水，不太明智的选择。<br>L113: cite64†上海洗霸 、固态陪跑。<br>L114: cite65†园林股份 、Y重组。<br>L115: cite66†均瑶健康 ，Y局配套，有发酵有想法，但是九阳不让你也没办法。<br>L116: ——<br>L117: 三板晋级<br>L118: cite67†深物业A 、抽象派分时，体现节点强大。实际套Y。<br>L119: cite68†九阳股份 、改股性加单，套Y身位。因为昨天cite69†新华传媒 放量了，而园林是新的套利对象。<br>L120: cite70†善水科技 、S弹性。<br>L121: cite71†时代万恒 ，S主升。首板发酵跟不上了，容易被卡位，看板块效应和对手态度吧。<br>L122: ——<br>L123: 四板晋级<br>L124: cite72†襄阳轴承 ，因为新华放量，园林新增，凸显Y的意义而活下来。其实意义和九阳差不多或许就是帮九阳，但是故意深水坑雪龙让位，这种做法有点cite73†泰慕士 逼华贸跳水的感觉，理论上算扯平了。<br>L125: ——<br>L126: 七板晋级<br>L127: 新华传媒，不给捡漏就是顶到死了。<br>L128: ——<br>L129: 观察计划<br>L130: 1观察一字板助攻方向。<br>L131: 2观察新华是否被监管升级，看园林封单大小套利。<br>L132: 3观察Y字明牌后的主升选择，234均有。<br>L133: ——<br>L134: 高标情绪<br>L135: 1襄阳坑雪龙（竞价)，九阳获利加速，符合短线逻辑。<br>L136: 2物业坑时代（发酵)，九阳获利加速，符合短线逻辑。<br>L137: ——<br>L138: 竞价最优解<br>L139: 1橡胶独立高开非一字板溢价第一，主动性。<br>L140: 2均瑶Jy一字板助攻Jy，各自排板开板做捡漏。<br>L141: ——<br>L142: 气质股<br>L143: 1Y主升，<br>L144: 襄阳坑自家兄弟，忠义角度直接跳过。<br>L145: 九阳开板换手，家电主业未发酵，60分。<br>L146: 均瑶发酵饮料，符合一字板续强，65分。<br>
来源：https://m.tgb.cn/a/2vvUuPqu2Tl

独立方向／采纳决定／反证／取消条件：待作者审阅。

## 慈文传媒 002343（5板）

同位：百洋股份 002696
同父锚候选：无
日线价量：{"boards": 5, "height_gap": 0, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 1.0, "is_space_leader": 1.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.161722, "signal_amount": 681668185.0, "amplitude": 2.9629629629629632, "turnover_vs_prev": 1.3541607355182288, "price": 8.91, "price_same_height_percentile": 0.5, "signal_amount_same_height_percentile": 0.5, "turnover_same_height_percentile": 0.5, "turnover_vs_prev_same_height_percentile": 0.5, "signal_one_word": 0.0, "log_signal_amount": 20.34005356662144, "amount_rank_competition": 0.0, "turnover_rank_competition": 0.0}
真实一字身份：0.0
因果短线角色（规则候选）：SPACE_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002242／ANCHOR_POST_BREAK
同位淘汰率：0.0；已观察分歧换手板数：4.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：UNKNOWN
本轮连续板核对：True
父锚假设：未知
同周期与价量对比：[]
历史名称／字辈扩散代理：慈文传媒／未知
扩展规则角色：SPACE_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

父锚假设：未知；SECTOR_CONTEXT_PARENT_UNKNOWN
节点：{"state": "UNKNOWN", "current_streak_start": null, "current_streak_dates": ["2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"], "complete_streak_checked": false, "prior_first_observed_board": null, "prior_observation_is_lifetime_first": false, "anchor_first_observed_streak_break": null, "anchor_boards_at_current_start": null, "child_boards_at_break": null, "missing": ["source-bound complete current streak", "explicit parent streak break; missing board is not a break"], "timing_is_causal_proof": false}
替代解释：新华传媒全局锚背景；具体父子连接未知；传媒自身轮动/相对承接，不能排除
削弱／取消条件：

讨论 opinion_5e39671456abb7359dd06f49：延边刺客／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-10-12T16:23:00+08:00；主帖：2026-10-12T16:23:00+08:00
取得：2026-10-14T12:19:01.705127+00:00；核阅后可用：2026-10-14T12:31:29.455528+00:00
> cite36†慈文传媒†xueqiu.com 、传媒伴飞，封单有一定强度，需要爆量检验后弱转强确认。
来源：https://www.xueqiu.com/5176071593/411957075

独立方向／采纳决定／反证／取消条件：待作者审阅。

## 百洋股份 002696（5板）

同位：慈文传媒 002343
同父锚候选：阳光乳业 001318
日线价量：{"boards": 5, "height_gap": 0, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 1.0, "is_space_leader": 1.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.223693, "signal_amount": 781879528.0, "amplitude": 10.051282051282051, "turnover_vs_prev": 22.465903384553577, "price": 10.73, "price_same_height_percentile": 1.0, "signal_amount_same_height_percentile": 1.0, "turnover_same_height_percentile": 1.0, "turnover_vs_prev_same_height_percentile": 1.0, "signal_one_word": 0.0, "log_signal_amount": 20.477211231654685, "amount_rank_competition": 0.34657359027997264, "turnover_rank_competition": 0.34657359027997264}
真实一字身份：0.0
因果短线角色（规则候选）：SPACE_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002242／ANCHOR_POST_BREAK
同位淘汰率：0.0；已观察分歧换手板数：4.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：CYCLE_SUPPLEMENT_CANDIDATE
本轮连续板核对：True
父锚假设：九阳股份：CYCLE_SUPPLEMENT_CANDIDATE
同周期与价量对比：[{"code": "001318", "name": "阳光乳业", "boards": 3, "signal_day": {"date": "2026-10-15", "status": "OBSERVED", "previous_official_session": "2026-10-14", "open": 1674, "close": 1697, "high": 1697, "low": 1630, "volume_native": 179004.0, "one_price_session": false, "intraday_return": 0.013739545997610403, "gap_vs_previous_official_close": 0.08489954633830199, "range_vs_previous_official_close": 0.04342190537913156, "volume_ratio_vs_previous_official_session": 0.44464979022438167}, "price_path_is_role_truth": false}]
历史名称／字辈扩散代理：百洋股份／未知
扩展规则角色：SPACE_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

父锚假设：九阳股份；EXPLICIT_CO_MENTION
节点：{"state": "EARLY_START_CONTINUES_THROUGH_ANCHOR_BREAK", "current_streak_start": "2026-10-09", "current_streak_dates": ["2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"], "complete_streak_checked": true, "prior_first_observed_board": null, "prior_observation_is_lifetime_first": false, "anchor_first_observed_streak_break": "2026-10-13", "anchor_boards_at_current_start": 5, "child_boards_at_break": 3, "missing": [], "timing_is_causal_proof": false}
替代解释：三羊/圣阳同链背景，但无已核父子时序；农业避险/魚粉等独立叙事；新华全局情绪周期，不能自动等同直接父龙
削弱／取消条件：完整名称关联池持续弱反馈，候选失去相对连续性；消费/名字线缺扩散而独立行业解释更充分；开盘实际承接或可成交未核

讨论 opinion_615005c65e583807a9d6975a：抄底冠军1573／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-10-12T17:28:51+08:00；主帖：2026-10-12T17:28:51+08:00
取得：2026-10-14T12:19:01.705127+00:00；核阅后可用：2026-10-14T12:31:29.455528+00:00
> 三羊——九阳——百洋——圣阳，三九百圣，越来越离谱了
来源：https://gubaposthk.eastmoney.com/news%2C002696%2C1781454602.html

讨论 opinion_3881102d5c2dfd4f09cbdd21：延边刺客／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-10-12T16:23:00+08:00；主帖：2026-10-12T16:23:00+08:00
取得：2026-10-14T12:19:01.705127+00:00；核阅后可用：2026-10-14T12:31:29.455528+00:00
> cite39†百洋股份†xueqiu.com 、农业避险，板块背刺严重。但是背刺越严重，越体现主动性，不行的自己会隔日自爆。
来源：https://www.xueqiu.com/5176071593/411957075

讨论 opinion_afb25256f83d606e24a62a4c：延边刺客／AUTHOR_BODY／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-10-12T16:23:00+08:00；主帖：2026-10-12T16:23:00+08:00
取得：2026-10-14T12:19:01.705127+00:00；核阅后可用：2026-10-15T09:31:30.166860+00:00
> 1时代和九阳是装糖还是真糖？细节是什么？它们7板意义大吗？二板非传媒里远程百洋都是Y？有没有独立走出来的可能性？
来源：https://www.xueqiu.com/5176071593/411957075

讨论 opinion_5a91f9088710f3769a037d7d：西北豪／COMMENT／OLDER_DISCUSSION_BACKGROUND
正文／评论各自发布时间：2026-10-12T21:09:00+08:00；主帖：2026-10-12T16:23:00+08:00
取得：2026-10-14T12:19:01.705127+00:00；核阅后可用：2026-10-14T12:31:29.455528+00:00
> 1，时代和九阳是主力装糖，接的人可能就不知道了，感觉今天是让你看的，不是让你买的，它俩的分时对抗那是写在明面上的。它们7板意义感觉不大，但无奈，新华抗伤抗的非常漂亮。二板非传媒里远程百洋都是Y，Y又像羊，太抽象了只能说，九阳布的好局啊。
来源：https://www.xueqiu.com/5176071593/411957075

独立方向／采纳决定／反证／取消条件：待作者审阅。

## 新华文轩 601811（2板）

同位：富春染织 605189
同父锚候选：无
日线价量：{"boards": 2, "height_gap": 3, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.4, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.080946, "signal_amount": 1566969940.0, "amplitude": 12.688172043010754, "turnover_vs_prev": 1.113795710934183, "price": 25.58, "price_same_height_percentile": 1.0, "signal_amount_same_height_percentile": 1.0, "turnover_same_height_percentile": 0.5, "turnover_vs_prev_same_height_percentile": 1.0, "signal_one_word": 0.0, "log_signal_amount": 21.17240961762195, "amount_rank_competition": 0.34657359027997264, "turnover_rank_competition": 0.0}
真实一字身份：0.0
因果短线角色（规则候选）：SUPPLEMENT_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：603096／ANCHOR_FIRST_BROKEN
同位淘汰率：0.5；已观察分歧换手板数：2.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：RELATED_UNRESOLVED
本轮连续板核对：True
父锚假设：慈文传媒：RELATED_UNRESOLVED；华北制药：RELATED_UNRESOLVED；新华传媒：RELATED_UNRESOLVED
同周期与价量对比：[]
历史名称／字辈扩散代理：新华文轩／未知
扩展规则角色：SUPPLEMENT_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

限定来源未覆盖；未点名不等于无催化，不自动看空或转正。

独立方向／采纳决定／反证／取消条件：待作者审阅。

## 富春染织 605189（2板）

同位：新华文轩 601811
同父锚候选：无
日线价量：{"boards": 2, "height_gap": 3, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.4, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.17102399999999998, "signal_amount": 724182159.0, "amplitude": 3.080714725816389, "turnover_vs_prev": 1.0776533556711947, "price": 17.85, "price_same_height_percentile": 0.5, "signal_amount_same_height_percentile": 0.5, "turnover_same_height_percentile": 1.0, "turnover_vs_prev_same_height_percentile": 0.5, "signal_one_word": 0.0, "log_signal_amount": 20.40055352091341, "amount_rank_competition": 0.0, "turnover_rank_competition": 0.34657359027997264}
真实一字身份：0.0
因果短线角色（规则候选）：USURP_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002696／ANCHOR_HEAVY_TURNOVER
同位淘汰率：0.5；已观察分歧换手板数：1.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：UNKNOWN
本轮连续板核对：True
父锚假设：未知
同周期与价量对比：[]
历史名称／字辈扩散代理：富春染织／未知
扩展规则角色：USURP_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

限定来源未覆盖；未点名不等于无催化，不自动看空或转正。

独立方向／采纳决定／反证／取消条件：待作者审阅。
