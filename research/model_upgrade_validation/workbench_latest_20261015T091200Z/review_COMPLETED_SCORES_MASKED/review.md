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

自主父锚与身份区分：3板换手/穿越规则候选；未采纳百洋补涨归属。
本轮连续板核对：True
父锚假设：未知
同周期与价量对比：[]
支持：与阳光共同3板；14日一字后开板仍封板，日线满足规则换手检验。
反证/其他解释：换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。
取消条件：需要实际盘中承接和相关板块同步确认；放量冲高回落、同题材走弱或盘口验证失败则取消。
历史名称／字辈扩散代理：冰山冷热／未知
扩展规则角色：TURNOVER_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

限定来源未覆盖；未点名不等于无催化，不自动看空或转正。

作者实际审阅：

```json
{
  "code": "000530",
  "name": "冰山冷热",
  "boards": 3,
  "promotion_judgement": null,
  "forum_preread_assessment": {
    "reviewed": true,
    "unknown_not_negative": true,
    "reason": "限定已读旧来源未点名本股，未覆盖不是无催化或断板证据；保留独立价格和竞争观察。"
  },
  "supplement_assessment": {
    "reviewed": true,
    "reason": "3板换手/穿越规则候选；未采纳百洋补涨归属。 换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
    "unknown_not_negative": true,
    "same_height_peers": [
      "001318"
    ],
    "same_anchor_candidate_peers": [],
    "relations": [],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "forum_reasoning_assessment": {
    "reviewed": true,
    "synthesis": "3板换手/穿越规则候选；未采纳百洋补涨归属。 换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
    "agreement": "UNDETERMINED",
    "agreement_reason": "没有限定来源观点可比较，不宣称与不存在的观点一致或相反。",
    "unknown_not_negative": true,
    "opinions": []
  },
  "identity_discovery_assessment": {
    "reviewed": true,
    "identity_call": "UNKNOWN",
    "accepted_anchor_codes": [],
    "rejected_or_pending_anchors": [],
    "relative_role_hypothesis": "3板换手/穿越规则候选；未采纳百洋补涨归属。",
    "support": "与阳光共同3板；14日一字后开板仍封板，日线满足规则换手检验。",
    "counterevidence": "换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
    "cancel_condition": "需要实际盘中承接和相关板块同步确认；放量冲高回落、同题材走弱或盘口验证失败则取消。",
    "same_cycle_competitors": [],
    "role_truth_certified": false,
    "reviewed_at": "2026-10-15T09:43:36.555574+00:00",
    "parent_unknown_does_not_erase_known_space_or_turnover_rule": true
  },
  "role_hypothesis": "3板换手/穿越规则候选；未采纳百洋补涨归属。",
  "support": "与阳光共同3板；14日一字后开板仍封板，日线满足规则换手检验。",
  "rebuttal": "换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
  "cancel_condition": "需要实际盘中承接和相关板块同步确认；放量冲高回落、同题材走弱或盘口验证失败则取消。",
  "relative_thesis": "共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
  "comparison_peers": [
    "001318"
  ],
  "opening_confirmation": {
    "09:25": "实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。",
    "09:35": "用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"
  },
  "short_term_assessment": {
    "checks": {
      "launch_and_continuation": {
        "state": "UNKNOWN",
        "reason": "本轮日线连板先后已单独核对；旧11类源门尚缺source_bound_actual_start、checked_specific_topic_narrative_present，不把规则角色变成严格身份真值。 换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
        "evidence_refs": []
      },
      "theme_transition": {
        "state": "UNKNOWN",
        "reason": "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。 换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
        "evidence_refs": []
      },
      "independent_strength": {
        "state": "UNKNOWN",
        "reason": "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。 换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
        "evidence_refs": []
      },
      "absorption_and_competition": {
        "state": "UNKNOWN",
        "reason": "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。 换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。",
        "evidence_refs": []
      }
    },
    "alternative_explanations": [
      "3板换手/穿越规则候选；未采纳百洋补涨归属。",
      "共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"
    ],
    "counterevidence": [
      "换手约41.05%、量约13.61倍，分时回封质量和冰雪主线同步尚缺；龙江失败提示不能整体外推题材持续。"
    ],
    "cancel_conditions": [
      "需要实际盘中承接和相关板块同步确认；放量冲高回落、同题材走弱或盘口验证失败则取消。"
    ],
    "confirmation_plan": [
      {
        "planned_at": "2026-10-16T09:25:00+08:00",
        "status": "PENDING",
        "requirement": "需要实际盘中承接和相关板块同步确认；放量冲高回落、同题材走弱或盘口验证失败则取消。"
      },
      {
        "planned_at": "2026-10-16T09:35:00+08:00",
        "status": "PENDING",
        "requirement": "真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。"
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "pairwise_competition_assessment": {
    "scopes": {
      "current": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      },
      "prior": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      }
    },
    "price_rank_is_not_core_identity": true,
    "role_causality_or_entry_confirmed": false
  },
  "narrative_type_assessment": {
    "current_trade_narrative_presence": "UNKNOWN",
    "content_types_reviewed": true,
    "unknown_not_negative": true,
    "reason": "四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",
    "fund_causality_confirmed": false
  }
}
```

## 阳光乳业 001318（3板）

同位：冰山冷热 000530
同父锚候选：百洋股份 002696
日线价量：{"boards": 3, "height_gap": 2, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.6, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.06332900000000001, "signal_amount": 300750817.0, "amplitude": 4.342190537913156, "turnover_vs_prev": 0.4446543044311663, "price": 16.97, "price_same_height_percentile": 1.0, "signal_amount_same_height_percentile": 0.5, "turnover_same_height_percentile": 0.5, "turnover_vs_prev_same_height_percentile": 0.5, "signal_one_word": 0.0, "log_signal_amount": 19.521792632682363, "amount_rank_competition": 0.0, "turnover_rank_competition": 0.0}
真实一字身份：0.0
因果短线角色（规则候选）：SUPPLEMENT_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002696／ANCHOR_HEAVY_TURNOVER
同位淘汰率：0.0；已观察分歧换手板数：3.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。
本轮连续板核对：True
父锚假设：襄阳轴承：RELATED_UNRESOLVED；九阳股份：OLD_FOLLOWER_RESTART；均瑶健康：RELATED_UNRESOLVED
同周期与价量对比：[{"code": "002696", "name": "百洋股份", "boards": 5, "signal_day": {"date": "2026-10-15", "status": "OBSERVED", "previous_official_session": "2026-10-14", "open": 1071, "close": 1073, "high": 1073, "low": 975, "volume_native": 747713.0, "one_price_session": false, "intraday_return": 0.0018674136321195078, "gap_vs_previous_official_close": 0.09846153846153838, "range_vs_previous_official_close": 0.10051282051282051, "volume_ratio_vs_previous_official_session": 22.465312622059308}, "price_path_is_role_truth": false}]
支持：真实非一字3板，与冰山共同存活，换手约6.33%、量为昨日约0.44倍。
反证/其他解释：缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。
取消条件：核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。
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

作者实际审阅：

```json
{
  "code": "001318",
  "name": "阳光乳业",
  "boards": 3,
  "promotion_judgement": null,
  "forum_preread_assessment": {
    "reviewed": true,
    "unknown_not_negative": true,
    "reason": "已逐条核对旧正文和分栏评论；本股有背景不等于15日当日题材已核。"
  },
  "supplement_assessment": {
    "reviewed": true,
    "reason": "旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。 缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
    "unknown_not_negative": true,
    "same_height_peers": [
      "000530"
    ],
    "same_anchor_candidate_peers": [
      "002696"
    ],
    "relations": [
      {
        "relation_id": "sunshine_jiuyang",
        "decision": "PENDING",
        "reason": "保留关系候选而非资金因果或唯一父龙。旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。",
        "counterevidence": [
          "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。"
        ],
        "cancel_conditions": [
          "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
        ]
      },
      {
        "relation_id": "sunshine_junyao_alternative",
        "decision": "PENDING",
        "reason": "保留关系候选而非资金因果或唯一父龙。旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。",
        "counterevidence": [
          "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。"
        ],
        "cancel_conditions": [
          "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
        ]
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "forum_reasoning_assessment": {
    "reviewed": true,
    "synthesis": "旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。 缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
    "agreement": "PARTIAL",
    "agreement_reason": "采纳已核叙事与条件问题，父锚、资金及当期主动性分别核验。",
    "unknown_not_negative": true,
    "opinions": [
      {
        "opinion_id": "opinion_83bf69c0ffa8a102e1b5769c",
        "author_claim": "9月30日旧正文把阳光放在消费/名字线，并讨论Y字、均瑶、九阳等替代解释。",
        "author_reasoning": "采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\sunshine_jiuyang_relation_20261014T152814Z\\web_selected_sources_tool_response.json"
          ]
        },
        "decision": "ACCEPT",
        "reason": "采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。",
        "counterevidence": [
          "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。"
        ],
        "cancel_conditions": [
          "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
        ],
        "independent_effect": "ROLE_ONLY"
      },
      {
        "opinion_id": "opinion_c411f771d7bcb085bfb42037",
        "author_claim": "9月30日旧正文把阳光放在消费/名字线，并讨论Y字、均瑶、九阳等替代解释。",
        "author_reasoning": "采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\sunshine_jiuyang_relation_20261014T152814Z\\web_selected_sources_tool_response.json"
          ]
        },
        "decision": "ACCEPT",
        "reason": "采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。",
        "counterevidence": [
          "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。"
        ],
        "cancel_conditions": [
          "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
        ],
        "independent_effect": "ROLE_ONLY"
      },
      {
        "opinion_id": "opinion_983be17090052372fbb8bbfa",
        "author_claim": "9月30日旧正文把阳光放在消费/名字线，并讨论Y字、均瑶、九阳等替代解释。",
        "author_reasoning": "采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\sunshine_jiuyang_relation_20261014T152814Z\\web_selected_sources_tool_response.json"
          ]
        },
        "decision": "ACCEPT",
        "reason": "采纳旧跟风背景与替代父锚范围；旧文不等于15日仍在交易同一因果，父锚归属仍待核。",
        "counterevidence": [
          "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。"
        ],
        "cancel_conditions": [
          "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
        ],
        "independent_effect": "ROLE_ONLY"
      }
    ]
  },
  "identity_discovery_assessment": {
    "reviewed": true,
    "identity_call": "OLD_FOLLOWER_RESTART",
    "accepted_anchor_codes": [],
    "rejected_or_pending_anchors": [
      {
        "anchor_code": "000678",
        "decision": "REJECT",
        "reason": "仅单字或语义近邻，缺少直接叙事/业务及相邻时序联系，不采纳其为父锚；未证实并非永久证伪。"
      },
      {
        "anchor_code": "002242",
        "decision": "PENDING",
        "reason": "旧跟风重起、并列周期及原父锚状态缺失等替代解释仍在；保留为待核竞争假设。"
      },
      {
        "anchor_code": "605388",
        "decision": "PENDING",
        "reason": "旧跟风重起、并列周期及原父锚状态缺失等替代解释仍在；保留为待核竞争假设。"
      }
    ],
    "relative_role_hypothesis": "旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。",
    "support": "真实非一字3板，与冰山共同存活，换手约6.33%、量为昨日约0.44倍。",
    "counterevidence": "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
    "cancel_condition": "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。",
    "same_cycle_competitors": [
      "002696"
    ],
    "role_truth_certified": false,
    "reviewed_at": "2026-10-15T09:43:36.555634+00:00",
    "parent_unknown_does_not_erase_known_space_or_turnover_rule": true
  },
  "role_hypothesis": "旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。",
  "support": "真实非一字3板，与冰山共同存活，换手约6.33%、量为昨日约0.44倍。",
  "rebuttal": "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
  "cancel_condition": "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。",
  "relative_thesis": "共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
  "comparison_peers": [
    "000530"
  ],
  "opening_confirmation": {
    "09:25": "实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。",
    "09:35": "用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"
  },
  "short_term_assessment": {
    "checks": {
      "launch_and_continuation": {
        "state": "UNKNOWN",
        "reason": "本轮日线连板先后已单独核对；旧11类源门尚缺source_bound_actual_start、checked_specific_topic_narrative_present，不把规则角色变成严格身份真值。 缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
        "evidence_refs": []
      },
      "theme_transition": {
        "state": "UNKNOWN",
        "reason": "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。 缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
        "evidence_refs": []
      },
      "independent_strength": {
        "state": "UNKNOWN",
        "reason": "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。 缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
        "evidence_refs": []
      },
      "absorption_and_competition": {
        "state": "UNKNOWN",
        "reason": "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。 缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。",
        "evidence_refs": []
      }
    },
    "alternative_explanations": [
      "旧跟风重起；规则给补涨候选，百洋/九阳父锚归属仍待核。",
      "共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"
    ],
    "counterevidence": [
      "缩量封板不能独自证明强承接；与百洋的同音及时序关系不足以排除旧周期独立修复。"
    ],
    "cancel_conditions": [
      "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
    ],
    "confirmation_plan": [
      {
        "planned_at": "2026-10-16T09:25:00+08:00",
        "status": "PENDING",
        "requirement": "核实次日有效换手及旧周期/新父锚两种解释；无量高开后失守、补涨关系被相反来源推翻则取消。"
      },
      {
        "planned_at": "2026-10-16T09:35:00+08:00",
        "status": "PENDING",
        "requirement": "真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。"
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "pairwise_competition_assessment": {
    "scopes": {
      "current": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      },
      "prior": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      }
    },
    "price_rank_is_not_core_identity": true,
    "role_causality_or_entry_confirmed": false
  },
  "narrative_type_assessment": {
    "current_trade_narrative_presence": "UNKNOWN",
    "content_types_reviewed": true,
    "unknown_not_negative": true,
    "reason": "四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",
    "fund_causality_confirmed": false
  }
}
```

## 慈文传媒 002343（5板）

同位：百洋股份 002696
同父锚候选：无
日线价量：{"boards": 5, "height_gap": 0, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 1.0, "is_space_leader": 1.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.161722, "signal_amount": 681668185.0, "amplitude": 2.9629629629629632, "turnover_vs_prev": 1.3541607355182288, "price": 8.91, "price_same_height_percentile": 0.5, "signal_amount_same_height_percentile": 0.5, "turnover_same_height_percentile": 0.5, "turnover_vs_prev_same_height_percentile": 0.5, "signal_one_word": 0.0, "log_signal_amount": 20.34005356662144, "amount_rank_competition": 0.0, "turnover_rank_competition": 0.0}
真实一字身份：0.0
因果短线角色（规则候选）：SPACE_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002242／ANCHOR_POST_BREAK
同位淘汰率：0.0；已观察分歧换手板数：4.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：并列空间龙、换手核心候选；新华周期传媒竞争假设。
本轮连续板核对：True
父锚假设：未知
同周期与价量对比：[]
支持：与百洋共同5板；当日OHLC非一字，规则记录4次分歧换手，换手约16.17%、相对昨日量1.35倍。
反证/其他解释：新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。
取消条件：仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。
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

作者实际审阅：

```json
{
  "code": "002343",
  "name": "慈文传媒",
  "boards": 5,
  "promotion_judgement": null,
  "forum_preread_assessment": {
    "reviewed": true,
    "unknown_not_negative": true,
    "reason": "已逐条核对旧正文和分栏评论；本股有背景不等于15日当日题材已核。"
  },
  "supplement_assessment": {
    "reviewed": true,
    "reason": "并列空间龙、换手核心候选；新华周期传媒竞争假设。 新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
    "unknown_not_negative": true,
    "same_height_peers": [
      "002696"
    ],
    "same_anchor_candidate_peers": [],
    "relations": [
      {
        "relation_id": "ciwen_sector_role",
        "decision": "ACCEPT",
        "reason": "保留关系候选而非资金因果或唯一父龙。并列空间龙、换手核心候选；新华周期传媒竞争假设。",
        "counterevidence": [
          "新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。"
        ],
        "cancel_conditions": [
          "仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。"
        ]
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "forum_reasoning_assessment": {
    "reviewed": true,
    "synthesis": "并列空间龙、换手核心候选；新华周期传媒竞争假设。 新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
    "agreement": "PARTIAL",
    "agreement_reason": "采纳已核叙事与条件问题，父锚、资金及当期主动性分别核验。",
    "unknown_not_negative": true,
    "opinions": [
      {
        "opinion_id": "opinion_5e39671456abb7359dd06f49",
        "author_claim": "12日作者称慈文传媒伴飞并提出需放量检验后弱转强。",
        "author_reasoning": "15日非一字封板是独立新事实，旧条件可作为观察问题；没有分时/盘口就不能认证已经完成真承接。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\forum_preread_workflow_20261014T122229Z\\xueqiu_1012_tool_text.txt"
          ]
        },
        "decision": "ACCEPT",
        "reason": "15日非一字封板是独立新事实，旧条件可作为观察问题；没有分时/盘口就不能认证已经完成真承接。",
        "counterevidence": [
          "新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。"
        ],
        "cancel_conditions": [
          "仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。"
        ],
        "independent_effect": "ROLE_ONLY"
      }
    ]
  },
  "identity_discovery_assessment": {
    "reviewed": true,
    "identity_call": "UNKNOWN",
    "accepted_anchor_codes": [],
    "rejected_or_pending_anchors": [],
    "relative_role_hypothesis": "并列空间龙、换手核心候选；新华周期传媒竞争假设。",
    "support": "与百洋共同5板；当日OHLC非一字，规则记录4次分歧换手，换手约16.17%、相对昨日量1.35倍。",
    "counterevidence": "新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
    "cancel_condition": "仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。",
    "same_cycle_competitors": [],
    "role_truth_certified": false,
    "reviewed_at": "2026-10-15T09:43:36.555687+00:00",
    "parent_unknown_does_not_erase_known_space_or_turnover_rule": true
  },
  "role_hypothesis": "并列空间龙、换手核心候选；新华周期传媒竞争假设。",
  "support": "与百洋共同5板；当日OHLC非一字，规则记录4次分歧换手，换手约16.17%、相对昨日量1.35倍。",
  "rebuttal": "新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
  "cancel_condition": "仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。",
  "relative_thesis": "共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
  "comparison_peers": [
    "002696"
  ],
  "opening_confirmation": {
    "09:25": "实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。",
    "09:35": "用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"
  },
  "short_term_assessment": {
    "checks": {
      "launch_and_continuation": {
        "state": "UNKNOWN",
        "reason": "本轮日线连板先后已单独核对；旧11类源门尚缺source_bound_actual_start、checked_specific_topic_narrative_present，不把规则角色变成严格身份真值。 新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
        "evidence_refs": []
      },
      "theme_transition": {
        "state": "UNKNOWN",
        "reason": "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。 新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
        "evidence_refs": []
      },
      "independent_strength": {
        "state": "UNKNOWN",
        "reason": "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。 新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
        "evidence_refs": []
      },
      "absorption_and_competition": {
        "state": "UNKNOWN",
        "reason": "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。 新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。",
        "evidence_refs": []
      }
    },
    "alternative_explanations": [
      "并列空间龙、换手核心候选；新华周期传媒竞争假设。",
      "共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"
    ],
    "counterevidence": [
      "新经典已断连续板，传媒后排延续待独立核实；并列最高不排除双双失败。"
    ],
    "cancel_conditions": [
      "仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。"
    ],
    "confirmation_plan": [
      {
        "planned_at": "2026-10-16T09:25:00+08:00",
        "status": "PENDING",
        "requirement": "仅在次日不极端高开、真实开板承接与必要传媒同向性核实后保留偏晋级观察；丢失封板并不能回封、板块显著背离则取消。"
      },
      {
        "planned_at": "2026-10-16T09:35:00+08:00",
        "status": "PENDING",
        "requirement": "真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。"
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "pairwise_competition_assessment": {
    "scopes": {
      "current": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      },
      "prior": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      }
    },
    "price_rank_is_not_core_identity": true,
    "role_causality_or_entry_confirmed": false
  },
  "narrative_type_assessment": {
    "current_trade_narrative_presence": "UNKNOWN",
    "content_types_reviewed": true,
    "unknown_not_negative": true,
    "reason": "四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",
    "fund_causality_confirmed": false
  }
}
```

## 百洋股份 002696（5板）

同位：慈文传媒 002343
同父锚候选：阳光乳业 001318
日线价量：{"boards": 5, "height_gap": 0, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 1.0, "is_space_leader": 1.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.223693, "signal_amount": 781879528.0, "amplitude": 10.051282051282051, "turnover_vs_prev": 22.465903384553577, "price": 10.73, "price_same_height_percentile": 1.0, "signal_amount_same_height_percentile": 1.0, "turnover_same_height_percentile": 1.0, "turnover_vs_prev_same_height_percentile": 1.0, "signal_one_word": 0.0, "log_signal_amount": 20.477211231654685, "amount_rank_competition": 0.34657359027997264, "turnover_rank_competition": 0.34657359027997264}
真实一字身份：0.0
因果短线角色（规则候选）：SPACE_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002242／ANCHOR_POST_BREAK
同位淘汰率：0.0；已观察分歧换手板数：4.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。
本轮连续板核对：True
父锚假设：九阳股份：CYCLE_SUPPLEMENT_CANDIDATE
同周期与价量对比：[{"code": "001318", "name": "阳光乳业", "boards": 3, "signal_day": {"date": "2026-10-15", "status": "OBSERVED", "previous_official_session": "2026-10-14", "open": 1674, "close": 1697, "high": 1697, "low": 1630, "volume_native": 179004.0, "one_price_session": false, "intraday_return": 0.013739545997610403, "gap_vs_previous_official_close": 0.08489954633830199, "range_vs_previous_official_close": 0.04342190537913156, "volume_ratio_vs_previous_official_session": 0.44464979022438167}, "price_path_is_role_truth": false}]
支持：14日一字后15日真实非一字封板，九阳旧关系有已核正文/评论及先后节点。
反证/其他解释：量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。
取消条件：不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。
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

作者实际审阅：

```json
{
  "code": "002696",
  "name": "百洋股份",
  "boards": 5,
  "promotion_judgement": null,
  "forum_preread_assessment": {
    "reviewed": true,
    "unknown_not_negative": true,
    "reason": "已逐条核对旧正文和分栏评论；本股有背景不等于15日当日题材已核。"
  },
  "supplement_assessment": {
    "reviewed": true,
    "reason": "并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。 量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
    "unknown_not_negative": true,
    "same_height_peers": [
      "002343"
    ],
    "same_anchor_candidate_peers": [
      "001318"
    ],
    "relations": [
      {
        "relation_id": "baiyang_jiuyang",
        "decision": "ACCEPT",
        "reason": "保留关系候选而非资金因果或唯一父龙。并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。",
        "counterevidence": [
          "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。"
        ],
        "cancel_conditions": [
          "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
        ]
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "forum_reasoning_assessment": {
    "reviewed": true,
    "synthesis": "并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。 量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
    "agreement": "PARTIAL",
    "agreement_reason": "采纳已核叙事与条件问题，父锚、资金及当期主动性分别核验。",
    "unknown_not_negative": true,
    "opinions": [
      {
        "opinion_id": "opinion_615005c65e583807a9d6975a",
        "author_claim": "正文确实出现三羊—九阳—百洋—圣阳的名称链。",
        "author_reasoning": "名称叙事存在及本轮先后节点支持九阳周期候选；不认证三羊/圣阳为直接父锚或唯一补涨龙。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\forum_preread_workflow_20261014T122229Z\\guba_name_tool_text.txt"
          ]
        },
        "decision": "ACCEPT",
        "reason": "名称叙事存在及本轮先后节点支持九阳周期候选；不认证三羊/圣阳为直接父锚或唯一补涨龙。",
        "counterevidence": [
          "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。"
        ],
        "cancel_conditions": [
          "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
        ],
        "independent_effect": "ROLE_ONLY"
      },
      {
        "opinion_id": "opinion_3881102d5c2dfd4f09cbdd21",
        "author_claim": "12日作者提出百洋农业避险、相对主动的假设。",
        "author_reasoning": "农业避险及主动性须当期完整题材和价格路径复核。15日巨量只说明成交放大，不能验证主动买盘或资金因果。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\forum_preread_workflow_20261014T122229Z\\xueqiu_1012_tool_text.txt"
          ]
        },
        "decision": "PENDING",
        "reason": "农业避险及主动性须当期完整题材和价格路径复核。15日巨量只说明成交放大，不能验证主动买盘或资金因果。",
        "counterevidence": [
          "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。"
        ],
        "cancel_conditions": [
          "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
        ],
        "independent_effect": "UNCERTAIN"
      },
      {
        "opinion_id": "opinion_afb25256f83d606e24a62a4c",
        "author_claim": "作者问时代/九阳是否真强、Y连接以及百洋能否独立；没有给出已确认资金布局或本次16日晋级结论。",
        "author_reasoning": "提问是检验独立性及名称共振的线索，不是作者肯定九阳已经安排百洋。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "QUESTION",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\forum_preread_workflow_20261014T122229Z\\xueqiu_1012_tool_text.txt"
          ]
        },
        "decision": "ACCEPT",
        "reason": "提问是检验独立性及名称共振的线索，不是作者肯定九阳已经安排百洋。",
        "counterevidence": [
          "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。"
        ],
        "cancel_conditions": [
          "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
        ],
        "independent_effect": "ROLE_ONLY"
      },
      {
        "opinion_id": "opinion_5a91f9088710f3769a037d7d",
        "author_claim": "评论者把九阳、百洋与Y/羊相联，并推测主力装糖与布局。",
        "author_reasoning": "保留名称/父锚研究线索；资金布局部分缺少交易级证据，不能采纳为事实或当前晋级理由。",
        "testable_expectation": "以原完整同位、真实日线先后节点及次日实际竞价/开板后的观察检验；保留共同晋级、共同失败和分化。",
        "author_claim_form": "HYPOTHESIS",
        "author_call": "UNSPECIFIED",
        "author_expectation_target_date": null,
        "verification": {
          "status": "VERIFIED",
          "scope": "DISCUSSION_PRESENCE",
          "reason": "已核原文字符偏移与正文/评论各自时间，只确认这段讨论确实出现。",
          "evidence_refs": [
            "D:\\Documents\\ChatGPT\\龙头战法\\results\\forum_preread_workflow_20261014T122229Z\\xueqiu_1012_tool_text.txt"
          ]
        },
        "decision": "PENDING",
        "reason": "保留名称/父锚研究线索；资金布局部分缺少交易级证据，不能采纳为事实或当前晋级理由。",
        "counterevidence": [
          "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。"
        ],
        "cancel_conditions": [
          "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
        ],
        "independent_effect": "UNCERTAIN"
      }
    ]
  },
  "identity_discovery_assessment": {
    "reviewed": true,
    "identity_call": "CYCLE_SUPPLEMENT_CANDIDATE",
    "accepted_anchor_codes": [
      "002242"
    ],
    "rejected_or_pending_anchors": [],
    "relative_role_hypothesis": "并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。",
    "support": "14日一字后15日真实非一字封板，九阳旧关系有已核正文/评论及先后节点。",
    "counterevidence": "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
    "cancel_condition": "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。",
    "same_cycle_competitors": [
      "001318"
    ],
    "role_truth_certified": false,
    "reviewed_at": "2026-10-15T09:43:36.555734+00:00",
    "parent_unknown_does_not_erase_known_space_or_turnover_rule": true
  },
  "role_hypothesis": "并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。",
  "support": "14日一字后15日真实非一字封板，九阳旧关系有已核正文/评论及先后节点。",
  "rebuttal": "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
  "cancel_condition": "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。",
  "relative_thesis": "共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
  "comparison_peers": [
    "002343"
  ],
  "opening_confirmation": {
    "09:25": "实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。",
    "09:35": "用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"
  },
  "short_term_assessment": {
    "checks": {
      "launch_and_continuation": {
        "state": "UNKNOWN",
        "reason": "本轮日线连板先后已单独核对；旧11类源门尚缺source_bound_actual_start、checked_specific_topic_narrative_present，不把规则角色变成严格身份真值。 量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
        "evidence_refs": []
      },
      "theme_transition": {
        "state": "UNKNOWN",
        "reason": "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。 量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
        "evidence_refs": []
      },
      "independent_strength": {
        "state": "UNKNOWN",
        "reason": "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。 量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
        "evidence_refs": []
      },
      "absorption_and_competition": {
        "state": "UNKNOWN",
        "reason": "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。 量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。",
        "evidence_refs": []
      }
    },
    "alternative_explanations": [
      "并列空间龙；九阳周期补涨来源候选，现已进入自身换手检验。",
      "共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"
    ],
    "counterevidence": [
      "量放大约22.47倍、振幅约10.05%，不能把巨量本身认证真承接；与慈文共生或共同失败均保留。"
    ],
    "cancel_conditions": [
      "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
    ],
    "confirmation_plan": [
      {
        "planned_at": "2026-10-16T09:25:00+08:00",
        "status": "PENDING",
        "requirement": "不因一字转非一字机械转弱；次日竞价、开板后实际承接和风险收益确认前只观察，极端高开或放量不能回封则取消。"
      },
      {
        "planned_at": "2026-10-16T09:35:00+08:00",
        "status": "PENDING",
        "requirement": "真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。"
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "pairwise_competition_assessment": {
    "scopes": {
      "current": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      },
      "prior": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      }
    },
    "price_rank_is_not_core_identity": true,
    "role_causality_or_entry_confirmed": false
  },
  "narrative_type_assessment": {
    "current_trade_narrative_presence": "UNKNOWN",
    "content_types_reviewed": true,
    "unknown_not_negative": true,
    "reason": "四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",
    "fund_causality_confirmed": false
  }
}
```

## 新华文轩 601811（2板）

同位：富春染织 605189
同父锚候选：无
日线价量：{"boards": 2, "height_gap": 3, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.4, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.080946, "signal_amount": 1566969940.0, "amplitude": 12.688172043010754, "turnover_vs_prev": 1.113795710934183, "price": 25.58, "price_same_height_percentile": 1.0, "signal_amount_same_height_percentile": 1.0, "turnover_same_height_percentile": 0.5, "turnover_vs_prev_same_height_percentile": 1.0, "signal_one_word": 0.0, "log_signal_amount": 21.17240961762195, "amount_rank_competition": 0.34657359027997264, "turnover_rank_competition": 0.0}
真实一字身份：0.0
因果短线角色（规则候选）：SUPPLEMENT_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：603096／ANCHOR_FIRST_BROKEN
同位淘汰率：0.5；已观察分歧换手板数：2.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。
本轮连续板核对：True
父锚假设：慈文传媒：RELATED_UNRESOLVED；华北制药：RELATED_UNRESOLVED；新华传媒：RELATED_UNRESOLVED
同周期与价量对比：[]
支持：同身位2只成功、2只冲板失败，已观察非一字；当日成交额在同位居前。
反证/其他解释：振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。
取消条件：需要相关主线实际扩散及自身回封质量核实；传媒活口走弱、低开无承接或改属其他催化则取消。
历史名称／字辈扩散代理：新华文轩／未知
扩展规则角色：SUPPLEMENT_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

限定来源未覆盖；未点名不等于无催化，不自动看空或转正。

作者实际审阅：

```json
{
  "code": "601811",
  "name": "新华文轩",
  "boards": 2,
  "promotion_judgement": null,
  "forum_preread_assessment": {
    "reviewed": true,
    "unknown_not_negative": true,
    "reason": "限定已读旧来源未点名本股，未覆盖不是无催化或断板证据；保留独立价格和竞争观察。"
  },
  "supplement_assessment": {
    "reviewed": true,
    "reason": "2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。 振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
    "unknown_not_negative": true,
    "same_height_peers": [
      "605189"
    ],
    "same_anchor_candidate_peers": [],
    "relations": [],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "forum_reasoning_assessment": {
    "reviewed": true,
    "synthesis": "2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。 振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
    "agreement": "UNDETERMINED",
    "agreement_reason": "没有限定来源观点可比较，不宣称与不存在的观点一致或相反。",
    "unknown_not_negative": true,
    "opinions": []
  },
  "identity_discovery_assessment": {
    "reviewed": true,
    "identity_call": "RELATED_UNRESOLVED",
    "accepted_anchor_codes": [],
    "rejected_or_pending_anchors": [
      {
        "anchor_code": "002343",
        "decision": "PENDING",
        "reason": "旧跟风重起、并列周期及原父锚状态缺失等替代解释仍在；保留为待核竞争假设。"
      },
      {
        "anchor_code": "600812",
        "decision": "REJECT",
        "reason": "仅单字或语义近邻，缺少直接叙事/业务及相邻时序联系，不采纳其为父锚；未证实并非永久证伪。"
      },
      {
        "anchor_code": "600825",
        "decision": "PENDING",
        "reason": "旧跟风重起、并列周期及原父锚状态缺失等替代解释仍在；保留为待核竞争假设。"
      }
    ],
    "relative_role_hypothesis": "2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。",
    "support": "同身位2只成功、2只冲板失败，已观察非一字；当日成交额在同位居前。",
    "counterevidence": "振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
    "cancel_condition": "需要相关主线实际扩散及自身回封质量核实；传媒活口走弱、低开无承接或改属其他催化则取消。",
    "same_cycle_competitors": [],
    "role_truth_certified": false,
    "reviewed_at": "2026-10-15T09:43:36.555758+00:00",
    "parent_unknown_does_not_erase_known_space_or_turnover_rule": true
  },
  "role_hypothesis": "2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。",
  "support": "同身位2只成功、2只冲板失败，已观察非一字；当日成交额在同位居前。",
  "rebuttal": "振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
  "cancel_condition": "需要相关主线实际扩散及自身回封质量核实；传媒活口走弱、低开无承接或改属其他催化则取消。",
  "relative_thesis": "共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
  "comparison_peers": [
    "605189"
  ],
  "opening_confirmation": {
    "09:25": "实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。",
    "09:35": "用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"
  },
  "short_term_assessment": {
    "checks": {
      "launch_and_continuation": {
        "state": "UNKNOWN",
        "reason": "本轮日线连板先后已单独核对；旧11类源门尚缺source_bound_actual_start、consecutive_same_scope_stock_observations、board_continuation_observed、checked_specific_topic_narrative_present，不把规则角色变成严格身份真值。 振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
        "evidence_refs": []
      },
      "theme_transition": {
        "state": "UNKNOWN",
        "reason": "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。 振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
        "evidence_refs": []
      },
      "independent_strength": {
        "state": "UNKNOWN",
        "reason": "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。 振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
        "evidence_refs": []
      },
      "absorption_and_competition": {
        "state": "UNKNOWN",
        "reason": "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。 振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。",
        "evidence_refs": []
      }
    },
    "alternative_explanations": [
      "2板补涨/卡位规则候选；新经典、慈文、新华三种周期解释并列。",
      "共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"
    ],
    "counterevidence": [
      "振幅约12.69%，仅名称/语义与同时点不能确认传媒资金切换。没有合格15日股吧正文覆盖，也不等于没有催化。"
    ],
    "cancel_conditions": [
      "需要相关主线实际扩散及自身回封质量核实；传媒活口走弱、低开无承接或改属其他催化则取消。"
    ],
    "confirmation_plan": [
      {
        "planned_at": "2026-10-16T09:25:00+08:00",
        "status": "PENDING",
        "requirement": "需要相关主线实际扩散及自身回封质量核实；传媒活口走弱、低开无承接或改属其他催化则取消。"
      },
      {
        "planned_at": "2026-10-16T09:35:00+08:00",
        "status": "PENDING",
        "requirement": "真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。"
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "pairwise_competition_assessment": {
    "scopes": {
      "current": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      },
      "prior": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      }
    },
    "price_rank_is_not_core_identity": true,
    "role_causality_or_entry_confirmed": false
  },
  "narrative_type_assessment": {
    "current_trade_narrative_presence": "UNKNOWN",
    "content_types_reviewed": true,
    "unknown_not_negative": true,
    "reason": "四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",
    "fund_causality_confirmed": false
  }
}
```

## 富春染织 605189（2板）

同位：新华文轩 601811
同父锚候选：无
日线价量：{"boards": 2, "height_gap": 3, "same_height_rivals": 1, "unique_space": 0.0, "height_share": 0.4, "is_space_leader": 0.0, "emotion_score": 0.4855333644568939, "promotion_2to3": 0.18181818181818182, "promotion_3to4": 0.0, "promotion_mid": 0.26666666666666666, "previous_limit_tail_loss": -0.09992523220974465, "failed_rate": 0.1282051282051282, "down_limit_rate": 1.0, "breadth_health": 0.4822649572649572, "ladder_health": 0.6361904761904762, "loss_effect_health": 0.3413817663817664, "crowding_risk": 0.0, "regime_fragility": 0.4037192492039551, "turnover": 0.17102399999999998, "signal_amount": 724182159.0, "amplitude": 3.080714725816389, "turnover_vs_prev": 1.0776533556711947, "price": 17.85, "price_same_height_percentile": 0.5, "signal_amount_same_height_percentile": 0.5, "turnover_same_height_percentile": 1.0, "turnover_vs_prev_same_height_percentile": 0.5, "signal_one_word": 0.0, "log_signal_amount": 20.40055352091341, "amount_rank_competition": 0.0, "turnover_rank_competition": 0.34657359027997264}
真实一字身份：0.0
因果短线角色（规则候选）：USURP_DRAGON；周期：EBB_HANDOVER
父锚候选及状态：002696／ANCHOR_HEAVY_TURNOVER
同位淘汰率：0.5；已观察分歧换手板数：1.0
自动规则标签不等于独立龙头真值；缺失、可成交性及原独立方向分别保留。

自主父锚与身份区分：2板卡位规则候选；不采纳百洋补涨归属。
本轮连续板核对：True
父锚假设：未知
同周期与价量对比：[]
支持：与新华文轩共同2板，同位失败2只；当日非一字且振幅约3.08%。
反证/其他解释：题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。
取消条件：需要公司公告与实际板块助攻核实；真实走势不能维持回封、业绩叙事被公告否定或风险收益不合适则取消。
历史名称／字辈扩散代理：富春染织／未知
扩展规则角色：USURP_DRAGON；首板支持：0.0
该层为名称共现代理，未认证公司行业、活跃题材、资金带动或可成交。
字段语义：{"turnover_vs_prev": "ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates", "relative_volume": "native-volume ratio to previous official session; distinct from turnover ratio", "amount_turnover": "quote field positions inferred and crosschecked; not exchange certification", "AI_page_business_text": "excluded from company facts, numeric industry fields and new semantic roles"}

限定来源未覆盖；未点名不等于无催化，不自动看空或转正。

作者实际审阅：

```json
{
  "code": "605189",
  "name": "富春染织",
  "boards": 2,
  "promotion_judgement": null,
  "forum_preread_assessment": {
    "reviewed": true,
    "unknown_not_negative": true,
    "reason": "限定已读旧来源未点名本股，未覆盖不是无催化或断板证据；保留独立价格和竞争观察。"
  },
  "supplement_assessment": {
    "reviewed": true,
    "reason": "2板卡位规则候选；不采纳百洋补涨归属。 题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
    "unknown_not_negative": true,
    "same_height_peers": [
      "601811"
    ],
    "same_anchor_candidate_peers": [],
    "relations": [],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "forum_reasoning_assessment": {
    "reviewed": true,
    "synthesis": "2板卡位规则候选；不采纳百洋补涨归属。 题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
    "agreement": "UNDETERMINED",
    "agreement_reason": "没有限定来源观点可比较，不宣称与不存在的观点一致或相反。",
    "unknown_not_negative": true,
    "opinions": []
  },
  "identity_discovery_assessment": {
    "reviewed": true,
    "identity_call": "UNKNOWN",
    "accepted_anchor_codes": [],
    "rejected_or_pending_anchors": [],
    "relative_role_hypothesis": "2板卡位规则候选；不采纳百洋补涨归属。",
    "support": "与新华文轩共同2板，同位失败2只；当日非一字且振幅约3.08%。",
    "counterevidence": "题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
    "cancel_condition": "需要公司公告与实际板块助攻核实；真实走势不能维持回封、业绩叙事被公告否定或风险收益不合适则取消。",
    "same_cycle_competitors": [],
    "role_truth_certified": false,
    "reviewed_at": "2026-10-15T09:43:36.555778+00:00",
    "parent_unknown_does_not_erase_known_space_or_turnover_rule": true
  },
  "role_hypothesis": "2板卡位规则候选；不采纳百洋补涨归属。",
  "support": "与新华文轩共同2板，同位失败2只；当日非一字且振幅约3.08%。",
  "rebuttal": "题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
  "cancel_condition": "需要公司公告与实际板块助攻核实；真实走势不能维持回封、业绩叙事被公告否定或风险收益不合适则取消。",
  "relative_thesis": "共同晋级、共同失败、彼此分化均保留；不机械指定同位必死一个。",
  "comparison_peers": [
    "601811"
  ],
  "opening_confirmation": {
    "09:25": "实际取得后核对0—7%开盘区间、竞价强弱及板块同步；不从15日收盘代填。",
    "09:35": "用真实分时/成交确认承接和可成交性，不以日线或公开开盘价声明成交。"
  },
  "short_term_assessment": {
    "checks": {
      "launch_and_continuation": {
        "state": "UNKNOWN",
        "reason": "本轮日线连板先后已单独核对；旧11类源门尚缺source_bound_actual_start、checked_specific_topic_narrative_present，不把规则角色变成严格身份真值。 题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
        "evidence_refs": []
      },
      "theme_transition": {
        "state": "UNKNOWN",
        "reason": "历史主线4日实际消费，但其中未知状态不是可确认的主线切换；旧叙事不移植成当期主题。 题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
        "evidence_refs": []
      },
      "independent_strength": {
        "state": "UNKNOWN",
        "reason": "尚缺当期完整同题材和原对手同步收益/压力窗口，无法确认独立性。 题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
        "evidence_refs": []
      },
      "absorption_and_competition": {
        "state": "UNKNOWN",
        "reason": "缺当期分时开板/回封及盘口承接；单独成交放大不足以确认吸收抛压。 题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。",
        "evidence_refs": []
      }
    },
    "alternative_explanations": [
      "2板卡位规则候选；不采纳百洋补涨归属。",
      "共时情绪修复、独立业务催化或其他父锚解释仍可能成立。"
    ],
    "counterevidence": [
      "题材/业绩页面摘要由AI生成，未认证公司业绩或主动性；没有独立完整行业助攻证据，不能用未覆盖当负面定论。"
    ],
    "cancel_conditions": [
      "需要公司公告与实际板块助攻核实；真实走势不能维持回封、业绩叙事被公告否定或风险收益不合适则取消。"
    ],
    "confirmation_plan": [
      {
        "planned_at": "2026-10-16T09:25:00+08:00",
        "status": "PENDING",
        "requirement": "需要公司公告与实际板块助攻核实；真实走势不能维持回封、业绩叙事被公告否定或风险收益不合适则取消。"
      },
      {
        "planned_at": "2026-10-16T09:35:00+08:00",
        "status": "PENDING",
        "requirement": "真实开盘成交、板块同步及分时承接核实，失败即取消条件观察。"
      }
    ],
    "role_identity_confirmed": false,
    "fund_causality_confirmed": false
  },
  "pairwise_competition_assessment": {
    "scopes": {
      "current": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      },
      "prior": {
        "state": "UNKNOWN",
        "reason": "原逐对同时间采样资料不足；同位完整名单另由当期收盘池保持，不以缺失样本删除对手。",
        "comparison_peer_codes": [],
        "incomplete_peer_codes": [],
        "stronger_observed_peer_codes": [],
        "next_session_paths": {
          "joint_promotion": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "joint_failure": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          },
          "divergence": {
            "state": "POSSIBLE",
            "reason": "当期并列身位与退潮环境允许此路径，缺盘中证据不预设唯一胜者。"
          }
        }
      }
    },
    "price_rank_is_not_core_identity": true,
    "role_causality_or_entry_confirmed": false
  },
  "narrative_type_assessment": {
    "current_trade_narrative_presence": "UNKNOWN",
    "content_types_reviewed": true,
    "unknown_not_negative": true,
    "reason": "四条原叙事均已语义审阅；旧正文显式线索保持旧背景，评论单独留存。当前无15日合格正文，不能确认完整主线，也不能据此否定该股。",
    "fund_causality_confirmed": false
  }
}
```
