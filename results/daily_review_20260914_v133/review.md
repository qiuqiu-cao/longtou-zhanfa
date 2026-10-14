# 2026-09-14 龙头战法盘后复盘（证据版）

结论：`CASH`。当前不产生买入候选。

## 市场与情绪

情绪状态：升温；主板涨停48只，首板37只、连板11只、炸板34只、跌停16只。封板率58.5%，炸板率41.5%。

昨日涨停平均反馈3.28%，中位反馈4.02%，晋级率27.5%；尾部反馈-3.77%。

按涨停价成交至炸板收盘的推断反馈：平均-5.08%，中位-4.66%，大于等于7%亏损4/34；该项为普通10%涨停价成交代理，不是实际成交记录。

真实题材证据：VERIFIED_CANDIDATE_ROLES；覆盖42/48只，强验证覆盖42/48只。

龙头竞争证据：VERIFIED_PARTIAL；覆盖42/48只。

公告索引覆盖47只候选，有效硬风险事件3条、涉及2只；普通异动公告不作机械否决。

## 三种角色

- 情绪锚：闽东电力(000993) 4板，硬风险0条
- 成龙潜力最大：未验证。
- 当前买入机会最好：无，保持现金。

## 禁止买入原因

- ACTUAL_EXECUTION_AND_COMPLETE_TRADE_MODEL_UNAVAILABLE
- BROKEN_BOARD_TAIL_LOSS_ADVERSE
- FULL_MARKET_POOL_COMPLETENESS_NOT_INDEPENDENTLY_VERIFIED
- LEADER_COMPETITION_EVIDENCE_INCOMPLETE
- NEXT_SESSION_AUCTION_AND_OPENING_EVIDENCE_UNAVAILABLE
- NO_VALIDATED_FULL_POOL_SELECTION_THRESHOLD
- REAL_SPECIFIC_THEME_EVIDENCE_INCOMPLETE

## 决策可执行性

模式：`CONDITIONAL_ONLY`；入场计划：`UNAVAILABLE_UNTIL_AUCTION_CONFIRMATION`；实际成交：`UNAVAILABLE`；退出证据：`UNAVAILABLE`。

缺失字段不会按0处理：competition_identity_for_unverified_candidates、specific_theme_membership_for_unverified_candidates、auction_match_and_cancellation、opening_vwap_and_reclaim、competitor_elimination、entry_fill、t_plus_one_exit。

## 连板池

|代码|名称|板位|题材状态|题材角色|竞争身份|证券状态|硬风险|
|---|---|---:|---|---|---|---|---:|
|000993|闽东电力|4|verified_explicit_source_roster|THEME_HEIGHT_LEADER|height_anchor|VERIFIED_ELIGIBLE|0|
|000823|超声电子|3|verified_explicit_source_roster|THEME_HEIGHT_LEADER|unique_survivor|VERIFIED_ELIGIBLE|0|
|002912|中新赛克|3|verified_explicit_source_roster|THEME_HEIGHT_LEADER|unique_survivor|VERIFIED_ELIGIBLE|0|
|600876|凯盛新能|3|unavailable_real_specific_theme|UNVERIFIED|unavailable|VERIFIED_ELIGIBLE|0|
|000980|众泰汽车|2|verified_explicit_source_roster|THEME_HEIGHT_LEADER|unique_survivor|VERIFIED_ELIGIBLE|0|
|002560|通达股份|2|verified_explicit_source_roster|THEME_SUPPORT|pioneer|VERIFIED_ELIGIBLE|0|
|002585|双星新材|2|verified_explicit_source_roster|THEME_SUPPORT|pioneer|VERIFIED_ELIGIBLE|0|
|600088|中视传媒|2|verified_explicit_source_roster|THEME_HEIGHT_LEADER|unique_survivor|VERIFIED_ELIGIBLE|0|
|603439|三力制药|2|verified_explicit_source_roster|THEME_HEIGHT_LEADER|unique_survivor|VERIFIED_ELIGIBLE|0|
|605058|澳弘电子|2|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|605069|正和生态|2|verified_explicit_source_roster|THEME_HEIGHT_LEADER|unique_survivor|VERIFIED_ELIGIBLE|0|

## 首板池

|代码|名称|题材状态|题材角色|竞争身份|证券状态|硬风险|
|---|---|---|---|---|---|---:|
|000572|海马汽车|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|000812|陕西金叶|unavailable_real_specific_theme|UNVERIFIED|unavailable|VERIFIED_ELIGIBLE|0|
|000839|国安股份|verified_explicit_source_roster|THEME_SUPPORT|capacity_core|VERIFIED_ELIGIBLE|0|
|000868|安凯客车|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|000962|东方钽业|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|000978|桂林旅游|verified_explicit_source_roster|SAME_HEIGHT_CORE|pioneer|VERIFIED_ELIGIBLE|0|
|000981|山子高科|verified_explicit_source_roster|THEME_SUPPORT|pioneer|VERIFIED_ELIGIBLE|0|
|001212|中旗新材|verified_explicit_source_roster|SAME_HEIGHT_CORE|supplement|VERIFIED_ELIGIBLE|0|
|001896|豫能控股|verified_explicit_source_roster|THEME_SUPPORT|capacity_core|VERIFIED_ELIGIBLE|0|
|002141|贤丰控股|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|002172|澳洋健康|verified_explicit_source_roster|THEME_SUPPORT|capacity_core|VERIFIED_ELIGIBLE|0|
|002212|天融信|verified_explicit_source_roster|THEME_SUPPORT|pioneer|VERIFIED_ELIGIBLE|0|
|002232|启明信息|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|002346|柘中股份|verified_explicit_source_roster|SAME_HEIGHT_CORE|capacity_core|VERIFIED_ELIGIBLE|0|
|002579|中京电子|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|002669|康达新材|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|002897|意华股份|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|002918|蒙娜丽莎|unavailable_real_specific_theme|UNVERIFIED|unavailable|VERIFIED_ELIGIBLE|0|
|002976|瑞玛精密|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|600172|黄河旋风|verified_explicit_source_roster|THEME_SUPPORT|capacity_core|VERIFIED_ELIGIBLE|0|
|600186|莲花控股|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|600228|返利科技|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|600330|天通股份|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|600488|津药药业|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|600697|欧亚集团|verified_explicit_source_roster|SAME_HEIGHT_CORE|supplement|VERIFIED_ELIGIBLE|0|
|600802|福建水泥|unavailable_real_specific_theme|UNVERIFIED|unavailable|VERIFIED_ELIGIBLE|0|
|601086|国芳集团|verified_explicit_source_roster|SAME_HEIGHT_CORE|capacity_core|VERIFIED_ELIGIBLE|0|
|601123|马矿股份|unavailable_real_specific_theme|UNVERIFIED|unavailable|VERIFIED_ELIGIBLE|0|
|601567|三星电气|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|603020|爱普股份|verified_explicit_source_roster|SAME_HEIGHT_CORE|supplement|VERIFIED_ELIGIBLE|0|
|603090|宏盛股份|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|603285|键邦股份|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|2|
|603400|华之杰|verified_explicit_source_roster|THEME_SUPPORT|capacity_core|VERIFIED_ELIGIBLE|0|
|603626|科森科技|unavailable_real_specific_theme|UNVERIFIED|unavailable|VERIFIED_ELIGIBLE|0|
|603657|春光科技|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|
|605168|三人行|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|1|
|605580|恒盛能源|verified_explicit_source_roster|THEME_SUPPORT|supplement|VERIFIED_ELIGIBLE|0|

本报告没有实时竞价、开盘成交或真实退出证据；题材证据仅在绑定的时点化来源通过校验后显示，不构成买卖建议。
