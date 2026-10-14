# 2026-09-09 公开早盘监控预检

- 状态：**COLLECTION_AND_CONDITIONAL_PLAN_READY**
- 复盘日/前一交易日：2026-09-08 / 2026-09-08
- 全池：71只（首板52、连板19）
- 冻结首选：600354

## 采集检查

- prior_session_bound: True
- candidate_review_pool_exact: True
- candidate_pool_nonempty: True
- public_universe_file_nonempty: True
- all_candidates_in_public_universe: True

## 交易计划检查

- conditional_primary_frozen: True
- maximum_acceptable_price_frozen: True
- position_frozen_100_share_lot: True
- holding_plan_frozen: True
- auction_conditions_frozen: True
- opening_conditions_frozen: True
- cancel_conditions_frozen: True

结论：capture registered points and evaluate the frozen conditional plan。公开行情不会单独证明排队成交，也不保证溢价。
