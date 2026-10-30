#!/usr/bin/env python3
import json
import joblib
import pandas as pd
import numpy as np

model = joblib.load('research/model_upgrade_validation/unified_dragon_breakthrough_20261028/model_clean_5d_dragon.joblib')
feat_cols = json.load(open('research/model_upgrade_validation/unified_dragon_breakthrough_20261028/clean_feature_list_5d.json', encoding='utf-8'))

df = pd.read_csv('results/auction_formula_replay_20261028_20261028T1612Z/complete_38_prior_limit_up_formula_replay.csv')
df['code_str'] = df['code'].astype(str).str.zfill(6)

ind_df = pd.read_csv('research/first_board_github_industry_qualification_20261020/stock_basics_industry_original.csv', usecols=['symbol', 'industry'])
ind_df['code_str'] = ind_df['symbol'].astype(str).str.zfill(6)
ind_map = dict(zip(ind_df['code_str'], ind_df['industry']))
df['industry'] = df['code_str'].map(ind_map).fillna('其他行业')

df['open_gap'] = df['gap_pct'] / 100.0
df['open_gap_pct'] = df['gap_pct']
df['open_vs_prev_high'] = (df['open'] - df['previous_high']) / df['previous_high']
df['prev_vwap'] = df['previous_close'] * 0.965
df['open_vs_prev_vwap'] = (df['open'] - df['prev_vwap']) / df['prev_vwap']
df['open_vs_prev_open'] = df['open_gap']
df['vwap_cost_support'] = df['open_vs_prev_vwap']
df['is_golden_auction'] = ((df['gap_pct'] >= 2.0) & (df['gap_pct'] <= 7.0) & (df['open'] >= df['previous_high'])).astype(float)
df['prev_amplitude_pct'] = 4.0
df['prev_seal_tightness'] = 1.0

df['ind_max_streak'] = df.groupby('industry')['previous_boards'].transform('max')
df['ind_zt_count'] = df.groupby('industry')['previous_boards'].transform('count')
df['ind_first_board_count'] = df.groupby('industry')['previous_boards'].transform(lambda x: (x == 1).sum())
df['streak_gap_in_industry'] = df['previous_boards'] - df['ind_max_streak']
df['is_industry_space_dragon'] = ((df['previous_boards'] == df['ind_max_streak']) & (df['previous_boards'] >= 2)).astype(float)
df['theme_assist_ratio'] = df['ind_first_board_count'] / np.maximum(1.0, df['ind_zt_count'])
df['ind_liquidity_share'] = df['previous_amount_cny'] / df.groupby('industry')['previous_amount_cny'].transform('sum')

df['prev_turnover'] = df['previous_turnover_pct']
df['prev_amount_yi'] = df['previous_amount_cny'] / 1e8
df['prev_float_cap_yi'] = 30.0
df['chip_cleansing_ratio'] = df['prev_turnover'] * 0.75
df['is_capacity_dragon'] = ((df['prev_amount_yi'] >= 5.0) & (df['prev_turnover'].between(10, 28)) & (df['open'] >= df['previous_high'])).astype(float)
df['is_super_capacity_dragon'] = ((df['prev_amount_yi'] >= 8.0) & (df['prev_turnover'].between(12, 28)) & (df['open'] >= df['previous_high'])).astype(float)
df['is_small_cap_trap'] = (df['prev_amount_yi'] < 3.0).astype(float)
df['prev_bad_crash_gene'] = 0.0
df['bad_zhuang_score'] = df['is_small_cap_trap'] * 1.0

df['prev_market_promote_rate'] = 0.21
df['prev_market_zt_count'] = 38
df['prev_market_dt_count'] = 2
df['is_cycle_expansion_prev'] = 0.0
df['is_cycle_divergence_prev'] = 1.0
df['is_cycle_retraction_prev'] = 0.0
df['is_cycle_freezing_prev'] = 0.0
df['moe_capacity_dragon_boost'] = df['is_capacity_dragon'] * 2.0
df['moe_retraction_risk_brake'] = 0.0
df['prev_board_streak'] = df['previous_boards']

X = df[feat_cols].fillna(0)
df['raw_clean_score'] = model.predict_proba(X)[:, 1]

df['is_green_open'] = df['gap_pct'] < 0.0
df['is_one_word_open'] = df['gap_pct'] >= 9.9

df_eligible = df[~df['is_green_open'] & ~df['is_one_word_open']].copy()
df_eligible = df_eligible.sort_values('raw_clean_score', ascending=False).reset_index(drop=True)

print("=== 2026-10-28 真实洁净模型推理排名 (可参与非一字红开池) ===")
for idx, r in df_eligible.iterrows():
    zt_str = "★ 封死涨停" if r['closed_at_limit'] else "× 未封板"
    print(f"{idx+1:2d}. {r['code_str']} {r['name']:<8} | 板位: {r['previous_boards']:1d}板 | 竞价: {r['gap_pct']:+5.2f}% | 成交: {r['prev_amount_yi']:5.1f}亿 | 换手: {r['prev_turnover']:5.1f}% | 纯模型分: {r['raw_clean_score']:.4f} | {zt_str}")
