#!/usr/bin/env python3
"""
build_and_train_5d_dragon_model.py

Unified 5D Breakthrough Quantitative Model (五维一体龙头破局量化模型):
Integrates Directions 1 through 5:
1. Direction 1: Microdynamics & Tail Order Momentum (微观时序动力学与尾盘加撤单盘口)
2. Direction 2: Theme Topology & Streak Gap (题材身位与拓扑博弈网络)
3. Direction 3: Chip Cleansing & Bad Zhuang Fingerprint (筹码全生命周期穿透与恶庄量化指纹)
4. Direction 4: Emotion Cycle State Machine & MoE Gate (周期状态机与混合专家条件门控)
5. Direction 5: Learning-to-Rank with Asymmetric Loss (排序学习与非对称防大面惩罚损失)

Fits on full daily dataset (2023-2026), performs 4-fold walk-forward cross-validation,
evaluates Top 1/2/3 selection accuracy, big-loss reduction, and produces empirical metrics.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import joblib

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
OUTPUT_DIR = ROOT / "research/model_upgrade_validation/unified_dragon_breakthrough_20261028"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"
INDUSTRY_PATH = ROOT / "research/first_board_github_industry_qualification_20261020/stock_basics_industry_original.csv"

def log_step(msg: str):
    print(f"\n{'='*75}\n[5D-DRAGON] {msg}\n{'='*75}")

def load_data_and_engineer_5d():
    log_step("Step 1: Ingesting Daily Bars and Industry Mapping")
    t0 = time.time()
    
    # 1. Load Industry Mapping
    ind_df = pd.read_csv(INDUSTRY_PATH, usecols=['symbol', 'industry'])
    ind_df['code_str'] = ind_df['symbol'].astype(str).str.zfill(6)
    ind_map = dict(zip(ind_df['code_str'], ind_df['industry']))
    print(f"Loaded {len(ind_map)} industry mappings in {time.time()-t0:.2f}s")
    
    # 2. Load Daily Bars
    cols = [
        'date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close',
        'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing'
    ]
    df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
    df = df[~df['st'] & ~df['new_listing']].copy()
    df['code_str'] = df['code'].astype(str).str.zfill(6)
    # Filter BJ/NEEQ
    df = df[~df['code_str'].str.startswith(('43', '83', '87', '92'))].copy()
    df['industry'] = df['code_str'].map(ind_map).fillna('其他行业')
    df['is_industry_unknown'] = (df['industry'] == '其他行业').astype(int)
    
    # Sort strictly by stock code and date
    df = df.sort_values(['code_str', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df):,} valid bars in {time.time()-t0:.2f}s")
    
    # 3. Limit Price & Board Streaks
    log_step("Step 2: Calculating Limit-Up Facts and Consecutive Board Streaks")
    is_20pct = df['code_str'].str.startswith(('300', '301', '688'))
    limit_mult = np.where(is_20pct, 1.195, 1.095)
    df['limit_price'] = (df['pre_close'] * limit_mult).round(2)
    df['limit_down_price'] = (df['pre_close'] * np.where(is_20pct, 0.805, 0.905)).round(2)
    
    df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
    df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)
    df['is_dt'] = (df['close'] <= df['limit_down_price'] + 0.001)
    
    # Intraday VWAP
    df['vwap'] = np.where(df['volume'] > 0, df['amount'] / (df['volume'] * 100.0), df['close'])
    
    # Board streak computation (fast numpy iteration)
    zt_arr = df['is_zt'].values
    codes = df['code_str'].values
    n = len(df)
    streak = np.zeros(n, dtype=int)
    curr_code = ""
    curr_streak = 0
    for i in range(n):
        if codes[i] != curr_code:
            curr_code = codes[i]
            curr_streak = 1 if zt_arr[i] else 0
        else:
            curr_streak = curr_streak + 1 if zt_arr[i] else 0
        streak[i] = curr_streak
    df['board_streak'] = streak
    
    # Bad history: historical count of 3+ boards immediately followed by limit-down/big crash in past 250 days
    print("Computing bad history (bad zhuang gene: 3板必核前科)...")
    dt_arr = df['is_dt'].values
    is_crash = np.zeros(n, dtype=int)
    for i in range(1, n):
        if codes[i] == codes[i-1] and streak[i-1] >= 3 and (dt_arr[i] or (df['close'].values[i]/df['pre_close'].values[i] <= 0.91)):
            is_crash[i-1] = 1
    df['is_severe_crash_after_3b'] = is_crash
    df['bad_crash_history_250d'] = df.groupby('code_str')['is_severe_crash_after_3b'].transform(lambda x: x.rolling(250, min_periods=1).sum())
    
    # Moving Averages
    df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
    
    # 4. Lag Features (Shift by 1 to get T-1 state for Day T Call Auction)
    log_step("Step 3: Generating T-1 Lag Features and 5D Feature Dimensions")
    grp = df.groupby('code_str')
    df['prev_date'] = grp['date'].shift(1)
    df['prev_close'] = grp['close'].shift(1)
    df['prev_open'] = grp['open'].shift(1)
    df['prev_high'] = grp['high'].shift(1)
    df['prev_low'] = grp['low'].shift(1)
    df['prev_amount'] = grp['amount'].shift(1)
    df['prev_volume'] = grp['volume'].shift(1)
    df['prev_turnover'] = grp['turnover'].shift(1)
    df['prev_vwap'] = grp['vwap'].shift(1)
    df['prev_is_zt'] = grp['is_zt'].shift(1).fillna(False)
    df['prev_is_touch_zt'] = grp['is_touch_zt'].shift(1).fillna(False)
    df['prev_board_streak'] = grp['board_streak'].shift(1).fillna(0).astype(int)
    df['prev_float_cap'] = grp['float_cap'].shift(1)
    df['prev_ma5'] = grp['ma5'].shift(1)
    df['prev_ma10'] = grp['ma10'].shift(1)
    df['prev_ma20'] = grp['ma20'].shift(1)
    df['prev_bad_crash_count'] = grp['bad_crash_history_250d'].shift(1).fillna(0)
    
    df['next_open'] = grp['open'].shift(-1)
    df['next_close'] = grp['close'].shift(-1)
    
    # Drop rows without yesterday's facts
    df = df[df['prev_close'].notnull()].copy()
    
    # -------------------------------------------------------------------------
    # DIRECTION 1: Microdynamics & Tail Order Momentum (微观时序动力学与尾盘加撤单盘口)
    # -------------------------------------------------------------------------
    print("Engineering Direction 1: Microdynamics & Tail Order Momentum...")
    # T-1 Intraday Seal & Amplitude
    df['prev_amplitude_pct'] = np.where(df['prev_close'] > 0, (df['prev_high'] - df['prev_low']) / df['prev_close'] * 100.0, 0.0)
    # Tail seal momentum proxy (Seal_Delta proxy):
    # Tight seal with small amplitude and high volume absorption -> strong tail addition (加单锁仓)
    # Late rebound / wide amplitude / low volume -> afternoon wobble / fake seal (偷袭回封 / 尾盘撤单)
    df['is_tight_morning_seal'] = (df['prev_amplitude_pct'] <= 3.5) & (df['prev_is_zt'])
    df['is_tail_sneaky_seal'] = (df['prev_amplitude_pct'] >= 6.5) & (df['prev_is_zt'])
    
    df['tail_seal_momentum'] = np.where(
        df['is_tight_morning_seal'], 1.5,
        np.where(df['is_tail_sneaky_seal'], 0.5, 1.0)
    )
    
    # Day T 09:25 Call Auction features
    df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['open_gap_pct'] = df['open_gap'] * 100.0
    df['open_vs_prev_high'] = (df['open'] - df['prev_high']) / df['prev_high']
    df['open_vs_prev_vwap'] = (df['open'] - df['prev_vwap']) / df['prev_vwap']
    df['open_vs_prev_open'] = (df['open'] - df['prev_open']) / df['prev_open']
    
    # Golden auction pattern (+2%~+7% with price > prev_vwap and price >= prev_high)
    df['is_golden_auction'] = (
        (df['open_gap'] >= 0.02) & 
        (df['open_gap'] <= 0.07) & 
        (df['open_vs_prev_vwap'] >= 0.0) & 
        (df['open_vs_prev_high'] >= -0.001)
    ).astype(float)
    
    # Fake auction decay risk (high open but below yesterday VWAP, or high open with small volume)
    df['fake_open_risk'] = (
        (df['open_gap'] >= 0.04) & 
        ((df['open_vs_prev_vwap'] < 0.0) | (df['prev_amount'] < 2.5e8))
    ).astype(float)
    
    # Non-one-word open filter (can actually be bought by retail)
    df['is_one_word'] = (df['open'] >= df['limit_price'] - 0.001)
    df['is_non_one_word'] = ~df['is_one_word']
    
    # -------------------------------------------------------------------------
    # DIRECTION 2: Theme Topology & Streak Gap (题材身位与拓扑博弈网络)
    # -------------------------------------------------------------------------
    print("Engineering Direction 2: Theme Topology & Streak Gap...")
    # Group by (date, industry) to compute theme ladder & relative position
    ind_day = df.groupby(['date', 'industry'])
    
    # Industry max streak yesterday
    df['ind_max_streak'] = ind_day['prev_board_streak'].transform('max')
    df['ind_zt_count'] = ind_day['prev_is_zt'].transform('sum')
    df['ind_first_board_count'] = ind_day['prev_board_streak'].transform(lambda x: (x == 1).sum())
    df['ind_total_amount'] = ind_day['prev_amount'].transform('sum')
    
    # Relative Position: Streak Gap within Industry
    df['streak_gap_in_industry'] = df['prev_board_streak'] - df['ind_max_streak']
    df['is_industry_space_dragon'] = (
        (df['prev_board_streak'] == df['ind_max_streak']) & 
        (df['prev_board_streak'] >= 2) & 
        (df['prev_is_zt'])
    ).astype(float)
    
    # Theme Assist Strength (首板助攻扩散比)
    df['theme_assist_ratio'] = df['ind_first_board_count'] / np.maximum(1.0, df['ind_zt_count'])
    # Liquidity share within industry (题材资金抽血/吸附能力)
    df['ind_liquidity_share'] = np.where(df['ind_total_amount'] > 0, df['prev_amount'] / df['ind_total_amount'], 0.0)
    
    # -------------------------------------------------------------------------
    # DIRECTION 3: Chip Cleansing & Bad Zhuang Fingerprint (筹码全生命周期穿透与恶庄指纹)
    # -------------------------------------------------------------------------
    print("Engineering Direction 3: Chip Cleansing & Bad Zhuang Fingerprints...")
    # Chip cleansing ratio: Turnover * position of VWAP in range
    range_span = np.maximum(0.01, df['prev_high'] - df['prev_low'])
    df['vwap_relative_position'] = (df['prev_vwap'] - df['prev_low']) / range_span
    df['chip_cleansing_ratio'] = df['prev_turnover'] * df['vwap_relative_position']
    
    # Cost support at open: percentage above yesterday's VWAP
    df['vwap_cost_support'] = df['open_vs_prev_vwap']
    
    # Large Capacity Dragon Features (襄阳轴承型)
    df['prev_amount_yi'] = df['prev_amount'] / 1e8
    df['float_cap_yi'] = df['float_cap'] / 1e8
    df['is_capacity_dragon'] = (
        (df['prev_amount_yi'] >= 5.0) & 
        (df['prev_turnover'] >= 10.0) & 
        (df['prev_turnover'] <= 28.0) & 
        (df['open_vs_prev_vwap'] >= 0.0)
    ).astype(float)
    
    df['is_super_capacity_dragon'] = (
        (df['prev_amount_yi'] >= 8.0) & 
        (df['prev_turnover'] >= 12.0) & 
        (df['prev_turnover'] <= 28.0) & 
        (df['open_vs_prev_vwap'] >= 0.0)
    ).astype(float)
    
    # Bad Zhuang Fingerprint Score (正裕工业型)
    df['is_small_cap_trap'] = ((df['float_cap_yi'] < 30.0) & (df['prev_amount_yi'] < 3.0)).astype(float)
    df['has_bad_crash_gene'] = (df['prev_bad_crash_count'] >= 1.0).astype(float)
    
    # Bad Zhuang Comprehensive Penalty
    df['bad_zhuang_score'] = (
        df['is_small_cap_trap'] * 1.0 + 
        df['has_bad_crash_gene'] * 2.5 + 
        df['is_tail_sneaky_seal'].astype(float) * 1.5 + 
        df['fake_open_risk'] * 2.0
    )
    
    # -------------------------------------------------------------------------
    # DIRECTION 4: Emotion Cycle State Machine & MoE Gate (周期状态机与混合专家门控)
    # -------------------------------------------------------------------------
    print("Engineering Direction 4: Market Emotion Cycle State Machine & MoE Gate...")
    day_grp = df.groupby('date')
    
    # Market-wide daily stats
    df['mkt_total_zt'] = day_grp['prev_is_zt'].transform('sum')
    df['mkt_total_dt'] = day_grp['is_dt'].transform('sum')
    
    # Promotion rate of yesterday's limit-up stocks
    zt_yesterday_mask = df['prev_is_zt']
    promote_success = df['is_zt'] & zt_yesterday_mask
    
    day_zt_prev = df.loc[zt_yesterday_mask].groupby('date')['code_str'].count()
    day_zt_prom = df.loc[promote_success].groupby('date')['code_str'].count()
    day_promote_rate = (day_zt_prom / np.maximum(1, day_zt_prev)).to_dict()
    df['mkt_promote_rate'] = df['date'].map(day_promote_rate).fillna(0.18)
    
    # Cycle classification:
    # 0: 冰点期 (Freezing: low promote rate < 15%, low total ZT)
    # 1: 主升期 (Expansion: promote rate >= 28%, ZT >= 35, high optimism)
    # 2: 分歧换手期 (Divergence: active turnover, moderate promote rate 15%~28%)
    # 3: 退潮核按钮期 (Retraction: high limit-downs, failing streaks)
    cycle_regime = np.where(
        df['mkt_total_dt'] >= 8, 3, # 退潮
        np.where(
            (df['mkt_promote_rate'] >= 0.28) & (df['mkt_total_zt'] >= 35), 1, # 主升
            np.where(
                (df['mkt_promote_rate'] < 0.15) & (df['mkt_total_zt'] < 25), 0, # 冰点
                2 # 分歧换手
            )
        )
    )
    df['cycle_regime'] = cycle_regime
    df['is_cycle_expansion'] = (cycle_regime == 1).astype(float)
    df['is_cycle_divergence'] = (cycle_regime == 2).astype(float)
    df['is_cycle_retraction'] = (cycle_regime == 3).astype(float)
    df['is_cycle_freezing'] = (cycle_regime == 0).astype(float)
    
    # MoE Gate Interactions:
    # In Expansion / Divergence -> Boost Capacity Dragon & waive turnover penalty!
    df['moe_capacity_dragon_boost'] = df['is_capacity_dragon'] * np.where((cycle_regime == 1) | (cycle_regime == 2), 2.0, 0.5)
    # In Retraction -> Double Bad Zhuang Penalty & penalize high streaks!
    df['moe_retraction_risk_brake'] = np.where(cycle_regime == 3, df['bad_zhuang_score'] * 2.0 + (df['prev_board_streak'] >= 3).astype(float) * 1.5, 0.0)
    
    # -------------------------------------------------------------------------
    # DIRECTION 5: Target Labels & Asymmetric Loss Weighting (目标与非对称权重)
    # -------------------------------------------------------------------------
    print("Engineering Direction 5: Target Labels & Asymmetric Loss Weighting...")
    df['target_limit_up'] = df['is_zt'].astype(int)
    df['target_touch_limit'] = df['is_touch_zt'].astype(int)
    df['target_day_return'] = (df['close'] - df['open']) / df['open']
    df['target_bomb'] = ((df['target_touch_limit'] == 1) & (df['target_limit_up'] == 0) & ((df['high'] - df['close']) / df['open'] >= 0.04)).astype(int)
    df['target_big_loss'] = (df['target_day_return'] <= -0.04).astype(int)
    df['target_next_open_return'] = np.where(df['next_open'].notnull(), (df['next_open'] - df['close']) / df['close'], 0.0)
    
    # Asymmetric Sample Weighting for Training:
    # Big Loss / Bomb Trap: 4.0x penalty weight
    # Real Dragon (Limit up + positive next open): 2.5x positive reward
    # Normal samples: 1.0x
    sample_weights = np.ones(len(df), dtype=float)
    sample_weights = np.where(df['target_big_loss'] == 1, 4.0, sample_weights)
    sample_weights = np.where(df['target_bomb'] == 1, 3.5, sample_weights)
    sample_weights = np.where((df['target_limit_up'] == 1) & (df['target_next_open_return'] > 0), 2.5, sample_weights)
    df['asymmetric_sample_weight'] = sample_weights
    
    print(f"Data Engineering finished: {len(df):,} total rows.")
    return df

def run_5d_walk_forward_evaluation(df: pd.DataFrame):
    log_step("Step 4: Filtering Eligible Non-One-Word Relay Candidate Pool")
    # Candidate pool:
    # 1. Non-one-word (open < limit_price - 0.01)
    # 2. Yesterday was limit-up (prev_is_zt == True, streak >= 1)
    # 3. Not negative open (open >= pre_close), strictly respecting the green-open cancellation rule!
    candidates = df[
        df['prev_is_zt'] & 
        df['is_non_one_word'] & 
        (df['open_gap'] >= 0.0)
    ].copy().reset_index(drop=True)
    
    print(f"Total eligible non-one-word candidate rows: {len(candidates):,}")
    print(f"Base limit-up win rate in pool: {candidates['target_limit_up'].mean()*100:.2f}%")
    print(f"Base touch limit-up rate: {candidates['target_touch_limit'].mean()*100:.2f}%")
    print(f"Base big loss rate (<= -4%): {candidates['target_big_loss'].mean()*100:.2f}%")
    
    # Feature list incorporating all 5 directions
    feature_cols = [
        # Direction 1: Microdynamics & Tail Order Momentum
        'tail_seal_momentum', 'open_gap', 'open_gap_pct', 'open_vs_prev_high',
        'open_vs_prev_vwap', 'open_vs_prev_open', 'is_golden_auction', 'fake_open_risk',
        # Direction 2: Theme Topology & Streak Gap
        'ind_max_streak', 'ind_zt_count', 'ind_first_board_count', 'streak_gap_in_industry',
        'is_industry_space_dragon', 'theme_assist_ratio', 'ind_liquidity_share',
        # Direction 3: Chip Cleansing & Bad Zhuang Fingerprint
        'prev_turnover', 'prev_amount_yi', 'float_cap_yi', 'chip_cleansing_ratio',
        'vwap_cost_support', 'is_capacity_dragon', 'is_super_capacity_dragon',
        'is_small_cap_trap', 'has_bad_crash_gene', 'bad_zhuang_score',
        # Direction 4: Emotion Cycle State Machine & MoE Gate
        'mkt_promote_rate', 'is_cycle_expansion', 'is_cycle_divergence',
        'is_cycle_retraction', 'is_cycle_freezing',
        'moe_capacity_dragon_boost', 'moe_retraction_risk_brake',
        # Technical Baseline & Capacity
        'prev_board_streak', 'prev_amplitude_pct'
    ]
    
    print(f"\nTotal 5D Engineered Feature Count: {len(feature_cols)}")
    
    # 4-Fold Walk-Forward Temporal Splits
    folds = [
        ("Fold 1", "2024-01-01", "2024-08-31", "2024-09-01", "2025-01-31"),
        ("Fold 2", "2024-06-01", "2025-01-31", "2025-02-01", "2025-07-31"),
        ("Fold 3", "2025-01-01", "2025-07-31", "2025-08-01", "2026-01-31"),
        ("Fold 4", "2025-06-01", "2026-01-31", "2026-02-01", "2026-09-16"),
    ]
    
    oof_predictions = []
    models = []
    
    log_step("Step 5: Training 5D HistGradientBoostingClassifier with Asymmetric Loss")
    for fold_name, train_start, train_end, test_start, test_end in folds:
        train_mask = (candidates['date'] >= train_start) & (candidates['date'] <= train_end)
        test_mask = (candidates['date'] >= test_start) & (candidates['date'] <= test_end)
        
        X_train = candidates.loc[train_mask, feature_cols].fillna(0)
        y_train = candidates.loc[train_mask, 'target_limit_up']
        w_train = candidates.loc[train_mask, 'asymmetric_sample_weight']
        
        X_test = candidates.loc[test_mask, feature_cols].fillna(0)
        y_test = candidates.loc[test_mask, 'target_limit_up']
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42
        )
        clf.fit(X_train, y_train, sample_weight=w_train)
        
        test_preds = clf.predict_proba(X_test)[:, 1]
        
        sub = candidates.loc[test_mask, ['date', 'code_str', 'name', 'open_gap_pct', 'prev_board_streak', 'prev_amount_yi', 'target_limit_up', 'target_touch_limit', 'target_day_return', 'target_big_loss', 'target_bomb']].copy()
        sub['pred_score'] = test_preds
        sub['fold'] = fold_name
        oof_predictions.append(sub)
        models.append(clf)
        
        auc = roc_auc_score(y_test, test_preds)
        prauc = average_precision_score(y_test, test_preds)
        print(f"[{fold_name}] Train: {len(X_train):,}, Test: {len(X_test):,} | Test AUC: {auc:.4f}, PR-AUC: {prauc:.4f} (Pos Rate: {y_test.mean()*100:.1f}%)")
        
    oof_df = pd.concat(oof_predictions, ignore_index=True)
    
    # -------------------------------------------------------------------------
    # Evaluation: Daily Top 1, Top 2, Top 3 Performance
    # -------------------------------------------------------------------------
    log_step("Step 6: Ranking Learning Evaluation: Daily Top 1, Top 2, Top 3 Metrics")
    
    # Sort within each date by pred_score descending
    oof_df['daily_rank'] = oof_df.groupby('date')['pred_score'].rank(ascending=False, method='first')
    
    top1 = oof_df[oof_df['daily_rank'] == 1]
    top2 = oof_df[oof_df['daily_rank'] <= 2]
    top3 = oof_df[oof_df['daily_rank'] <= 3]
    top5 = oof_df[oof_df['daily_rank'] <= 5]
    
    # Big loss rate & limit-up hit rate by rank
    metrics = {
        "all_candidates": {
            "count": len(oof_df),
            "limit_up_rate": round(oof_df['target_limit_up'].mean() * 100, 2),
            "touch_rate": round(oof_df['target_touch_limit'].mean() * 100, 2),
            "big_loss_rate": round(oof_df['target_big_loss'].mean() * 100, 2),
            "avg_day_return": round(oof_df['target_day_return'].mean() * 100, 2)
        },
        "top1": {
            "count": len(top1),
            "limit_up_rate": round(top1['target_limit_up'].mean() * 100, 2),
            "touch_rate": round(top1['target_touch_limit'].mean() * 100, 2),
            "big_loss_rate": round(top1['target_big_loss'].mean() * 100, 2),
            "avg_day_return": round(top1['target_day_return'].mean() * 100, 2)
        },
        "top2": {
            "count": len(top2),
            "limit_up_rate": round(top2['target_limit_up'].mean() * 100, 2),
            "touch_rate": round(top2['target_touch_limit'].mean() * 100, 2),
            "big_loss_rate": round(top2['target_big_loss'].mean() * 100, 2),
            "avg_day_return": round(top2['target_day_return'].mean() * 100, 2)
        },
        "top3": {
            "count": len(top3),
            "limit_up_rate": round(top3['target_limit_up'].mean() * 100, 2),
            "touch_rate": round(top3['target_touch_limit'].mean() * 100, 2),
            "big_loss_rate": round(top3['target_big_loss'].mean() * 100, 2),
            "avg_day_return": round(top3['target_day_return'].mean() * 100, 2)
        },
        "top5": {
            "count": len(top5),
            "limit_up_rate": round(top5['target_limit_up'].mean() * 100, 2),
            "touch_rate": round(top5['target_touch_limit'].mean() * 100, 2),
            "big_loss_rate": round(top5['target_big_loss'].mean() * 100, 2),
            "avg_day_return": round(top5['target_day_return'].mean() * 100, 2)
        }
    }
    
    print("\n--- Daily Ranking Metrics Table ---")
    print(f"Pool Baseline   | Sample: {metrics['all_candidates']['count']:5d} | Limit-Up Rate: {metrics['all_candidates']['limit_up_rate']:5.2f}% | Touch Rate: {metrics['all_candidates']['touch_rate']:5.2f}% | Big Loss Rate: {metrics['all_candidates']['big_loss_rate']:5.2f}% | Day Ret: {metrics['all_candidates']['avg_day_return']:+5.2f}%")
    print(f"Top 1 Selected  | Sample: {metrics['top1']['count']:5d} | Limit-Up Rate: {metrics['top1']['limit_up_rate']:5.2f}% | Touch Rate: {metrics['top1']['touch_rate']:5.2f}% | Big Loss Rate: {metrics['top1']['big_loss_rate']:5.2f}% | Day Ret: {metrics['top1']['avg_day_return']:+5.2f}%")
    print(f"Top 2 Selected  | Sample: {metrics['top2']['count']:5d} | Limit-Up Rate: {metrics['top2']['limit_up_rate']:5.2f}% | Touch Rate: {metrics['top2']['touch_rate']:5.2f}% | Big Loss Rate: {metrics['top2']['big_loss_rate']:5.2f}% | Day Ret: {metrics['top2']['avg_day_return']:+5.2f}%")
    print(f"Top 3 Selected  | Sample: {metrics['top3']['count']:5d} | Limit-Up Rate: {metrics['top3']['limit_up_rate']:5.2f}% | Touch Rate: {metrics['top3']['touch_rate']:5.2f}% | Big Loss Rate: {metrics['top3']['big_loss_rate']:5.2f}% | Day Ret: {metrics['top3']['avg_day_return']:+5.2f}%")
    print(f"Top 5 Selected  | Sample: {metrics['top5']['count']:5d} | Limit-Up Rate: {metrics['top5']['limit_up_rate']:5.2f}% | Touch Rate: {metrics['top5']['touch_rate']:5.2f}% | Big Loss Rate: {metrics['top5']['big_loss_rate']:5.2f}% | Day Ret: {metrics['top5']['avg_day_return']:+5.2f}%")
    
    # -------------------------------------------------------------------------
    # Feature Importance via Random Forest
    # -------------------------------------------------------------------------
    log_step("Step 7: Computing 5D Feature Importance")
    rf = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
    sample_sub = candidates.sample(min(len(candidates), 15000), random_state=42)
    rf.fit(sample_sub[feature_cols].fillna(0), sample_sub['target_limit_up'], sample_weight=sample_sub['asymmetric_sample_weight'])
    
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\n--- Top 15 Most Important 5D Features ---")
    for idx, row in fi_df.head(15).iterrows():
        print(f"{idx+1:2d}. {row['feature']:<30} : {row['importance']:.4f}")
    
    fi_df.to_csv(OUTPUT_DIR / "feature_importances_5d.csv", index=False, encoding='utf-8-sig')
    
    # -------------------------------------------------------------------------
    # Retrain Full Model and Persist Artifacts
    # -------------------------------------------------------------------------
    log_step("Step 8: Persisting Full Production Model and Diagnostic Files")
    full_clf = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42
    )
    full_clf.fit(candidates[feature_cols].fillna(0), candidates['target_limit_up'], sample_weight=candidates['asymmetric_sample_weight'])
    joblib.dump(full_clf, OUTPUT_DIR / "model_5d_dragon_breakthrough.joblib")
    
    # Save OOF predictions
    oof_df.to_csv(OUTPUT_DIR / "oof_5d_predictions.csv", index=False, encoding='utf-8-sig')
    
    # Save metrics json
    with open(OUTPUT_DIR / "unified_5d_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    # Save feature list
    with open(OUTPUT_DIR / "feature_list_5d.json", "w", encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        
    print(f"\nAll artifacts successfully saved to {OUTPUT_DIR}")
    return metrics, fi_df, oof_df

if __name__ == "__main__":
    df = load_data_and_engineer_5d()
    metrics, fi_df, oof_df = run_5d_walk_forward_evaluation(df)
