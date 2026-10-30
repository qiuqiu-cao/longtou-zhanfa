#!/usr/bin/env python3
"""
build_and_train_auction_models.py

Comprehensive Training and Quantitative Analysis of Call Auction (竞价) Limit-Up Predictability:
1. Focuses strictly on NON-ONE-WORD (非一字) call auction patterns (as explicitly requested: "不只是一字板放量的").
2. Investigates:
   - Cohort 1: 连板接力非一字竞价模型 (1进2, 2进3, 3进4, 4进5, 高位连板)
   - Cohort 2: 首板起爆与低位突破非一字竞价模型 (首板起爆)
   - Cohort 3: 大容量换手真龙竞价模型 (大成交额、健康换手、真龙超预期)
3. Fits HistGradientBoostingClassifiers and interpretable DecisionTrees to extract optimal thresholds for Tonghuashun.
4. Generates empirical win-rate tables, feature importances, and validated formulas.
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
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import joblib

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
OUTPUT_DIR = ROOT / "research/model_upgrade_validation/auction_dragon_model_20261028"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"

def print_step(msg: str):
    print(f"\n{'='*70}\n[STEP] {msg}\n{'='*70}")

def build_auction_dataset():
    print_step("1. Loading Daily Bars and Filtering Universe")
    t0 = time.time()
    cols = [
        'date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close',
        'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing'
    ]
    df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
    print(f"Loaded {len(df):,} raw bars in {time.time() - t0:.2f}s")
    
    # Filter ST and new listings
    df = df[~df['st'] & ~df['new_listing']].copy()
    df['code_str'] = df['code'].astype(str).str.zfill(6)
    
    # Exclude BJ/NEEQ (keep mainboard and ChiNext/STAR)
    df = df[~df['code_str'].str.startswith(('43', '83', '87', '92'))].copy()
    print(f"Valid non-ST bars: {len(df):,}")
    
    # Sort strictly by stock code and date
    df = df.sort_values(['code_str', 'date']).reset_index(drop=True)
    
    print_step("2. Calculating Limit Prices and Daily Indicators")
    is_20pct = df['code_str'].str.startswith(('300', '301', '688'))
    limit_mult = np.where(is_20pct, 1.195, 1.095)
    df['limit_price'] = (df['pre_close'] * limit_mult).round(2)
    
    # Limit up definition: close reached limit price and close == high
    df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
    df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)
    
    # VWAP of the day
    df['vwap'] = np.where(df['volume'] > 0, df['amount'] / (df['volume'] * 100.0), df['close'])
    
    # Moving averages
    print("Computing moving averages...")
    df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
    
    # Consecutive ZT count (board streak)
    print("Computing consecutive board streaks...")
    # Group by code and calculate cumulative consecutive limit ups
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
            if zt_arr[i]:
                curr_streak += 1
            else:
                curr_streak = 0
        streak[i] = curr_streak
    df['board_streak'] = streak
    
    print_step("3. Generating Yesterday Lag Features and Today 09:25 Auction Metrics")
    # Shift features by 1 day within each stock to obtain YESTERDAY'S facts
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
    df['prev_board_streak'] = grp['board_streak'].shift(1).fillna(0).astype(int)
    df['prev_amplitude'] = grp['high'].shift(1) - grp['low'].shift(1)
    df['prev_amplitude_pct'] = np.where(df['prev_close'] > 0, df['prev_amplitude'] / grp['pre_close'].shift(1) * 100.0, 0.0)
    
    df['prev_ma5'] = grp['ma5'].shift(1)
    df['prev_ma10'] = grp['ma10'].shift(1)
    df['prev_ma20'] = grp['ma20'].shift(1)
    
    # Next day open for T+1 premium calculation
    df['next_open'] = grp['open'].shift(-1)
    df['next_date'] = grp['date'].shift(-1)
    
    # Drop rows without yesterday's data
    df = df[df['prev_close'].notnull()].copy()
    
    # --- AUCTION FEATURES (Day t, 09:25 Call Auction) ---
    df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['open_gap_pct'] = df['open_gap'] * 100.0
    
    # Breakout features
    df['open_vs_prev_high'] = (df['open'] - df['prev_high']) / df['prev_high']
    df['open_vs_prev_vwap'] = (df['open'] - df['prev_vwap']) / df['prev_vwap']
    df['open_vs_prev_open'] = (df['open'] - df['prev_open']) / df['prev_open']
    
    # Moving average breakout
    df['open_vs_ma5'] = np.where(df['prev_ma5'] > 0, (df['open'] - df['prev_ma5']) / df['prev_ma5'], 0.0)
    df['open_vs_ma20'] = np.where(df['prev_ma20'] > 0, (df['open'] - df['prev_ma20']) / df['prev_ma20'], 0.0)
    df['is_ma_bull'] = (df['prev_ma5'] > df['prev_ma10']) & (df['prev_ma10'] > df['prev_ma20'])
    
    # Non-one-word open: open price is LESS than limit price (can actually be bought!)
    df['is_one_word'] = (df['open'] >= df['limit_price'] - 0.001)
    df['is_non_one_word'] = ~df['is_one_word']
    
    # Capacity dragon indicators (襄阳轴承型 vs 正裕工业型)
    df['prev_amount_yi'] = df['prev_amount'] / 1e8
    df['float_cap_yi'] = df['float_cap'] / 1e8
    df['is_capacity_volume'] = (df['prev_amount_yi'] >= 5.0)
    df['is_super_capacity'] = (df['prev_amount_yi'] >= 8.0)
    df['is_healthy_turnover'] = (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 28.0)
    df['is_capacity_dragon'] = df['is_capacity_volume'] & df['is_healthy_turnover']
    df['is_small_cap_trap'] = (df['float_cap_yi'] < 30.0) & (df['prev_amount_yi'] < 3.0)
    
    # Price bins
    df['is_low_price'] = (df['pre_close'] < 12.0)
    
    # --- TARGET LABELS (Day t ground truth) ---
    # 1. Closed at limit-up (涨停封死)
    df['target_limit_up'] = df['is_zt'].astype(int)
    # 2. Touched limit-up during the day (盘中触板)
    df['target_touch_limit'] = df['is_touch_zt'].astype(int)
    # 3. Day holding return: buy at 09:25 open, hold to 15:00 close
    df['target_day_return'] = (df['close'] - df['open']) / df['open']
    # 4. Max intraday return: buy at 09:25 open, max potential profit
    df['target_max_return'] = (df['high'] - df['open']) / df['open']
    # 5. Bombing board trap (炸板大面): touched limit-up, closed not limit-up, and fell >= 4% from high
    df['target_bomb'] = (
        (df['target_touch_limit'] == 1) & 
        (df['target_limit_up'] == 0) & 
        ((df['high'] - df['close']) / df['open'] >= 0.04)
    ).astype(int)
    # 6. Big intraday loss (吃大面 <= -4%)
    df['target_big_loss'] = (df['target_day_return'] <= -0.04).astype(int)
    # 7. Next day open return
    df['target_next_open_return'] = np.where(df['next_open'].notnull(), (df['next_open'] - df['close']) / df['close'], np.nan)
    
    print(f"Dataset preparation complete: {len(df):,} rows.")
    return df

def analyze_and_train(df: pd.DataFrame):
    print_step("4. Empirical Statistical Distribution by Auction Gap Bins")
    
    # Define auction gap bins
    bins = [-0.30, -0.04, 0.0, 0.02, 0.04, 0.06, 0.08, 0.095, 0.30]
    labels = [
        '1.深核跌停 (<-4%)',
        '2.绿盘低开 (-4%~0%)',
        '3.平开微红 (0%~+2%)',
        '4.温和高开 (+2%~+4%)',
        '5.黄金抢筹 (+4%~+6%)',
        '6.强势进攻 (+6%~+8%)',
        '7.极强分歧 (+8%~+9.5%)',
        '8.顶格一字 (>=+9.5%)'
    ]
    df['gap_bin'] = pd.cut(df['open_gap'], bins=bins, labels=labels)
    
    # --------------------------------------------------------------------------
    # Sub-Analysis A: 连板接力候选池 (Relay Candidates: prev_board_streak >= 1)
    # --------------------------------------------------------------------------
    relay_df = df[df['prev_is_zt']].copy()
    print(f"\n--- Total Relay Candidates: {len(relay_df):,} ---")
    
    # Group by gap bin for relay candidates
    relay_stats = relay_df.groupby('gap_bin', observed=False).agg(
        count=('code_str', 'count'),
        limit_up_rate=('target_limit_up', 'mean'),
        touch_limit_rate=('target_touch_limit', 'mean'),
        bomb_rate=('target_bomb', 'mean'),
        big_loss_rate=('target_big_loss', 'mean'),
        avg_day_return=('target_day_return', 'mean'),
        avg_max_return=('target_max_return', 'mean'),
        avg_next_open=('target_next_open_return', 'mean')
    ).reset_index()
    
    print("\n[连板接力个股：各竞价高开区间实证表现]")
    print(relay_stats.to_string(index=False))
    relay_stats.to_csv(OUTPUT_DIR / "relay_auction_gap_statistics.csv", index=False, encoding='utf-8-sig')
    
    # --------------------------------------------------------------------------
    # Sub-Analysis B: NON-ONE-WORD Relay Candidates (非一字连板接力)
    # --------------------------------------------------------------------------
    relay_non_one = relay_df[relay_df['is_non_one_word']].copy()
    print(f"\n--- Non-One-Word Relay Candidates: {len(relay_non_one):,} (占比 {len(relay_non_one)/len(relay_df)*100:.1f}%) ---")
    
    # By Board Height (1进2, 2进3, 3进4, 4进5, 5+)
    relay_non_one['streak_bin'] = np.where(
        relay_non_one['prev_board_streak'] == 1, '1进2',
        np.where(relay_non_one['prev_board_streak'] == 2, '2进3',
        np.where(relay_non_one['prev_board_streak'] == 3, '3进4',
        np.where(relay_non_one['prev_board_streak'] == 4, '4进5', '5板以上高位龙')))
    )
    streak_stats = relay_non_one.groupby('streak_bin').agg(
        count=('code_str', 'count'),
        limit_up_rate=('target_limit_up', 'mean'),
        touch_limit_rate=('target_touch_limit', 'mean'),
        bomb_rate=('target_bomb', 'mean'),
        avg_day_return=('target_day_return', 'mean')
    ).reset_index()
    print("\n[非一字连板接力：不同板位晋级率]")
    print(streak_stats.to_string(index=False))
    streak_stats.to_csv(OUTPUT_DIR / "streak_height_statistics.csv", index=False, encoding='utf-8-sig')
    
    # --------------------------------------------------------------------------
    # Sub-Analysis C: 大容量换手真龙 vs 小市值恶庄对比 (襄阳轴承 vs 正裕工业)
    # --------------------------------------------------------------------------
    dragon_stats = relay_non_one.groupby('is_capacity_dragon').agg(
        count=('code_str', 'count'),
        limit_up_rate=('target_limit_up', 'mean'),
        touch_limit_rate=('target_touch_limit', 'mean'),
        bomb_rate=('target_bomb', 'mean'),
        big_loss_rate=('target_big_loss', 'mean'),
        avg_day_return=('target_day_return', 'mean')
    ).reset_index()
    dragon_stats['type'] = np.where(dragon_stats['is_capacity_dragon'], '大容量换手真龙(额>=5亿,换手10~28%)', '常规/小容量/缩量')
    print("\n[非一字接力：大容量换手真龙 vs 常规个股]")
    print(dragon_stats[['type', 'count', 'limit_up_rate', 'bomb_rate', 'avg_day_return']].to_string(index=False))
    dragon_stats.to_csv(OUTPUT_DIR / "capacity_dragon_statistics.csv", index=False, encoding='utf-8-sig')
    
    # --------------------------------------------------------------------------
    # Sub-Analysis D: 首板突破候选池 (First Board Breakout: prev_is_zt == False)
    # --------------------------------------------------------------------------
    # Filter non-ZT stocks opening strong (open_gap >= 0.02, non-one-word)
    first_df = df[~df['prev_is_zt'] & (df['open_gap'] >= 0.01) & df['is_non_one_word']].copy()
    print(f"\n--- First Board Strong Open Candidates (open >= +1%): {len(first_df):,} ---")
    first_stats = first_df.groupby('gap_bin', observed=False).agg(
        count=('code_str', 'count'),
        limit_up_rate=('target_limit_up', 'mean'),
        touch_limit_rate=('target_touch_limit', 'mean'),
        bomb_rate=('target_bomb', 'mean'),
        avg_day_return=('target_day_return', 'mean')
    ).reset_index()
    print("\n[首板起爆候选：各竞价高开区间表现]")
    print(first_stats.to_string(index=False))
    first_stats.to_csv(OUTPUT_DIR / "first_board_gap_statistics.csv", index=False, encoding='utf-8-sig')
    
    # --------------------------------------------------------------------------
    # STEP 5: Machine Learning Training (Non-One-Word Relay Dragon Model)
    # --------------------------------------------------------------------------
    print_step("5. Machine Learning Training: Non-One-Word Relay Dragon Model")
    
    # Feature columns available strictly at 09:25
    feature_cols = [
        'open_gap',
        'open_vs_prev_high',
        'open_vs_prev_vwap',
        'open_vs_prev_open',
        'open_vs_ma5',
        'open_vs_ma20',
        'prev_board_streak',
        'prev_turnover',
        'prev_amount_yi',
        'prev_amplitude_pct',
        'float_cap_yi',
        'pre_close',
        'is_capacity_dragon',
        'is_small_cap_trap',
        'is_low_price',
        'is_ma_bull'
    ]
    
    train_data = relay_non_one.dropna(subset=feature_cols + ['target_limit_up']).copy()
    print(f"Clean training samples for Relay Non-One-Word: {len(train_data):,}")
    
    # 4-Fold Temporal Walk-Forward Validation
    dates = sorted(train_data['date'].unique())
    n_dates = len(dates)
    fold_splits = [
        (dates[0], dates[int(n_dates * 0.40)], dates[int(n_dates * 0.40) + 1], dates[int(n_dates * 0.55)]),
        (dates[0], dates[int(n_dates * 0.55)], dates[int(n_dates * 0.55) + 1], dates[int(n_dates * 0.70)]),
        (dates[0], dates[int(n_dates * 0.70)], dates[int(n_dates * 0.70) + 1], dates[int(n_dates * 0.85)]),
        (dates[0], dates[int(n_dates * 0.85)], dates[int(n_dates * 0.85) + 1], dates[-1]),
    ]
    
    oof_preds = np.zeros(len(train_data))
    models = []
    
    print("\nRunning 4-fold walk-forward cross validation...")
    for fold, (tr_start, tr_end, te_start, te_end) in enumerate(fold_splits):
        tr_mask = (train_data['date'] >= tr_start) & (train_data['date'] <= tr_end)
        te_mask = (train_data['date'] >= te_start) & (train_data['date'] <= te_end)
        
        X_tr = train_data.loc[tr_mask, feature_cols].values
        y_tr = train_data.loc[tr_mask, 'target_limit_up'].values
        X_te = train_data.loc[te_mask, feature_cols].values
        y_te = train_data.loc[te_mask, 'target_limit_up'].values
        
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=24,
            min_samples_leaf=30,
            l2_regularization=1.5,
            random_state=42 + fold
        )
        clf.fit(X_tr, y_tr)
        models.append(clf)
        
        preds = clf.predict_proba(X_te)[:, 1]
        oof_preds[te_mask] = preds
        
        auc = roc_auc_score(y_te, preds) if len(np.unique(y_te)) > 1 else 0.5
        ap = average_precision_score(y_te, preds)
        print(f"  Fold {fold+1} [{te_start} to {te_end}]: Test Samples = {len(X_te)}, Positives = {y_te.sum()} ({y_te.mean()*100:.1f}%), AUC = {auc:.4f}, PR-AUC = {ap:.4f}")
    
    # Overall OOF metrics on non-initial period
    eval_mask = (train_data['date'] >= fold_splits[0][2])
    eval_y = train_data.loc[eval_mask, 'target_limit_up'].values
    eval_p = oof_preds[eval_mask]
    
    overall_auc = float(roc_auc_score(eval_y, eval_p))
    overall_ap = float(average_precision_score(eval_y, eval_p))
    print(f"\n>>> Total Out-of-Fold AUC: {overall_auc:.4f}, PR-AUC: {overall_ap:.4f} (Base positive rate: {eval_y.mean()*100:.1f}%)")
    
    # Top decile precision
    train_data['oof_score'] = oof_preds
    eval_df = train_data[eval_mask].copy()
    
    top_1pct = eval_df['oof_score'] >= eval_df['oof_score'].quantile(0.99)
    top_5pct = eval_df['oof_score'] >= eval_df['oof_score'].quantile(0.95)
    top_10pct = eval_df['oof_score'] >= eval_df['oof_score'].quantile(0.90)
    top_20pct = eval_df['oof_score'] >= eval_df['oof_score'].quantile(0.80)
    
    p_top1 = float(eval_df.loc[top_1pct, 'target_limit_up'].mean())
    p_top5 = float(eval_df.loc[top_5pct, 'target_limit_up'].mean())
    p_top10 = float(eval_df.loc[top_10pct, 'target_limit_up'].mean())
    p_top20 = float(eval_df.loc[top_20pct, 'target_limit_up'].mean())
    
    ret_top5 = float(eval_df.loc[top_5pct, 'target_day_return'].mean())
    ret_top10 = float(eval_df.loc[top_10pct, 'target_day_return'].mean())
    
    print(f"\n[模型精准度分层表现 (Precision by Score Decile)]:")
    print(f"  - Top 1%  最高分选股封板率: {p_top1*100:.2f}% (提升 {p_top1/eval_y.mean():.2f} 倍基准胜率)")
    print(f"  - Top 5%  高分选股封板率:   {p_top5*100:.2f}% (平均日内收益: +{ret_top5*100:.2f}%)")
    print(f"  - Top 10% 候选池封板率:     {p_top10*100:.2f}% (平均日内收益: +{ret_top10*100:.2f}%)")
    print(f"  - Top 20% 候选池封板率:     {p_top20*100:.2f}%")
    print(f"  - 全市场非一字基准封板率:   {eval_y.mean()*100:.2f}%")
    
    # --------------------------------------------------------------------------
    # STEP 6: Fit Final Production Model & Interpretable Rule Tree
    # --------------------------------------------------------------------------
    print_step("6. Training Final Model and Interpretable Rule Tree for Tonghuashun Formula")
    final_clf = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.05,
        max_leaf_nodes=24,
        min_samples_leaf=30,
        l2_regularization=1.5,
        random_state=2026
    )
    final_clf.fit(train_data[feature_cols].values, train_data['target_limit_up'].values)
    joblib.dump(final_clf, OUTPUT_DIR / "model_auction_relay_dragon.joblib")
    print("Saved final gradient boosting model to model_auction_relay_dragon.joblib")
    
    # Fit shallow decision tree for transparent formula extraction
    tree_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    tree_clf.fit(train_data[feature_cols].values, train_data['target_limit_up'].values)
    tree_rules = export_text(tree_clf, feature_names=feature_cols)
    print("\n[可解释决策树规则 (用于提炼同花顺选股公式)]:\n", tree_rules)
    (OUTPUT_DIR / "extracted_decision_tree_rules.txt").write_text(tree_rules, encoding='utf-8')
    
    # Feature importances via RandomForest surrogate
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(train_data[feature_cols].values, train_data['target_limit_up'].values)
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n[特征重要性排序 (Feature Importance Rankings)]:")
    print(feat_imp.to_string(index=False))
    feat_imp.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False, encoding='utf-8-sig')
    
    # --------------------------------------------------------------------------
    # STEP 7: Save Comprehensive Summary Metrics JSON
    # --------------------------------------------------------------------------
    metrics = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_relay_candidates": len(relay_df),
        "non_one_word_relay_candidates": len(relay_non_one),
        "non_one_word_share": float(len(relay_non_one) / len(relay_df)),
        "baseline_limit_up_rate": float(eval_y.mean()),
        "model_evaluation": {
            "oof_auc": overall_auc,
            "oof_pr_auc": overall_ap,
            "precision_top_1pct": p_top1,
            "precision_top_5pct": p_top5,
            "precision_top_10pct": p_top10,
            "precision_top_20pct": p_top20,
            "mean_day_return_top_5pct": ret_top5,
            "mean_day_return_top_10pct": ret_top10,
        },
        "key_auction_takeaways": {
            "golden_gap_range": "+2.0% ~ +6.0%",
            "nuclear_low_win_rate": float(relay_stats.loc[relay_stats['gap_bin'].str.contains('绿盘|深核'), 'limit_up_rate'].mean()),
            "golden_gap_win_rate": float(relay_stats.loc[relay_stats['gap_bin'].str.contains('黄金|温和'), 'limit_up_rate'].mean()),
            "capacity_dragon_win_rate": float(dragon_stats.loc[dragon_stats['is_capacity_dragon'] == 1, 'limit_up_rate'].values[0]),
            "normal_relay_win_rate": float(dragon_stats.loc[dragon_stats['is_capacity_dragon'] == 0, 'limit_up_rate'].values[0])
        }
    }
    with open(OUTPUT_DIR / "auction_dragon_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nAll metrics and models successfully persisted to {OUTPUT_DIR}")

if __name__ == "__main__":
    df = build_auction_dataset()
    analyze_and_train(df)
