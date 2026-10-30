#!/usr/bin/env python3
"""
backtest_auction_dragon_refined.py

Refined Portfolio Backtest for Auction Dragon Strategy:
Compares:
1. Benchmark: Naked Formula (All 3,525 triggered signals)
2. Refined Model: Daily Top-1 / Top-2 Focus (Machine learning scoring selects the single highest-conviction dragon each day)
3. Gated Model: Daily Top-1 + Market Emotion Gate (Avoids severe market retreat regimes)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(".")
OUTPUT_DIR = ROOT / "research/model_upgrade_validation/auction_dragon_model_20261028"
DAILY_BARS_PATH = ROOT / "data/v3/imported/daily-bars.csv"
MODEL_PATH = OUTPUT_DIR / "model_auction_relay_dragon.joblib"

def run_refined_backtest():
    print("=" * 70)
    print("REFINED STRATEGY BACKTEST: TOP-1 DRAGON FOCUS & REGIME GATING")
    print("=" * 70)
    
    # 1. Load data
    cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
    df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
    df = df[~df['st'] & ~df['new_listing']].copy()
    df['code_str'] = df['code'].astype(str).str.zfill(6)
    df = df[~df['code_str'].str.startswith(('43', '83', '87', '92'))].copy()
    df = df.sort_values(['code_str', 'date']).reset_index(drop=True)
    
    is_20 = df['code_str'].str.startswith(('300', '301', '688'))
    df['limit_price'] = (df['pre_close'] * np.where(is_20, 1.195, 1.095)).round(2)
    df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
    df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)
    
    # Calculate MA5 and MA20
    df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
    
    # Board streak
    zt_arr = df['is_zt'].values
    codes = df['code_str'].values
    streak = np.zeros(len(df), dtype=int)
    c_code, c_s = "", 0
    for i in range(len(df)):
        if codes[i] != c_code:
            c_code, c_s = codes[i], (1 if zt_arr[i] else 0)
        else:
            c_s = (c_s + 1) if zt_arr[i] else 0
        streak[i] = c_s
    df['streak'] = streak
    
    grp = df.groupby('code_str')
    df['prev_zt'] = grp['is_zt'].shift(1).fillna(False)
    df['prev_streak'] = grp['streak'].shift(1).fillna(0)
    df['prev_close'] = grp['close'].shift(1)
    df['prev_open'] = grp['open'].shift(1)
    df['prev_high'] = grp['high'].shift(1)
    df['prev_low'] = grp['low'].shift(1)
    df['prev_amount'] = grp['amount'].shift(1)
    df['prev_volume'] = grp['volume'].shift(1)
    df['prev_turnover'] = grp['turnover'].shift(1)
    df['prev_amplitude_pct'] = np.where(df['prev_close'] > 0, (df['prev_high'] - df['prev_low']) / grp['pre_close'].shift(1) * 100.0, 0.0)
    df['prev_vwap'] = np.where(df['prev_volume'] > 0, df['prev_amount'] / (df['prev_volume'] * 100.0), df['prev_close'])
    df['prev_ma5'] = grp['ma5'].shift(1)
    df['prev_ma10'] = grp['ma10'].shift(1)
    df['prev_ma20'] = grp['ma20'].shift(1)
    
    df['next_open'] = grp['open'].shift(-1)
    df['next_close'] = grp['close'].shift(-1)
    df['next_is_zt'] = grp['is_zt'].shift(-1).fillna(False)
    
    df = df[df['prev_close'].notnull()].copy()
    
    # Auction variables
    df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['open_vs_prev_high'] = (df['open'] - df['prev_high']) / df['prev_high']
    df['open_vs_prev_vwap'] = (df['open'] - df['prev_vwap']) / df['prev_vwap']
    df['open_vs_prev_open'] = (df['open'] - df['prev_open']) / df['prev_open']
    df['open_vs_ma5'] = np.where(df['prev_ma5'] > 0, (df['open'] - df['prev_ma5']) / df['prev_ma5'], 0.0)
    df['open_vs_ma20'] = np.where(df['prev_ma20'] > 0, (df['open'] - df['prev_ma20']) / df['prev_ma20'], 0.0)
    df['is_ma_bull'] = (df['prev_ma5'] > df['prev_ma10']) & (df['prev_ma10'] > df['prev_ma20'])
    df['prev_amount_yi'] = df['prev_amount'] / 1e8
    df['float_cap_yi'] = df['float_cap'] / 1e8
    df['is_capacity_dragon'] = (df['prev_amount_yi'] >= 5.0) & (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 28.0)
    df['is_small_cap_trap'] = (df['float_cap_yi'] < 30.0) & (df['prev_amount_yi'] < 3.0)
    df['is_low_price'] = (df['pre_close'] < 12.0)
    
    # Strategy condition
    df['is_non_one_word'] = (df['open'] < df['limit_price'] - 0.001)
    df['golden_gap'] = (df['open_gap'] >= 0.02) & (df['open_gap'] <= 0.07)
    df['break_high'] = (df['open'] >= df['prev_high'])
    df['healthy_capacity'] = (df['prev_amount_yi'] >= 3.0) & (df['prev_turnover'] >= 8.0) & (df['prev_turnover'] <= 32.0)
    
    df['signal'] = (df['prev_zt'] & df['is_non_one_word'] & df['golden_gap'] & df['break_high'] & df['healthy_capacity'])
    
    candidate_df = df[df['signal']].copy()
    print(f"Total candidate matches: {len(candidate_df)}")
    
    # Score candidates with trained model
    feature_cols = [
        'open_gap', 'open_vs_prev_high', 'open_vs_prev_vwap', 'open_vs_prev_open',
        'open_vs_ma5', 'open_vs_ma20', 'prev_streak', 'prev_turnover',
        'prev_amount_yi', 'prev_amplitude_pct', 'float_cap_yi', 'pre_close',
        'is_capacity_dragon', 'is_small_cap_trap', 'is_low_price', 'is_ma_bull'
    ]
    model = joblib.load(MODEL_PATH)
    candidate_df['ml_score'] = model.predict_proba(candidate_df[feature_cols].values)[:, 1]
    
    # Sort within each day by ml_score descending
    candidate_df = candidate_df.sort_values(['date', 'ml_score'], ascending=[True, False]).reset_index(drop=True)
    candidate_df['daily_rank'] = candidate_df.groupby('date').cumcount() + 1
    
    # Multi-day forward tracing for Mode A
    df_indexed = df.set_index(['code_str', 'date'])
    trading_dates = sorted(df['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    FRICTION = 0.0012
    
    trailing_returns = []
    holding_days_list = []
    
    for idx, row in candidate_df.iterrows():
        c = row['code_str']
        d = row['date']
        d_idx = date_to_idx.get(d)
        entry_price = row['open']
        t_is_zt = row['is_zt']
        
        if not t_is_zt:
            exit_price = row['next_open'] if pd.notnull(row['next_open']) else row['close']
            ret = (exit_price - entry_price) / entry_price - FRICTION
            h = 1
        else:
            curr_idx = d_idx + 1
            h = 1
            final_exit_price = entry_price
            while curr_idx < len(trading_dates):
                next_d = trading_dates[curr_idx]
                if (c, next_d) in df_indexed.index:
                    next_row = df_indexed.loc[(c, next_d)]
                    if next_row['is_zt']:
                        h += 1
                        curr_idx += 1
                        continue
                    else:
                        final_exit_price = next_row['open'] if next_row['open'] > 0 else next_row['close']
                        h += 1
                        break
                else:
                    final_exit_price = entry_price
                    break
            ret = (final_exit_price - entry_price) / entry_price - FRICTION
        trailing_returns.append(ret)
        holding_days_list.append(h)
        
    candidate_df['return_trailing'] = trailing_returns
    candidate_df['holding_days'] = holding_days_list
    
    # Market emotion regime proxy: daily market limit up count and failed rate
    market_daily = df.groupby('date').agg(
        total_zt=('is_zt', 'sum'),
        total_touch=('is_touch_zt', 'sum')
    )
    market_daily['failed_zt'] = market_daily['total_touch'] - market_daily['total_zt']
    market_daily['failed_rate'] = market_daily['failed_zt'] / market_daily['total_touch'].replace(0, 1)
    market_daily['regime_ok'] = (market_daily['total_zt'] >= 15) & (market_daily['failed_rate'] <= 0.45)
    
    # -------------------------------------------------------------
    # Evaluate 3 Tiers of Strategy Execution
    # -------------------------------------------------------------
    def evaluate_portfolio(sub_df, name):
        daily_ret = sub_df.groupby('date')['return_trailing'].mean()
        cal = pd.DataFrame({'date': trading_dates}).merge(daily_ret.rename('ret'), on='date', how='left').fillna(0.0)
        cal['nav'] = (1.0 + cal['ret']).cumprod()
        cal['cum_max'] = cal['nav'].cummax()
        cal['dd'] = (cal['nav'] - cal['cum_max']) / cal['cum_max']
        
        n_trades = len(sub_df)
        win_rate = (sub_df['return_trailing'] > 0).mean()
        zt_rate = sub_df['is_zt'].mean()
        touch_rate = sub_df['is_touch_zt'].mean()
        
        avg_w = sub_df.loc[sub_df['return_trailing'] > 0, 'return_trailing'].mean()
        avg_l = abs(sub_df.loc[sub_df['return_trailing'] <= 0, 'return_trailing'].mean())
        pl_ratio = avg_w / avg_l if avg_l > 0 else 0
        
        tot_ret = cal['nav'].iloc[-1] - 1.0
        cagr = (1.0 + tot_ret) ** (242.0 / len(trading_dates)) - 1.0
        mdd = cal['dd'].min()
        sharpe = (cal['ret'].mean() / cal['ret'].std()) * np.sqrt(242) if cal['ret'].std() > 0 else 0
        
        return {
            'name': name,
            'trades': n_trades,
            'zt_rate': zt_rate,
            'touch_rate': touch_rate,
            'win_rate': win_rate,
            'pl_ratio': pl_ratio,
            'tot_return': tot_ret,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe,
            'cal_df': cal
        }
        
    res_tier0 = evaluate_portfolio(candidate_df, "Tier 0: 裸公式全部买入 (All Triggers)")
    res_tier1 = evaluate_portfolio(candidate_df[candidate_df['daily_rank'] == 1], "Tier 1: 每日仅做 Top-1 聚焦真龙 (Daily Top-1 Focus)")
    res_tier2 = evaluate_portfolio(candidate_df[(candidate_df['daily_rank'] == 1) & candidate_df['date'].isin(market_daily[market_daily['regime_ok']].index)], "Tier 2: Top-1 聚焦 + 情绪潮汐闸门 (Top-1 + Regime Gate)")
    
    print("\n" + "=" * 85)
    print(f"{'策略版本':<35} | {'交易笔数':<6} | {'封板率':<7} | {'胜率':<7} | {'盈亏比':<7} | {'CAGR':<9} | {'最大回撤':<8} | {'夏普比率':<6}")
    print("-" * 85)
    for r in [res_tier0, res_tier1, res_tier2]:
        print(f"{r['name']:<35} | {r['trades']:<8} | {r['zt_rate']*100:5.2f}% | {r['win_rate']*100:5.2f}% | {r['pl_ratio']:5.2f}:1 | +{r['cagr']*100:6.1f}% | {r['mdd']*100:6.2f}% | {r['sharpe']:5.2f}")
    print("=" * 85)
    
    # Save comparison metrics
    summary = {
        "tier0_all_triggers": {k: v for k, v in res_tier0.items() if k != 'cal_df'},
        "tier1_top1_focus": {k: v for k, v in res_tier1.items() if k != 'cal_df'},
        "tier2_top1_gated": {k: v for k, v in res_tier2.items() if k != 'cal_df'}
    }
    with open(OUTPUT_DIR / "backtest_tiers_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"Saved refined backtest comparisons to {OUTPUT_DIR / 'backtest_tiers_comparison.json'}")

if __name__ == "__main__":
    run_refined_backtest()
