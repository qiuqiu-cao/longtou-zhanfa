import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

DAILY_BARS_PATH = Path("data/v3/imported/daily-bars.csv")

print("Loading data for enhanced formula optimization...")
t0 = time.time()
cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
df = df[~df['st'] & ~df['new_listing']].copy()
df['code_str'] = df['code'].astype(str).str.zfill(6)
df = df[~df['code_str'].str.startswith(('43', '83', '87', '92'))].copy()
df = df.sort_values(['code_str', 'date']).reset_index(drop=True)

# Limit price
is_20 = df['code_str'].str.startswith(('300', '301', '688'))
df['limit_price'] = (df['pre_close'] * np.where(is_20, 1.195, 1.095)).round(2)
df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)

# Moving averages
df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
df['vwap'] = np.where(df['volume'] > 0, df['amount'] / df['volume'], df['close'])

grp = df.groupby('code_str')
df['prev_zt'] = grp['is_zt'].shift(1).fillna(False)
df['prev_close'] = grp['close'].shift(1)
df['prev_open'] = grp['open'].shift(1)
df['prev_high'] = grp['high'].shift(1)
df['prev_low'] = grp['low'].shift(1)
df['prev_amount'] = grp['amount'].shift(1)
df['prev_volume'] = grp['volume'].shift(1)
df['prev_turnover'] = grp['turnover'].shift(1)
df['prev_vwap'] = grp['vwap'].shift(1)
df['prev_ma5'] = grp['ma5'].shift(1)
df['prev_ma10'] = grp['ma10'].shift(1)
df['prev_ma20'] = grp['ma20'].shift(1)
df['next_open'] = grp['open'].shift(-1)
df['next_close'] = grp['close'].shift(-1)
df['next_high'] = grp['high'].shift(-1)
df['next_is_zt'] = grp['is_zt'].shift(-1).fillna(False)

df = df[df['prev_close'].notnull()].copy()
df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
df['is_non_one_word'] = (df['open'] < df['limit_price'] - 0.001)
df['break_high'] = (df['open'] >= df['prev_high'])
df['above_vwap'] = (df['open'] >= df['prev_vwap'])
df['prev_amount_yi'] = df['prev_amount'] / 1e8

# Next day return (Open to Open, with friction 0.12%)
df['ret_t1_open'] = (df['next_open'] - df['open']) / df['open'] - 0.0012
# Intraday return (Open to Close)
df['ret_day_t'] = (df['close'] - df['open']) / df['open'] - 0.0012

print(f"Data ready in {time.time() - t0:.2f}s. Total rows: {len(df):,}")

# Test multiple configurations
configs = [
    {
        'name': '现有基准版 (Baseline: Gap 2%~7%, Amt>=3亿, TO 8%~32%)',
        'mask': (
            df['prev_zt'] & df['is_non_one_word'] &
            (df['open_gap'] >= 0.02) & (df['open_gap'] <= 0.07) &
            df['break_high'] &
            (df['prev_amount_yi'] >= 3.0) &
            (df['prev_turnover'] >= 8.0) & (df['prev_turnover'] <= 32.0)
        )
    },
    {
        'name': '优化方案一: 黄金微调版 (Gap 3%~7.5%, Amt>=4亿, TO 10%~28%)',
        'mask': (
            df['prev_zt'] & df['is_non_one_word'] &
            (df['open_gap'] >= 0.03) & (df['open_gap'] <= 0.075) &
            df['break_high'] & df['above_vwap'] &
            (df['prev_amount_yi'] >= 4.0) &
            (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 28.0)
        )
    },
    {
        'name': '优化方案二: 趋势多头强龙版 (+MA5>MA10, Gap 3.2%~7.2%, 站上VWAP)',
        'mask': (
            df['prev_zt'] & df['is_non_one_word'] &
            (df['open_gap'] >= 0.032) & (df['open_gap'] <= 0.072) &
            df['break_high'] & df['above_vwap'] &
            (df['prev_ma5'] > df['prev_ma10']) &
            (df['prev_amount_yi'] >= 3.5) &
            (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 28.0)
        )
    },
    {
        'name': '优化方案三: 极稳高盈亏比终极版 (主板专注+趋势多头+大容量换手+均线发散)',
        'mask': (
            df['prev_zt'] & df['is_non_one_word'] &
            ~df['code_str'].str.startswith(('300', '301', '688')) & # 专注主板10%防大面
            (df['open_gap'] >= 0.03) & (df['open_gap'] <= 0.07) &
            df['break_high'] & df['above_vwap'] &
            (df['prev_ma5'] > df['prev_ma10']) & (df['prev_close'] > df['prev_ma20']) &
            (df['prev_amount_yi'] >= 4.0) &
            (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 26.0)
        )
    }
]

print("\n" + "="*85)
print("【多方案严谨全周期历史回测比较 (2,013,716 个日线真实样本)】")
print("="*85)

for cfg in configs:
    sub = df[cfg['mask']].copy()
    n_trades = len(sub)
    if n_trades == 0:
        continue
    
    zt_rate = sub['is_zt'].mean() * 100
    touch_rate = sub['is_touch_zt'].mean() * 100
    
    # Returns
    mean_ret_day_t = sub['ret_day_t'].mean() * 100
    mean_ret_t1_open = sub['ret_t1_open'].mean() * 100
    
    # Win rate (next day open positive)
    win_rate = (sub['ret_t1_open'] > 0).mean() * 100
    
    # Profit factor (wins / losses)
    gains = sub.loc[sub['ret_t1_open'] > 0, 'ret_t1_open'].sum()
    losses = abs(sub.loc[sub['ret_t1_open'] < 0, 'ret_t1_open'].sum())
    pl_ratio = gains / losses if losses > 0 else np.nan
    
    # Average win vs average loss
    avg_win = sub.loc[sub['ret_t1_open'] > 0, 'ret_t1_open'].mean() * 100
    avg_loss = abs(sub.loc[sub['ret_t1_open'] < 0, 'ret_t1_open'].mean() * 100)
    
    # Maximum drawdown on cumulative returns
    cum = (1 + sub['ret_t1_open']).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100
    
    print(f"\n★ {cfg['name']}")
    print(f"  - 触发交易次数: {n_trades:,} 次")
    print(f"  - 当日封死涨停率: {zt_rate:.2f}% | 盘中触板率: {touch_rate:.2f}%")
    print(f"  - 当日开盘到收盘均收益: {mean_ret_day_t:+.2f}%")
    print(f"  - 次日开盘卖出胜率 (胜率): {win_rate:.2f}%")
    print(f"  - 次日开盘卖出均收益 (单笔期望): {mean_ret_t1_open:+.2f}%")
    print(f"  - 盈亏比 (Profit Factor): {pl_ratio:.2f}:1 (平均赢 {avg_win:.2f}% / 平均输 {avg_loss:.2f}%)")
    print(f"  - 策略累计最大回撤: {max_dd:.2f}%")
