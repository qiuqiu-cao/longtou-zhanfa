import numpy as np
import pandas as pd
from pathlib import Path

DAILY_BARS_PATH = Path("data/v3/imported/daily-bars.csv")

cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
df = df[~df['st'] & ~df['new_listing']].copy()
df['code_str'] = df['code'].astype(str).str.zfill(6)
# Filter mainboard only (000, 001, 002, 003, 600, 601, 603, 605)
df = df[df['code_str'].str.startswith(('000', '001', '002', '003', '600', '601', '603', '605'))].copy()
df = df.sort_values(['code_str', 'date']).reset_index(drop=True)

df['limit_price'] = (df['pre_close'] * 1.095).round(2)
df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)

# Moving averages
df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
df['vwap'] = np.where(df['volume'] > 0, df['amount'] / df['volume'], df['close'])

# Streak
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
df['prev_high'] = grp['high'].shift(1)
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

df = df[df['prev_close'].notnull() & df['next_open'].notnull()].copy()
df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
df['is_non_one_word'] = (df['open'] < df['limit_price'] - 0.001)
df['break_high'] = (df['open'] >= df['prev_high'])
df['above_vwap'] = (df['open'] >= df['prev_vwap'])
df['prev_amount_yi'] = df['prev_amount'] / 1e8

# Compare Strategies
# 1. Formula V1: 基准版 (Gap 2%~7%, Amt>=3亿, TO 8%~32%)
# 2. Formula V2: 稳健进阶版 (Gap 3%~7%, Amt>=5亿, TO 10%~26%, MA5>MA10, Break High, Above VWAP)
# 3. Formula V3: 襄阳轴承·大换手龙头成妖版 (Streak>=2, Gap 3.5%~7.2%, Amt>=6亿, TO 12%~28%, MA5>MA10>MA20)

f1_mask = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['open_gap'] >= 0.02) & (df['open_gap'] <= 0.07) &
    df['break_high'] &
    (df['prev_amount_yi'] >= 3.0) &
    (df['prev_turnover'] >= 8.0) & (df['prev_turnover'] <= 32.0)
)

f2_mask = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['open_gap'] >= 0.03) & (df['open_gap'] <= 0.07) &
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) &
    (df['prev_amount_yi'] >= 5.0) &
    (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 26.0)
)

f3_mask = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['prev_streak'] >= 2) & # 2进3及以上龙头
    (df['open_gap'] >= 0.035) & (df['open_gap'] <= 0.072) & # 极精黄金高开
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) & (df['prev_close'] > df['prev_ma20']) &
    (df['prev_amount_yi'] >= 5.0) &
    (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 28.0)
)

forms = [
    ("基准公式 (原版本)", f1_mask),
    ("稳健进阶公式 (主板大容量+趋势共振)", f2_mask),
    ("成妖极稳公式 (2进3/3进4大换手真龙版)", f3_mask)
]

print("="*85)
print("【三种公式核心回测指标全面横向评测 (主板2023-2026全样本)】")
print("="*85)

for title, m in forms:
    sub = df[m].copy()
    n = len(sub)
    if n == 0:
        continue
    
    zt_rate = sub['is_zt'].mean() * 100
    touch_rate = sub['is_touch_zt'].mean() * 100
    
    # Execution 1: Sealing Buy (打板买入模式)
    # Buy when touched limit up, hold if sealed, exit next open
    touched = sub[sub['is_touch_zt']].copy()
    n_touched = len(touched)
    seal_rate_given_touch = touched['is_zt'].mean() * 100
    # Return of touched trades: buy at limit_price, sell at next open
    ret_zt_buy = (touched['next_open'] - touched['limit_price']) / touched['limit_price'] - 0.0012
    win_rate_zt = (ret_zt_buy > 0).mean() * 100
    mean_ret_zt = ret_zt_buy.mean() * 100
    
    # Execution 2: Overnight Premium of successfully sealed dragons (龙头持筹次日溢价)
    sealed = sub[sub['is_zt']].copy()
    sealed_premium = ((sealed['next_open'] - sealed['close']) / sealed['close'] - 0.0012).mean() * 100
    sealed_win = ((sealed['next_open'] - sealed['close']) > 0).mean() * 100
    
    # Execution 3: Standard Trailing Return (hold if sealed to t+1 close, sell at next open if unsealed)
    trailing_ret = np.where(
        sub['is_zt'],
        (sub['next_close'] - sub['open']) / sub['open'] - 0.0012,
        np.maximum((sub['low'] - sub['open']) / sub['open'], -0.04) # -4% stop loss simulated
    )
    trailing_mean = trailing_ret.mean() * 100
    trailing_win = (trailing_ret > 0).mean() * 100
    
    gains = trailing_ret[trailing_ret > 0].sum()
    losses = abs(trailing_ret[trailing_ret < 0].sum())
    pl_ratio = gains / losses if losses > 0 else np.nan
    
    print(f"\n★ {title}")
    print(f"  - 触发标的数: {n:,} 只")
    print(f"  - 触板率 (盘中冲板率): {touch_rate:.2f}%")
    print(f"  - 封板率 (收盘封死率): {zt_rate:.2f}%")
    print(f"  - 触板后封死成功率: {seal_rate_given_touch:.2f}%")
    print(f"  - 封死涨停后【次日平均溢价】: {sealed_premium:+.2f}% (次日高开胜率: {sealed_win:.2f}%)")
    print(f"  - 结合【-4%均线止损】策略期望单笔收益: {trailing_mean:+.2f}%")
    print(f"  - 盈亏比 (Profit-to-Loss Ratio): {pl_ratio:.2f}:1")
