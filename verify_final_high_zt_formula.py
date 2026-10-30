import numpy as np
import pandas as pd
from pathlib import Path

DAILY_BARS_PATH = Path("data/v3/imported/daily-bars.csv")

cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
df = df[~df['st'] & ~df['new_listing']].copy()
df['code_str'] = df['code'].astype(str).str.zfill(6)
df = df[df['code_str'].str.startswith(('000', '001', '002', '003', '600', '601', '603', '605'))].copy()
df = df.sort_values(['code_str', 'date']).reset_index(drop=True)

# Exact limit price
df['limit_price'] = (df['pre_close'] * 1.10).round(2)
df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])
df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)

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
df['prev_vwap'] = np.where(df['prev_volume'] > 0, df['prev_amount'] / df['prev_volume'], df['prev_close'])

df['ma5'] = grp['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = grp['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = grp['close'].transform(lambda x: x.rolling(20).mean())
df['prev_ma5'] = grp['ma5'].shift(1)
df['prev_ma10'] = grp['ma10'].shift(1)
df['prev_ma20'] = grp['ma20'].shift(1)

df['next_open'] = grp['open'].shift(-1)
df['next_close'] = grp['close'].shift(-1)

df = df[df['prev_close'].notnull() & df['next_open'].notnull()].copy()
df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
df['is_non_one_word'] = (df['open'] < df['limit_price'] - 0.001)
df['break_high'] = (df['open'] >= df['prev_high'])
df['above_vwap'] = (df['open'] >= df['prev_vwap'])
df['prev_amount_yi'] = df['prev_amount'] / 1e8

FRICTION = 0.0012
df['ret_t1_open'] = (df['next_open'] - df['open']) / df['open'] - FRICTION

# Test the two high-probability candidates:
# Candidate A: 2-board and above (2进3及以上大换手主升)
mask_a = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['prev_streak'] >= 2) &
    (df['open_gap'] >= 0.045) & (df['open_gap'] <= 0.085) &
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) & (df['prev_ma10'] > df['prev_ma20']) &
    (df['prev_amount_yi'] >= 3.5) &
    (df['prev_turnover'] >= 12.0) & (df['prev_turnover'] <= 26.0)
)

# Candidate B: 3-board and above (3进4成妖专用)
mask_b = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['prev_streak'] >= 3) &
    (df['open_gap'] >= 0.045) & (df['open_gap'] <= 0.085) &
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) & (df['prev_ma10'] > df['prev_ma20']) &
    (df['prev_amount_yi'] >= 3.5) &
    (df['prev_turnover'] >= 12.0) & (df['prev_turnover'] <= 26.0)
)

for label, m in [("方案A: 2进3及以上【高胜率真龙版】", mask_a),
                 ("方案B: 3进4及以上【成妖最高胜率版】", mask_b)]:
    sub = df[m].copy()
    n = len(sub)
    n_days = sub['date'].nunique()
    touch_rate = sub['is_touch_zt'].mean() * 100
    seal_rate = sub['is_zt'].mean() * 100
    touch_to_seal = sub[sub['is_touch_zt']]['is_zt'].mean() * 100
    
    sealed = sub[sub['is_zt']]
    sealed_prem = ((sealed['next_open'] - sealed['close']) / sealed['close'] - FRICTION).mean() * 100
    sealed_win = ((sealed['next_open'] - sealed['close']) > 0).mean() * 100
    
    print(f"\n================ {label} ================")
    print(f"触发样本量: {n} 只 (分布在 {n_days} 个交易日，不强求每天有票，大部分交易日空仓防守!)")
    print(f"【当日盘中触板率】: {touch_rate:.2f}% ({sub['is_touch_zt'].sum()}/{n})")
    print(f"【当日收盘封死率】: {seal_rate:.2f}% ({sub['is_zt'].sum()}/{n})")
    print(f"触板后封死成功率: {touch_to_seal:.2f}%")
    print(f"封死涨停后次日平均高开溢价: {sealed_prem:+.2f}% (次日高开胜率: {sealed_win:.2f}%)")
