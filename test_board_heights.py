import numpy as np
import pandas as pd
from pathlib import Path

DAILY_BARS_PATH = Path("data/v3/imported/daily-bars.csv")

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

grp = df.groupby('code_str')
df['prev_zt'] = grp['is_zt'].shift(1).fillna(False)
df['prev_close'] = grp['close'].shift(1)
df['prev_high'] = grp['high'].shift(1)
df['prev_amount'] = grp['amount'].shift(1)
df['prev_volume'] = grp['volume'].shift(1)
df['prev_turnover'] = grp['turnover'].shift(1)
df['next_open'] = grp['open'].shift(-1)
df['next_close'] = grp['close'].shift(-1)
df['next_high'] = grp['high'].shift(-1)

df = df[df['prev_close'].notnull() & df['next_open'].notnull()].copy()
df['open_gap'] = (df['open'] - df['pre_close']) / df['pre_close']
df['is_non_one_word'] = (df['open'] < df['limit_price'] - 0.001)
df['break_high'] = (df['open'] >= df['prev_high'])
df['prev_amount_yi'] = df['prev_amount'] / 1e8

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
df['prev_streak'] = grp['streak'].shift(1).fillna(0)

# Compare board heights: 1-adv-2 vs 2-adv-3 vs 3-adv-4+
df['ret_t1_open'] = (df['next_open'] - df['open']) / df['open'] - 0.0012
df['ret_t1_close'] = (df['next_close'] - df['open']) / df['open'] - 0.0012

# Condition: Mainboard, Non-one-word, Golden gap 3%~7%, Break high
mask_base = (
    df['prev_zt'] & df['is_non_one_word'] &
    ~df['code_str'].str.startswith(('300', '301', '688')) &
    (df['open_gap'] >= 0.03) & (df['open_gap'] <= 0.075) &
    df['break_high'] &
    (df['prev_amount_yi'] >= 4.0)
)

print("\n=== 分板位深度统计 (主板大容量高开 3%~7.5%) ===")
for st, label in [(1, '1进2 (首板接力)'), (2, '2进3 (确立主线)'), (3, '3进4及以上 (高位龙头成妖)')]:
    if st == 3:
        sub = df[mask_base & (df['prev_streak'] >= 3)].copy()
    else:
        sub = df[mask_base & (df['prev_streak'] == st)].copy()
        
    zt_rate = sub['is_zt'].mean() * 100
    touch_rate = sub['is_touch_zt'].mean() * 100
    win_rate = (sub['ret_t1_open'] > 0).mean() * 100
    avg_ret = sub['ret_t1_open'].mean() * 100
    
    # Premium if sealed
    sealed = sub[sub['is_zt']]
    sealed_premium = ((sealed['next_open'] - sealed['close']) / sealed['close'] - 0.0012).mean() * 100 if len(sealed) > 0 else 0
    sealed_win = ((sealed['next_open'] - sealed['close']) > 0).mean() * 100 if len(sealed) > 0 else 0
    
    print(f"\n【{label}】: 样本量 = {len(sub)}")
    print(f"  - 当日封板率: {zt_rate:.1f}% | 触板率: {touch_rate:.1f}%")
    print(f"  - 次日开盘胜率: {win_rate:.1f}% | 平均单笔收益: {avg_ret:+.2f}%")
    print(f"  - 成功封板样本量: {len(sealed)}")
    print(f"  - 封板后次日溢价均值: {sealed_premium:+.2f}% | 溢价胜率: {sealed_win:.1f}%")
