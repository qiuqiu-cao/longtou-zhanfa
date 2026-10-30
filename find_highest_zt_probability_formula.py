import numpy as np
import pandas as pd
from pathlib import Path
import time

DAILY_BARS_PATH = Path("data/v3/imported/daily-bars.csv")

print("Searching for the Highest Day t Limit-Up Sealing Probability Formulation...")

cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
df = df[~df['st'] & ~df['new_listing']].copy()
df['code_str'] = df['code'].astype(str).str.zfill(6)
# Mainboard only
df = df[df['code_str'].str.startswith(('000', '001', '002', '003', '600', '601', '603', '605'))].copy()
df = df.sort_values(['code_str', 'date']).reset_index(drop=True)

# EXACT A-SHARE LIMIT PRICE:
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
df['prev_low'] = grp['low'].shift(1)
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

# Next day open-to-open return with friction
FRICTION = 0.0012
df['ret_t1_open'] = (df['next_open'] - df['open']) / df['open'] - FRICTION
df['ret_t1_close'] = (df['next_close'] - df['open']) / df['open'] - FRICTION

print(f"Dataset prepared. Testing candidate combinations...")

# Grid search across key parameters:
# 1. Streak: >= 1 vs >= 2 vs >= 3
# 2. Open Gap: [3.0%, 7.0%], [4.0%, 7.5%], [4.5%, 8.0%], [5.0%, 8.5%]
# 3. Prev Amount: >= 3.0亿, >= 5.0亿, >= 8.0亿
# 4. Prev Turnover: [10%, 25%], [12%, 28%]
# 5. MA Alignment: MA5 > MA10 vs MA5 > MA10 > MA20

test_results = []

for streak_min in [1, 2, 3]:
    for gap_low, gap_high in [(0.03, 0.07), (0.04, 0.075), (0.045, 0.08), (0.05, 0.085)]:
        for amt_min in [3.0, 5.0, 7.0]:
            for to_low, to_high in [(8.0, 28.0), (12.0, 26.0)]:
                for ma_strict in [False, True]:
                    mask = (
                        df['prev_zt'] &
                        df['is_non_one_word'] &
                        (df['prev_streak'] >= streak_min) &
                        (df['open_gap'] >= gap_low) & (df['open_gap'] <= gap_high) &
                        df['break_high'] &
                        df['above_vwap'] &
                        (df['prev_amount_yi'] >= amt_min) &
                        (df['prev_turnover'] >= to_low) & (df['prev_turnover'] <= to_high)
                    )
                    if ma_strict:
                        mask &= (df['prev_ma5'] > df['prev_ma10']) & (df['prev_ma10'] > df['prev_ma20'])
                    else:
                        mask &= (df['prev_ma5'] > df['prev_ma10'])
                        
                    sub = df[mask]
                    n = len(sub)
                    if n < 30: # Need statistically meaningful sample size
                        continue
                    
                    zt_rate = sub['is_zt'].mean() * 100
                    touch_rate = sub['is_touch_zt'].mean() * 100
                    win_rate = (sub['ret_t1_open'] > 0).mean() * 100
                    avg_ret = sub['ret_t1_open'].mean() * 100
                    
                    # Next-day premium if sealed
                    sealed = sub[sub['is_zt']]
                    sealed_prem = ((sealed['next_open'] - sealed['close']) / sealed['close'] - FRICTION).mean() * 100 if len(sealed) > 0 else 0
                    
                    test_results.append({
                        'streak_min': streak_min,
                        'gap_range': f"{gap_low*100:.1f}%~{gap_high*100:.1f}%",
                        'amt_min': amt_min,
                        'to_range': f"{to_low:.0f}%~{to_high:.0f}%",
                        'ma_strict': ma_strict,
                        'n_trades': n,
                        'n_days': sub['date'].nunique(),
                        'touch_rate': touch_rate,
                        'zt_rate': zt_rate,
                        'win_rate': win_rate,
                        'avg_ret': avg_ret,
                        'sealed_prem': sealed_prem
                    })

res_df = pd.DataFrame(test_results)
res_df = res_df.sort_values('zt_rate', ascending=False).reset_index(drop=True)

print("\n" + "="*95)
print("【Top 10 封板率最高的参数组合 (全市场精确主板 1.10 涨停价回测)】")
print("="*95)
for i, r in res_df.head(10).iterrows():
    print(f"Top {i+1}: 封板率={r['zt_rate']:.2f}% | 触板率={r['touch_rate']:.2f}% | 样本量={r['n_trades']}次 (覆盖{r['n_days']}个交易日)")
    print(f"       连板起点: {r['streak_min']}板及以上 | 竞价高开: {r['gap_range']} | 昨日成交: >={r['amt_min']}亿 | 换手: {r['to_range']} | 均线严格多头: {r['ma_strict']}")
    print(f"       封死次日平均溢价: {r['sealed_prem']:+.2f}% | 全池开盘买次日开盘均收益: {r['avg_ret']:+.2f}%")
    print("-" * 80)
