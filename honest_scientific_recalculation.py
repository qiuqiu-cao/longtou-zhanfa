import numpy as np
import pandas as pd
from pathlib import Path

DAILY_BARS_PATH = Path("data/v3/imported/daily-bars.csv")

print("Executing Honest & Rigorous Recalculation fixing all audited defects...")

cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'turnover', 'float_cap', 'st', 'new_listing']
df = pd.read_csv(DAILY_BARS_PATH, usecols=cols)
df = df[~df['st'] & ~df['new_listing']].copy()
df['code_str'] = df['code'].astype(str).str.zfill(6)
# Mainboard only
df = df[df['code_str'].str.startswith(('000', '001', '002', '003', '600', '601', '603', '605'))].copy()
df = df.sort_values(['code_str', 'date']).reset_index(drop=True)

# 1. FIX DEFECT 1: EXACT A-Share Mainboard Limit Price Calculation!
# In A-shares: round(pre_close * 1.10, 2)
df['limit_price'] = (df['pre_close'] * 1.10).round(2)
# Real touch: high reached true limit price
df['is_touch_zt'] = (df['high'] >= df['limit_price'] - 0.001)
# Real seal: close equals true limit price AND high equals close
df['is_zt'] = (df['close'] >= df['limit_price'] - 0.001) & (df['close'] == df['high'])

# Moving averages
df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
df['vwap'] = np.where(df['volume'] > 0, df['amount'] / df['volume'], df['close'])

# Consecutive board streak
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

# Condition Alignment:
# Match formula exactly:
# - prev_streak >= 2 (2 boards and above)
# - non one word
# - golden gap 3.5% ~ 7.2%
# - break high
# - above vwap
# - prev_ma5 > prev_ma10 and prev_close > prev_ma20
# - prev_amount >= 4.5亿 (aligned with 450000000)
# - prev_turnover 10% ~ 28%
mask = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['prev_streak'] >= 2) &
    (df['open_gap'] >= 0.035) & (df['open_gap'] <= 0.072) &
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) & (df['prev_close'] > df['prev_ma20']) &
    (df['prev_amount_yi'] >= 4.5) &
    (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 28.0)
)

candidates = df[mask].copy().reset_index(drop=True)
print(f"Total matched candidates: {len(candidates)}")

# 1. Real touch rate & real seal rate
real_touch_rate = candidates['is_touch_zt'].mean() * 100
real_seal_rate = candidates['is_zt'].mean() * 100
real_touch_to_seal = candidates[candidates['is_touch_zt']]['is_zt'].mean() * 100

print(f"\n[真实涨停价格修正后指标 (四舍五入 1.10 主板精确价)]:")
print(f"  - 真实盘中触板率: {real_touch_rate:.2f}% ({candidates['is_touch_zt'].sum()}/{len(candidates)})")
print(f"  - 真实收盘封板率: {real_seal_rate:.2f}% ({candidates['is_zt'].sum()}/{len(candidates)})")
print(f"  - 触板后成功封死率: {real_touch_to_seal:.2f}%")

# 2. FIX DEFECT 2 & 3: Strict A-Share T+1 Execution (NO Day t intraday stop loss fantasy!)
FRICTION = 0.0012 # 0.12% commission + stamp tax

# Execution Mode 1: 开盘全买 + T+1 开盘卖出 (Unconditional Open-to-Open, 100% realistic)
candidates['ret_open_to_t1_open'] = (candidates['next_open'] - candidates['open']) / candidates['open'] - FRICTION
win_rate_open_t1 = (candidates['ret_open_to_t1_open'] > 0).mean() * 100
mean_ret_open_t1 = candidates['ret_open_to_t1_open'].mean() * 100

# Execution Mode 2: 开盘全买 + T+1 尾盘卖出 (Unconditional Open-to-T+1 Close)
candidates['ret_open_to_t1_close'] = (candidates['next_close'] - candidates['open']) / candidates['open'] - FRICTION
win_rate_open_t1_close = (candidates['ret_open_to_t1_close'] > 0).mean() * 100
mean_ret_open_t1_close = candidates['ret_open_to_t1_close'].mean() * 100

# Execution Mode 3: 真实打板策略 (Sealing Trigger):
# You ONLY buy if the stock touches limit price on Day t!
# If it never touches limit price, you DO NOT BUY (0 trade)!
# If you buy at limit price on Day t: you CANNOT sell on Day t!
# Exit strictly at Day t+1 OPEN:
touched_trades = candidates[candidates['is_touch_zt']].copy()
touched_trades['ret_zt_buy_t1_open'] = (touched_trades['next_open'] - touched_trades['limit_price']) / touched_trades['limit_price'] - FRICTION
win_rate_zt_t1 = (touched_trades['ret_zt_buy_t1_open'] > 0).mean() * 100
mean_ret_zt_t1 = touched_trades['ret_zt_buy_t1_open'].mean() * 100

# Execution Mode 4: 封板持筹溢价 (Only for stocks that sealed)
sealed_trades = candidates[candidates['is_zt']].copy()
sealed_trades['overnight_premium'] = (sealed_trades['next_open'] - sealed_trades['close']) / sealed_trades['close'] - FRICTION
win_rate_sealed = (sealed_trades['overnight_premium'] > 0).mean() * 100
mean_premium_sealed = sealed_trades['overnight_premium'].mean() * 100

print(f"\n[真实 A 股 T+1 交易纪律结算 (绝无日内 T+0 止损作弊)]:")
print(f"1. 开盘买入 -> 次日开盘卖出 (全池基准):")
print(f"   - 样本量: {len(candidates)}")
print(f"   - 胜率: {win_rate_open_t1:.2f}%")
print(f"   - 单笔平均净收益: {mean_ret_open_t1:+.2f}%")

print(f"\n2. 开盘买入 -> 次日收盘卖出:")
print(f"   - 样本量: {len(candidates)}")
print(f"   - 胜率: {win_rate_open_t1_close:.2f}%")
print(f"   - 单笔平均净收益: {mean_ret_open_t1_close:+.2f}%")

print(f"\n3. 打板买入 (仅在触及涨停时以涨停价买入) -> 次日开盘卖出:")
print(f"   - 触发买入样本量: {len(touched_trades)} (未触板不买，自动过滤 {len(candidates) - len(touched_trades)} 只走弱票)")
print(f"   - 封板成功率: {real_touch_to_seal:.2f}% (炸板率 {100 - real_touch_to_seal:.2f}%)")
print(f"   - 次日开盘卖出胜率: {win_rate_zt_t1:.2f}%")
print(f"   - 单笔平均净收益: {mean_ret_zt_t1:+.2f}%")

print(f"\n4. 封死涨停标的的【次日真实高开溢价】(条件胜率):")
print(f"   - 封死涨停样本量: {len(sealed_trades)}")
print(f"   - 次日高开获利胜率: {win_rate_sealed:.2f}%")
print(f"   - 次日开盘平均溢价: {mean_premium_sealed:+.2f}%")
