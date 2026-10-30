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

df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
df['vwap'] = np.where(df['volume'] > 0, df['amount'] / df['volume'], df['close'])

grp = df.groupby('code_str')
df['prev_zt'] = grp['is_zt'].shift(1).fillna(False)
df['prev_close'] = grp['close'].shift(1)
df['prev_high'] = grp['high'].shift(1)
df['prev_amount'] = grp['amount'].shift(1)
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

# Base condition
base_mask = (
    df['prev_zt'] & df['is_non_one_word'] &
    ~df['code_str'].str.startswith(('300', '301', '688')) & # 主板专注
    (df['open_gap'] >= 0.03) & (df['open_gap'] <= 0.075) & # 黄金高开 3%~7.5%
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) & # 趋势向上
    (df['prev_amount_yi'] >= 3.5) & # 适度大容量
    (df['prev_turnover'] >= 8.0) & (df['prev_turnover'] <= 28.0) # 换手健康
)

candidates = df[base_mask].copy()
print(f"Total base candidates: {len(candidates)}")

# 1. Execution A: 开盘无脑全买 (Blind Open Buy)
candidates['ret_blind_open'] = (candidates['next_open'] - candidates['open']) / candidates['open'] - 0.0012

# 2. Execution B: 打板确认买入 (Sealing Buy - Only buy if stock touches limit price!)
# If touched limit price, buy at limit price. If sealed, exit next day. If bombed, exit next day open or close.
candidates['bought_at_zt'] = candidates['is_touch_zt']
# Buy price = limit_price
# Next day exit price:
# If next day open is positive or negative, sell at next open
candidates['ret_zt_buy'] = np.where(
    candidates['bought_at_zt'],
    (candidates['next_open'] - candidates['limit_price']) / candidates['limit_price'] - 0.0012,
    0.0 # Did not touch limit price, 0 trade!
)

# 3. Execution C: 龙头战法标准持仓 (Dragon Trailing Exit):
# If sealed ZT on Day t, hold until broken, then sell!
# Let's calculate: for candidates where is_zt is True (successfully sealed)
sealed_trades = candidates[candidates['is_zt']].copy()
sealed_trades['next_day_premium'] = (sealed_trades['next_open'] - sealed_trades['close']) / sealed_trades['close'] - 0.0012
sealed_trades['next_day_close_ret'] = (sealed_trades['next_close'] - sealed_trades['close']) / sealed_trades['close'] - 0.0012

print("\n--- 模式对比统计 ---")
print(f"1. 开盘盲买模式 (全买，不论是否走强):")
print(f"   - 交易次数: {len(candidates)}")
print(f"   - 次日开盘胜率: {(candidates['ret_blind_open'] > 0).mean()*100:.2f}%")
print(f"   - 单笔均收益: {candidates['ret_blind_open'].mean()*100:+.2f}%")

print(f"\n2. 打板确认模式 (只买触板/封板标的，不碰走弱票):")
touched_trades = candidates[candidates['bought_at_zt']].copy()
print(f"   - 触发打板次数: {len(touched_trades)} (过滤掉了 {len(candidates) - len(touched_trades)} 只未上板走弱票!)")
print(f"   - 最终成功封板率: {(touched_trades['is_zt']).mean()*100:.2f}% (炸板率 {100 - (touched_trades['is_zt']).mean()*100:.2f}%)")
print(f"   - 次日开盘胜率: {(touched_trades['ret_zt_buy'] > 0).mean()*100:.2f}%")
print(f"   - 单笔均收益: {touched_trades['ret_zt_buy'].mean()*100:+.2f}%")

print(f"\n3. 成功封板后的次日真实溢价 (Overnight Premium for Sealed Dragons):")
print(f"   - 封死涨停总次数: {len(sealed_trades)}")
print(f"   - 次日高开/获利胜率: {(sealed_trades['next_day_premium'] > 0).mean()*100:.2f}%")
print(f"   - 次日开盘平均溢价: {sealed_trades['next_day_premium'].mean()*100:+.2f}%")
print(f"   - 次日收盘平均收益: {sealed_trades['next_day_close_ret'].mean()*100:+.2f}%")
