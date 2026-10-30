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

# Moving averages on Day t-1
df['ma5'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(5).mean())
df['ma10'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(10).mean())
df['ma20'] = df.groupby('code_str')['close'].transform(lambda x: x.rolling(20).mean())
df['vwap'] = np.where(df['volume'] > 0, df['amount'] / df['volume'], df['close'])

grp = df.groupby('code_str')
df['prev_zt'] = grp['is_zt'].shift(1).fillna(False)
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

# 1. Baseline Formula
mask_baseline = (
    df['prev_zt'] & df['is_non_one_word'] &
    (df['open_gap'] >= 0.02) & (df['open_gap'] <= 0.07) &
    df['break_high'] &
    (df['prev_amount_yi'] >= 3.0) &
    (df['prev_turnover'] >= 8.0) & (df['prev_turnover'] <= 32.0)
)

# 2. Enhanced Formula: "极稳高收益·主板大容量真龙接力版"
# Key innovations:
# - Exclude non-mainboard (pure 10% mainboard)
# - Golden gap narrowed to +3.2% ~ +7.2% (cuts out 2.0%~3.1% weak trap gaps!)
# - Trend confirmation: MA5 > MA10 and CLOSE > MA20 (in active bull run)
# - Stand above yesterday VWAP (OPEN >= PREV_VWAP)
# - Healthier capacity: Amount >= 4.0亿, Turnover 10.0% ~ 26.0% (avoids exhausted 30%+ turnover)
mask_enhanced = (
    df['prev_zt'] & df['is_non_one_word'] &
    ~df['code_str'].str.startswith(('300', '301', '688')) &
    (df['open_gap'] >= 0.032) & (df['open_gap'] <= 0.072) &
    df['break_high'] & df['above_vwap'] &
    (df['prev_ma5'] > df['prev_ma10']) & (df['prev_close'] > df['prev_ma20']) &
    (df['prev_amount_yi'] >= 4.0) &
    (df['prev_turnover'] >= 10.0) & (df['prev_turnover'] <= 26.0)
)

# 3. Super Leader Variant: "2进3及以上·高位成妖主升版"
mask_super_dragon = (
    mask_enhanced & (df['prev_streak'] >= 2)
)

results = []
for name, m in [("1. 现有基准版 (全市场宽松版)", mask_baseline),
                ("2. 极稳高收益优化版 (主板多头+大容量黄金高开)", mask_enhanced),
                ("3. 高位成妖极稳版 (2板及以上主板龙头)", mask_super_dragon)]:
    sub = df[m].copy()
    n = len(sub)
    if n == 0:
        continue
    zt_rate = sub['is_zt'].mean() * 100
    touch_rate = sub['is_touch_zt'].mean() * 100
    
    # Intraday open to close return
    day_ret = ((sub['close'] - sub['open']) / sub['open'] - 0.0012).mean() * 100
    
    # Next day open return
    ret_t1_open = ((sub['next_open'] - sub['open']) / sub['open'] - 0.0012)
    win_rate = (ret_t1_open > 0).mean() * 100
    avg_ret_t1 = ret_t1_open.mean() * 100
    
    # Dragon trailing hold return (hold if sealed ZT)
    # If sealed, hold to t+1 close
    sealed_mask = sub['is_zt']
    unsealed_mask = ~sub['is_zt']
    
    # Trailing strategy return:
    # If sealed on day t: sell on day t+1 close!
    # If not sealed on day t: sell on day t+1 open!
    strategy_ret = np.where(
        sealed_mask,
        (sub['next_close'] - sub['open']) / sub['open'] - 0.0012,
        (sub['next_open'] - sub['open']) / sub['open'] - 0.0012
    )
    strat_mean = strategy_ret.mean() * 100
    strat_win = (strategy_ret > 0).mean() * 100
    
    # Gains / Losses
    gains = strategy_ret[strategy_ret > 0].sum()
    losses = abs(strategy_ret[strategy_ret < 0].sum())
    pf = gains / losses if losses > 0 else np.nan
    
    # Drawdown of strategy
    cum = (1 + strategy_ret).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min() * 100
    
    results.append({
        'name': name,
        'count': n,
        'zt_rate': zt_rate,
        'touch_rate': touch_rate,
        'day_ret': day_ret,
        'strat_win': strat_win,
        'strat_mean': strat_mean,
        'profit_factor': pf,
        'mdd': mdd
    })

for r in results:
    print(f"\n★ {r['name']}")
    print(f"  - 触发样本总数: {r['count']:,} 次")
    print(f"  - 当日封板率: {r['zt_rate']:.2f}% | 盘中触板率: {r['touch_rate']:.2f}%")
    print(f"  - 策略整体综合胜率: {r['strat_win']:.2f}%")
    print(f"  - 策略单笔平均净收益: {r['strat_mean']:+.2f}%")
    print(f"  - 盈亏比 (Profit Factor): {r['profit_factor']:.2f}:1")
