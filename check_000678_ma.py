import json

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

d = daily.get('000678', {}).get('days', {})
dates = sorted(d.keys())
print("Total dates for 000678:", len(dates))
closes = [d[dt]['close']/100.0 for dt in dates]
dates_series = dates

import pandas as pd
s = pd.Series(closes, index=dates)
ma5 = s.rolling(5).mean()
ma10 = s.rolling(10).mean()
ma20 = s.rolling(20).mean()

df_ma = pd.DataFrame({'close': s, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20})
print(df_ma.tail(8))

# Check on 2026-10-27:
c_27 = df_ma.loc['2026-10-27', 'close']
m5_27 = df_ma.loc['2026-10-27', 'ma5']
m10_27 = df_ma.loc['2026-10-27', 'ma10']
m20_27 = df_ma.loc['2026-10-27', 'ma20']

print("\nOn 2026-10-27:")
print("Close:", c_27)
print("MA5:", m5_27)
print("MA10:", m10_27)
print("MA20:", m20_27)
print("m5 > m10:", m5_27 > m10_27)
print("m10 > m20:", m10_27 > m20_27)
print("close > m5:", c_27 > m5_27)
print("TREND_BULL:", (m5_27 > m10_27) and (m10_27 > m20_27) and (c_27 > m5_27))
