import json
import pandas as pd

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

d = daily.get('000678', {}).get('days', {})
dates = sorted(d.keys())
closes = [d[dt]['close']/100.0 for dt in dates]
s = pd.Series(closes, index=dates)
df_ma = pd.DataFrame({
    'close': s,
    'ma5': s.rolling(5).mean(),
    'ma10': s.rolling(10).mean(),
    'ma20': s.rolling(20).mean()
})

print("Checking 000678 on 2026-10-26 (which is REF(...,1) when running on 10-27):")
r26 = df_ma.loc['2026-10-26']
print(r26)
print("MA5:", r26['ma5'])
print("MA10:", r26['ma10'])
print("MA20:", r26['ma20'])
print("MA5 > MA10:", r26['ma5'] > r26['ma10'])
print("MA10 > MA20:", r26['ma10'] > r26['ma20'])
print("Is MA10 > MA20 on 10-26?", r26['ma10'] > r26['ma20'])
