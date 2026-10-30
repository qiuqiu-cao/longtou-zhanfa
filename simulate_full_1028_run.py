import json

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

print(f"Total stocks in 10-28 normalized_daily: {len(daily)}")

passed_1028 = []

for code, info in daily.items():
    # 1. Mainboard
    if not (code.startswith('000') or code.startswith('001') or code.startswith('002') or code.startswith('003') or code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605')):
        continue
        
    days = info.get('days', {})
    d28 = days.get('2026-10-28')
    d27 = days.get('2026-10-27')
    d26 = days.get('2026-10-26')
    d25 = days.get('2026-10-23')
    
    if not d28 or not d27 or not d26 or not d25:
        continue
        
    c28 = d28['close'] / 100.0
    o28 = d28['open'] / 100.0
    h28 = d28['high'] / 100.0
    
    c27 = d27['close'] / 100.0
    o27 = d27['open'] / 100.0
    h27 = d27['high'] / 100.0
    v27 = d27.get('volume_native', 0)
    
    c26 = d26['close'] / 100.0
    h26 = d26['high'] / 100.0
    
    c25 = d25['close'] / 100.0
    h25 = d25['high'] / 100.0
    
    # Condition 2: LIANBAN (ZT1 and ZT2)
    zt1 = (c27 / c26 >= 1.098) and (c27 == h27)
    zt2 = (c26 / c25 >= 1.098) and (c26 == h26)
    lianban = zt1 and zt2
    if not lianban:
        continue
        
    # Condition 3: NOT_ONE
    not_one = o28 < c27 * 1.098
    if not not_one:
        continue
        
    # Condition 4: GOLDEN_GAP (3.5% to 7.2%)
    gap_pct = (o28 - c27) / c27 * 100.0
    golden_gap = (gap_pct >= 3.5) and (gap_pct <= 7.2)
    if not golden_gap:
        continue
        
    # Condition 5: BREAK_HIGH and ABOVE_VWAP
    break_high = o28 >= h27
    if not break_high:
        continue
        
    passed_1028.append({
        'code': code,
        'c27': c27,
        'o28': o28,
        'gap_pct': gap_pct,
        'c28': c28
    })

print(f"Passed on 10-28: {len(passed_1028)}")
for p in passed_1028:
    print(p)
