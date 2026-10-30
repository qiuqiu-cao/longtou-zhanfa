import json

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

with open('results/daily_review_20261027T0928Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_27 = json.load(f)
    rows_27 = {r['code']: r for r in facts_27.get('rows', [])}

targets = ['000678', '600838', '600828', '001219']

for code in targets:
    info = daily.get(code, {})
    days = info.get('days', {})
    d28 = days.get('2026-10-28')
    d27 = days.get('2026-10-27')
    d26 = days.get('2026-10-26')
    d25 = days.get('2026-10-23') # Friday
    
    r27 = rows_27.get(code, {})
    
    print(f"\n--- Checking code {code} on 2026-10-28 ---")
    if not d28 or not d27:
        print("Missing day data")
        continue
        
    c28 = d28['close'] / 100.0
    o28 = d28['open'] / 100.0
    h28 = d28['high'] / 100.0
    
    c27 = d27['close'] / 100.0
    o27 = d27['open'] / 100.0
    h27 = d27['high'] / 100.0
    v27 = d27.get('volume_native', 0)
    
    c26 = d26['close'] / 100.0 if d26 else 0
    h26 = d26['high'] / 100.0 if d26 else 0
    
    amt_27 = r27.get('amount_cny', 0)
    to_27 = r27.get('turnover_pct', 0)
    
    # Check each condition:
    # 1. ZT1 & ZT2
    zt1 = (c27 / c26 >= 1.098)
    # ZT2
    zt2 = False
    if d25:
        c25 = d25['close'] / 100.0
        zt2 = (c26 / c25 >= 1.098)
    
    # Gap
    gap_pct = (o28 - c27) / c27 * 100.0
    golden_gap = (gap_pct >= 3.5) and (gap_pct <= 7.2)
    not_one = o28 < c27 * 1.098
    break_high = o28 >= h27
    
    # Healthy dragon
    # AMOUNT >= 450000000 and turnover between 10.0 and 28.0
    amt_pass = amt_27 >= 450000000
    to_pass = (to_27 >= 10.0) and (to_27 <= 28.0)
    healthy_dragon = amt_pass and to_pass
    
    # Above VWAP:
    # In Tonghuashun: OPEN * REF(VOL,1) * 100 >= REF(AMOUNT,1)
    # Let's check what REF(VOL,1)*100 vs REF(AMOUNT,1) is in Tonghuashun:
    # In Tonghuashun, VOL is in lots (1 lot = 100 shares).
    # Does REF(VOL,1)*100 match shares?
    
    print(f"Name: {r27.get('name')}")
    print(f"  10-27 Close={c27}, 10-28 Open={o28} (Gap={gap_pct:+.2f}%)")
    print(f"  ZT1 (10-27 ZT): {zt1} | ZT2 (10-26 ZT): {zt2}")
    print(f"  Golden Gap [3.5-7.2]: {golden_gap}")
    print(f"  Not One Word: {not_one}")
    print(f"  Break High (o28 >= h27): {break_high} ({o28} >= {h27})")
    print(f"  10-27 Amount: {amt_27/1e8:.2f}亿 (>=4.5亿: {amt_pass})")
    print(f"  10-27 Turnover: {to_27:.2f}% (10-28%: {to_pass})")
    print(f"  Healthy Dragon: {healthy_dragon}")
