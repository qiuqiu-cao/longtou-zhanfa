import json

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

with open('results/daily_review_20261027T0928Z/facts/2026-10-27_previous_limit_primary.json', 'r', encoding='utf-8') as f:
    prev_limit_data = json.load(f)
    prev_limit_rows = prev_limit_data.get('rows', [])

# Load name mapping from security master or another file if available
name_map = {}
for r in prev_limit_rows:
    name_map[r['code']] = r['name']

with open('results/daily_review_20261027T0928Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    enriched_data = json.load(f)
    for r in enriched_data.get('rows', []):
        if r.get('code') and r.get('name'):
            name_map[r['code']] = r['name']

print("Evaluating all 47 candidates for 2026-10-27 morning (Yesterday = 2026-10-26, Today = 2026-10-27)...")

passed_formula = []
rejected = []

for r in prev_limit_rows:
    code = r['code']
    name = name_map.get(code, r.get('name', ''))
    stock_daily = daily.get(code, {}).get('days', {})
    d26 = stock_daily.get('2026-10-26')
    d27 = stock_daily.get('2026-10-27')
    
    if not d26 or not d27:
        continue
    
    # In normalized_daily, prices are in cents (e.g. 1524 = 15.24)
    # Check 10-26 values
    c26 = d26['close'] / 100.0
    h26 = d26['high'] / 100.0
    o26 = d26['open'] / 100.0
    v26 = d26.get('volume_native', 0)
    
    # Check 10-27 values
    o27 = d27['open'] / 100.0
    c27 = d27['close'] / 100.0
    h27 = d27['high'] / 100.0
    l27 = d27['low'] / 100.0
    v27 = d27.get('volume_native', 0)
    
    gap_pct = (o27 - c26) / c26 * 100.0
    day_pct = (c27 - c26) / c26 * 100.0
    
    # Formula Conditions:
    # 1. NOT_ONE: o27 < c26 * 1.10 - 0.005 (not one word board)
    not_one = o27 < (c26 * 1.098)
    
    # 2. GOLDEN_GAP: gap_pct >= 2.0 and gap_pct <= 7.0
    golden_gap = (gap_pct >= 2.0) and (gap_pct <= 7.0)
    
    # 3. BREAK_HIGH: o27 >= h26 (since 10-26 was limit up, h26 == c26)
    break_high = o27 >= h26
    
    # 4. Capacity & Turnover on 10-26:
    # Let's check 10-26 turnover & amount from previous review or facts
    # Note: r['turnover_pct'] in 2026-10-27_previous_limit_primary is 10-27 turnover or 10-26?
    # Let's check!
    
    passed = not_one and golden_gap and break_high
    
    item = {
        'code': code,
        'name': name,
        'c26': c26,
        'o27': o27,
        'gap_pct': gap_pct,
        'c27': c27,
        'day_pct': day_pct,
        'h27': h27,
        'is_zt': day_pct >= 9.8,
        'not_one': not_one,
        'golden_gap': golden_gap,
        'break_high': break_high,
        'd27_amt': r.get('amount', 0),
        'd27_to': r.get('turnover_pct', 0)
    }
    
    if passed:
        passed_formula.append(item)
    else:
        rejected.append(item)

print(f"\n=== PASSED GOLDEN GAP (+2%~+7%) & NON-ONE-WORD on 2026-10-27: {len(passed_formula)} stocks ===")
for p in passed_formula:
    print(f"[{p['code']}] {p['name']}: 昨收={p['c26']:.2f}, 开盘={p['o27']:.2f} (高开={p['gap_pct']:+.2f}%), 收盘={p['c27']:.2f} (涨幅={p['day_pct']:+.2f}%), 涨停={p['is_zt']}")

print(f"\n=== REJECTED Breakdown (Total {len(rejected)}) ===")
low_open = [x for x in rejected if x['gap_pct'] < 2.0]
high_one = [x for x in rejected if x['gap_pct'] > 7.0]
print(f"- 低开或平开 (<+2%): {len(low_open)} 只 (大多冲高回落大跌，如正裕工业、赛福天等)")
print(f"- 顶格或超高开 (>+7%): {len(high_one)} 只 (如青岛食品 +9.99% 一字、多伦科技等)")
