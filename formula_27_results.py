import json

with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    d26_data = json.load(f)
    rows_26 = {r['code']: r for r in d26_data.get('rows', [])}

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

with open('results/daily_review_20261027T0928Z/facts/2026-10-27_previous_limit_primary.json', 'r', encoding='utf-8') as f:
    prev_limit_data = json.load(f)
    prev_limit_rows = {r['code']: r for r in prev_limit_data.get('rows', [])}

name_map = {}
for code, r in rows_26.items():
    name_map[code] = r.get('name')
for code, r in prev_limit_rows.items():
    if code not in name_map:
        name_map[code] = r.get('name')

output_lines = []

for code, r_prev in prev_limit_rows.items():
    name = name_map.get(code, code)
    r26 = rows_26.get(code)
    stock_daily = daily.get(code, {}).get('days', {})
    d26_bar = stock_daily.get('2026-10-26')
    d27_bar = stock_daily.get('2026-10-27')
    
    if not d26_bar or not d27_bar:
        continue
        
    c26 = d26_bar['close'] / 100.0
    h26 = d26_bar['high'] / 100.0
    amt_26 = r26.get('amount_cny') if r26 else None
    if amt_26 is None:
        amt_26 = r26.get('primary_amount_cny') if r26 else None
    to_26 = r26.get('turnover_pct') if r26 else None
    
    o27 = d27_bar['open'] / 100.0
    c27 = d27_bar['close'] / 100.0
    h27 = d27_bar['high'] / 100.0
    l27 = d27_bar['low'] / 100.0
    
    gap_pct = (o27 - c26) / c26 * 100.0
    day_pct = (c27 - c26) / c26 * 100.0
    
    is_20 = code.startswith('300') or code.startswith('301') or code.startswith('688')
    limit_mult = 1.20 if is_20 else 1.10
    limit_price_27 = round(c26 * limit_mult, 2)
    
    not_one = o27 < (limit_price_27 - 0.01)
    golden_gap = (gap_pct >= 2.0) and (gap_pct <= 7.0)
    break_high = o27 >= h26
    
    capacity_pass = False
    if amt_26 is not None and to_26 is not None:
        capacity_pass = (amt_26 >= 300000000) and (to_26 >= 8.0) and (to_26 <= 32.0)
        
    is_zt = (c27 >= limit_price_27 - 0.01)
    is_touch = (h27 >= limit_price_27 - 0.01)
    
    output_lines.append({
        'code': code,
        'name': name,
        'c26': c26,
        'amt_26': amt_26,
        'to_26': to_26,
        'o27': o27,
        'gap_pct': gap_pct,
        'c27': c27,
        'h27': h27,
        'day_pct': day_pct,
        'is_zt': is_zt,
        'is_touch': is_touch,
        'strict_pass': not_one and golden_gap and break_high and capacity_pass,
        'gap_pass': not_one and golden_gap and break_high
    })

with open('eval_output_27.json', 'w', encoding='utf-8') as f:
    json.dump(output_lines, f, ensure_ascii=False, indent=2)

print("Saved eval_output_27.json successfully.")
