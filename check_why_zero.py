import json

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily_27 = json.load(f)

with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_26 = json.load(f)
    rows_26 = {r['code']: r for r in facts_26.get('rows', [])}

print("=== 10-27 Morning: Strict vs Relaxed ===")

for code, r in rows_26.items():
    if not (code.startswith('000') or code.startswith('001') or code.startswith('002') or code.startswith('003') or code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605')):
        continue
    d_info = daily_27.get(code, {}).get('days', {})
    d27 = d_info.get('2026-10-27')
    d26 = d_info.get('2026-10-26')
    if not d27 or not d26:
        continue
    c26 = d26['close'] / 100.0
    o27 = d27['open'] / 100.0
    c27 = d27['close'] / 100.0
    h27 = d27['high'] / 100.0
    gap = (o27 - c26) / c26 * 100.0
    amt_26 = r.get('amount_cny', 0)
    to_26 = r.get('turnover_pct', 0)
    b_26 = r.get('boards', 1)
    
    # Check if 10-26 was limit up
    is_zt_26 = r.get('kind') == 'limit_up' or r.get('boards', 0) >= 1
    if not is_zt_26:
        continue
        
    # Check gap 3.5%~7.2%
    in_gap = (gap >= 3.5) and (gap <= 7.2)
    
    if in_gap:
        print(f"[{code}] {r.get('name')}: 板数={b_26}, 昨额={amt_26/1e8:.2f}亿, 昨换={to_26:.1f}%, 高开={gap:+.2f}%, 10-27涨幅={(c27-c26)/c26*100:+.2f}%")
