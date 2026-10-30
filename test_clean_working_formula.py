import json

with open('results/daily_review_20261027T0928Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily_27 = json.load(f)

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily_28 = json.load(f)

with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_26 = json.load(f)
    rows_26 = {r['code']: r for r in facts_26.get('rows', [])}

with open('results/daily_review_20261027T0928Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_27 = json.load(f)
    rows_27 = {r['code']: r for r in facts_27.get('rows', [])}

print("=== Evaluating Clean Formula on 10-27 ===")
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
    gap = (o27 - c26) / c26 * 100.0
    amt_26 = r.get('amount_cny', 0)
    to_26 = r.get('turnover_pct', 0)
    
    b_val = r.get('boards')
    is_zt_26 = (r.get('kind') == 'limit_up') or (b_val is not None and b_val >= 1)
    if not is_zt_26:
        continue
    if gap < 3.5 or gap > 8.0:
        continue
    if o27 >= c26 * 1.098:
        continue
    if amt_26 < 300000000 or to_26 < 8.0 or to_26 > 28.0:
        continue
    print(f"10-27 Hit: [{code}] {r.get('name')}: 高开={gap:+.2f}%, 昨额={amt_26/1e8:.2f}亿, 昨换={to_26:.1f}%, 10-27涨幅={(c27-c26)/c26*100:+.2f}%")

print("\n=== Evaluating Clean Formula on 10-28 ===")
for code, r in rows_27.items():
    if not (code.startswith('000') or code.startswith('001') or code.startswith('002') or code.startswith('003') or code.startswith('600') or code.startswith('601') or code.startswith('603') or code.startswith('605')):
        continue
    d_info = daily_28.get(code, {}).get('days', {})
    d28 = d_info.get('2026-10-28')
    d27 = d_info.get('2026-10-27')
    if not d28 or not d27:
        continue
    c27 = d27['close'] / 100.0
    o28 = d28['open'] / 100.0
    c28 = d28['close'] / 100.0
    gap = (o28 - c27) / c27 * 100.0
    amt_27 = r.get('amount_cny', 0)
    to_27 = r.get('turnover_pct', 0)
    
    b_val28 = r.get('boards')
    is_zt_27 = (r.get('kind') == 'limit_up') or (b_val28 is not None and b_val28 >= 1)
    if not is_zt_27:
        continue
    if gap < 3.5 or gap > 8.0:
        continue
    if o28 >= c27 * 1.098:
        continue
    if amt_27 < 300000000 or to_27 < 8.0 or to_27 > 28.0:
        continue
    print(f"10-28 Hit: [{code}] {r.get('name')}: 高开={gap:+.2f}%, 昨额={amt_27/1e8:.2f}亿, 昨换={to_27:.1f}%, 10-28涨幅={(c28-c27)/c27*100:+.2f}%")
