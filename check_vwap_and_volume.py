import json

with open('results/daily_review_20261028T1020Z/facts/normalized_daily.json', 'r', encoding='utf-8') as f:
    daily = json.load(f)

d = daily.get('000678', {}).get('days', {})
d27 = d.get('2026-10-27')
d28 = d.get('2026-10-28')

print("000678 10-27 bar:", d27)
print("000678 10-28 bar:", d28)

with open('results/daily_review_20261027T0928Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    facts_27 = json.load(f)
    for r in facts_27.get('rows', []):
        if r['code'] == '000678':
            amt = r.get('amount_cny')
            vol_shares = r.get('source_rows', {}).get('primary_limit_up', {})
            print("Facts 27 row:", r.get('close'), "amount:", amt)
            # VWAP = amt / volume
            # If in Tonghuashun VOL is in 手 (lots), VOL_lots = volume_shares / 100
            # Then OPEN * VOL_lots * 100 = OPEN * volume_shares
            # If OPEN >= VWAP, then OPEN * volume_shares >= amt!
            # Let's check 10-27 VWAP of 000678:
            vwap = amt / (d27['volume_native'] * 100) # Wait, is volume_native in shares or lots?
            print("volume_native:", d27['volume_native'])
            vwap_test1 = amt / d27['volume_native']
            vwap_test2 = amt / (d27['volume_native'] * 100)
            print("vwap_test1 (if volume_native is shares):", vwap_test1)
            print("vwap_test2 (if volume_native is lots):", vwap_test2)
            print("10-28 Open:", d28['open']/100.0)
