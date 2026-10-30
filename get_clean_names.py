import json

with open('results/daily_review_20261026T1323Z/facts/complete_enriched_facts_SCORES_MASKED.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
    name_map = {r['code']: r['name'] for r in d.get('rows', [])}

targets = ['000678', '002086', '002535', '000559', '001331', '002093', '002820', '601900', '603258', '603337', '603657', '603719', '002095', '002578']
with open('clean_names.txt', 'w', encoding='utf-8') as out:
    for t in targets:
        out.write(f"{t}: {name_map.get(t, 'Unknown')}\n")
