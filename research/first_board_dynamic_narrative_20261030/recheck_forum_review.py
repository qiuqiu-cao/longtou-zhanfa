"""Append a new projection after contract revision; preserve the first audit."""
import json
from pathlib import Path
from evidence_contract import check_claims

base=Path(__file__).resolve().parent
old=json.loads((base/'FORUM_CLAIM_TIME_AUDIT.json').read_text(encoding='utf-8'))
claims=next(iter(old['by_cutoff'].values()))['claims']
claims=[{k:v for k,v in c.items() if k not in ('source_eligibility','claim_actually_available_by_cutoff','is_confirmed_causal_leadership','claim_is_author_interpretation')} for c in claims]
results={}
for cutoff in old['by_cutoff']:
    results[cutoff]=check_claims({'cutoff':cutoff,'formed_at':old['reviewed_at'],
                                  'sources':{'post':old['source']},'claims':claims})
out=base/'FORUM_CLAIM_TIME_AUDIT_V2.json'
with out.open('x',encoding='utf-8') as f:
    json.dump({'source':old['source'],'reviewed_at':old['reviewed_at'],'by_cutoff':results},f,ensure_ascii=False,indent=2)
print(json.dumps({'cutoffs':len(results),'new_claims_available_at_cutoff':sum(c['claim_actually_available_by_cutoff'] for x in results.values() for c in x['claims'])}))
