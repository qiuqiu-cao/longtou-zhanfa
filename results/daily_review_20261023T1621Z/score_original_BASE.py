"""Original BASE inference only after verified complete independent archive."""
from pathlib import Path
import sys,json
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import short_term_thinking_dataset_builder as b
import model_upgrade_compat as compat
from model_champion_compat import runtime
RUN=ROOT/'research/model_upgrade_validation'
POLICY=RUN/'shadow_compat_20261009T093237Z/compatibility_BEFORE_USE.json'
IMPLEMENTATION=POLICY.parent/'champion_implementation_BEFORE_USE.json'
PROTOCOL=RUN/'fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json'
def main():
    own=OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json';receipt=b.read(OUT/'INDEPENDENT_FREEZE_RECEIPT.json');independent=b.read(own)
    facts=b.read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json')
    assert b.sha(own)==receipt['sha256'] and receipt['full_rows']==len(facts['rows'])==len(independent['rows'])
    assert sorted(r['code'] for r in independent['rows'])==sorted(r['code'] for r in facts['rows'])
    assert independent['signal_date']=='2026-10-23' and independent['target_date']=='2026-10-26'
    assert independent['model_output_seen'] is False
    pack=b.read(OUT/'features/current29_BEFORE_MODEL.json');prep=b.read(OUT/'features/preparation_receipt.json')
    assert len(pack['relay'])==prep['full_relay']==11
    protected={str(p):b.sha(p) for p in [PROTOCOL,POLICY,IMPLEMENTATION,ROOT/'data/independent_stock_judgement/registry.jsonl',ROOT/'data/independent_stock_judgement/settlement_ledger.jsonl',own]}
    champion,proof=runtime(POLICY,IMPLEMENTATION);_,shadow=compat.runtime(POLICY)
    pp,protocol,_,root=shadow._protocol(PROTOCOL);pointer=shadow._pointer(root)
    assert pointer.get('modules',{}).get('relay',{'active_arm':'BASE'}).get('active_arm')=='BASE'
    models=shadow._models(pp,protocol,'relay',pp.parent/'models',shadow.now_utc());binding=models['BASE'];champion._validate_binding(pp,'relay','BASE',binding)
    b.write_new(OUT/'BASE_inference_BEFORE_LOAD.json',dict(formed_at=b.now(),independent_sha=b.sha(own),protected_hashes=protected,binding=binding,new_fits=0,source_gate='ALTERNATIVE_SOURCE_SUPPLEMENT_NOT_FORMAL',source_qualification=prep))
    forecast=champion._forecast('relay',binding,pack);lookup={r['code']:r for r in independent['rows']};rows=[]
    for r in forecast['candidates']:
        v=lookup[r['code']];rows.append(dict(r,score_display=round(r['raw_score']*100,1),rank=forecast['ranking'].index(r['code'])+1,BASE_direction='YES' if r['predicted_positive'] else 'NO',independent_direction=v['next_close_limit_up_call'],independent_rank=v.get('priority_rank')))
    b.write_new(OUT/'BASE_scores_AFTER_INDEPENDENT.json',dict(formed_at=b.now(),signal_date='2026-10-23',target_date='2026-10-26',rows=rows,ranking=forecast['ranking'],binding=binding,uncalibrated=True,new_fits=0,new_formal_predictions=0,original_models_retained=True,source_status='SUPPLEMENTAL_ALTERNATIVE_NOT_ORIGINAL_PRODUCTION_SOURCE',source_qualification=prep))
    assert all(b.sha(p)==h for p,h in protected.items())
    b.write_new(OUT/'BASE_verification.json',dict(formed_at=b.now(),complete11=True,independent_full81_unchanged=True,protected_files_unchanged=True,new_fits=0,source_gate_pass=False))
    print(json.dumps([dict(code=r['code'],name=r['name'],score=r['score_display'],rank=r['rank'],BASE=r['BASE_direction'],independent=r['independent_direction']) for r in sorted(rows,key=lambda x:x['rank'])],ensure_ascii=False))
if __name__=='__main__':main()
