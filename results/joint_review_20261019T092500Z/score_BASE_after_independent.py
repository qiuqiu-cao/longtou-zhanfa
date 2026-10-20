"""Read immutable BASE only after full independent review and external review."""
from datetime import datetime,timezone
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import short_term_thinking_dataset_builder as b
import model_upgrade_compat as compat
from model_champion_compat import runtime
OUT=Path(__file__).resolve().parent
RUN=ROOT/'research/model_upgrade_validation'
POLICY=RUN/'shadow_compat_20261009T093237Z/compatibility_BEFORE_USE.json'
IMPLEMENTATION=POLICY.parent/'champion_implementation_BEFORE_USE.json'
PROTOCOL=RUN/'fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json'
def main():
 own=OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json';receipt=b.read(OUT/'independent_archive_receipt_BEFORE_MODEL.json');gemini=OUT/'gemini/gemini_independent_20261019_BEFORE_CODEX_MODEL.md'
 if b.sha(own)!=receipt['sha256'] or not gemini.exists():raise ValueError('complete independent records required')
 pack_path=OUT/'features/current29_BEFORE_MODEL.json';pack=b.read(pack_path);views=b.read(own);vm={r['code']:r for r in views['rows']};assert len(pack['relay'])==11
 protected={str(p):b.sha(p) for p in [PROTOCOL,ROOT/'data/independent_stock_judgement/registry.jsonl',ROOT/'data/independent_stock_judgement/settlement_ledger.jsonl',own,gemini]}
 champion,proof=runtime(POLICY,IMPLEMENTATION);_,shadow=compat.runtime(POLICY)
 pp,protocol,_,root=shadow._protocol(PROTOCOL);pointer=shadow._pointer(root)
 if pointer.get('modules',{}).get('relay',{'active_arm':'BASE'}).get('active_arm')!='BASE':raise ValueError('production selection changed')
 models=shadow._models(pp,protocol,'relay',pp.parent/'models',shadow.now_utc());binding=models['BASE'];champion._validate_binding(pp,'relay','BASE',binding)
 b.write_new(OUT/'BASE_inference_BEFORE_LOAD.json',dict(formed_at=b.now(),independent_sha=b.sha(own),gemini_independent_sha=b.sha(gemini),feature_sha=b.sha(pack_path),full_relay=11,new_fits=0,binding=binding,protected_hashes=protected,source_scope='SUPPLEMENTAL_ALTERNATIVE_FORMULAS_NOT_ORIGINAL_FORMAL_GATE'))
 forecast=champion._forecast('relay',binding,pack);rows=[]
 for r in forecast['candidates']:
  v=vm[r['code']];rows.append(dict(r,score_display=round(r['raw_score']*100,2),rank=forecast['ranking'].index(r['code'])+1,BASE_direction='YES' if r['predicted_positive'] else 'NO',independent_direction=v['next_close_limit_up_call'],independent_priority=v['priority_rank'],conflict=v['next_close_limit_up_call']!='UNKNOWN' and (r['predicted_positive']!=(v['next_close_limit_up_call']=='YES'))))
 b.write_new(OUT/'BASE_scores_AFTER_INDEPENDENT.json',dict(formed_at=b.now(),signal_date='2026-10-19',target_date='2026-10-20',rows=rows,ranking=forecast['ranking'],uncalibrated_scores_not_probabilities=True,new_fits=0,new_formal_predictions=0,source_gate='ORIGINAL_FORMAL_SOURCE_GATE_NOT_PASSED',binding=binding))
 if any(b.sha(p)!=h for p,h in protected.items()):raise ValueError('protected file changed')
 b.write_new(OUT/'BASE_inference_verification.json',dict(formed_at=b.now(),full_count=11,protected_hashes_unchanged=True,protected_hashes=protected,model_refits=0,independent_calls_unchanged=True))
 print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
