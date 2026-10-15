"""Read saved weights only after the complete independent views are on file."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import model_upgrade_compat as compat
from model_champion_compat import runtime
import short_term_thinking_workbench as thinking

OUT=Path(__file__).resolve().parent
RUN=ROOT/'research/model_upgrade_validation'
PREP=RUN/'workbench_latest_20261015T091200Z'
POLICY=RUN/'shadow_compat_20261009T093237Z/compatibility_BEFORE_USE.json'
IMPLEMENTATION=POLICY.parent/'champion_implementation_BEFORE_USE.json'
PROTOCOL=RUN/'fixed_20261007T112017Z/protocol_BEFORE_EFFECTS.json'
FEATURES=RUN/'historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/features_BEFORE_MODEL_DISPLAY.json'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    review_path=OUT/'independent_forward_views_BEFORE_MODEL.json'
    receipt=read(OUT/'independent_archive_receipt_BEFORE_MODEL.json')
    if receipt['independent_forward_sha256']!=sha(review_path) or not receipt['before_current_model_display']:
        raise ValueError('independent before-score archive changed')
    review=read(review_path)
    pack=read(FEATURES)
    if (pack['signal_date'],pack['target_date'])!=('2026-10-15','2026-10-16'):
        raise ValueError('wrong date cohort')
    if {r['code'] for r in pack['relay']}!={r['code'] for r in review['rows']}:
        raise ValueError('incomplete full-six comparison')
    protected={str(path):sha(path) for path in [PROTOCOL,
        ROOT/'data/independent_stock_judgement/registry.jsonl',
        ROOT/'data/independent_stock_judgement/settlement_ledger.jsonl',review_path]}
    champion,proof=runtime(POLICY,IMPLEMENTATION)
    _,shadow=compat.runtime(POLICY)
    protocol_path,protocol,_,root=shadow._protocol(PROTOCOL)
    pointer=shadow._pointer(root)
    if pointer.get('modules',{}).get('relay',{'active_arm':'BASE'}).get('active_arm')!='BASE':
        raise ValueError('BASE no longer active; report selection change first')
    models=shadow._models(protocol_path,protocol,'relay',protocol_path.parent/'models',shadow.now_utc())
    binding=models['BASE']
    champion._validate_binding(protocol_path,'relay','BASE',binding)
    before=dict(formed_at=now(),independent_review_path=str(review_path),independent_review_sha256=sha(review_path),
                full_pool_count=6,feature_path=str(FEATURES),feature_sha256=sha(FEATURES),
                base_binding=binding,strict_threshold=0.5,new_fits=0,
                scope='ALTERNATIVE_CLOSE_SUPPLEMENTAL_INFERENCE_NOT_FORMAL_FREEZE',protected_before=protected)
    write(OUT/'inference_receipt_BEFORE_MODEL_LOAD.json',before)
    forecast=champion._forecast('relay',binding,pack)
    rows=[]
    for row in forecast['candidates']:
        rows.append(dict(row,score_display=round(row['raw_score']*100,2),
                         rank=forecast['ranking'].index(row['code'])+1,
                         direction='PROMOTE' if row['predicted_positive'] else 'FAIL'))
    result=dict(status='BASE_ALTERNATIVE_SOURCE_SCORE_AFTER_INDEPENDENT_REVIEW',formed_at=now(),
                signal_date=pack['signal_date'],target_date=pack['target_date'],rows=rows,
                ranking=forecast['ranking'],score_calibrated_probability=False,
                source_gate='FORMAL_ORIGINAL_SOURCE_UNAVAILABLE',model_binding=binding,
                source_bindings=proof|{str(FEATURES):sha(FEATURES),str(review_path):sha(review_path)},
                new_fits=0,new_formal_predictions=0,registry_mutated=False)
    write(OUT/'BASE_scores_AFTER_INDEPENDENT_REVIEW.json',result)
    research=thinking.score_context(PREP/'symbolic/facts_with_symbolic_context.json',
        OUT/'completed_diagnostic_review_record.json',
        RUN/'short_term_thinking_20261015T071000Z/execution_after_binning_repair/2026-10_ST_DUAL_model.joblib',
        OUT/'ST_DUAL_shadow_scores_AFTER_INDEPENDENT_REVIEW.json')
    changed=[path for path,wanted in protected.items() if sha(path)!=wanted]
    if changed:raise ValueError('protected original artifacts changed: '+str(changed))
    write(OUT/'model_inference_verification.json',dict(formed_at=now(),full_pool_count=6,
        original_protected_hashes_unchanged=True,protected_hashes=protected,
        independent_before_model=True,base_rows=len(rows),shadow_rows=len(research['rows']),
        shadow_is_failed_research_model_not_deployed=True,new_fits=0,new_formal_predictions=0,
        new_actual_or_paper_fills=0))
    print(json.dumps(dict(BASE=rows,ST_DUAL=research['rows']),ensure_ascii=False))


if __name__=='__main__':main()
