"""Apply the original formulas to newly bound alternative sources, without scoring."""
from copy import deepcopy
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import short_term_thinking_latest as latest
import short_term_thinking_dataset_builder as b
import model_upgrade_compat as compat
import public_outcome_sources_v293 as public
OUT=Path(__file__).resolve().parent
OLD15=ROOT/'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/alternative_source_projection'
def main():
 daily=b.read(OUT/'facts/normalized_daily.json');facts=b.read(OUT/'facts/qualified_current_facts_SCORES_MASKED.json');fmap={r['code']:r for r in facts['rows']}
 m15,p15,d15,fail15,_=latest.reconstruct()
 # Only exact observed bars are merged. Missing old member feedback stays unknown.
 extra={}
 prevpool=p15[('2026-10-15','limit_up')]
 for day in ['2026-10-16','2026-10-19']:
  parsed=b.read(OUT/'facts'/(day+'_complete_page_pools.json'));pools={}
  received=max(v['available_at'] for v in daily.values())
  for kind in ['limit_up','failed_board','limit_down']:
   rows=[]
   for r in parsed['rows']:
    if r['kind']!=kind or not r['scope']:continue
    c=r['code'];bar=daily[c]['days'][day]
    row=dict(code=c,name=r['name'],price_cents=bar['close'],turnover_pct=r['turnover_pct'],amount=fmap[c]['amount_cny'] if day=='2026-10-19' else None,industry=None)
    if kind=='limit_up':row['boards']=r['boards']
    rows.append(row)
   pools[(day,kind)]=dict(date=day,kind=kind,rows=rows,available_at=received,raw_path=str((OUT/'facts'/(day+'_complete_page_pools.json')).resolve()),capture_complete_within_alternative_page=True,exchange_wide_completeness_verified=False,amount_missing_not_replaced_with_later_quote=day=='2026-10-16')
  prior='2026-10-15' if day=='2026-10-16' else '2026-10-16';feedback=[]
  for r in prevpool['rows']:
   c=r['code'];pre=daily.get(c,{}).get('days',{}).get(prior);aft=daily.get(c,{}).get('days',{}).get(day)
   if pre and aft:feedback.append(dict(code=c,name=r['name'],price_cents=aft['close'],pct=(aft['close']/pre['close']-1)*100,derived_from_exact_day_prices=True))
  pools[(day,'previous_limit')]=dict(date=day,kind='DERIVED_PREVIOUS_COMPLETE_CLOSING_COHORT',rows=feedback,available_at=received)
  projected={c:dict(v,days={d:r for d,r in v['days'].items() if d<=day}) for c,v in daily.items()}
  sources={str((OUT/'facts'/p).resolve()):b.sha(OUT/'facts'/p) for p in ['normalized_daily.json','daily_manifest.json','pool_capture_manifest.json',day+'_complete_page_pools.json','603337_source_conflict_APPEND_CORRECTION.json']}
  sources.update({str(Path(v['raw_path']).resolve()):b.sha(v['raw_path']) for v in daily.values()})
  manifest=dict(version='NEW_ALTERNATIVE_DIAGNOSTIC_NOT_V293',target_date=day,signal_date=prior,recorded_at=received,requests=[],source_hashes=sources,source_conflict_603337='retained_outside39; original failed source disagrees closing daily')
  alt=OUT/'features'/('alternative_'+day)
  b.write_new(alt/'manifest.json',manifest);b.write_new(alt/'normalized.json',dict(pools={'/'.join(k):v for k,v in pools.items()}))
  extra[str(alt.resolve())]=(manifest,pools,projected,[]);prevpool=pools[(day,'limit_up')]
 adapter=latest.prior_adapter();inputs,_=compat.runtime(adapter.POLICY)
 om=b.read(adapter.OLD/'model/features/manifest.json')
 history=om['history_archives']+[om['source_archive'],str(adapter.ALT.resolve()),str(OLD15.resolve()),str((OUT/'features/alternative_2026-10-16').resolve())]
 def reader(path):
  p=str(Path(path).resolve())
  if p in extra:return deepcopy(extra[p])
  if Path(path).resolve()==OLD15.resolve():return latest.reconstruct()[:4]
  if Path(path).resolve()==adapter.ALT.resolve():return adapter.rebuild_alternative()[:4]
  return public.reconstruct(path)
 diagnostic=compat.clone_module(inputs,{'reconstruct':reader})
 pack=diagnostic._derive(adapter.PROTOCOL,OUT/'features/alternative_2026-10-19',adapter.CALENDAR,history,b.now())
 if (pack['signal_date'],pack['target_date'],len(pack['relay']))!=('2026-10-19','2026-10-20',11):raise ValueError('exact current date/full11 required')
 pack['source_status']='SUPPLEMENTAL_ALTERNATIVE_NOT_ORIGINAL_PRODUCTION_SOURCE'
 pack['source_conflicts_separate']=['603337'];b.write_new(OUT/'features/current29_BEFORE_MODEL.json',pack)
 b.write_new(OUT/'features/preparation_receipt.json',dict(formed_at=b.now(),full_relay=11,market_status=pack['market']['status'],unknown_fields=pack['market']['unknown_fields'],features_formula_unchanged=True,models_displayed=False,new_fits=0,original_source_gate_pass=False))
 print(json.dumps(dict(full_relay=11,market_status=pack['market']['status'],unknown_fields=pack['market']['unknown_fields']),ensure_ascii=False))
if __name__=='__main__':main()
