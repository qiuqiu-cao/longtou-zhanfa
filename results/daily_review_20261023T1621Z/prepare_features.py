"""Original frozen formulas; new full declared source scope; no model scores."""
from copy import deepcopy
from pathlib import Path
import json,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
import short_term_thinking_latest as latest
import short_term_thinking_dataset_builder as b
import model_upgrade_compat as compat
import public_outcome_sources_v293 as public
OLD=ROOT/'results/joint_review_20261019T092500Z'
OLD15=ROOT/'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/alternative_source_projection'
DAY='2026-10-23';PRIOR='2026-10-22';TARGET='2026-10-26'

def main():
    daily=b.read(OUT/'facts/normalized_daily.json');facts=b.read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json')
    assert facts['signal_date']==DAY and facts['target_date']==TARGET
    expected_relay=[r for r in facts['rows'] if r['kind']=='limit_up' and r['boards']>=2]
    pools={};missing_current=[]
    for kind in ['limit_up','failed_board','limit_down']:
        rows=[]
        for r in facts['rows']:
            if r['kind']!=kind:continue
            item=daily.get(r['code'],{}).get('days',{}).get(DAY)
            if not item:
                missing_current.append(dict(code=r['code'],kind=kind));continue
            row=dict(code=r['code'],name=r['name'],price_cents=item['close'],turnover_pct=r['turnover_pct'],amount=r.get('amount_cny'),industry=r.get('industry'))
            if kind=='limit_up':row['boards']=r['boards']
            rows.append(row)
        pools[(DAY,kind)]=dict(date=DAY,kind=kind,rows=rows,available_at=facts['formed_at'],capture_complete_within_alternative_page=True,exchange_wide_completeness_verified=False)
    oldfacts=b.read(ROOT/'results/daily_review_20261022T1608Z/facts/qualified_full_facts_SCORES_MASKED.json')
    priorlimits=[r for r in oldfacts['rows'] if r['kind']=='limit_up'];feedback=[];missing_prior=[]
    for r in priorlimits:
        item=daily.get(r['code'],{}).get('days',{});a=item.get(PRIOR);z=item.get(DAY)
        if a and z:feedback.append(dict(code=r['code'],name=r['name'],price_cents=z['close'],pct=(z['close']/a['close']-1)*100,derived_from_exact_day_prices=True))
        else:missing_prior.append(r['code'])
    pools[(DAY,'previous_limit')]=dict(date=DAY,kind='DERIVED_PREVIOUS_COMPLETE_CLOSING_COHORT',rows=feedback,available_at=facts['formed_at'],complete_prior_rows=len(priorlimits),missing_codes=missing_prior)
    manifest=dict(version='NEW_ALTERNATIVE_DIAGNOSTIC_NOT_V293',target_date=DAY,signal_date=PRIOR,recorded_at=facts['formed_at'],requests=[],source_hashes=facts['source_bindings'],source_scope='Full mainboard primary/page union; exact-day price limitations retained.',missing_current_rows=missing_current,missing_previous_feedback_codes=missing_prior)
    alt=OUT/('features/alternative_'+DAY);b.write_new(alt/'manifest.json',manifest);b.write_new(alt/'normalized.json',dict(pools={'/'.join(k):v for k,v in pools.items()}))
    olddaily=b.read(OLD/'facts/normalized_daily.json')
    def old_extra(path):
        m=b.read(path/'manifest.json');p=b.read(path/'normalized.json')['pools'];p={tuple(k.split('/')):v for k,v in p.items()};d=max(k[0] for k in p)
        projected={c:dict(v,days={k:r for k,r in v['days'].items() if k<=d}) for c,v in olddaily.items()}
        return m,p,projected,[]
    extra={str(alt.resolve()):(manifest,pools,daily,[])}
    for date in ['2026-10-16','2026-10-19']:
        p=OLD/'features'/('alternative_'+date);extra[str(p.resolve())]=old_extra(p)
    newer=[('2026-10-20','daily_review_20261020T101630Z'),('2026-10-21','joint_review_20261021T084800Z'),('2026-10-22','daily_review_20261022T1608Z')]
    newer_paths=[]
    for day,folder in newer:
        directory=ROOT/'results'/folder;p=directory/'features'/('alternative_'+day)
        m=b.read(p/'manifest.json');pp={tuple(k.split('/')):v for k,v in b.read(p/'normalized.json')['pools'].items()}
        dd=b.read(directory/'facts/normalized_daily.json');extra[str(p.resolve())]=(m,pp,dd,[]);newer_paths.append(str(p.resolve()))
    adapter=latest.prior_adapter();inputs,_=compat.runtime(adapter.POLICY)
    om=b.read(adapter.OLD/'model/features/manifest.json')
    history=om['history_archives']+[om['source_archive'],str(adapter.ALT.resolve()),str(OLD15.resolve())]+[str((OLD/'features'/('alternative_'+d)).resolve()) for d in ['2026-10-16','2026-10-19']]+newer_paths
    def reader(path):
        s=str(Path(path).resolve())
        if s in extra:return deepcopy(extra[s])
        if Path(path).resolve()==OLD15.resolve():return latest.reconstruct()[:4]
        if Path(path).resolve()==adapter.ALT.resolve():return adapter.rebuild_alternative()[:4]
        return public.reconstruct(path)
    runtime=compat.clone_module(inputs,{'reconstruct':reader})
    pack=runtime._derive(adapter.PROTOCOL,alt,adapter.CALENDAR,history,b.now())
    assert (pack['signal_date'],pack['target_date'])==(DAY,TARGET)
    assert sorted(x['code'] for x in pack['relay'])==sorted(x['code'] for x in expected_relay),'Missing relay cannot be silently omitted'
    pack['source_status']='SUPPLEMENTAL_ALTERNATIVE_NOT_ORIGINAL_PRODUCTION_SOURCE'
    b.write_new(OUT/'features/current29_BEFORE_MODEL.json',pack)
    b.write_new(OUT/'features/preparation_receipt.json',dict(formed_at=b.now(),signal_date=DAY,target_date=TARGET,full_relay=len(expected_relay),models_displayed=False,original_formula_unchanged=True,new_fits=0,market_status=pack['market']['status'],unknown_fields=pack['market']['unknown_fields'],formal_source_gate_pass=False,missing_current_rows=missing_current,complete_previous_limits=len(priorlimits),previous_feedback_known=len(feedback),missing_previous_feedback_codes=missing_prior,history_archives=history,source_script_sha256=b.sha(__file__)))
    print(json.dumps({'full_relay':len(expected_relay),'market_status':pack['market']['status'],'unknown_fields':pack['market']['unknown_fields'],'models_displayed':False},ensure_ascii=False))
if __name__=='__main__':main()
