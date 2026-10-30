"""Additive close-time inference of Gemini's saved R7; no fitting or orders."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, math, sys
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DAY='2026-10-28'; PRIOR='2026-10-27'; TARGET='2026-10-29'
REVIEW=ROOT/'results/daily_review_20261028T1020Z'
R7=ROOT/'research/model_upgrade_validation/dragon_round7_capacity_dragon_20261028T094500Z'
R6=ROOT/'research/model_upgrade_validation/dragon_round6_repaired_20261027T180000Z'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    p=OUT/name
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not math.isfinite(v):return None
    if isinstance(v,np.generic):return v.item()
    return v
def locked(r):
    return float(r.get('seal_first') is not None and r['seal_first']<=92501
                 and r.get('seal_last') is not None and r['seal_last']<=92501
                 and r.get('reported_open_count')==0 and r.get('turnover_pct',100)<3)

def main():
    facts=read(REVIEW/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json')
    prior=read(ROOT/'results/daily_review_20261027T0928Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json')
    pack=read(REVIEW/'features/current29_BEFORE_MODEL.json')
    feedback=read(REVIEW/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json')
    cur=[r for r in facts['rows'] if r['kind']=='limit_up']
    prev=[r for r in prior['rows'] if r['kind']=='limit_up']
    byprev={r['code']:r for r in prev};bycur={r['code']:r for r in cur}
    relay=[r for r in cur if r['boards']>=2]
    three=[r for r in relay if r['boards']==3]
    assert len(cur)==59 and len(relay)==8 and len(three)==2
    cols=read(R7/'feature_order.json');raw=cols[:70]
    basic=ROOT/'research/first_board_github_industry_qualification_20261020/stock_basics_industry_original.csv'
    basics=pd.read_csv(basic,dtype={'symbol':str})
    indmap=dict(zip(basics['symbol'].str.zfill(6),basics['industry']))
    def industry(code):
        v=indmap.get(code)
        return v if isinstance(v,str) and v.strip() else 'UNKNOWN_'+code
    def theme_features(r):
        ind=industry(r['code']); peers=[p for p in cur if industry(p['code'])==ind]
        old=[p for p in prev if industry(p['code'])==ind]
        n=len(peers); f=sum(p['boards']==1 for p in peers);two=sum(p['boards']==2 for p in peers);th=sum(p['boards']==3 for p in peers)
        # The original training denominator contains amounts for >=2-board candidates only.
        a=sum(p['amount_cny'] for p in peers if p['boards']>=2)
        at=sum(p['amount_cny'] for p in cur if p['boards']>=2)
        firstold=sum(p['boards']==1 for p in old)
        return dict(theme_cluster_total_zt=float(n),theme_first_board_count=float(f),theme_two_board_count=float(two),theme_three_board_count=float(th),
          theme_zt_share=n/len(cur),theme_first_board_share=f/sum(p['boards']==1 for p in cur),theme_amount_share=a/at,
          theme_ladder_density=(float(f>0)+float(two>0)+float(th>0))/3,theme_ladder_complete=float(f>0 and two>0),
          theme_first_board_expansion=float(f-firstold),theme_cluster_expansion=float(n-len(old)),
          is_theme_sole_leader=float(max(p['boards'] for p in peers)==3 and th==1),theme_same_height_rivals=float(max(0,th-1)),
          is_isolated_lone_wolf=float(f==0 and two==0),theme_diffusion_surge=float(f>=3 and f-firstold>=1),
          is_industry_unknown=float(ind.startswith('UNKNOWN_')))
    def rate(group):
        return float(np.mean([bycur.get(r['code'],{}).get('boards')==r['boards']+1 for r in group])) if group else np.nan
    m=dict(pack['market']['features'])
    m.update(emotion_score=pack['market']['quality'],limit_count_scaled=min(1,len(cur)/100),consecutive_scaled=min(1,len(relay)/30),
      down_limit_rate=min(1,m['down_limit_count']/10),height_change_scaled=min(1,max(0,(m['height_change']+3)/6)),
      limit_change_scaled=min(1,max(0,(len(cur)-len(prev)+80)/160)),promotion_overall=rate(prev),
      promotion_1to2=rate([p for p in prev if p['boards']==1]),promotion_2to3=rate([p for p in prev if p['boards']==2]),
      promotion_3to4=rate([p for p in prev if p['boards']==3]),promotion_mid=rate([p for p in prev if 2<=p['boards']<=4]),
      space_feedback=rate([p for p in prev if p['boards']==max(x['boards'] for x in prev)]),
      failed_tail_loss=m['previous_limit_tail_loss'],market_max_board=float(max(p['boards'] for p in cur)),
      market_max_board_count=float(sum(p['boards']==max(x['boards'] for x in cur) for p in cur)),
      consecutive_count=float(len(relay)),total_limit_up=float(len(relay)))
    # R7 total_limit_up is a legacy name for the >=2 board candidate count, not all 59.
    amounts=pd.Series({r['code']:r['amount_cny'] for r in three});amps=pd.Series({r['code']:r['amplitude_pct'] for r in three})
    prices=pd.Series({r['code']:r['close'] for r in three});turn=pd.Series({r['code']:r['turnover_pct'] for r in three})
    ranks=dict(amount=amounts.rank(ascending=False,method='min'),amplitude=amps.rank(method='min'),price=prices.rank(method='min'),turnover=turn.rank(method='min'))
    rows=[]
    for r in three:
        code=r['code']; priorrow=byprev.get(code,{}); cap=r['reported_float_cap_cny']
        h={k:m.get(k,np.nan) for k in raw}
        h.update(date=DAY,code=code,name=r['name'],boards=3,industry=industry(code),height_gap=m['market_max_board']-3,
          is_space_leader=float(m['market_max_board']==3),unique_space=float(m['market_max_board']==3 and len(three)==1),height_share=3/m['market_max_board'],
          same_height_rivals=float(len(three)-1),turnover=r['turnover_pct']/100,signal_amount=r['amount_cny'],
          turnover_vs_prev=r['turnover_pct']/priorrow['turnover_pct'] if priorrow.get('turnover_pct') else np.nan,
          amount_float_cap=r['amount_cny']/cap,one_word_proxy=locked(r),one_word_streak=1.0 if locked(r) and not locked(priorrow) else (np.nan if locked(r) else 0.0),
          amplitude=r['amplitude_pct'],relative_strength=r['change_pct']/10,price=r['close'],float_cap_log=math.log(cap),
          amount_cohort_share=r['amount_cny']/amounts.sum(),amount_cohort_rank=float(ranks['amount'][code]),
          amplitude_cohort_rank=float(ranks['amplitude'][code]),price_cohort_rank=float(ranks['price'][code]),turnover_cohort_rank=float(ranks['turnover'][code]),
          is_cohort_volume_leader=float(ranks['amount'][code]==1),is_cohort_tightest=float(ranks['amplitude'][code]==1),
          is_cohort_lowest_price=float(ranks['price'][code]==1),is_already_space_leader=float(m['market_max_board']==3),bomb_count=r.get('reported_open_count',np.nan))
        h.update(theme_features(r)); rows.append(h)
    frame=pd.DataFrame(rows)
    bindings=[R7/'round7_all_fold_models.joblib',R7/'feature_medians.json',R7/'feature_order.json',R7/'round7_manifest.json',R7/'verify_round7_model_load_and_predict.py',
       R7/'train_and_evaluate_round7_capacity.py',R7/'oof_round7_predictions.csv',R7/'round7_dataset.csv',basic,
       REVIEW/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',REVIEW/'features/current29_BEFORE_MODEL.json',
       ROOT/'results/daily_review_20261027T0928Z/facts/RESOLVED_FACTS_v2_SCORES_MASKED.json',REVIEW/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json',
       REVIEW/'BASE_scores_AFTER_INDEPENDENT.json',REVIEW/'20261028复盘与1029预期预测_完整双榜.md',
       ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/model.joblib',
       ROOT/'data/independent_stock_judgement/registry.jsonl',ROOT/'data/independent_stock_judgement/settlement_ledger.jsonl']
    before={str(p):sha(p) for p in bindings}
    write('PROTOCOL_BEFORE_SCORES.json',dict(formed_at_utc=datetime.now(timezone.utc).isoformat(),signal_date=DAY,target_date=TARGET,
       model='Gemini R7 capacity dragon, current repaired files',fold=4,training_cutoff='2025-11-30',
       native_scope='Exactly three boards at signal close',main_arm='arm3_capacity',other_arms='display only, no outcome-based selection',
       first_board_inference=False,auction_model_inference=False,reason='2026-10-29 auction has not occurred',
       source_scope='Archived full 112-row review; 59 closing-price limit events; 8 relay; 2 native three-board',
       old_scores_already_seen=True,blind_review=False,protected_and_source_hashes=before,
       limitations=['Alternative pool not exchange-wide certified.','Old market context quality/dimensions retain the 37-row prior archive limitation; complete prior rates reconstructed from 38 rows separately.',
       'Vintage industry mapping is used exactly as training; real robot/topic links are not injected into frozen features.',
       'Unknown raw fields are imputed only with saved per-head training medians; no new medians fitted.',
       'No fitted probability calibration; scores are not event probabilities.'],new_fits=0,orders=0))
    write('INPUTS_BEFORE_SCORES.json',clean(dict(rows=rows,raw_feature_order=raw,
       missing_raw_by_code={r['code']:[c for c in raw if pd.isna(r[c])] for r in rows},
       industry_mapping_unknown_codes=[r['code'] for r in cur if industry(r['code']).startswith('UNKNOWN_')],
       notes={'total_limit_up':'Legacy R7 field equals relay count 8; limit_count_scaled uses 59.',
       'theme_amount_share':'Original R7 builder sums amounts only for >=2-board candidates.',
       'crowding_risk':'Not exposed by market pack; use saved per-head median, not per-stock zero.',
       'theme_industry':'Original vintage mapping only; unknowns remain individually isolated.'})))
    models=joblib.load(R7/'round7_all_fold_models.joblib');meds=read(R7/'feature_medians.json')
    spec=importlib.util.spec_from_file_location('gemini_r7_saved_transform',R7/'verify_round7_model_load_and_predict.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=frame[['code','name','boards','industry']].copy(); imputed={};head_models={}
    heads=[('3to4','score_3to4'),('space','score_space'),('5b','score_5b'),('gain46','score_gain46'),('reb18','score_reb18')]
    training=pd.read_csv(R7/'round7_dataset.csv',dtype={'code':str})
    oof=pd.read_csv(R7/'oof_round7_predictions.csv',dtype={'code':str})
    test=training[training.date.between('2025-12-01','2026-08-31')].sort_values(['date','code']).reset_index(drop=True)
    saved=oof[oof['fold']==4].sort_values(['date','code']).reset_index(drop=True)
    replay={}
    origcols={'3to4':'score_3to4','space':'score_space_leader','5b':'score_height_5','gain46':'score_path_gain46','reb18':'score_rebound18'}
    for head,col in heads:
        key='fold_4_'+head;model=models[key]; trans,order=module.transform_features_with_medians(frame,meds[key],raw)
        assert order==cols and list(model.feature_names_in_)==cols
        assert np.isfinite(trans[cols].to_numpy()).all()
        output[col]=model.predict_proba(trans[cols])[:,1]
        imputed[key]={r['code']:{c:meds[key][c] for c in raw if pd.isna(r[c])} for r in rows}
        t,order=module.transform_features_with_medians(test,meds[key],raw)
        diff=float(np.max(np.abs(model.predict_proba(t[cols])[:,1]-saved[origcols[head]].to_numpy())))
        assert diff<1e-12;replay[key]=dict(rows=len(test),max_abs_difference=diff)
        head_models[key]=dict(features=len(cols),train_cutoff='2025-11-30')
    output['score_arm0_3to4']=output.score_3to4
    output['score_arm1_full']=.45*output.score_3to4+.25*output.score_5b+.15*output.score_gain46+.15*output.score_reb18
    output['score_arm2_space']=.60*output.score_space+.40*output.score_5b
    output['score_arm3_capacity']=.40*output.score_space+.35*output.score_5b+.25*output.score_3to4
    output['rank_arm3']=output.score_arm3_capacity.rank(ascending=False,method='min').astype(int)
    output=output.sort_values(['rank_arm3','code'])
    # Load the repaired regime gate that R7 actually uses; native NaNs supported, no invented gate pass.
    gate=joblib.load(R6/'repaired_regime_gate_models.joblib')['regime_gate_fold_4']
    gatecols=list(gate.feature_names_in_);gateinput={c:m.get(c,np.nan) for c in gatecols};gateinput['height_gap']=m['market_max_board']-3;gateinput['cohort_size']=len(three)
    gate_score=float(gate.predict_proba(pd.DataFrame([gateinput])[gatecols])[0,1])
    write('REGIME_GATE.json',clean(dict(score=gate_score,threshold=.55,pass_gate=gate_score>=.55,
      model_path=str(R6/'repaired_regime_gate_models.joblib'),sha256=sha(R6/'repaired_regime_gate_models.joblib'),fold=4,
      input=gateinput,missing_fields=[c for c in gatecols if pd.isna(gateinput[c])],calibrated_probability=False)))
    write('SAVED_HEAD_MEDIAN_IMPUTATIONS.json',imputed)
    write('GEMINI_R7_SCORES.json',clean(dict(signal_date=DAY,target_date=TARGET,scope='three_board_only',rows=output.to_dict('records'),
      calibrated_probability=False,source_qualified_for_production=False,old_full_pool_frozen_before_model=True,new_fits=0,orders=0)))
    output.to_csv(OUT/'GEMINI_R7_SCORES.csv',index=False,encoding='utf-8-sig')
    write('ALL_RELAY_MODEL_APPLICABILITY.json',dict(rows=[dict(code=r['code'],name=r['name'],boards=r['boards'],
       status='SCORED_NATIVE_THREE_BOARD' if r['boards']==3 else 'NOT_APPLICABLE_TO_R7_THREE_BOARD_MODEL',
       score=None if r['boards']!=3 else float(output.loc[output.code==r['code'],'score_arm3_capacity'].iloc[0])) for r in relay],
       first_board_count=51,first_board_R7_scores=None,auction_20261029_data=None))
    after={str(p):sha(p) for p in bindings};assert before==after
    manifest=read(R7/'round7_manifest.json')
    write('VERIFICATION.json',dict(formed_at_utc=datetime.now(timezone.utc).isoformat(),model_manifest_hash_matches=manifest['artifacts_sha256']['round7_all_fold_models.joblib']==sha(R7/'round7_all_fold_models.joblib'),
       actual_oof_sha=sha(R7/'oof_round7_predictions.csv'),declared_oof_sha=manifest['artifacts_sha256']['oof_round7_predictions.csv'],
       loaded_last_fold_replay=replay,protected_and_source_files_unchanged=before==after,raw_features=70,engineered_features=10,
       full_pool=112,relay=8,native_three_board=2,first_board=51,new_fits=0,orders=0))
    print(output.to_json(orient='records',force_ascii=False,indent=2))
    print(json.dumps({'regime_score':gate_score,'gate_pass':gate_score>=.55,'imputed':{r['code']:[c for c in raw if pd.isna(r[c])] for r in rows},'replay':replay},ensure_ascii=False))

if __name__=='__main__':main()
