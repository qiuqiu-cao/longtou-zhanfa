"""Post-hoc input sensitivity of a fixed head. Not feature causality or a new rule."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent;OUT=BASE/'v3';sys.path.insert(0,str(ROOT))
import first_second_stage_model as stage
rows=pd.read_csv(OUT/'FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv',dtype={'code':str},low_memory=False)
rows=rows[(rows.signal_date.eq('2026-10-15')) & rows.research_selection_allowed].copy().sort_values(['s1_score','code'],ascending=[False,True]);rows['rank']=range(1,len(rows)+1)
f=stage.VARIANTS['STAGE_CYCLE_GENE'];model=joblib.load(ROOT/'research/first_second_stage_20261020_run1/models/latest_S1_full.joblib'); j=rows[rows.code.eq('003029')].iloc[0];orig=pd.DataFrame([j[f].to_dict()]);score=float(model.predict_proba(orig)[0,1]);assert abs(score-j.s1_score)<1e-10
records=[]
for code in ['605003','603999','603172','002889']:
    peer=rows[rows.code.eq(code)].iloc[0]; batch=pd.concat([orig.copy() for _ in f],ignore_index=True)
    for i,name in enumerate(f):batch.loc[i,name]=peer[name]
    scores=model.predict_proba(batch)[:,1]
    for i,name in enumerate(f):records.append({'peer_code':code,'peer_name':peer['name'],'feature':name,'jida_value':j[name],'peer_value':peer[name],'score_if_replaced':scores[i],'score_delta':scores[i]-score})
result=pd.DataFrame(records);result.to_csv(OUT/'JIDA_SINGLE_FEATURE_REPLACEMENTS_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'jida_rank':int(j['rank']),'jida_score':score,'limitation':'Single-feature replacements can generate unrealistic combinations; changes are not causal or proved stable. No feature weights or production rules modified.','largest_changes':result.assign(absdelta=result.score_delta.abs()).sort_values('absdelta',ascending=False).head(16).replace({np.nan:None}).to_dict('records')},ensure_ascii=False,indent=2,default=str))
