"""Fixed S1 scoring only; input file contains no future outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,hashlib
import pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'v3';sys.path.insert(0,str(ROOT))
import first_second_stage_model as stage
import first_board_capital_risk_model as risk
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inp=OUT/'MODEL_INPUTS_BEFORE_SCORE_DIAGNOSTIC.csv'; rows=pd.read_csv(inp,dtype={'code':str},low_memory=False)
assert not any('LABEL' in c or c in ['label_end','label_observed'] for c in rows.columns)
assert rows.signal_date.nunique()==16 and not rows.duplicated(['signal_date','code']).any()
model=ROOT/'research/first_second_stage_20261020_run1/models/latest_S1_full.joblib'; m=joblib.load(model)
assert sha(model)=='9a3c47da25e9ed9eaf5a312cbee28cd67f9baa080bfd97fb27731c38ba9a35da'
rows=risk.capital_context(rows,9326859896.520002)
rows['s1_score']=m.predict_proba(rows[stage.VARIANTS['STAGE_CYCLE_GENE']])[:,1]
rows['s1_model_status']='UNQUALIFIED_INPUT_RECONSTRUCTION_DIAGNOSTIC_NOT_CALIBRATED'
rows['screen_missing_volume_history']=(rows.float_cap.le(3e9)&rows.turnover.lt(5)&rows.one_price.eq(0)&rows.volume_ratio_5.isna())
eligible=rows[rows.research_selection_allowed & ~rows.screen_missing_volume_history].copy()
eligible=eligible.sort_values(['signal_date','s1_score','code'],ascending=[True,False,True]);eligible['rank']=eligible.groupby('signal_date').cumcount()+1
selected=eligible[eligible['rank'].le(3)].copy()
rows.to_csv(OUT/'FULL_POOL_SCORES_UNQUALIFIED_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
selected.to_csv(OUT/'TOP3_FROZEN_BEFORE_OUTCOME_REPLAY.csv',index=False,encoding='utf-8-sig')
receipt={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':sha(inp),'model_sha256':sha(model),'selected_sha256':sha(OUT/'TOP3_FROZEN_BEFORE_OUTCOME_REPLAY.csv'),'pool_rows':len(rows),'eligible_rows':len(eligible),'selected_rows':len(selected),'selected_dates':int(selected.signal_date.nunique()),'unknown_risk_history':int(rows.screen_missing_volume_history.sum()),'target_fields_used':False,'status':'AFTER_OUTCOMES_FACTUAL_CORRECTION_NOT_BLIND','note':'V3 also repairs estimated market capitalization for rows outside page pool, preserving v1 and v2 results.'}
(OUT/'RANKING_FREEZE_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
