"""Check signal-day features do not change when later daily bars are hidden."""
from pathlib import Path
import json,sys
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'v3';sys.path.insert(0,str(ROOT))
import first_second_stage_model as stage
inputs={'bars':OUT/'replay_bars_PROXY.csv','recovered':OUT/'recovered_history.csv','calendar':ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json','security_intervals':OUT/'security_master_with_evidenced_additions.csv'}
date='2026-10-15';prefix,_,_,_=stage.build_rows(inputs,feature_cutoff=date)
prefix=prefix[prefix.signal_date.eq(date)&prefix.stage.eq(1)].sort_values('code')
full=pd.read_csv(OUT/'MODEL_INPUTS_BEFORE_SCORE_DIAGNOSTIC.csv',dtype={'code':str},low_memory=False);full=full[full.signal_date.eq(date)].sort_values('code')
assert prefix.code.tolist()==full.code.tolist()
features=stage.VARIANTS['STAGE_CYCLE_GENE'];details={}
for col in features:
    a=pd.to_numeric(prefix[col],errors='coerce').to_numpy();b=pd.to_numeric(full[col],errors='coerce').to_numpy()
    close=np.isclose(a,b,equal_nan=True,rtol=1e-8,atol=1e-8);details[col]=int((~close).sum())
result={'date':date,'rows':len(prefix),'feature_count':len(features),'mismatches':{k:v for k,v in details.items() if v},'all_equal':all(v==0 for v in details.values()),'meaning':'Feature computation prefix check only; does not certify historical inputs, universe, corporate actions or untouched blindness.'}
(OUT/'PREFIX_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));assert result['all_equal']
