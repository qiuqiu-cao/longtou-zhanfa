from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
paths=['research/first_second_stage_20261020_run1/models/latest_S1_full.joblib','research/first_second_stage_20261020_run1/models/latest_S1_full_COMPLETED.json','first_second_stage_model.py','first_board_leader_model.py','first_board_capital_risk_model.py','data/v16/point_in_time_rebuild_cutoff_20260831_v2/point_in_time_current_daily_bars.csv','data/v16/point_in_time_rebuild_cutoff_20260831_v2/verified_recovered_daily_bars.csv','data/v11/security_master_intervals.csv','data/v298/exchange_calendar_extension_20261003_v2/calendar.json']
d={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'model':'saved latest_S1_full.joblib, label cutoff 2026-08-28, constructed 2026-10-20','scope':'2026-10-08 through 2026-10-29 inclusive; 2026-10-30 target not matured','target':'next official session close at exact limit','selection':'full qualified original first-board pool; S1 descending, code ascending tie; cap <= 9326859896.52 yuan; existing thin-risk rule','no_signal_later_than_close':True,'outputs':'all dates, all slots, unknowns and non-one-price; inspect outcomes only after rankings frozen','no_retraining':True,'no_tuning':True,'retrospective_limit':'October has been seen and model was formed Oct 20; temporal out-of-sample reconstruction, not untouched blinded prospective test','source_hashes':{p:sha(ROOT/p) for p in paths}}
with (OUT/'PROTOCOL_BEFORE_DATA_RECONSTRUCTION.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
print(json.dumps(d,ensure_ascii=False,indent=2))
