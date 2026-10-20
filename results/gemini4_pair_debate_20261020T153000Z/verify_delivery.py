"""Read-only source/receipt/calendar/preservation checks; no inference or fit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRE=ROOT/'results/daily_review_20261020T101630Z'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 original=read(PRE/'BASE_inference_BEFORE_LOAD.json')
 checks={p:sha(p)==h for p,h in original['protected_hashes'].items()}
 binding=original['binding'];checks[binding['path']]=sha(binding['path'])==binding['sha256'];checks[binding['model_path']]=sha(binding['model_path'])==binding['model_sha256']
 assert all(checks.values()),checks
 browser=read(OUT/'browser_rounds_receipt.json')
 assert browser['real_sent_rounds']==browser['real_completed_replies']==2 and browser['usage_limit_visible_after_round2'] and not browser['switched_model']
 response=(OUT/'GEMINI_round2_FINAL_received_DOM.txt').read_text(encoding='utf-8')
 assert all(w in response for w in ['弱偏大业股份','弱偏东安动力','9.93%','无人盯盘','换手>6%'])
 source_checks={}
 for r in read(OUT/'source_checks/manifest.json'):
  if r.get('sha256'):source_checks[r['key']]=sha(OUT/'source_checks'/(r['key']+'.raw.html'))==r['sha256']
 assert len(source_checks)==6 and all(source_checks.values())
 calendar=read(ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json')['sessions'];i=calendar.index('2026-10-20')
 dates={h:calendar[i+h] for h in [1,2,3,5]}
 assert dates=={1:'2026-10-21',2:'2026-10-22',3:'2026-10-23',5:'2026-10-27'}
 values=[230929,156542,75771];denom=466476161;turnover=[v*100/denom*100 for v in values]
 assert round(sum(turnover),2)==9.93
 files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
 result=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),real_rounds=2,real_gemini_mode_observed='4 Argon',quota_observed_after_two_completed_replies=True,new_model_calls_after_quota=0,protected_unchanged=checks,source_byte_hashes=source_checks,official_targets=dates,turnover_unified_denominator_pct=turnover,files=files,new_fits=0,production_switches=0,orders=0,new_matured_results=0,no_joint_winner=True)
 with (OUT/'completion_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:result[k] for k in ['real_rounds','quota_observed_after_two_completed_replies','new_model_calls_after_quota','official_targets','new_fits','orders','no_joint_winner']},ensure_ascii=False))
if __name__=='__main__':main()
