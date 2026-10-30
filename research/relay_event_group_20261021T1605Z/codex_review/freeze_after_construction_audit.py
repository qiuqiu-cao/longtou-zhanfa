from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'validation_final'
QUALIFIED = HERE.parent / 'qualified_final'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
p = read(OUT/'PRECHECK.json')
audit = read(QUALIFIED/'CODEX_CONSTRUCTION_AUDIT.json')
assert audit['rows'] == 9245 and audit['fits'] == 0
assert audit['checks']['event_rosters_recomputed'] == 1979
assert all(x['mismatches'] == 0 for x in read(QUALIFIED/'PREFIX_INVARIANCE_CHECK.json').values())
for path,digest in p['sources'].items(): assert sha(Path(path)) == digest,path
for path,digest in audit['sources'].items(): assert sha(Path(path)) == digest,path
paths=[OUT/'PRECHECK.json', OUT/'CODEX_FINAL_EXECUTION_REVIEW.md',
       QUALIFIED/'CODEX_CONSTRUCTION_AUDIT.json',QUALIFIED/'PREFIX_INVARIANCE_CHECK.json',
       QUALIFIED/'EVENT_GROUP_LEDGER_DETAILS.jsonl',QUALIFIED/'MEMBER_DAILY_OBSERVATIONS.csv',
       HERE/'audit_qualified_features.py',HERE/'audit_result_independently.py',
       HERE.parent/'gemini/RESPONSE_TO_FINITE_REVIEW.md',HERE.parent/'gemini/RESPONSE_TO_FINAL_SOURCE_FIX.md']
p['sources'].update({str(path.resolve()):sha(path) for path in paths})
p.update(status='CODEX_SIGNED_AWAITING_REAL_GEMINI_ACK_BEFORE_FIT', at_utc=datetime.now(timezone.utc).isoformat(),
         actual_executor='Antigravity Gemini',director='Codex',no_new_monthly_training_authorized=True,
         production_deployment_allowed=False, new_blind_test=False, history_seen=True,
         source_qualification='Known historical date intervals and archived event-name proxy; unknown external anchors/membership retained.',
         fitting_conditions='Actual peer reads this exact SHA and review, writes authentic accepted ACK, then designated runner only; never execute corrected draft runner.')
with (OUT/'FROZEN_PROTOCOL.json').open('x',encoding='utf-8') as f:
    json.dump(p,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'protocol_sha256':sha(OUT/'FROZEN_PROTOCOL.json'),'status':p['status'],'new_fits':0}))
