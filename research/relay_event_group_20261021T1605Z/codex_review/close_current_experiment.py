from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent
OUT=RESEARCH/'validation_final'
ROOT=HERE.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
peer=read(OUT/'PEER_RESULT_ACK.json')
result=read(OUT/'RESULT.json')
assert peer['protocol_sha256']==sha(OUT/'FROZEN_PROTOCOL.json')
assert peer['verification_acknowledgment']['nonword_top1_new']==result['new_metrics']['nonword_top1']['hits']==137
assert peer['verification_acknowledgment']['nonword_top1_control']==134
assert peer['closeout_determination']['experiment_status']=='SEALED_REJECTED_NO_RETRY'
receipt=dict(at_utc=datetime.now(timezone.utc).isoformat(),status='BATCH_COMPLETED_SEALED_NOT_ACCEPTED_FOR_UPGRADE',
    arm='ST_A_EVENT_COHORT_FIXED3',actual_executor='Antigravity Gemini',director='Codex',
    actual_fits=3,monthly_fits=0,model_reload_and_prediction_audits_passed=True,
    actual_peer_result_agreement=True,new_nonword_top1=137,control_nonword_top1=134,dates=302,
    unknown_candidate_outcomes=67,all_gates_passed=False,production_deployed=False,
    no_retry_or_tuning_this_arm=True,first_board='PAUSED_AFTER_CORRECTION',
    no_tasks_or_goal_changes=True,active_exec_sessions=[],
    artifacts={str(p.resolve()):sha(p) for p in [OUT/'FROZEN_PROTOCOL.json',OUT/'RESULT.json',OUT/'PEER_RESULT_ACK.json',
        OUT/'CODEX_INDEPENDENT_PREDICTION_AUDIT.json',OUT/'CODEX_SAVED_MODEL_AUDIT.json',OUT/'CODEX_VERIFIED_RESULT.md',
        OUT/'EVENT_COHORT_VALIDATION_REPORT.md',OUT/'REPORT_CORRECTIONS.json']})
with (OUT/'COMPLETION_FINAL.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
pointer=ROOT/'research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json'
d=read(pointer)
d.update(updated_at=receipt['at_utc'],status=receipt['status'],active_process=None,new_model_fits=3,
    completion_receipt=str((OUT/'COMPLETION_FINAL.json').resolve()),
    active_arms=[dict(name=receipt['arm'],actual_executor='Antigravity Gemini',fits=3,status='SEALED_REJECTED_NO_RETRY')],
    actual_peer_ui_state='Result ACK received; no further training authorized',
    actual_peer_result_agreement=True,result_notified_in_current_chat=True,
    peer_response_expected=None,active_exec_sessions=[])
pointer.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
with (RESEARCH/'MONITOR_LAST_NOTIFIED_RESULT.json').open('x',encoding='utf-8') as f:
    json.dump(dict(at_utc=receipt['at_utc'],result_sha256=sha(OUT/'RESULT.json'),
        notified_channel='Current chat commentary; verified137/302 versus134/302 and no upgrade already delivered',
        completion_receipt=str((OUT/'COMPLETION_FINAL.json').resolve()),
        same_result_requires_no_repeat_notification=True),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=receipt['status'],fits=3,peer_agreement=True,new_training_authorized=False)))
