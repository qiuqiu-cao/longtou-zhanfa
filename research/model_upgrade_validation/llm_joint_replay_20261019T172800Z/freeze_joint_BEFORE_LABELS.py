from run_joint_replay import *
from evaluate_AFTER_JOINT_FREEZE import validate, FILES
import copy
assert not (HERE/'label_reveal_started_AFTER_ALL_FREEZES.json').exists(), 'No post-reveal decisions permitted'
response_path=HERE/'gemini_consultation_response_BEFORE_LABELS.json'
response=read(response_path)
assert response['outcomes_seen'] is False
assert response['proposal_sha256']==sha(HERE/'CODEX_joint_proposal_BEFORE_LABELS.json')
accept=read(HERE/'codex_acceptance_of_gemini_response_BEFORE_LABELS.json')
assert accept['outcomes_seen'] is False and accept['accept_all_declared_amendments'] is True
assert accept['response_sha256']==sha(response_path)
out=read(HERE/'CODEX_joint_proposal_BEFORE_LABELS.json')
out['reviewer']='Codex + Antigravity Gemini 3.8 Flash High, actually discussed joint decision'
out['created_at_utc']=now();out['outcomes_seen']=False
out['method']='Both independent 122-row reviews frozen first; Codex reasoned 20-row consultation proposal; Gemini actual response and counterproposals; Codex accepts declared amendments before label reveal. No score averaging or fitted ensemble.'
out['consultation_response_sha256']=sha(response_path)
out['codex_acceptance_sha256']=sha(HERE/'codex_acceptance_of_gemini_response_BEFORE_LABELS.json')
cl={(d['case'],r['id']):r for d in read(HERE/FILES['codex_independent'])['days'] for r in d['stocks']}
gl={(d['case'],r['id']):r for d in read(HERE/FILES['gemini_independent'])['days'] for r in d['stocks']}
dl={d['case']:d for d in out['days']}
lookup={(d['case'],r['id']):r for d in out['days'] for r in d['stocks']}
for key,r in lookup.items():
    a,b=cl[key],gl[key]
    r['codex_original_reason']=a['reason'];r['gemini_original_reason']=b['reason']
    if r['reason'].startswith('双方独立方向一致；'):
        r['reason']='双方独立方向一致并在讨论后保留；'+a['reason']+' 角色、主动资金及成交仍未认证。'
    elif r['reason'].startswith('暂保留Codex倾向，待Gemini逐项反驳。'):
        r['reason']='讨论后保留该倾向；'+a['reason']+' Gemini另一解释见独立原稿，身份/资金推断未作为事实。'
seen=set()
for amendment in response['row_amendments']:
    key=(amendment['case'],amendment['id'])
    assert key in lookup and key not in seen
    assert amendment['decision'] in ('YES','NO','UNKNOWN') and len(amendment['reason'])>=5
    seen.add(key);lookup[key]['decision']=amendment['decision']
    lookup[key]['reason']='采纳Gemini在讨论阶段的具体修正：'+amendment['reason']+' 该叙事为情景判断，角色与资金因果仍未认证。'
seen_cases=set()
for amendment in response['top3_amendments']:
    case=amendment['case'];assert case in dl and case not in seen_cases
    seen_cases.add(case);dl[case]['top3']=amendment['top3'];dl[case]['joint_top3_reason']=amendment['reason']
for amendment in accept.get('reason_clarifications',[]):
    key=(amendment['case'],amendment['id']);assert key in lookup
    lookup[key]['reason']=amendment['reason']
out['factual_language_clarifications']=accept.get('factual_language_clarifications',[])
out['gemini_response_accept_proposal']=response['accept_proposal']
out['gemini_declared_row_amendments']=len(seen)
out['gemini_declared_top3_amendments']=len(seen_cases)
write(HERE/'joint_decisions_BEFORE_LABELS.json',out)
packet=read(HERE/'review_packet_SIGNAL_ONLY_v2.json')
for arm,f in FILES.items():validate(HERE/f,packet)
prep=read(HERE/'preparation_receipt.json')
assert sha(HERE/'protocol_BEFORE_DATA_AND_DECISIONS.json')==prep['protocol_sha256']
assert sha(HERE/'codex_independent_BEFORE_GEMINI_AND_LABELS.json')==read(HERE/'codex_independent_freeze_receipt.json')['sha256']
write(HERE/'joint_freeze_receipt_BEFORE_LABELS.json',{'created_at_utc':now(),'discussion_completed':True,'outcome_reads':0,
    'decision_hashes':{f:sha(HERE/f) for f in FILES.values()},'packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY_v2.json'),
    'proposal_sha256':sha(HERE/'CODEX_joint_proposal_BEFORE_LABELS.json'),'response_sha256':sha(response_path),
    'gemini_independent_first_payload_unchanged':read(HERE/'gemini_independent_BEFORE_CODEX_AND_LABELS.json')==read(HERE/'gemini_first_received_snapshot_BEFORE_DISCUSSION.json'),
    'outcome_reveal_file_absent':not (HERE/'label_reveal_started_AFTER_ALL_FREEZES.json').exists(),
    'rows':sum(len(d['stocks']) for d in out['days']),'days':len(out['days'])})
print(json.dumps({'joint_frozen':True,'rows':122,'counts':pd.Series([r['decision'] for d in out['days'] for r in d['stocks']]).value_counts().to_dict(),'outcome_reads':0},ensure_ascii=False))
