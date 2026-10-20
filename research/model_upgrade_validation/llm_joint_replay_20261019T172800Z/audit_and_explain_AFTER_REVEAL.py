"""Independent bookkeeping audit and descriptive discussion effects, no predictions."""
from run_joint_replay import *
import math
rows=pd.read_csv(HERE/'complete_122_revealed_all_arms.csv',dtype={'code':str,'date':str,'target_date':str})
res=read(HERE/'results_AFTER_JOINT_FREEZE.json')
freeze=read(HERE/'joint_freeze_receipt_BEFORE_LABELS.json')
assert len(rows)==122 and not rows.duplicated(['case','id']).any()
assert len(rows['case'].unique())==12
for f,h in freeze['decision_hashes'].items():assert sha(HERE/f)==h
decisions={a:read(HERE/f) for a,f in __import__('evaluate_AFTER_JOINT_FREEZE').FILES.items()}
for a,p in decisions.items():
    lookup={(d['case'],s['id']):s['decision'] for d in p['days'] for s in d['stocks']}
    assert rows.apply(lambda r:lookup[(r['case'],r['id'])]==r[a],axis=1).all()
    for event in ['strict_close','archived_promotion','touch']:
        k=rows.loc[rows[event].notna()].copy()
        called=k.loc[k[a].isin(['YES','NO'])]
        y=called[event].astype(int).map({1:'YES',0:'NO'})
        correct=int(called[a].eq(y).sum())
        v=res['events'][event]['arms'][a]
        assert len(k)==v['labels_known'] and len(called)==v['judged_known']
        assert correct==v['TP']+v['TN']
        assert abs(correct/len(called)-v['accuracy_judged'])<.000051
        assert int(((k[a]=='YES')&(k[event]==1)).sum())==v['TP']
        assert int(((k[a]=='YES')&(k[event]==0)).sum())==v['FP']
        assert int(((k[a]=='NO')&(k[event]==1)).sum())==v['FN_judged']
        assert int(((k[a]=='UNKNOWN')&(k[event]==1)).sum())==v['abstain_positive']
effects={}
for other in ['codex_independent','gemini_independent']:
    k=rows.loc[rows.strict_close.notna()].copy()
    truth=k.strict_close.astype(int).map({1:'YES',0:'NO'})
    k['joint_correct']=k.discussed_joint.eq(truth);k['other_correct']=k[other].eq(truth)
    common=k[other].ne('UNKNOWN')&k.discussed_joint.ne('UNKNOWN')
    resolve=k[other].eq('UNKNOWN')&k.discussed_joint.ne('UNKNOWN')
    withdraw=k[other].ne('UNKNOWN')&k.discussed_joint.eq('UNKNOWN')
    effects[other]={'common_judged':int(common.sum()),'corrections':int((common&k.joint_correct&~k.other_correct).sum()),
        'induced_errors':int((common&~k.joint_correct&k.other_correct).sum()),
        'resolved_abstentions':int(resolve.sum()),'resolved_correct':int((resolve&k.joint_correct).sum()),
        'withdrawn_calls':int(withdraw.sum()),'withdrawn_previously_correct':int((withdraw&k.other_correct).sum()),
        'withdrawn_previously_wrong':int((withdraw&~k.other_correct).sum()),
        'full_known_correct_change':int(k.joint_correct.sum()-k.other_correct.sum())}
    v=effects[other]
    assert v['full_known_correct_change']==v['corrections']-v['induced_errors']+v['resolved_correct']-v['withdrawn_previously_correct']
all_known_days=[case for case,g in rows.groupby('case') if g.strict_close.notna().all()]
top_complete={}
for a in ['codex_independent','gemini_independent','discussed_joint']:
    top=rows.loc[rows.case.isin(all_known_days)&rows[a+'_rank'].gt(0)]
    top_complete[a]={'days':len(all_known_days),'slots':len(top),'hits':int(top.strict_close.eq(1).sum())}
word=rows.loc[rows.one_price.eq(1)&rows.strict_close.notna()]
word_effects={a:{'rows':len(word),'actual_positive':int(word.strict_close.eq(1).sum()),
    'TP':int((word[a].eq('YES')&word.strict_close.eq(1)).sum()),'FP':int((word[a].eq('YES')&word.strict_close.eq(0)).sum()),
    'FN_judged':int((word[a].eq('NO')&word.strict_close.eq(1)).sum()),'abstain_positive':int((word[a].eq('UNKNOWN')&word.strict_close.eq(1)).sum())}
    for a in ['codex_independent','gemini_independent','discussed_joint']}
write(HERE/'discussion_effects_AFTER_REVEAL.json',{'created_at_utc':now(),'purpose':'Post-reveal descriptive explanation; no new trial, ranking, fitting or changes to frozen forecasts.',
    'effects_vs_individuals':effects,'complete_price_days':all_known_days,'Top3_same_complete_days':top_complete,
    'daily_one_price_group':word_effects,'all_unknown_rows_retained':True})
errors=rows.copy()
for a in ['codex_independent','gemini_independent','discussed_joint']:
    lookup={(d['case'],s['id']):s['reason'] for d in decisions[a]['days'] for s in d['stocks']}
    errors[a+'_reason']=[lookup[(r['case'],r['id'])] for _,r in errors.iterrows()]
    errors[a+'_result']=['UNLABELLED' if pd.isna(r.strict_close) else 'ABSTAIN' if r[a]=='UNKNOWN' else 'CORRECT' if r[a]==('YES' if r.strict_close==1 else 'NO') else 'FP' if r[a]=='YES' else 'FN' for _,r in errors.iterrows()]
errors.to_csv(HERE/'complete_122_errors_and_original_reasons_APPEND_ONLY.csv',index=False,encoding='utf-8-sig',mode='x')
prep=read(HERE/'preparation_receipt.json')
assert all(sha(path)==h for path,h in prep['protected_files'].items())
write(HERE/'independent_bookkeeping_audit_AFTER_REVEAL.json',{'created_at_utc':now(),'rows':len(rows),'days':12,'decisions_match_original_freezes':True,
    'all_three_events_confusions_independently_recounted':True,'full_known_coverage_metrics_independently_recounted':True,
    'abstention_transition_identity_passed':True,'protected_registry_and_ledger_unchanged':True,'new_fits':0})
print(json.dumps({'effects':effects,'complete_Top3':top_complete,'one_price':word_effects},ensure_ascii=False))
