from evaluate_AFTER_JOINT_FREEZE import *
p=read(HERE/'review_packet_SIGNAL_ONLY_v2.json')
c=validate(HERE/FILES['codex_independent'],p)
sample=[{'strict_close':y,'a':a} for y,a in [(1,'YES'),(0,'YES'),(0,'NO'),(1,'NO'),(1,'UNKNOWN'),(0,'UNKNOWN'),(None,'YES')]]
m=metrics(sample,'a')
assert (m['TP'],m['FP'],m['TN'],m['FN_judged'],m['abstain_positive'],m['abstain_negative'],m['labels_unknown'])==(1,1,1,1,1,1,1)
assert m['accuracy_judged']==.5 and m['correct_fraction_full_known']==1/3
assert m['precision_known']==.5 and m['recall_full_known']==1/3 and m['F1_full_known']==.4
assert m['predicted_yes_unlabelled']==1
for day in p['days']:
    for stock in day['stocks']:
        assert not {'target','target_date','promoted_next_day','base_score','raw_score','code','name','date'} & set(stock)
        assert all(h['lag_sessions']>=0 for h in stock['past5_ohlc_pct_vs_previous_close'])
assert sha(HERE/'codex_independent_BEFORE_GEMINI_AND_LABELS.json')==read(HERE/'codex_independent_freeze_receipt.json')['sha256']
write(HERE/'pre_reveal_checks.json',{'created_at_utc':now(),'metric_hand_example':m,'full_packet_rows':122,'codex_full_coverage_schema':True,'no_outcome_or_identifier_fields':True,'all_history_lags_nonnegative':True,'codex_hash_unchanged':True,'evaluation_script_sha256':sha(HERE/'evaluate_AFTER_JOINT_FREEZE.py'),'outcome_reads':0})
print('Hand-calculated abstention/unknown metrics, signal allowlist, 122-row coverage and freeze hash passed. Outcome reads: 0.')
