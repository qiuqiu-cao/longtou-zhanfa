"""Reveal only after validated decisions from both actual reviewers and consultation."""
from run_joint_replay import *
sys.path.insert(0,str(ROOT))
from short_term_thinking_dataset_builder import cents, cap_cents, price_state

FILES={'codex_independent':'codex_independent_BEFORE_GEMINI_AND_LABELS.json',
       'gemini_independent':'gemini_first_received_snapshot_BEFORE_DISCUSSION.json',
       'discussed_joint':'joint_decisions_BEFORE_LABELS.json'}

def validate(path,packet):
    p=read(path)
    assert p['packet_sha256']==sha(HERE/'review_packet_SIGNAL_ONLY_v2.json'), 'wrong signal packet'
    assert p['outcomes_seen'] is False
    assert len(p['days'])==len(packet['days'])
    assert {d['case'] for d in p['days']}=={d['case'] for d in packet['days']}
    lookup={d['case']:d for d in p['days']}
    for d in packet['days']:
        v=lookup[d['case']]; ids={r['id'] for r in d['stocks']}
        assert len(v['stocks'])==len(ids) and {r['id'] for r in v['stocks']}==ids
        assert len(v['top3'])==min(3,len(ids)) and len(set(v['top3']))==len(v['top3'])
        assert set(v['top3']) <= ids
        assert all(r['decision'] in ('YES','NO','UNKNOWN') and len(r['reason'])>=5 for r in v['stocks'])
    return p

def fraction(a,b): return a/b if b else None
def metrics(rows,arm,label='strict_close'):
    known=[r for r in rows if r[label] is not None]
    judged=[r for r in known if r[arm] in ('YES','NO')]
    tp=sum(r[arm]=='YES' and r[label]==1 for r in known)
    fp=sum(r[arm]=='YES' and r[label]==0 for r in known)
    tn=sum(r[arm]=='NO' and r[label]==0 for r in known)
    fn=sum(r[arm]=='NO' and r[label]==1 for r in known)
    up=sum(r[arm]=='UNKNOWN' and r[label]==1 for r in known)
    un=sum(r[arm]=='UNKNOWN' and r[label]==0 for r in known)
    precision=fraction(tp,tp+fp); recall=fraction(tp,tp+fn+up)
    return dict(rows=len(rows),labels_known=len(known),labels_unknown=len(rows)-len(known),positive_labels=tp+fn+up,
        judged_known=len(judged),abstain_positive=up,abstain_negative=un,TP=tp,FP=fp,TN=tn,FN_judged=fn,
        accuracy_judged=fraction(tp+tn,len(judged)),correct_fraction_full_known=fraction(tp+tn,len(known)),
        coverage_known=fraction(len(judged),len(known)),coverage_all=fraction(sum(r[arm]!='UNKNOWN' for r in rows),len(rows)),
        precision_known=precision,recall_full_known=recall,F1_full_known=fraction(2*tp,2*tp+fp+fn+up),
        balanced_full_known_correct=(fraction(tp,tp+fn+up)+fraction(tn,tn+fp+un))/2 if (tp+fn+up)*(tn+fp+un) else None,
        predicted_yes_all=sum(r[arm]=='YES' for r in rows),predicted_yes_unlabelled=sum(r[arm]=='YES' and r[label] is None for r in rows))

def tops(rows,arms,label):
    out={}
    for arm in arms:
        out[arm]={}
        for k in [1,3]:
            selected=[r for r in rows if 0<r[arm+'_rank']<=k]
            known=[r for r in selected if r[label] is not None]
            hits=sum(r[label]==1 for r in known)
            out[arm][f'Top{k}']={'slots':len(selected),'known_slots':len(known),'unknown_slots':len(selected)-len(known),
                'hits':hits,'hits_over_all_slots_lower_bound':fraction(hits,len(selected)),'hit_rate_known_slots':fraction(hits,len(known)),
                'days_with_any_hit':len({r['case'] for r in known if r[label]==1})}
    return out

def bootstrap(rows,left,right,label):
    cases=sorted({r['case'] for r in rows})
    counts=[]
    for case in cases:
        g=[r for r in rows if r['case']==case and r[label] is not None]
        counts.append([sum((r[left]=='YES' and r[label]==1) or (r[left]=='NO' and r[label]==0) for r in g),
                       sum((r[right]=='YES' and r[label]==1) or (r[right]=='NO' and r[label]==0) for r in g),len(g)])
    arr=np.array(counts,dtype=float)
    rng=np.random.default_rng(20261019)
    sampled=arr[rng.integers(0,len(cases),size=(10000,len(cases)))].sum(axis=1)
    valid=sampled[:,2]>0
    delta=(sampled[valid,0]-sampled[valid,1])/sampled[valid,2]
    return {'left':left,'right':right,'event':label,'unit':'signal day cluster, not independent stocks','resamples':10000,'seed':20261019,
        'delta_full_known_correct':float((arr[:,0].sum()-arr[:,1].sum())/arr[:,2].sum()) if arr[:,2].sum() else None,
        'percentile_95_interval':np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
        'limitation':'Conditional resampling uncertainty on this small retrospective sample; not independent blind-test certification or prospective significance.'}

def main():
    packet=read(HERE/'review_packet_SIGNAL_ONLY_v2.json')
    decisions={a:validate(HERE/f,packet) for a,f in FILES.items()}
    freeze=read(HERE/'joint_freeze_receipt_BEFORE_LABELS.json')
    assert freeze['outcome_reads']==0 and freeze['discussion_completed'] is True
    for arm,f in FILES.items():assert freeze['decision_hashes'][f]==sha(HERE/f)
    prep=read(HERE/'preparation_receipt.json')
    assert sha(BUNDLE)==prep['source_sha256']
    for p,h in prep['protected_files'].items():assert sha(p)==h
    # First possible target extraction starts after all preceding gates.
    write(HERE/'label_reveal_started_AFTER_ALL_FREEZES.json',{'created_at_utc':now(),'decision_hashes':freeze['decision_hashes'],'no_new_fits':True})
    data=joblib.load(BUNDLE)
    full=data['prepared']['relay'].copy()
    full['date']=full.date.astype(str).str[:10];full['code']=full.code.astype(str).str.zfill(6)
    full=full.set_index(['date','code']);assert full.index.is_unique
    mapping=read(HERE/'PRIVATE_mapping_DO_NOT_READ_BEFORE_JOINT_FREEZE.json')
    calendar_path=ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'
    calendar=read(calendar_path);assert calendar['status']=='verified'
    sessions=calendar['sessions']; nextday=dict(zip(sessions[:-1],sessions[1:]))
    assert set(r['date'] for r in mapping) <= set(sessions)
    codes={r['code'] for r in mapping}
    bars=data['bars'].copy();bars['date']=bars.date.astype(str).str[:10];bars['code']=bars.code.astype(str).str.zfill(6)
    bars=bars.loc[bars.code.isin(codes)].set_index(['date','code']);assert bars.index.is_unique
    refs=[];refbind={}
    for f in sorted((ROOT/'research/model_upgrade_validation/recency_walkforward_20261013T134351Z').glob('*_ROLL252_predictions_BEFORE_VALIDATION.csv')):
        v=pd.read_csv(f,dtype={'date':str,'code':str,'target_date':str});v['code']=v.code.str.zfill(6)
        refs.append(v);refbind[str(f)]=sha(f)
    ref=pd.concat(refs).set_index(['date','code']);assert ref.index.is_unique
    decisions_lookup={a:{(d['case'],r['id']):(r,d['top3'].index(r['id'])+1 if r['id'] in d['top3'] else 0) for d in p['days'] for r in d['stocks']} for a,p in decisions.items()}
    siglookup={(d['case'],r['id']):r for d in packet['days'] for r in d['stocks']}
    rows=[]
    for m in mapping:
        key=(m['date'],m['code']); signal=full.loc[key]; td=nextday[m['date']]
        archived=None;archive_status='MISSING_OR_INVALID_ARCHIVED_TARGET'
        target=num(signal.get('target'))
        if target in (0.,1.) and bool(signal.get('label_valid',False)) and str(signal.get('target_date'))[:10]==td:
            archived=int(target);archive_status='VALID_ARCHIVED_ADJACENT_TARGET'
        future=bars.loc[(td,m['code'])].to_dict() if (td,m['code']) in bars.index else None
        strict=None;touch=None;price_status='MISSING_OR_INVALID_NEXT_OHLC';price_close=None;price_high=None;cap=None
        if future and price_state(future)['one_word'] is not None:
            try:
                previous=cents(future['pre_close']);signal_price=cents(m['pool_price'])
                if previous != signal_price:price_status='NEXT_PRECLOSE_VS_SIGNAL_POOL_PRICE_CONFLICT'
                else:
                    cap=cap_cents(future['pre_close']);price_close=cents(future['close']);price_high=cents(future['high'])
                    if price_high>cap:price_status='EXCEEDS_10PCT_CAP_SPECIAL_STATUS_OR_PRICE_CONFLICT'
                    else:
                        raw=int(price_close==cap);touch=int(price_high==cap)
                        if archived is None:price_status='PRICE_OBSERVED_BUT_ARCHIVED_TARGET_UNQUALIFIED'
                        elif raw!=archived:price_status='PRICE_VS_ARCHIVED_PROMOTION_CONFLICT'
                        else:strict=raw;price_status='EXACT_CENT_10PCT_PRICE_AND_ARCHIVED_TARGET_AGREE'
            except (ValueError,KeyError):price_status='INVALID_EXACT_CENT_PRICE'
        r={**m,'target_date':td,'boards':int(signal['boards']),'one_price':siglookup[(m['case'],m['id'])]['one_price'],
           'archived_promotion':archived,'archive_status':archive_status,'strict_close':strict,'touch':touch,'price_status':price_status,
           'next_close_cents':price_close,'next_high_cents':price_high,'cap_cents':cap}
        for arm,lookup in decisions_lookup.items():r[arm]=lookup[(m['case'],m['id'])][0]['decision'];r[arm+'_rank']=lookup[(m['case'],m['id'])][1]
        r['always_no']='NO'
        if key in ref.index:
            rr=ref.loc[key]; raw_score=num(rr['raw_score'])
            r['archived_ROLL252']='YES' if raw_score is not None and raw_score>.5 else 'NO' if raw_score is not None else 'UNKNOWN'
            r['ROLL252_score']=raw_score
        else:r['archived_ROLL252']='UNKNOWN';r['ROLL252_score']=None
        rows.append(r)
    for case in sorted({r['case'] for r in rows}):
        g=[r for r in rows if r['case']==case]
        # A hash-tied height rank reference, no score or label in ranking.
        ordered=sorted(g,key=lambda r:(-r['boards'],score(r['date']+':'+r['code'])))
        for i,r in enumerate(ordered):r['height_rank_Top3_rank']=i+1 if i<3 else 0
        ordered=sorted(g,key=lambda r:(r['ROLL252_score'] is None,-(r['ROLL252_score'] or 0),r['code']))
        for i,r in enumerate(ordered):r['archived_ROLL252_rank']=i+1 if i<3 else 0
    arms=list(FILES)+['archived_ROLL252','always_no']
    results={'created_at_utc':now(),'type':'ACTUAL_TWO_LLM_CONSULTATION_MASKED_HISTORICAL_REPLAY','rows':len(rows),'days':len(packet['days']),
        'events':{},'bootstrap':[],'paired':{},'by_day':{},'by_period':{},'source_hashes':{**refbind,str(calendar_path):sha(calendar_path)},
        'price_status_counts':pd.Series([r['price_status'] for r in rows]).value_counts().to_dict(),'new_fits':0,'production_changes':0}
    for label in ['strict_close','archived_promotion','touch']:
        results['events'][label]={'arms':{a:metrics(rows,a,label) for a in arms},
            'top':tops(rows,list(FILES)+['archived_ROLL252','height_rank_Top3'],label),
            'non_one_price':{a:metrics([r for r in rows if r['one_price']==0],a,label) for a in arms}}
    for label in ['strict_close','archived_promotion']:
        for arm in ['codex_independent','gemini_independent','archived_ROLL252','always_no']:
            results['bootstrap'].append(bootstrap(rows,'discussed_joint',arm,label))
            wins=[];losses=[];resolved=[]
            for r in rows:
                if r[label] is None:continue
                def ok(a):return r[a]==('YES' if r[label] else 'NO')
                if r[arm] in ('YES','NO') and r['discussed_joint'] in ('YES','NO'):
                    if ok('discussed_joint') and not ok(arm):wins.append(r['case']+'/'+r['id'])
                    if ok(arm) and not ok('discussed_joint'):losses.append(r['case']+'/'+r['id'])
                elif r[arm]=='UNKNOWN' and r['discussed_joint']!='UNKNOWN':resolved.append({'case_id':r['case']+'/'+r['id'],'correct':ok('discussed_joint')})
            common=[r for r in rows if r[label] is not None and r[arm]!='UNKNOWN' and r['discussed_joint']!='UNKNOWN']
            results['paired'][label+':joint_vs_'+arm]={'corrections':wins,'induced_errors':losses,'resolved_abstentions':resolved,
                'common_judged_known':len(common),'joint_accuracy_common':metrics(common,'discussed_joint',label)['accuracy_judged'],'other_accuracy_common':metrics(common,arm,label)['accuracy_judged']}
    for case in sorted({r['case'] for r in rows}):results['by_day'][case]={label:{a:metrics([r for r in rows if r['case']==case],a,label) for a in arms} for label in ['strict_close','archived_promotion']}
    for period,_,_ in PERIODS:results['by_period'][period]={label:{a:metrics([r for r in rows if r['period']==period],a,label) for a in arms} for label in ['strict_close','archived_promotion']}
    write(HERE/'results_AFTER_JOINT_FREEZE.json',results)
    pd.DataFrame(rows).to_csv(HERE/'complete_122_revealed_all_arms.csv',index=False,encoding='utf-8-sig',mode='x')
    write(HERE/'completion_evaluation_receipt.json',{'created_at_utc':now(),'decision_hashes':freeze['decision_hashes'],
        'evaluation_script_sha256':sha(Path(__file__)),'results_sha256':sha(HERE/'results_AFTER_JOINT_FREEZE.json'),
        'rows_csv_sha256':sha(HERE/'complete_122_revealed_all_arms.csv'),'protocol_unchanged':sha(HERE/'protocol_BEFORE_DATA_AND_DECISIONS.json')==prep['protocol_sha256'],
        'protected_files_unchanged':{p:sha(p)==h for p,h in prep['protected_files'].items()},'source_unchanged':sha(BUNDLE)==prep['source_sha256']})
    print(json.dumps({'price_status':results['price_status_counts'],'strict_close':results['events']['strict_close']['arms'],'archived_promotion':results['events']['archived_promotion']['arms']},ensure_ascii=False))

if __name__=='__main__':main()
