from run_joint_replay import *
old = HERE/'review_packet_SIGNAL_ONLY.json'
p = read(old)
for d in p['days']:
    for r in d['stocks']:
        if r['turnover_pct'] is not None: r['turnover_pct'] *= 100
p['notes'] += ' 单位补正：turnover_pct已由原模型比例乘100转成百分数，与past5内turnover同为百分数；市场premium/failed_rate/seal_rate仍是比例（0.01=1%）。observed_nonword_streak实际是本轮已观察且满足日线形态规则的换手板累计数，不能视为连续换手天数或真实承接认证；past5内最后记录未必为T，见lag。原市场前两行缺失仍保留未知，不补0。one_price=null表示当日日线与原池价型资格不足。'
# Record official session lags for existing past bar rows without touching outcomes.
data=joblib.load(BUNDLE)
bars=data['bars'][['date','code']].copy()
bars['date']=bars.date.astype(str).str[:10]; bars['code']=bars.code.astype(str).str.zfill(6)
sessions=sorted(str(v)[:10] for v in data['sessions'])
index={d:i for i,d in enumerate(sessions)}
mapping=read(HERE/'PRIVATE_mapping_DO_NOT_READ_BEFORE_JOINT_FREEZE.json')
lookup={(r['case'],r['id']):r for r in mapping}
for d in p['days']:
    for r in d['stocks']:
        m=lookup[(d['case'],r['id'])]
        days=bars.loc[bars.code.eq(m['code']) & bars.date.le(m['date']),'date'].sort_values().tail(5).tolist()
        assert len(days)==len(r['past5_ohlc_pct_vs_previous_close'])
        for h,day in zip(r['past5_ohlc_pct_vs_previous_close'],days):h['lag_sessions']=index[m['date']]-index[day]
write(HERE/'review_packet_SIGNAL_ONLY_v2.json',p)
text=(HERE/'review_packet_SIGNAL_ONLY.md').read_text(encoding='utf-8')
lines=text.splitlines()
for i,line in enumerate(lines):
    if line.startswith('|S'):
        vals=line.split('|')
        if vals[4]!='?':vals[4]=str(round(float(vals[4])*100,4))
        lines[i]='|'.join(vals)
lines[2]=p['notes']
(HERE/'review_packet_SIGNAL_ONLY_v2.md').write_text('\n'.join(lines),encoding='utf-8')
s=read(HERE/'decision_SCHEMA_NOT_A_PREDICTION.json'); s['packet_sha256']=sha(HERE/'review_packet_SIGNAL_ONLY_v2.json')
write(HERE/'decision_SCHEMA_v2_NOT_A_PREDICTION.json',s)
write(HERE/'unit_correction_BEFORE_ANY_REVIEW.json',{'created_at_utc':now(),'reason':'relay_field_context.py explicitly defines model turnover fraction; convert to displayed percentage and expose official session lags. No outcome was read, no decision exists yet.','old_packet_sha256':sha(old),'final_packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY_v2.json'),'rows':len(mapping),'outcome_read':False,'original_files_preserved':True})
print(json.dumps({'packet_sha256':s['packet_sha256'],'rows':len(mapping),'outcome_read':False}))
