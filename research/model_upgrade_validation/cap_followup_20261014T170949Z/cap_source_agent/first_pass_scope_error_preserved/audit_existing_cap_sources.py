"""Read-only audit of existing cap sources; never fit or rewrite original data."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json
import joblib
import numpy as np
import pandas as pd
import akshare as ak

ROOT = Path(r'D:\Documents\ChatGPT\龙头战法')
OUT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def writej(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def summary(frame):
    return {'rows':int(len(frame)), 'dates':int(frame.date.nunique()), 'codes':int(frame.code.nunique()),
            'date_min':str(frame.date.min()), 'date_max':str(frame.date.max()),
            'duplicates_date_code':int(frame.duplicated(['date','code']).sum())}

def exact(x,y):
    return np.isclose(x, y, rtol=1e-12, atol=1e-6, equal_nan=True)

receipt_path=ROOT/'research/model_upgrade_validation/relay_context_20261011T151204Z/source_reconstruction_BEFORE_FITS.json'
receipt=readj(receipt_path)
prepared_path=Path(receipt['prepared_path'])
prepared=joblib.load(prepared_path)
signal=prepared['relay'][['date','code','boards','price','signal_float_cap','float_cap_log','turnover','signal_amount','amount_float_cap']].copy()
signal.code=signal.code.astype(str).str.zfill(6)
keys=pd.MultiIndex.from_frame(signal[['date','code']])
v16=ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2'
daily_paths=[v16/'point_in_time_current_daily_bars.csv',v16/'verified_recovered_daily_bars.csv']
parts=[]
all_d=[]
daily_columns=['date','code','open','high','low','close','float_cap','volume','amount','turnover','source']
for path in daily_paths:
    d=pd.read_csv(path,dtype={'code':str},usecols=daily_columns)
    d.code=d.code.str.zfill(6)
    all_d.append(d)
    ix=pd.MultiIndex.from_frame(d[['date','code']])
    matched=d.loc[ix.isin(keys)].copy()
    matched['daily_source_path']=str(path.relative_to(ROOT))
    parts.append(matched)
d=pd.concat(all_d,ignore_index=True)
matched=pd.concat(parts,ignore_index=True)
j=signal.merge(matched,on=['date','code'],how='left',validate='one_to_one',indicator=True)
j['candidate_cap_matches_daily']=exact(j.signal_float_cap,j.float_cap)
j['candidate_price_matches_daily']=exact(j.price,j.close)
j['candidate_log_matches_formula']=exact(j.float_cap_log,np.log(j.signal_float_cap.clip(lower=1)))
j['candidate_turnover_fraction_matches_daily_pct']=exact(j.turnover_x,j.turnover_y/100)
j['direct_vwap_sina_yuan']=np.where(j.source.eq('Sina unadjusted daily via AKShare fallback')&j.volume.gt(0)&j.amount.gt(0),j.amount/j.volume,np.nan)
j['vwap_within_raw_ohlc_1cent']=(j.direct_vwap_sina_yuan.ge(j.low-.01)&j.direct_vwap_sina_yuan.le(j.high+.01))
j['stored_cap_ratio']=j.signal_float_cap*j.turnover_x/j.signal_amount
j['direct_close_over_vwap']=j.price/j.direct_vwap_sina_yuan
j['cap_ratio_equal_direct_close_over_vwap']=exact(j.stored_cap_ratio,j.direct_close_over_vwap)
j.to_csv(OUT/'full9245_cap_source_binding_NO_TARGETS.csv',index=False,encoding='utf-8-sig')

pos=d.float_cap.gt(0)&d.close.gt(0)
d['implied_float_shares']=np.where(pos,d.float_cap/d.close,np.nan)
d['volume_turnover_implied_shares']=np.where(d.turnover.gt(0),100*d.volume/d.turnover,np.nan)
d['share_identity_relative_error']=np.abs(d.implied_float_shares/d.volume_turnover_implied_shares-1)
dc=d.loc[pos].sort_values(['code','date']).copy()
dc['previous_implied_shares']=dc.groupby('code').implied_float_shares.shift()
dc['previous_date']=dc.groupby('code').date.shift()
dc['relative_share_change']=dc.implied_float_shares/dc.previous_implied_shares-1
changes=dc.loc[dc.relative_share_change.abs().gt(1e-6)].copy()
changes.to_csv(OUT/'all_archived_share_transitions_INFERRED_NO_TARGETS.csv',index=False,encoding='utf-8-sig')
bycode=dc.groupby('code').agg(rows=('date','size'),date_min=('date','min'),date_max=('date','max'),shares_min=('implied_float_shares','min'),shares_max=('implied_float_shares','max'))
bycode['shares_range_fraction']=bycode.shares_max/bycode.shares_min-1
bycode['changing_over_1ppm']=bycode.shares_range_fraction.gt(1e-6)
bycode.to_csv(OUT/'full_code_share_variation_INFERRED.csv',encoding='utf-8-sig')

bridge_path=ROOT/'research/model_upgrade_validation/peer_discrimination_20261013T170021Z/September182_cap_source_bridge_NO_TARGETS.csv'
bridge=pd.read_csv(bridge_path,dtype={'code':str})
bridge.code=bridge.code.str.zfill(6)
v275_path=ROOT/'data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv'
v275=pd.read_csv(v275_path,dtype={'code':str},usecols=['date','code','price','float_cap','amount','turnover','captured_at','source_path','source_sha256'])
v275.code=v275.code.str.zfill(6)
pool_checks=[]
for rel,g in v275.groupby('source_path'):
    path=ROOT/rel
    meta_path=path.parent/'manifest.json'
    meta=readj(meta_path)
    expected=g.source_sha256.unique().tolist()
    dataset=meta['datasets']['limit_up']
    pool_checks.append({'source_path':rel,'actual_sha256':sha(path),'expected_sha256':expected,
        'matches_reference':expected==[sha(path)],'capture_manifest':str(meta_path.relative_to(ROOT)),
        'manifest_date':str(meta['date']),'captured_at':meta.get('collected_at'),
        'manifest_rows':dataset.get('rows'),'manifest_has_pool_sha256':bool(dataset.get('sha256')),
        'manifest_source':dataset.get('source'), 'manifest_sha256':sha(meta_path)})
bridge_bound=bridge.merge(v275[['date','code','float_cap','price','amount','turnover']],on=['date','code'],validate='one_to_one',suffixes=('_bridge','_v275'))
cross=v275.merge(matched,on=['date','code'],how='left',validate='one_to_one',indicator=True,suffixes=('_pool','_daily'))
cross['cap_ratio_daily_to_pool']=cross.float_cap_daily/cross.float_cap_pool
cross['close_matches_pool_price']=exact(cross.close,cross.price)
cross.to_csv(OUT/'full217_v275_to_cutoff_daily_comparison_NO_TARGETS.csv',index=False,encoding='utf-8-sig')

# Existing later daily file, exact existing source candidates only, used for cross-source diagnosis.
later_daily=ROOT/'data/v3/imported/daily-bars.csv'
v275_keys=pd.MultiIndex.from_frame(v275[['date','code']])
later=[]
for chunk in pd.read_csv(later_daily,dtype={'code':str},usecols=daily_columns,chunksize=150000):
    chunk.code=chunk.code.str.zfill(6)
    ix=pd.MultiIndex.from_frame(chunk[['date','code']])
    later.append(chunk.loc[ix.isin(v275_keys)])
later=pd.concat(later,ignore_index=True)
later_cross=v275.merge(later,on=['date','code'],how='left',validate='one_to_one',indicator=True,suffixes=('_pool','_daily'))
later_cross['cap_ratio_daily_to_pool']=later_cross.float_cap_daily/later_cross.float_cap_pool
later_cross['close_matches_pool_price']=exact(later_cross.close,later_cross.price)
later_cross.to_csv(OUT/'full217_v275_to_later_existing_daily_comparison_NO_TARGETS.csv',index=False,encoding='utf-8-sig')

# Direct amount/volume qualification uses original daily rows, not candidate labels.
v278_dir=ROOT/'data/v278/september_unadjusted_20261002_v3'
v278_report=readj(v278_dir/'report.json')
v278_frames=[]
v278_raw_checks=[]
for row in v278_report['audit']:
    if row.get('status')!='UNADJUSTED_DAY_AVAILABLE':
        continue
    raw_path=v278_dir/'raw'/(row['symbol']+'.json')
    raw=readj(raw_path)
    values=(raw.get('data') or {}).get(row['symbol'],{}).get('day')
    frame=pd.DataFrame([r[:6] for r in values],columns=['date','open','close','high','low','volume'])
    frame=frame.loc[frame.date.between('2026-08-20','2026-09-30')].copy()
    for col in ['open','close','high','low','volume']:
        frame[col]=pd.to_numeric(frame[col])
    frame.insert(0,'symbol',row['symbol'])
    v278_frames.append(frame)
    v278_raw_checks.append({'symbol':row['symbol'],'actual_sha256':sha(raw_path),'expected_sha256':row['sha256'],
        'hash_match':sha(raw_path)==row['sha256'],'requested_at_utc':row.get('requested_at_utc'),'captured_at_utc':row.get('captured_at_utc'),
        'timestamp_semantics':row.get('timestamp_semantics')})
v278_rebuilt=pd.concat(v278_frames,ignore_index=True)
v278_observed=pd.read_csv(v278_dir/'unadjusted_daily.csv')
pd.testing.assert_frame_equal(v278_rebuilt.sort_values(['symbol','date']).reset_index(drop=True),v278_observed.sort_values(['symbol','date']).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-10)
v278_rebuilt['code']=v278_rebuilt.symbol.str[2:]
sept=signal.loc[signal.date.ge('2026-09-01')].merge(v278_rebuilt,on=['date','code'],how='left',validate='one_to_one')
sept['vwap_if_volume_is_shares']=sept.signal_amount/sept.volume
sept['vwap_if_volume_is_100_share_lots']=sept.signal_amount/(sept.volume*100)
sept['shares_unit_within_ohlc_1cent']=sept.vwap_if_volume_is_shares.ge(sept.low-.01)&sept.vwap_if_volume_is_shares.le(sept.high+.01)
sept['lots_unit_within_ohlc_1cent']=sept.vwap_if_volume_is_100_share_lots.ge(sept.low-.01)&sept.vwap_if_volume_is_100_share_lots.le(sept.high+.01)
sept['candidate_close_matches_raw_1cent']=sept.price.sub(sept.close).abs().le(.011)
sept.to_csv(OUT/'September182_direct_vwap_UNIT_HYPOTHESES_NO_TARGETS.csv',index=False,encoding='utf-8-sig')

v16_manifest_path=ROOT/'results/v16/point_in_time_rebuild_manifest_cutoff_20260831_v2.json'
v16_manifest=readj(v16_manifest_path)
library_path=Path(inspect.getfile(ak.stock_zh_a_daily))
library_source=inspect.getsource(ak.stock_zh_a_daily)
(OUT/'current_installed_stock_zh_a_daily_SOURCE_NOT_ORIGINAL.py.txt').write_text(library_source,encoding='utf-8')
zt_lib_path=Path(inspect.getfile(ak.stock_zt_pool_em))
paths=[receipt_path,prepared_path,*daily_paths,v16/'leader_candidates.csv',bridge_path,v275_path,v16_manifest_path,ROOT/'leader_system_v3.py',ROOT/'survivorship_rebuild_v6.py',library_path,zt_lib_path,later_daily,ROOT/'data/v3/imported/daily-bars-manifest.json',ROOT/'data/v3/imported/history-progress.csv',v278_dir/'report.json',v278_dir/'unadjusted_daily.csv']
source_checks={str(p):sha(p) for p in paths}
v16_manifest_checks={k:{'path':block['path'],'actual_sha256':sha(block['path']),'expected_sha256':block['sha256'],'matches':sha(block['path'])==block['sha256']} for k,block in v16_manifest['files'].items() if k in ['corrected_current','recovered_raw','candidates']}

identity_observable=d.share_identity_relative_error.notna() & np.isfinite(d.share_identity_relative_error)
hist=j.date.lt('2026-09-01')
def cross_stats(frame):
    m=frame['_merge'].eq('both')
    c=m&frame.float_cap_daily.gt(0)&frame.float_cap_pool.gt(0)
    ratios=frame.loc[c,'cap_ratio_daily_to_pool']
    return {'source_rows':int(len(frame)),'exact_date_code_matches':int(m.sum()),'positive_caps_both':int(c.sum()),
        'price_match_on_positive_caps':int((c&frame.close_matches_pool_price).sum()),
        'cap_match_rtol1e_12_atol1e_6':int((c&exact(frame.float_cap_daily,frame.float_cap_pool)).sum()),
        'cap_match_within_1pct':int((c&frame.cap_ratio_daily_to_pool.sub(1).abs().le(.01)).sum()),
        'ratio_min':float(ratios.min()) if len(ratios) else None,'ratio_median':float(ratios.median()) if len(ratios) else None,
        'ratio_max':float(ratios.max()) if len(ratios) else None,
        'by_daily_source':frame.loc[m,'source'].value_counts().to_dict()}

finding={
    'formed_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'existing source audit; no fits or source rewrite',
    'new_fits':0,'new_data_requests':0,'original_prepared_sha_match':sha(prepared_path)==receipt['prepared_sha256'],
    'units':{'signal_float_cap':'RMB yuan inferred from historical outstanding shares × unadjusted close in yuan; supplier field labelled 流通市值; dimensional equivalence not sufficient to certify identical float-share definitions',
        'float_cap_log':'natural log(max(signal_float_cap,1)); yuan scale', 'historical_turnover':'percent in daily archive; fraction in prepared relay',
        'historical_sina_volume':'shares, inferred by identity with cap/close and turnover',
        'current_library_outstanding_share':'vendor historical share series ×10000 to shares; outer date join then forward fill; no current-snapshot backward fill in inspected current implementation'},
    'prepared_scope':summary(signal),'historical_candidates':int(hist.sum()),'historical_positive_cap':int((hist&j.signal_float_cap.gt(0)).sum()),
    'historical_missing_cap':int((hist&j.signal_float_cap.isna()).sum()),'september_original_cap_nonmissing':int((~hist&j.signal_float_cap.notna()).sum()),
    'daily_scope':summary(d),'daily_source_counts':d.source.value_counts(dropna=False).to_dict(),
    'full_candidate_binding':{'exact_date_code_matches':int(j['_merge'].eq('both').sum()),'cap_equal_including_missing':int(j.candidate_cap_matches_daily.sum()),
        'price_equal_including_missing':int(j.candidate_price_matches_daily.sum()),'log_formula_equal_including_missing':int(j.candidate_log_matches_formula.sum()),
        'turnover_conversion_equal_including_missing':int(j.candidate_turnover_fraction_matches_daily_pct.sum()),
        'historical_cap_equal':int((hist&j.candidate_cap_matches_daily).sum()),'historical_price_equal':int((hist&j.candidate_price_matches_daily).sum())},
    'daily_share_identity':{'positive_cap_rows':int(pos.sum()),'identity_observable':int(identity_observable.sum()),
        'identity_match_within_1e_10':int((identity_observable&d.share_identity_relative_error.le(1e-10)).sum()),
        'identity_error_max':float(d.loc[identity_observable,'share_identity_relative_error'].max()),
        'positive_cap_codes':int(bycode.shape[0]),'codes_with_changing_shares_over_1ppm':int(bycode.changing_over_1ppm.sum()),
        'daily_share_change_transitions_over_1ppm':int(len(changes)),'interpretation':'Archived implied share counts vary within codes; a blanket statement that current shares were applied backward is unsupported. Derived identity is circular and does not verify vendor change-effective dates.'},
    'bridge_scope':summary(bridge),'bridge_cap_matches_v275':int(exact(bridge_bound.float_cap_bridge,bridge_bound.float_cap_v275).sum()),
    'v275_to_cutoff_daily':cross_stats(cross),'v275_to_later_existing_daily':cross_stats(later_cross),
    'direct_daily_vwap_audit':{'historical_sina_direct_vwap_observable':int(j.direct_vwap_sina_yuan.notna().sum()),
        'historical_within_raw_ohlc_1cent':int((j.direct_vwap_sina_yuan.notna()&j.vwap_within_raw_ohlc_1cent).sum()),
        'historical_direct_vwap_cap_ratio_identity_match':int((j.direct_vwap_sina_yuan.notna()&j.cap_ratio_equal_direct_close_over_vwap).sum()),
        'historical_vwap_identity_max_abs_error':float(j.loc[j.direct_vwap_sina_yuan.notna(),'stored_cap_ratio'].sub(j.loc[j.direct_vwap_sina_yuan.notna(),'direct_close_over_vwap']).abs().max()),
        'september_exact_daily_volume_matches':int(sept.volume.notna().sum()),'september_positive_amount_volume':int((sept.signal_amount.gt(0)&sept.volume.gt(0)).sum()),
        'september_shares_unit_ohlc_pass':int(sept.shares_unit_within_ohlc_1cent.sum()),'september_100share_lot_unit_ohlc_pass':int(sept.lots_unit_within_ohlc_1cent.sum()),
        'september_close_matches_original_within_1cent':int(sept.candidate_close_matches_raw_1cent.sum()),
        'raw_tencent_archive_hashes_all_match':all(r['hash_match'] for r in v278_raw_checks),'raw_tencent_normalized_reconstruction_matches':True,
        'strict_full9245_volume_unit_and_original_asof_certification':False,
        'interpretation':'Sina direct amount/volume is observationally reconstructable, but close/VWAP duplicates the proposed cap×turnover/amount ratio if share denominators coincide. Tencent ×100 unit convention is an OHLC consistency hypothesis here, not independent unit documentation or strict as-of qualification. No target effect measured.'},
    'v278_original_raw_checks':v278_raw_checks,
    'pool_source_hash_checks':pool_checks,'v16_original_manifest_checks':v16_manifest_checks,
    'v16_generated_at_utc':v16_manifest.get('generated_at_utc'),'current_daily_manifest':readj(ROOT/'data/v3/imported/daily-bars-manifest.json'),
    'source_sha256':source_checks,
    'strict_historical_asof_certification':False,'comparable_full_training_plus_september_certified':False,
    'numerical_challenger_source_qualification_passed':False,
    'known_limits':['Original Sina raw HTTP share timeline and received time absent from cutoff csv; outstanding_share column was dropped after conversion.',
        'Inspected AKShare source is installed now; original library version/source hash at collection not bound, so cannot assert identical original implementation.',
        'Forward-filled vendor date series has no archival publication/effective-date/correction history; corporate action treatment cannot be independently verified.',
        'V16 security eligibility is separately verified; security-status completeness does not certify float capital point-in-time completeness.',
        'V275 captured_at mixes naive and timezone-aware strings; naive strings do not supply original timezone.',
        'Current imported daily-bars file has later manifest and changed source scope; is diagnostic cross-source evidence, not the exact original V16 raw input.',
        'Common date-code coverage is limited; supplier cap may use different circulating-share definitions. No whole-history common-vendor cap source certified.',
        'Sina cap/close and volume/turnover identity is mechanical because turnover and cap use the same outstanding-share input; it is not independent validation.'],
    'next_concrete_source_gate':'Need original date-indexed vendor outstanding-share responses or already-stored same-definition historical cap series including effective dates and timestamps. Without those, retain source-specific diagnostic layer and avoid claiming strict historical as-of or authorized challenger.'
}
writej('cap_source_findings.json',finding)
print(json.dumps({k:finding[k] for k in ['prepared_scope','historical_positive_cap','historical_missing_cap','full_candidate_binding','daily_share_identity','v275_to_cutoff_daily','v275_to_later_existing_daily','numerical_challenger_source_qualification_passed']},ensure_ascii=False,indent=2))
