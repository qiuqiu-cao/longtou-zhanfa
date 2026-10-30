import pandas as pd, numpy as np, json, hashlib
from pathlib import Path
R=Path('results/wclt0925_historical_diagnostic_20261028'); source=Path('data/v3/imported/daily-bars.csv'); formula=Path('templates/同花顺公式_龙头竞价非一字选股_v2026.txt')
cols=['date','code','name','open','high','low','close','pre_close','volume','amount','turnover','float_cap','st','new_listing','source']
d=pd.read_csv(source,usecols=cols,dtype={'code':str},low_memory=False).sort_values(['code','date']).reset_index(drop=True)
dates=sorted(d.date.dropna().unique()); previous={dates[i]: dates[i-1] for i in range(1,len(dates))}; following={dates[i]:dates[i+1] for i in range(len(dates)-1)}
g=d.groupby('code',sort=False)
for c in ['date','close','high','volume','amount','turnover','source','float_cap']:
 d['p_'+c]=g[c].shift(1)
d['pp_close']=g.close.shift(2); d['pp_date']=g.date.shift(2)
for c in ['date','open','close']:
 d['n_'+c]=g[c].shift(-1)
d['adjacent']=d.p_date.eq(d.date.map(previous)) & d.pp_date.eq(d.p_date.map(previous))
d['next_adjacent']=d.n_date.eq(d.date.map(following))
# Converting Sina shares to THS lot-equivalent yields amount/shares = yuan/share.
d['prev_vwap']=d.p_amount/d.p_volume
d['prev_shares_proxy']=d.p_float_cap/d.p_close
d['current_shares_proxy']=d.float_cap/d.close
# CAPITAL is current bar's shares. Estimated float-cap-based proxy has unknown point-in-time provenance.
d['prev_turnover_capital_proxy']=d.p_volume/d.current_shares_proxy*100
src=d.source.str.startswith('Sina') & d.p_source.str.startswith('Sina')
board=d.code.str.startswith(('600','601','603','605','000','001','002','003'))
valid=src & board & ~d.st & ~d.new_listing & d.adjacent & (d.p_volume>0) & (d.current_shares_proxy>0) & (d.pp_close>0) & (d.p_close>0)
# Sanity-check previous daily traded VWAP must lie within previous daily range; obtain previous low.
d['p_low']=g.low.shift(1)
valid &= d.prev_vwap.ge(d.p_low-0.02) & d.prev_vwap.le(d.p_high+0.02)
p=d.p_close
base=valid & (p/d.pp_close>=1.098) & (p==d.p_high)
cond=base & (d.open<p*1.1) & (d.open/p>=1.02) & (d.open/p<=1.07) & (d.open>=d.p_high) & (d.open>=d.prev_vwap) & (d.p_amount>=300000000) & d.prev_turnover_capital_proxy.between(8,32)
s=d.loc[cond].copy(); s['limit_ref']=np.floor(s.p_close*110+0.50000001)/100; s['label_known']=(s.high<=s.limit_ref+0.011)&(s.low>0)&(s.close>0);s['hit_close']=np.where(s.label_known,(s.close>=s.limit_ref-0.001)&(s.close==s.high),np.nan);s['hit_touch']=np.where(s.label_known,s.high>=s.limit_ref-0.001,np.nan);s['close_open_pct']=(s.close/s.open-1)*100;s['next_open_pct']=np.where(s.next_adjacent,(s.n_open/s.open-1)*100,np.nan);s['next_close_pct']=np.where(s.next_adjacent,(s.n_close/s.open-1)*100,np.nan);s['month']=s.date.str[:7];s['year']=s.date.str[:4]
fields=['date','code','name','open','p_close','p_high','p_amount','prev_vwap','prev_turnover_capital_proxy','close','high','limit_ref','label_known','hit_close','hit_touch','close_open_pct','n_date','next_adjacent','n_open','next_open_pct','next_close_pct'];s[fields].to_csv(R/'signals.csv',index=False,encoding='utf-8-sig')
def summary(x):
 return {'signals':len(x),'days':int(x.date.nunique()),'label_known':int(x.hit_close.notna().sum()),'label_unknown':int(x.hit_close.isna().sum()),'close_limit':int(x.hit_close.sum()),'close_limit_pct':round(x.hit_close.mean()*100,2) if len(x) else None,'touch_limit':int(x.hit_touch.sum()),'touch_limit_pct':round(x.hit_touch.mean()*100,2) if len(x) else None,'close_open_avg_pct':round(x.close_open_pct.mean(),2) if len(x) else None,'next_known':int(x.next_open_pct.notna().sum()),'next_open_avg_pct':round(x.next_open_pct.mean(),2) if x.next_open_pct.notna().any() else None,'next_open_positive_pct':round(x.next_open_pct.gt(0).sum()/x.next_open_pct.notna().sum()*100,2) if x.next_open_pct.notna().any() else None}
res={'source_rows':len(d),'source_date_range':[d.date.min(),d.date.max()],'eligible_rows':int(valid.sum()),'previous_approx_limit_baseline':int(base.sum()),'all':summary(s),'year':{str(k):summary(v) for k,v in s.groupby('year')},'month':{str(k):summary(v) for k,v in s.groupby('month')},'formula_sha256':hashlib.sha256(formula.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'caveats':['incomplete surviving historical universe; historical ST and name not verified','Sina daily data only; no historical auction observations or fills','CAPITAL estimated with end-of-day float_cap/close, point-in-time provenance unverified','previous close is prior trade close, ex-right reference unverified','latest bar date 2026-09-16; date continuity based on available dataset trading dates, not independently verified official calendar','daily bars unadjusted but THS precision adjustment platform behavior unverified','all historic results exploratory and previously observed, not independent blind test']}
(R/'summary.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(res,ensure_ascii=False,indent=2))
