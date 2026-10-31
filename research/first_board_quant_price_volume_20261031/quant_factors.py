"""Past daily-VWAP distribution, not current holders' cost or tick-level chips."""
import numpy as np
import pandas as pd

KEYS=['signal_date','code']
FIELDS=['daily_vwap_log_dispersion20','recent5_vs_prior15_vwap',
        'signal_close_vs_weighted_vwap_q80_20','prior_vwap_above_signal_volume_fraction20']

def window_values(w):
    """22 exact consecutive bars: S-21 validates first reference, then S-20..S."""
    if len(w)!=22 or not np.array_equal(np.diff(w.p.to_numpy(int)),np.ones(21,dtype=int)):
        return [np.nan]*4,'MISSING_OFFICIAL_SESSION_WINDOW'
    a=w[['open','high','low','close','pre_close','volume','amount']].to_numpy(float)
    if not np.isfinite(a).all() or not (a>0).all():return [np.nan]*4,'INVALID_PRICE_VOLUME'
    o,h,l,c,ref,v,amount=a.T
    if not ((h>=np.maximum(o,c))&(l<=np.minimum(o,c))&(h>=l)).all():return [np.nan]*4,'INVALID_OHLC'
    if not np.isclose(a[:,:4]*100,np.rint(a[:,:4]*100),atol=1e-6,rtol=0).all():return [np.nan]*4,'NON_CENT_OHLC'
    if not (np.abs(ref[1:]-c[:-1])<=.011).all():return [np.nan]*4,'REFERENCE_DISCONTINUITY'
    # The original CSV volume is shares, amount is yuan; never auto-try factors.
    price=amount/v
    if not ((price>=l-.011)&(price<=h+.011)).all():return [np.nan]*4,'AMOUNT_VOLUME_UNIT_OR_PRICE_MISMATCH'
    pv=price[1:-1];vv=v[1:-1];aa=amount[1:-1]
    weights=vv/vv.sum();logp=np.log(pv);center=np.dot(weights,logp)
    dispersion=np.sqrt(np.dot(weights,(logp-center)**2))
    migration=np.log((aa[-5:].sum()/vv[-5:].sum())/(aa[:-5].sum()/vv[:-5].sum()))
    order=np.argsort(pv,kind='stable');k=min(int(np.searchsorted(np.cumsum(weights[order]),.8,side='left')),19)
    q80=pv[order[k]]
    return [float(dispersion),float(migration),float(np.log(c[-1]/q80)),float(weights[pv>c[-1]].sum())],'COMPLETE_NOMINAL_DAILY_VWAP_WINDOW'

def build(bars,keys):
    if list(keys)!=KEYS:raise ValueError('Only signal identities accepted; no scores or labels')
    if keys.duplicated(KEYS).any() or bars.duplicated(['code','p']).any():raise ValueError('duplicate identity')
    groups={code:g.sort_values('p').reset_index(drop=True) for code,g in bars.groupby('code',sort=False)}
    rows=[]
    for r in keys.itertuples(index=False):
        g=groups.get(r.code)
        values,status=[np.nan]*4,'MISSING_STOCK_OR_SIGNAL'
        source_close=np.nan
        if g is not None:
            loc=np.flatnonzero(g.date.to_numpy()==r.signal_date)
            if len(loc):
                i=int(loc[0]);source_close=float(g.close.iloc[i])
                values,status=window_values(g.iloc[max(0,i-21):i+1])
        rows.append(dict(signal_date=r.signal_date,code=r.code,source_close=source_close,
                         factor_source_status=status,**dict(zip(FIELDS,values))))
    return pd.DataFrame(rows)
