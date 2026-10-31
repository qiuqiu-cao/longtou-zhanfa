"""Previous-return similarity among same-day candidates; never uses labels."""
import numpy as np
import pandas as pd
FIELDS=['similar_peer_s1_logit_gap']
KEYS=['signal_date','code']

def feature_day(ret, codes, offsets, min_market=3, k=3):
    """ret: 60 x complete archived market matrix; NaNs are unknown sessions."""
    if ret.shape[0]!=60 or ret.shape[1]!=len(codes) or len(offsets)!=len(codes):raise ValueError('bad inputs')
    available=np.isfinite(ret); sums=np.where(available,ret,0).sum(axis=1); counts=available.sum(axis=1)
    edges=[];n=len(codes)
    for i in range(n):
        if not available[:,i].all():continue
        for j in range(i+1,n):
            if not available[:,j].all():continue
            cnt=counts-2
            if (cnt<min_market).any():continue
            market=(sums-ret[:,i]-ret[:,j])/cnt
            centered=market-market.mean();den=centered@centered
            if den<=1e-18:continue
            yi=ret[:,i]-ret[:,i].mean();yj=ret[:,j]-ret[:,j].mean()
            ei=yi-centered*(centered@yi/den);ej=yj-centered*(centered@yj/den)
            scale=np.sqrt((ei@ei)*(ej@ej))
            if scale<=1e-12:continue
            corr=float(np.clip((ei@ej)/scale,-1,1))
            edges.append((i,j,corr))
    # Candidate count may be smaller than complete market count. callers mask
    # all non-candidate indices below; they remain in market background.
    return edges

def build_day(ret, allcodes, candidate_codes, score_by_code, k=3):
    if len(set(allcodes))!=len(allcodes) or len(set(candidate_codes))!=len(candidate_codes):raise ValueError('duplicate')
    order=np.argsort(allcodes,kind='stable');ret=ret[:,order];allcodes=[allcodes[i] for i in order]
    candidate_codes=sorted(candidate_codes)
    idx={c:i for i,c in enumerate(allcodes)};choices=[]
    available=np.isfinite(ret);sums=np.where(available,ret,0).sum(axis=1);counts=available.sum(axis=1)
    for a,code in enumerate(candidate_codes):
        if code not in idx:continue
        i=idx[code]
        if not available[:,i].all():continue
        for other in candidate_codes[a+1:]:
            if other not in idx:continue
            j=idx[other]
            if not available[:,j].all():continue
            cnt=counts-2
            if (cnt<3).any():continue
            m=(sums-ret[:,i]-ret[:,j])/cnt;x=m-m.mean();den=x@x
            if den<=1e-18:continue
            yi=ret[:,i]-ret[:,i].mean();yj=ret[:,j]-ret[:,j].mean()
            ei=yi-x*(x@yi/den);ej=yj-x*(x@yj/den);v=np.sqrt((ei@ei)*(ej@ej))
            if v<=1e-12:continue
            corr=float(np.clip((ei@ej)/v,-1,1))
            choices.append((code,other,corr))
    rows=[]
    for code in candidate_codes:
        peers=[(j,c) for i,j,c in choices if i==code and c>0]+[(i,c) for i,j,c in choices if j==code and c>0]
        peers.sort(key=lambda a:(-a[1],a[0]));peers=peers[:k]
        gap=score_by_code[code]-np.mean([score_by_code[p] for p,_ in peers]) if peers else np.nan
        rows.append({'code':code,FIELDS[0]:gap,'peer_count':len(peers),'peers':str(peers)})
    return pd.DataFrame(rows),choices
