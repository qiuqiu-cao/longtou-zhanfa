"""Conditional pairwise preferences; output is cohort-relative, never event probability."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import short_term_thinking_research as previous

VERSION="conditional-pairwise-left-right-difference-v1"


def pair_features(left,right):
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    if left.shape!=right.shape or left.ndim!=2:raise ValueError("equal two-dimensional pairs required")
    # Left/right keep shared cycle context and each side's native missingness.
    return np.concatenate([left,right,left-right],axis=1)


def build_pairs(frame,fields):
    xparts=[];labels=[];weights=[];groups=[]
    for day,full in frame.groupby("date",sort=True):
        known=full[full.target.notna() & full.label_valid.fillna(False).astype(bool)]
        pos=known[known.target.eq(1)];neg=known[known.target.eq(0)]
        n=len(pos)*len(neg)
        status="MIXED_KNOWN_LABELS" if n else "NO_KNOWN_ROWS" if not len(known) else "SINGLE_KNOWN_CANDIDATE" if len(known)==1 else "ALL_KNOWN_POSITIVE" if len(pos)==len(known) else "ALL_KNOWN_NEGATIVE"
        groups.append(dict(date=day,complete_nonword_candidates=len(full),known=len(known),unknown_labels=len(full)-len(known),
            positives=len(pos),negatives=len(neg),directed_pairs=2*n,training_group_status=status))
        if not n:continue
        a=np.repeat(pos[fields].to_numpy(float),len(neg),axis=0)
        z=np.tile(neg[fields].to_numpy(float),(len(pos),1))
        xparts.extend([pair_features(a,z),pair_features(z,a)])
        labels.extend([np.ones(n,dtype=int),np.zeros(n,dtype=int)])
        weights.extend([np.full(n,1/(2*n)),np.full(n,1/(2*n))])
    if not xparts:raise ValueError("no mixed day can supervise ranking")
    return np.concatenate(xparts),np.concatenate(labels),np.concatenate(weights),pd.DataFrame(groups)


def fit_ranker(frame,fields,params=None):
    x,y,w,groups=build_pairs(frame,fields)
    model=HistGradientBoostingClassifier(**(previous.PARAMS if params is None else params))
    with threadpool_limits(limits=2):model.fit(x,y,sample_weight=w)
    return model,groups,dict(directed_pairs=len(y),effective_mixed_days=int(groups.directed_pairs.gt(0).sum()),
        supervised_weight_mass=float(w.sum()),base_fields=len(fields),pair_input_fields=x.shape[1])


def predict_cohort(model,frame,fields):
    """Each complete day is inferred without consulting any target or outcome."""
    result=[]
    for day,group in frame.groupby("date",sort=True):
        group=group.sort_values("code");n=len(group)
        if n==1:
            row=group.iloc[0]
            result.append(dict(date=day,code=row.code,ranking_score=np.nan,opponent_count=0,
                pairwise_antisymmetry_error=0.,ranking_status="SINGLETON_NO_COMPARATIVE_SCORE"))
            continue
        values=group[fields].to_numpy(float)
        left,right=np.where(~np.eye(n,dtype=bool))
        x=pair_features(values[left],values[right])
        with threadpool_limits(limits=2):
            probabilities=model.predict_proba(x)[:,list(model.classes_).index(1)]
        matrix=np.full((n,n),np.nan);matrix[left,right]=probabilities
        symmetric=.5*(matrix+1-matrix.T)
        error=float(np.nanmax(np.abs(symmetric+symmetric.T-1)))
        for i,row in enumerate(group.itertuples()):
            result.append(dict(date=day,code=row.code,ranking_score=float(np.nanmean(symmetric[i])),opponent_count=n-1,
                pairwise_antisymmetry_error=error,ranking_status="UNVALIDATED_COHORT_RELATIVE_PREFERENCE"))
    return pd.DataFrame(result)
