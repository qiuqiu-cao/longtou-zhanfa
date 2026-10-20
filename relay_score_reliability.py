"""Retrospective score reliability diagnostics; never changes or calibrates scores."""
import numpy as np
import pandas as pd


def reliability(frame):
    summaries=[];bins=[]
    for arm,part in frame.groupby("arm",sort=True):
        if arm=="PAIRWISE_DAY_RANK":
            summaries.append(dict(arm=arm,applicability="NO_ABSOLUTE_EVENT_HEAD",candidates=len(part),known_scored=0,
                Brier=None,log_loss=None,ECE10=None,event_probability_validated=False))
            continue
        part=part[part.daily_one_word.eq(0)].copy()
        score=pd.to_numeric(part.score,errors="coerce")
        if "b_score" in part:
            raw=pd.to_numeric(part.b_score,errors="coerce")
            score=raw.combine_first(score)
        if ((score<0)|(score>1)).any():raise ValueError("event score diagnostic requires original scale0to1")
        valid=part.target.isin([0,1])&part.label_valid.fillna(False).astype(bool)&score.notna()
        s,y=score[valid].to_numpy(float),part.loc[valid,"target"].to_numpy(float)
        count=len(s);ece=0.
        indices=np.minimum((s*10).astype(int),9)
        for i in range(10):
            a=indices==i;n=int(a.sum())
            mean=float(s[a].mean()) if n else None
            rate=float(y[a].mean()) if n else None
            if n:ece+=n/count*abs(mean-rate)
            bins.append(dict(arm=arm,bin_index=i,lower=i/10,upper=(i+1)/10,upper_inclusive=i==9,
                known_scored=n,mean_internal_score=mean,observed_event_fraction=rate,
                interpretation="Retrospective internal-score bin, not a new calibrated probability"))
        clipped=np.clip(s,1e-15,1-1e-15)
        summaries.append(dict(arm=arm,applicability="UNCALIBRATED_RAW_CLASSIFIER_SCORE_DIAGNOSTIC",candidates=len(part),
            known_scored=count,unknown_or_unscored=len(part)-count,
            Brier=float(np.mean((s-y)**2)) if count else None,
            log_loss=float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))) if count else None,
            ECE10=ece if count else None,event_probability_validated=False))
    return pd.DataFrame(summaries),pd.DataFrame(bins)
