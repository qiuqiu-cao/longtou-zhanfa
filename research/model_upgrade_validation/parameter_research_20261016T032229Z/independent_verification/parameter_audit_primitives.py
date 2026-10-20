"""Independent audit primitives, with no training or effect-based search."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_labels(frame):
    return frame.label_valid.astype(str).str.lower().eq('true') & frame.target.isin([0,1])


def matrix(frame, call_column='evaluation_call', score_column=None, threshold=None):
    """Unknown labels never enter denominators; abstained positives stay missed."""
    known=frame[valid_labels(frame)]
    if score_column is None:
        p=known[call_column].eq('PROMOTE');n=known[call_column].eq('FAIL')
    else:
        if threshold is None:raise ValueError('explicit frozen threshold required')
        values=known[score_column]
        p=values.gt(threshold);n=values.notna()&values.le(threshold)
    y=known.target.eq(1)
    tp=int((p&y).sum());fp=int((p&~y).sum());fn=int((n&y).sum());tn=int((n&~y).sum())
    ap=int((~p&~n&y).sum());an=int((~p&~n&~y).sum());positives=int(y.sum());negative=int((~y).sum())
    rate=lambda a,d:a/d if d else np.nan
    return dict(candidates=len(frame),known=len(known),unknown=len(frame)-len(known),positives=positives,negatives=negative,
        TP=tp,FP=fp,FN_called=fn,TN=tn,abstained_positives=ap,abstained_negatives=an,
        Recall=rate(tp,positives),Precision=rate(tp,tp+fp),F1=rate(2*tp,positives+tp+fp),
        F0p5=rate(1.25*tp,1.25*tp+fp+.25*(positives-tp)),FPR=rate(fp,negative))


def outer_window(full, cutoff):
    known=full[valid_labels(full)&full.date.lt(cutoff)&full.target_date.lt(cutoff)]
    dates=sorted(known.date.unique())[-252:]
    out=known[known.date.isin(dates)].copy()
    if len(dates)!=252 or out.target_date.max()>=cutoff:raise AssertionError('invalid complete matured outer window')
    return out


def chronological_blocks(window):
    dates=sorted(window.date.unique())
    if len(dates)!=252:raise AssertionError('exact252 dates required')
    out=[]
    for block in [1,2,3]:
        validation_dates=dates[63*block:63*(block+1)]
        start=validation_dates[0]
        past=window[window.date.lt(start)&window.target_date.lt(start)]
        valid=window[window.date.isin(validation_dates)]
        if past.target.nunique()!=2 or past.target_date.max()>=start or set(past.date)&set(valid.date):
            raise AssertionError('inner block violates strict chronological maturity')
        out.append((block,past,valid))
    return out


def full_slot_topk(frame,k,score_column='ranking_score'):
    """Select before settlement; all signal dates contribute k slots."""
    eligible=frame[frame.daily_one_word.eq(0)]
    chosen=eligible.sort_values(['date',score_column,'code'],ascending=[True,False,True],na_position='last').groupby('date',sort=True).head(k)
    valid=valid_labels(chosen)
    slots=k*frame.date.nunique();hits=int((valid&chosen.target.eq(1)).sum())
    summary=dict(k=k,signal_days=frame.date.nunique(),full_slots=slots,selected=len(chosen),empty_slots=slots-len(chosen),
        known_selected=int(valid.sum()),unknown_selected=int((~valid).sum()),hits=hits,full_slot_hit_rate=hits/slots if slots else np.nan)
    return summary,chosen


def assert_old_predictions_exact(old,final):
    if len(old)!=30216 or len(final)!=37770:raise AssertionError('complete8old plus2new head sizes required')
    pd.testing.assert_frame_equal(old,final.iloc[:len(old)][old.columns],check_exact=True,check_dtype=False)
    if final.duplicated(['date','code','arm']).any():raise AssertionError('duplicate prediction identity')
    if not final.groupby('arm').size().eq(3777).all() or final.arm.nunique()!=10:raise AssertionError('all10 full3777 heads required')


def assert_september_and_current_unknown(frame):
    s=frame[frame.date.between('2026-09-01','2026-09-30')&frame.daily_one_word.eq(0)]
    if (len(s),int(valid_labels(s).sum()),int(s.target.eq(1).sum()),int((~valid_labels(s)).sum()))!=(143,107,40,36):
        raise AssertionError('original september denominator changed')
    c=frame[frame.date.eq('2026-10-15')]
    if len(c)!=6 or c.target.notna().any() or valid_labels(c).any():raise AssertionError('current6 unknown outcomes changed')


def scores_equal(actual,expected,tolerance=1e-14):
    a=np.asarray(actual,float);z=np.asarray(expected,float)
    np.testing.assert_allclose(a,z,rtol=0,atol=tolerance,equal_nan=True)
    finite=np.isfinite(a)&np.isfinite(z)
    return float(np.max(np.abs(a[finite]-z[finite]))) if finite.any() else 0.
