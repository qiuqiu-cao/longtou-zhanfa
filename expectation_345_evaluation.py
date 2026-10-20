"""Whole-cohort metrics, explicit abstentions and repeated-review agreement."""
import numpy as np
import pandas as pd


def classification(frame):
    known = frame[frame.target.notna() & frame.label_valid.fillna(False).astype(bool)]
    y = known.target.eq(1)
    p, n = known.call.eq("PROMOTE"), known.call.eq("FAIL")
    tp, fp, direct_fn, tn = [int(v.sum()) for v in [y & p, ~y & p, y & n, ~y & n]]
    abstain_pos, abstain_neg = int((y & ~p & ~n).sum()), int((~y & ~p & ~n).sum())
    positives = int(y.sum()); negatives = len(known)-positives
    missed = direct_fn + abstain_pos
    return dict(candidates=len(frame), known=len(known), unknown_outcomes=len(frame)-len(known), positives=positives,
        positive_calls=int(frame.call.eq("PROMOTE").sum()), negative_calls=int(frame.call.eq("FAIL").sum()),
        abstentions=int(frame.call.eq("UNKNOWN").sum()), tp=tp, fp=fp, fn_called=direct_fn, tn=tn,
        abstained_positives=abstain_pos, abstained_negatives=abstain_neg, missed_positives=missed,
        Recall=tp/positives if positives else None, Precision=tp/(tp+fp) if tp+fp else None,
        F1=2*tp/(2*tp+fp+missed) if 2*tp+fp+missed else None,
        false_positive_rate=fp/negatives if negatives else None,
        decision_coverage=(int(p.sum())+int(n.sum()))/len(known) if len(known) else None,
        full_known_accuracy=(tp+tn)/len(known) if len(known) else None,
        called_accuracy=(tp+tn)/(int(p.sum())+int(n.sum())) if p.sum()+n.sum() else None,
        utility_TP_minus_1p5FP=tp-1.5*fp)


def selection(frame, k=1, same_height=False, height=False):
    eligible = frame[frame.daily_one_word.eq(0)].copy()
    group = ["date", "boards"] if same_height else ["date"]
    order = group + (["boards"] if height and not same_height else []) + ([] if height else ["ranking_score"]) + ["code"]
    asc = [True]*len(group) + ([False] if height and not same_height else []) + ([] if height else [False]) + [True]
    return eligible.sort_values(order, ascending=asc, na_position="last").groupby(group, sort=True).head(k)


def ranking(frame, k=1, same_height=False, height=False):
    chosen = selection(frame, k, same_height, height)
    known = chosen[chosen.target.notna() & chosen.label_valid.fillna(False).astype(bool)]
    group = ["date", "boards"] if same_height else ["date"]
    sizes = frame[frame.daily_one_word.eq(0)].groupby(group).size()
    return dict(selected=len(chosen), known=len(known), unknown=len(chosen)-len(known), groups=len(sizes),
        groups_fewer_than_k=int(sizes.lt(k).sum()), candidate_hit_rate=float(known.target.mean()) if len(known) else None,
        unknown_or_ineligible_selected=int(chosen.call.eq("UNKNOWN").sum()),
        grouping="DATE_HEIGHT" if same_height else "DATE", tie_rule="descending score or height, ascending code")


def block_intervals(reference, challenger, draws=1500, family_tests=24):
    keys = ["date", "code"]
    z = reference[keys + ["target", "label_valid", "call"]].merge(challenger[keys + ["call"]], on=keys, suffixes=("_ref", "_new"), validate="one_to_one")
    days = sorted(z.date.unique())
    valid = z.target.notna() & z.label_valid.fillna(False).astype(bool)
    y = z.target.eq(1)
    z["positives"] = (valid & y).astype(int)
    for arm in ["ref", "new"]:
        p = z["call_"+arm].eq("PROMOTE")
        z["tp_"+arm] = (valid & y & p).astype(int)
        z["fp_"+arm] = (valid & ~y & p).astype(int)
    values = z.groupby("date")[["positives", "tp_ref", "fp_ref", "tp_new", "fp_new"]].sum().reindex(days, fill_value=0).to_numpy(float)
    if not len(values): return {}
    n=len(values); length=min(5,n); rng=np.random.default_rng(20261015)
    starts=rng.integers(0,n,size=(draws,(n+length-1)//length))
    ids=((starts[:,:,None]+np.arange(length))%n).reshape(draws,-1)[:,:n]
    sums=values[ids].sum(axis=1)
    out={}
    for key in ["Recall", "Precision", "F1", "utility_per_signal_day"]:
        def compute(tp,fp):
            den=sums[:,0] if key=="Recall" else tp+fp if key=="Precision" else sums[:,0]+tp+fp
            return (tp-1.5*fp)/n if key=="utility_per_signal_day" else np.divide(tp*(2 if key=="F1" else 1),den,out=np.full(len(den),np.nan),where=den>0)
        delta=compute(sums[:,3],sums[:,4])-compute(sums[:,1],sums[:,2])
        finite=delta[np.isfinite(delta)];alpha=.05/family_tests
        out[key]=dict(lower=float(np.quantile(finite,alpha/2)) if len(finite) else None,upper=float(np.quantile(finite,1-alpha/2)) if len(finite) else None,
            draws=draws,blocks="5 consecutive observed full signal dates including zero-positive/unknown days",family_tests=family_tests,
            correction="Within-current-batch Bonferroni; prior sequential historical experiments not corrected, descriptive retrospective only")
    return out


def agreement(repeats):
    pivot=pd.concat(repeats).pivot(index=["date","code"],columns="repeat_id",values="call")
    pairs=[]
    for i,a in enumerate(pivot.columns):
        for c in pivot.columns[i+1:]:
            x,y=pivot[a],pivot[c];observed=float(x.eq(y).mean())
            expected=sum(float(x.eq(v).mean()*y.eq(v).mean()) for v in ["PROMOTE","FAIL","UNKNOWN"])
            pairs.append(dict(left=int(a),right=int(c),rows=len(x),agreement=observed,kappa=(observed-expected)/(1-expected) if expected<1 else None))
    consensus=[]
    for (day,code),row in pivot.iterrows():
        counts=row.value_counts();best=counts.index[0]
        majority=best if counts.iloc[0]>=2 else "UNKNOWN"
        consensus.append(dict(date=day,code=code,majority_call=majority,unanimous_call=best if counts.iloc[0]==len(row) else "UNKNOWN",unanimous=counts.iloc[0]==len(row),votes=row.to_dict()))
    return pairs,pd.DataFrame(consensus)
