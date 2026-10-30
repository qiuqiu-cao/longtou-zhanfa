"""Exact finite-data ceiling of every convex mixture of two saved scores.

Uses rational breakpoints and integer scores, including ties at all boundaries.
The maximum is intentionally hindsight/optimistic, NOT a proposed live weight.
No classifier fit, feature search, production change, or independent-test claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "research/first_board_optimization_20261030/results/S1_FULL_SCORED.csv"
S1_SOURCE = ROOT / "research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv"
PROTECTION = ROOT / "research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json"
P2 = "promotion_next_LABEL_ONLY"
P4 = "reach4_LABEL_ONLY"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exact_day_states(s1, height, codes, k=3):
    """Enumerate all Top-k sets of h + w*(s-h), w in [0,1].

    Binary64 source values are treated as exact rationals. A common power-of-two
    denominator makes every point and interval ordering an integer comparison.
    Code ascending is the outcome-independent tie rule.
    """
    s1, height = np.asarray(s1, float), np.asarray(height, float)
    if not np.isfinite(s1).all() or not np.isfinite(height).all():
        raise ValueError("Missing/nonfinite score")
    if len(s1) != len(height) or len(s1) != len(codes) or len(s1) < k:
        raise ValueError("Incompatible arrays or insufficient daily pool")
    pairs = [v.as_integer_ratio() for v in np.r_[height, s1]]
    denominator = max(d for _, d in pairs)
    ints = [n*(denominator//d) for n, d in pairs]
    intercept = ints[:len(s1)]
    slope = [s-h for s,h in zip(ints[len(s1):], intercept)]
    points = {Fraction(0), Fraction(1)}
    for i,j in combinations(range(len(s1)), 2):
        delta = slope[i]-slope[j]
        if delta:
            w = Fraction(intercept[j]-intercept[i], delta)
            if 0 < w < 1:
                points.add(w)
    points = sorted(points)

    def selected(w):
        p,q = w.numerator,w.denominator
        score = [a*q+b*p for a,b in zip(intercept,slope)]
        order = sorted(range(len(score)), key=lambda i: (-score[i],codes[i]))
        return tuple(sorted(order[:k]))

    at_points = [selected(w) for w in points]
    between = [selected((a+b)/2) for a,b in zip(points[:-1],points[1:])]
    return points, at_points, between


def structured_topk_regret_loss(scores, labels, k=3):
    """Loss upper-bounding the Top-k missed-hit count on fully known labels.

    L = max_|S|=k [sum(s_i) + q - sum(y_i)]
        - max_|S|=k,sum(y_i)=q sum(s_i), q=min(k, number of positives).

    Exact O(n log n) latent structured loss. Difference of convex functions;
    NOT claimed globally convex, calibrated, or empirically superior.
    """
    s,y = np.asarray(scores,float),np.asarray(labels,float)
    if len(s)!=len(y) or len(s)<k or not np.isfinite(s).all() or not np.isin(y,[0,1]).all():
        raise ValueError("Loss needs finite scores and fully observed binary labels")
    q=min(k,int(y.sum()))
    augmented=float(np.sort(s-y)[-k:].sum()+q)
    pos=np.sort(s[y==1]); neg=np.sort(s[y==0])
    ideal=float(pos[-q:].sum()) if q else 0.
    if k-q:
        ideal+=float(neg[-(k-q):].sum())
    return max(0.,augmented-ideal)


def metric(selected, d):
    q=d.iloc[list(selected)]
    h=int(q[P2].eq(1).sum()); n=int(q[P2].notna().sum())
    h4=int(q[P4].eq(1).sum()); n4=int(q[P4].notna().sum())
    return np.array([h,n,h4,n4],dtype=int)


def block_bootstrap(d, repetitions=5000, block_dates=25):
    q=d[d.final_rank.le(3)].copy()
    a=q.groupby("signal_date")[P2].agg(hits=lambda x:x.eq(1).sum(),known="count").to_numpy()
    n=len(a); rng=np.random.default_rng(20261030)
    # Moving contiguous 25-session blocks, truncated to original length.
    starts=rng.integers(0,n-block_dates+1,size=(repetitions,int(np.ceil(n/block_dates))))
    indices=(starts[:,:,None]+np.arange(block_dates)).reshape(repetitions,-1)[:,:n]
    sums=a[indices].sum(axis=1)
    rates=sums[:,0]/sums[:,1]
    return dict(repetitions=repetitions,block_dates=block_dates,
                interval95=[float(x) for x in np.quantile(rates,[.025,.975])],
                source="Already inspected history; descriptive temporal resampling, not independent validation")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    args=parser.parse_args(); out=args.output
    out.mkdir(parents=True,exist_ok=False)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,S1_SOURCE,Path(__file__),PROTECTION]}
    protocol=dict(formed_at=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
        question="Can any fixed nonnegative linear fusion of the two frozen raw scores reach 40% Top3 on these 286 days?",
        method="All exact rational crossing points, endpoints and open intervals; code-ascending ties",
        use="Oracle capacity diagnosis only. Never deploy the hindsight maximizing weight.",
        fits=0,constant_weight_family="w*S1+(1-w)*height for every real w in [0,1]",
        new_training_or_joint_freeze=False,unknowns="Retain and report both known precision and full-slot bounds")
    (out/"PROTOCOL_BEFORE_MATH.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    d=pd.read_csv(SOURCE,dtype={"code":str},float_precision="round_trip")
    assert len(d)==7254 and d.signal_date.nunique()==286 and not d.duplicated(["signal_date","code"]).any()
    assert d.groupby("signal_date").size().min()>=3
    assert d[P2].dropna().isin([0,1]).all()
    n_dates=d.signal_date.nunique(); slots=3*n_dates
    current=set(); at_zero=set(); at_one=set(); events={}; daily=[]; all_crossings=0
    for day,g in d.groupby("signal_date",sort=True):
        points, at, between=exact_day_states(g.s1_score.to_numpy(),g.height_score.to_numpy(),g.code.to_list())
        ids=g.index.to_numpy()
        translated=lambda s: tuple(int(ids[i]) for i in s)
        at=[translated(s) for s in at]; between=[translated(s) for s in between]
        at_zero.update(at[0]); at_one.update(at[-1]); current.update(between[0])
        all_crossings+=len(points)-2
        possible=set(at+between)
        counts=[metric(s,d) for s in possible]
        daily.append(dict(signal_date=day,pool_rows=len(g),all_pair_crossings=len(points)-2,
            realizable_top3_sets=len(possible),
            oracle_daily_varying_weight_known_hits_LABEL_ONLY=max(int(m[0]) for m in counts),
            oracle_daily_varying_weight_hits_plus_unknown_LABEL_ONLY=max(int(m[0]+3-m[1]) for m in counts)))
        for i,w in enumerate(points[1:-1],start=1):
            left,right,point=between[i-1],between[i],at[i]
            if left != right or point != left:
                events.setdefault(w,[]).append((left,point,right))
    rows=[]
    def row(a,b,state,m):
        return dict(weight_left=float(a),weight_right=float(b),
            weight_left_exact=str(a),weight_right_exact=str(b),state=state,
            hits=int(m[0]),known=int(m[1]),unknown=slots-int(m[1]),
            precision_known=float(m[0]/m[1]),confirmed_hits_per_all_slots=float(m[0]/slots),
            possible_hits_per_all_slots=float((m[0]+slots-m[1])/slots),
            reach4_hits=int(m[2]),reach4_known=int(m[3]))
    rows.append(row(Fraction(0),Fraction(0),"POINT",metric(at_zero,d)))
    state=metric(current,d); previous=Fraction(0)
    for w, transitions in sorted(events.items()):
        rows.append(row(previous,w,"OPEN_INTERVAL",state))
        point_state=state.copy(); right_state=state.copy()
        for left,point,right in transitions:
            left_metric=metric(left,d)
            point_state+=metric(point,d)-left_metric
            right_state+=metric(right,d)-left_metric
        rows.append(row(w,w,"POINT",point_state))
        state=right_state; previous=w
    rows.append(row(previous,Fraction(1),"OPEN_INTERVAL",state))
    rows.append(row(Fraction(1),Fraction(1),"POINT",metric(at_one,d)))
    frontier=pd.DataFrame(rows)
    frontier.to_csv(out/"ALL_FIXED_WEIGHT_STATES_ORACLE_ONLY.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(daily).to_csv(out/"DAILY_WEIGHT_CAPACITY_ORACLE_ONLY.csv",index=False,encoding="utf-8-sig")
    best=frontier.iloc[int(frontier.precision_known.to_numpy().argmax())].to_dict()
    max_hits=int(frontier.hits.max()); max_possible=float(frontier.possible_hits_per_all_slots.max())
    # Independent replay of representative ordinary weights, not candidate search.
    checks=[]
    for w in [0.,.1,.5,.6,.7,.9,1.]:
        picks=d.assign(score=w*d.s1_score+(1-w)*d.height_score).sort_values(
            ["signal_date","score","code"],ascending=[True,False,True]).groupby("signal_date").head(3)
        match=frontier[(frontier.state.eq("POINT") & frontier.weight_left.eq(w)) |
                       (frontier.state.eq("OPEN_INTERVAL") & frontier.weight_left.lt(w) & frontier.weight_right.gt(w))]
        assert len(match)==1
        expected=metric(picks.index,d)
        actual=match.iloc[0][["hits","known","reach4_hits","reach4_known"]].to_numpy(dtype=int)
        assert np.array_equal(actual,expected),(w,actual,expected)
        checks.append(dict(weight=w,hits=int(expected[0]),known=int(expected[1])))
    sp=pd.read_csv(S1_SOURCE,dtype={"code":str},float_precision="round_trip")
    z=d.merge(sp[["signal_date","code","calibrated_score","train_prior"]],on=["signal_date","code"],validate="one_to_one")
    known=z[P2].notna(); discrimination={}
    for score in ["s1_score","height_score","calibrated_score"]:
        q=z[known & z[score].notna()]
        discrimination[score]=dict(known=len(q),global_auc=float(roc_auc_score(q[P2],q[score])))
        if q[score].between(0,1).all():
            discrimination[score]["brier_diagnostic"]=float(brier_score_loss(q[P2],q[score]))
    day_auc=[]
    random_mean=0.; random_variance=0.
    for day,g in z.groupby("signal_date",sort=True):
        q=g[g[P2].notna()]; N=len(q); M=int(q[P2].sum())
        selected_known=int(g[g.final_rank.le(3)][P2].notna().sum())
        if N:
            random_mean+=selected_known*M/N
            if N>1:
                random_variance+=selected_known*M/N*(1-M/N)*(N-selected_known)/(N-1)
        if 0<M<N:
            day_auc.append(dict(signal_date=day,positive=M,negative=N-M,
                                s1_auc=float(roc_auc_score(q[P2],q.s1_score)),
                                height_auc=float(roc_auc_score(q[P2],q.height_score))))
    pd.DataFrame(day_auc).to_csv(out/"WITHIN_DATE_DISCRIMINATION.csv",index=False,encoding="utf-8-sig")
    # Geometric ceiling: a candidate dominated by >=3 others cannot enter Top3
    # under ANY strictly increasing function of both scores. This is an upper
    # bound, not a sufficient condition for realizability with one global function.
    geometry=[]
    for day,g in d.groupby("signal_date",sort=True):
        x=g[["s1_score","height_score"]].to_numpy()
        dominates=(x[None,:,:]>=x[:,None,:]).all(2)&(x[None,:,:]>x[:,None,:]).any(2)
        count=dominates.sum(1); eligible=g[count<3]
        geometry.append(dict(signal_date=day,not_dominated_by_three=len(eligible),
            optimistic_monotone_two_score_known_hits_LABEL_ONLY=min(3,int(eligible[P2].eq(1).sum()))))
    pd.DataFrame(geometry).to_csv(out/"TWO_SCORE_DOMINANCE_CAPACITY_ORACLE_ONLY.csv",index=False,encoding="utf-8-sig")
    protection=json.loads(PROTECTION.read_text(encoding="utf-8"))["protected_sha256"]
    changed=[p for p,h in protection.items() if sha(ROOT/p)!=h]
    assert not changed,changed
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    report=dict(formed_at=datetime.now(timezone.utc).isoformat(),rows=len(d),dates=int(n_dates),slots=int(slots),
        source_hashes=hashes,all_within_day_pair_crossings=all_crossings,
        distinct_global_top3_change_weights=len(events),point_and_interval_states=len(frontier),
        exact_arithmetic="Integer coefficients and rational crossing/midpoints; every boundary included; code-ascending ties",
        hindsight_best_constant_mixture=best,
        maximum_known_hits_with_constant_mixture=max_hits,
        maximum_unknown_optimistic_precision_all_slots=max_possible,
        any_constant_mixture_reaches_40=bool(frontier.precision_known.ge(.4).any()),
        oracle_daily_varying_weight_known_hits=sum(r["oracle_daily_varying_weight_known_hits_LABEL_ONLY"] for r in daily),
        optimistic_monotone_two_score_known_hit_capacity=sum(r["optimistic_monotone_two_score_known_hits_LABEL_ONLY"] for r in geometry),
        representative_replay_checks=checks,discrimination=discrimination,
        mean_within_date_s1_auc=float(np.mean([r["s1_auc"] for r in day_auc])),
        mean_within_date_height_auc=float(np.mean([r["height_auc"] for r in day_auc])),
        s1_top3_descriptive_block_interval=block_bootstrap(d),
        conditional_randomization_reference=dict(expected_known_hits=random_mean,
            sd_if_independent_days=float(np.sqrt(random_variance)),
            definition="Shuffle only known labels within each day; unknown positions fixed; not a tradable benchmark or serial-dependence-adjusted p-value"),
        new_model_fits=0,live_weight_selected=False,production_changed=False,
        independent_validation=False,protected_files_unchanged=len(protection))
    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["source_hashes"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
