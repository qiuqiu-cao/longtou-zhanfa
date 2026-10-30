import importlib.util
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/"research/first_board_math_20261030/math_frontier.py"
SPEC=importlib.util.spec_from_file_location("math_frontier",PATH)
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def test_crossing_point_tie_can_differ_from_both_neighbor_sets():
    # All three lines cross at .5; code order picks the flat middle line at
    # exactly .5, which is absent from the optimum on either side.
    p,a,b=M.exact_day_states([1.,0.,.5],[0.,1.,.5],["B","C","A"],k=1)
    assert a[p.index(M.Fraction(1,2))]==(2,)
    assert b==[(1,),(0,)]


def test_enumeration_matches_direct_ranking_on_many_interior_weights():
    rng=np.random.default_rng(1)
    s=rng.random(8); h=rng.random(8); codes=[str(i) for i in range(8)]
    points,at,between=M.exact_day_states(s,h,codes)
    for w in np.linspace(0,1,301):
        expected=tuple(sorted(np.lexsort((codes,-(w*s+(1-w)*h)))[:3]))
        if w==0: actual=at[0]
        elif w==1: actual=at[-1]
        else:
            i=next(i for i,(a,b) in enumerate(zip(points[:-1],points[1:])) if float(a)<w<float(b))
            actual=between[i]
        assert actual==expected


def test_identical_scores_use_codes_without_outcomes():
    _,at,between=M.exact_day_states([.2]*4,[.8]*4,["D","A","C","B"])
    assert all(s==(1,2,3) for s in at+between)


def test_structured_loss_matches_bruteforce_and_upper_bounds_regret():
    rng=np.random.default_rng(42)
    for _ in range(100):
        s=rng.normal(size=7); y=rng.integers(0,2,size=7); k=3
        sets=list(combinations(range(7),k)); q=min(k,int(y.sum()))
        augmented=max(sum(s[list(a)])+q-sum(y[list(a)]) for a in sets)
        correct=max(sum(s[list(a)]) for a in sets if sum(y[list(a)])==q)
        loss=M.structured_topk_regret_loss(s,y)
        regret=q-int(y[np.argsort(s)[-k:]].sum())
        assert loss==pytest.approx(augmented-correct)
        assert loss+1e-12>=regret


def test_unknown_labels_are_not_converted_to_negative_training_targets():
    with pytest.raises(ValueError):
        M.structured_topk_regret_loss([.1,.2,.3],[0,1,np.nan])


def test_fractional_daily_ceiling_matches_exhaustive_choices():
    sys.modules["math_frontier"]=M
    spec=importlib.util.spec_from_file_location("fractional_ceiling",PATH.with_name("daily_fractional_ceiling.py"))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    options=[[(1,3),(1,2)],[(0,0),(1,3)],[(1,2),(2,3)]]
    h,n,_=mod.best_fraction(options)
    expected=max(M.Fraction(sum(v[0] for v in choice),sum(v[1] for v in choice)) for choice in product(*options))
    assert M.Fraction(h,n)==expected
