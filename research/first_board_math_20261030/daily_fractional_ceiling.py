"""Exact hindsight ceiling even if each day chooses its own fusion weight.

Fractional optimization over finite per-day attainable outcome counts. Not a
live selector: the objective explicitly uses future outcomes.
"""
from fractions import Fraction
import json
from pathlib import Path
import pandas as pd
from math_frontier import exact_day_states, SOURCE, P2


def best_fraction(options):
    ratio=Fraction(0)
    for iteration in range(100):
        chosen=[max(day,key=lambda z:(z[0]*ratio.denominator-ratio.numerator*z[1],z[1],z[0])) for day in options]
        hits=sum(z[0] for z in chosen); known=sum(z[1] for z in chosen)
        if not known:
            raise ValueError("No observed outcomes")
        new=Fraction(hits,known)
        if new==ratio:
            return hits,known,iteration+1
        ratio=new
    raise AssertionError("Finite fractional program did not converge")


def main():
    d=pd.read_csv(SOURCE,dtype={"code":str},float_precision="round_trip")
    options=[]
    for _,g in d.groupby("signal_date",sort=True):
        _,at,between=exact_day_states(g.s1_score.to_numpy(),g.height_score.to_numpy(),g.code.to_list())
        metrics=set()
        for selected in set(at+between):
            y=g.iloc[list(selected)][P2]
            metrics.add((int(y.eq(1).sum()),int(y.notna().sum())))
        options.append(sorted(metrics))
    h,n,iterations=best_fraction(options)
    result=dict(hindsight_daily_weight_maximum_known_precision=h/n,hits=h,known=n,
        unknown=3*len(options)-n,dates=len(options),slots=3*len(options),iterations=iterations,
        exact_ratio=str(Fraction(h,n)),reaches40=Fraction(h,n)>=Fraction(2,5),
        max_known_hit_count=sum(max(v[0] for v in day) for day in options),
        max_hits_plus_unknown_count=sum(max(v[0]+3-v[1] for v in day) for day in options),
        oracle_only=True,deployable=False,
        scope="Every independent daily nonnegative fusion weight of two fixed raw scores; not all nonlinear rules or original features")
    p=Path(__file__).resolve().parent/"results/DAILY_FRACTIONAL_CEILING_ORACLE_ONLY.json"
    with p.open("x",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
