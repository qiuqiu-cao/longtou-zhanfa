"""Joint Top3 MILP with sparse S1 residuals and optional monthly robustness.

This is a NEW, untrained repair implementation, not a replay of Gemini's DE fit.
The positive boundary margin is an explicit restriction on feasible rankings;
an optimum is only for that restricted problem. Boundary ties cannot be chosen
using outcomes. Deployment/replay always uses score descending, code ascending.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def solve_joint_top3(f, outcomes, *, l1_bound, max_active_features, boundary_margin,
                     robust_lambda, time_limit, mip_rel_gap=0., feature_columns):
    """Fit ONLY the supplied mature training pool. No test frame is accepted.

    A: maximize confirmed Top3 hits (robust_lambda=0).
    B: maximize mean monthly confirmed-hit-rate gain over S1, less lambda times
    squared negative monthly gains. Denominators are all 3*days slots, so unknown
    labels cannot boost the objective by shrinking a denominator. Unknowns remain
    unknown; their contribution is a lower-bound reward of zero, not a failure.
    Squared penalties are represented exactly at integer hit counts by a convex
    piecewise-linear epigraph. Binary activation caps enforce actual sparsity.
    """
    keys = ["signal_date", "code"]
    cols = list(feature_columns)
    p = len(cols)
    if (not 0 < l1_bound or not 1 <= max_active_features <= p or
            int(max_active_features) != max_active_features or
            boundary_margin < 1e-5 or robust_lambda < 0 or time_limit <= 0 or
            not 0 <= mip_rel_gap < 1):
        raise ValueError("Invalid frozen solver settings")
    if not np.isfinite([l1_bound, boundary_margin, robust_lambda, time_limit, mip_rel_gap]).all():
        raise ValueError("Non-finite solver settings")
    if f.duplicated(keys).any() or outcomes.duplicated(keys).any():
        raise ValueError("Duplicate candidate keys")
    d = f[keys + ["s1_score"] + cols].merge(outcomes[keys + ["promotion_next_LABEL_ONLY"]],
        on=keys, how="outer", validate="one_to_one", indicator=True)
    if not d.pop("_merge").eq("both").all():
        raise ValueError("Training features and outcomes differ")
    d = d.sort_values(keys).reset_index(drop=True)
    y = d.promotion_next_LABEL_ONLY
    if not y.dropna().isin([0, 1]).all():
        raise ValueError("Invalid training labels")
    reward = y.eq(1).to_numpy(dtype=float)
    x = d[cols].to_numpy(dtype=float)
    base = d.s1_score.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(base).all():
        raise ValueError("Non-finite training features")
    days = list(d.groupby("signal_date", sort=True).indices.values())
    if not days or any(len(g) < 3 for g in days):
        raise ValueError("Every training day needs its complete Top3 pool")
    months = list(d.groupby(d.signal_date.str[:7], sort=True).indices.values())
    n, nd, nm = len(d), len(days), len(months)
    # Variables: theta[p], abs(theta)[p], active[p], selected[n], threshold[days], penalty[months].
    u, active, b, t, v = p, 2*p, 3*p, 3*p+n, 3*p+n+nd
    nv = v + nm
    lower = np.zeros(nv); upper = np.full(nv, np.inf)
    lower[:p] = -l1_bound; upper[:2*p] = l1_bound
    upper[active:b] = 1; upper[b:t] = 1
    integrality = np.zeros(nv, dtype=int)
    integrality[active:t] = 1
    c = np.zeros(nv)
    rows, columns, values, lo, hi = [], [], [], [], []

    def constraint(entries, low=-np.inf, high=np.inf):
        row = len(lo)
        for col, value in entries.items():
            if value:
                rows.append(row); columns.append(col); values.append(value)
        lo.append(low); hi.append(high)

    for j in range(p):
        constraint({j: 1, u+j: -1}, high=0)
        constraint({j: -1, u+j: -1}, high=0)
        constraint({j: 1, active+j: -l1_bound}, high=0)
        constraint({j: -1, active+j: -l1_bound}, high=0)
    constraint({u+j: 1 for j in range(p)}, high=l1_bound)
    constraint({active+j: 1 for j in range(p)}, high=max_active_features)
    baseline_selected = np.zeros(n)
    for day, ix in enumerate(days):
        # Safe Big-M from the full feasible score range; no hand-picked constant.
        correction = l1_bound * np.max(np.abs(x[ix]))
        low_score = float(base[ix].min() - correction)
        high_score = float(base[ix].max() + correction)
        lower[t+day] = low_score - boundary_margin
        upper[t+day] = high_score + boundary_margin
        big_m = high_score-low_score+2*boundary_margin
        constraint({b+int(i): 1 for i in ix}, low=3, high=3)
        for i in ix:
            e = {j: float(x[i, j]) for j in range(p)}
            e.update({t+day: -1, b+int(i): -big_m})
            constraint(e, low=-big_m-base[i]+boundary_margin/2,
                       high=-base[i]-boundary_margin/2)
        # d is code sorted, so lexsort's index tie-break is code ascending.
        chosen = ix[np.lexsort((ix, -base[ix]))[:3]]
        baseline_selected[chosen] = 1
    baseline_monthly = []
    for month, ix in enumerate(months):
        slots = 3 * d.iloc[ix].signal_date.nunique()
        base_hits = int(reward[ix] @ baseline_selected[ix])
        baseline_monthly.append(dict(month=d.iloc[ix[0]].signal_date[:7],
                                     slots=slots, baseline_hits=base_hits))
        if robust_lambda:
            c[b+ix] = -reward[ix]/(nm*slots)
            c[v+month] = robust_lambda
            # At integer h, max of adjacent secants equals max(0,(base_hits-h)/slots)^2.
            for h in range(base_hits):
                f0 = ((base_hits-h)/slots)**2
                slope = ((base_hits-h-1)/slots)**2-f0
                e = {v+month: 1}
                e.update({b+int(i): -slope*reward[i] for i in ix if reward[i]})
                constraint(e, low=f0-slope*h)
        else:
            c[b+ix] = -reward[ix]
            upper[v+month] = 0
    matrix = coo_matrix((values, (rows, columns)), shape=(len(lo), nv)).tocsc()
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, np.array(lo), np.array(hi)),
                  options=dict(time_limit=float(time_limit), mip_rel_gap=float(mip_rel_gap)))
    metadata = dict(status=int(result.status), message=str(result.message),
        solver_reported_success=bool(result.success),
        mip_gap=None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        mip_dual_bound=None if getattr(result, "mip_dual_bound", None) is None else float(result.mip_dual_bound),
        minimized_objective=None if result.fun is None else float(result.fun),
        objective_drops_constant_baseline=bool(robust_lambda),
        optimality_scope="MARGIN_CONSTRAINED_TRAINING_PROBLEM_ONLY",
        proven_optimal=bool(result.status == 0 and getattr(result, "mip_gap", None) == 0),
        predictive_validation_passed=False, training_rows=n, training_dates=nd,
        unknown_training_outcomes=int(y.isna().sum()), monthly_baseline=baseline_monthly)
    if result.x is None:
        return dict(metadata=metadata, theta=None, selections=None)
    theta = result.x[:p]
    scores = base + x @ theta
    if np.abs(theta).sum() > l1_bound+1e-7 or np.count_nonzero(np.abs(theta)>1e-7) > max_active_features:
        raise RuntimeError("Solver coefficient constraints failed replay")
    take = []
    gaps = []
    for ix in days:
        order = ix[np.lexsort((ix, -scores[ix]))]
        chosen = order[:3]
        declared = ix[result.x[b+ix] > .5]
        if set(chosen) != set(declared):
            raise RuntimeError("MILP membership does not match actual deterministic Top3")
        if len(ix) > 3:
            gap = float(scores[order[2]]-scores[order[3]])
            gaps.append(gap)
            if gap < boundary_margin-1e-7:
                raise RuntimeError("MILP boundary margin failed replay")
        for slot, i in enumerate(chosen, 1):
            take.append(dict(signal_date=d.iloc[i].signal_date, code=d.iloc[i].code,
                             slot=slot, score=float(scores[i])))
    metadata["verified_against_actual_score_ranking"] = True
    metadata["minimum_training_boundary_gap"] = min(gaps) if gaps else None
    metadata["l1_norm"] = float(np.abs(theta).sum())
    metadata["active_coefficients_at_1e_minus7"] = int(np.count_nonzero(np.abs(theta)>1e-7))
    return dict(metadata=metadata, theta=theta.tolist(), selections=pd.DataFrame(take))
