"""Causal, lagged price co-movement within the archived attempt universe.

This is a statistical relation proxy. It cannot identify an industry, an active
trading narrative, a funds transfer, or an independently validated parent.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

VERSION = "lagged-price-relation-1"
POLICY = dict(window_sessions=20, excluded_recent_prior_sessions=5,
    minimum_common_sessions=12, return_definition="close/pre_close-1",
    market_component="subtract same-date median of available archived stock returns",
    pair_weight="square of positive Pearson correlation; no selected threshold",
    source_scope="ARCHIVED_CURRENT_UNIVERSE_WITH_SURVIVORSHIP_AND_PROVIDER_LIMITATIONS",
    relation_semantics="STATISTICAL_PRICE_COHORT_NOT_BUSINESS_THEME_OR_FUNDS_CAUSALITY",
    future_labels_allowed=False, missing_scope_is_zero=False,
    available_mass="sum of weights of observed pairs, with separate observation fraction",
    parent_selection="all prior active high anchors from causal engine; no future outcome or name selection")
FIELDS = ["price_relation_history_fraction", "price_relation_closing_pair_fraction",
    "price_relation_failed_pair_fraction", "price_relation_first_observed_mass",
    "price_relation_relay_observed_mass", "price_relation_higher_observed_mass",
    "price_relation_same_height_mean_corr", "price_relation_failed_observed_mass",
    "price_relation_parent_max_corr", "price_relation_selected_parent_corr",
    "price_relation_parent_pair_fraction", "price_relation_net_observed_mass"]


def _parents(value):
    if isinstance(value, str):
        value = json.loads(value)
    return [str(x).zfill(6) for x in (value if isinstance(value, (list, tuple)) else [])]


def pairwise_complete_correlation(values, minimum=12):
    """Pairwise-complete Pearson; no imputation and no correlation for constants."""
    x = np.asarray(values, dtype=float)
    observed = np.isfinite(x).astype(float)
    v = np.where(np.isfinite(x), x, 0.)
    count = observed.T @ observed
    sums = v.T @ observed
    squares = (v * v).T @ observed
    cross = v.T @ v
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = cross - sums * sums.T / count
        variance = squares - sums * sums / count
        denominator = np.sqrt(np.maximum(variance, 0) * np.maximum(variance.T, 0))
        result = covariance / denominator
    result[(count < minimum) | (denominator <= 1e-18)] = np.nan
    return np.clip(result, -1., 1.), count.astype(int)


def build_causal(candidates, attempts, bars, sessions, complete_dates):
    """Shared historical/live function; the only candidate metadata is causal.

    T uses returns on official sessions T-25..T-6. The five most recent prior
    sessions and T itself are excluded from relation estimation. Current T
    closing/failed membership is known at signal close and is allowed.
    """
    sessions = list(sessions)
    if sessions != sorted(set(sessions)):
        raise ValueError("official sessions must be sorted and unique")
    required = ["date", "code", "name", "boards", "parent_candidates", "parent_anchor_code"]
    c = candidates[required].copy()
    a = attempts[["signal_date", "code", "board_height", "pool_assignment"]].copy()
    prices = bars[["date", "code", "close", "pre_close"]].copy()
    for frame in (c, a, prices):
        frame["code"] = frame.code.astype(str).str.zfill(6)
    if c.duplicated(["date", "code"]).any() or prices.duplicated(["date", "code"]).any():
        raise ValueError("duplicate candidate or dated daily price")
    if a.duplicated(["signal_date", "code"]).any():
        raise ValueError("ambiguous closing/failed attempt identity")
    if not set(a.pool_assignment).issubset({"CLOSING_LIMIT_UP", "FAILED_BOARD_OBSERVATION"}):
        raise ValueError("unqualified source attempt assignment")
    index = {d: i for i, d in enumerate(sessions)}
    if any(day not in index for day in c.date):
        raise ValueError("candidate is outside official calendar")
    maximum = c.date.max()
    prices = prices[prices.date.le(maximum) & prices.date.isin(index)].copy()
    for field in ("close", "pre_close"):
        prices[field] = pd.to_numeric(prices[field], errors="coerce")
    good = prices.close.gt(0) & prices.pre_close.gt(0)
    prices["ret"] = np.where(good, prices.close / prices.pre_close - 1., np.nan)
    panel = prices.pivot(index="date", columns="code", values="ret").reindex(sessions)
    benchmark_members = panel.notna().sum(axis=1)
    # Availability is reported, never represented as an exchange-complete index.
    panel = panel.sub(panel.median(axis=1), axis=0)
    lookup = {code: i for i, code in enumerate(panel.columns)}
    data = panel.to_numpy(dtype=float)
    by_attempt = {d: f for d, f in a[a.signal_date.le(maximum)].groupby("signal_date")}
    complete = set(complete_dates)
    rows, pair_records = [], []
    with threadpool_limits(limits=1):
        for day, part in c.groupby("date", sort=True):
            t = index[day]
            end = max(0, t - POLICY["excluded_recent_prior_sessions"])
            start = max(0, end - POLICY["window_sessions"])
            days = sessions[start:end]
            source = by_attempt.get(day, a.iloc[0:0])
            closing = {r.code: int(r.board_height) for r in source.itertuples()
                if r.pool_assignment == "CLOSING_LIMIT_UP"}
            failed = {r.code: int(r.board_height) for r in source.itertuples()
                if r.pool_assignment == "FAILED_BOARD_OBSERVATION"}
            augmented = set(part.code) - set(closing)
            # Original candidates survive even if the attempt source is incomplete.
            closing.update({r.code: int(r.boards) for r in part.itertuples()})
            universe = set(closing) | set(failed) | set(part.code)
            for row in part.itertuples():
                universe.update(_parents(row.parent_candidates))
                if pd.notna(row.parent_anchor_code):
                    universe.add(str(row.parent_anchor_code).zfill(6))
            codes = sorted(universe)
            local = {code: i for i, code in enumerate(codes)}
            values = np.full((len(days), len(codes)), np.nan)
            for code, j in local.items():
                if code in lookup:
                    values[:, j] = data[start:end, lookup[code]]
            corr, common = pairwise_complete_correlation(values, POLICY["minimum_common_sessions"])
            for row in part.itertuples():
                i = local[row.code]
                own_observed = int(np.isfinite(values[:, i]).sum())
                own_valid = own_observed >= POLICY["minimum_common_sessions"]
                close_codes = [code for code in closing if code != row.code]
                fail_codes = [code for code in failed if code != row.code]
                parent_codes = sorted(set(_parents(row.parent_candidates)) - {row.code})
                def correlations(members):
                    return np.array([corr[i, local[code]] for code in members], dtype=float)
                def fraction(members):
                    if not own_valid:
                        return np.nan
                    return float(np.isfinite(correlations(members)).mean()) if members else 1.
                def mass(members):
                    if not own_valid:
                        return np.nan
                    if not members:
                        return 0.
                    v = correlations(members)
                    if not np.isfinite(v).any():
                        return np.nan
                    return float(np.nansum(np.maximum(v, 0.) ** 2))
                scope = day in complete
                same = [code for code in close_codes if closing[code] == int(row.boards)]
                same_values = correlations(same)
                parent_values = correlations(parent_codes)
                selected = str(row.parent_anchor_code).zfill(6) if pd.notna(row.parent_anchor_code) else None
                closing_mass, failed_mass = mass(close_codes), mass(fail_codes)
                total = closing_mass + failed_mass
                result = dict(date=day, code=row.code, name=row.name, boards=int(row.boards),
                    relation_status=POLICY["relation_semantics"], estimate_start=days[0] if days else None,
                    estimate_end=days[-1] if days else None, signal_cutoff=day,
                    minimum_benchmark_members=int(benchmark_members.loc[days].min()) if days else 0,
                    scope_complete=scope, candidate_augmented_count=len(augmented),
                    closing_peers=len(close_codes), failed_peers=len(fail_codes), parent_peers=len(parent_codes))
                result.update(dict(zip(FIELDS, [own_observed / POLICY["window_sessions"],
                    fraction(close_codes) if scope else np.nan,
                    fraction(fail_codes) if scope else np.nan,
                    mass([code for code in close_codes if closing[code] == 1]) if scope else np.nan,
                    mass([code for code in close_codes if closing[code] >= 2]) if scope else np.nan,
                    mass([code for code in close_codes if closing[code] > row.boards]) if scope else np.nan,
                    float(np.nanmean(same_values)) if scope and np.isfinite(same_values).any() else np.nan,
                    failed_mass if scope else np.nan,
                    float(np.nanmax(parent_values)) if np.isfinite(parent_values).any() else np.nan,
                    float(corr[i, local[selected]]) if selected in local and selected != row.code else np.nan,
                    fraction(parent_codes),
                    float((closing_mass - failed_mass) / total) if scope and np.isfinite(total) and total > 0 else np.nan])))
                rows.append(result)
                # Every pool and parent pair is retained, including missing/negative.
                for code in sorted(set(close_codes) | set(fail_codes) | set(parent_codes)):
                    pair_records.append(dict(date=day, code=row.code, peer_code=code,
                        peer_pool="FAILED" if code in failed else "CLOSING" if code in closing else "PRIOR_PARENT_ONLY",
                        peer_height=failed.get(code, closing.get(code)), is_active_parent=code in parent_codes,
                        is_selected_parent=code == selected, common_sessions=int(common[i, local[code]]),
                        correlation=float(corr[i, local[code]]), estimate_start=result["estimate_start"], estimate_end=result["estimate_end"]))
    return pd.DataFrame(rows), pd.DataFrame(pair_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    import joblib
    import short_term_thinking_dataset_builder as b
    source = Path(args.bundle).resolve()
    bundle = joblib.load(source)
    b.verify(bundle["sources"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dates = bundle["attempts"].attrs.get("failed_scope_complete_dates")
    if dates is None:
        raise ValueError("explicit complete source dates required")
    rows, pairs = build_causal(bundle["features"], bundle["attempts"], bundle["bars"], bundle["sessions"], dates)
    rows.to_csv(output / "complete_price_relation_NO_LABELS.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "all_pool_and_parent_pairs_NO_LABELS.csv", index=False, encoding="utf-8-sig")
    joblib.dump(dict(version=VERSION, policy=POLICY, rows=rows, pairs=pairs,
        sources=bundle["sources"] | {str(source): b.sha(source), str(Path(__file__).resolve()): b.sha(__file__)},
        engine_path=str(source)), output / "price_relation_context.joblib")
    b.write_new(output / "construction_receipt.json", dict(formed_at=b.now(), version=VERSION,
        candidates=len(rows), pairs=len(pairs), fields=FIELDS, policy=POLICY,
        labels_read_by_builder=False, new_fits=0,
        source_bindings={str(source): b.sha(source), str(Path(__file__).resolve()): b.sha(__file__)}))
    print(dict(rows=len(rows), pairs=len(pairs), fits=0))


if __name__ == "__main__":
    main()
