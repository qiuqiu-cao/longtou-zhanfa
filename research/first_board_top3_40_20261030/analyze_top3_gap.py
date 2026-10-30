"""Read-only diagnosis of the user's Top-3 promotion target. No ranking search or fit.

Future outcomes appear only in explicitly diagnostic outputs. Source features are
exported separately; oracle opportunities are NOT a realizable selection rule.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAVED = ROOT / "research/first_board_optimization_20261030/results"
SEAL = ROOT / "research/first_board_joint_design_20261020T185417Z/qualification_resumed_20261021/COMPLETE28503_NEW_FIELDS_NO_LABELS.csv"
PRESSURE = ROOT / "research/first_board_leader_emotion_20261021T1706Z/UNFROZEN_PRESSURE_RESPONSE_FEATURES_WITHOUT_TARGETS.csv"
PROTECTION = ROOT / "research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json"
KEYS = ["signal_date", "code"]
P2 = "promotion_next_LABEL_ONLY"
P4 = "reach4_LABEL_ONLY"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return pd.read_csv(p, dtype={"code": str}, float_precision="round_trip", low_memory=False)


def evaluate_selection(pool, selected_keys, dates, k=3, target=.4):
    """Keep every contracted date/slot; never trust labels supplied with picks.

    This evaluates counts only. It cannot certify as-of timestamps, out-of-sample
    integrity, calibrated probabilities, or investment performance.
    """
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("Expected dates must be nonempty and unique")
    if pool.duplicated(KEYS).any() or selected_keys.duplicated(KEYS).any():
        raise ValueError("Duplicate candidate or selection")
    if pool[KEYS].isna().any().any() or selected_keys[KEYS].isna().any().any():
        raise ValueError("Missing candidate key")
    if set(pool.signal_date) != set(dates):
        raise ValueError("Pool must retain every contracted date and no extra dates")
    if not pool[P2].dropna().isin([0, 1]).all():
        raise ValueError("Promotion outcome must be 0, 1, or unknown")
    picks = selected_keys[KEYS].merge(pool[KEYS + [P2]], on=KEYS,
                                      how="left", validate="one_to_one", indicator=True)
    if not picks._merge.eq("both").all():
        raise ValueError("Selection outside the original qualified pool")
    per_day = picks.groupby("signal_date").size().reindex(dates, fill_value=0)
    if per_day.gt(k).any():
        raise ValueError("More picks than the daily contract")
    slots = k * len(dates)
    hits = int(picks[P2].eq(1).sum())
    known = int(picks[P2].notna().sum())
    unknown = len(picks) - known
    precision = hits / known if known else None
    complete = bool(per_day.eq(k).all())
    return dict(dates=len(dates), slots=slots, selected=len(picks), known=known,
                unknown=unknown, unfilled_slots=slots-len(picks), hits=hits,
                known_precision=precision, confirmed_hits_per_all_slots=hits/slots,
                possible_hits_per_all_slots=(hits+unknown)/slots,
                complete_daily_top3=complete,
                extra_hits_for_known_target=max(0, math.ceil(target*known)-hits),
                extra_hits_for_all_slots_target=max(0, math.ceil(target*slots)-hits),
                observed_target_met=bool(complete and known and precision >= target),
                all_slots_target_confirmed=bool(complete and hits/slots >= target),
                predictive_validation_passed=False)


def top3(d):
    return d.sort_values(["signal_date", "final_rank", "code"]).groupby("signal_date").head(3)


def outcome_counts(d, field=P2):
    return dict(rows=len(d), hits=int(d[field].eq(1).sum()),
                failures=int(d[field].eq(0).sum()), unknown=int(d[field].isna().sum()))


def diagnostic_opportunities(pool):
    """Post-outcome capacity only; retain all dates, failures, and unknowns."""
    rows = []
    for date, g in pool.groupby("signal_date", sort=True):
        selected = g[g.final_rank.le(3)]
        row = dict(signal_date=date, pool_rows=len(g), **{
            f"top3_{key}": val for key, val in outcome_counts(selected).items()})
        row["pool_known_winners"] = int(g[P2].eq(1).sum())
        row["pool_unknown"] = int(g[P2].isna().sum())
        for cap in [5, 10, 1000]:
            label = str(cap) if cap != 1000 else "full_pool"
            q = g[g.final_rank.le(cap)]
            capacity = min(3, int(q[P2].eq(1).sum()))
            row[f"oracle_known_hits_within_{label}_DIAGNOSTIC_ONLY"] = capacity
            row[f"oracle_extra_within_{label}_DIAGNOSTIC_ONLY"] = capacity-row["top3_hits"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [SAVED / f"{name}_FULL_SCORED.csv" for name in ["S1", "HEIGHT", "FIXED_ENGINE_70_30"]]
    inputs += [SEAL, PRESSURE, Path(__file__), PROTECTION]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    s1, height, overlay = map(read, inputs[:3])
    assert len(s1) == 7254 and s1.signal_date.nunique() == 286
    assert not s1.duplicated(KEYS).any()
    dates = sorted(s1.signal_date.unique())
    expected = s1.sort_values(KEYS).reset_index(drop=True)
    for d in [height, overlay]:
        pd.testing.assert_frame_equal(expected[KEYS + [P2, P4]],
                                      d.sort_values(KEYS)[KEYS + [P2, P4]].reset_index(drop=True))
    # Ensure the diagnostic baseline is exactly the archived S1 ordering.
    reranked = s1.sort_values(["signal_date", "s1_score", "code"], ascending=[True, False, True])
    assert reranked.groupby("signal_date").cumcount().add(1).to_list() == reranked.final_rank.to_list()
    baseline = evaluate_selection(s1, top3(s1)[KEYS], dates)
    # Whitelisted archived S-day fields, explicitly isolated from outcomes.
    feature_fields = KEYS + ["name", "s1_score", "height_score", "final_rank", "one_price",
        "turnover", "amount", "float_cap", "position_60", "close_ma5", "close_ma20",
        "close_ma60", "breakout_20", "market_down_limits", "market_failed_rate",
        "previous_first_promotion", "legacy_industry_group_PROXY", "legacy_industry_source_status"]
    features = s1[feature_fields].rename(columns={"final_rank": "saved_s1_rank"})
    seal_fields = KEYS + ["reported_first_seal_trading_minutes", "reported_first_seal_observed_pool_rank",
        "reported_close_seal_log_amount_ratio", "reported_close_seal_log_floatcap_ratio",
        "source_path", "source_sha256", "source_received_at", "source_record_pointer"]
    seals = read(SEAL)[seal_fields]
    features = features.merge(seals, on=KEYS, how="left", validate="one_to_one")
    features["seal_source_status"] = "LATER_RETRIEVED_HISTORICAL_PROVIDER_REPORT_NOT_ORIGINAL_HTTP"
    pressure_fields = KEYS + ["prior_high_shock_exists", "prior_high_shock_age",
        "prelaunch_shock_return", "prelaunch_shock_drawdown", "prelaunch_shock_rebound",
        "prelaunch_shock_return_same_day_rank", "E", "path_unknown_reason"]
    features = features.merge(read(PRESSURE)[pressure_fields], on=KEYS, how="left", validate="one_to_one")
    features["pressure_feature_status"] = "EXISTING_UNFROZEN_HYPOTHESIS_NOT_VALIDATED_FOR_RANKING"
    assert len(features) == 7254 and not any("LABEL" in c for c in features)
    features.to_csv(args.output / "FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv", index=False, encoding="utf-8-sig")
    s1[KEYS + [P2, P4]].to_csv(args.output / "OUTCOMES_DIAGNOSTIC_ONLY.csv", index=False, encoding="utf-8-sig")
    daily = diagnostic_opportunities(s1)
    daily.to_csv(args.output / "DAILY_ERROR_DECOMPOSITION_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    rank_metrics = []
    for rank in [1, 2, 3, 4, 5]:
        q = s1[s1.final_rank.eq(rank)]
        known = int(q[P2].notna().sum())
        rank_metrics.append(dict(rank=rank, **outcome_counts(q), known=known,
                                 known_precision=int(q[P2].eq(1).sum())/known))
    monthly = []
    for month, g in s1.groupby(s1.signal_date.str[:7]):
        monthly.append(dict(month=month, **evaluate_selection(g, top3(g)[KEYS], sorted(g.signal_date.unique()))))
    pd.DataFrame(monthly).to_csv(args.output / "MONTHLY_BASELINE.csv", index=False, encoding="utf-8-sig")
    comparisons = {}
    for name, d in [("HEIGHT", height), ("FIXED_ENGINE_70_30", overlay)]:
        take = top3(d[~d.vetoed])
        a = set(map(tuple, top3(s1)[KEYS].to_numpy()))
        b = set(map(tuple, take[KEYS].to_numpy()))
        outgoing = s1[[tuple(k) in a-b for k in s1[KEYS].to_numpy()]]
        incoming = s1[[tuple(k) in b-a for k in s1[KEYS].to_numpy()]]
        comparisons[name] = dict(metrics=evaluate_selection(s1, take[KEYS], dates),
            outgoing_p2=outcome_counts(outgoing), incoming_p2=outcome_counts(incoming),
            outgoing_p4=outcome_counts(outgoing, P4), incoming_p4=outcome_counts(incoming, P4))
    protection = json.loads(PROTECTION.read_text(encoding="utf-8"))["protected_sha256"]
    changed = [p for p, h in protection.items() if not (ROOT / p).exists() or sha(ROOT / p) != h]
    assert not changed, changed
    assert all(sha(ROOT / p) == h for p, h in hashes.items())
    report = dict(formed_at=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
        target="Daily Top-3 pooled next-official-session closing-limit-up precision >= 40%",
        baseline=baseline, rank_metrics=rank_metrics,
        daily_known_hits_histogram={str(k): int(v) for k, v in daily.top3_hits.value_counts().sort_index().items()},
        oracle_opportunities_DIAGNOSTIC_ONLY={
            c: int(daily[c].sum()) for c in daily if c.startswith("oracle_")},
        existing_evidence_coverage={c: int(features[c].notna().sum()) for c in
            ["reported_first_seal_trading_minutes", "reported_close_seal_log_amount_ratio",
             "prelaunch_shock_return", "prelaunch_shock_return_same_day_rank"]},
        comparisons=comparisons, protected_files_unchanged=len(protection),
        new_fits=0, new_rank_rules=0, production_changed=False, new_forward_validation=False,
        limits=["All 286 days were previously inspected; no new independent validation.",
                "Oracle capacities use future labels and cannot be used in scoring.",
                "Historical provider first seal is not a continuous intraday path or queue.",
                "Static legacy industry is not contemporaneous theme membership.",
                "Pressure response remains an unfrozen, unvalidated hypothesis.",
                "Top-3 promotion precision is not trading win rate or calibrated probability."])
    (args.output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["baseline", "rank_metrics", "oracle_opportunities_DIAGNOSTIC_ONLY", "existing_evidence_coverage", "protected_files_unchanged"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
