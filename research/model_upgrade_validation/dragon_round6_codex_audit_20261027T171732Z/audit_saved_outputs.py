"""Recompute saved Gemini R6 evidence. Never fit estimators or change source files."""
from pathlib import Path
from datetime import datetime, timezone
from bisect import bisect_right
import ast
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
R6 = ROOT / "research/model_upgrade_validation/dragon_auction_and_second_wave_20261027T091600Z"
R4 = ROOT / "research/model_upgrade_validation/dragon_theme_matrix_20261027T083500Z"
R5 = ROOT / "research/model_upgrade_validation/dragon_regime_gate_20261027T085500Z"
FOLDS = [
    ("2024-08-31", "2024-09-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-05-31"),
    ("2025-05-31", "2025-06-01", "2025-11-30"),
    ("2025-11-30", "2025-12-01", "2026-08-31"),
]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    d = pd.read_csv(p, dtype={"code": str, "top1_code": str}, low_memory=False)
    if "code" in d:
        d["code"] = d["code"].str.zfill(6)
    return d


def pick(d, col, ascending=False):
    return d.sort_values(["date", col, "code"], ascending=[True, ascending, True]).groupby("date", sort=True).head(1).copy()


def stats(d):
    ans = {"n": len(d)}
    for k in ["is_3to4", "is_3to5", "is_market_space_leader", "is_emotion_dragon_proxy", "is_10d_100_abnormal", "is_second_wave_leader"]:
        ans[k] = {"hits": int(d[k].sum()), "rate": float(d[k].mean()) if len(d) else None}
    if len(d):
        n = len(d)
        p = d.is_3to4.mean()
        z = 1.959963984540054
        center = (p + z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        ans["3to4_descriptive_wilson95"] = [center-half, center+half]
    ans["whole_day_one_price_up"] = int(d.trade_one_word_up.eq(1).sum())
    ans["next_open_ge_9p5pct"] = int(d.next_open_gap.ge(.095).sum())
    ans["base_lt50"] = int(d.base_historical_score.lt(.5).sum())
    ans["base_missing"] = int(d.base_historical_score.isna().sum())
    ans["execution_status"] = d.execution_status.value_counts(dropna=False).to_dict()
    ans["max_10d_return_mean_not_executable_pnl"] = float(d.forward_10d_max_return.mean()) if len(d) else None
    ans["positive_max_but_negative_10th_close"] = int((d.forward_10d_max_return.gt(0) & d.forward_10d_close_return.lt(0)).sum())
    ans["10th_close_negative"] = int(d.forward_10d_close_return.lt(0).sum())
    return ans


def main():
    input_files = [p for base in [R6, R4, R5] for p in base.iterdir() if p.is_file()]
    input_files += [ROOT / f for f in ["build_round4_dataset.py", "build_round5_regime_dataset.py", "build_round6_dataset.py", "train_and_evaluate_round4.py", "train_and_evaluate_round5_gate.py", "train_and_evaluate_round6.py", "evaluate_round6_comprehensive.py", "audit_round6_strict.py", "audit_three_board_traps.py", "multiround_stress_test_and_enhancement.py", "calculate_exact_premiums.py", "leader_system_v3.py", "AGENTS.md", "data/v3/imported/daily-bars.csv", "data/v279/exchange_calendar_audit_20261002_v1/calendar.json", "research/first_board_github_industry_qualification_20261020/qualified_legacy_snapshot.json"]]
    protected = ROOT / "research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE"
    input_files += [p for p in protected.rglob("*") if p.is_file()]
    industry_raw_path = ROOT / "research/first_board_github_industry_qualification_20261020/stock_basics_industry_original.csv"
    input_files += [industry_raw_path] + list((ROOT / "data/2026-08").glob("*.csv"))
    before = {str(p.relative_to(ROOT)): sha(p) for p in input_files}
    d, o, daily = [read(R6 / x) for x in ["round6_dataset.csv", "oof_round6_predictions.csv", "daily_competition_round6.csv"]]
    assert not d.duplicated(["date", "code"]).any()
    assert not o.duplicated(["date", "code"]).any()
    tree = ast.parse((ROOT / "train_and_evaluate_round6.py").read_text(encoding="utf-8"))
    raw_features = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "raw_feature_cols" for t in n.targets))
    formula = .30*o.score_space_leader + .25*o.score_height_5 + .25*o.score_10d_100_abnormal + .20*o.score_second_wave
    top = pick(o, "score_round6_total_dragon")
    reg = read(R5 / "oof_round5_daily_gate_predictions.csv")[["date", "prob_dragon_day", "gate_arm_b_open_th55"]]
    top = top.merge(reg, on="date", validate="one_to_one")
    stored = daily.set_index("date").sort_index()
    got = top.set_index("date").sort_index()
    assert got.index.equals(stored.index)
    assert (got.code == stored.top1_code.str.zfill(6)).all()
    assert np.allclose(formula, o.score_round6_total_dragon)
    top.to_csv(OUT / "top1_recomputed.csv", index=False, encoding="utf-8-sig")
    masks = {
        "all_top1": pd.Series(True, index=top.index),
        "positive_open": top.next_open_gap.gt(0),
        "nonpositive_open": top.next_open_gap.le(0),
        "missing_open": top.next_open_gap.isna(),
        "positive_below_9p5pct": top.next_open_gap.gt(0) & top.next_open_gap.lt(.095),
        "open_ge_9p5pct_not_one_word_identity": top.next_open_gap.ge(.095),
        "gold_original_0p40": top.next_open_gap.gt(0) & top.prob_dragon_day.ge(.55) & top.score_round6_total_dragon.ge(.40),
        "gold_original_without_whole_day_one_price": top.next_open_gap.gt(0) & top.prob_dragon_day.ge(.55) & top.score_round6_total_dragon.ge(.40) & ~top.trade_one_word_up.eq(1),
        "gold_original_positive_below_9p5pct": top.next_open_gap.gt(0) & top.next_open_gap.lt(.095) & top.prob_dragon_day.ge(.55) & top.score_round6_total_dragon.ge(.40),
        "enhanced_posthoc_0p35": top.next_open_gap.ge(.02) & top.prob_dragon_day.ge(.55) & top.score_round6_total_dragon.ge(.35),
        "enhanced_posthoc_0p40": top.next_open_gap.ge(.02) & top.prob_dragon_day.ge(.55) & top.score_round6_total_dragon.ge(.40),
    }
    tier_stats = {k: stats(top[m]) for k, m in masks.items()}
    metrics = {}
    for target, score in [("is_10d_100_abnormal", "score_10d_100_abnormal"), ("is_second_wave_leader", "score_second_wave")]:
        metrics[target] = {"n": len(o), "positives": int(o[target].sum()), "auc": float(roc_auc_score(o[target], o[score])), "average_precision": float(average_precision_score(o[target], o[score])), "positive_rate": float(o[target].mean())}
    r4 = read(R4 / "oof_round4_predictions.csv")
    r4_top = pick(r4, "score_round4_composite")
    compare = {}
    # Report saved R4 and R6 over identical dates, without introducing a new threshold.
    for name, frame in [("round4", r4_top), ("round6", top)]:
        compare[name] = {k: {"n": len(frame), "hits": int(frame[k].sum()), "rate": float(frame[k].mean())} for k in ["is_3to4", "is_3to5", "is_market_space_leader", "is_emotion_dragon_proxy"]}
    comp = o.groupby("date").filter(lambda g: len(g) >= 2)
    complete = comp[comp.base_cohort_match_status.eq("COMPLETE_MATCH")]
    complete_m = pick(complete, "score_round6_total_dragon")
    complete_b = pick(complete, "base_historical_score")
    compare["same_complete_competitive_dates_r6_vs_BASE"] = {"R6": stats(complete_m), "BASE": stats(complete_b)}
    compare["diagnostic_min_amplitude_same_competitive_dates"] = {"R6": stats(pick(comp, "score_round6_total_dragon")), "minimum_amplitude": stats(pick(comp, "amplitude", True)), "limitation": "Read after outcomes; diagnostic only, no rule optimization."}
    # Reconstruct exact future-bar availability and label observation dates; no model fitting.
    bars = pd.read_csv(ROOT / "data/v3/imported/daily-bars.csv", dtype={"code": str}, usecols=["date", "code", "high", "close"])
    bars.code = bars.code.str.zfill(6)
    duplicate_bar_n = int(bars.duplicated(["date", "code"]).sum())
    assert duplicate_bar_n == 0
    bars = bars.sort_values(["code", "date"])
    by_code = {c: g for c, g in bars.groupby("code")}
    sessions = json.loads((ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json").read_text(encoding="utf-8"))["sessions"]
    details = []
    for row in d.itertuples():
        g = by_code.get(row.code)
        sub = g[g.date.gt(row.date)].head(10) if g is not None else bars.head(0)
        i = bisect_right(sessions, row.date)
        expected = sessions[i:i+10]
        observed = sub.date.tolist()
        official_missing = len(set(expected) - set(observed))
        ret = sub.high.max()/row.price-1 if len(sub) else np.nan
        close_ret = sub.close.iloc[-1]/row.price-1 if len(sub) else np.nan
        item = {"date": row.date, "code": row.code, "name": row.name, "forward_bars": len(sub), "end_10_observed": sub.date.iloc[-1] if len(sub)==10 else "", "missing_official_sessions_within10": official_missing, "official10_complete": len(sub)==10 and observed==expected, "saved_y100": row.is_10d_100_abnormal, "saved_y2w": row.is_second_wave_leader, "recomputed_max_return": ret, "stored_max_return": row.forward_10d_max_return, "terminal_board": row.terminal_board, "determined_date_final": row.determined_date_final}
        for cutoff, _, _ in FOLDS:
            old_train = row.date <= cutoff and row.determined_date_final <= cutoff
            pre = sub[sub.date.le(cutoff)]
            pre_max = pre.high.max()/row.price-1 if len(pre) else -np.inf
            item[cutoff+"_old_train"] = old_train
            item[cutoff+"_10d_unobserved"] = old_train and (len(sub)<10 or observed[-1]>cutoff)
            item[cutoff+"_y100_positive_only_after_cutoff"] = old_train and row.is_10d_100_abnormal==1 and row.terminal_board<7 and pre_max<.464
            item[cutoff+"_y2w_positive_only_after_cutoff"] = old_train and row.is_second_wave_leader==1 and pre_max<.18
        details.append(item)
    paths = pd.DataFrame(details)
    paths.to_csv(OUT / "all1851_forward_label_diagnostics.csv", index=False, encoding="utf-8-sig")
    incomplete = paths[paths.forward_bars.lt(10)]
    incomplete.to_csv(OUT / "incomplete_forward_paths.csv", index=False, encoding="utf-8-sig")
    folds = []
    for cutoff, start, end in FOLDS:
        f = {"cutoff": cutoff, "test_start": start, "train_n": int(paths[cutoff+"_old_train"].sum()), "train_10d_window_not_observed": int(paths[cutoff+"_10d_unobserved"].sum()), "definite_positive_100_only_after_cutoff": int(paths[cutoff+"_y100_positive_only_after_cutoff"].sum()), "definite_positive_2w_only_after_cutoff": int(paths[cutoff+"_y2w_positive_only_after_cutoff"].sum())}
        # R5 includes signal dates before cutoff without ANY target-maturity filter.
        last_signal = max(v for v in sessions if v <= cutoff)
        latest = d[d.date.eq(last_signal)]
        f["r5_last_signal_date"] = last_signal
        f["r5_last_day_has_candidates"] = len(latest)>0
        f["r5_last_day_label_is_dragon"] = int(latest.is_market_space_leader.max()==1 or latest.is_3to5.max()==1) if len(latest) else None
        f["r5_last_day_has_future_target_maturity"] = bool((latest.determined_date_3to4.gt(cutoff)).all()) if len(latest) else None
        folds.append(f)
    # Unknown legacy industries are pooled as one artificial industry; expose affected candidate rows.
    unknown = d[d.industry.eq("UNKNOWN_INDUSTRY")]
    unknown.to_csv(OUT / "unknown_industry_pooled_rows.csv", index=False, encoding="utf-8-sig")
    gate_conflicts = top[masks["gold_original_0p40"] & (top.base_historical_score.lt(.5) | top.base_historical_score.isna())]
    gate_conflicts.to_csv(OUT / "golden_BASE_gate_conflicts.csv", index=False, encoding="utf-8-sig")
    gold = top[masks["gold_original_0p40"]]
    # Paired checks are assertions about reproduction only, not strategy success.
    assert len(o) == 1480 and len(top) == 449
    assert int(top.is_3to4.sum()) == 255
    assert len(gold) == 52 and int(gold.is_3to4.sum()) == 39
    assert np.allclose(paths.recomputed_max_return, paths.stored_max_return, equal_nan=True)
    label_summary = {"total": len(paths), "no_forward_bars": int(paths.forward_bars.eq(0).sum()), "fewer_than10": len(incomplete), "incomplete_y100_zero": int(incomplete.saved_y100.eq(0).sum()), "incomplete_y2w_zero": int(incomplete.saved_y2w.eq(0).sum()), "unknown_2w_eligible_but_marked_zero": int((incomplete.terminal_board.le(4)&incomplete.saved_y2w.eq(0)).sum()), "not_exact10_official_sessions": int((~paths.official10_complete).sum()), "path_max_formula_reproduced": True, "duplicate_bar_keys": duplicate_bar_n}
    old_industry = pd.read_csv(industry_raw_path, dtype={"symbol": str})
    primary_codes = set(old_industry.symbol.str.zfill(6))
    augmented = d[~d.code.isin(primary_codes) & ~d.industry.eq("UNKNOWN_INDUSTRY")]
    industry_aug = {"rows": len(augmented), "codes": int(augmented.code.nunique()), "rows_before_august2026": int(augmented.date.lt("2026-08-01").sum()), "earliest_date": augmented.date.min()}
    source_after = {str(p.relative_to(ROOT)): sha(p) for p in input_files}
    assert before == source_after, "Source changed during audit; do not certify snapshot."
    audit = {"formed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Read-only audit of already seen historical R6 outputs; no fresh forward validation", "fit_count": 0, "new_forward_predictions": 0, "orders": 0, "source_unchanged": True, "total_dataset_rows": len(d), "oof_rows": len(o), "test_days": len(top), "score_formula_and_top1_reproduced": True, "raw_feature_columns": raw_features, "obvious_future_columns_in_raw_features": [x for x in raw_features if x.startswith(("next_", "forward_", "target_"))], "tier_statistics": tier_stats, "new_head_metrics": metrics, "comparison": compare, "label_availability": label_summary, "fold_label_maturity": folds, "unknown_industry": {"rows": len(unknown), "multiple_unknown_peers": int(unknown.theme_cluster_total_zt.gt(1).sum()), "fabricated_positive_first_board_peer_count": int(unknown.theme_first_board_count.gt(0).sum())}, "august2026_industry_used_for_older_events": industry_aug, "saved_R6_models": [p.name for p in R6.glob("*.joblib")], "source_sha256": before, "limitations": ["Source declarations are not an independent external verification of the entire historical universe.", "Rates inherit saved labels, some labels are incomplete; not corrected-model estimates.", "Daily one-price and high-open flags are separate; excluding one-price sessions does not establish actual fills.", "Wilson interval is descriptive and ignores serial dependence, repeated research and selection.", "No model retraining; old strict audit script itself retrains and is not invoked here."]}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: audit[k] for k in ["formed_at_utc", "source_unchanged", "new_head_metrics", "comparison", "label_availability", "fold_label_maturity", "unknown_industry", "saved_R6_models"]}
    summary["tier_statistics"] = {k: {a: v for a,v in s.items() if a in ["n", "is_3to4", "is_3to5", "is_market_space_leader", "whole_day_one_price_up", "next_open_ge_9p5pct", "base_lt50", "base_missing", "10th_close_negative", "positive_max_but_negative_10th_close"]} for k,s in tier_stats.items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
