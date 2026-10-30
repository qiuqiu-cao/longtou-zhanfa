"""Reproduce Gemini's temporary tests, then audit on the complete original pool.

No training, threshold search, theme invention, production score changes, or old
forecast overwrites. The original rule and a full-pool scope correction are kept
separate; neither is treated as validated prediction improvement.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd

from industry_timing_proxy import KEYS, GROUP, TIME, timing_features, baseline, choose_proxy_swap

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT/"research/first_board_top3_40_20261030/results"
P2 = "promotion_next_LABEL_ONLY"
P4 = "reach4_LABEL_ONLY"


def read(p):
    return pd.read_csv(p, dtype={"code": str}, float_precision="round_trip", low_memory=False)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def existing_evaluator():
    p = ROOT/"research/first_board_math_20261030/review_gemini_cascade.py"
    s = importlib.util.spec_from_file_location("theme_review_contract", p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m.evaluate


def original_flags(d):
    """Literal original behavior; diagnostic use only, including false defaults."""
    f = d[KEYS+["s1_score", GROUP, TIME]].copy()
    f[GROUP] = f[GROUP].fillna("UNKNOWN")
    f["proxy_status"] = "OBSERVED_LOCAL_STATIC_GROUP"
    f["is_earliest_in_observed_group"] = False
    f["later_first_seals_within_window"] = 0
    for (_, group), g in f.groupby(["signal_date", GROUP]):
        if group == "UNKNOWN":
            continue
        first = g[TIME].min()
        earliest = g[TIME].eq(first)
        followers = int((g[TIME].gt(first) & g[TIME].le(first+30)).sum())
        f.loc[g.index, "is_earliest_in_observed_group"] = earliest
        f.loc[g.index[earliest], "later_first_seals_within_window"] = followers
    return f


def subgroup_counts(f, pool):
    q = f.merge(pool[KEYS+[P2]], on=KEYS, validate="one_to_one")
    observed = q.proxy_status.eq("OBSERVED_LOCAL_STATIC_GROUP")
    solo = q.local_group_size.eq(1).fillna(False)
    multi = q.local_group_size.gt(1).fillna(False)
    early = q.is_earliest_in_observed_group.fillna(False)
    later = q.later_first_seals_within_window
    groups = {"LOCAL_SINGLE_MEMBER": observed & solo, "LOCAL_MULTIPLE_MEMBERS": observed & multi,
              "EARLIEST_WITH_LATER_SEALS": observed & multi & early & later.ge(1).fillna(False),
              "EARLIEST_WITHOUT_LATER_SEALS": observed & multi & early & later.eq(0).fillna(False),
              "LATER_SEALING_MEMBERS": observed & multi & ~early,
              "UNKNOWN_PROXY_EVIDENCE": ~observed}
    result = {}
    for name, mask in groups.items():
        g = q[mask]
        known = int(g[P2].notna().sum()); hits = int(g[P2].eq(1).sum())
        result[name] = dict(rows=len(g), known=known, unknown=len(g)-known, hits=hits,
                            known_precision=hits/known if known else None)
    return result


def swap_ledger(pool, changes):
    out = changes.copy()
    for direction, code in [("promoted", "promoted_code"), ("displaced", "displaced_code")]:
        source = pool[KEYS+[P2,P4]].rename(columns={"code":code, P2:direction+"_promotion", P4:direction+"_reach4"})
        out = out.merge(source, on=["signal_date", code], how="left", validate="many_to_one")
    return out


def summarize_swaps(g):
    answer = dict(swaps=len(g))
    for direction in ["promoted", "displaced"]:
        values = g[direction+"_promotion"]
        answer[direction] = dict(hits=int(values.eq(1).sum()), failures=int(values.eq(0).sum()),
                                 unknown=int(values.isna().sum()))
    answer["net_confirmed_hits"] = answer["promoted"]["hits"]-answer["displaced"]["hits"]
    both = g.promoted_promotion.notna() & g.displaced_promotion.notna()
    answer["known_pairs_corrected_failure_to_success"] = int((both & g.promoted_promotion.eq(1) & g.displaced_promotion.eq(0)).sum())
    answer["known_pairs_damaged_success_to_failure"] = int((both & g.promoted_promotion.eq(0) & g.displaced_promotion.eq(1)).sum())
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=HERE/"results")
    out = ap.parse_args().output; out.mkdir(parents=True, exist_ok=False)
    x, y = read(DATA/"FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv"), read(DATA/"OUTCOMES_DIAGNOSTIC_ONLY.csv")
    pool = x.merge(y, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not pool.pop("_merge").eq("both").all() or len(pool)!=7254 or pool.signal_date.nunique()!=286:
        raise ValueError("Original pool contract changed")
    files = [Path(__file__), HERE/"industry_timing_proxy.py",
             DATA/"FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv", DATA/"OUTCOMES_DIAGNOSTIC_ONLY.csv"]
    files += sorted((HERE/"source_snapshot").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/"PROTOCOL_BEFORE_REPLAY.json").write_text(json.dumps(dict(
        created_at=datetime.now(timezone.utc).isoformat(), source_hashes=hashes, new_market_fits=0,
        original_rule=dict(window_trading_minutes=30, challenger_ranks="4-8", displaced_slots_order=[2,3]),
        corrected_rule=dict(window_trading_minutes=30, challenger_ranks="ALL_AFTER_TOP3", displaced_slots_order=[2,3]),
        rule_status="POSTHOC_RESEARCH_DIAGNOSTIC_ONLY", dynamic_theme_claim_allowed=False), indent=2),encoding="utf-8")
    reproduced = {}
    for step in [505,510,515]:
        capture = io.StringIO()
        with redirect_stdout(capture):
            runpy.run_path(str(HERE/f"source_snapshot/gemini_step_{step}.py"), run_name="__main__")
        reproduced[str(step)] = capture.getvalue()
        (out/f"ORIGINAL_STEP_{step}_REPLAY.txt").write_text(capture.getvalue(), encoding="utf-8")
    # Reproduce the old bug only in a clearly isolated diagnostic arm.
    known = pool[pool[P2].notna()].copy()
    legacy = original_flags(known)
    full_legacy = original_flags(x)
    corrected = timing_features(x)
    corrected.to_csv(out/"FULL_POOL_PROXY_FEATURES_NO_LABELS.csv",index=False,encoding="utf-8-sig")
    picks = {"S1_COMPLETE_BASELINE":(baseline(x),None),
             "ORIGINAL_DROPPED_UNKNOWN_BASELINE_DIAGNOSTIC":(baseline(known),None),
             "ORIGINAL_DROPPED_UNKNOWN_RULE_DIAGNOSTIC":choose_proxy_swap(legacy,8),
             "FULL_POOL_ORIGINAL_MISSING_POLICY_CAP8_DIAGNOSTIC":choose_proxy_swap(full_legacy,8),
             "CORRECT_UNKNOWN_POLICY_CAP8_MATCHED_DIAGNOSTIC":choose_proxy_swap(corrected,8),
             "CORRECT_UNKNOWN_POLICY_FULL_POOL_DIAGNOSTIC":choose_proxy_swap(corrected,None)}
    evaluate = existing_evaluator()
    results, swaps, monthly = {}, {}, []
    splits = {"EARLIER_203_DAYS":pool.signal_date.le("2026-04-30"),
              "LATER_83_DAYS_PREVIOUSLY_INSPECTED":pool.signal_date.ge("2026-05-01"),
              "FULL_286_DAYS":pd.Series(True,index=pool.index)}
    for method,(take,changes) in picks.items():
        rep, detail = evaluate(pool,take,sorted(pool.signal_date.unique()))
        detail.to_csv(out/f"{method}_SELECTIONS.csv",index=False,encoding="utf-8-sig")
        ledger = swap_ledger(pool,changes) if changes is not None else None
        if ledger is not None: ledger.to_csv(out/f"{method}_SWAPS.csv",index=False,encoding="utf-8-sig")
        results[method] = {}; swaps[method] = {}
        for split,mask in splits.items():
            sub = pool[mask]; dates = sorted(sub.signal_date.unique())
            q = take[take.signal_date.isin(dates)]
            metric, selected = evaluate(sub,q,dates)
            states = selected.merge(x[KEYS+["one_price"]],on=KEYS,validate="one_to_one")
            non_one = states[states.one_price.eq(0)]
            metric["signal_day_non_one_price"] = dict(selected=len(non_one),hits=int(non_one[P2].eq(1).sum()),
                known=int(non_one[P2].notna().sum()),unknown=int(non_one[P2].isna().sum()))
            results[method][split] = metric
            if ledger is not None: swaps[method][split] = summarize_swaps(ledger[ledger.signal_date.isin(dates)])
        for month,g in detail.groupby(detail.signal_date.str[:7]):
            monthly.append(dict(method=method,month=month,selected=len(g),hits=int(g[P2].eq(1).sum()),
                                known=int(g[P2].notna().sum()),unknown=int(g[P2].isna().sum())))
    pd.DataFrame(monthly).to_csv(out/"MONTHLY_COMPARISON.csv",index=False,encoding="utf-8-sig")
    protected = json.loads((ROOT/"research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json").read_text(encoding="utf-8"))["protected_sha256"]
    if any(sha(ROOT/p)!=h for p,h in protected.items()): raise RuntimeError("Protected file changed")
    if any(sha(ROOT/p)!=h for p,h in hashes.items()): raise RuntimeError("Source changed during audit")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),results=results,swaps=swaps,
        source_hashes=hashes,original_replay_stdout=reproduced,
        corrected_subgroups=subgroup_counts(corrected,pool),
        input_rows=len(pool),input_dates=int(pool.signal_date.nunique()),
        original_removed_unknown_rows=int(pool[P2].isna().sum()),original_remaining_dates=int(known.signal_date.nunique()),
        unknown_industry_rows=int(x[GROUP].isna().sum()),unknown_first_seal_rows=int(x[TIME].isna().sum()),
        feature_status_counts={str(k):int(v) for k,v in corrected.proxy_status.value_counts().items()},
        protected_files_unchanged=len(protected),new_market_fits=0,production_changed=False,
        independent_blind_test=False,dynamic_theme_detection_implemented=False,
        temporal_parent_anchor_discovery_implemented=False,market_intention_forecast_implemented=False,
        validated_upgrade=False,target_40_percent_achieved=False,
        limitations=["2021 static industry is not a historical dynamic trade narrative.",
            "Context contains qualified first boards only, excluding high anchors, failed boards and outside-cap peers.",
            "Earlier and later first-seal times do not establish influence, funds flow, persistence or reseal quality.",
            "30 means trading minutes, excluding lunch, not continuous 30-minute clock time.",
            "Missing evidence is unknown, never a negative role label.",
            "Both these months and the earlier full history have been inspected for multiple rules.",
            "Full-pool correction is not a new validated strategy or permission to deploy."])
    (out/"REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({name:{split:dict(hits=r['hits'],known=r['known'],unknown=r['unknown'],unfilled=r['unfilled_slots'])
        for split,r in values.items()} for name,values in results.items()},indent=2))


if __name__=="__main__": main()
