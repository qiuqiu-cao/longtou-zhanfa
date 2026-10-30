"""Replay Gemini's archived rounded residual weights. Never fit market data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from residual_core import ROOT, HERE, DATA, KEYS, P2, P4, Z, features, select, load_inputs, mature_training, evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protection_check():
    p = ROOT / "research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json"
    protected = json.loads(p.read_text(encoding="utf-8"))["protected_sha256"]
    changed = [name for name, digest in protected.items() if sha(ROOT/name) != digest]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    return len(protected)


def main(output=None):
    out = Path(output) if output else HERE / "review_results"
    out.mkdir(parents=True, exist_ok=False)
    source = HERE / "review_source_snapshot/REPORT.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    x, pool, maturity = load_inputs()
    trained, purged = mature_training(pool, maturity, "2026-04-30")
    purged.to_csv(out/"TRAINING_CUTOFF_PURGED_ROWS.csv", index=False, encoding="utf-8-sig")
    source_files = [source, DATA/"FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv",
                    DATA/"OUTCOMES_DIAGNOSTIC_ONLY.csv", HERE/"residual_core.py",
                    HERE/"review_residual_arms.py",
                    ROOT/"research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv"]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    action="SAVED_FOUR_DECIMAL_COEFFICIENT_REPLAY_ONLY", market_fits=0,
                    source_hashes=source_hashes, original_cutoff="2026-04-30",
                    evaluation_start="2026-05-06", full_precision_weights_available=False,
                    independent_blind_test=False, fixes_do_not_remove_original_fit_contamination=True)
    (out/"PROTOCOL_BEFORE_REPLAY.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    masks = dict(NOMINAL_TRAIN=pool.signal_date.le("2026-04-30"),
                 LATER_83_DAYS_ALREADY_INSPECTED=pool.signal_date.ge("2026-05-01"),
                 FULL_286_DAYS_MIXED_TRAIN_AND_LATER=np.ones(len(pool), dtype=bool))
    f_neutral = features(x)
    f_old = features(x, "original_imputation_diagnostic")
    methods = {"S1_BASELINE": select(f_neutral, np.zeros(4))}
    for name, key in [("ARM_A", "theta_a"), ("ARM_B", "theta_b")]:
        w = [report[key][z] for z in Z]
        methods[name+"_ORIGINAL_IMPUTATION_ROUNDED_REPLAY"] = select(f_old, w)
        methods[name+"_NEUTRAL_MISSING_ROUNDED_REPLAY"] = select(f_neutral, w)
    results, details, monthly = {}, {}, []
    for method, picks in methods.items():
        result, detail = evaluate(pool, picks)
        detail.to_csv(out/f"{method}_SELECTIONS.csv", index=False, encoding="utf-8-sig")
        details[method] = detail
        splits = {}
        for name, mask in masks.items():
            sub = pool[mask]
            q = picks[picks.signal_date.isin(sub.signal_date)]
            splits[name] = evaluate(sub, q)[0]
        results[method] = splits
        for month, frame in detail.groupby(detail.signal_date.str[:7]):
            monthly.append(dict(method=method, month=month, dates=frame.signal_date.nunique(),
                                slots=len(frame), hits=int(frame[P2].eq(1).sum()),
                                known=int(frame[P2].notna().sum()), unknown=int(frame[P2].isna().sum()),
                                reach4_hits=int(frame[P4].eq(1).sum())))
    pd.DataFrame(monthly).to_csv(out/"MONTHLY_REPLAY.csv", index=False, encoding="utf-8-sig")
    swaps = {}
    base_keys = set(map(tuple, details["S1_BASELINE"][KEYS].to_numpy()))
    for name, detail in details.items():
        if name == "S1_BASELINE":
            continue
        keys = set(map(tuple, detail[KEYS].to_numpy()))
        by_split = {}
        for split, mask in masks.items():
            sub = pool[mask]
            comparison = {}
            for direction, wanted in [("promoted", keys-base_keys), ("removed", base_keys-keys)]:
                q = sub[[tuple(key) in wanted for key in sub[KEYS].to_numpy()]]
                comparison[direction] = dict(rows=len(q), hits=int(q[P2].eq(1).sum()),
                    failures=int(q[P2].eq(0).sum()), unknown=int(q[P2].isna().sum()),
                    reach4_hits=int(q[P4].eq(1).sum()))
            by_split[split] = comparison
        swaps[name] = by_split
    differences = []
    for name, old in [("S1_BASELINE", "S1_BASELINE"),
                      ("ARM_A_ORIGINAL_IMPUTATION_ROUNDED_REPLAY", "ARM_A_DIRECT_TOP3"),
                      ("ARM_B_ORIGINAL_IMPUTATION_ROUNDED_REPLAY", "ARM_B_GROUP_ROBUST")]:
        for new_split, old_split in [("NOMINAL_TRAIN", "in_sample_train"),
                                     ("LATER_83_DAYS_ALREADY_INSPECTED", "out_of_time_test"),
                                     ("FULL_286_DAYS_MIXED_TRAIN_AND_LATER", "full_sample_286d")]:
            for field in ["hits", "known", "unknown", "reach4_hits", "reach4_known"]:
                actual = results[name][new_split][field]
                claimed = report["results"][old][old_split][field]
                if actual != claimed:
                    differences.append(dict(method=name, split=new_split, field=field,
                                            archived_value=claimed, rounded_replay_value=actual))
    summary = dict(protocol=protocol, results=results, swaps=swaps,
        archived_count_vs_rounded_replay_differences=differences,
        training_boundary=dict(nominal_dates=203, nominal_rows=4682,
            mature_dates=int(trained.signal_date.nunique()), mature_rows=len(trained),
            purged_dates=int(purged.signal_date.nunique()), purged_rows=len(purged),
            purged_known_outcomes=int(purged[P2].notna().sum()),
            actual_label_end_dates=sorted(purged.label_end.dropna().unique().tolist()),
            warning="Deleting rows for this audit DOES NOT purge already-fitted coefficients"),
        original_implementation=dict(solver="DIFFERENTIAL_EVOLUTION", integer_programming=False,
            coefficient_count=4, l1_bound=.15, nonzero_coefficients_A=4, nonzero_coefficients_B=4,
            enforced_sparsity=False, neutral_missing_handling=False, full_precision_weights_saved=False),
        repaired_milp_market_fits=0, production_changed=False,
        predictive_validation_passed=False, target_40_percent_achieved=False,
        protected_files_unchanged=protection_check(),
        limitations=["Later four months already informed previous cascade experiments; not pristine blind history.",
            "At an April-30 fit cutoff, April-30 outcomes are unavailable until May-6 close.",
            "An explicit May-6-after-close refit would require its own documented fit clock; none was archived.",
            "Archived weights are rounded; full-precision original selections cannot be certified.",
            "Later-retrieved provider seal data do not establish original historical availability.",
            "S-day one-price status is not a measure of next-day fillability.",
            "New MILP code is tested on synthetic data only, not an improvement result."])
    if any(sha(ROOT/p) != h for p, h in source_hashes.items()):
        raise RuntimeError("Source changed during replay")
    original = HERE/"results/REPORT.json"
    if sha(original) != sha(source):
        raise RuntimeError("Original report was modified")
    (out/"REPORT.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: {s: f"{r['hits']}/{r['known']} ({r['known_precision']:.4%})" for s, r in v.items()}
                      for k, v in results.items()}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
