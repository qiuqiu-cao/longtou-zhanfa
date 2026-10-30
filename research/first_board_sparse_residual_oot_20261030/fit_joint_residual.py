"""Explicit, separately frozen research fit; never invoked by default replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np

from residual_core import HERE, ROOT, DATA, Z, KEYS, P2, features, select, load_inputs, mature_training, evaluate
from joint_top3_milp import solve_joint_top3
from review_residual_arms import sha, protection_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_FOR_RESEARCH" or not protocol.get("actual_discussion_record"):
        raise ValueError("This new MILP specification needs a real recorded freeze, not the DE report")
    required = [HERE/"residual_core.py", HERE/"joint_top3_milp.py", HERE/"fit_joint_residual.py",
                HERE/"review_residual_arms.py",
                ROOT/"research/first_board_math_20261030/review_gemini_cascade.py",
                ROOT/"research/first_board_top3_40_20261030/analyze_top3_gap.py",
                DATA/"FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv", DATA/"OUTCOMES_DIAGNOSTIC_ONLY.csv",
                ROOT/"research/first_second_stage_20261020_run1/S1_ALL_PREDICTIONS.csv"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in required}
    if protocol.get("source_hashes") != hashes:
        raise ValueError("Frozen source hashes do not match this implementation and data")
    x, pool, maturity = load_inputs()
    cutoff, start, end = protocol["fit_as_of"], protocol["test_start"], protocol["test_end"]
    if not cutoff < start <= end:
        raise ValueError("Training cutoff must precede all evaluation signal dates")
    train, purged = mature_training(pool, maturity, cutoff)
    test = pool[pool.signal_date.between(start, end)]
    if test.empty:
        raise ValueError("Empty temporal evaluation period")
    f = features(x)
    train_f = f.merge(train[KEYS], on=KEYS, validate="one_to_one")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output/"PROTOCOL_BEFORE_FIT.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for source in required:
        if source.suffix == ".py":
            dest = output/"source_snapshot"/source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    purged.to_csv(output/"PURGED_TRAINING_ROWS.csv", index=False, encoding="utf-8-sig")
    protection_check()
    params = protocol["parameters"]
    records = {}
    baseline = select(f, np.zeros(len(Z)))
    records["S1_BASELINE"] = {name: evaluate(frame, baseline[baseline.signal_date.isin(frame.signal_date)])[0]
                               for name, frame in [("mature_train", train), ("later_period", test)]}
    for name, lam in [("A_JOINT_SPARSE", 0.), ("B_JOINT_SPARSE_ROBUST", params["robust_lambda"])]:
        result = solve_joint_top3(train_f, train[KEYS+[P2]], feature_columns=Z,
            l1_bound=params["l1_bound"], max_active_features=params["max_active_features"],
            boundary_margin=params["boundary_margin"], robust_lambda=lam,
            time_limit=params["time_limit_seconds"], mip_rel_gap=params["mip_rel_gap"])
        # Persist failures and timed-out incumbents too. Never silently change settings/retry.
        record = dict(theta_full_precision=result["theta"], solver=result["metadata"])
        if result["theta"] is not None:
            picks = select(f, result["theta"])
            _, detail = evaluate(pool, picks)
            detail.to_csv(output/f"{name}_FULL_SELECTIONS.csv", index=False, encoding="utf-8-sig")
            record["metrics"] = {split: evaluate(frame, picks[picks.signal_date.isin(frame.signal_date)])[0]
                                  for split, frame in [("mature_train", train), ("later_period", test),
                                                        ("mixed_full_period_diagnostic_only", pool)]}
        records[name] = record
        (output/f"{name}.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    if any(sha(ROOT/p) != h for p, h in hashes.items()):
        raise RuntimeError("Frozen sources changed during fit")
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), results=records,
        source_hashes=hashes, protocol=protocol, protected_files_unchanged=protection_check(),
        production_changed=False, independent_blind_test=False, predictive_validation_passed=False,
        warning="Repeatedly inspected historical period; no untouched future-period validation")
    (output/"REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
