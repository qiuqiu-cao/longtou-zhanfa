"""Verify selected rows against immutable candidate evidence, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from height_anchor_signal_diagnostic_v133 import diagnose


def audit(selected, candidates):
    keys = ["signal_date", "code"]
    if candidates.duplicated(keys).any() or selected.signal_date.duplicated().any():
        raise ValueError("duplicate candidate keys or multiple daily selections")
    common = [c for c in candidates.columns if c in selected.columns and c not in keys]
    joined = selected.merge(candidates, on=keys, how="left", suffixes=("_pick", "_source"),
                            indicator=True, validate="one_to_one")
    mismatches = {}
    for column in common:
        left, right = joined[column + "_pick"], joined[column + "_source"]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.isclose(left.astype(float), right.astype(float), rtol=1e-10,
                               atol=1e-12, equal_nan=True)
        else:
            equal = left.eq(right) | (left.isna() & right.isna())
        mismatches[column] = int((~equal).sum())
    missing = int(joined._merge.ne("both").sum())
    return {"passed": missing == 0 and not any(mismatches.values()),
            "selected_rows": len(selected), "missing_source_rows": missing,
            "field_mismatch_counts": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = pd.read_csv(args.selected, dtype={"code": str})
    candidates = pd.read_csv(args.candidates, dtype={"code": str})
    result = {"row_integrity": audit(selected, candidates),
              "corrected_candidate_diagnostic": diagnose(candidates),
              "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.selected, args.candidates]},
              "refit_performed": False, "deployment_authorized": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result["row_integrity"], ensure_ascii=False))
    if not result["row_integrity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
