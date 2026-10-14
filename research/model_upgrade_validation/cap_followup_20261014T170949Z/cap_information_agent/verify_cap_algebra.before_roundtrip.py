"""Independently verify saved algebra, coverage, source binding and train sets."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / "cap_algebra_results.json").read_text(encoding="utf-8"))
    for source, digest in report["input_bindings"].items():
        assert sha(source) == digest
    for output, digest in report["output_hashes"].items():
        assert sha(HERE / output) == digest
    assert sha(HERE / "audit_cap_algebra.py") == report["script_sha256"]
    prepared = joblib.load(next(p for p in report["input_bindings"] if p.endswith("prepared_source_reconstruction.joblib")))
    original = prepared["relay"]
    saved = pd.read_csv(HERE / "full9245_cap_algebra_NO_OUTCOMES.csv", dtype={"code": str})
    assert len(saved) == len(original) == 9245
    assert not saved.duplicated(["date", "code"]).any()
    assert not set(saved.columns) & {"target", "target_date", "label_valid", "prediction", "truth", "promoted_next_day", "next_open_gap"}
    assert list(zip(saved.date, saved.code)) == list(zip(original.date, original.code))
    # Compute proxy through division, logs through ratios (different order of operations).
    proxy = saved.signal_amount.to_numpy(float) / saved.turnover_fraction.to_numpy(float)
    cap = saved.cap.to_numpy(float)
    known = np.isfinite(cap) & np.isfinite(proxy) & (cap > 0) & (proxy > 0)
    discrepancy = np.abs(cap[known] / proxy[known] - 1)
    np.testing.assert_allclose(saved.amount_turnover_proxy, proxy, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(saved.loc[known, "relative_disagreement"], discrepancy, rtol=1e-9, atol=1e-14)
    log_proxy = np.log(proxy[known])
    residual = np.log(cap[known] / proxy[known])
    assert np.max(np.abs(np.log(cap[known])-log_proxy-residual)) < 1e-11
    full = report["full_pool"]
    assert known.sum() == full["cap_and_proxy_comparable"] == 9102
    assert (~known).sum() == full["cap_unknown"] == 143
    for row in full["fixed_tolerance_counts"]:
        assert int((discrepancy <= row["tolerance"]).sum()) == row["within"]
        assert int((discrepancy > row["tolerance"]).sum()) == row["outside"]
    # Independent train selection without the helper used by producer.
    train_checks = []
    for t in report["training_only"]:
        cutoff = t["cutoff"]
        eligible = original[original.label_valid.fillna(False).astype(bool)
                            & original.target.notna() & original.date.lt(cutoff)
                            & original.target_date.lt(cutoff)]
        eligible_dates = sorted(set(eligible.date))[-252:]
        train = eligible[eligible.date.isin(eligible_dates)]
        keys = set(zip(train.date, train.code))
        expected = [key in keys for key in zip(saved.date, saved.code)]
        actual = saved["member_train_" + t["fold"]].tolist()
        assert actual == expected
        assert len(keys) == t["train_n"]
        assert len(eligible_dates) == t["training_dates"] == 252
        assert train.target_date.max() == t["training_max_label_date"]
        subset = saved.loc[expected]
        good = subset.cap.notna() & subset.amount_turnover_proxy.notna()
        assert good.sum() == t["summary"]["cap_and_proxy_comparable"]
        train_checks.append({"fold": t["fold"], "complete_train_n": len(train),
                             "known_cap_proxy": int(good.sum()), "unknown": int((~good).sum())})
    old_scope = saved[~saved.cap_bridge_appended]
    old_known = old_scope.cap.notna() & old_scope.amount_turnover_proxy.notna()
    assert len(old_scope) == 9063 and old_known.sum() == 8920
    assert saved.cap_bridge_appended.sum() == 182
    old_discrepancy = abs(old_scope.loc[old_known, "cap"] / old_scope.loc[old_known, "amount_turnover_proxy"] - 1)
    supplement = {"formed_at": datetime.now(timezone.utc).isoformat(),
        "original9063_denominator_including_unknowns": {"rows": len(old_scope),
            "known_cap_proxy": int(old_known.sum()), "unknown": int((~old_known).sum()),
            "fixed_tolerances": [{"tolerance": row["tolerance"],
                "within": int((old_discrepancy <= row["tolerance"]).sum()),
                "outside": int((old_discrepancy > row["tolerance"]).sum()),
                "unknown": int((~old_known).sum())} for row in full["fixed_tolerance_counts"]]},
        "whole_original_cap_missing_before_bridge": int(original.signal_float_cap.isna().sum()),
        "september_exact_bridge": 182, "full_pool_retained": len(saved),
        "train_checks": train_checks,
        "checks": ["all input hashes match", "output and producer script hashes match", "full candidate order and all unknowns retained",
                   "no labels or target dates in diagnostic ledger", "fixed numerical identities match independently",
                   "all fixed tolerance counts reproduce", "independent threefold member selection matches", "original9063 denominator includes143 unknowns"],
        "verification_script_sha256": sha(__file__), "new_fits": 0,
        "outcome_conditioned_analysis": False, "predictions_evaluated": False,
        "gain_verified": False}
    with (HERE / "independent_verification_and_full_denominator.json").open("x", encoding="utf-8") as handle:
        json.dump(supplement, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(supplement, indent=2))


if __name__ == "__main__":
    main()
