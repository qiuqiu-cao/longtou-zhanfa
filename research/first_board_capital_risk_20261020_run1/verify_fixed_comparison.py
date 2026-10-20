"""Read-only replay of the fixed capital-risk experiment; never fit."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_capital_risk_model as cap
import first_second_stage_model as stage
import first_board_leader_model as base
import pandas as pd
import numpy as np
import joblib

OUT = Path(__file__).resolve().parent
protocol = json.loads((OUT / "protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
report = json.loads((OUT / "report.json").read_text("utf-8"))
for group in ("inputs", "protected", "original_raw_hashes"):
    for path, digest in protocol[group].items():
        assert base.sha(path) == digest, (group, path)
assert base.sha(cap.__file__) == protocol["source_sha256"]
assert base.sha(OUT / "source_before_first_fit.py") == protocol["source_sha256"]
original, upper, scale = cap.load_rows()
assert scale == protocol["scale"]
facts = cap.capital_context(original, upper)
saved = pd.read_csv(OUT / "COMPLETE_FIRST_BOARD_FACTS_FEATURES_LABELS_AND_SCREEN.csv", dtype={"code":str}, float_precision="round_trip")
assert list(zip(facts.signal_date, facts.code)) == list(zip(saved.signal_date, saved.code))
np.testing.assert_allclose(facts[cap.FIELDS + [stage.TARGET]], saved[cap.FIELDS + [stage.TARGET]], equal_nan=True, atol=1e-12, rtol=1e-12)
# Independent screen expression, including unknowns and rejected positive rows.
thin = saved.float_cap.le(3e9) & saved.turnover.lt(5) & (saved.one_price.eq(1) | saved.volume_ratio_5.lt(1))
allowed = saved.float_cap.gt(0) & saved.float_cap.le(upper) & ~thin
assert allowed.tolist() == saved.research_selection_allowed.tolist()
assert saved.verified_restructuring_preclose.isna().all()
assert saved.verified_threeboard_risk_preclose.isna().all()
assert saved.threeboard_evidence_status.eq("INSUFFICIENT_EVIDENCE").all()

train = facts[facts.signal_date.lt("2025-01-01") & facts.label_end.lt("2025-01-01") & facts[stage.TARGET].notna()]
val = facts[facts.signal_date.between("2025-01-01", "2025-06-30")]
stored = pd.read_csv(OUT / "validation_CONTEXT_ALL_PREDICTIONS_BEFORE_METRICS.csv", dtype={"code":str}, float_precision="round_trip")
bundle = joblib.load(OUT / "models/validation_context_bundle.joblib")
replay = stage.predict(bundle, val)
np.testing.assert_allclose(replay[["model_score", "calibrated_score"]], stored[["model_score", "calibrated_score"]], equal_nan=True, atol=1e-12, rtol=1e-12)
assert bundle["features"] == cap.FIELDS and len(cap.FIELDS) == 85
assert bundle["train_max_label_end"] < val.signal_date.min()
cal = pd.read_csv(OUT / "models/validation_context_calibration_rows.csv", dtype={"code":str}, float_precision="round_trip")
receipt = json.loads((OUT / "models/validation_context_calibration_receipt.json").read_text("utf-8"))
prefix = joblib.load(OUT / "models/validation_context_calibration_prefix.joblib")
cal_x = facts.merge(cal[["signal_date", "code"]], on=["signal_date", "code"], validate="one_to_one")
np.testing.assert_allclose(prefix.predict_proba(cal_x[cap.FIELDS])[:,1], cal.uncalibrated_oof, atol=1e-12, rtol=1e-12)
core_checks = []
for name, expected in [("validation_context_calibration_prefix", train[train.label_end.lt(receipt["cutoff"])]), ("validation_context_full", train)]:
    info = json.loads((OUT / "models" / (name + "_COMPLETED.json")).read_text("utf-8"))
    assert base.sha(OUT / "models" / (name + ".joblib")) == info["sha256"]
    members = pd.read_csv(OUT / "models" / (name + "_members.csv"), dtype={"code":str}, float_precision="round_trip")
    assert list(zip(members.signal_date, members.code)) == list(zip(expected.signal_date, expected.code))
    np.testing.assert_allclose(members[stage.TARGET], expected[stage.TARGET], atol=0, rtol=0)
    if name.endswith("prefix"):
        assert members.label_end.max() < receipt["cutoff"]
    core_checks.append(dict(name=name, members=len(members), max_label_end=members.label_end.max()))
assert len(list((OUT / "models").glob("*_COMPLETED.json"))) == 2
assert len(list((OUT / "models").glob("2025-*.joblib"))) == 0

reference = cap.attach_frozen(facts, cap.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
metric_checks = {}
for name, rows in [("SCREEN_FROZEN", reference), ("CONTEXT_SCREEN", replay)]:
    picked = rows[rows.research_selection_allowed].sort_values(["signal_date", "model_score", "code"], ascending=[True,False,True]).groupby("signal_date", sort=False).head(1)
    m = report["decision"]["validation"][name]
    independent = dict(hits=int(picked[stage.TARGET].eq(1).sum()), known=int(picked[stage.TARGET].notna().sum()),
        selected=len(picked), false_positives=int(picked[stage.TARGET].eq(0).sum()),
        false_negatives=int(rows[stage.TARGET].eq(1).sum()-picked[stage.TARGET].eq(1).sum()),
        excluded_known_promoters=int((~rows.research_selection_allowed & rows[stage.TARGET].eq(1)).sum()))
    for key, value in independent.items():
        assert value == m[key], (name, key)
    metric_checks[name] = independent
assert report["decision"]["winner"] is None and report["goal_60_passed"] is False
bins = pd.read_csv(OUT / "fixed_cap_bins_DESCRIPTIVE_NOT_THRESHOLD_SEARCH.csv")
assert bins.rows.sum() == len(saved)
assert bins.known.sum() == saved[stage.TARGET].notna().sum()
assert bins.hits.sum() == saved[stage.TARGET].eq(1).sum()
base.write_json(OUT / "verification.json", dict(status="PASS", verified_at=base.now(), complete_rows=len(saved), validation_rows=len(val),
    exact_same_labels_and_original85_fields=True, independent_screen_equal=True, untouched_reference_hashes=True,
    two_saved_core_replays=core_checks, metrics_independently_recomputed=metric_checks,
    unknown_fund_identity_retained=len(saved), new_fits=0, production_deployment_allowed=False,
    original_labels_verified_by=str((cap.REFERENCE / "verification.json").resolve()),
    source_sha256=base.sha(cap.__file__), verifier_sha256=base.sha(__file__)))
print(json.dumps(dict(status="PASS", rows=len(saved), validation=len(val), metrics=metric_checks), ensure_ascii=True, indent=2))
