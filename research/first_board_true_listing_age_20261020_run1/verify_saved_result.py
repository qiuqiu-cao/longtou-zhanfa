"""Independent saved-model replay, whole-pool reconciliation and daily settlement; no fits."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import first_board_true_listing_age as trial
import first_board_leader_model as base
import first_second_stage_model as stage
import joblib
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def selections(frame, non_one=False):
    mask = frame.research_selection_allowed & (frame.one_price.eq(0) if non_one else True)
    return frame[mask].sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).groupby("signal_date", sort=True).head(1)


def main():
    protocol = json.loads((OUT / "protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
    assert base.sha(trial.__file__) == protocol["source_sha256"]
    for name, digest in {**protocol["inputs"], **protocol["protected"], **protocol["raw_sources"]}.items():
        assert base.sha(name) == digest
    rows = pd.read_csv(OUT / "COMPLETE_FIRST_BOARD_UNCHANGED_FACTS_LABELS_AND_ACTUAL_AGE.csv", dtype={"code": str}, float_precision="round_trip")
    source = pd.read_csv(trial.SOURCE, dtype={"code": str}, float_precision="round_trip")
    source = source[source.stage.eq(1)].reset_index(drop=True)
    columns = trial.KEYS + ["stage", "label_end"] + trial.ORIGINAL_FIELDS + [stage.TARGET]
    pd.testing.assert_frame_equal(rows[columns], source[columns], check_dtype=False, check_exact=True)
    extra = pd.read_csv(trial.QUAL / "COMPLETE_FIRST_BOARD_ACTUAL_AGE_NO_OUTCOMES.csv", dtype={"code": str}, float_precision="round_trip")
    expected = rows[trial.KEYS].merge(extra[trial.KEYS + [trial.NEW]], on=trial.KEYS, validate="one_to_one")
    np.testing.assert_allclose(expected[trial.NEW], rows[trial.NEW], atol=0, rtol=0)
    scored = pd.read_csv(OUT / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", dtype={"code": str}, float_precision="round_trip")
    val = rows[rows.signal_date.between("2025-01-01", "2025-06-30")].reset_index(drop=True)
    pd.testing.assert_frame_equal(scored[val.columns], val, check_dtype=False, check_exact=True)
    head = joblib.load(OUT / "models/validation_actual_listing_age_bundle.joblib")
    assert len(head["features"]) == 62 and head["features"] == trial.FIELDS
    assert trial.OLD not in head["features"] and trial.NEW in head["features"]
    replay = stage.predict(head, val)
    for field in ("model_score", "calibrated_score", "train_prior"):
        np.testing.assert_allclose(replay[field], scored[field], atol=1e-12, rtol=1e-12, equal_nan=True)
    params_all = {}
    for suffix in ("calibration_prefix", "full"):
        stem = "validation_actual_listing_age_" + suffix
        core = joblib.load(OUT / "models" / (stem + ".joblib"))
        receipt = json.loads((OUT / "models" / (stem + "_COMPLETED.json")).read_text("utf-8"))
        assert base.sha(OUT / "models" / (stem + ".joblib")) == receipt["sha256"]
        params = core.named_steps["hist"].get_params()
        for key, value in dict(max_iter=160, max_leaf_nodes=15, min_samples_leaf=60, learning_rate=.06, l2_regularization=8., class_weight="balanced", random_state=42, early_stopping=False).items():
            assert params[key] == value
        assert receipt["features"] == trial.FIELDS
        train = pd.read_csv(OUT / "models" / (stem + "_members.csv"), dtype={"code": str}, float_precision="round_trip")
        old = pd.read_csv(trial.REFERENCE / "models" / ("validation_S1_STAGE_CYCLE_GENE_" + suffix + "_members.csv"), dtype={"code": str}, float_precision="round_trip")
        order = trial.KEYS + ["stage", "label_end", stage.TARGET]
        pd.testing.assert_frame_equal(train[order].sort_values(trial.KEYS).reset_index(drop=True), old[order].sort_values(trial.KEYS).reset_index(drop=True), check_dtype=False, check_exact=True)
        assert train.label_end.lt("2025-01-01").all()
        params_all[suffix] = {key: params[key] for key in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization", "class_weight", "random_state", "early_stopping")}
    baseline = trial.risk.attach_frozen(rows, trial.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    merged = scored[trial.KEYS + ["label_end", stage.TARGET, "one_price", "research_selection_allowed", trial.OLD, trial.NEW, "model_score", "calibrated_score"]].merge(
        baseline[trial.KEYS + ["model_score", "calibrated_score"]].rename(columns={"model_score": "baseline_raw_score", "calibrated_score": "baseline_calibrated_score"}), on=trial.KEYS, validate="one_to_one")
    merged.to_csv(OUT / "AUDIT_COMPLETE4724_SCORE_COMPARISON_WITH_UNKNOWN.csv", index=False)
    pairs, settlement = [], {}
    for non, scope in [(False, "primary"), (True, "non_one_price")]:
        a, b = selections(scored, non).set_index("signal_date"), selections(baseline, non).set_index("signal_date")
        assert a.index.equals(b.index) and len(a) == 117
        known_a, known_b = a[stage.TARGET].notna(), b[stage.TARGET].notna()
        ah, bh = a[stage.TARGET].eq(1), b[stage.TARGET].eq(1)
        settlement[scope] = dict(dates=len(a), challenger_hits=int(ah.sum()), baseline_hits=int(bh.sum()),
                                challenger_known=int(known_a.sum()), baseline_known=int(known_b.sum()),
                                common_wins=int((ah & bh).sum()), challenger_only_win=int((ah & ~bh & known_b).sum()),
                                baseline_only_win=int((~ah & bh & known_a).sum()), common_failures=int((~ah & ~bh & known_a & known_b).sum()))
        for day in a.index:
            pairs.append(dict(scope=scope, date=day, challenger_code=a.loc[day, "code"], baseline_code=b.loc[day, "code"],
                              challenger_outcome=a.loc[day, stage.TARGET], baseline_outcome=b.loc[day, stage.TARGET]))
    pd.DataFrame(pairs).to_csv(OUT / "INDEPENDENT_ALL117_DAILY_PAIRED_SELECTIONS.csv", index=False)
    assert settlement["primary"]["challenger_hits"] == 48 and settlement["primary"]["baseline_hits"] == 52
    assert settlement["non_one_price"]["challenger_hits"] == 43 and settlement["non_one_price"]["baseline_hits"] == 48
    report = json.loads((OUT / "report.json").read_text("utf-8"))
    assert report["core_fit_count"] == 2 and not report["decision"]["pass_validation"] and not report["goal_60_passed"]
    assert not list((OUT / "models").glob("2025-*.joblib")) and not list((OUT / "models").glob("2026-*.joblib"))
    result = dict(at=datetime.now(timezone.utc).isoformat(), status="PASS", verifier_fits=0, all_raw_source_and_protected_hashes_unchanged=True,
                  all28503_original_fields_and_labels_unchanged=True, all4724_scores_replayed=True, feature_count=62,
                  removed_from_estimator_only=trial.OLD, new_feature_only=trial.NEW, actual_age_coverage=len(rows[rows[trial.NEW].notna()]),
                  validation_rows=len(scored), validation_known=int(scored[stage.TARGET].notna().sum()), validation_unknown=int(scored[stage.TARGET].isna().sum()),
                  first_unknown=int(rows[stage.TARGET].isna().sum()), original_full_and_prefix_training_members_unchanged=True,
                  original_parameters=params_all, settlement=settlement, core_fits=2, temporal_calibrator_fits=1, monthly_fits=0, baseline_refits=0, second_stage_refits=0,
                  interpretation="REJECTED:primary and non-one-price validation both declined;not a60%achievement,no deployment")
    base.write_json(OUT / "INDEPENDENT_SAVED_MODEL_VERIFICATION.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
