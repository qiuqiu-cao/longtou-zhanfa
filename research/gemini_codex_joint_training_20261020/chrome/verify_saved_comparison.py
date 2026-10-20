"""Replay saved models and independently settle the fixed validation; zero fits."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import first_board_co_limit_network as trial
import first_board_leader_model as base
import first_second_stage_model as stage

OUT = ROOT / "research/first_board_co_limit_network_20261020_run1"


def picks(frame, non_one=False):
    good = frame.research_selection_allowed & (frame.one_price.eq(0) if non_one else True)
    return frame[good].sort_values(["signal_date", "model_score", "code"], ascending=[True, False, True]).groupby("signal_date", sort=True).head(1)


def main():
    protocol = json.loads((OUT / "protocol_BEFORE_ANY_FIT.json").read_text("utf-8"))
    for path, digest in {**protocol["input_hashes"], **protocol["protected"], **protocol["raw_sources"]}.items():
        assert base.sha(path) == digest
    complete = pd.read_csv(OUT / "ALL_FIRST_BOARD_70_FIELDS_UNCHANGED_FACTS_LABELS_SCREEN.csv", dtype={"code": str}, float_precision="round_trip")
    scored = pd.read_csv(OUT / "VALIDATION_ALL_PREDICTIONS_BEFORE_METRICS.csv", dtype={"code": str}, float_precision="round_trip")
    expected = complete[complete.signal_date.between("2025-01-01", "2025-06-30")].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, scored[expected.columns], check_dtype=False, check_exact=True)
    bundle = joblib.load(OUT / "models/validation_network70_bundle.joblib")
    assert bundle["features"] == trial.FIELDS and len(bundle["features"]) == 70
    replay = stage.predict(bundle, scored)
    for field in ("model_score", "calibrated_score", "train_prior"):
        np.testing.assert_allclose(replay[field], scored[field], rtol=1e-12, atol=1e-12, equal_nan=True)
    for ending in ("calibration_prefix", "full"):
        stem = "validation_network70_" + ending
        model = joblib.load(OUT / "models" / (stem + ".joblib"))
        receipt = json.loads((OUT / "models" / (stem + "_COMPLETED.json")).read_text("utf-8"))
        assert base.sha(OUT / "models" / (stem + ".joblib")) == receipt["sha256"]
        params = model.named_steps["hist"].get_params()
        for key, value in dict(max_iter=160, max_leaf_nodes=15, min_samples_leaf=60, learning_rate=.06, l2_regularization=8., class_weight="balanced", random_state=42, early_stopping=False).items():
            assert params[key] == value
        assert receipt["features"] == trial.FIELDS
        members = pd.read_csv(OUT / "models" / (stem + "_members.csv"), dtype={"code": str}, float_precision="round_trip")
        original_name = "validation_S1_STAGE_CYCLE_GENE_" + ending + "_members.csv"
        old = pd.read_csv(trial.risk.REFERENCE / "models" / original_name, dtype={"code": str}, float_precision="round_trip")
        trial.same_members(members, old)
        assert members.label_end.lt("2025-01-01").all()
    frozen = trial.risk.attach_frozen(complete, trial.risk.REFERENCE / "validation_S1_STAGE_CYCLE_GENE_PREDICTIONS.csv")
    settlement = {}
    paired = []
    for non_one, name in [(False, "primary"), (True, "non_one_price")]:
        a = picks(scored, non_one).set_index("signal_date").sort_index()
        b = picks(frozen, non_one).set_index("signal_date").sort_index()
        assert a.index.equals(b.index) and len(a) == 117
        ah, bh = a[stage.TARGET].eq(1), b[stage.TARGET].eq(1)
        settlement[name] = dict(challenger_hits=int(ah.sum()), baseline_hits=int(bh.sum()), dates=len(a),
                                challenger_only_win=int((ah & ~bh).sum()), baseline_only_win=int((~ah & bh).sum()),
                                common_wins=int((ah & bh).sum()), common_failures=int((~ah & ~bh).sum()))
        for date in a.index:
            paired.append(dict(scope=name, signal_date=date, challenger_code=a.loc[date, "code"], baseline_code=b.loc[date, "code"],
                               challenger_outcome=a.loc[date, stage.TARGET], baseline_outcome=b.loc[date, stage.TARGET]))
    pd.DataFrame(paired).to_csv(OUT / "INDEPENDENT_COMPLETE_DAILY_PAIRED_TOP1.csv", index=False)
    assert settlement["primary"]["challenger_hits"] == settlement["primary"]["baseline_hits"] == 52
    assert settlement["non_one_price"]["challenger_hits"] == 46 and settlement["non_one_price"]["baseline_hits"] == 48
    report = json.loads((OUT / "report.json").read_text("utf-8"))
    assert not report["validation"]["pass_validation"] and not report["goal_60_passed"]
    assert report["core_fit_count"] == 2 and report["logical_heads"] == 1
    assert not list((OUT / "models").glob("2025-*.joblib")) and not list((OUT / "models").glob("2026-*.joblib"))
    status = dict(at=datetime.now(timezone.utc).isoformat(), status="PASS", replay_zero_fits=True, settlement=settlement,
                  complete_first_rows=len(complete), first_unknown=int(complete[stage.TARGET].isna().sum()),
                  validation_rows=len(scored), validation_known=int(scored[stage.TARGET].notna().sum()),
                  validation_unknown=int(scored[stage.TARGET].isna().sum()), validation_positive=int(scored[stage.TARGET].eq(1).sum()),
                  logical_challenge_heads=1, core_hgb_fits=2, monthly_fits=0, baseline_refits=0, second_stage_refits=0,
                  protected_hashes_unchanged=True, exact70_fields=True, original_training_members_and_labels_unchanged=True,
                  concluded="REJECTED: primary tied baseline; non-one-price declined; no monthly evaluation or deployment")
    base.write_json(OUT / "INDEPENDENT_SAVED_MODEL_VERIFICATION.json", status)
    print(json.dumps(status, ensure_ascii=True))


if __name__ == "__main__":
    main()
