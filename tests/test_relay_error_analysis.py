import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import relay_error_analysis as r
import model_upgrade_history as h

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/model.joblib"
FEATURE = ROOT/"results/review_20261008_forecast_20261009_20261008T091618Z/model/features/features.json"
POLICY = ROOT/"research/model_upgrade_validation/relay_error_20261009T045526Z/reader_compatibility_BEFORE_REPLAY.json"


def frame():
    return pd.DataFrame([row["features"] for row in json.loads(FEATURE.read_text(encoding="utf-8"))["relay"]])


def test_explanation_independently_reconstructs_all_original_scores():
    model = joblib.load(BASE); x = frame()
    _, terms, logits, raw, _, _, intercept = r.components(model, x, r.RELAY_FIELDS)
    decision = model.named_steps["logisticregression"].decision_function(model[:-1].transform(x[r.RELAY_FIELDS]))
    np.testing.assert_allclose(decision, logits, atol=1e-12, rtol=0)
    np.testing.assert_allclose(expit(intercept + np.sum(terms, axis=1)), raw, atol=1e-12, rtol=0)


def test_future_truth_and_next_day_prices_do_not_change_explanation():
    model = joblib.load(BASE); x = frame(); original = r.components(model, x, r.RELAY_FIELDS)
    x["target"] = 1; x["trade_high"] = 99999; x["next_session_turnover"] = -999
    changed = r.components(model, x, r.RELAY_FIELDS)
    np.testing.assert_array_equal(original[1], changed[1]); np.testing.assert_array_equal(original[3], changed[3])


def test_unknown_and_nonfinite_inputs_are_not_explained_as_known():
    model = joblib.load(BASE); x = frame()
    x.loc[0, r.RELAY_FIELDS] = np.nan
    with pytest.raises(ValueError, match="unknown"):
        r.components(model, x, r.RELAY_FIELDS)
    x = frame(); x.loc[0, "turnover"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        r.components(model, x, r.RELAY_FIELDS)


def test_model_serialization_keeps_original_scores(tmp_path):
    model = joblib.load(BASE); dest = tmp_path/"model.joblib"; joblib.dump(model, dest)
    np.testing.assert_array_equal(model.predict_proba(frame()[r.RELAY_FIELDS]), joblib.load(dest).predict_proba(frame()[r.RELAY_FIELDS]))


def test_future_labels_cannot_enter_past_training_members():
    full = pd.DataFrame({"date": ["2026-08-28", "2026-08-31", "2026-09-01"],
                         "target_date": ["2026-08-31", "2026-09-01", "2026-09-02"],
                         "label_valid": [True]*3, "target": [1, 0, 1]})
    old = h.training_rows(full, "2026-09-01", "relay")
    full.loc[1:, "target"] = [1, 0]
    pd.testing.assert_frame_equal(old, h.training_rows(full, "2026-09-01", "relay"))
    assert old.target_date.max() == "2026-08-31"


def test_replay_adapter_keeps_raw_features_and_global_verifier_unchanged():
    import model_upgrade_inputs as inputs
    old_verifier = inputs.verify_protocol
    rebuilt = r.replay_feature_archive(FEATURE.parent, "2026-10-08T09:23:04.917365+00:00", POLICY)
    saved = json.loads(FEATURE.read_text(encoding="utf-8"))
    assert {k: rebuilt[k] for k in saved} == saved
    assert inputs.verify_protocol is old_verifier


def test_replay_adapter_rejects_unapproved_current_hash(tmp_path):
    policy = json.loads(POLICY.read_text(encoding="utf-8")); policy["current_sha256"] = "0"*64
    path = tmp_path/"invalid_policy.json"; path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="hash conflict"):
        r.replay_feature_archive(FEATURE.parent, "2026-10-08T09:23:04.917365+00:00", path)


def test_hash_conflicts_and_repeated_outputs_are_refused(tmp_path):
    path = tmp_path/"input.json"; r.write_new(path, {"valid": True}); digest = r.sha(path)
    with pytest.raises(FileExistsError):
        r.write_new(path, {"valid": False})
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash conflict"):
        r.verify_sources({str(path): digest})
