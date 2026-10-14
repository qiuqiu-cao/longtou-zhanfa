from pathlib import Path
import json

import pytest

import model_champion_compat as adapter
import model_champion_runtime as native
from model_upgrade_research import read

ROOT = Path(__file__).resolve().parents[1]
COMPAT = ROOT/"research/model_upgrade_validation/shadow_compat_20261009T093237Z"
POLICY = COMPAT/"compatibility_BEFORE_USE.json"
IMPLEMENTATION = COMPAT/"champion_implementation_BEFORE_USE.json"
FIXED = ROOT/"research/model_upgrade_validation/fixed_20261007T112017Z"
FEATURES = ROOT/"results/review_20261008_forecast_20261009_20261008T091618Z/model/features"
BASE = read(FEATURES.parent/"base_predictions.json")


@pytest.fixture(scope="module")
def isolated():
    return adapter.runtime(POLICY, IMPLEMENTATION)[0]


def test_exact_original_selector_and_acceptance_code_are_used_in_local_namespace(isolated):
    assert isolated.predict.__code__ is native.predict.__code__
    assert isolated._forecast.__code__ is native._forecast.__code__
    assert isolated._prove_active.__code__ is native._prove_active.__code__
    assert isolated.choose_champion.__code__ is native.choose_champion.__code__
    assert isolated.shadow is not native.shadow
    assert isolated.load_feature_archive is not native.load_feature_archive
    assert native.shadow.__name__ == "model_upgrade_shadow"


def test_saved_input_replay_preserves_all_ten_original_scores_and_order(isolated):
    pack = isolated.load_feature_archive(FEATURES, BASE["knowledge_cutoff"])
    result = isolated._forecast("relay", BASE["model_bindings"]["relay"], pack)
    expected = {r["code"]: r for r in BASE["relay"]["candidates"]}
    assert len(result["candidates"]) == len(expected) == 10
    for candidate in result["candidates"]:
        old = expected[candidate["code"]]
        assert candidate["raw_score"] == pytest.approx(old["raw_score"], rel=0, abs=1e-12)
        assert candidate["predicted_positive"] is old["predicted_positive"]
    assert result["ranking"] == [r["code"] for r in sorted(expected.values(), key=lambda r: (-r["raw_score"], r["code"]))]


def test_compatibility_does_not_allow_a_past_signal_to_be_registered_as_current(isolated):
    with pytest.raises(ValueError, match="true current Shanghai date"):
        isolated.predict(FIXED/"protocol_BEFORE_EFFECTS.json", "relay", FEATURES)


@pytest.mark.parametrize("key,value", [("numeric_refits", 1), ("selection_rules_changed", True),
                                       ("formed_at", "2099-01-01T00:00:00+00:00")])
def test_scope_change_and_future_receipts_are_rejected(tmp_path, key, value):
    receipt = read(IMPLEMENTATION); receipt[key] = value
    path = tmp_path/"receipt.json"; path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        adapter.runtime(POLICY, path)
