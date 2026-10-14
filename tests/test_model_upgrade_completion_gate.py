"""Three synthetic negative completion cases; no real data or fitting."""
import json

import pytest

import model_upgrade_research as research


def _bundle(tmp_path, module):
    directory = tmp_path / (module + "_history")
    directory.mkdir()
    predictions = directory / "predictions.csv"
    predictions.write_text("arm,prediction\nBASE,SYNTHETIC\n", encoding="utf-8")
    prediction_sha = research.sha(predictions)
    protocol = {"protocol_sha256": "a" * 64, "folds": [{"id": "SYNTHETIC"}]}
    report = {"protocol_sha256": protocol["protocol_sha256"],
              "predictions_sha256": prediction_sha}
    checks = [{"arm": arm, "confusion_and_scored_count_match": True,
               "candidate_raw_Brier_match": True if module == "relay" else None}
              for arm in ("BASE", "C1", "C2")]
    checks.extend({"arm": arm, "identity": "actual_one_word_daily" if module == "relay" else None,
                   "paired_dates": 1, "primary_difference": 0.0}
                  for arm in ("C1", "C2"))
    if module == "relay":
        checks.extend({"arm": arm, "identity": "one_word_proxy", "paired_dates": 1,
                       "primary_difference": 0.0} for arm in ("C1", "C2"))
    metrics = {"status": "HISTORICAL_DIAGNOSTIC_NO_CORE_UPGRADE", "module": module,
               "prediction_sha256": prediction_sha,
               "folds": {"SYNTHETIC": {"independent_recomputation": checks}}}
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return protocol, predictions, metrics_path, metrics


def test_missing_independent_history_metrics_rejects_completion(tmp_path):
    protocol, _, metrics_path, _ = _bundle(tmp_path, "market")
    metrics_path.unlink()
    with pytest.raises(FileNotFoundError):
        research.require_history_metrics("market", tmp_path, protocol)


def test_false_relay_brier_recomputation_rejects_completion(tmp_path):
    protocol, _, metrics_path, metrics = _bundle(tmp_path, "relay")
    metrics["folds"]["SYNTHETIC"]["independent_recomputation"][1]["candidate_raw_Brier_match"] = False
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    before = metrics_path.read_bytes()
    with pytest.raises(ValueError, match="reconstruction did not complete"):
        research.require_history_metrics("relay", tmp_path, protocol)
    assert metrics_path.read_bytes() == before


def test_prediction_sha_mutation_rejects_completion(tmp_path):
    protocol, predictions, metrics_path, _ = _bundle(tmp_path, "market")
    predictions.write_text("arm,prediction\nBASE,CHANGED_AFTER_METRICS\n", encoding="utf-8")
    before = metrics_path.read_bytes()
    with pytest.raises(ValueError, match="independently reconstructed metrics required"):
        research.require_history_metrics("market", tmp_path, protocol)
    assert metrics_path.read_bytes() == before
