"""Root API integration on synthetic files/data; no real fitting or signals."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import model_upgrade_history as history
import model_upgrade_inputs as inputs
import model_upgrade_research as research


class SyntheticEstimator:
    classes_ = np.asarray(["CONTINUE", "IMPROVE", "WORSEN"])

    def __init__(self, training_mean):
        self.training_mean = training_mean

    def predict_proba(self, frame):
        a = 1 / (1 + np.exp(-(frame["promotion_mid"].to_numpy() - self.training_mean)))
        return np.column_stack([.2 + .2*a, .6 - .2*a, np.full(len(frame), .2)])


class UpgradeIntegrationSynthetic(unittest.TestCase):
    def test_receipt_prepared_time_can_change_but_source_hash_cannot(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp), Path(temp) / "results"
            names = ["model_upgrade_research.py", "model_upgrade_history.py", "model_upgrade_inputs.py",
                     "model_upgrade_shadow.py", "model_upgrade_metrics.py", "model_upgrade_apply.py"]
            for name in names:
                (root / name).write_text("synthetic module\n", encoding="utf-8")
            source = root / "synthetic_source.csv"
            source.write_text("x\n1\n", encoding="utf-8")
            prepared = {"provenance": {"prepared_at": "2026-10-07T10:00:00+00:00",
                "source_hashes": {str(source): history.sha(source)}, "dependency_hashes": {},
                "training_inputs": {"market": "same immutable synthetic digest"}}}
            protocol = {"protocol_sha256": "a"*64, "protected_sha256": {}}
            with patch.object(research, "ROOT", root):
                research.implementation_receipt(out, prepared, protocol)
                original_receipt_hash = history.sha(out / "provenance_BEFORE_FITS.json")
                later = copy.deepcopy(prepared)
                later["provenance"]["prepared_at"] = "2026-10-07T11:00:00+00:00"
                research.implementation_receipt(out, later, copy.deepcopy(protocol))
                self.assertEqual(later["provenance"]["prepared_at"], prepared["provenance"]["prepared_at"])
                self.assertEqual(history.sha(out / "provenance_BEFORE_FITS.json"), original_receipt_hash)
                source.write_text("x\n2\n", encoding="utf-8")
                changed = copy.deepcopy(later)
                changed["provenance"]["source_hashes"][str(source)] = history.sha(source)
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    research.implementation_receipt(out, changed, copy.deepcopy(protocol))

    def _fold_and_logs(self):
        fold = {"id": "SYNTHETIC", "start": "2026-09-24", "end": "2026-09-30"}
        logs = [{"fold": fold, "arm": "BASE", "training_majority": {"class": "CONTINUE"}}]
        return fold, logs

    def test_market_converter_official_grid_and_missing_previous_stay_unknown(self):
        sessions = ["2026-09-24", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08"]
        market = pd.DataFrame({"date": [sessions[i] for i in [0, 2, 3]],
                               "emotion_quality_score": [.2, .3, .6]})
        prepared = {"market": market, "provenance": {"sessions": sessions}}
        predictions = pd.DataFrame([{ "fold": "SYNTHETIC", "date": sessions[2], "arm": arm,
            "status": "PREDICTED_UNCALIBRATED", "predicted": "CONTINUE", "label_valid": False,
            "target": None, "raw_score_IMPROVE": .2, "raw_score_CONTINUE": .6, "raw_score_WORSEN": .2}
            for arm in history.ARMS])
        fold, logs = self._fold_and_logs()
        records = research.historical_records("market", predictions, prepared, fold, logs)
        self.assertEqual([r["signal_date"] for r in records], sessions[:4])
        self.assertEqual(records[0]["target_date"], "2026-09-28")
        self.assertEqual(records[1]["arms"]["BASE"]["status"], "UNKNOWN")
        self.assertIsNone(records[2]["naive"]["persistence"])
        self.assertIsNone(records[2]["outcomes"]["market_target"])
        self.assertEqual(records[3]["naive"]["persistence"], "IMPROVE")
        self.assertTrue(all(r["forward"] is False for r in records))

    def test_relay_converter_full_group_separates_true_ohlc_and_legacy_proxy(self):
        sessions = ["2026-09-24", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08"]
        rows = []
        for arm in history.ARMS:
            for code, boards, word, legacy, target, valid in [
                    ("000001", 3, 1., 0., 0., True),
                    ("000002", 2, 0., 0., 1., True),
                    ("000003", 2, np.nan, 1., np.nan, False)]:
                rows.append({"fold": "SYNTHETIC", "date": sessions[0], "arm": arm, "code": code,
                    "boards": boards, "actual_one_word_daily": word, "one_word_proxy": legacy,
                    "target": target, "label_valid": valid, "status": "PREDICTED_UNCALIBRATED",
                    "raw_score": .6, "predicted": 1})
        predictions = pd.DataFrame(rows)
        prepared = {"market": pd.DataFrame({"date": [], "emotion_quality_score": []}),
                    "provenance": {"sessions": sessions}}
        fold, logs = self._fold_and_logs()
        records = research.historical_records("relay", predictions, prepared, fold, logs)
        first = records[0]
        self.assertEqual(first["original_codes"], ["000001", "000002", "000003"])
        self.assertEqual(first["arms"]["BASE"]["candidates"][0]["actual_one_word_daily"], 1)
        self.assertIsNone(first["outcomes"]["candidates"]["000003"])
        self.assertIsNone(first["naive"]["highest_nonword"])
        proxy = research.historical_records("relay", predictions, prepared, fold, logs, True)
        self.assertEqual(proxy[0]["arms"]["BASE"]["candidates"][0]["actual_one_word_daily"], 0)
        self.assertEqual(proxy[0]["naive"]["highest_nonword"], "000001")
        incomplete = predictions[~(predictions.arm.eq("C2") & predictions.code.eq("000003"))]
        with self.assertRaisesRegex(ValueError, "lost full original group"):
            research.historical_records("relay", incomplete, prepared, fold, logs)

    def test_feature_formation_before_knowledge_cutoff_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            payload = {"sources": {}, "market": {"status": "verified"}}
            (path / "features.json").write_text(json.dumps(payload), encoding="utf-8")
            manifest = {"version": inputs.VERSION, "features_sha256": history.sha(path / "features.json"),
                "source_hashes": {}, "formed_at": "2026-10-08T15:10:00+08:00",
                "knowledge_cutoff": "2026-10-08T15:20:00+08:00", "protocol_path": "unused",
                "source_archive": "unused", "calendar_path": "unused", "history_archives": []}
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(inputs, "_derive", return_value=payload) as derive:
                with self.assertRaisesRegex(ValueError, "formation precedes"):
                    inputs.load_feature_archive(path, "2026-10-08T15:30:00+08:00")
                derive.assert_not_called()

    def test_synthetic_estimator_save_load_and_future_mutation_contract(self):
        sessions = pd.bdate_range("2026-08-03", periods=30).strftime("%Y-%m-%d").tolist()
        n = len(sessions)-1
        frame = pd.DataFrame({"date": sessions[:-1], "target_date": sessions[1:],
            "target": np.resize(["CONTINUE", "IMPROVE", "WORSEN"], n), "label_valid": True,
            "promotion_mid": np.linspace(.7, .3, n), "failed_rate": np.linspace(.1, .2, n),
            "yesterday_premium": -.01})
        signal, cutoff = sessions[15], sessions[16]
        fields = ["promotion_mid", "failed_rate", "yesterday_premium"]
        original = history.interactions(frame, "market", sessions)
        train = history.training_rows(original, cutoff, "market")
        self.assertLess(train.target_date.max(), cutoff)
        changed = frame.copy()
        future = changed.date.gt(signal)
        changed.loc[future, fields] = 1e8
        changed.loc[future, "target"] = "HINDSIGHT_ONLY"
        revised = history.interactions(changed, "market", sessions)
        revised_train = history.training_rows(revised, cutoff, "market")
        pd.testing.assert_frame_equal(train, revised_train)
        pd.testing.assert_frame_equal(original[original.date.le(signal)], revised[revised.date.le(signal)])
        estimator = SyntheticEstimator(float(train.promotion_mid.mean()))
        current = original[original.date.eq(signal)]
        before = history._predict(estimator, current, fields, "market")
        with tempfile.TemporaryDirectory() as temp:
            prepared = {"provenance": {"source_hashes": {}, "dependency_hashes": {}}}
            manifest = history._save_model(Path(temp) / "BASE", estimator, fields, train, "market", "BASE",
                                           prepared, "a"*64, cutoff)
            loaded = joblib.load(manifest["model_path"])
            after = history._predict(loaded, revised[revised.date.eq(signal)], fields, "market")
            pd.testing.assert_frame_equal(before, after)
            self.assertEqual(manifest["training_max_label_date"], train.target_date.max())
            self.assertEqual(manifest["sha256"], history.sha(manifest["model_path"]))


if __name__ == "__main__":
    unittest.main()
