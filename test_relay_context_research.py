import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import relay_context_research as r
import relay_context_shadow as s
from model_prediction_presentation import render_tendencies, model_call_label


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.sessions = ["2026-09-28", "2026-09-29", "2026-09-30"]
        self.relay = pd.DataFrame([{"date": "2026-09-30", "code": code,
            "signal_amount_same_height_percentile": rank, "turnover_same_height_percentile": 1 - rank,
            "target": target, "predicted": None} for code, rank, target in [("000002", .25, 1), ("000001", .75, 0)]])
        self.market = pd.DataFrame([{"date": day, "data_status": "verified", **{k: value for k in r.LEVELS}}
            for day, value in zip(self.sessions, [1., 2., 4.])])

    def test_exact_official_previous(self):
        full = r.context_features(self.relay, self.market, self.sessions)
        self.assertEqual(full.iloc[0]["height_change"], 2.)
        self.assertEqual(full.iloc[0]["context_previous_official_date"], "2026-09-29")

    def test_missing_previous_not_last_available(self):
        full = r.context_features(self.relay, self.market.drop(index=1), self.sessions)
        self.assertTrue(full[r.ENV[1:]].isna().all().all())
        self.assertEqual(full.iloc[0].yesterday_premium, 4.)

    def test_nonverified_source_nan(self):
        self.market.loc[2, "data_status"] = "partial"
        full = r.context_features(self.relay, self.market, self.sessions)
        self.assertTrue(full[r.ENV].isna().all().all())

    def test_future_label_and_future_market_do_not_change_features(self):
        expected = r.context_features(self.relay, self.market, self.sessions)
        modified = self.relay.copy(); modified["target"] = [999, -888]
        later = pd.DataFrame([{"date": "2026-10-08", "data_status": "verified", **{k: 99999 for k in r.LEVELS}}])
        actual = r.context_features(modified, pd.concat([self.market, later]), self.sessions + ["2026-10-08"])
        pd.testing.assert_frame_equal(expected[r.ENV + r.INTERACTIONS], actual[r.ENV + r.INTERACTIONS])

    def test_original_full_group_percentiles_not_selected_group(self):
        selected = r.context_features(self.relay.iloc[:1], self.market, self.sessions)
        self.assertEqual(selected.iloc[0][r.RANKS[0]], .25)
        self.assertEqual(selected.iloc[0][r.INTERACTIONS[0]], -1.)

    def test_unknown_percentile_not_zero(self):
        self.relay.loc[0, r.RANKS[0]] = np.nan
        full = r.context_features(self.relay, self.market, self.sessions)
        self.assertTrue(full.iloc[0][r.INTERACTIONS[:7]].isna().all())

    def test_duplicate_members_rejected(self):
        with self.assertRaises(ValueError):
            r.context_features(pd.concat([self.relay, self.relay]), self.market, self.sessions)

    def test_training_cutoff_excludes_future_mature_labels(self):
        frame = pd.DataFrame({"date": ["2026-08-28", "2026-08-31"], "target_date": ["2026-08-31", "2026-09-01"],
                              "target": [1., 0.], "label_valid": [True, True]})
        train = r.history.training_rows(frame, "2026-09-01", "relay")
        self.assertEqual(list(train.date), ["2026-08-28"])

    def test_convergence_warning_is_failure(self):
        from sklearn.exceptions import ConvergenceWarning
        class Estimator:
            def fit(self, *a, **kw):
                import warnings
                warnings.warn("not converged", ConvergenceWarning)
        frame = pd.DataFrame([{**{k: 1. for k in r.ARMS["E1"]}, "date": "2026-08-28", "target": target} for target in (0., 1.)])
        with patch("rolling_forecast_challenger_v274.make", return_value=Estimator()):
            with self.assertRaises(ConvergenceWarning): r.fit_once(frame, "E1")

    def test_tie_sort_and_negative_not_positive_first(self):
        rows = [{"signal_date": "2026-09-30", "code": code, "raw_score": .4, "prediction": False,
                 "truth": True, "actual_one_word_daily": 0., "boards": 2} for code in ["000002", "000001"]]
        result = r.score(rows, self.sessions, {"start": "2026-09-30", "end": "2026-09-30"})
        self.assertEqual(result["true_nonword_rank_first"]["daily_rows"][0]["selected_code"], "000001")
        self.assertEqual(result["positive_nonword_first"]["selected_days"], 0)

    def test_metrics_independently_include_unknown_registered_and_positives(self):
        rows = [{"signal_date": "2026-09-30", "truth": a, "prediction": b} for a, b in
                [(True, True), (True, False), (False, False), (False, True), (True, None), (None, True)]]
        result = r.metrics(rows)
        self.assertEqual(result["confusion"], {"TP": 1, "TN": 1, "FP": 1, "FN": 1})
        self.assertEqual(result["coverage"], 4 / 6)
        self.assertEqual(result["complete_positive_Recall"], 1 / 3)

    def test_weighted_paired_interval_keeps_official_empty_days(self):
        base = [{"signal_date": "2026-09-30", "code": str(i), "prediction": False, "truth": True} for i in range(10)]
        challenge = [{**x, "prediction": True} for x in base]
        self.assertEqual(s.paired_accuracy_interval(base, challenge, self.sessions), [1., 1.])

    def test_old_render_byte_identical_and_new_only_prediction_column(self):
        text = "【六、连板晋级预测】\n| 000001 股票 | 2 | 候选 | 断板 | 未知 | 证据 |\n【十一、结算】\n实际断板\n"
        self.assertEqual(render_tendencies(text, {"recorded_at": "2026-10-08T12:00:00+00:00"}), text)
        new = render_tendencies(text, {"recorded_at": "2026-10-12T12:00:00+00:00"})
        self.assertIn("模型偏断板", new); self.assertIn("实际断板", new)
        self.assertEqual(model_call_label("PROMOTE"), "模型偏晋级")

    def test_immutable_artifacts_and_hash_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.json"; r.write_new(path, {"x": 1})
            with self.assertRaises(FileExistsError): r.write_new(path, {"x": 2})
            with self.assertRaises(ValueError): r.verify({str(path): "0" * 64})

    def test_unknown_true_identity_separate_reason(self):
        record = {"signal_date": self.sessions[-1], "target_date": "2026-10-08", "original_codes": ["000001"],
            "predictions": {arm: [{"code": "000001", "prediction": True, "raw_score": .6, "actual_one_word_daily": None}]
                            for arm in ["BASE", "E1"]}}
        rows, reasons = s.paired_day_rows([record], {self.sessions[-1]: {"000001": True}}, self.sessions)
        self.assertEqual(reasons[self.sessions[-1]], "UNKNOWN_TRUE_ONEWORD_IDENTITY")
        self.assertEqual(len(rows["BASE"]), 1)

    def synthetic_window(self):
        days = [x.date().isoformat() for x in pd.date_range("2025-01-01", periods=120)]
        rows = {arm: [] for arm in ["BASE", "E1", "E2"]}
        for day in days:
            for i, (truth, base_call, base_score) in enumerate([(True, True, .8), (False, True, .7), (True, False, .3), (False, False, .2)]):
                for arm in rows:
                    call = base_call if arm == "BASE" else truth
                    raw = base_score if arm == "BASE" else [.8, .2, .7, .1][i]
                    rows[arm].append({"signal_date": day, "target_date": day, "code": str(i).zfill(6), "boards": 2,
                                      "prediction": call, "truth": truth, "raw_score": raw, "actual_one_word_daily": 0.})
        return days, rows, {day: "COMPLETE_PAIRED" for day in days}

    def test_fixed120_gate_and_tie_prefers_smaller_e1(self):
        days, rows, reasons = self.synthetic_window()
        result = s.assess(rows, reasons, days, days)
        self.assertTrue(result["challenges"]["E1"]["passed"])
        self.assertEqual(result["winner"], "E1")
        self.assertEqual(result["challenges"]["E1"]["Accuracy_delta"], .5)

    def test_negative_third40_cannot_pass_despite_overall_improvement(self):
        days, rows, reasons = self.synthetic_window()
        for arm in ["E1", "E2"]:
            for row in rows[arm]:
                if row["signal_date"] in days[80:]: row["prediction"] = not row["truth"]
        result = s.assess(rows, reasons, days, days)
        self.assertGreater(result["challenges"]["E1"]["Accuracy_delta"], .03)
        self.assertFalse(result["challenges"]["E1"]["guards"]["three40"])
        self.assertIsNone(result["winner"])

    def test_missing_official_dates_prevent_full120_acceptance(self):
        days, rows, reasons = self.synthetic_window()
        result = s.assess(rows, reasons, days[:119], days)
        self.assertFalse(result["challenges"]["E1"]["passed"])


if __name__ == "__main__": unittest.main()
