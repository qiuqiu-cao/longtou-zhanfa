import copy
import unittest

from emotion_evidence_pack import build, METRICS


class EmotionEvidencePackTests(unittest.TestCase):
    asof="2026-09-07T09:35:00+08:00"

    def snapshot(self, mode="supportive"):
        metrics={}
        for dimension,definitions in METRICS.items():
            for metric,(direction,unit) in definitions.items():
                p25,median,p75=(.2,.5,.8) if unit=="ratio" else (-.05,0,.05)
                if mode=="supportive":
                    value=p75 if direction==1 else p25
                elif mode=="adverse":
                    value=p25 if direction==1 else p75
                else:
                    value=median
                metrics[metric]=dict(metric=metric,value=value,unit=unit,scope_id="mainboard-fixed",
                    observed_at=self.asof,available_at=self.asof,source="synthetic",data_status="verified",
                    definition_id=metric+"-v1",cohort_id=metric+"-fixed",max_age_seconds=300,
                    denominator=100,observed_count=100,missing_count=0,
                    reference=dict(p25=p25,median=median,p75=p75,sample_count=100,
                        definition_id=metric+"-v1",cohort_id=metric+"-fixed",source="historical archive",
                        window_start="2026-06-01T15:00:00+08:00",
                        window_end="2026-09-04T15:00:00+08:00",available_at="2026-09-06T18:00:00+08:00"))
        return dict(asof=self.asof,scope_id="mainboard-fixed",metrics=metrics)

    def test_supportive_pattern_has_no_probability_or_entry(self):
        result=build(self.snapshot("supportive"),self.asof)
        self.assertEqual(result["pattern"],"BROAD_SUPPORTIVE_HYPOTHESIS")
        self.assertIsNone(result["cycle_state_probability"])
        self.assertIsNone(result["entry_decision"])
        self.assertTrue(result["entry_support_eligible"])

    def test_adverse_pattern(self):
        result=build(self.snapshot("adverse"),self.asof)
        self.assertEqual(result["pattern"],"BROAD_ADVERSE_HYPOTHESIS")
        self.assertTrue(all(value=="ADVERSE" for value in result["dimension_states"].values()))

    def test_missing_metric_is_unavailable_not_zero(self):
        snapshot=self.snapshot()
        snapshot["metrics"].pop("buyable_profit_rate")
        result=build(snapshot,self.asof)
        self.assertEqual(result["dimension_states"]["money_effect"],"UNAVAILABLE")
        self.assertEqual(result["pattern"],"INSUFFICIENT_EVIDENCE")

    def test_future_or_inverted_evidence_is_unavailable(self):
        for update in ({"available_at":"2026-09-07T09:36:00+08:00"},
                       {"observed_at":"2026-09-07T09:35:01+08:00"},
                       {"max_age_seconds":-1}):
            snapshot=self.snapshot()
            snapshot["metrics"]["theme_sync"].update(update)
            self.assertEqual(build(snapshot,self.asof)["dimension_states"]["repair_quality"],"UNAVAILABLE")

    def test_future_reference_and_definition_mismatch_are_unavailable(self):
        for update in ({"window_end":"2026-09-07T09:35:00+08:00"},{"definition_id":"other"},
                       {"sample_count":59}):
            snapshot=self.snapshot()
            snapshot["metrics"]["high_feedback"]["reference"].update(update)
            self.assertEqual(build(snapshot,self.asof)["dimension_states"]["height_feedback"],"UNAVAILABLE")

    def test_neutral_dimensions_produce_rotation_hypothesis(self):
        result=build(self.snapshot("neutral"),self.asof)
        self.assertEqual(result["pattern"],"ROTATION_OR_DIVERGENCE_HYPOTHESIS")
        self.assertTrue(all(value=="MIXED" for value in result["dimension_states"].values()))

    def test_low_coverage_and_delayed_data_are_descriptive_only(self):
        for update in ({"observed_count":89,"missing_count":11},{"data_status":"delayed"}):
            snapshot=self.snapshot()
            snapshot["metrics"]["buyable_profit_rate"].update(update)
            result=build(snapshot,self.asof)
            self.assertFalse(result["emotion_analysis"]["money_effect"]["actionable_for_entry_support"])
            self.assertFalse(result["entry_support_eligible"])


if __name__ == "__main__": unittest.main()
