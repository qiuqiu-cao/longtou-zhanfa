import unittest
from opening_minute_measurements import derive, derive_absorption
from entry_plan_audit import numeric_condition


class MinuteTests(unittest.TestCase):
    def rows(self):
        return [dict(code="600001",bar_end=f"2026-09-07T09:{31+i}:00+08:00",
                     available_at=f"2026-09-07T09:{31+i}:01+08:00",open=10+i*.1,high=10+i*.1,
                     low=10+i*.1,close=10+i*.1,volume=100,amount=(10+i*.1)*100,
                     source="synthetic",data_status="verified",volume_unit="shares",amount_unit="yuan",
                     interval_seconds=60,timestamp_side="end") for i in range(5)]

    def test_vwap_and_numeric_condition(self):
        measurements=derive(self.rows(),"600001","2026-09-07","2026-09-07T09:35:01+08:00")
        evidence=measurements["opening_vwap_change_5m"]
        self.assertAlmostEqual(evidence["value"],.02)
        rule=dict(metric=evidence["metric"],unit="ratio",window_seconds=300,definition_id=evidence["definition_id"],
                  threshold=.01,operator=">=",max_age_seconds=5)
        self.assertEqual(numeric_condition(rule,evidence,"2026-09-07T09:35:01+08:00")["status"],"PASS")

    def test_gap_units_and_future_rejected(self):
        for mode in ("gap","units","future"):
            rows=self.rows()
            if mode=="gap": rows.pop(2)
            if mode=="units": rows[0]["volume_unit"]="lots"
            if mode=="future": rows[-1]["available_at"]="2026-09-07T09:36:00+08:00"
            with self.assertRaises(ValueError):
                derive(rows,"600001","2026-09-07","2026-09-07T09:35:01+08:00")

    def absorption_rows(self):
        rows=[]
        closes=[10.8,10.7,11]+[11]*12
        for i,close in enumerate(closes,1):
            prior=10.8 if i==1 else closes[i-2]
            high=11 if i in (1,3) or close==11 else max(prior,close)
            low=min(prior,close)
            rows.append(dict(code="600001",bar_end=f"2026-09-07T09:{30+i}:00+08:00",
                available_at=f"2026-09-07T09:{30+i}:01+08:00",open=prior,high=high,low=low,close=close,
                volume=100,amount=close*100,limit_price=11,source="synthetic",data_status="verified",
                volume_unit="shares",amount_unit="yuan",interval_seconds=60,timestamp_side="end"))
        return rows

    def test_absorption_reseal_proxies(self):
        result=derive_absorption(self.absorption_rows(),"600001","2026-09-07","2026-09-07T09:45:01+08:00")
        self.assertEqual(result["opening_failed_seal_episodes_15m"]["value"],1)
        self.assertEqual(result["opening_resealed_after_failure_15m"]["value"],1)
        self.assertEqual(result["opening_closed_at_limit_0945"]["value"],1)
        self.assertAlmostEqual(result["opening_pullback_volume_share_15m"]["value"],1/15)

    def test_absorption_requires_limit_and_complete_window(self):
        rows=self.absorption_rows()
        del rows[0]["limit_price"]
        with self.assertRaises(ValueError):
            derive_absorption(rows,"600001","2026-09-07","2026-09-07T09:45:01+08:00")
        rows=self.absorption_rows()[:-1]
        with self.assertRaises(ValueError):
            derive_absorption(rows,"600001","2026-09-07","2026-09-07T09:45:01+08:00")


if __name__=="__main__": unittest.main()
