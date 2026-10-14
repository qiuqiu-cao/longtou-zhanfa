import unittest
from datetime import date,timedelta

import pandas as pd

from emotion_reference_builder import build


class EmotionReferenceBuilderTests(unittest.TestCase):
    asof="2026-09-07T09:35:00+08:00"

    def rows(self, count=60):
        rows=[]
        start=date(2026,6,1)
        for index in range(count):
            day=(start+timedelta(days=index)).isoformat()
            rows.append(self.row(day,index/100))
        rows.append(self.row("2026-09-07",.8,"09:35:00"))
        return rows

    def sessions(self, rows):
        return sorted({row["date"] for row in rows})

    def row(self,day,value,clock="15:00:00"):
        return dict(date=day,metric="theme_sync",value=value if value<=1 else 1,unit="ratio",
                    denominator=100,observed_count=100,missing_count=0,scope_id="mainboard-fixed",
                    definition_id="theme-sync-v1",cohort_id="theme-fixed",source="synthetic",
                    observed_at=f"{day}T{clock}+08:00",available_at=f"{day}T{clock}+08:00",data_status="verified")

    def test_causal_reference_uses_prior_dates_only(self):
        rows=self.rows()
        result=build(pd.DataFrame(rows),self.asof,self.sessions(rows),60,60)
        item=result["metrics"]["theme_sync"]
        self.assertEqual(item["reference"]["sample_count"],60)
        self.assertLess(item["reference"]["p75"],.8)
        self.assertFalse(result["all_metrics_ready"])

    def test_future_row_and_changed_cohort_are_excluded(self):
        rows=self.rows(59)
        future=self.row("2026-09-06",.9)
        future["available_at"]="2026-09-07T09:36:00+08:00"
        changed=self.row("2026-09-05",.9)
        changed["cohort_id"]="changed"
        rows.extend([future,changed])
        result=build(pd.DataFrame(rows),self.asof,self.sessions(rows),120,60)
        self.assertEqual(result["metric_audit"]["theme_sync"]["status"],"REFERENCE_INSUFFICIENT")

    def test_latest_available_current_snapshot_is_selected(self):
        rows=self.rows()
        rows.append(self.row("2026-09-07",.9,"09:30:00"))
        result=build(pd.DataFrame(rows),self.asof,self.sessions(rows),60,60)
        self.assertEqual(result["metrics"]["theme_sync"]["value"],.8)

    def test_missing_denominator_or_short_minimum_rejected(self):
        with self.assertRaises(ValueError):
            rows=self.rows()
            build(pd.DataFrame(rows).drop(columns=["denominator"]),self.asof,self.sessions(rows),60,60)
        with self.assertRaises(ValueError):
            rows=self.rows()
            build(pd.DataFrame(rows),self.asof,self.sessions(rows),59,59)

    def test_non_session_rows_do_not_enter_reference(self):
        rows=self.rows(59)
        extra=self.row("2026-08-31",.5)
        rows.append(extra)
        sessions=self.sessions(rows)
        sessions.remove("2026-08-31")
        result=build(pd.DataFrame(rows),self.asof,sessions,120,60)
        self.assertEqual(result["metric_audit"]["theme_sync"]["status"],"REFERENCE_INSUFFICIENT")

    def test_multiple_current_market_scopes_are_rejected(self):
        rows=self.rows()
        other=dict(rows[-1],metric="repair_volume_support",definition_id="repair-v1",
                   cohort_id="repair-fixed",scope_id="different-scope")
        rows.append(other)
        with self.assertRaisesRegex(ValueError,"multiple market scopes"):
            build(pd.DataFrame(rows),self.asof,self.sessions(rows),60,60)


if __name__=="__main__": unittest.main()
