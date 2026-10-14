import unittest
import json
import numpy as np
import pandas as pd

from experience_hypotheses_v297 import indicators,bootstrap,classified,HYPOTHESES,TESTED,split_periods


class ExperienceIntegrity(unittest.TestCase):
    def calendar(self):
        return {"sessions":["2026-04-01","2026-04-02","2026-04-03","2026-04-07","2026-04-08","2026-04-09","2026-04-10","2026-04-13","2026-04-30","2026-05-06"]}

    def rows(self):
        return pd.DataFrame({"date":["2026-04-01"]*3+["2026-04-02"]*2,"code":["000008","600001","600002","000008","600001"],
            "boards":[5,4,2,5,4],"digital_height":[1,0,np.nan,1,0],"code_eight":[1,0,0,1,0],
            "daily_true_one_word":0,"target_original":[1,0,0,1,np.nan],"label_valid":[True,True,True,True,False]})

    def test_fifteen_named_requirements_and_seven_proxies(self):
        self.assertEqual(len(HYPOTHESES),15)
        self.assertEqual(len({r[0] for r in HYPOTHESES}),15)
        self.assertEqual(len(TESTED),7)

    def test_calendar_holiday_excludes_ordinary_weekend(self):
        f=pd.DataFrame({"date":["2026-04-03","2026-04-10"],"code":["000008","600001"],"boards":[2,3]})
        masks=indicators(f,self.calendar())
        self.assertEqual(masks.pre_holiday.tolist(),[1.,0.])
        self.assertEqual(masks.friday.tolist(),[1.,1.])

    def test_masks_cannot_see_future_outcomes(self):
        f=self.rows();first=indicators(f,self.calendar())
        f["target_original"]=1-f.target_original
        f["terminal_board"]=100;f["next_open_gap"]=500
        pd.testing.assert_frame_equal(first,indicators(f,self.calendar()))

    def test_missing_labels_and_scope_rows_are_retained(self):
        result=bootstrap(self.rows(),"code_eight","target_original",self.calendar(),297)
        self.assertEqual(result["full_candidate_rows"],5)
        self.assertEqual(result["label_or_proxy_unknown_rows"],1)
        self.assertEqual(result["exposed"]["rows"]+result["control"]["rows"],4)

    def test_same_day_height_comparison_not_mixed_with_unpaired_dates(self):
        result=bootstrap(self.rows(),"digital_height","target_original",self.calendar(),297)
        self.assertEqual(result["unmatched_group_rows_retained_not_scored"],1)
        self.assertEqual(result["exposed"]["rows"],1)
        self.assertEqual(result["control"]["rows"],1)

    def test_multiplicity_level_counts_all_fixed_proxies(self):
        result=bootstrap(self.rows(),"code_eight","target_original",self.calendar(),297)
        self.assertEqual(result["family_tests"],7)
        self.assertAlmostEqual(result["family_interval_nominal_level"],1-.05/7)

    def test_apparent_statistical_success_cannot_upgrade_core(self):
        result=classified({"difference":.1},{"difference":.15,"support_sufficient":True,"family_block_interval":[.01,.25]},"test")
        self.assertFalse(result["core_rule"])
        self.assertEqual(result["grade"],"B")
        self.assertTrue(result["candidate_for_future_confirmation"])

    def test_duplicate_or_nontrading_date_rejected(self):
        f=pd.DataFrame({"date":["2026-04-04"],"code":["000008"],"boards":[2]})
        with self.assertRaisesRegex(ValueError,"nontrading"):
            indicators(f,self.calendar())

    def test_boundary_candidates_retained_but_future_labels_unused(self):
        rows=pd.DataFrame({"date":["2025-12-30","2025-12-31","2026-01-05"],"target_date":["2025-12-31","2026-01-05","2026-01-06"],"label_valid":True})
        development,confirmation,count=split_periods(rows)
        self.assertEqual(len(development),2)
        self.assertEqual(count,1)
        self.assertEqual(development.label_valid.tolist(),[True,False])
        self.assertEqual(len(confirmation),1)
        self.assertTrue(rows.label_valid.all())

    def test_statistics_persist_without_numpy_bool_or_nonfinite_numbers(self):
        result=bootstrap(self.rows(),"code_eight","target_original",self.calendar(),297)
        self.assertIsInstance(result["support_sufficient"],bool)
        json.dumps(result,allow_nan=False)


if __name__=="__main__":unittest.main()
