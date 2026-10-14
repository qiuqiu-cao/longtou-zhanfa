import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from historical_ohlc_archive_v278 import parse
from september_forecast_holdout_v278 import attach_ohlc, rebuild_market, relay_summary, relay_features, feedback_cohort


class SourceTests(unittest.TestCase):
    def test_existing_output_is_not_modified_by_failed_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "report.json"
            original.write_text('{"fixture":true}',encoding="utf-8")
            process = subprocess.run([sys.executable,str(Path(__file__).with_name("september_forecast_holdout_v278.py")),
                "--output",directory],capture_output=True,text=True,timeout=15)
            self.assertNotEqual(process.returncode,0)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),["report.json"])
            self.assertEqual(original.read_text(encoding="utf-8"),'{"fixture":true}')
    def test_adjusted_bars_cannot_substitute_for_raw(self):
        p = {"data":{"sh600001":{"qfqday":[["2026-09-01","10","10","10","10","1"]]}}}
        with self.assertRaises(ValueError): parse(p,"sh600001")

    def test_invalid_bounds_and_duplicate_dates_rejected(self):
        rows = [["2026-09-01","11","10","10","9","1"]]
        with self.assertRaises(ValueError): parse({"data":{"sh600001":{"day":rows}}},"sh600001")
        rows = [["2026-09-01","10","10","10","10","1"]]*2
        with self.assertRaises(ValueError): parse({"data":{"sh600001":{"day":rows}}},"sh600001")

    def test_true_one_word_and_legacy_proxy_are_separate(self):
        pool = pd.DataFrame({"date":["2026-09-01"],"code":["600001"],"price":[11.],"turnover":[5.]})
        daily = pd.DataFrame({"symbol":["sh600001"]*2,"date":["2026-08-31","2026-09-01"],
                              "open":[10.,11.],"close":[10.,11.],"high":[10.,11.],"low":[10.,11.],"volume":[1.,1.]})
        r = attach_ohlc(pool,daily).iloc[0]
        self.assertEqual(r.actual_one_word_daily,1)
        self.assertEqual(r.one_word_proxy,0)
        pool.loc[0,"price"] = 12
        r = attach_ohlc(pool,daily).iloc[0]
        self.assertTrue(np.isnan(r.actual_one_word_daily))

    def test_missing_pool_does_not_become_previous_session(self):
        pool = pd.DataFrame({"date":["2026-09-07","2026-09-09"],"code":["600001"]*2,
                             "boards":[2,3],"one_word_proxy":[0.,0.],"source_path":["TEST"]*2,"captured_at":[None]*2})
        base = pd.DataFrame({"date":["2026-08-27"],"max_board":[2],"limit_count":[20]})
        def aux(_path,day,kind):
            return pd.DataFrame({"涨跌幅":[1.]}),{"date":day,"kind":kind,"status":"TEST"}
        with patch("september_forecast_holdout_v278.auxiliary",side_effect=aux):
            rows,_ = rebuild_market(pool,base)
        row = rows.set_index("date").loc["2026-09-09"]
        self.assertTrue(np.isnan(row.promotion_mid))
        self.assertTrue(np.isnan(row.height_change))
        self.assertEqual(row.data_status,"partial")

    def test_unknown_next_pool_retained_as_unresolved(self):
        frame = pd.DataFrame({"date":["2026-09-01","2026-09-04"],"code":["600001"]*2,"boards":[2,3],
                              "label_valid":[False,True],"target":[np.nan,0.],"probability":[.9,.8],
                              "one_word_proxy":[0,0],"actual_one_word_daily":[0,0]})
        result = relay_summary(frame)
        self.assertEqual(result["dates_total"],2)
        self.assertEqual(result["dates_with_complete_next_pool"],1)
        self.assertEqual(result["identity"]["successes"],0)

    def test_feedback_equal_counts_do_not_prove_equal_cohort(self):
        previous = pd.DataFrame({"code":["600001","600002"]})
        feedback = pd.DataFrame({"代码":["600001","600003"],"涨跌幅":[1.,10.]})
        selected,audit = feedback_cohort(previous,feedback)
        self.assertEqual(selected["代码"].tolist(),["600001"])
        self.assertEqual(audit["coverage"],.5)
        self.assertEqual(audit["status"],"PARTIAL_PREVIOUS_CLOSING_POOL")

    def test_turnover_ratio_matches_last_observed_legacy_definition(self):
        history = pd.DataFrame({"date":["2026-08-31","2026-09-15"],"code":["600001"]*2,"turnover":[.05,.99]})
        pools = pd.DataFrame({"date":["2026-09-07","2026-09-09"],"code":["600001"]*2,"name":["TEST"]*2,
            "boards":[2,3],"target_date":["2026-09-08","2026-09-10"],"promoted_next_day":[np.nan,0.],
            "turnover":[2.,4.],"price":[10.,11.],"amount":[100.,200.],"amplitude":[2.,3.],
            "one_word_proxy":[0.,0.],"actual_one_word_daily":[0.,0.],"ohlc_status":["TEST"]*2,"captured_at":[None]*2})
        market = pd.DataFrame({"date":pools.date,"data_status":["partial"]*2})
        f = relay_features(pools,market,history)
        self.assertAlmostEqual(f.iloc[0].turnover_vs_prev,.4)
        self.assertEqual(f.iloc[1].turnover_vs_prev,2.)
        self.assertEqual(f.iloc[1].turnover_reference_date,"2026-09-07")


if __name__ == "__main__":
    unittest.main()
