import unittest
import pandas as pd
from import_minute_csv_v47 import audit_requested_coverage, normalize


class MinuteImportTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(code="17",date="2024-01-15",time="09:30",open=10,high=10,low=10,close=10,volume=1,amount=1000)])

    def run_import(self,frame):
        return normalize(frame,1,"start","lots","yuan")

    def test_units_and_bar_end(self):
        row = self.run_import(self.frame()).iloc[0]
        self.assertEqual(row["code"],"000017")
        self.assertEqual(row["time"],"20240115093100000")
        self.assertEqual(row["bar_vwap"],10)

    def test_auction_rejected(self):
        frame=self.frame(); frame["time"]="09:25"
        with self.assertRaises(ValueError): self.run_import(frame)

    def test_lunch_rejected(self):
        frame=self.frame(); frame["time"]="11:30"
        with self.assertRaises(ValueError): self.run_import(frame)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): self.run_import(pd.concat([self.frame(),self.frame()]))

    def test_ambiguous_column_rejected(self):
        frame=self.frame(); frame["开盘"]=10
        with self.assertRaises(ValueError): self.run_import(frame)

    def test_unit_mismatch_flag_not_silently_fixed(self):
        frame=self.frame(); frame["amount"]=100000
        self.assertFalse(self.run_import(frame).iloc[0]["amount_volume_consistent"])

    def test_request_audit_keeps_missing_distinct_from_zero(self):
        frame = self.run_import(self.frame())
        audit = audit_requested_coverage(
            frame, {("000017", "2024-01-15"), ("000020", "2024-01-15")}, 1)
        self.assertEqual(audit["complete_groups"], 0)
        self.assertEqual(audit["incomplete_groups"], 1)
        self.assertEqual(audit["missing_groups"], 1)

    def test_request_audit_rejects_unrequested_group(self):
        frame = self.run_import(self.frame())
        with self.assertRaises(ValueError):
            audit_requested_coverage(frame, {("000020", "2024-01-15")}, 1)
