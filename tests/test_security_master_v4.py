import tempfile
import unittest
from pathlib import Path

import pandas as pd

import security_master_v4 as security


class SecurityMasterV4Tests(unittest.TestCase):
    def _row(self, code: str = "600001", name: str = "甲") -> dict:
        return {
            "code": code, "name": name, "valid_from": "2024-01-01", "valid_to": "",
            "exchange": "SSE" if code.startswith("6") else "SZSE", "board": "MAINBOARD",
            "listing_status": "LISTED", "st_status": "NORMAL",
            "price_limit_regime": "MAINBOARD_10", "listing_date": "2000-01-01",
            "delisting_date": "", "source_type": "sse_official" if code.startswith("6") else "szse_official",
            "source_ref": f"https://example.test/{code}", "source_timestamp": "2026-09-01 15:00:00",
            "data_status": "verified",
        }

    def test_verified_mainboard_interval_is_usable(self):
        checked = security.validate_master(pd.DataFrame([self._row()]))
        self.assertTrue(checked.iloc[0]["source_verified"])
        self.assertEqual(checked.iloc[0]["board"], "MAINBOARD")

    def test_overlapping_inclusive_intervals_are_rejected(self):
        first = self._row()
        first["valid_to"] = "2024-06-30"
        second = self._row()
        second["valid_from"] = "2024-06-30"
        second["st_status"] = "ST"
        second["price_limit_regime"] = "ST_5"
        second["name"] = "ST甲"
        with self.assertRaisesRegex(ValueError, "overlap"):
            security.validate_master(pd.DataFrame([first, second]))

    def test_interval_cannot_follow_open_ended_interval(self):
        first = self._row()
        second = self._row()
        second["valid_from"] = "2025-01-01"
        with self.assertRaisesRegex(ValueError, "open-ended"):
            security.validate_master(pd.DataFrame([first, second]))

    def test_st_name_cannot_claim_normal_status(self):
        row = self._row(name="*ST甲")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            security.validate_master(pd.DataFrame([row]))

    def test_candidate_audit_separates_eligible_st_no_limit_and_missing(self):
        normal = self._row("600001", "甲")
        st = self._row("600002", "ST乙")
        st.update({"st_status": "ST", "price_limit_regime": "ST_5"})
        no_limit = self._row("600003", "丙")
        no_limit["price_limit_regime"] = "NO_LIMIT"
        master = security.validate_master(pd.DataFrame([normal, st, no_limit]))
        candidates = pd.DataFrame([
            {"date": "2025-01-02", "code": "600001", "name": "甲", "boards": 2},
            {"date": "2025-01-02", "code": "600002", "name": "ST乙", "boards": 2},
            {"date": "2025-01-02", "code": "600003", "name": "丙", "boards": 2},
            {"date": "2025-01-02", "code": "600999", "name": "缺失", "boards": 2},
        ])
        audit = security.audit_candidates(candidates, master).set_index("code")
        self.assertEqual(audit.loc["600001", "security_status"], "VERIFIED_ELIGIBLE")
        self.assertEqual(audit.loc["600002", "security_status"], "EXCLUDED_ST_OR_DELISTING_RISK")
        self.assertEqual(audit.loc["600003", "security_status"], "EXCLUDED_NO_PRICE_LIMIT")
        self.assertEqual(audit.loc["600999", "security_status"], "UNVERIFIED_POINT_IN_TIME_STATUS")

    def test_non_mainboard_candidate_is_excluded(self):
        row = self._row("300001", "创业样本")
        row.update({"board": "CHINEXT", "price_limit_regime": "OTHER"})
        master = security.validate_master(pd.DataFrame([row]))
        candidate = pd.DataFrame([{"date": "2025-01-02", "code": "300001", "name": "创业样本", "boards": 2}])
        result = security.audit_candidates(candidate, master).iloc[0]
        self.assertEqual(result["security_status"], "EXCLUDED_NON_MAINBOARD")

    def test_untrusted_interval_is_not_treated_as_verified(self):
        row = self._row()
        row["source_type"] = "current_code_list"
        master = security.validate_master(pd.DataFrame([row]))
        candidate = pd.DataFrame([{"date": "2025-01-02", "code": "600001", "name": "甲", "boards": 2}])
        result = security.audit_candidates(candidate, master).iloc[0]
        self.assertEqual(result["security_status"], "UNVERIFIED_SOURCE")
        self.assertFalse(result["historically_eligible"])

    def test_registered_interval_start_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            master = root / "master.csv"
            pd.DataFrame([self._row()]).to_csv(source, index=False)
            security.import_master(source, master)
            changed = self._row()
            changed["source_ref"] = "https://example.test/correction"
            pd.DataFrame([changed]).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "interval start already registered"):
                security.import_master(source, master)

    def test_complete_status_intervals_do_not_prove_universe_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.csv"
            master = root / "master.csv"
            pd.DataFrame([
                {"date": "2025-01-02", "code": "600001", "name": "甲", "boards": 2}
            ]).to_csv(candidates, index=False)
            pd.DataFrame([self._row()]).to_csv(master, index=False)
            result = security.run_audit(candidates, master, root / "out")
            self.assertEqual(result["verified_coverage"], 1.0)
            self.assertFalse(result["candidate_universe_complete"])
            self.assertEqual(result["status"], "SURVIVORSHIP_BIAS_UNRESOLVED")

    def test_full_pool_observation_schema_signal_date_without_name_is_supported(self):
        master = security.validate_master(pd.DataFrame([self._row("600001", "甲")]))
        candidates = pd.DataFrame([{
            "signal_date": "2025-01-02", "code": "600001", "boards": 1,
            "board_group": "FIRST_BOARD",
        }])
        result = security.audit_candidates(candidates, master).iloc[0]
        self.assertEqual(result["date"], "2025-01-02")
        self.assertEqual(result["name"], "")
        self.assertEqual(result["security_status"], "VERIFIED_ELIGIBLE")

    def test_run_audit_honours_historical_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.csv"
            master = root / "master.csv"
            pd.DataFrame([
                {"date": "2025-01-02", "code": "600001", "name": "甲", "boards": 2},
                {"date": "2025-01-03", "code": "600001", "name": "甲", "boards": 3},
            ]).to_csv(candidates, index=False)
            pd.DataFrame([self._row()]).to_csv(master, index=False)
            result = security.run_audit(candidates, master, root / "out", "2025-01-02")
            self.assertEqual(result["candidate_rows"], 1)
            self.assertEqual(result["historical_end_cutoff"], "2025-01-02")


if __name__ == "__main__":
    unittest.main()
