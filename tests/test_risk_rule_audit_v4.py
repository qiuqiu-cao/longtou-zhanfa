import unittest

import pandas as pd

import risk_rule_audit_v4 as audit


class RiskRuleAuditV4Tests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        dates = pd.bdate_range("2026-01-05", periods=9)
        for idx, date in enumerate(dates):
            rows.append({
                "date": date.date().isoformat(),
                "trade_date": (date + pd.offsets.BDay(1)).date().isoformat(),
                "code": f"600{idx:03d}",
                "boards": 2 + idx % 4,
                "large_loss": float(idx % 2),
                "promoted_next_day": float((idx + 1) % 2),
                "mark_to_market_return": -0.08 if idx % 2 else 0.02,
                "execution_status": "inferred_daily_limit_touch_fill",
                "large_loss_label_status": "inferred_from_unadjusted_daily_ohlc",
                "execution_source": "inferred_from_unadjusted_daily_ohlc",
                "emotion_state": "大分歧" if idx % 2 else "升温",
                "false_acceleration": float(idx % 2),
                "regime_fragility": 0.7 if idx % 2 else 0.2,
                "loss_effect_health": 0.3 if idx % 2 else 0.8,
                "previous_limit_tail_loss": -0.06 if idx % 2 else -0.01,
                "emotion_risk_level": "HIGH" if idx % 2 else "LOW",
                "one_word_streak": 2 if idx % 2 else 0,
                "next_open_gap": 0.08 if idx % 2 else 0.01,
            })
        return pd.DataFrame(rows)

    def test_eligible_population_excludes_unlabeled_unexecuted_and_out_of_scope(self):
        frame = self._frame()
        excluded = frame.iloc[[0]].copy()
        excluded["code"] = "000999"
        excluded["boards"] = 6
        unexecuted = frame.iloc[[1]].copy()
        unexecuted["code"] = "000998"
        unexecuted["execution_status"] = "no_limit_touch_no_trade"
        unlabeled = frame.iloc[[2]].copy()
        unlabeled["code"] = "000997"
        unlabeled["large_loss"] = None
        result = audit.eligible_executions(pd.concat([frame, excluded, unexecuted, unlabeled]))
        self.assertEqual(len(result), len(frame))
        self.assertTrue(result["boards"].isin([2, 3, 4, 5]).all())

    def test_folds_keep_whole_dates_and_are_strictly_chronological(self):
        folded = audit.assign_contiguous_folds(audit.eligible_executions(self._frame()))
        self.assertEqual(folded.groupby("date")["fold"].nunique().max(), 1)
        windows = folded.groupby("fold")["date"].agg(["min", "max"])
        self.assertLess(windows.loc[1, "max"], windows.loc[2, "min"])
        self.assertLess(windows.loc[2, "max"], windows.loc[3, "min"])

    def test_missing_rule_value_is_not_treated_as_zero(self):
        frame = self._frame()
        frame.loc[0, "one_word_streak"] = None
        report = audit.audit_rules(frame)
        rule = next(item for item in report["rules"] if item["rule_id"] == "EOD_ONE_WORD_STREAK")
        self.assertEqual(rule["missing_n"], 1)
        self.assertEqual(rule["assessed_n"], len(frame) - 1)

    def test_stability_gate_requires_every_fold_to_agree(self):
        good = {
            "flagged_n": 60,
            "risk_difference": 0.10,
            "risk_ratio": 1.4,
        }
        pooled = {"flagged_n": 180, "risk_difference": 0.10, "risk_ratio": 1.4}
        passed, reasons = audit.stability_gate([good, good, good], pooled)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        bad = dict(good, risk_difference=-0.01, risk_ratio=0.98)
        passed, reasons = audit.stability_gate([good, bad, good], pooled)
        self.assertFalse(passed)
        self.assertTrue(any("全部时间段同向" in reason for reason in reasons))

    def test_report_is_research_only_and_does_not_promote_failed_rule(self):
        report = audit.audit_rules(self._frame())
        self.assertEqual(report["status"], "RESEARCH_ONLY")
        self.assertEqual(report["decision"], "NO_STABLE_RULE")
        self.assertEqual(report["stable_rule_ids"], [])


if __name__ == "__main__":
    unittest.main()
