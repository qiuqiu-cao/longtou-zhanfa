import unittest

import pandas as pd

from intraday_flow_causality_v176 import build, readiness_report


class IntradayFlowCausalityV176Tests(unittest.TestCase):
    signal = "2026-09-16"
    trade = "2026-09-17"

    def candidates(self):
        return pd.DataFrame([
            {"signal_date": self.signal, "code": "600001", "name": "高标股份", "boards": 3},
            {"signal_date": self.signal, "code": "000001", "name": "低位甲", "boards": 1},
            {"signal_date": self.signal, "code": "000002", "name": "低位乙", "boards": 1},
        ])

    def roster(self):
        rows = []
        for row in self.candidates().itertuples():
            rows.append({
                "signal_date": self.signal, "trade_date": self.trade,
                "code": row.code, "name": row.name, "boards": row.boards,
                "theme_group_id": "theme.alpha", "theme_evidence_id": "ev-alpha",
                "membership_verified": True, "driver_verified": True,
                "roster_complete": True,
                "evidence_available_at": "2026-09-16 20:00:00",
            })
        return pd.DataFrame(rows)

    def auction(self):
        rows = []
        for code, name in (("600001", "高标股份"), ("000001", "低位甲"),
                           ("000002", "低位乙")):
            for point, amount in (("09:20", 100.0), ("09:24:50", 50.0),
                                  ("09:25", 60.0)):
                rows.append({
                    "signal_date": self.signal, "trade_date": self.trade,
                    "asof": point, "code": code, "name": name,
                    "source": "terminal", "source_timestamp":
                    f"{self.trade} {point}" if point.count(":") == 2
                    else f"{self.trade} {point}:00",
                    "data_status": "verified_terminal_export",
                    "matched_amount": amount, "indicative_price": 9.5,
                    "prior_close": 9.0, "prior_day_amount": 1000.0,
                })
        return pd.DataFrame(rows)

    def minutes(self):
        paths = {
            # Anchor touches, breaks, then reseals after low-position expansion.
            "600001": [
                ("09:31", 9.8, 10.0, 9.8, 10.0),
                ("09:32", 10.0, 10.0, 9.7, 9.8),
                ("09:33", 9.8, 9.9, 9.7, 9.8),
                ("09:34", 9.8, 10.0, 9.8, 10.0),
                ("09:35", 10.0, 10.0, 10.0, 10.0),
            ],
            "000001": [
                ("09:31", 9.5, 9.6, 9.4, 9.5),
                ("09:32", 9.5, 9.8, 9.5, 9.7),
                ("09:33", 9.7, 10.0, 9.7, 10.0),
                ("09:34", 10.0, 10.0, 10.0, 10.0),
                ("09:35", 10.0, 10.0, 10.0, 10.0),
            ],
            "000002": [
                ("09:31", 9.3, 9.4, 9.2, 9.3),
                ("09:32", 9.3, 9.4, 9.2, 9.3),
                ("09:33", 9.3, 9.4, 9.2, 9.3),
                ("09:34", 9.3, 9.4, 9.2, 9.3),
                ("09:35", 9.3, 9.4, 9.2, 9.3),
            ],
        }
        names = {"600001": "高标股份", "000001": "低位甲", "000002": "低位乙"}
        rows = []
        for code, bars in paths.items():
            for point, open_, high, low, close in bars:
                rows.append({
                    "signal_date": self.signal, "trade_date": self.trade,
                    "bar_end": f"{self.trade} {point}:00", "code": code,
                    "name": names[code], "source": "terminal",
                    "source_timestamp": f"{self.trade} {point}:00",
                    "data_status": "verified_terminal_export", "open": open_,
                    "high": high, "low": low, "close": close,
                    "volume_shares": 100, "amount_yuan": close * 100,
                    "limit_up_price": 10.0,
                })
        return pd.DataFrame(rows)

    def test_level1_change_is_not_promoted_to_cancel_or_replenishment(self):
        candidates, _, report = build(
            self.auction(), self.minutes(), self.candidates(), self.roster(),
            self.signal, self.trade, "09:35")
        anchor = candidates.set_index("code").loc["600001"]
        self.assertAlmostEqual(anchor.apparent_match_decay_ratio, .5)
        self.assertIsNone(anchor.gross_cancel_amount)
        self.assertIsNone(anchor.gross_replenishment_amount)
        self.assertEqual(anchor.level2_queue_status, "UNAVAILABLE_NO_LEVEL2")
        self.assertFalse(report["governance"][
            "level1_net_change_may_be_called_gross_cancel"])

    def test_low_touch_between_anchor_break_and_reseal_is_sequence_not_causality(self):
        _, sequences, report = build(
            self.auction(), self.minutes(), self.candidates(), self.roster(),
            self.signal, self.trade, "09:35")
        row = sequences.iloc[0]
        self.assertEqual(row.sequence_status, "VERIFIED_MINUTE_SEQUENCE_NOT_CAUSAL")
        self.assertEqual(row.low_touches_between_anchor_break_and_reseal, 1)
        self.assertTrue(row.low_feedback_sequence_supported)
        self.assertFalse(row.causal_feedback_claim_allowed)
        self.assertFalse(row.common_catalyst_simultaneity_excluded)
        self.assertFalse(report["governance"]["minute_order_proves_causal_feedback"])

    def test_future_roster_evidence_is_rejected(self):
        roster = self.roster()
        roster.loc[0, "evidence_available_at"] = "2026-09-17 10:01:00"
        with self.assertRaisesRegex(ValueError, "after asof"):
            build(self.auction(), self.minutes(), self.candidates(), roster,
                  self.signal, self.trade, "10:00")

    def test_readiness_keeps_unavailable_microstructure_explicit(self):
        report = readiness_report()
        self.assertEqual(report["status"], "NO_FORWARD_OBSERVATIONS")
        self.assertIn("gross cancellations", report["unavailable_without_level2"])
        self.assertFalse(report["policy"]["direct_model_use_allowed"])
        self.assertFalse(report["policy"]["entry_authorized"])


if __name__ == "__main__":
    unittest.main()
