import math
import unittest

import pandas as pd

import ths_auction_adapter_v15 as ths


class ThsAuctionAdapterV15Tests(unittest.TestCase):
    @staticmethod
    def candidate(**overrides):
        row = {
            "date": "2026-09-01", "code": "003005", "name": "竞业达", "boards": 3,
            "emotion_state": "升温", "valid_identity": 1, "one_word_proxy": 0,
            "announcement_risk": 0,
        }
        row.update(overrides)
        return pd.Series(row)

    @staticmethod
    def plan():
        return {
            "session_id": "s1", "signal_date": "2026-09-01", "trade_date": "2026-09-02",
            "candidate_code": "003005", "candidate_name": "竞业达",
        }

    @staticmethod
    def export(gap="3%", auction="3000万", previous="10亿", match20="2500万", match2450="2300万"):
        raw = pd.DataFrame([{
            "股票代码": "003005", "股票简称": "竞业达", "竞价涨幅": gap,
            "今日竞价金额": auction, "昨日成交额": previous,
            "9:20匹配金额": match20, "9:24:50匹配金额": match2450,
        }])
        return ths.normalize_export(raw)

    def test_chinese_money_and_percent_are_normalized(self):
        frame = self.export()
        self.assertAlmostEqual(frame.iloc[0]["auction_gap"], 0.03)
        self.assertEqual(frame.iloc[0]["auction_amount"], 30_000_000)
        self.assertAlmostEqual(frame.iloc[0]["auction_amount_ratio"], 0.03)
        self.assertAlmostEqual(frame.iloc[0]["auction_cancel_ratio"], 0.08)

    def test_normal_open_only_becomes_watch_and_never_enter(self):
        result = ths.screen_candidate(
            self.plan(), self.candidate(), self.export(),
            "THEME_SYNC_ELIGIBLE", "NO_HARD_RISK_FOUND",
        )
        self.assertEqual(result["auction_status"], "WATCH_AUCTION")
        self.assertEqual(result["execution_permission"], "NO_ENTRY_AT_0925")
        self.assertAlmostEqual(result["structural_position_cap_pct"], 0.005 / 0.12)
        self.assertIsNone(result["final_position_cap_pct"])

    def test_high_open_without_amount_is_cancelled(self):
        result = ths.screen_candidate(
            self.plan(), self.candidate(), self.export(gap="6%", auction="1500万"),
            "THEME_SYNC_ELIGIBLE", "NO_HARD_RISK_FOUND",
        )
        self.assertEqual(result["auction_status"], "CANCEL")
        self.assertEqual(result["final_position_cap_pct"], 0)

    def test_high_open_with_large_match_decay_is_cancelled(self):
        result = ths.screen_candidate(
            self.plan(), self.candidate(), self.export(gap="6%", match2450="1000万"),
            "THEME_SYNC_ELIGIBLE", "NO_HARD_RISK_FOUND",
        )
        self.assertEqual(result["auction_status"], "CANCEL")
        self.assertIn("衰减超过45%", result["reasons"])

    def test_missing_match_history_is_data_limited(self):
        result = ths.screen_candidate(
            self.plan(), self.candidate(), self.export(match20=None, match2450=None),
            "THEME_SYNC_ELIGIBLE", "NO_HARD_RISK_FOUND",
        )
        self.assertEqual(result["auction_status"], "DATA_LIMITED")

    def test_failed_identity_or_theme_binding_is_ineligible(self):
        result = ths.screen_candidate(
            self.plan(), self.candidate(valid_identity=0), self.export(),
            "INSUFFICIENT_PEER_ROSTER", "NO_HARD_RISK_FOUND",
        )
        self.assertEqual(result["auction_status"], "INELIGIBLE")
        self.assertEqual(result["structural_position_cap_pct"], 0)

    def test_hard_announcement_overrides_strong_auction(self):
        result = ths.screen_candidate(
            self.plan(), self.candidate(), self.export(),
            "THEME_SYNC_ELIGIBLE", "HARD_RISK_REJECT",
        )
        self.assertEqual(result["auction_status"], "INELIGIBLE")
        self.assertIn("公告绑定存在硬风险", result["reasons"])

    def test_gap_caps_are_provisional(self):
        self.assertAlmostEqual(ths.structural_position_cap(self.candidate(boards=4, emotion_state="加速"), 0.06), 0.03)
        self.assertAlmostEqual(ths.structural_position_cap(self.candidate(boards=5, emotion_state="修复"), 0.08), 0.02)
        self.assertEqual(ths.structural_position_cap(self.candidate(emotion_state="退潮"), 0.03), 0)

    def test_capture_deadline_blocks_posthoc_screen(self):
        with self.assertRaisesRegex(ValueError, "after the 09:27"):
            ths.validate_capture_time(
                "2026-09-02", "2026-09-02T09:25:30+08:00",
                "2026-09-02T09:28:00+08:00", True,
            )


if __name__ == "__main__":
    unittest.main()
