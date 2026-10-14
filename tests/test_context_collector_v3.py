import unittest

import pandas as pd

import collect_context_v3 as context


class ContextCollectorV3Tests(unittest.TestCase):
    def test_observed_completed_reduction_wording(self):
        for title in (
            "控股股东及其一致行动人减持股份结果公告",
            "控股股东、实际控制人减持股份计划期限届满暨实施情况的公告",
            "股东减持计划期限届满公告",
        ):
            with self.subTest(title=title):
                self.assertEqual(context.classify_announcement(title)[:2],
                                 ("historical_corporate_action_closed", 0))

    def test_closed_phrase_cannot_hide_new_or_unrelated_risk(self):
        for title, expected in (
            ("减持股份结果暨下一期减持计划公告", "减持"),
            ("减持计划完成暨拟减持股份公告", "减持"),
            ("减持结果暨立案调查公告", "风险提示_退市或重大违法"),
            ("股票交易异常波动暨股东减持计划公告", "减持"),
            ("立案调查专项说明", "风险提示_退市或重大违法"),
            ("关于行政处罚事项回复的公告", "风险提示_监管"),
            ("关于对公司予以监管警示的决定", "风险提示_监管"),
            ("关于公司股票交易异常波动事项的监管工作函", "风险提示_监管"),
        ):
            with self.subTest(title=title):
                self.assertEqual(context.classify_announcement(title)[:2], (expected, 1))

    def test_reply_does_not_certify_regulatory_case_closed(self):
        self.assertEqual(context.classify_announcement("问询函回复")[:2],
                         ("风险提示_监管", 1))

    def test_ordinary_abnormal_notice_is_context_not_mechanical_veto(self):
        label, hard, _ = context.classify_announcement("某公司股票交易异常波动公告", "异常波动")
        self.assertEqual(label, "监管背景_普通异常波动")
        self.assertEqual(hard, 0)

    def test_serious_abnormal_notice_remains_hard(self):
        label, hard, _ = context.classify_announcement("股票交易严重异常波动公告")
        self.assertEqual(label, "风险提示_严重异常波动")
        self.assertEqual(hard, 1)

    def test_announcement_classifier_separates_reduction_unlock_and_earnings(self):
        self.assertEqual(context.classify_announcement("股东拟减持股份公告")[0], "减持")
        self.assertEqual(context.classify_announcement("限售股份上市流通公告")[0], "解禁")
        self.assertEqual(context.classify_announcement("年度业绩预亏公告")[0], "业绩雷")

    def test_unlock_conditions_met_is_not_actual_listing(self):
        pending = context.classify_announcement(
            "限制性股票第一个解除限售期解除限售条件成就的公告"
        )
        self.assertEqual(pending[:2], ("解禁条件成就_待上市流通公告", 0))
        actual = context.classify_announcement(
            "解除限售条件成就暨解除限售股份上市流通公告"
        )
        self.assertEqual(actual[:2], ("解禁", 1))

    def test_explicit_risk_notice_is_hard_even_without_abnormal_wording(self):
        label, hard, _ = context.classify_announcement(
            "关于股票交易风险提示公告", "风险提示性公告"
        )
        self.assertEqual(label, "风险提示_风险公告")
        self.assertEqual(hard, 1)

    def test_completed_reduction_is_not_mistaken_for_new_sell_pressure(self):
        label, hard, _ = context.classify_announcement(
            "关于控股股东、实际控制人减持计划完成的公告"
        )
        self.assertEqual(label, "historical_corporate_action_closed")
        self.assertEqual(hard, 0)

    def test_completed_reduction_is_not_reactivated_by_broad_category(self):
        label, hard, _ = context.classify_announcement(
            "减持股份计划实施期限届满暨减持计划实施完毕的公告",
            "股东增减持",
        )
        self.assertEqual(label, "historical_corporate_action_closed")
        self.assertEqual(hard, 0)
        self.assertEqual(
            context.classify_announcement("关于持股5%以上股东股份减持计划实施完毕的公告")[:2],
            ("historical_corporate_action_closed", 0),
        )
        self.assertEqual(context.classify_announcement("控股股东减持计划公告")[:2], ("减持", 1))

    def test_closed_or_old_risks_do_not_remain_hard_forever(self):
        label, hard, _ = context.classify_announcement("股东减持计划提前终止暨减持结果公告")
        self.assertEqual(label, "historical_corporate_action_closed")
        self.assertEqual(hard, 0)
        self.assertEqual(context.apply_risk_window("风险提示_严重异常波动", 1, "", 3)[0],
                         "风险提示_严重异常波动")
        expired = context.apply_risk_window("风险提示_严重异常波动", 1, "", 8)
        self.assertEqual(expired[0], "historical_notice_expired")
        self.assertEqual(expired[1], 0)

    def test_generic_concepts_do_not_become_trade_themes(self):
        self.assertTrue(context.is_generic_concept("融资融券"))
        self.assertTrue(context.is_generic_concept("昨日涨停_含一字"))
        self.assertFalse(context.is_generic_concept("创新药"))

    def test_theme_selection_requires_same_day_reinforcement(self):
        evidence = pd.DataFrame([
            {"date": "2026-08-31", "code": "600001", "name": "甲", "concept": "创新药",
             "same_day_limit_up_support": 3, "same_day_ladder_support": 2,
             "generic_concept": False, "membership_status": "verified_current_snapshot",
             "source": "test", "source_timestamp": "2026-08-31T16:00:00"},
            {"date": "2026-08-31", "code": "600001", "name": "甲", "concept": "融资融券",
             "same_day_limit_up_support": 20, "same_day_ladder_support": 8,
             "generic_concept": True, "membership_status": "verified_current_snapshot",
             "source": "test", "source_timestamp": "2026-08-31T16:00:00"},
            {"date": "2026-08-31", "code": "600002", "name": "乙", "concept": "孤立概念",
             "same_day_limit_up_support": 1, "same_day_ladder_support": 1,
             "generic_concept": False, "membership_status": "verified_current_snapshot",
             "source": "test", "source_timestamp": "2026-08-31T16:00:00"},
        ])
        selected = context.choose_theme(evidence)
        self.assertEqual(selected[["code", "theme"]].to_dict("records"),
                         [{"code": "600001", "theme": "创新药"}])

    def test_missing_concept_evidence_is_not_reported_as_no_theme(self):
        report = context.make_report(
            "2026-08-31",
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "concepts": {"status": "skipped"},
                "announcements": {"status": "verified_index_date_only"},
            },
        )
        self.assertIn("无法判断", report)
        self.assertIn("不得解释为无题材", report)

    def test_full_pool_scope_includes_first_and_relay_but_not_failed(self):
        frame = pd.DataFrame([
            {"signal_date": "2026-09-04", "code": "1", "name": "A",
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True},
            {"signal_date": "2026-09-04", "code": "600001", "name": "B",
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": True},
            {"signal_date": "2026-09-04", "code": "2", "name": "C",
             "pool_assignment": "FAILED_BOARD_OBSERVATION", "is_mainboard": True},
            {"signal_date": "2026-09-04", "code": "300001", "name": "D",
             "pool_assignment": "CLOSING_LIMIT_UP", "is_mainboard": False},
        ])
        day, selected = context.select_candidate_scope(frame, "2026-09-04")
        self.assertEqual(day, "2026-09-04")
        self.assertEqual(set(selected.code), {"000001", "600001"})


if __name__ == "__main__":
    unittest.main()
