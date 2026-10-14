import unittest

from sse_name_history_v4 import acw_cookie, is_event_title, parse_event_text


class SSENameHistoryV4Tests(unittest.TestCase):
    def test_cookie_challenge_matches_known_response(self):
        arg1 = "CB100A9EE2C0525B07B1C92667FF9EEA08A23655"
        self.assertEqual(acw_cookie(arg1), "6a961ca62e3b3f9463b9564d1ca4d790279f8077")

    def test_explicit_transition_and_date_are_verified(self):
        text = "公司股票将于2024年5月6日起实施退市风险警示，股票简称由博信股份变更为*ST博信。"
        event = parse_event_text(text)
        self.assertEqual(event["parse_status"], "VERIFIED_EVENT")
        self.assertEqual(event["effective_date"], "2024-05-06")
        self.assertEqual(event["after_status"], "*ST")

    def test_title_like_text_without_effective_date_is_not_verified(self):
        text = "公司拟实施退市风险警示，股票简称由甲公司变更为*ST甲。"
        self.assertEqual(parse_event_text(text)["parse_status"], "REVIEW_REQUIRED")

    def test_effective_date_without_transition_is_not_verified(self):
        text = "实施退市风险警示的起始日：2024年5月6日。"
        self.assertEqual(parse_event_text(text)["parse_status"], "REVIEW_REQUIRED")

    def test_top_level_implementation_date_wins_over_historical_date(self):
        text = (
            "实施起始日为2024年5月6日。实施后A股简称为*ST世茂。"
            "A股股票简称由‘ST世茂’变更为‘*ST世茂’。"
            "公司股票于2023年5月5日起实施其他风险警示。"
        )
        event = parse_event_text(text)
        self.assertEqual(event["parse_status"], "VERIFIED_EVENT")
        self.assertEqual(event["effective_date"], "2024-05-06")

    def test_only_action_announcements_become_pdf_candidates(self):
        self.assertTrue(is_event_title("博信股份关于实施退市风险警示暨停牌的公告"))
        self.assertTrue(is_event_title("关于叠加实施其他风险警示的公告"))
        self.assertTrue(is_event_title("关于撤销退市风险警示暨继续实施其他风险警示的公告"))
        self.assertTrue(is_event_title("ST宁科关于实施退市风险警示并继续实施其他风险警示暨停牌的公告"))
        self.assertFalse(is_event_title("ST示例关于继续实施其他风险警示的公告"))
        self.assertFalse(is_event_title("关于可能被实施退市风险警示的提示性公告"))
        self.assertFalse(is_event_title("关于被实施其他风险警示相关事项的进展公告"))
        self.assertFalse(is_event_title("关于公司股票进入退市整理期交易的第二次风险提示公告"))

    def test_multicolumn_pdf_prefers_complete_later_transition(self):
        text = (
            "证券简称变更日期：2025年5月8日。将公司证券简称由‘格力地产’变更为‘珠\n"
            "，证券代码600185保持不变。\n免集团’。"
            "5月8日起由‘格力地产’变更为‘珠免集团’，公司股票证券代码600185保持不变。"
        )
        event = parse_event_text(text)
        self.assertEqual(event["parse_status"], "VERIFIED_EVENT")
        self.assertEqual(event["after_name"], "珠免集团")

    def test_labelled_transition_does_not_require_the_word_from(self):
        text = (
            "实施起始日为2025年4月30日。"
            "股票简称：‘波导股份’变更为‘*ST波导’。"
        )
        event = parse_event_text(text)
        self.assertEqual(event["parse_status"], "VERIFIED_EVENT")
        self.assertEqual(event["before_name"], "波导股份")
        self.assertEqual(event["after_name"], "*ST波导")

    def test_interleaved_effective_date_and_transition_is_parsed(self):
        text = (
            "变更后的股票证券简称：衢州发展。证券简称变更日期：2024年9月6日。"
            "公司证券简称将自2024‘新湖中宝’变更为‘衢州发展’，年9月6日起由公司证券代码保持不变。"
        )
        event = parse_event_text(text)
        self.assertEqual(event["parse_status"], "VERIFIED_EVENT")
        self.assertEqual(event["effective_date"], "2024-09-06")
        self.assertEqual(event["after_name"], "衢州发展")


if __name__ == "__main__":
    unittest.main()
