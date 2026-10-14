import pandas as pd

from secondary_driver_claim_archive_v231 import classify_claim, parse_page, theme_summary


HTML = b"""<!doctype html><html><body>
<div data-stock-card><a href='/stock/detail/SZ000001'>\xe6\xa6\x9c1</a><div><p><a>000001</a><a href='/stock/detail/SZ000001'>\xe5\x8d\x8e\xe7\x94\xb2</a></p>
<div class='text-sm line-clamp-2'><span class='font-bold' title='\xe5\x8c\xbb\xe8\x8d\xaf'>[x]</span>
<span class='text-secondary' title='\xe5\x85\xac\xe5\x8f\xb8\xe6\x8b\x9f\xe5\x9b\x9e\xe8\xb4\xad\xe8\x82\xa1\xe4\xbb\xbd'>claim</span></div></div></div>
<div data-stock-card><div><p><a>000002</a><a href='/stock/detail/SZ000002'>\xe5\x8d\x8e\xe4\xb9\x99</a></p>
<div class='text-sm line-clamp-2'><span class='font-bold' title='\xe5\x8c\xbb\xe8\x8d\xaf'>[x]</span>
<span class='text-secondary' title='\xe5\x85\xac\xe5\x8f\xb8\xe4\xb8\x8d\xe7\x94\x9f\xe4\xba\xa7\xe5\x85\x89\xe5\x88\xbb\xe8\x83\xb6\xe4\xba\xa7\xe5\x93\x81'>claim</span></div></div></div>
</body></html>"""


def _source():
    return {"source_id": "s1", "url": "https://example.com", "source_tier": "C",
            "claim_status": "secondary_market_narrative",
            "available_at": "2026-09-21T23:38:00+08:00", "published_at": None,
            "page_sha256": "x"}


def test_parse_page_preserves_claim_and_negative_clarification():
    rows = parse_page(HTML, "2026-09-21", _source())
    assert set(rows.code) == {"000001", "000002"}
    first = rows[rows.code.eq("000001")].iloc[0]
    second = rows[rows.code.eq("000002")].iloc[0]
    assert first.theme_tags == "医药"
    assert first["name"] == "华甲"
    assert first.driver_claim_category == "CONTROL_OR_CAPITAL_ACTION"
    assert bool(second.negative_or_limited_exposure)
    assert not bool(second.positive_buy_reason_allowed)


def test_recency_words_do_not_become_verified_first_time():
    result = classify_claim("公司近期拟回购股份")
    assert result["recency_language_status"] == "RECENCY_LANGUAGE_PRESENT_NOT_FIRST_TIME_VERIFIED"
    assert result["driver_claim_category"] == "CONTROL_OR_CAPITAL_ACTION"


def test_static_parent_description_is_not_mislabeled_as_control_action():
    result = classify_claim("公司间接控股股东为中国台湾上市公司，主要产品为印制电路板")
    assert result["driver_claim_category"] == "PRODUCT_OR_BUSINESS_DESCRIPTION"
    assert "CONTROL_OR_CAPITAL_ACTION" not in result["all_driver_claim_categories"]


def test_theme_summary_does_not_invent_common_catalyst():
    claims = parse_page(HTML, "2026-09-21", _source())
    source_doc = {
        "signal_date": "2026-09-21", "sources": [_source()],
        "groups": [{"theme_group_id": "drug", "theme": "医药",
                    "codes": ["000001", "000002"], "source_id": "s1"}],
    }
    summary, overlay = theme_summary(claims, source_doc)
    assert summary.iloc[0].common_driver_status == (
        "INDIVIDUAL_DRIVER_CLAIMS_COMMON_CATALYST_NOT_DEMONSTRATED")
    assert not bool(summary.iloc[0].common_driver_verified)
    assert overlay.iloc[0].catalyst_published_at == ""
    assert overlay.iloc[0].expectation_status == "UNAVAILABLE"


def test_same_stock_repeated_in_two_theme_sections_is_merged():
    repeated = b"""<div data-stock-card><div><p><a>000001</a><a href='/stock/detail/SZ000001'>
    \xe5\x8d\x8e\xe7\x94\xb2</a></p><div class='text-sm line-clamp-2'>
    <span class='font-bold' title='\xe5\x8c\xbb\xe8\x8d\xaf'>[x]</span>
    <span class='text-secondary' title='\xe5\x85\xac\xe5\x8f\xb8\xe6\x8b\x9f\xe5\x9b\x9e\xe8\xb4\xad\xe8\x82\xa1\xe4\xbb\xbd'>claim</span>
    </div></div></div>"""
    duplicated = HTML.replace(b"</body>", repeated + b"</body>")
    rows = parse_page(duplicated, "2026-09-21", _source())
    first = rows[rows.code.eq("000001")].iloc[0]
    assert first.card_occurrence_count == 2
