import json

import pandas as pd
import pytest

from announcement_fulltext_review_v228 import classify_text, review


def test_hot_concept_compliance_sentence_is_not_theme_denial():
    result = classify_text("公司不存在违规交易行为和通过炒作热点概念等不当行为配合减持的情况。")
    assert result["content_risk_class"] == "NO_CLASSIFIED_SUBSTANTIVE_RISK"


def test_routine_market_notice_with_profit_decline_is_not_hard_risk():
    result = classify_text(
        "股票交易风险提示：公司股价波动较大，归属于上市公司股东的净利润"
        "2,291万元，同比下降81.75%，公司经营正常。"
    )
    assert result["content_risk_class"] == "ROUTINE_MARKET_AND_PERFORMANCE_RISK"
    assert result["research_score"] == -4


def test_substantive_legal_risk_is_detected():
    result = classify_text("公司收到监管机构立案调查通知书。")
    assert result["content_risk_class"] == "SUBSTANTIVE_HARD_RISK"
    assert result["research_score"] == -20


def test_material_profit_impact_is_hard_risk():
    result = classify_text(
        "根据仲裁裁决，预计将对公司2026年度净利润减少6.2亿元左右，"
        "相关义务能否如期履行尚存在不确定性。"
    )
    assert result["content_risk_class"] == "SUBSTANTIVE_HARD_RISK"
    assert result["research_score"] == -20


def test_proxy_product_and_non_self_developed_denies_theme():
    result = classify_text(
        "公司销售的矢量网络分析仪均为代理产品，非公司自研产品，"
        "公司不进行相关产品的生产制造。"
    )
    assert result["content_risk_class"] == "HOT_THEME_CLARIFICATION_NEGATIVE"
    assert result["theme_thesis_veto"] is True


def test_archive_hash_mismatch_fails_closed(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    evidence = archive / "fulltext_evidence.csv"
    pd.DataFrame([{"code": "000001"}]).to_csv(evidence, index=False)
    (archive / "report.json").write_text(json.dumps({"evidence_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        review(str(archive), "2026-09-21T23:40:00+08:00", str(tmp_path / "out"))
