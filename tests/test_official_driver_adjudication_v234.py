from official_driver_adjudication_v234 import adjudicate, available_at


def _row(**updates):
    row = {
        "signal_date": "2026-09-21", "official_published_at": "2026-09-18",
        "availability_basis": "DATE_ONLY_CONSERVATIVE_END_OF_DAY",
        "official_available_at": "", "conflict_terms": "",
        "expected_verdict": "OFFICIAL_FULLTEXT_MATCH_PREEXISTING",
    }
    row.update(updates)
    return row


def test_prior_date_fulltext_match_is_usable_but_not_buy_signal():
    result = adjudicate(
        _row(), "CONTROL_OR_CAPITAL_ACTION", "2026-09-21T15:00:00+08:00")
    assert result["official_evidence_usable_at_decision"]
    assert result["secondary_claim_fully_supported"]
    assert not result["positive_buy_reason_allowed"]


def test_same_day_date_only_source_is_after_cutoff():
    result = adjudicate(_row(
        official_published_at="2026-09-21",
        expected_verdict="OFFICIAL_MATCH_AFTER_CUTOFF_UNUSABLE"),
        "CONTROL_OR_CAPITAL_ACTION", "2026-09-21T15:00:00+08:00")
    assert result["after_cutoff_unusable"]
    assert not result["official_evidence_usable_at_decision"]


def test_material_amount_conflict_fails_positive_support():
    result = adjudicate(_row(
        conflict_terms="10,000万元", expected_verdict="OFFICIAL_DETAIL_CONFLICT"),
        "CONTROL_OR_CAPITAL_ACTION", "2026-09-21T15:00:00+08:00")
    assert result["material_detail_conflict"]
    assert not result["secondary_claim_fully_supported"]


def test_negative_clarification_is_separate_verdict():
    result = adjudicate(_row(
        expected_verdict="OFFICIAL_NEGATIVE_CLARIFICATION_MATCHED"),
        "NEGATIVE_OR_LIMITED_EXPOSURE", "2026-09-21T15:00:00+08:00")
    assert result["official_adjudication_verdict"] == "OFFICIAL_NEGATIVE_CLARIFICATION_MATCHED"


def test_exact_available_at_requires_timezone_and_is_converted():
    row = _row(
        availability_basis="EXACT_AVAILABLE_AT",
        official_available_at="2026-09-21T14:30:00+08:00")
    assert available_at(row).isoformat() == "2026-09-21T14:30:00+08:00"
