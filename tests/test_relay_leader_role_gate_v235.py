import pandas as pd

from relay_leader_role_gate_v235 import build


def _scores():
    common = {
        "signal_date": "2026-09-21", "announcement_hard_veto": False,
        "announcement_risk_tier": "NO_ACTIVE_RISK", "missing_evidence": "",
        "factor_verified_real_theme": 0, "factor_turnover_quality": 2,
        "repair_span_minutes": 0, "symbolic_tags": "",
    }
    return pd.DataFrame([
        {**common, "code": "001216", "name": "华瓷股份", "boards": 5,
         "research_score_NOT_PROBABILITY": 4,
         "board_path_status": "ONE_PRICE_BUYABILITY_UNPROVEN", "symbolic_tags": "华"},
        {**common, "code": "002285", "name": "世联行", "boards": 4,
         "research_score_NOT_PROBABILITY": 14,
         "board_path_status": "LOCKED_NO_REOPEN"},
        {**common, "code": "603230", "name": "内蒙新华", "boards": 4,
         "research_score_NOT_PROBABILITY": 16, "repair_span_minutes": 12.15,
         "board_path_status": "QUICK_REPAIR", "symbolic_tags": "华|新"},
        {**common, "code": "600448", "name": "华纺股份", "boards": 3,
         "research_score_NOT_PROBABILITY": -5, "repair_span_minutes": 200,
         "board_path_status": "PROLONGED_CONTEST", "factor_turnover_quality": -4},
    ])


def _review(verified=False):
    eligibility = "VERIFIED_ELIGIBLE" if verified else "SECURITY_STATUS_UNAVAILABLE_OR_INELIGIBLE"
    return {
        "signal_date": "2026-09-21",
        "relay_candidates": [
            {"code": "001216", "turnover": .86, "amount": 51_000_000,
             "eligibility_status": eligibility, "theme_evidence": {"status": "INFERRED_OR_LIMITED"}},
            {"code": "002285", "turnover": 14.7, "amount": 859_000_000,
             "eligibility_status": eligibility, "theme_evidence": {"status": "INFERRED_OR_LIMITED"}},
            {"code": "603230", "turnover": 10.6, "amount": 557_000_000,
             "eligibility_status": eligibility, "theme_evidence": {"status": "INFERRED_OR_LIMITED"}},
            {"code": "600448", "turnover": 39.3, "amount": 938_000_000,
             "eligibility_status": eligibility, "theme_evidence": {"status": "UNAVAILABLE"}},
        ],
    }


def test_height_anchor_is_not_automatic_close_entry():
    rows, duel, report = build(_scores(), _review())
    assert report["height_emotion_anchor"]["code"] == "001216"
    assert report["height_emotion_anchor"]["relay_role"] == "HEIGHT_EMOTION_ANCHOR_NOT_CLOSE_ENTRY"
    assert not bool(rows.loc[rows.code.eq("001216"), "close_relay_challenger_eligible"].iloc[0])
    assert not duel.empty


def test_quick_repair_wins_cross_height_tradable_challenger_role():
    _, _, report = build(_scores(), _review())
    assert report["close_relay_challenger"]["code"] == "603230"
    assert report["close_relay_challenger"]["relay_role"] == "DIVERGENCE_ABSORBED_RELAY_CHALLENGER"
    assert report["official_relay_primary"] is None


def test_prolonged_contest_and_blowoff_turnover_are_not_close_candidates():
    rows, _, _ = build(_scores(), _review())
    row = rows[rows.code.eq("600448")].iloc[0]
    assert row.relay_role == "PROLONGED_CONTEST_WATCH_ONLY"
    assert not bool(row.close_relay_challenger_eligible)


def test_verified_security_alone_cannot_create_official_primary():
    scores = _scores()
    scores.loc[scores.code.eq("603230"), "factor_verified_real_theme"] = 6
    rows, _, report = build(scores, _review(verified=True))
    candidate = rows[rows.code.eq("603230")].iloc[0]
    assert candidate.verified_real_theme
    assert candidate.security_scope_verified
    assert not candidate.official_primary_gate_passed
    assert report["official_relay_primary"] is None
