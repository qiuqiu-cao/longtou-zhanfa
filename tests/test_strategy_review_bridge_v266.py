import strategy_review_bridge_v266 as bridge


def _review():
    return {
        "asof": "2026-09-21T23:40:00+08:00",
        "signal_date": "2026-09-21",
        "status": "CASH",
        "market": {"emotion_regime": "加速", "limit_count_mainboard": 93, "failed_rate": 0.2,
                    "previous_limit_feedback": {"promotion_rate": 0.29}},
        "roles": {"emotion_anchors": [{"code": "001216", "name": "华瓷股份", "boards": 5,
                                         "role": "EMOTION_ANCHOR_NOT_BUY_CANDIDATE"}],
                  "highest_leader_potential": {"code": "001216", "name": "华瓷股份", "boards": 5,
                                                "research_score_not_probability": 77,
                                                "evidence_status": "INFERRED"},
                  "current_best_buy_opportunity_status": "UNAVAILABLE_AUCTION_OPENING_AND_EXECUTION_EVIDENCE"},
        "decision_readiness": {"actual_fill_status": "UNAVAILABLE"},
        "blocking_reasons": ["EXECUTION_MISSING"],
    }


def _audit():
    return {"live_ready": False, "governance": {"strategy_upgrade_allowed": False}}


def _contract():
    return {
        "status": "DIAGNOSTIC_ONLY", "latest_signal_date": "2026-08-31",
        "identity_review": {"allowed": True},
        "conditional_relay_review": {"allowed": False, "missing_fields": ["auction_gap", "theme"]},
        "trade_authorization": {"allowed": False},
    }


def test_bridge_is_fail_closed_and_explicitly_stale():
    report = bridge.build_bridge(_review(), _audit(), _contract(), reference_asof="2026-09-27T15:00:00+08:00")
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["decision"] == "CASH"
    assert report["freshness"]["status"] == "STALE"
    assert report["gates"]["identity_review_allowed"] is True
    assert report["gates"]["conditional_relay_review_allowed"] is False
    assert report["gates"]["trade_authorization"] is False
    assert "DAILY_REVIEW_STALE" in report["blocking_reasons"]
    assert "V265_MISSING_FIELDS_conditional_relay" in report["blocking_reasons"]


def test_bridge_never_authorizes_trade_without_live_audit():
    contract = _contract()
    contract["conditional_relay_review"]["allowed"] = True
    contract["conditional_relay_review"]["missing_fields"] = []
    report = bridge.build_bridge(_review(), {"live_ready": True, "governance": {"strategy_upgrade_allowed": True}}, contract,
                                 reference_asof="2026-09-21T23:40:00+08:00")
    assert report["gates"]["trade_authorization"] is False
    assert report["governance"]["no_order_submission"] is True
