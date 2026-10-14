from relay_leader_open_confirmation_v237 import evaluate


def snapshot():
    return {
        "signal_date": "2026-09-21",
        "role_candidates": [
            {"role": "height_emotion_anchor", "candidate": {"code": "001216", "name": "华瓷股份", "boards": 5}},
            {"role": "close_relay_challenger", "candidate": {"code": "603230", "name": "内蒙新华", "boards": 4}},
        ],
    }


def auction(**updates):
    row = {"code": "603230", "trade_date": "2026-09-22",
           "available_at": "2026-09-22T09:25:00+08:00",
           "auction_gap_pct": 3, "auction_amount_ratio_pct": 4,
           "match_amount_decay_proxy": .1, "source_status": "PUBLIC_SNAPSHOT"}
    row.update(updates)
    return row


def bars(anchor_return=1, candidate_weak=False):
    rows = []
    for code, previous, ret in (("603230", 10, 2), ("001216", 20, anchor_return)):
        for minute in range(31, 37):
            lift = (minute - 31) * .002
            close = previous * (1 + ret / 100 + lift)
            if candidate_weak and code == "603230":
                close = previous * (1 - .02 - lift)
            rows.append({
                "code": code, "minute": f"09:{minute:02d}",
                "available_at": f"2026-09-22T09:{minute:02d}:59+08:00",
                "open": close - .01, "high": close + .02, "low": close - .02,
                "close": close, "volume": 100 + minute,
                "amount": (100 + minute) * close,
                "previous_close": previous, "limit_price": previous * 1.1,
                "volume_unit": "share", "amount_unit": "CNY",
            })
    return rows


def sync(kind="symbolic", ratio=.85):
    return {
        "code": "603230", "trade_date": "2026-09-22",
        "available_at": "2026-09-22T09:35:59+08:00",
        "business_sync_ratio": ratio if kind == "business" else 0,
        "business_coverage_ratio": .8 if kind == "business" else 0,
        "business_peer_count": 3 if kind == "business" else 0,
        "business_evidence_lane": "VERIFIED_REAL_THEME" if kind == "business" else "UNAVAILABLE",
        "symbolic_sync_ratio": ratio if kind == "symbolic" else 0,
        "symbolic_coverage_ratio": .9 if kind == "symbolic" else 0,
        "symbolic_peer_count": 5 if kind == "symbolic" else 0,
        "symbolic_evidence_lane": "VERIFIED_SYMBOLIC_ROSTER" if kind == "symbolic" else "UNAVAILABLE",
        "symbolic_evidence_id": "v217-freeze" if kind == "symbolic" else "",
    }


def test_missing_data_stays_unavailable():
    result = evaluate(snapshot(), None, None, None, "2026-09-22")
    assert result["decision"] == "DATA_UNAVAILABLE"
    assert not result["entry_authorized"]


def test_symbolic_ladder_never_becomes_real_theme_confirmation():
    result = evaluate(snapshot(), auction(), bars(), sync("symbolic"), "2026-09-22")
    assert result["decision"] == "PAPER_CONFIRM_SYMBOLIC_LADDER_ONLY"
    assert result["first_causal_price_opportunity"]["minute"] == "09:36"


def test_verified_business_ladder_can_paper_confirm():
    result = evaluate(snapshot(), auction(), bars(), sync("business"), "2026-09-22")
    assert result["decision"] == "PAPER_CONFIRM_AFTER_0935"
    assert not result["entry_authorized"]


def test_anchor_break_without_strong_challenger_is_retreat_contagion():
    result = evaluate(snapshot(), auction(), bars(anchor_return=-8, candidate_weak=True),
                      sync("symbolic", .5), "2026-09-22")
    assert result["decision"] == "CANCEL_RETREAT_CONTAGION"


def test_anchor_break_with_strong_challenger_and_ladder_is_core_switch_only():
    result = evaluate(snapshot(), auction(), bars(anchor_return=-8),
                      sync("business", .9), "2026-09-22")
    assert result["decision"] == "PAPER_CORE_SWITCH_AFTER_0935"


def test_overheated_auction_cancels_before_open_confirmation():
    result = evaluate(snapshot(), auction(auction_gap_pct=8,
                                         auction_amount_ratio_pct=1,
                                         match_amount_decay_proxy=.5),
                      bars(), sync(), "2026-09-22")
    assert result["decision"] == "CANCEL_AT_0925"
