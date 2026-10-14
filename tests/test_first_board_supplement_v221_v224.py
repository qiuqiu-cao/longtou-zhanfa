import json

import pandas as pd
import pytest

from first_board_open_confirmation_v221 import evaluate
from first_board_theme_catalyst_v222 import verify
from first_board_forward_settlement_v224 import settle
from first_board_selective_policy_v223 import apply_policy
from first_board_daily_supplement_v224 import parser as daily_parser


def snapshot():
    return {"freeze_id": "f", "signal_date": "2026-09-21",
            "candidate": {"code": "002349", "name": "精华制药"}}


def auction(**updates):
    row = {"code": "002349", "trade_date": "2026-09-22",
           "available_at": "2026-09-22T09:25:00+08:00",
           "auction_gap_pct": 3, "auction_amount_ratio_pct": 4,
           "match_amount_decay_proxy": .1, "source_status": "PUBLIC_SNAPSHOT"}
    row.update(updates)
    return row


def bars():
    result = []
    for i in range(31, 37):
        price = 10 + (i - 31) * .02
        result.append({"code": "002349", "minute": f"09:{i:02d}",
                       "available_at": f"2026-09-22T09:{i:02d}:59+08:00",
                       "open": price, "high": price + .03, "low": price - .01,
                       "close": price + .02, "volume": 100, "amount": 100 * (price + .01),
                       "volume_unit": "share", "amount_unit": "CNY",
                       "limit_price": 11})
    return result


def theme(lane="VERIFIED_REAL_THEME"):
    return {"code": "002349", "trade_date": "2026-09-22",
            "available_at": "2026-09-22T09:35:59+08:00", "sync_ratio": .8,
            "coverage_ratio": .9, "peer_count": 4, "evidence_lane": lane}


def test_v221_missing_is_unavailable():
    assert evaluate(snapshot(), None, None, None, "2026-09-22")["decision"] == "DATA_UNAVAILABLE"


def test_v221_uses_0936_price_after_0935_confirmation():
    result = evaluate(snapshot(), auction(), bars(), theme(), "2026-09-22")
    assert result["decision"] == "PAPER_CONFIRM_AFTER_0935"
    assert result["first_causal_price_opportunity"]["minute"] == "09:36"
    assert result["entry_authorized"] is False


def test_v221_observed_roster_never_becomes_verified_theme():
    result = evaluate(snapshot(), auction(), bars(), theme("OBSERVED_THEME_ROSTER"), "2026-09-22")
    assert result["decision"] == "PAPER_RESEARCH_CONFIRM_UNVERIFIED_THEME"


def test_v221_rejects_future_auction_timestamp():
    with pytest.raises(ValueError):
        evaluate(snapshot(), auction(available_at="2026-09-22T09:26:00+08:00"),
                 bars(), theme(), "2026-09-22")


def test_v222_requires_membership_and_catalyst_before_cutoff():
    matrix = pd.DataFrame([{"signal_date": "2026-09-21", "theme_name": "医药"}])
    evidence = pd.DataFrame([{
        "signal_date": "2026-09-21", "theme_name": "医药",
        "membership_status": "VERIFIED_PRIMARY_SOURCE", "membership_source_url": "https://a",
        "membership_available_at": "2026-09-21T14:00:00+08:00",
        "catalyst_status": "VERIFIED_ACTIVE", "catalyst_source_url": "https://b",
        "catalyst_published_at": "2026-09-21T13:00:00+08:00",
        "catalyst_available_at": "2026-09-21T14:00:00+08:00", "expectation_status": "NEW",
    }])
    out = verify(matrix, evidence, "2026-09-21T15:05:00+08:00")
    assert bool(out.iloc[0].verified_real_theme)
    evidence.loc[0, "catalyst_available_at"] = "2026-09-21T16:00:00+08:00"
    assert not bool(verify(matrix, evidence, "2026-09-21T15:05:00+08:00").iloc[0].verified_real_theme)


def test_v224_pending_does_not_invent_outcome():
    out = settle(snapshot(), None, None, None)
    assert out["status"] == "PENDING_OR_PARTIAL"
    assert out["phases"]["promotion"]["status"] == "PENDING_DATA"
    assert out["entry_authorized"] is False


def test_v223_abstention_never_replaces_frozen_top1():
    scored = pd.DataFrame([
        {"signal_date": "2026-09-01", "code": "000001", "name": "A",
         "community_probability": .9, "b2_positive_probability": .1,
         "b1_big_loss_probability": .9, "close_quality_gate": "True",
         "fill_status": "FILLED_OPEN_PROXY", "promotion_label": 0,
         "net_return_B_plus_2": -.1, "big_loss_B_plus_1": 1},
        {"signal_date": "2026-09-01", "code": "000002", "name": "B",
         "community_probability": .8, "b2_positive_probability": .9,
         "b1_big_loss_probability": .1, "close_quality_gate": "True",
         "fill_status": "FILLED_OPEN_PROXY", "promotion_label": 1,
         "net_return_B_plus_2": .1, "big_loss_B_plus_1": 0},
    ])
    frozen, report = apply_policy(scored)
    assert frozen.iloc[0].code == "000001"
    assert frozen.iloc[0].policy_decision == "ABSTAIN_TO_CASH"
    assert report["selective_policy"]["decisions"] == 0


def test_v224_daily_parser_accepts_official_fulltext_adjudication():
    args = daily_parser().parse_args([
        "--snapshot", "snapshot.json", "--decision-date", "2026-09-22",
        "--theme-matrix", "matrix.csv", "--theme-evidence", "evidence.csv",
        "--theme-cutoff", "2026-09-22T09:15:00+08:00",
        "--history-rows", "history.csv",
        "--promotion-predictions", "predictions.csv",
        "--current-watchlist", "watchlist.csv",
        "--official-driver-adjudication", "v234/report.json",
        "--output", "out",
    ])
    assert args.official_driver_adjudication == "v234/report.json"
