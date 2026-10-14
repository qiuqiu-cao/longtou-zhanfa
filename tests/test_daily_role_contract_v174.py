import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v174_authority_hashes_and_current_replay_are_consistent():
    authority = json.loads((
        ROOT / "results/v174/daily_role_contract_20260916_authority.json"
    ).read_text(encoding="utf-8"))
    review_path = ROOT / authority["daily_review"]
    markdown_path = ROOT / authority["daily_markdown"]
    assert _sha256(review_path) == authority["daily_review_sha256"]
    assert _sha256(markdown_path) == authority["daily_markdown_sha256"]

    review = json.loads(review_path.read_text(encoding="utf-8"))
    replay = authority["current_replay"]
    assert review["pool_summary"]["closing_limit_mainboard"] == replay[
        "closing_candidate_count"]
    assert review["pool_summary"]["first_board"] == replay["first_board_count"]
    assert review["pool_summary"]["relay"] == replay["relay_count"]
    assert [row["code"] for row in review["roles"]["emotion_anchors"]] == replay[
        "emotion_anchor_codes"]
    for key in (
        "current_market_core", "first_board_primary_candidate",
        "relay_primary_candidate", "comprehensive_primary_candidate",
        "current_best_buy_opportunity",
    ):
        assert review["roles"][key] is replay[key] is None
    assert review["status"] == replay["decision"] == "CASH"


def test_v174_contract_never_uses_height_as_automatic_trading_role():
    authority = json.loads((
        ROOT / "results/v174/daily_role_contract_20260916_authority.json"
    ).read_text(encoding="utf-8"))
    contract = authority["contract"]
    assert contract["highest_board_cannot_auto_fill_market_core"] is True
    assert contract["highest_board_cannot_auto_fill_group_primary"] is True
    assert contract["maximum_one_comprehensive_primary"] is True
    assert contract["entry_authorized"] is False
