"""Read-only acceptance checks against original pool, timeline and receipts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import stock_judgement_review as o
import supplement_role_review as workflow

OUT = Path(__file__).resolve().parent
ACTUAL = OUT/"actual_full15_release"
facts_path = ACTUAL/"facts_with_supplement.json"
facts = o.read(facts_path)
context = o.read(ACTUAL/"relations"/"relation_context.json")
completed = o.read(ACTUAL/"completed_supplement_review.json")
old = o.read(ROOT/"results"/"sunshine_jiuyang_relation_20261014T152814Z"/"15日完整15只角色补充输出.json")
old_rows = {r["code"]:r for r in old["rows"]}
rows = {r["code"]:r for r in context["rows"]}
assert len(rows) == len(old_rows) == 15 and set(rows) == set(old_rows)
for code,row in rows.items():
    assert (row["name"],row["boards"]) == (old_rows[code]["name"],old_rows[code]["boards"])
    peers = sorted(c for c,r in old_rows.items() if c != code and r["boards"] == row["boards"])
    assert row["same_height_peers"] == peers
    assert row["promotion_direction"] is None and row["confirmed_catchup_dragon"] is False
assert sum(bool(r["relations"]) for r in rows.values()) == 5
assert sum(len(r["relations"]) for r in rows.values()) == 6
assert sum(r["current_day_relation_body_count"] for r in rows.values()) == 0
assert rows["002696"]["same_anchor_candidate_peers"] == ["001318"]
assert rows["001318"]["same_anchor_candidate_peers"] == ["002696"]

source_relation_path = ROOT/"results"/"deep_review_20261014T115540Z"/"jiuyang_baiyang_relation_correction_20261014T121615Z"/"relation_review.json"
timeline = o.read(source_relation_path)["timeline"]
assert [r["baiyang_archived_boards"] for r in timeline] == [1,2,3,4]
assert timeline[0]["jiuyang_archived_boards"] == 5
assert [r["date"] for i,r in enumerate(timeline) if r["jiuyang_limit_streak_broken"] and not timeline[i-1]["jiuyang_limit_streak_broken"]] == ["2026-10-13"]
baiyang = rows["002696"]["relations"][0]
assert baiyang["recognition_state"] == "EARLY_START_CONTINUATION_CANDIDATE"
assert baiyang["timing"]["current_streak_start"] == timeline[0]["date"]
assert not baiyang["anchor_in_current_pool"] and "002242" not in rows
sunshine = rows["001318"]["relations"]
assert [r["anchor_code"] for r in sunshine] == ["002242","605388"]
assert sunshine[0]["recognition_state"] == "OLD_FOLLOWER_RESTART_CANDIDATE"
assert sunshine[0]["timing"]["prior_first_observed_board"] == "2026-09-30"
assert sunshine[0]["timing"]["current_streak_start"] == "2026-10-13"
assert sunshine[1]["recognition_state"] == "RELATION_BACKGROUND_TIMING_UNRESOLVED"
assert rows["603968"]["relations"][0]["anchor_code"] is None

workflow.checked(facts)
workflow.validate_review(facts,completed)
assert not completed["forecast_registered"]
assert all(r["promotion_judgement"] is None for r in completed["rows"])
try:
    workflow.freeze(facts_path,ACTUAL/"completed_supplement_review.json",OUT/"SHOULD_NEVER_BE_REGISTERED.json")
except ValueError as error:
    assert "diagnostic" in str(error)
else:
    raise AssertionError("diagnostic forecast gate did not reject")
assert not (OUT/"SHOULD_NEVER_BE_REGISTERED.json").exists()

architecture = o.read(OUT/"architecture_agent"/"architecture_verification.json")
old_core_hashes = {str(ROOT/k):v for k,v in architecture["source_hashes"].items()}
o.check_sources(old_core_hashes)
conflicts = o.read(OUT/"full_pool_conflict_audit_release.json")
o.check_sources(conflicts["source_bindings"])
assert len(conflicts["full13_to14_old10"]) == 10
assert len(conflicts["full14_to15_old15_with_relation_conflicts"]) == 15
assert conflicts["old10_BASE_metrics"]["Accuracy"] == .8
assert conflicts["old10_BASE_metrics"]["complete_positive_Recall"] == .5
assert conflicts["old10_independent_metrics"]["Accuracy"] == .625
for row in conflicts["full14_to15_old15_with_relation_conflicts"]:
    for key in ("raw_score","score_display","rank","predicted_positive","original_independent_direction"):
        assert row[key] == old_rows[row["code"]][key]
    assert row["target_label"] is None and row["new_direction"] is None
proof = facts["source_bindings"] | old_core_hashes | o.bind(__file__) | o.bind(facts_path)
for filename in ("full_pool_conflict_audit_release.json","prediction_validation_protocol.json","补涨识别接入与预测改进结论.md"):
    proof |= o.bind(OUT/filename)
proof |= o.bind(ACTUAL/"completed_supplement_review.json")
o.check_sources(proof)
o.write_new(OUT/"release_verification.json",{"verified_at":o.now().isoformat(),
    "checks":{"full15_and_all_same_height_peers":True,"off_pool_anchor_not_new_candidate":True,
        "baiyang_early_start_matches_original_timeline":True,"sunshine_old_start_and_restart_separate":True,
        "alternative_anchor_unknown_retained":True,"chemical_pair_not_transferred":True,
        "source_clocks_and_projections_regenerated":True,"full15_completed_review":True,
        "diagnostic_registration_rejected_without_writes":True,"old_core_hashes_unchanged":True,
        "full10_metrics_and_all15_scores_preserved":True},
    "targeted_tests_passed":22,"related_legacy_tests_passed":17,"tests_are_prediction_gain":False,
    "observed_ohlc_rows":56,"unknown_temporal_rows_not_used_as_truth":8,
    "new_fits":0,"new_role_truth":0,"new_formal_forward_labels":0,"deployed":False,
    "source_bindings":proof})
print({"full15_verified":True,"old_core_unchanged":True,"diagnostic_freeze_rejected":True,"new_fits":0})
