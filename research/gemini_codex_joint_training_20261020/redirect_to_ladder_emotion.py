"""Apply the human's explicit research-priority correction without any fits."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
POINTER=Path(__file__).resolve().parent/"CURRENT_JOINT_EXPERIMENT.json"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
previous=json.loads(POINTER.read_text("utf-8"))
prior=Path(previous["experiment_root"])
assert prior.name=="first_board_corporate_action_structured_probe_20261021T070745Z"
assert not (prior/"COLLECTION_STARTED.json").exists()
instant=datetime.now(timezone.utc)
write(prior/"DEFERRED_BY_LATEST_USER_PRIORITY.json",dict(at=instant.isoformat(),
    status="SOURCE_PROBE_NOT_STARTED_DEFERRED_BY_LATEST_HUMAN_DIRECTION",actual_requests=0,model_fits=0,
    reason="Human asks whether direction is wrong;prioritizes historical ladder,sealing quality and precise emotion judgment;stop expanding dividend/announcement audit",
    original_protocol_preserved=True,goal_remains_active=True))
destination=ROOT/"research"/instant.strftime("gemini_codex_joint_ladder_emotion_design_%Y%m%dT%H%M%SZ")
destination.mkdir(exist_ok=False)
facts=ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
plan=dict(started_at=instant.isoformat(),deadline_utc=(instant+timedelta(minutes=30)).isoformat(),budget_minutes=30,
    status="DESIGN_DISCUSSION_PENDING_ACTUAL_PEER_NO_TRAINING_YET",
    human_priority=["Precise emotion judgment first","Historical board-ladder structure and tier feedback",
        "Sealing quality as observable market-acceptance evidence","Emotion-dependent interaction,not announcement business-logic veto"],
    proposed_distinct_evidence="Full tier2/3/4/5+ structure,ladder gaps and sequential tier-specific promotion/loss feedback;existing qualified seal fields interpreted conditionally",
    old_experiments_not_retried=["Standalone SEAL_TIME64/SEAL_TIME_FUND66","Standalone REOPEN","Static phases or static symbolic17","Event path67/72"],
    original_facts=str(facts),original_facts_sha256=sha(facts),
    original_goal="S close->next official S+1close-limit1to2;full-dateTop1>=60%,same-stock2to3recheckseparate",
    model_fits_allowed_in_design=0,max_later_challenger_arms=2,production_writes_allowed=False,
    source_audit_not_a_block_on_legal_historical_research=True,
    unresolved_source_issues_preserved=True,actual_peer="Antigravity Gemini 3.8 Flash High",
    previous_goal_turn_classification="PROGRESS:one confirmed target correction,one corroborated first-pool omission;no predictive gain claimed")
write(destination/"DESIGN_SCOPE_BEFORE_FEATURE_RESULTS.json",plan)
write(destination/"PREVIOUS_JOINT_EXPERIMENT_POINTER.json",previous)
current=dict(updated_at=instant.isoformat(),status="USER_REDIRECTED_TO_EMOTION_LADDER_SEAL_DESIGN_AWAIT_ACTUAL_PEER",
    experiment_root=str(destination),design_scope=str(destination/"DESIGN_SCOPE_BEFORE_FEATURE_RESULTS.json"),
    actual_peer="Antigravity Gemini 3.8 Flash High",actual_owner="Joint design discussion",training_arms={},
    active_model_training=False,process_handle=None,model_fits=0,goal_60_achieved=False,
    data_audit_expansion_deferred=True,prior_unstarted_source_probe=str(prior),
    original_deadline_utc=plan["deadline_utc"],current_research_goal_still_active=True)
POINTER.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding="utf-8")
print(str(destination))
