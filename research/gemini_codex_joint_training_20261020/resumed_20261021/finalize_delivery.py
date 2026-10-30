"""Finalize the real joint batch and the peer's append-only statistical correction."""
from pathlib import Path
from datetime import datetime
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import first_board_joint_seal_model as m
HERE=Path(__file__).resolve().parent
peer=m.JOINT/"gemini"
freeze=json.loads((m.JOINT/"JOINT_FROZEN_PROTOCOL.json").read_text("utf-8"))
assert freeze["status"]=="AGREED_BEFORE_NEW_FITS" and m.base.sha(m.__file__)==freeze["source_sha256"]
report=json.loads((m.OUTPUT/"report.json").read_text("utf-8"))
verification=json.loads((m.OUTPUT/"JOINT_SAVED_RESULT_VERIFICATION.json").read_text("utf-8"))
settlement=json.loads((peer/"GEMINI_JOINT_FINAL_SETTLEMENT.json").read_text("utf-8"))
correction=json.loads((peer/"GEMINI_JOINT_SETTLEMENT_CORRECTION.json").read_text("utf-8"))
progress=json.loads((peer/"GEMINI_PROGRESS.json").read_text("utf-8"))
assert report["status"]=="BOTH_FIXED_ARMS_REJECTED" and report["selection"]["selected"] is None and not report["goal_60_achieved"]
assert verification["status"]=="PASS_NO_FIT_REPLAY" and verification["verifier_fits"]==0
assert correction["status"]=="CORRECTION_APPENDED_ORIGINALS_UNCHANGED" and progress["status"]=="CORRECTED"
mapping={"protocol":m.JOINT/"JOINT_FROZEN_PROTOCOL.json","report":m.OUTPUT/"report.json","verification":m.OUTPUT/"JOINT_SAVED_RESULT_VERIFICATION.json",
    "complete_4724":m.OUTPUT/"COMPLETE4724_BOTH_ARMS_WITH_FROZEN_BASE.csv","complete_117_paired":m.OUTPUT/"COMPLETE117_DAILY_PAIRED_SELECTIONS_ALL_ARMS.csv"}
for key,path in mapping.items():assert m.base.sha(path)==settlement["source_sha256"][key]
for name in ["GEMINI_JOINT_FINAL_SETTLEMENT.json","GEMINI_JOINT_FINAL_SETTLEMENT.md"]:
    assert m.base.sha(peer/name)==m.base.sha(HERE/(name+".BEFORE_UNSUPPORTED_STATS_CORRECTION"))
assert m.base.sha(HERE/"UNSUPPORTED_GEMINI_STATISTICS_NOTICE.json")==correction["source_sha256"]["notice"]
assert m.base.sha(HERE/"ALL18_LOST_BASE_HITS_AND_REPORTED_FIRST_TIMES.csv")==correction["source_sha256"]["all18_csv"]
notice=json.loads((HERE/"UNSUPPORTED_GEMINI_STATISTICS_NOTICE.json").read_text("utf-8"))
assert notice["direct_time_recalculation"]["strictly_after10"]==3 and notice["direct_time_recalculation"]["denominator"]==18
proof=HERE/"ANTIGRAVITY_JOINT_FINAL.jpg"
assert proof.read_bytes().startswith(bytes.fromhex("ffd8ff"))
core_receipts=[]
for arm in m.ARMS:
    directory=m.OUTPUT/arm
    arm_report=json.loads((directory/"report.json").read_text("utf-8"))
    assert arm_report["trainer"]==m.OWNERS[arm] and not arm_report["pass_validation"]
    assert m.base.sha(directory/"source_before_fit.py")==freeze["source_sha256"]
    for path in (directory/"models").glob("*_BEFORE_FIT.json"):
        receipt=json.loads(path.read_text("utf-8"))
        assert datetime.fromisoformat(receipt["started_at"])>datetime.fromisoformat(freeze["agreed_at"])
        core_receipts.append(str(path))
assert len(core_receipts)==4 and not (m.OUTPUT/"evaluation").exists()
artifacts=[*mapping.values(),peer/"GEMINI_JOINT_FINAL_SETTLEMENT.json",peer/"GEMINI_JOINT_FINAL_SETTLEMENT.md",peer/"GEMINI_JOINT_SETTLEMENT_CORRECTION.json",peer/"GEMINI_JOINT_SETTLEMENT_CORRECTION.md",proof,m.OUTPUT/"联合训练_首封时序与封单_结果.md"]
result=dict(at=m.base.now(),status="JOINT_BATCH_TRAINED_VERIFIED_SETTLED_AND_CORRECTED",previous_goal_turn="PROGRESS:Gemini4 audit completed and rejectedIPOtrial;newsourceprobe succeededbeforeuserpause",current_turn="PROGRESS:actual AntigravityGemini collaboration resumed,jointproposal/rebuttals/acknowledgment,668datedsources qualified,both agents fitted separatearms,fullreplay and independent settlement completed,unsupported peer statistics retracted",
    actual_partner="Antigravity Gemini3.8 Flash High",collaboration_conversation="cbfaa95f-7eb8-46fd-8195-6ace0c4fe35e",peer_agreed_before_all_four_core_fits=True,
    results=dict(BASE=dict(primary_hits=52,non_one_price_hits=48),SEAL_TIME_64=dict(trainer="Gemini",primary_hits=40,non_one_price_hits=34),SEAL_TIME_FUND_66=dict(trainer="Codex",primary_hits=52,non_one_price_hits=41),known_selected_each=117),
    logical_heads=2,hgb_core_fits=4,temporal_platt_fits=2,total_estimator_fits=6,monthly_fits=0,baseline_refits=0,second_stage_refits=0,verifier_fits=0,
    source_dates=668,original_first_board_rows=28503,validation_rows=4724,validation_unknown=10,original_first_unknown=126,
    statistical_correction=dict(peer_originals_preserved=True,unsupported72_and68percent_retracted=True,actual_late_first_seals_among18_lost_baseline_hits=3,theme_fields_unavailable=True,causal_market_explanations_not_certified=True),
    source_scope="Historical provider daily reports acquired later;no verified exchange queue/capitalidentity/originalHTTPcapture;originaluniverse-survivorship remains",
    goal_60_achieved=False,goal_completion_claimed=False,deployed=False,production_changes=False,actual_trades=0,new_forward_results=0,
    batch_end="Fixed2armcomparisoncomplete;bothrejected,no3rdarmorretuning",monitoring=dict(automation_id="gemini",interval_minutes=10,unchanged_results_quiet=True),
    artifacts={str(path.relative_to(ROOT)):m.base.sha(path) for path in artifacts})
with (m.OUTPUT/"completion.json").open("x",encoding="utf-8") as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
m.base.write_json(HERE/"MONITOR_LAST_NOTIFIED_RESULT.json",dict(at=m.base.now(),result_reported_in_current_thread=True,root_report_sha256=m.base.sha(m.OUTPUT/"report.json"),peer_correction_sha256=m.base.sha(peer/"GEMINI_JOINT_SETTLEMENT_CORRECTION.json"),same_result_should_remain_quiet=True,goal_60_achieved=False))
print(json.dumps({key:result[key] for key in ("status","logical_heads","hgb_core_fits","temporal_platt_fits","monthly_fits","goal_60_achieved")},ensure_ascii=True))
