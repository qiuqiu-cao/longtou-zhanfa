"""Close three official probes and freeze four structured source controls."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import hashlib,json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text("utf-8"))
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
review=read(OUT/"gemini/GEMINI_OFFICIAL_THREE_CASE_REVIEW.json")
agreement=read(OUT/"gemini/NEXT_STRUCTURED_SOURCE_SCOPE_AGREEMENT.json")
assert review["status"]=="INDEPENDENT_THREE_CASE_OFFICIAL_AUDIT_SETTLED"
assert agreement["status"]=="AGREED_ON_STRUCTURED_SOURCE_SCOPE_WITH_STRICT_BOUNDS"
assert [r["code"] for r in agreement["target_codes"]]==["600361","603203","688368","603214"]
assert agreement["request_budget_max"]==4 and agreement["strict_technical_bounds"]["page_size"]==100
original=read(OUT/"JOINT_FROZEN_PROTOCOL.json")
assert sha(original["original_facts"])==original["original_facts_sha256"]
instant=datetime.now(timezone.utc)
names=["JOINT_FROZEN_PROTOCOL.json","ALL_THREE_CASE_SOURCE_RECEIPTS.json",
    "OFFICIAL_FIELDS_BOUND_BEFORE_PRICE_COMPARISON.json","THREE_CASE_FIELD_AND_PRICE_QUALIFICATION.json",
    "ONE_FIRST_COHORT_OMISSION_EVIDENCE.json","三例公告核验与首板漏入_结果.md",
    "gemini/GEMINI_OFFICIAL_THREE_CASE_REVIEW.md","gemini/GEMINI_OFFICIAL_THREE_CASE_REVIEW.json",
    "gemini/NEXT_STRUCTURED_SOURCE_SCOPE_AGREEMENT.md","gemini/NEXT_STRUCTURED_SOURCE_SCOPE_AGREEMENT.json",
    "CODEX_OFFICIAL_PEER_REVIEW.md","raw/600361_implementation.pdf","raw/688368_implementation.pdf"]
write(OUT/"completion.json",dict(completed_at=instant.isoformat(),status="JOINT_THREE_OFFICIAL_PROBES_SETTLED_PARTIAL_SOURCE_QUALIFICATION",
    source_collection_completed_at=read(OUT/"ALL_THREE_CASE_SOURCE_RECEIPTS.json")["at"],
    actual_exec_session=89109,actual_exec_exit_code=0,actual_peer="Antigravity Gemini 3.8 Flash High",
    actual_peer_independent_review_read=True,source_requests=5,index_requests=3,pdfs_captured=2,
    source_fields_verified_cases=2,differential_distribution_cases=2,implementation_unavailable_cases=1,
    index_coverage_truncated_cases=2,first_cohort_omission_evidence_count=1,
    original_rows_inserted=0,target_corrections_in_this_batch=0,model_fits=0,production_writes=0,
    complete_action_database=False,complete_label_or_cohort_repair=False,goal_60_achieved=False,
    artifact_sha256={name:sha(OUT/name) for name in names}))
destination=ROOT/"research"/instant.strftime("first_board_corporate_action_structured_probe_%Y%m%dT%H%M%SZ")
destination.mkdir(exist_ok=False)
protocol=dict(frozen_at=instant.isoformat(),deadline_utc=(instant+timedelta(minutes=30)).isoformat(),budget_minutes=30,
    status="ACTUAL_JOINT_STRUCTURED_FOUR_CODE_SCOPE_FROZEN_BEFORE_HTTP",codes=["600361","603203","688368","603214"],
    control_cases=agreement["target_codes"],source_scope="One raw RPT_SHAREBONUS_DET response per declared company;not a marketwide crawl",
    actual_peer="Antigravity Gemini 3.8 Flash High",collector_owner="Codex",independent_reviewer="Gemini",
    max_direct_http_requests=4,page_size=100,page_number=1,timeout_seconds=10,no_paging=True,no_retries=True,
    max_requests_per_code=1,on_access_denial="Stop;no bypass or alternate route",model_fits_allowed=0,
    training_arms=[],target_label_changes_allowed=0,production_writes_allowed=0,
    agreement=str(OUT/"gemini/NEXT_STRUCTURED_SOURCE_SCOPE_AGREEMENT.json"),
    agreement_sha256=sha(OUT/"gemini/NEXT_STRUCTURED_SOURCE_SCOPE_AGREEMENT.json"),
    collector_source=str(ROOT/"first_board_corporate_action_structured_probe.py"),
    collector_source_sha256=sha(ROOT/"first_board_corporate_action_structured_probe.py"),
    technical_source_url="https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock_feature/stock_fhps_em.py",
    local_technical_source_sha256=sha(OUT/"akshare_local_stock_fhps_em_source.py"),
    qualifications=["Companyidentity and dates","Units against official PDFs","Implementation vs plan status",
        "Presence of differential distribution flags and virtual diluted parameters","Revised historical fields and true capture clocks"],
    rejection="Nominal distributions alone cannot certify reference prices for differential cases or authorize full dataset repair",
    later_capture_not_original_signal_time=True,original_facts=original["original_facts"],
    original_facts_sha256=original["original_facts_sha256"],goal_60_achieved=False)
write(destination/"JOINT_FROZEN_PROTOCOL.json",protocol)
pointer_path=ROOT/"research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
write(destination/"PREVIOUS_JOINT_EXPERIMENT_POINTER.json",read(pointer_path))
pointer=dict(updated_at=instant.isoformat(),status="STRUCTURED_FOUR_CODE_SCOPE_FROZEN_REQUESTS_NOT_YET_STARTED",
    experiment_root=str(destination),protocol=str(destination/"JOINT_FROZEN_PROTOCOL.json"),
    actual_owner="Codex",actual_peer="Antigravity Gemini 3.8 Flash High",training_arms={},
    active_model_training=False,process_handle=None,model_fits=0,goal_60_achieved=False,
    prior_official_probe_completion=str(OUT/"completion.json"),original_deadline_utc=protocol["deadline_utc"],
    no_failed_model_retry=True,current_research_goal_still_active=True)
pointer_path.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding="utf-8")
print(str(destination))
