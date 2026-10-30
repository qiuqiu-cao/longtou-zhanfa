"""Freeze the actual peer-agreed three-company source qualification before HTTP."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import hashlib,json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text("utf-8"))
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
agreement=read(OUT/"gemini/NEXT_BOUNDED_OFFICIAL_SCOPE_AGREEMENT.json")
assert agreement["status"]=="AGREED_ON_3_CASE_BOUNDED_SCOPE_BEFORE_NEW_PROTOCOL"
assert {(c["code"],c["source_date"]) for c in agreement["agreed_3_cases"]}=={
    ("600361","2024-06-28"),("603203","2026-06-08"),("688368","2026-06-09")}
instant=datetime.now(timezone.utc)
destination=ROOT/"research"/instant.strftime("first_board_corporate_action_official_probe_%Y%m%dT%H%M%SZ")
destination.mkdir(exist_ok=False)
facts=ROOT/"research/first_second_stage_20261020_run1/all_first_and_second_boards_WITH_SEPARATE_LABELS.csv"
protocol=dict(frozen_at=instant.isoformat(),deadline_utc=(instant+timedelta(minutes=30)).isoformat(),budget_minutes=30,
    status="ACTUAL_JOINT_THREE_CASE_SOURCE_SCOPE_FROZEN_BEFORE_HTTP",cases=agreement["agreed_3_cases"],
    actual_peer="Antigravity Gemini 3.8 Flash High",collector_owner="Codex",independent_source_reviewer="Gemini",
    agreement=str(OUT/"gemini/NEXT_BOUNDED_OFFICIAL_SCOPE_AGREEMENT.json"),
    agreement_sha256=sha(OUT/"gemini/NEXT_BOUNDED_OFFICIAL_SCOPE_AGREEMENT.json"),
    collector_source=str(ROOT/"first_board_corporate_action_official_probe.py"),
    collector_source_sha256=sha(ROOT/"first_board_corporate_action_official_probe.py"),
    max_direct_http_requests=6,max_index_queries_per_code=1,max_pdf_downloads_per_code=1,timeout_seconds=10,
    request_policy="One bounded120-calendar-day company index page100,then latest matching implementation PDF published no later than source date;no paging/retries/alternate companies",
    on_403_or_429="Stop all remaining requests,record unknown;no bypass",
    pdf_selection_is_not_field_verification=True,
    requested_field_scope=["actual_company_code","publication_date","cash_per_share_gross","record_date","ex_date",
        "bonus_and_reserve_share_ratio","differential_distribution","rights_subscription_if_any","reference_formula"],
    price_reference_and_asof_cumulative_factor_separate=True,star_case_not_in_main_board_training_pool=True,
    model_fits_allowed=0,target_label_changes_allowed=0,production_writes_allowed=0,
    original_facts=str(facts),original_facts_sha256=sha(facts),goal_60_achieved=False,
    limitation="3 source/formula probes do not certify full action history,all labels or prediction improvement")
write(destination/"JOINT_FROZEN_PROTOCOL.json",protocol)
pointer_path=ROOT/"research/gemini_codex_joint_training_20261020/CURRENT_JOINT_EXPERIMENT.json"
write(destination/"PREVIOUS_JOINT_EXPERIMENT_POINTER.json",read(pointer_path))
pointer=dict(updated_at=instant.isoformat(),status="THREE_OFFICIAL_CASES_FROZEN_REQUESTS_NOT_YET_STARTED",
    experiment_root=str(destination),protocol=str(destination/"JOINT_FROZEN_PROTOCOL.json"),actual_owner="Codex",
    actual_peer="Antigravity Gemini 3.8 Flash High",training_arms={},active_model_training=False,process_handle=None,
    model_fits=0,goal_60_achieved=False,prior_source_qualification=str(OUT/"completion.json"),
    original_deadline_utc=protocol["deadline_utc"],no_failed_model_retry=True)
pointer_path.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding="utf-8")
print(str(destination))
