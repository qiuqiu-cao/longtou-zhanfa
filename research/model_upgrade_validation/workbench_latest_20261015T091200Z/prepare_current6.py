"""Complete the existing diagnostic preparation chain without model inference."""
from copy import deepcopy
from pathlib import Path
import json

import short_term_thinking_dataset_builder as b
import supplement_role_review as supplement
import supplement_identity_discovery as identity
import forum_reasoning_review as forum
import short_term_thinking_workbench as thinking
import historical_symbolic_context as symbolic
import relay_review_workbench as workbench

ROOT=Path(__file__).resolve().parent
PREVIOUS=symbolic.RUN
LATEST=PREVIOUS/"latest15_prepared_v2"
RAW=PREVIOUS/"latest15_alternative_page_once"
DAY="2026-10-15"
TARGET="2026-10-16"


def main():
    b.verify(b.read(ROOT/"execution_start.json")["protected_before"])
    if (ROOT/"prepared_receipt.json").exists():raise FileExistsError("immutable current6 preparation exists")
    pack_path=LATEST/"features_BEFORE_MODEL_DISPLAY.json";pack=b.read(pack_path)
    prepool=b.read(PREVIOUS/"latest15_forum_preread/complete_current6_pool.json")
    pool=deepcopy(prepool)
    for row in pool["rows"]:
        source=next(x for x in pack["relay"] if x["code"]==row["code"])
        row.update(same_height_peers=sorted(x["code"] for x in pool["rows"] if x["boards"]==row["boards"] and x["code"]!=row["code"]),
            daily_price_volume=deepcopy(source["features"]),actual_one_word_daily=source.get("actual_one_word_daily"),
            board_scope="ALTERNATIVE_REPORTED_STREAK_EXACT_DAILY_CHECKED_NOT_ORIGINAL_FORMAL_SOURCE",
            feature_semantics={"turnover_vs_prev":"ratio to previous entry in archived relay pool; only compare adjacent official days after independently checking dates",
                "relative_volume":"native-volume ratio to previous official session; distinct from turnover ratio",
                "amount_turnover":"quote field positions inferred and crosschecked; not exchange certification",
                "AI_page_business_text":"excluded from company facts, numeric industry fields and new semantic roles"})
    pool["source_bindings"]|=b.read(LATEST/"source_context_receipt.json")["source_bindings"]
    pool_path=ROOT/"inputs/complete_current6_score_free_pool.json";b.write_new(pool_path,pool)
    # This is only an extraction for the already supported alternative adapter.
    parsed=b.read(RAW/"parsed_complete_pools.json")
    alt_rows=[]
    for row in parsed["rows"]:
        alt_rows.append(dict(code=row["code"],name=row["name"],kind=row["kind"],consecutive_boards=row["boards"],
            mainboard_name_non_st_scope=row["scope"],reason_secondary="",unverified_AI_page_summary=row["reason_summary"]))
    alt=dict(signal_date=DAY,generated_at=b.now(),source_raw_path=str((RAW/"page.raw.html").resolve()),source_sha256=b.sha(RAW/"page.raw.html"),
        rows=alt_rows,relay_complete=[x for x in alt_rows if x["kind"]=="limit_up" and x["mainboard_name_non_st_scope"] and x["consecutive_boards"]>=2],
        raw_pool_counts=parsed["raw_pool_counts"],AI_business_text_not_verified_and_not_used_as_industry=True,
        source_status="ALTERNATIVE_COMPLETE_RETURNED_PAGE_NOT_V293_OR_EXCHANGE_CERTIFICATION")
    alt_path=ROOT/"inputs/alternate_full_pool15.json";b.write_new(alt_path,alt)
    oldspec=b.read(b.ROOT/"research/model_upgrade_validation/supplement_discovery_20261015T063312Z/existing_pool_and_daily_evidence.json")
    oldbank_path=b.ROOT/"research/model_upgrade_validation/supplement_recognition_20261014T174129Z/qualified_relation_bank_v2.json"
    bank=deepcopy(b.read(oldbank_path));codes={x["code"] for x in pool["rows"]}
    bank.update(signal_date=DAY,target_date=TARGET,formed_at=b.now(),new_requests=0,new_fits=0,
        source_scope="Existing verified old bodies and separate comments; exact15 closing prices and complete source scopes; no current author body invented")
    bank["edges"]=[x for x in bank["edges"] if x["child_code"] in codes]
    bank["source_bindings"]|=pool["source_bindings"]
    for path in [oldbank_path,RAW/"normalized_daily.json",RAW/"daily_manifest.json",alt_path,Path(__file__)]:bank["source_bindings"][str(path.resolve())]=b.sha(path)
    spec=deepcopy(oldspec);spec.update(signal_date=DAY,target_date=TARGET,formed_at=b.now(),scope="CURRENT15_DIAGNOSTIC_PREPARATION_BEFORE_STOCK_MODEL_DISPLAY",
        historical_asof_training_authorized=False,new_requests=0)
    spec["pool_snapshots"].append(dict(date=DAY,path=str(alt_path.resolve()),sha256=b.sha(alt_path),kind="ALTERNATE_FULL_POOL"))
    proof=spec["source_bindings"]|bank["source_bindings"]
    snapshots=identity.read_snapshots(spec,proof,identity.o.stamp(b.now()))
    daily=b.read(RAW/"normalized_daily.json")
    present={(x["code"],x["date"]) for x in bank["temporal_facts"]}
    relevant=codes|{x["code"] for x in bank["anchors"] if x.get("code")}
    added=[]
    for code in sorted(relevant & set(daily)):
        for day,bar in daily[code]["days"].items():
            if (code,day) in present or day not in snapshots:continue
            reported=snapshots[day]["rows"].get(code)
            board_source=None
            if reported:
                sp=reported["source_path"]
                tail="/lbc" if "EASTMONEY" in snapshots[day]["scope"] or "/data/pool/" in reported["source_pointer"] else "/consecutive_boards"
                board_source=dict(path=sp,sha256=proof[sp],json_pointer=reported["source_pointer"]+tail)
            source=dict(path=str((RAW/"normalized_daily.json").resolve()),sha256=b.sha(RAW/"normalized_daily.json"),json_pointer="/"+code+"/days/"+day)
            added.append(dict(code=code,date=day,ohlc_cents={k:bar[k] for k in ["open","high","low","close"]},
                archived_boards=reported["boards"] if reported else None,board_source=board_source,
                board_status="SOURCE_REPORTED_CLOSING_HEIGHT" if reported else "ABSENT_FROM_FULL_POOL_NOT_AN_IDENTITY_LABEL",
                available_at=daily[code]["available_at"],source_path=source["path"],raw_path=daily[code]["raw_path"],
                source_json_pointer=source["json_pointer"],ohlc_source=source,auction_verified=False,reseal_verified=False))
    bank["temporal_facts"]+=added
    bank_path=ROOT/"inputs/relation_bank_current6.json";b.write_new(bank_path,bank)
    spec.update(relation_bank_path=str(bank_path.resolve()),formed_at=b.now(),source_bindings=proof|{str(bank_path.resolve()):b.sha(bank_path)})
    spec["daily_archive"]=dict(normalized_path=str((RAW/"normalized_daily.json").resolve()),manifest_paths=[str((RAW/"daily_manifest.json").resolve())],codes=sorted(daily))
    spec_path=ROOT/"inputs/identity_evidence_current15.json";b.write_new(spec_path,spec)
    bank_forum=PREVIOUS/"latest15_forum_preread/checked_background_bank.json"
    calendar=b.ROOT/"data/v298/exchange_calendar_extension_20261003_v2/calendar.json"
    exposure="MODEL_NOT_REVIEWED_SELF_ATTESTED"
    supplement.context(pool_path,bank_forum,bank_path,calendar,ROOT/"supplement",exposure)
    forum.attach(ROOT/"supplement/facts_with_supplement.json",ROOT/"forum",ROOT/"supplement/author_supplement_PENDING.json")
    identity.attach(ROOT/"forum/facts_with_forum_reasoning.json",spec_path,ROOT/"identity",ROOT/"forum/author_draft_PENDING.json")
    history=[b.ROOT/"results/daily_review_20261013T074936Z/leader_role_history.json"]
    # Reuse actual prior review inputs rather than inventing file names.
    existing=b.read(b.DEFAULT_RUN/"delivery/workbench_verified/context.json")
    history=existing["history_datasets"]
    mainline=existing["mainline_history"]
    narrative=existing["narrative_reviews"]
    thinking.attach(ROOT/"identity/facts_with_identity_discovery.json",pack_path,LATEST/"engine_context_WITH_CURRENT15.joblib",ROOT/"thinking",ROOT/"identity/author_identity_PENDING.json",
        identity_evidence=spec_path,history_datasets=history,mainline_history=mainline,narrative_reviews=narrative)
    symbolic.attach_review(ROOT/"thinking/facts_with_causal_thinking.json",pack_path,PREVIOUS/"qualified_final/symbolic_context.joblib",ROOT/"symbolic",ROOT/"thinking/author_draft_PENDING.json")
    facts=ROOT/"symbolic/facts_with_symbolic_context.json"
    workbench.render(facts,ROOT/"review_SCORES_MASKED")
    result=dict(formed_at=b.now(),status="FULL6_EXISTING_DIAGNOSTIC_PREPARATION_COMPLETE_AUTHOR_PENDING",signal_date=DAY,target_date=TARGET,
        full_pool_count=6,source_scope="QUALIFIED_ALTERNATIVE_RESEARCH_NOT_FORMAL",facts_path=str(facts.resolve()),facts_sha256=b.sha(facts),
        draft_path=str((ROOT/"symbolic/author_draft_PENDING.json").resolve()),all_legacy_guards_retained=True,old_six_role_review_source=str((PREVIOUS/"delivery_final/current6_actual_role_review_BEFORE_MODEL_DISPLAY.json").resolve()),
        history_datasets=history,mainline_history=mainline,narrative_reviews=narrative,models_loaded=0,new_fits=0,new_formal_predictions=0)
    b.write_new(ROOT/"prepared_receipt.json",result);print(result)


if __name__=="__main__":main()
