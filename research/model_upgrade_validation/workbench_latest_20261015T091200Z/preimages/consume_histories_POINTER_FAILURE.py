"""Consume prior role/mainline archives and typed old narratives, diagnostic only.

This never invokes an alternative source through a trusted production adapter.
Raw-backed external groups stay reconstructed/INFERRED and are analysed only
with the existing DIAGNOSTIC role track.
"""
from pathlib import Path
from copy import deepcopy
import json

import short_term_thinking_dataset_builder as b
import leader_role_engine as e
import stock_catalyst_context as catalyst
import stock_narrative_review as narrative
import stock_short_term_context as short
import short_term_thinking_workbench as thinking
import historical_symbolic_context as symbolic

ROOT=Path(__file__).resolve().parent
PREVIOUS=symbolic.RUN
LATEST=PREVIOUS/"latest15_prepared_v2"
SCOPE="PROVIDER_REPORTED_MAINBOARD_NAME_NON_ST_NOT_EXCHANGE_CERTIFIED"
MARKET_SCOPE="PROVIDER_MAINBOARD_NAME_NON_ST_CLOSING_AND_FAILED_POOLS_NOT_EXCHANGE_CERTIFIED"
CALENDAR=b.ROOT/"data/v298/exchange_calendar_extension_20261003_v2/calendar.json"


def external_dataset(day,pack,closing,failed,source_paths,out):
    codes=sorted({x["code"] for x in closing+failed})
    if len(codes)!=len(closing)+len(failed):raise ValueError("disjoint complete closing and failed pools required")
    prices=e.closing_price_facts(closing,{x["code"]:x for x in pack["relay"]})
    names={x["code"]:x["name"] for x in closing+failed}
    heights={x["code"]:int(x.get("boards",0)) for x in closing}
    for x in failed:
        heights[x["code"]]=0
        prices[x["code"]]={k:None for k in e.RELAY_FIELDS}
        prices[x["code"]]["boards"]=0
    # Board labels are provider observations crosschecked separately, with no
    # exchange certification. Retain the original role gate as unknown here.
    pool=dict(signal_date=day,codes=codes,boards=heights,amounts={c:prices[c].get("signal_amount") for c in codes},
        turnovers={c:prices[c].get("turnover") for c in codes},complete_within_scope=True,
        board_numbers_independently_verified=False,scope="COMPLETE_ALTERNATIVE_REPORTED_CLOSING_AND_FAILED_ROSTER")
    when=b.now();sources={str(Path(p).resolve()):b.sha(p) for p in source_paths}
    rich=dict(formed_at=when,signal_date=day,groups=dict(pool=pool,prices=prices),source_bindings=sources,
        source_policy="Alternative quoted price facts; reconstructed and diagnostic, no trusted production adapter or role gold")
    rich_path=out/(day+"_rich_raw_projection.json");b.write_new(rich_path,rich)
    rows=[]
    for code in codes:
        groups={}
        for kind,value,pointer in [("pool",pool,"/groups/pool"),("price",prices[code],"/groups/prices/"+code)]:
            groups[kind]=dict(value=value,status="INFERRED",observed_at=day+"T15:00:00+08:00",available_at=when,
                reconstructed=True,adapter="ALTERNATIVE_RAW_DIAGNOSTIC_GROUP",source_path=str(rich_path.resolve()),source_sha256=b.sha(rich_path),pointer=pointer)
        rows.append(dict(candidate_id=day+":"+code,signal_date=day,code=code,name=names[code],boards=heights[code],scope_id=SCOPE,
            decision_at=when,available_at=when,actual_one_word_daily=next((x.get("actual_one_word_daily") for x in pack["relay"] if x["code"]==code),None),groups=groups))
    result=dict(version=e.VERSION,formed_at=when,rows=rows,calendar_path=str(CALENDAR.resolve()),calendar_sha256=b.sha(CALENDAR),
        source_bindings=sources|{str(rich_path.resolve()):b.sha(rich_path)},adapter=dict(kind="UNTRUSTED_ALTERNATIVE_GROUPS_DIAGNOSTIC_ONLY"),
        baseline_promotion={},official_full_market=False,historical_diagnostic_only=True)
    path=out/(day+"_role_dataset_DIAGNOSTIC.json");b.write_new(path,result)
    # Original gates remain active; this is deliberately not STRICT eligibility.
    loaded=e.load_dataset(path);e.analyse(loaded,"DIAGNOSTIC")
    return path


def main():
    out=ROOT/"histories_consumed"
    if out.exists():raise FileExistsError("immutable history consumption output")
    out.mkdir()
    feature15=LATEST/"features_BEFORE_MODEL_DISPLAY.json";p15=b.read(feature15)
    alt15=b.read(LATEST/"alternative_source_projection/normalized.json")
    p14path=b.ROOT/"results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json";p14=b.read(p14path)
    alt14path=b.ROOT/"results/score_update_20261014T1153Z/alternative_source_projection/normalized.json";alt14=b.read(alt14path)
    context14=external_dataset("2026-10-14",p14,p14["original_pool"]["rows"],alt14["pools"]["2026-10-14/failed_board"]["rows"],[p14path,alt14path],out)
    dataset=external_dataset("2026-10-15",p15,p15["original_pool"]["rows"],alt15["pools"]["2026-10-15/failed_board"]["rows"],[feature15,LATEST/"alternative_source_projection/normalized.json",LATEST/"source_context_receipt.json"],out)
    histories=[b.ROOT/"results/daily_review_20261013T074936Z/prior12_role_history_REBUILT_NOW.json",
        b.ROOT/"results/daily_review_20261013T074936Z/independent_source_repaired/role_dataset.original.json",context14]
    mainlines=[b.ROOT/"results/daily_review_20261012T105005Z/mainline/report.json",b.ROOT/"results/daily_review_20261013T074936Z/mainline/report.json"]
    for day,pack,source in [("2026-10-14",p14,p14path),("2026-10-15",p15,feature15)]:
        now=b.now();report=dict(signal_date=day,status="UNKNOWN",mainline=None,policy_version="mainline-facts-analysis-1",generated_at=now,knowledge_cutoff=now,
            scope=MARKET_SCOPE,source_hashes={str(source.resolve()):b.sha(source)},reason="Exact complete price pool available; actual active-theme membership and same-day author bodies unavailable in the bounded evidence. Old background not relabelled current mainline.",
            pool_counts=dict(closing=len(pack["original_pool"]["rows"]),relay=len(pack["relay"])),inferred_market_phase=pack["market"]["persistence"],phase_does_not_certify_theme=True)
        path=out/(day+"_mainline_UNKNOWN_WITH_REASON.json");b.write_new(path,report);mainlines.append(path)
    bank_path=PREVIOUS/"latest15_forum_preread/checked_background_bank.json"
    bank,sources,claims,proof=catalyst.validate_bank(bank_path,p15["relay"],e.stamp(b.now()))
    cutoff=b.now();crows=catalyst.build_rows(p15["relay"],claims,sources,cutoff,cutoff)
    context=dict(version=catalyst.VERSION,formed_at=cutoff,knowledge_cutoff=cutoff,signal_date=p15["signal_date"],target_date=p15["target_date"],
        original_feature_cutoff=cutoff,rows=crows,source_bindings=proof|{str(feature15.resolve()):b.sha(feature15)},
        status="DIAGNOSTIC_CATALYST_REVIEW_FROM_VALIDATED_BANK_NOT_FEATURE_ARCHIVE_ADMISSION")
    catalyst_path=out/"catalyst_context_DIAGNOSTIC.json";b.write_new(catalyst_path,context)
    annotations=[]
    for index,claim in enumerate(bank["claims"]):
        source=sources[claim["source_id"]];body=Path(source["text_path"]).read_text("utf-8")
        value=dict(claim_index=index,claim_sha256=narrative.digest(claim),code=claim["code"],reviewed_at=b.now())
        if claim.get("source_section")=="COMMENT":
            value.update(content_type="UNKNOWN",reason="Comment keeps its own clock and alias; it offers a name/parent conjecture, while the strict full-name typed span is not established in this comment. The separately validated forum argument remains available.")
        else:
            start=min(claim["offset"],claim["entity_offset"]);end=max(claim["offset"]+len(claim["excerpt"]),claim["entity_offset"]+len(claim["entity_anchor"]))
            span=body[start:end]
            if claim["source_id"]=="guba_name":topic,label,anchor,level="NAME_YANG_CHAIN","阳洋羊名称链","九阳","NAME_NARRATIVE"
            elif claim["code"]=="002343":topic,label,anchor,level="MEDIA_BACKGROUND","传媒旧背景","传媒","PARENT"
            else:topic,label,anchor,level="AGRICULTURAL_DEFENSE_BACKGROUND","农业避险旧假设","农业避险","PARENT"
            value.update(content_type="EXPLICIT_TRADE_NARRATIVE",reason="Author explicitly connects the named stock to this narrative; publication is October12 background, not current October15 market consensus.",
                source_span=dict(offset=start,text=span),topic_id=topic,topic_label=label,topic_anchor=anchor,topic_level=level,
                relationship_review="Reviewed author-reported topic link only; price identity, business truth and capital movement are independent questions.")
        annotations.append(value)
    reviews=dict(version=narrative.VERSION,formed_at=b.now(),reviewer="Codex actual claim-by-claim semantic review",bank_path=str(bank_path.resolve()),bank_sha256=b.sha(bank_path),reviews=annotations)
    reviews_path=out/"narrative_reviews_COMPLETED.json";b.write_new(reviews_path,reviews)
    typed=narrative.build(catalyst_path,bank_path,reviews_path,out/"typed_narrative_context.json")
    short_context=short.build(dataset,catalyst_path,out/"short_term_context_WITH_HISTORIES.json","DIAGNOSTIC",histories,mainlines)
    if not all(x["prior_rows_used"]>0 for x in short_context["history_datasets"]):raise ValueError("bound but unused role history")
    # Bind these actually consumed files into the live causal layer, preserving
    # the original forum/identity rows and every old guard.
    thinking.attach(ROOT/"symbolic/facts_with_symbolic_context.json",feature15,LATEST/"engine_context_WITH_CURRENT15.joblib",out/"thinking",ROOT/"symbolic/author_draft_PENDING.json",
        identity_evidence=ROOT/"inputs/identity_evidence_current15.json",history_datasets=histories,mainline_history=mainlines,narrative_reviews=reviews_path)
    symbolic.attach_review(out/"thinking/facts_with_causal_thinking.json",feature15,PREVIOUS/"qualified_final/symbolic_context.joblib",out/"symbolic",out/"thinking/author_draft_PENDING.json")
    result=dict(formed_at=b.now(),status="ACTUALLY_CONSUMED_HISTORIES_AND_TYPED_NARRATIVE_NOT_FORMAL_FORECAST",history_datasets=short_context["history_datasets"],
        mainline_records_consumed=4,mainline_changes_certified=False,narrative_claims_classified=len(annotations),same_day_active_narratives=sum(len(x["signal_day_trade_narratives"]) for x in typed["rows"]),
        current_source_role_track="DIAGNOSTIC",source_board_independent_verification_not_upgraded=True,
        facts_path=str((out/"symbolic/facts_with_symbolic_context.json").resolve()),draft_path=str((out/"symbolic/author_draft_PENDING.json").resolve()),
        old_rows_unchanged=True,new_fits=0,models_loaded=0,new_formal_predictions=0,source_bindings=short_context["source_bindings"]|typed["source_bindings"])
    b.write_new(ROOT/"histories_consumed_receipt.json",result);print({k:v for k,v in result.items() if k!="source_bindings"})


if __name__=="__main__":main()
