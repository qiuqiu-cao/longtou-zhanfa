"""Evaluate three authorized, manually authored Codex reviews without an API."""
from pathlib import Path
from itertools import combinations
import json
import re
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import expectation_345_research as trial
import expectation_345_evaluation as evaluation

ROOT=trial.ROOT


def validate_review(packet,review):
    if review["protocol"]!="345-v1" or review["repeat_id"]!=packet["repeat_id"]:
        raise ValueError("review identity mismatch")
    if review["canonical_facts_sha256"]!=packet["canonical_facts_sha256"]:
        raise ValueError("canonical facts mismatch")
    if not isinstance(review["model_scores_seen"],bool) or not isinstance(review["outcomes_seen"],bool):
        raise ValueError("exposure must be explicit")
    if not review.get("exposure_notes") or "\ufffd" in json.dumps(review,ensure_ascii=False):
        raise ValueError("missing exposure or invalid text")
    expected={(day["signal_date"],x["code"]) for day in packet["days"] for x in day["candidates"]}
    actual=[(x["date"],x["code"]) for x in review["rows"]]
    if len(actual)!=len(expected) or set(actual)!=expected or len(actual)!=len(set(actual)):
        raise ValueError("incomplete or duplicate complete-cohort review")
    by_date={x["signal_date"]:x for x in packet["days"]}
    for day in packet["days"]:
        rows=[x for x in review["rows"] if x["date"]==day["signal_date"]]
        if sorted(x["rank"] for x in rows)!=list(range(1,len(rows)+1)):
            raise ValueError("daily ranking is not a unique complete permutation")
    for row in review["rows"]:
        day=by_date[row["date"]]
        codes={x["code"] for x in day["candidates"]+day["all_reported_touch_pool"]+day["full_previous_members"]}
        codes|={x["parent_code"] for x in day["all_reported_active_anchor_profiles"]}
        heights={str(int(x["boards"])) for x in day["candidates"]}
        heights|={str(int(x["target_height"])) for x in day["full_previous_height_groups"]}
        if row["call"] not in ("PROMOTE","FAIL","UNKNOWN") or not row.get("reason") or not row.get("cancel"):
            raise ValueError("call and individual reasons/cancel required")
        for key in ("support_fact_ids","counter_fact_ids"):
            if not row[key]:raise ValueError("both support and counter-evidence required")
            for ref in row[key]:
                parts=ref.split(":")
                if len(parts)<2 or parts[1]!=row["date"]:
                    raise ValueError("fact from a different signal date")
                if parts[0]=="market" and len(parts)==2:continue
                if len(parts)!=3:raise ValueError("invalid fact id")
                if parts[0] in ("stock","parent","gap") and parts[2] in codes:continue
                if parts[0]=="cohort" and parts[2] in heights:continue
                raise ValueError("fact id not in provided as-of packet: "+ref)
    return pd.DataFrame(review["rows"]).assign(repeat_id=review["repeat_id"])


def rank_agreement(frame):
    stats=[]
    for day,part in frame.groupby("date"):
        pivot=part.pivot(index="code",columns="repeat_id",values="rank")
        for left,right in combinations(sorted(pivot.columns),2):
            x,y=pivot[left],pivot[right]
            # Unique complete rankings: Pearson correlation of ranks equals Spearman.
            rho=float(x.corr(y)) if len(pivot)>1 else None
            topa,setb=set(x.nsmallest(min(3,len(x))).index),set(y.nsmallest(min(3,len(y))).index)
            stats.append(dict(date=day,left=int(left),right=int(right),candidates=len(pivot),
                spearman=rho,top3_intersection=len(topa&setb),top3_union=len(topa|setb),
                top3_jaccard=len(topa&setb)/len(topa|setb)))
    return pd.DataFrame(stats)


def build():
    output=ROOT/"LLM_evaluation";output.mkdir(exist_ok=False)
    sources=[ROOT/"LLM_repeated_review_protocol.md",Path(__file__)]
    frames=[];exposure=[]
    for number in (1,2,3):
        packet_path=ROOT/f"LLM_repeat_{number}_closed_packet.json"
        review_path=ROOT/f"LLM_repeat_{number}_actual_review.json"
        packet,review=b.read(packet_path),b.read(review_path)
        frames.append(validate_review(packet,review))
        exposure.append({k:v for k,v in review.items() if k!="rows"})
        sources.extend([packet_path,review_path])
    reviews=pd.concat(frames,ignore_index=True)
    pairwise,consensus=evaluation.agreement(frames)
    _,current_consensus=evaluation.agreement([x[x.date.eq("2026-10-15")] for x in frames])
    current_pairs,_=evaluation.agreement([x[x.date.eq("2026-10-15")] for x in frames])
    ranks=rank_agreement(reviews)
    predictions=ROOT/"evaluation_scope_alignment/complete18885_predictions_ORIGINAL_FROZEN_COHORT.csv"
    pred=pd.read_csv(predictions,dtype={"code":str},low_memory=False)
    work_path=trial.OLD/"workbench/完整81只_全部证据与原审阅.json"
    work=b.read(work_path)["rows"]
    metadata=pd.DataFrame([dict(date=x["date"],code=x["code"],name=x["name"],boards=x["boards"],
        target_date=next(d["target_date"] for d in packet["days"] if d["signal_date"]==x["date"]),
        source_supplement_target=x["outcome_for_display"],
        original_author_call=x["archived_author_record"]["original_direction"] or "UNKNOWN",
        original_author_status=x["archived_author_record"]["archive_status"],
        archived_author_source_path=x["archived_author_record"]["source_path"]) for x in work])
    assert len(metadata)==81 and metadata.source_supplement_target.notna().sum()==74
    keys=["date","code"]
    scored=pred.merge(metadata[keys+["source_supplement_target"]],on=keys,validate="many_to_one")
    assert len(scored)==81*5
    base=scored[scored.arm.eq("BASE")][keys+["daily_one_word","target","label_valid"]]
    data=metadata.merge(base,on=keys,validate="one_to_one")
    parts=[scored]
    for number,review in enumerate(frames,1):
        f=data.merge(review,on=keys,validate="one_to_one")
        f["arm"]=f"LLM_REPEAT_{number}";f["ranking_score"]=-f["rank"]
        parts.append(f)
    mean_rank=reviews.groupby(keys,as_index=False)["rank"].mean().rename(columns={"rank":"mean_rank"})
    combined=data.merge(consensus,on=keys,validate="one_to_one").merge(mean_rank,on=keys,validate="one_to_one")
    for label,key in [("LLM_MAJORITY_FIXED","majority_call"),("LLM_UNANIMOUS_FIXED","unanimous_call")]:
        f=combined.copy();f["call"]=f[key];f["arm"]=label;f["ranking_score"]=-f.mean_rank;parts.append(f)
    original=data.copy();original["call"]=original.original_author_call;original["arm"]="ARCHIVED_ORIGINAL_AUTHOR_WITH_MISSING";original["ranking_score"]=np.nan
    parts.append(original)
    complete=pd.concat(parts,ignore_index=True)
    summaries=[];selected=[]
    for truth_name in ("FROZEN_ORIGINAL_LABELS","SUPPLEMENTAL_DISPLAY_LABELS"):
        labelset=complete.copy()
        if truth_name=="SUPPLEMENTAL_DISPLAY_LABELS":
            labelset["target"]=labelset.source_supplement_target
            labelset["label_valid"]=labelset.target.notna()
        for period,condition in [("ALL81_RETROSPECTIVE",pd.Series(True,index=labelset.index)),
            ("HISTORICAL75_EXPOSED",labelset.date.lt("2026-10-15")),
            ("CURRENT15_SIX_TARGET16_PENDING",labelset.date.eq("2026-10-15"))]:
            pool=labelset[condition]
            for arm,f in pool.groupby("arm"):
                for scope,sub in [("FULL",f),("ORIGINAL_NONWORD",f[f.daily_one_word.eq(0)])]:
                    summaries.append(dict(truth=truth_name,period=period,arm=arm,scope=scope,**evaluation.classification(sub)))
                if arm!="ARCHIVED_ORIGINAL_AUTHOR_WITH_MISSING":
                    for k in (1,2,3):
                        chosen=evaluation.selection(f,k).copy();chosen["truth"]=truth_name;chosen["period"]=period;chosen["k"]=k;selected.append(chosen)
    reviews.to_csv(output/"all243_actual_manual_reviews.csv",index=False,encoding="utf-8-sig")
    complete.to_csv(output/"all891_comparable_calls_WITH_ORIGINAL_AND_SUPPLEMENT_LABELS.csv",index=False,encoding="utf-8-sig")
    metrics=pd.DataFrame(summaries);metrics.to_csv(output/"standalone_effect_coverage_abstention_false_alarm.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pairwise).to_csv(output/"pairwise_call_agreement_ALL81.csv",index=False,encoding="utf-8-sig")
    ranks.to_csv(output/"all_daily_rank_agreement.csv",index=False,encoding="utf-8-sig")
    consensus.to_json(output/"all81_votes_and_fixed_consensus.json",orient="records",force_ascii=False,indent=2)
    pd.concat(selected,ignore_index=True).to_csv(output/"all_selected_rows_WITH_UNKNOWN.csv",index=False,encoding="utf-8-sig")
    disagreements=consensus[~consensus.unanimous].merge(metadata,on=keys,validate="one_to_one")
    disagreements.to_json(output/"all_disagreements_WITH_REASONS_REFERENCED.json",orient="records",force_ascii=False,indent=2)
    current=[]
    for row in combined[combined.date.eq("2026-10-15")].sort_values(["mean_rank","code"]).to_dict("records"):
        row.update(three_authored_reviews=reviews[reviews.date.eq(row["date"]) & reviews.code.eq(row["code"])].to_dict("records"),
            target=None,source_supplement_target=None,label_valid=False,
            status="SHADOW_FUTURE_OUTCOME_PENDING_NOT_NEW_FORMAL_REGISTRATION",model_probabilities=None)
        current.append(row)
    shadow=dict(formed_at=b.now(),target_date="2026-10-16",rows=current,
        actual_outcomes_available=0,original_formal_predictions_modified=False,
        review_context_exposure=exposure,stability_not_accuracy=True,calibrated_probability=False,
        use="Prospective outcome comparison for these six archived candidate identities only; no schedule, order or production model replacement")
    b.write_new(output/"current15_six_target16_forward_shadow_FROZEN.json",shadow)
    result=dict(formed_at=b.now(),actual_authored_reviews=243,independent_review_contexts=3,external_API_inference_calls=0,
        full_cohort_rows=81,historical_retrospective_rows=75,supplemental_known_outcomes=74,
        historical_blind=False,historical_independence_of_label_knowledge=False,
        exposure=exposure,controlled_fact_packet_content_identical=True,candidate_order_changed=True,
        indirect_later_date_outcome_exposure=True,unvalidated_historical_LLMeffect=True,
        pairwise_ALL81=pairwise,pairwise_CURRENT15_SIX=current_pairs,
        unanimous_ALL81=int(consensus.unanimous.sum()),unanimous_CURRENT15_SIX=int(current_consensus.unanimous.sum()),
        current15_future_target16_known_outcomes=0,fixed_majority_used=True,best_repeat_selected_by_outcome=False,
        automatic_call_generation=False,LLM_weight_finetuning=False,production_adopted=False,
        limitation="Three Codex contexts inherit the project's historical results/old scores; later signal packets also imply earlier outcomes. This quantifies exposed retrospective disagreement/effect only. Current15 six target16 unknown, separate shadow record. Consensus is a fixed comparison, never certified as superior from this batch.")
    sources.extend([predictions,work_path,Path(evaluation.__file__),*output.glob("*")])
    result["source_bindings"]={str(q.resolve()):b.sha(q) for q in sources}
    b.verify(result["source_bindings"]);b.write_new(output/"LLM_repeat_verification_and_results.json",result)
    print(json.dumps({k:result[k] for k in ["actual_authored_reviews","unanimous_ALL81","unanimous_CURRENT15_SIX","pairwise_ALL81","pairwise_CURRENT15_SIX"]}))


if __name__=="__main__":build()
