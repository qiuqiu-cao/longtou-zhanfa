"""Additive score-masked causal context and explicit research-score view."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import stock_judgement_review as o


def context(features_path, bundle_path, *, identity_evidence=None, history_datasets=(), mainline_history=(), narrative_reviews=None, theme_report=None):
    """Future dates call exactly the same causal_features function as history.

    For existing dates a saved current context can be used. For new dates the
    closing feature pack plus optional raw-bound identity evidence is appended;
    unavailable fields stay missing and never become certified role identity.
    """
    pack=b.read(features_path);day=pack["signal_date"]
    bundle=joblib.load(bundle_path)
    sources={str(Path(bundle_path).resolve()):b.sha(bundle_path),str(Path(features_path).resolve()):b.sha(features_path),str(Path(__file__).resolve()):b.sha(__file__),str(Path(b.__file__).resolve()):b.sha(b.__file__)}
    if "prepared" not in bundle:
        if b.sha(features_path) not in bundle["sources"].values():
            raise ValueError("cached context requires its exact source feature pack")
        cached=bundle["features"]
        rows=cached[cached.date.eq(day)].copy()
        if rows.empty:
            raise ValueError("new date requires historical dataset.joblib and current closing evidence")
        sources.update(bundle["sources"])
    else:
        current=[]
        closing_rows={r["code"]:r for r in pack.get("original_pool",{}).get("rows",[])}
        for r in pack["relay"]:
            current.append(dict(r["features"],date=day,code=r["code"],name=r["name"],boards=r["boards"],industry=closing_rows.get(r["code"],{}).get("industry")))
        candidates=bundle["prepared"]["relay"]
        candidates=pd.concat([candidates[candidates.date.lt(day)],pd.DataFrame(current)],ignore_index=True)
        market=pd.concat([bundle["prepared"]["market"].query("date < @day"),pd.DataFrame([dict(pack["market"]["features"],date=day)])],ignore_index=True)
        attempts=bundle["attempts"][bundle["attempts"].signal_date.le(day)].copy()
        failed_dates=set(bundle["attempts"].attrs.get("failed_scope_complete_dates",bundle["attempts"].signal_date.unique()))
        bars=bundle["bars"][bundle["bars"].date.le(day)].copy()
        if identity_evidence:
            import supplement_identity_discovery as identity
            spec=b.read(identity_evidence)
            if spec["signal_date"]!=day:
                raise ValueError("current parent-chain evidence date mismatch")
            b.verify(spec["source_bindings"])
            snapshots=identity.read_snapshots(spec,spec["source_bindings"],o.stamp(b.now()))
            daily=identity.read_daily(spec,spec["source_bindings"],o.stamp(b.now()))
            alternate_dates={x["date"] for x in spec["pool_snapshots"] if x["kind"]=="ALTERNATE_FULL_POOL"}
            def industry_for(date, value):
                families=identity.families(value or "") if date in alternate_dates else []
                return families[0] if families else value
            extra=[]
            for date, snapshot in snapshots.items():
                for code,r in snapshot["rows"].items():
                    price=daily.get(code,{}).get("days",{}).get(date,{}).get("close")
                    extra.append(dict(signal_date=date,code=code,name=r["name"],board_height=r["boards"],pool_assignment="CLOSING_LIMIT_UP",close=price/100 if price else np.nan,industry=industry_for(date,r.get("industry",""))))
            if extra:
                attempts=pd.concat([attempts,pd.DataFrame(extra)],ignore_index=True).drop_duplicates(["signal_date","code","pool_assignment"],keep="last")
            daily_rows=[]
            for code,item in daily.items():
                for date,r in item["days"].items():
                    if date<=day:
                        daily_rows.append(dict(date=date,code=code,**{k:r[k]/100 for k in ("open","high","low","close")},volume=r["volume_native"]))
            if daily_rows:
                dr=pd.DataFrame(daily_rows)
                # A native-volume series is kept whole from its first covered
                # date, so the last old share-count cannot be its denominator.
                first=dr.groupby("code").date.min().to_dict()
                bars=bars[[r.code not in first or r.date<first[r.code] for r in bars.itertuples()]]
                bars=pd.concat([bars,dr],ignore_index=True).drop_duplicates(["date","code"],keep="last")
            sources.update(spec["source_bindings"]);sources[str(Path(identity_evidence).resolve())]=b.sha(identity_evidence)
            for code,r in snapshots.get(day,{}).get("rows",{}).items():
                candidates.loc[candidates.date.eq(day)&candidates.code.eq(code),"industry"]=industry_for(day,r.get("industry"))
        if theme_report:
            theme=b.read(theme_report)
            if theme["signal_date"]!=day:
                raise ValueError("current complete theme/failed-pool date mismatch")
            previous=bundle["sessions"][bundle["sessions"].index(day)-1]
            prior=attempts[attempts.signal_date.eq(previous)&attempts.pool_assignment.eq("CLOSING_LIMIT_UP")].set_index("code").board_height.to_dict()
            extra=[]
            for r in theme["all_candidates"]:
                if r.get("pool_kind")=="failed_board":
                    extra.append(dict(signal_date=day,code=r["code"],name=r["name"],pool_assignment="FAILED_BOARD_OBSERVATION",board_height=int(prior.get(r["code"],0))+1))
            if extra:
                attempts=pd.concat([attempts,pd.DataFrame(extra)],ignore_index=True).drop_duplicates(["signal_date","code","pool_assignment"],keep="last")
            # The upstream theme builder validates the whole disjoint source
            # pool; an empty but complete failed pool is therefore explicit.
            failed_dates.add(day)
            sources[str(Path(theme_report).resolve())]=b.sha(theme_report)
        attempts.attrs["failed_scope_complete_dates"]=sorted(failed_dates)
        features,_=b.causal_features(candidates,market,attempts,bars,bundle["sessions"])
        rows=features[features.date.eq(day)].copy()
        sources.update(bundle["sources"])
    expected={(r["code"],r["name"],r["boards"]) for r in pack["relay"]}
    actual={(r.code,r.name,r.boards) for r in rows.itertuples()}
    if expected!=actual or len(rows)!=len(pack["relay"]):
        raise ValueError("complete current identities must match causal context")
    # All supplied histories are explicitly forwarded/bound. Narrative comments
    # do not acquire the parent post's clock and do not become numeric facts.
    for path in tuple(history_datasets)+tuple(mainline_history)+((narrative_reviews,) if narrative_reviews else ()):
        sources[str(Path(path).resolve())]=b.sha(path)
    b.verify(sources)
    return dict(version=b.VERSION,formed_at=b.now(),signal_date=day,target_date=pack["target_date"],rows=b.clean(rows.to_dict("records")),
        source_bindings=sources,history_datasets=[str(Path(x).resolve()) for x in history_datasets],
        mainline_history=[str(Path(x).resolve()) for x in mainline_history],narrative_reviews=str(Path(narrative_reviews).resolve()) if narrative_reviews else None,
        parent_chain_evidence=str(Path(identity_evidence).resolve()) if identity_evidence else None,
        independent_role_gold=False,model_scores_masked=True,new_fits=0)


def attach(facts_path, features_path, bundle_path, output_dir, draft_path=None, **kwargs):
    out=Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable thinking context directory required")
    facts=o.read(facts_path)
    projection=context(features_path,bundle_path,**kwargs)
    if projection["signal_date"]!=facts["signal_date"]:
        raise ValueError("fact date mismatch")
    src=o.unique_rows(projection["rows"])
    old=o.unique_rows(facts["rows"])
    if set(src)!=set(old) or any((src[c]["name"],src[c]["boards"])!=(old[c]["name"],old[c]["boards"]) for c in src):
        raise ValueError("facts must retain complete same current pool")
    b.write_new(out/"context.json",projection)
    result=deepcopy(facts)
    # Legacy forum checks the entire row projection; join context only in views.
    result.update(causal_thinking_required=True,causal_thinking_context_path=str((out/"context.json").resolve()))
    result["source_bindings"].update(projection["source_bindings"])
    result["source_bindings"].update(o.bind(out/"context.json"))
    b.write_new(out/"facts_with_causal_thinking.json",result)
    if draft_path:
        draft=deepcopy(o.read(draft_path))
        draft["facts_path"]=str((out/"facts_with_causal_thinking.json").resolve())
        draft["facts_sha256"]=b.sha(out/"facts_with_causal_thinking.json")
        draft["author_review_completed"]=False
        draft["source_bindings"].update(result["source_bindings"])
        draft["source_bindings"].update(o.bind(out/"facts_with_causal_thinking.json"))
        b.write_new(out/"author_draft_PENDING.json",draft)
    return result


def checked(facts):
    if not facts.get("causal_thinking_required"):
        return None
    path=facts["causal_thinking_context_path"]
    if facts["source_bindings"].get(str(Path(path).resolve()))!=b.sha(path):
        raise ValueError("causal context binding changed")
    projection=b.read(path);b.verify(projection["source_bindings"])
    if projection["signal_date"]!=facts["signal_date"] or projection["version"]!=b.VERSION:
        raise ValueError("causal context date or version changed")
    originals=o.unique_rows(projection["rows"])
    actual=o.unique_rows(facts["rows"])
    if set(actual)!=set(originals) or any((actual[c]["name"],actual[c]["boards"])!=(originals[c]["name"],originals[c]["boards"]) for c in actual):
        raise ValueError("causal role or feature tampering / incomplete pool")
    return projection


def score_context(facts_path, reviewed_record_path, model_path, output_path):
    """A separate view only after an actual archived independent review."""
    facts=o.read(facts_path);projection=checked(facts)
    if projection is None:
        raise ValueError("masked causal context must be attached first")
    record=o.read(reviewed_record_path)
    if not (record.get("author_review_completed") is True or record.get("status")=="INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE"):
        raise ValueError("actual independent review must be archived before scores")
    if not record.get("facts_path") or record.get("facts_sha256")!=b.sha(record["facts_path"]):
        raise ValueError("independent review must bind its original exact facts")
    if record.get("status")=="INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE":
        import json
        registry=b.ROOT/"data/independent_stock_judgement/registry.jsonl"
        entries=[json.loads(line) for line in registry.read_text(encoding="utf-8-sig").splitlines()]
        if not any(e.get("path")==str(Path(reviewed_record_path).resolve()) and e.get("sha256")==b.sha(reviewed_record_path) for e in entries):
            raise ValueError("independent record is not in the original registry")
    else:
        import relay_review_workbench
        relay_review_workbench.validate_completed(record["facts_path"],record)
    if set(o.unique_rows(record["rows"]))!=set(o.unique_rows(facts["rows"])) or record.get("signal_date")!=facts["signal_date"]:
        raise ValueError("independent review does not cover complete current pool")
    b.verify(record.get("source_bindings",{}))
    saved=joblib.load(model_path)
    if saved.get("role_version")!=b.VERSION or saved["cutoff"]>facts["signal_date"]:
        raise ValueError("research model version or training cutoff mismatch")
    # The original feature pack is bound in context; only its signal fields are
    # used. No outcomes are accepted into this score view.
    packs=[]
    for path in projection["source_bindings"]:
        if Path(path).suffix==".json" and ("features" in Path(path).name or "feature_pack" in Path(path).name):
            j=b.read(path)
            if j.get("signal_date")==facts["signal_date"] and isinstance(j.get("relay"),list):
                packs.append(j)
    if not packs:
        raise ValueError("source-bound original current feature pack required")
    pack=packs[0]
    frame=pd.DataFrame([dict(r["features"],code=r["code"]) for r in pack["relay"]])
    dynamic=pd.DataFrame(projection["rows"])
    overlap=(set(frame)&set(dynamic))-{ "code" }
    frame=frame.drop(columns=list(overlap)).merge(dynamic,on="code",validate="one_to_one")
    scores=saved["model"].predict_proba(frame[saved["features"]])[:,list(saved["model"].classes_).index(1)]
    rows=[dict(code=r.code,name=r.name,boards=r.boards,primary_role=r.primary_role,channel_b_score=float(scores[i]) if r.daily_one_word==0 else None,
        direction=("PROMOTE" if scores[i]>.5 else "FAIL") if r.daily_one_word==0 else "EMOTION_MONITOR_ONLY" if r.daily_one_word==1 else "UNKNOWN_IDENTITY",
        channel="A_EMOTION_ONLY" if r.daily_one_word==1 else "B_NONWORD" if r.daily_one_word==0 else "UNRESOLVED_IDENTITY") for i,r in enumerate(frame.itertuples())]
    result=dict(version=b.VERSION,formed_at=b.now(),status="RESEARCH_SCORE_AFTER_INDEPENDENT_REVIEW_NOT_PRODUCTION_REPLACEMENT",signal_date=facts["signal_date"],
        rows=rows,model_source=str(Path(model_path).resolve()),model_sha256=b.sha(model_path),review_source=str(Path(reviewed_record_path).resolve()),
        review_sha256=b.sha(reviewed_record_path),uncalibrated=True,new_fits=0,new_formal_prediction=0)
    result["original_review_model_exposure"]=record.get("model_exposure","MODEL_OUTPUT_ALREADY_SEEN")
    result["historical_replay_not_new_forward_forecast"]=True
    b.write_new(output_path,result)
    return result
