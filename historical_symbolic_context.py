"""Date-causal stock-name diffusion as an explicitly typed theme proxy.

This does not manufacture an industry, active trading theme, or independent
leader identity. It reconstructs the names valid on each historical date and
measures complete archived first-board/relay co-occurrence using fixed V217
structural thresholds. Original models and role rows are never overwritten.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
from symbolic_name_theme_v217 import _name_tokens, THRESHOLDS

VERSION="historical-symbolic-diffusion-1"
RUN=b.ROOT/"research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z"
BASE=b.DEFAULT_RUN
PHONETIC_FAMILIES=("阳洋羊扬","华花","龙隆","新鑫","国果","生升盛","德得")
POLICY={"min_members":THRESHOLDS["min_member_count"],"min_share":THRESHOLDS["min_member_share"],
    "min_relay":THRESHOLDS["min_relay_member_count"],"min_heights":THRESHOLDS["min_distinct_board_levels"],
    "first_board_required":1,"industry_diversity":"UNAVAILABLE_SEPARATE_FROM_FULL_V217_GATE",
    "phonetic_families":list(PHONETIC_FAMILIES),"primary_order":"two-character token first, relay support, first support, member share, token ascending",
    "role_semantics":"SYMBOLIC_NAME_DIFFUSION_PROXY_CANDIDATE_NOT_BUSINESS_OR_ACTIVE_THEME_IDENTITY",
    "missing_scope":"NaN structural counts; a known zero requires an explicit complete archived closing scope",
    "historical_eligibility":"Remove independently interval-known ST/non-mainboard/non10pct/not-listed pool supporters; retain unknown scope members explicitly and all original candidates",
    "dates":"exact official session; previous missing scope is not skipped",
    "old_roles":"retain original role set, append typed proxy roles separately; no reclassification of old frozen facts"}
FIELDS=["symbolic_scope_complete","symbolic_name_interval_observed","symbolic_cluster_count","symbolic_primary_members",
    "symbolic_primary_first_support","symbolic_primary_relay_support_ex_self","symbolic_primary_same_height_rivals",
    "symbolic_primary_height_gap","symbolic_primary_ladder_integrity","symbolic_primary_share",
    "symbolic_primary_first_change_1","symbolic_primary_share_change_1","symbolic_qualified_streak",
    "symbolic_role_theme_leader_proxy","symbolic_role_theme_seed_proxy","symbolic_isolated_within_name_proxy",
    "symbolic_industry_evidence_available"]


class DatedNames:
    def __init__(self, master):
        self.intervals={}
        for code,part in master.groupby("code",sort=False):
            items=part.sort_values("valid_from").to_dict("records")
            for previous,current in zip(items,items[1:]):
                if not previous.get("valid_to") or previous["valid_to"]>=current["valid_from"]:
                    raise ValueError("ambiguous or overlapping historical name intervals")
            self.intervals[code]=([r["valid_from"] for r in items],items)

    def resolve(self, code, day, archive_name):
        entry=self.intervals.get(code)
        if entry:
            starts,items=entry;i=bisect_right(starts,day)-1
            if i>=0:
                row=items[i]
                if not row.get("valid_to") or day<=row["valid_to"]:
                    verified=row.get("data_status") in {"verified","verified_import","verified_terminal_export"}
                    eligibility=None
                    if verified and all(k in row for k in ("st_status","board","price_limit_regime","listing_status")):
                        eligibility=(row["st_status"]=="NORMAL" and row["board"]=="MAINBOARD" and row["price_limit_regime"]=="MAINBOARD_10" and row["listing_status"]=="LISTED")
                    return dict(name=row["name"],status="DATED_INTERVAL_RECONSTRUCTED" if verified else "DATED_INTERVAL_INFERRED",historical_eligible=eligibility,
                        interval_observed=int(verified),interval_id=row.get("interval_id"),valid_from=row["valid_from"],valid_to=row.get("valid_to"),
                        source_ref=row.get("source_ref"),actual_source_timestamp=row.get("source_timestamp"),strict_original_receipt_verified=False)
        return dict(name=archive_name,status="DATE_ASSOCIATED_ARCHIVE_NAME_NOT_INTERVAL_QUALIFIED",interval_observed=0,historical_eligible=None,
            interval_id=None,valid_from=None,valid_to=None,source_ref=None,actual_source_timestamp=None,strict_original_receipt_verified=False)


def tokens(name):
    result={f"H{length}:{token}" for token,length in _name_tokens(name)}
    chars={token for token,length in _name_tokens(name) if length==1}
    result.update("P:"+group for group in PHONETIC_FAMILIES if chars&set(group))
    return result


def build_causal(candidates, attempts, names, sessions, complete_dates, *, industry_evidence=None):
    """Shared batch/live engine. Read only date/code/name/boards and past state."""
    if list(sessions)!=sorted(set(sessions)) or candidates.duplicated(["date","code"]).any():
        raise ValueError("unique official dates and current candidates required")
    c=candidates[["date","code","name","boards"]].copy()
    c["code"]=c.code.astype(str).str.zfill(6)
    a=attempts[["signal_date","code","name","board_height","pool_assignment"]].copy()
    if a.duplicated(["signal_date","code","pool_assignment"]).any():
        raise ValueError("duplicate source pool identity")
    a["code"]=a.code.astype(str).str.zfill(6)
    index={d:i for i,d in enumerate(sessions)}
    if any(d not in index for d in c.date):raise ValueError("candidate is outside official calendar")
    lookup=DatedNames(names)
    closing={day:{r["code"]:dict(code=r["code"],name=r["name"],boards=int(r["board_height"])) for r in f.to_dict("records")}
        for day,f in a[a.pool_assignment.eq("CLOSING_LIMIT_UP")].groupby("signal_date")}
    byday={day:f.to_dict("records") for day,f in c.groupby("date")}
    complete=set(complete_dates)
    industry={}
    if industry_evidence is not None:
        if industry_evidence.duplicated(["date","code"]).any():raise ValueError("duplicate industry evidence")
        industry={(r["date"],r["code"]):r for r in industry_evidence.to_dict("records")}
    rows=[];clusters=[];name_changes=[];previous={};streaks={};previous_complete_day=None
    start=min(c.date.min(),a.signal_date.min())
    all_days=[d for d in sessions if start<=d<=c.date.max()]
    for day in all_days:
        pool=deepcopy(closing.get(day,{}));current=byday.get(day,[])
        # Original >=2 candidates are retained even when the daily reconstruction
        # omits a code. This is a declared mixed-source archive union, not a
        # claim of exchange-wide or historically survivor-free completeness.
        for row in current:pool[row["code"]]=row
        day_complete=day in complete
        resolved={code:lookup.resolve(code,day,r["name"]) for code,r in pool.items()}
        groups=defaultdict(list)
        for code,row in pool.items():
            named=resolved[code]
            if named["name"]!=row["name"]:
                name_changes.append(dict(date=day,code=code,archive_name=row["name"],dated_name=named["name"],name_status=named["status"],interval_id=named["interval_id"],historical_scope_excluded=named["historical_eligible"] is False))
            if named["historical_eligible"] is False:continue
            for token in tokens(named["name"]):groups[token].append(code)
        current_clusters={}
        scope_members=sum(x["historical_eligible"] is not False for x in resolved.values())
        if day_complete:
            for token,codes in groups.items():
                members=[pool[code] for code in codes];heights=sorted({r["boards"] for r in members})
                first=sum(r["boards"]==1 for r in members);relays=sum(r["boards"]>=2 for r in members)
                qualified=(len(codes)>=max(POLICY["min_members"],int(np.ceil(scope_members*POLICY["min_share"])))
                    and relays>=POLICY["min_relay"] and len(heights)>=POLICY["min_heights"] and first>=POLICY["first_board_required"])
                adjacent=previous_complete_day is not None and index[day]-index[previous_complete_day]==1
                prior=previous.get(token,{"first_support":0,"share":0.})
                if qualified:streaks[token]=(streaks.get(token,(0,None))[0]+1 if streaks.get(token,(0,None))[1]==index[day]-1 else 1,index[day])
                else:streaks.pop(token,None)
                state=dict(date=day,token=token,members=len(codes),member_codes=sorted(codes),first_support=first,relay_support=relays,
                    max_height=max(heights),heights=heights,share=len(codes)/scope_members,qualified=bool(qualified),
                    first_change_1=first-prior["first_support"] if adjacent else None,
                    share_change_1=len(codes)/scope_members-prior["share"] if adjacent else None,
                    qualified_streak=streaks[token][0] if token in streaks else 0,
                    dated_name_interval_members=sum(resolved[code]["interval_observed"] for code in codes),
                    distinct_industries_verified=False,active_trading_theme_verified=False,
                    scope="COMPLETE_WITHIN_DECLARED_ARCHIVE_UNION_CURRENT_UNIVERSE_BIAS_REMAINS")
                current_clusters[token]=state
                if qualified:clusters.append(state)
            previous=current_clusters
            previous_complete_day=day
        else:
            # Unknown dates break continuity. Never use the last observed pool as
            # the previous official session when an intervening source is absent.
            previous={};streaks={};previous_complete_day=None
        for row in current:
            code,n=row["code"],int(row["boards"]);name=resolved[code]
            qualifies=[s for s in current_clusters.values() if s["qualified"] and code in s["member_codes"]]
            qualifies.sort(key=lambda s:(-int(s["token"].startswith("H2:")),-s["relay_support"],-s["first_support"],-s["share"],s["token"]))
            top=qualifies[0] if qualifies else None
            known_absence=day_complete and not top
            def val(key,zero=0):return top[key] if top else zero if known_absence else None
            leader=bool(top and n>=3 and n==top["max_height"])
            seed=bool(top and n==2 and n==top["max_height"])
            peers=sum(pool[k]["boards"]==n for k in top["member_codes"])-1 if top else 0 if known_absence else None
            integrity=len(top["heights"])/(top["max_height"]-min(top["heights"])+1) if top else 0 if known_absence else None
            candidate_industry=industry.get((day,code))
            result=dict(date=day,code=code,name=row["name"],boards=n,historical_name=name["name"],historical_name_status=name["status"],
                name_interval_id=name["interval_id"],name_source_time=name["actual_source_timestamp"],primary_symbolic_token=top["token"] if top else None,
                qualified_symbolic_tokens=[s["token"] for s in qualifies],symbolic_roles=(["THEME_LEADER"] if leader else ["THEME_SEED"] if seed else []),
                role_basis="SYMBOLIC_NAME_DIFFUSION_PROXY_NOT_VERIFIED_ACTIVE_THEME",industry_name=candidate_industry.get("industry") if candidate_industry else None,
                industry_is_active_theme=False,feature_cutoff=day,independent_role_truth=False,
                historical_candidate_eligibility=name["historical_eligible"],
                symbolic_scope_complete=int(day_complete),symbolic_name_interval_observed=name["interval_observed"],
                symbolic_cluster_count=len(qualifies) if day_complete else None,symbolic_primary_members=val("members"),
                symbolic_primary_first_support=val("first_support"),symbolic_primary_relay_support_ex_self=top["relay_support"]-1 if top else 0 if known_absence else None,
                symbolic_primary_same_height_rivals=peers,symbolic_primary_height_gap=top["max_height"]-n if top else 0 if known_absence else None,
                symbolic_primary_ladder_integrity=integrity,symbolic_primary_share=val("share"),
                symbolic_primary_first_change_1=val("first_change_1"),symbolic_primary_share_change_1=val("share_change_1"),
                symbolic_qualified_streak=val("qualified_streak"),symbolic_role_theme_leader_proxy=int(leader) if day_complete else None,
                symbolic_role_theme_seed_proxy=int(seed) if day_complete else None,symbolic_isolated_within_name_proxy=int(not qualifies) if day_complete else None,
                symbolic_industry_evidence_available=int(candidate_industry is not None))
            rows.append(result)
    return pd.DataFrame(rows),clusters,name_changes


def build(output_dir=RUN):
    out=Path(output_dir);out.mkdir(exist_ok=True,parents=True)
    if (out/"symbolic_context.joblib").exists():raise FileExistsError("immutable symbolic dataset already exists")
    engine_path=BASE/"scope_correction/engine_context.joblib";engine=joblib.load(engine_path)
    b.verify(engine["sources"])
    master_path=b.ROOT/"data/v4/security_master_intervals.csv"
    from security_master_v4 import validate_master
    master=validate_master(pd.read_csv(master_path,dtype={"code":str})).fillna("")
    industry_path=b.ROOT/"data/v4/candidate_industry_evidence.csv"
    industry=pd.read_csv(industry_path,dtype={"code":str})
    complete=engine["attempts"].attrs["failed_scope_complete_dates"]
    f,clusters,changes=build_causal(engine["prepared"]["relay"],engine["attempts"],master,engine["sessions"],complete,industry_evidence=industry)
    historical=f[f.date.le("2026-09-30")];assert len(historical)==9245
    expected=set(zip(engine["prepared"]["relay"].date,engine["prepared"]["relay"].code));assert expected==set(zip(f.date,f.code))
    original=engine["features"][["date","code","primary_role","roles"]]
    f=f.merge(original,on=["date","code"],validate="one_to_one")
    f["extended_rule_roles"]=[list(dict.fromkeys(list(old)+list(new))) for old,new in zip(f.roles,f.symbolic_roles)]
    f["extended_primary_role"]=[(new[0] if old=="RELAY_CONTENDER" and new else old) for old,new in zip(f.primary_role,f.symbolic_roles)]
    sources={str(engine_path.resolve()):b.sha(engine_path),**engine["sources"]}
    for path in [Path(__file__),master_path,industry_path,b.ROOT/"security_master_v4.py",b.ROOT/"symbolic_name_theme_v217.py",
        b.ROOT/"data/v4/security_evidence/szse_short_name_changes_20260831.csv",b.ROOT/"data/v4/security_evidence/szse_mainboard_list_20260831.csv",
        b.ROOT/"data/v4/security_evidence/sse_announcements/sse_name_events.csv"]:
        sources[str(path.resolve())]=b.sha(path)
    f.to_csv(out/"complete9309_symbolic_context_and_extended_roles.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(clusters).to_csv(out/"all_qualified_symbolic_clusters.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(out/"pool_names_resolved_at_historical_date.csv",index=False,encoding="utf-8-sig")
    joblib.dump(dict(version=VERSION,rows=f,clusters=clusters,names=master,sessions=engine["sessions"],policy=POLICY,sources=sources,
        original_full=engine["prepared"]["relay"],engine_path=str(engine_path.resolve())),out/"symbolic_context.joblib",compress=3)
    import model_upgrade_history as history
    windows={}
    for fold in history.FOLDS:
        train=history.training_rows(engine["prepared"]["relay"],fold["start"],"relay")
        p=f.merge(train[["date","code"]],on=["date","code"],validate="one_to_one")
        windows[fold["id"]]=dict(rows=len(p),full_archive_name_diffusion_computable=float(p.symbolic_scope_complete.mean()),
            dated_name_interval_coverage=float(p.symbolic_name_interval_observed.mean()),
            symbolic_cluster_present=int(p.symbolic_cluster_count.gt(0).sum()),symbolic_theme_leader_candidates=int(p.symbolic_role_theme_leader_proxy.sum()),
            symbolic_theme_seed_candidates=int(p.symbolic_role_theme_seed_proxy.sum()),
            feature_nonmissing={key:float(p[key].notna().mean()) for key in FIELDS})
    report=dict(version=VERSION,formed_at=b.now(),original_historical_rows=9245,current_rows=len(f)-9245,total_rows=len(f),
        original_role_rows_unchanged=True,original_models_refitted=0,strict_original_name_receipt_coverage=0,
        windows=windows,historical_symbolic_theme_leader_candidates=int(f[f.date.le("2026-09-30")].symbolic_role_theme_leader_proxy.sum()),
        historical_symbolic_theme_seed_candidates=int(f[f.date.le("2026-09-30")].symbolic_role_theme_seed_proxy.sum()),
        actual_historical_industry_candidate_rows=int(f[f.date.le("2026-09-30")].symbolic_industry_evidence_available.sum()),
        source_name_corrections=len(changes),known_ineligible_pool_supporters_removed=sum(x.get("historical_scope_excluded",False) for x in changes),qualified_cluster_days=len(clusters),
        current_name_is_not_past_name=True,industry_or_active_theme_not_fabricated=True,independent_identity_accuracy=None,
        policy=POLICY,source_bindings=sources,dataset_path=str((out/"symbolic_context.joblib").resolve()),dataset_sha256=b.sha(out/"symbolic_context.joblib"))
    b.write_new(out/"source_qualification_and_coverage.json",report)
    return report


def context_for_review(features_path, artifact_path, causal_context):
    """New dates and historical batches use build_causal, with no score input."""
    pack=b.read(features_path);day=pack["signal_date"];artifact=joblib.load(artifact_path)
    if artifact["version"]!=VERSION or artifact["policy"]!=POLICY:raise ValueError("symbolic policy changed")
    b.verify(artifact["sources"])
    stocks={r["code"]:r for r in pack["relay"]}
    causal={r["code"]:r for r in causal_context["rows"]}
    if set(stocks)!=set(causal) or causal_context["signal_date"]!=day:raise ValueError("complete current causal pool required")
    data=artifact["rows"]
    cached=data[data.date.eq(day)].copy()
    if len(cached) and b.sha(features_path) in artifact["sources"].values():
        rows=cached
        route="EXACT_SAVED_FEATURE_PACK_SHARED_ENGINE_CACHE"
    else:
        # The normal workbench has already validated the current original pool.
        # Standalone calls still reject an incomplete or changed relay roster.
        pool=pack.get("original_pool",{}).get("rows")
        if not isinstance(pool,list) or not pool:raise ValueError("complete original closing pool required")
        if len({r["code"] for r in pool})!=len(pool):raise ValueError("duplicate current closing pool")
        if {r["code"] for r in pool if int(r["boards"])>=2}!=set(stocks):raise ValueError("incomplete relay roster in current original pool")
        engine=joblib.load(artifact["engine_path"])
        c=engine["prepared"]["relay"]
        current=pd.DataFrame([dict(date=day,code=r["code"],name=r["name"],boards=r["boards"]) for r in stocks.values()])
        c=pd.concat([c[c.date.lt(day)],current],ignore_index=True)
        a=engine["attempts"]
        fresh=pd.DataFrame([dict(signal_date=day,code=r["code"],name=r["name"],board_height=int(r["boards"]),pool_assignment="CLOSING_LIMIT_UP") for r in pool])
        a=pd.concat([a[a.signal_date.lt(day)],fresh],ignore_index=True)
        complete=set(engine["attempts"].attrs["failed_scope_complete_dates"]);complete.add(day)
        industry=artifact["rows"].loc[artifact["rows"].symbolic_industry_evidence_available.eq(1),["date","code","industry_name"]].rename(columns={"industry_name":"industry"})
        computed,_,_=build_causal(c,a,artifact["names"],artifact["sessions"],complete,industry_evidence=industry)
        rows=computed[computed.date.eq(day)].copy();route="SAME_ENGINE_NEW_DATE_COMPUTED_FROM_COMPLETE_CURRENT_POOL"
    if len(rows)!=len(stocks) or set(rows.code)!=set(stocks):raise ValueError("symbolic context cannot omit a candidate")
    for r in rows.itertuples():
        if (r.name,int(r.boards))!=(stocks[r.code]["name"],int(stocks[r.code]["boards"])):
            raise ValueError("current name or height differs from source")
    output=[]
    for row in rows.to_dict("records"):
        base=causal[row["code"]];roles=list(base["roles"])
        extra=list(row["symbolic_roles"])
        row.update(original_causal_primary_role=base["primary_role"],extended_rule_roles=list(dict.fromkeys(roles+extra)),
            extended_primary_role=extra[0] if base["primary_role"]=="RELAY_CONTENDER" and extra else base["primary_role"])
        output.append(row)
    sources=artifact["sources"]|{str(Path(features_path).resolve()):b.sha(features_path),str(Path(artifact_path).resolve()):b.sha(artifact_path),str(Path(__file__).resolve()):b.sha(__file__)}
    return dict(version=VERSION,formed_at=b.now(),signal_date=day,target_date=pack["target_date"],rows=b.clean(output),
        features_path=str(Path(features_path).resolve()),artifact_path=str(Path(artifact_path).resolve()),route=route,source_bindings=sources,policy=POLICY,
        model_scores_masked=True,independent_identity_verified=False,new_fits=0)


def attach_review(facts_path, features_path, artifact_path, output_dir, draft_path=None):
    import stock_judgement_review as o
    import short_term_thinking_workbench as thinking
    out=Path(output_dir)
    if out.exists():raise FileExistsError("immutable symbolic review directory required")
    facts=o.read(facts_path);causal=thinking.checked(facts)
    if causal is None:raise ValueError("masked causal thinking context required before symbolic extension")
    context=context_for_review(features_path,artifact_path,causal)
    if context["signal_date"]!=facts["signal_date"] or {r["code"] for r in context["rows"]}!={r["code"] for r in facts["rows"]}:
        raise ValueError("symbolic extension must retain complete exact facts")
    b.write_new(out/"symbolic_context.json",context)
    result=deepcopy(facts)
    result.update(symbolic_context_required=True,symbolic_context_path=str((out/"symbolic_context.json").resolve()))
    result["source_bindings"].update(context["source_bindings"])
    result["source_bindings"].update(o.bind(out/"symbolic_context.json"))
    b.write_new(out/"facts_with_symbolic_context.json",result)
    if draft_path:
        draft=deepcopy(o.read(draft_path));draft.update(facts_path=str((out/"facts_with_symbolic_context.json").resolve()),facts_sha256=b.sha(out/"facts_with_symbolic_context.json"),author_review_completed=False)
        draft["source_bindings"].update(result["source_bindings"]);draft["source_bindings"].update(o.bind(out/"facts_with_symbolic_context.json"))
        b.write_new(out/"author_draft_PENDING.json",draft)
    return result


def checked_review(facts):
    if not facts.get("symbolic_context_required"):return None
    path=facts["symbolic_context_path"]
    if facts["source_bindings"].get(str(Path(path).resolve()))!=b.sha(path):raise ValueError("symbolic context source binding changed")
    context=b.read(path);b.verify(context["source_bindings"])
    if context["version"]!=VERSION or context["policy"]!=POLICY or context["signal_date"]!=facts["signal_date"]:
        raise ValueError("symbolic context date or definition changed")
    a={r["code"]:r for r in facts["rows"]};z={r["code"]:r for r in context["rows"]}
    if len(a)!=len(facts["rows"]) or len(z)!=len(context["rows"]) or set(a)!=set(z):raise ValueError("complete symbolic candidate roster required")
    if any((a[c]["name"],a[c]["boards"])!=(z[c]["name"],z[c]["boards"]) for c in a):raise ValueError("symbolic identities changed")
    import short_term_thinking_workbench as thinking
    causal=thinking.checked(facts)
    if causal is None:raise ValueError("original masked causal context missing")
    rebuilt=context_for_review(context["features_path"],context["artifact_path"],causal)
    if rebuilt["rows"]!=context["rows"]:
        raise ValueError("symbolic counts or roles differ from source reconstruction")
    return context


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",default=str(RUN));a=p.parse_args(argv);r=build(a.output_dir)
    print(b.clean({k:r[k] for k in ("total_rows","source_name_corrections","historical_symbolic_theme_leader_candidates","historical_symbolic_theme_seed_candidates","windows")}))


if __name__=="__main__":main()
