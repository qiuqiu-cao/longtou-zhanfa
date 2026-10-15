"""Discover parent-cycle hypotheses from complete saved pools, without model scores."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import re

import stock_judgement_review as o
import supplement_relation_review as old
from public_outcome_sources_v293 import parse_pool

VERSION = "supplement-identity-discovery-1"
CALLS = {"CYCLE_SUPPLEMENT_CANDIDATE", "OLD_FOLLOWER_RESTART",
         "THEME_RELAY_COMPETING", "RELATED_UNRESOLVED", "UNKNOWN", "NOT_SUPPORTED"}
FAMILIES = {
    "MEDIA": ("传媒", "出版", "影视", "微短剧", "图书", "文化传媒"),
    "CONSUMER": ("消费", "食品", "饮料", "乳品", "乳业", "家电", "白色家电"),
    "CHEMICAL": ("化工", "乙腈", "化学", "染料"),
    "DEFENSE": ("军工", "航空", "航发"),
    "AGRICULTURE": ("农业", "水产", "饲料", "鱼粉"),
    "ELECTRONICS": ("电子", "印制电路", "PCB"),
    "BATTERY": ("电池", "固态", "储能"),
    "TRANSPORT": ("交通", "物流", "航运"),
}


def families(text):
    return sorted(k for k, words in FAMILIES.items() if any(w in text for w in words))


def aliases(name):
    values = {name}
    for suffix in ("股份", "科技", "乳业", "健康", "传媒"):
        if name.endswith(suffix) and len(name)-len(suffix) >= 2:
            values.add(name[:-len(suffix)])
    return values


def read_snapshots(spec, proof, cutoff):
    snapshots = {}
    for item in spec["pool_snapshots"]:
        date, path = item["date"], str(Path(item["path"]).resolve())
        if date in snapshots or date > spec["signal_date"]:
            raise ValueError("duplicate or target-day pool snapshot")
        if proof.get(path) != item["sha256"]:
            raise ValueError("pool snapshot is not source bound")
        payload = o.read(path)
        if item["kind"] == "EASTMONEY_CLOSING_POOL":
            parsed = parse_pool(payload, date, "limit_up")
            rows = [{"code":r["code"], "name":r["name"], "boards":r["boards"],
                     "industry":r.get("industry") or "", "source_path":path,
                     "source_pointer":"/data/pool/"+str(i)}
                    for i,r in enumerate(parsed["rows"])]
            # Pointers must index raw rows before security-scope filtering.
            indices = {str(r["c"]):i for i,r in enumerate(payload["data"]["pool"])}
            for r in rows:
                r["source_pointer"] = "/data/pool/"+str(indices[r["code"]])
            manifest_path = str(Path(item["manifest"]).resolve())
            if manifest_path not in proof:
                raise ValueError("original request manifest required")
            manifest = o.read(manifest_path)
            matches = [r for r in manifest["requests"] if r.get("raw_path")==path]
            if len(matches)!=1 or matches[0].get("raw_sha256")!=item["sha256"]:
                raise ValueError("snapshot differs from original HTTP receipt")
            request = matches[0]
            if request.get("date") != date or request.get("kind") != "limit_up":
                raise ValueError("pool request date or kind changed")
            if not o.stamp(request["requested_at"]) <= o.stamp(request["available_at"]) <= cutoff:
                raise ValueError("snapshot request/receive clocks unavailable")
            if o.stamp(request["requested_at"]) < o.stamp(date+"T15:00:00+08:00"):
                raise ValueError("intraday snapshot cannot establish closing continuity")
            scope = "COMPLETE_PROVIDER_MAINBOARD_NAME_NON_ST_CLOSING_POOL"
        elif item["kind"] == "ALTERNATE_FULL_POOL":
            if payload["signal_date"] != date or not isinstance(payload.get("relay_complete"),list):
                raise ValueError("complete current alternative ladder required")
            raw_path = str(Path(payload["source_raw_path"]).resolve())
            if proof.get(raw_path)!=payload["source_sha256"]:
                raise ValueError("alternative raw page binding required")
            if o.stamp(payload["generated_at"]) > cutoff:
                raise ValueError("alternative extraction unavailable")
            rows = [{"code":r["code"], "name":r["name"], "boards":r["consecutive_boards"],
                     "industry":r.get("reason_secondary") or "", "source_path":path,
                     "source_pointer":"/rows/"+str(i)}
                    for i,r in enumerate(payload["rows"])
                    if r["kind"]=="limit_up" and r.get("mainboard_name_non_st_scope") is True]
            expected = {(r["code"],r["name"],r["boards"]) for r in rows if r["boards"]>=2}
            declared = {(r["code"],r["name"],r["consecutive_boards"]) for r in payload["relay_complete"]}
            if expected != declared or len(declared)!=len(payload["relay_complete"]):
                raise ValueError("alternative complete ladder differs from closing rows")
            scope = "ALTERNATE_PROVIDER_LADDER_NOT_ORIGINAL_FORMAL_SOURCE"
        else:
            raise ValueError("unsupported pool adapter")
        if any(type(r["boards"]) is not int or r["boards"]<1 for r in rows):
            raise ValueError("explicit consecutive boards required")
        if len({r["code"] for r in rows})!=len(rows):
            raise ValueError("duplicate closing stock")
        snapshots[date] = {"rows":{r["code"]:r for r in rows}, "scope":scope,
                           "full_pool_path":path, "complete_within_declared_scope":True}
    return snapshots


def analyze(pool, snapshots, sessions, claims, seed_edges=(), prior_events=()):
    """Pure hypothesis formation. Full anchor roster is derived from market history."""
    signal = pool["signal_date"]
    stocks = o.unique_rows(pool["rows"])
    if signal not in snapshots:
        raise ValueError("current complete closing pool required")
    current = snapshots[signal]["rows"]
    if any(c not in current or (current[c]["name"],current[c]["boards"])!=(r["name"],r["boards"])
           for c,r in stocks.items()):
        raise ValueError("original full candidate identities/boards differ")
    dates = [d for d in sessions if d in snapshots and d<=signal]
    identities = {}
    for day in dates:
        for code,row in snapshots[day]["rows"].items():
            identities[code] = row["name"]
    token_owners = defaultdict(set)
    for code,name in identities.items():
        for token in aliases(name):
            token_owners[token].add(code)
    def mentioned(code, text):
        return any(token in text and len(token_owners[token])==1 for token in aliases(identities[code]))
    anchors = {}
    for day in dates:
        if day==signal:
            continue
        for code,row in snapshots[day]["rows"].items():
            if row["boards"]>=2:
                anchors.setdefault(code,[]).append({"date":day,**row})
    rows = []
    for code,stock in sorted(stocks.items()):
        idx = sessions.index(signal); n=stock["boards"]
        streak_dates = sessions[max(0,idx-n+1):idx+1]
        observed = [snapshots.get(d,{}).get("rows",{}).get(code) for d in streak_dates]
        complete = len(streak_dates)==n and all(r and r["boards"]==i+1 for i,r in enumerate(observed))
        expected_start=streak_dates[0]
        older = [d for d in dates if d<expected_start and code in snapshots[d]["rows"]]
        prior_seen = [e["date"] for e in prior_events if e["code"]==code and
                      e["event"]=="PRIOR_FIRST_OBSERVED_BOARD" and e["value"] is True and e["date"]<expected_start]
        old_hit = min(older+prior_seen) if older or prior_seen else None
        child_families=set(families(current[code]["industry"]))
        hypotheses, considered = [], []
        for anchor,history in sorted(anchors.items()):
            if anchor==code:
                continue
            earlier=[r for r in history if r["date"]<=expected_start]
            if not earlier:
                considered.append({"anchor_code":anchor,"reason":"NO_SOURCE_BOUND_HIGHER_ANCHOR_BEFORE_CHILD_START"})
                continue
            prior=earlier[-1]
            # A lower/equal stock at the child's first board is a peer hypothesis,
            # not a parent inferred from its later success.
            if prior["boards"]<=1:
                continue
            scope_dates=sessions[sessions.index(prior["date"]):idx+1]
            missing_days=[d for d in scope_dates if d not in snapshots]
            not_closing=[d for d in scope_dates if d in snapshots and anchor not in snapshots[d]["rows"]]
            break_day=not_closing[0] if not_closing and not missing_days else None
            co_mentions=[c for c in claims if mentioned(code,c["excerpt"]) and mentioned(anchor,c["excerpt"])]
            questions=[c["claim_id"] for c in co_mentions if "?" in c["excerpt"] or "？" in c["excerpt"]
                       or c.get("source_claim_scope","").startswith("AUTHOR_ASKS_")]
            direct=[c["claim_id"] for c in co_mentions if c["claim_id"] not in questions]
            seeded=[e for e in seed_edges if e["child_code"]==code and e.get("anchor_code")==anchor
                    and e["connection_strength"] in {"EXPLICIT_CO_MENTION","GROUP_ASSOCIATION"}]
            shared=sorted(child_families & set(families(prior["industry"])))
            name_chars=sorted((set(stock["name"])&set(identities[anchor]))-set("股份科技乳业传媒电子健康控新能源食品中国集团"))
            supported = bool(direct or seeded)
            if not supported and not shared and not name_chars and not questions:
                considered.append({"anchor_code":anchor,"reason":"NO_CHECKED_STOCK_CONNECTION","anchor_observation":prior})
                continue
            if not complete:
                timing="CHILD_STREAK_UNQUALIFIED"
            elif break_day==expected_start and old_hit:
                timing="OLD_FOLLOWER_RESTART"
            elif break_day and expected_start<break_day<=signal:
                timing="EARLY_START_SURVIVES_PARENT_BREAK"
            elif break_day and break_day<=expected_start:
                timing="NEW_START_AFTER_PARENT_BREAK"
            else:
                timing="PARENT_BREAK_UNRESOLVED"
            if supported and timing=="OLD_FOLLOWER_RESTART":
                identity="OLD_FOLLOWER_RESTART"
            elif supported and timing=="EARLY_START_SURVIVES_PARENT_BREAK":
                identity="CYCLE_SUPPLEMENT_CANDIDATE"
            elif shared and complete and break_day:
                identity="THEME_RELAY_COMPETING"
            else:
                identity="RELATED_UNRESOLVED"
            hypotheses.append({"anchor_code":anchor,"anchor_name":identities[anchor],
                "anchor_in_current_candidate_pool":anchor in stocks,
                "anchor_observation_before_child_start":prior,
                "parent_not_closing_in_complete_provider_pool_since":break_day,
                "parent_break_scope":"PROVIDER_POOL_NONCONTINUATION_NOT_EXCHANGE_ROLE_TRUTH",
                "missing_pool_days":missing_days, "identity_hypothesis":identity,"timing_type":timing,
                "shared_provider_theme_families":shared,
                "same_theme_is_active_trading_causality":False,
                "direct_co_mention_claim_ids":direct,"question_only_claim_ids":questions,
                "previous_checked_relation_ids":[e["edge_id"] for e in seeded],
                "name_only_shared_characters":name_chars,
                "source_connection_supported":supported,
                "cancellation":["父锚时序或本轮连续板来源被反证","完整同周期对手显示相对地位假设不成立","独立业务/事件解释得到更强证据"],
                "unique_parent_certified":False,"leader_control_verified":False})
        structural=[h["identity_hypothesis"] for h in hypotheses]
        call=("CYCLE_SUPPLEMENT_CANDIDATE" if "CYCLE_SUPPLEMENT_CANDIDATE" in structural else
              "OLD_FOLLOWER_RESTART" if "OLD_FOLLOWER_RESTART" in structural else
              "THEME_RELAY_COMPETING" if "THEME_RELAY_COMPETING" in structural else
              "RELATED_UNRESOLVED" if hypotheses else "UNKNOWN")
        rows.append({"code":code,"name":stock["name"],"boards":n,
            "current_streak":{"inferred_start":expected_start,"dates":streak_dates,
                "complete_closing_streak_checked":complete,"earlier_board_observed":old_hit,
                "start_is_lifetime_first":False},
            "identity_hypothesis":call,"anchor_hypotheses":hypotheses,
            "considered_and_rejected_anchors":considered,
            "same_height_peers":sorted(c for c,s in stocks.items() if c!=code and s["boards"]==n),
            "same_cycle_competitors":[], "formal_role_label":None,
            "promotion_direction":None,"is_calibrated_probability":False,
            "confirmed_catchup_dragon":False})
    bycode={r["code"]:r for r in rows}
    for row in rows:
        relevant={h["anchor_code"] for h in row["anchor_hypotheses"] if h["identity_hypothesis"] in
                  {"CYCLE_SUPPLEMENT_CANDIDATE","OLD_FOLLOWER_RESTART","THEME_RELAY_COMPETING"}}
        for anchor in sorted(relevant):
            peers=[r for r in rows if any(h["anchor_code"]==anchor and h["identity_hypothesis"] in
                   {"CYCLE_SUPPLEMENT_CANDIDATE","OLD_FOLLOWER_RESTART","THEME_RELAY_COMPETING"} for h in r["anchor_hypotheses"])]
            leaders=[r for r in peers if r["boards"]==max(x["boards"] for x in peers)]
            row["same_cycle_competitors"].append({"anchor_code":anchor,"members":[r["code"] for r in peers],
                "all_highest_streak_candidates":[r["code"] for r in leaders],
                "height_lead_is_leader_control":False,"coexistence_paths":["JOINT_PROMOTION","JOINT_FAILURE","DIFFERENTIATION"]})
    return {"anchor_candidates":[{"code":c,"name":identities[c],"observations":h} for c,h in sorted(anchors.items())],
            "rows":rows,"pool_size":len(rows),"edge_seed_only_discovery":False,
            "model_scores_used":False,"author_role_titles_used_as_truth":False}


def project(facts_path, evidence_path, cutoff=None):
    cutoff=o.now() if cutoff is None else o.stamp(cutoff)
    if cutoff>o.now():
        raise ValueError("future discovery cutoff")
    facts,spec=o.read(facts_path),o.read(evidence_path)
    import forum_reasoning_review as forum
    forum.checked(facts)
    if spec["version"]!="supplement-discovery-evidence-1" or any(facts[k]!=spec[k] for k in ("signal_date","target_date")):
        raise ValueError("discovery evidence date/schema mismatch")
    if o.stamp(spec["formed_at"])>cutoff:
        raise ValueError("discovery packet unavailable")
    bank=o.read(spec["relation_bank_path"])
    proof=facts["source_bindings"]|spec["source_bindings"]|bank["source_bindings"]
    proof |= o.bind(facts_path)|o.bind(evidence_path)|o.bind(spec["relation_bank_path"])|o.bind(__file__)
    o.check_sources(proof)
    calendar=o.read(bank["calendar_path"])
    if calendar.get("status")!="verified" or bank["calendar_path"] not in proof:
        raise ValueError("bound verified calendar required")
    if calendar["sessions"][calendar["sessions"].index(facts["signal_date"])+1]!=facts["target_date"]:
        raise ValueError("target must be next official session")
    claims=old._checked_claims(bank,proof,cutoff)
    _,events=old._checked_temporal(bank,proof,cutoff)
    snapshots=read_snapshots(spec,proof,cutoff)
    pool={"signal_date":facts["signal_date"],"rows":[{k:r[k] for k in ("code","name","boards")} for r in facts["rows"]]}
    result=analyze(pool,snapshots,calendar["sessions"],list(claims.values()),bank["edges"],events)
    result.update(version=VERSION,status="DISCOVERED_PARENT_AND_ROLE_HYPOTHESES_AUTHOR_REVIEW_REQUIRED",
        signal_date=facts["signal_date"],target_date=facts["target_date"],formed_at=o.now().isoformat(),
        knowledge_cutoff=cutoff.isoformat(),facts_path=str(Path(facts_path).resolve()),
        evidence_path=str(Path(evidence_path).resolve()),source_bindings=proof,
        historical_snapshot_dates=sorted(snapshots),model_exposure=facts.get("supplement_model_exposure","UNKNOWN"),
        new_fits=0,role_training_labels_created=0,new_formal_forward_samples=0)
    return result


def build(facts_path,evidence_path,output_dir):
    out=Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable discovery directory required")
    result=project(facts_path,evidence_path)
    out.mkdir(parents=True)
    o.write_new(out/"discovery.json",result)
    lines=["# 自主父锚发现与补涨身份区分","","模型分数未参与。以下为身份假设，作者须逐股审阅，不认证补涨龙或资金因果。","",
           "| 股票 | 结构类型 | 父锚假设 | 完整连续板 | 同周期候选 |","|---|---|---|---|---|"]
    for row in result["rows"]:
        supported=[h for h in row["anchor_hypotheses"] if h["identity_hypothesis"]!="RELATED_UNRESOLVED"]
        names="、".join(h["anchor_name"]+"："+h["identity_hypothesis"] for h in supported) or "仅线索/未知"
        peers=sorted({c for p in row["same_cycle_competitors"] for c in p["members"] if c!=row["code"]})
        lines.append(f"| {row['name']} {row['code']} | {row['identity_hypothesis']} | {names} | {row['current_streak']['complete_closing_streak_checked']} | {','.join(peers)} |")
    (out/"自主父锚与身份区分.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def checked(facts):
    if not facts.get("identity_discovery_required"):
        return None
    path=facts["identity_discovery_context_path"]
    if facts["source_bindings"].get(path)!=o.sha(path):
        raise ValueError("identity discovery is not source bound")
    saved=o.read(path)
    rebuilt=project(saved["facts_path"],saved["evidence_path"],saved["knowledge_cutoff"])
    if any(saved[k]!=rebuilt[k] for k in saved if k!="formed_at"):
        raise ValueError("identity discovery differs from original sources")
    actual=o.unique_rows(facts["rows"]);rows=o.unique_rows(saved["rows"])
    if set(actual)!=set(rows) or any(actual[c].get("identity_discovery_context")!=rows[c] for c in rows):
        raise ValueError("identity discovery must retain the complete exact pool")
    return saved


def attach(facts_path,evidence_path,output_dir,draft_path):
    out=Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable identity preparation required")
    facts=o.read(facts_path)
    result=build(facts_path,evidence_path,out/"discovery")
    rows=o.unique_rows(result["rows"])
    updated=deepcopy(facts)
    context_path=out/"discovery/discovery.json"
    updated.update(identity_discovery_required=True,identity_discovery_context_path=str(context_path.resolve()))
    updated["rows"]=[{**r,"identity_discovery_context":rows[r["code"]]} for r in facts["rows"]]
    updated["source_bindings"]=facts["source_bindings"]|result["source_bindings"]|o.bind(context_path)|o.bind(__file__)
    fp=out/"facts_with_identity_discovery.json";o.write_new(fp,updated)
    draft=deepcopy(o.read(draft_path))
    if draft.get("facts_sha256")!=o.sha(facts_path):
        raise ValueError("identity draft must bind its original facts")
    draft.update(facts_path=str(fp.resolve()),facts_sha256=o.sha(fp),
        source_bindings=updated["source_bindings"]|o.bind(fp),author_review_completed=False)
    for row in draft["rows"]:
        row["identity_discovery_assessment"]={"reviewed":False,"identity_call":None,
            "accepted_anchor_codes":[],"rejected_or_pending_anchors":[],
            "support":"","counterevidence":"","cancel_condition":"","same_cycle_competitors":[]}
    o.write_new(out/"author_identity_PENDING.json",draft)
    checked(updated)
    return updated


def validate_review(facts,draft):
    context=checked(facts)
    if context is None:
        return
    authored=o.unique_rows(draft["rows"])
    expected=o.unique_rows(context["rows"])
    if set(authored)!=set(expected):
        raise ValueError("complete identity author review required")
    for code,row in expected.items():
        review=authored[code].get("identity_discovery_assessment",{})
        if review.get("reviewed") is not True or review.get("identity_call") not in CALLS:
            raise ValueError("explicit independent identity call required for every stock")
        for field in ("support","counterevidence","cancel_condition"):
            if not isinstance(review.get(field),str) or not review[field].strip():
                raise ValueError("identity support/counterevidence/cancel reason required")
        discovered={h["anchor_code"] for h in row["anchor_hypotheses"]}
        accepted=review.get("accepted_anchor_codes",[])
        other=review.get("rejected_or_pending_anchors",[])
        if len(accepted)!=len(set(accepted)) or not set(accepted).issubset(discovered):
            raise ValueError("accepted anchor outside discovered hypotheses")
        dispositions={r["anchor_code"]:r for r in other}
        if len(dispositions)!=len(other) or set(accepted)&set(dispositions) or set(accepted)|set(dispositions)!=discovered:
            raise ValueError("every discovered parent needs accept/reject/pending review")
        if any(r.get("decision") not in {"REJECT","PENDING"} or not isinstance(r.get("reason"),str) or not r["reason"].strip() for r in other):
            raise ValueError("parent rejection or uncertainty needs a reason")
        peers=sorted({c for p in row["same_cycle_competitors"] for c in p["members"] if c!=code})
        if review.get("same_cycle_competitors")!=peers:
            raise ValueError("complete same-cycle competition review required")
        if review["identity_call"]=="CYCLE_SUPPLEMENT_CANDIDATE" and not any(
                h["anchor_code"] in accepted and h["identity_hypothesis"]=="CYCLE_SUPPLEMENT_CANDIDATE" for h in row["anchor_hypotheses"]):
            raise ValueError("supplement call requires a supported source-bound cycle hypothesis")

