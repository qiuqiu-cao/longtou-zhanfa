"""Source-bound parent/child hypotheses; no promotion scores or role labels.

Historical anchors live outside the forecast pool. A name/sector connection,
its timing, role identity and execution are deliberately reviewed separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import stock_judgement_review as original

VERSION = "supplement-relation-context-1"
BANK_VERSION = "supplement-relation-bank-1"
DECISIONS = {"ACCEPT", "REJECT", "PENDING"}
FORBIDDEN = {"raw_score", "score", "score_display", "rank", "predicted_positive",
             "original_independent_direction", "promotion_judgement", "target_outcome"}


def _proof(items):
    if isinstance(items, dict):
        return dict(items)
    result = {}
    for item in items:
        path = str(Path(item["path"]).resolve())
        if path in result and result[path] != item["sha256"]:
            raise ValueError("conflicting source binding")
        result[path] = item["sha256"]
    return result


def _pointer(document, pointer):
    value = document
    for token in pointer.strip("/").split("/") if pointer else ():
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _bound_value(binding, proof):
    path = str(Path(binding["path"]).resolve())
    if proof.get(path) != binding["sha256"]:
        raise ValueError("observation is outside checked source bindings")
    if "json_pointer" in binding:
        return _pointer(original.read(path), binding["json_pointer"])
    text = Path(path).read_text(encoding="utf-8-sig")
    excerpt, offset = binding["excerpt"], binding["offset"]
    if not excerpt or not isinstance(offset, int) or offset < 0 or text[offset:offset+len(excerpt)] != excerpt:
        raise ValueError("observation source span changed")
    return binding.get("declared_value")


def _no_scores(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("relation pool must be score and outcome masked")
        for child in value.values():
            _no_scores(child)
    elif isinstance(value, list):
        for child in value:
            _no_scores(child)


def _checked_claims(bank, proof, cutoff):
    sources = {s["source_id"]: s for s in bank["sources"]}
    if len(sources) != len(bank["sources"]):
        raise ValueError("duplicate relation source")
    result = {}
    for claim in bank["claims"]:
        cid = claim["claim_id"]
        if cid in result:
            raise ValueError("duplicate relation claim")
        source = sources[claim["source_id"]]
        path = str(Path(source["text_path"]).resolve())
        if proof.get(path) != source["text_sha256"] or source.get("body_available") is not True:
            raise ValueError("only source-bound readable original discussion is admissible")
        if source.get("kind") != "ORIGINAL_FORUM":
            raise ValueError("search snippets cannot certify a discussion connection")
        if source.get("offset_encoding") == "DECODED_JSON_STRING_UNICODE_CHARACTERS":
            text = original.read(path)
            if not isinstance(text, str):
                raise ValueError("decoded source must be a JSON string")
        else:
            text = Path(path).read_text(encoding="utf-8-sig")
        offset, excerpt = claim["offset"], claim["excerpt"]
        if not isinstance(offset, int) or offset < 0 or not excerpt or text[offset:offset+len(excerpt)] != excerpt:
            raise ValueError("relation claim span changed")
        section = claim.get("source_section", "UNKNOWN_SECTION")
        if section not in {"AUTHOR_BODY", "COMMENT", "UNKNOWN_SECTION"}:
            raise ValueError("author and comment evidence must be distinct")
        # Never inherit the parent post's clock for a comment of unknown age.
        published = claim.get("comment_published_at") if section == "COMMENT" else source.get("published_at")
        obtained = original.stamp(source["obtained_at"])
        reviewed = original.stamp(claim.get("reviewed_at", bank["formed_at"]))
        available = max(obtained, reviewed)
        if published and original.stamp(published) > obtained:
            raise ValueError("discussion publication follows source capture")
        if available > cutoff:
            raise ValueError("relation evidence unavailable at requested cutoff")
        if reviewed < obtained:
            raise ValueError("review precedes source capture")
        day = original.stamp(published).astimezone(original.SHANGHAI).date().isoformat() if published else None
        result[cid] = {"claim_id": cid, "source_id": source["source_id"],
            "source_section": section, "author": claim.get("comment_author", "UNKNOWN_COMMENT_AUTHOR") if section == "COMMENT" else source.get("author", "UNKNOWN"),
            "url": source["url"], "parent_post_published_at": source.get("published_at"),
            "discussion_published_at": published, "obtained_at": obtained.isoformat(),
            "reviewed_at": reviewed.isoformat(), "available_at": available.isoformat(),
            "publication_day": day, "use": "SIGNAL_DAY_DISCUSSION" if day == bank["signal_date"] else "OLDER_DISCUSSION_BACKGROUND" if day else "COMMENT_DATE_UNKNOWN",
            "excerpt": excerpt, "offset": offset, "text_path": path,
            "is_role_truth": False, "is_capital_causality": False,
            "source_claim_scope": claim.get("claim_scope", "AUTHOR_OR_COMMENT_HYPOTHESIS")}
    return result


def _checked_temporal(bank, proof, cutoff):
    observations = {}
    for row in bank.get("temporal_facts", []):
        key = row["code"], row["date"]
        if key in observations or row["date"] > bank["signal_date"]:
            raise ValueError("duplicate or future market observation")
        if original.stamp(row["available_at"]) > cutoff:
            raise ValueError("market observation unavailable at cutoff")
        value = _bound_value(row["ohlc_source"], proof)
        if {k: value[k] for k in ("open", "high", "low", "close")} != row["ohlc_cents"]:
            raise ValueError("OHLC projection changed")
        raw = original.read(row["raw_path"])
        quotes = [v["day"] for k, v in raw["data"].items() if k.endswith(row["code"]) and "day" in v]
        matching = [bar for bars in quotes for bar in bars if bar[0] == row["date"]]
        if len(matching) != 1:
            raise ValueError("raw unadjusted daily identity missing")
        bar = matching[0]
        expected = {k: int(Decimal(bar[i])*100) for k, i in (("open",1),("close",2),("high",3),("low",4))}
        if expected != row["ohlc_cents"]:
            raise ValueError("raw daily differs from normalized source")
        if str(Path(row["raw_path"]).resolve()) not in proof:
            raise ValueError("raw daily is unbound")
        boards = row.get("archived_boards")
        if boards is not None:
            if not isinstance(boards, int) or boards < 1 or not row.get("board_source"):
                raise ValueError("positive boards need their own source")
            if _bound_value(row["board_source"], proof) != boards:
                raise ValueError("board projection changed")
        observations[key] = {"code": row["code"], "date": row["date"], "boards": boards,
            "ohlc_cents": expected, "one_price_ohlc": len(set(expected.values())) == 1,
            "available_at": row["available_at"], "board_source": row.get("board_source"),
            "price_source": row["ohlc_source"], "tradeability_verified": False}
    events = []
    for event in bank.get("events", []):
        if event["date"] > bank["signal_date"] or original.stamp(event["available_at"]) > cutoff:
            raise ValueError("future or unavailable anchor event")
        if _bound_value(event["source"], proof) != event["value"]:
            raise ValueError("anchor event projection changed")
        if event["event"] not in {"STREAK_BREAK", "PRIOR_FIRST_OBSERVED_BOARD"}:
            raise ValueError("unsupported timing event")
        events.append(event)
    return observations, events


def _timing(child, anchor, stock, observations, events, sessions, signal):
    idx = sessions.index(signal)
    n = stock["boards"]
    if idx < n-1:
        return {"state": "UNKNOWN", "missing": ["full current streak calendar"]}
    dates = sessions[idx-n+1:idx+1]
    streak = [observations.get((child, day)) for day in dates]
    valid = all(r is not None and r["boards"] == i+1 for i, r in enumerate(streak))
    breaks = sorted({e["date"] for e in events if e["code"] == anchor and e["event"] == "STREAK_BREAK" and e["value"] is True})
    break_day = breaks[0] if len(breaks) == 1 else None
    prior = sorted(e["date"] for e in events if e["code"] == child and e["event"] == "PRIOR_FIRST_OBSERVED_BOARD" and e["value"] is True)
    start = dates[0] if valid else None
    parent_start = observations.get((anchor, start)) if start and anchor else None
    missing = []
    if not valid:
        missing.append("source-bound complete current streak")
    if not break_day:
        missing.append("explicit parent streak break; missing board is not a break")
    if len(breaks) > 1:
        missing.append("multiple anchor cycles require separately bound cycle scope")
    if start and break_day and start < break_day <= signal and parent_start and parent_start["boards"] and parent_start["boards"] >= 2:
        state = "EARLY_START_CONTINUES_THROUGH_ANCHOR_BREAK"
    elif start and break_day and start == break_day and prior and prior[0] < start:
        state = "OLD_FOLLOWER_RESTART_AT_ANCHOR_BREAK"
    elif start and break_day and start >= break_day:
        state = "START_AT_OR_AFTER_ANCHOR_BREAK_PRIOR_ROLE_UNKNOWN"
    else:
        state = "UNKNOWN"
    return {"state": state, "current_streak_start": start, "current_streak_dates": dates,
        "complete_streak_checked": valid, "prior_first_observed_board": prior[0] if prior else None,
        "prior_observation_is_lifetime_first": False, "anchor_first_observed_streak_break": break_day,
        "anchor_boards_at_current_start": parent_start["boards"] if parent_start else None,
        "child_boards_at_break": observations.get((child,break_day),{}).get("boards"),
        "missing": missing, "timing_is_causal_proof": False}


def project(pool_path, bank_path, cutoff=None, exposure="MODEL_OUTPUT_ALREADY_SEEN"):
    cutoff = original.now() if cutoff is None else original.stamp(cutoff) if isinstance(cutoff, str) else cutoff
    if cutoff > original.now():
        raise ValueError("future knowledge cutoff is not allowed")
    if exposure not in original.EXPOSURES:
        raise ValueError("actual model exposure declaration required")
    pool, bank = original.read(pool_path), original.read(bank_path)
    _no_scores(pool.get("rows", pool.get("candidates")))
    stocks = original.unique_rows(pool.get("rows", pool.get("candidates")))
    if bank["version"] != BANK_VERSION or (pool["signal_date"],pool["target_date"]) != (bank["signal_date"],bank["target_date"]):
        raise ValueError("relation bank version or cohort dates differ")
    if original.stamp(bank["formed_at"]) > cutoff:
        raise ValueError("relation bank unavailable at cutoff")
    if bank.get("anchors_are_forecast_candidates") is not False:
        raise ValueError("anchors must not expand the forecast candidate pool")
    proof = _proof(bank["source_bindings"]) | _proof(pool.get("source_bindings", {}))
    proof |= original.bind(pool_path) | original.bind(bank_path) | original.bind(__file__) | original.bind(original.__file__)
    original.check_sources(proof)
    calendar = original.read(bank["calendar_path"])
    if calendar.get("status") != "verified" or str(Path(bank["calendar_path"]).resolve()) not in proof:
        raise ValueError("source-bound official sessions required")
    sessions = calendar["sessions"]
    i = sessions.index(pool["signal_date"])
    if sessions[i+1] != pool["target_date"]:
        raise ValueError("target must be next official session")
    claims = _checked_claims(bank,proof,cutoff)
    observations,events = _checked_temporal(bank,proof,cutoff)
    anchors = {a["code"]: a for a in bank["anchors"] if a.get("code")}
    if len(anchors) != sum(bool(a.get("code")) for a in bank["anchors"]):
        raise ValueError("duplicate anchor identity")
    for code,anchor in anchors.items():
        if not isinstance(code,str) or len(code) != 6 or not code.isdigit():
            raise ValueError("invalid anchor code")
        if not anchor.get("identity_source") or _bound_value(anchor["identity_source"],proof) != {"code":code,"name":anchor["name"]}:
            raise ValueError("anchor identity must have its own checked source")
    relations, ids = defaultdict(list), set()
    for edge in bank["edges"]:
        rid,child,parent = edge["edge_id"],edge["child_code"],edge.get("anchor_code")
        if rid in ids or child not in stocks or child == parent or parent and parent not in anchors:
            raise ValueError("invalid relation identity or outside-pool child")
        ids.add(rid)
        if edge.get("child",{}).get("name",stocks[child]["name"]) != stocks[child]["name"]:
            raise ValueError("child name/code mismatch")
        if parent and edge.get("parent",{}).get("name",anchors[parent]["name"]) != anchors[parent]["name"]:
            raise ValueError("anchor name/code mismatch")
        texts = [claims[cid] for cid in edge["claim_ids"]]
        if not texts:
            raise ValueError("relation needs checked original discussion")
        strength = edge["connection_strength"]
        if strength not in {"EXPLICIT_CO_MENTION", "GROUP_ASSOCIATION", "SECTOR_CONTEXT_PARENT_UNKNOWN", "NAME_MATCH_ONLY"}:
            raise ValueError("unsupported connection strength")
        child_token = edge.get("child_text_token", stocks[child]["name"][:2])
        parent_token = edge.get("anchor_text_token", anchors[parent]["name"][:2] if parent else None)
        if strength == "EXPLICIT_CO_MENTION" and not any(child_token in t["excerpt"] and parent_token and parent_token in t["excerpt"] for t in texts):
            raise ValueError("explicit relation must name both subjects in one checked span")
        if strength == "GROUP_ASSOCIATION" and (not parent or not any(child_token in t["excerpt"] for t in texts) or not any(parent_token in t["excerpt"] for t in texts)):
            raise ValueError("group relation must have checked child and anchor text")
        timing = _timing(child,parent,stocks[child],observations,events,sessions,pool["signal_date"])
        if strength == "NAME_MATCH_ONLY":
            hypothesis = "NAME_LEAD_ONLY_NOT_SUPPORTED_RELATION"
        elif not parent:
            hypothesis = "SECTOR_DISCUSSION_PARENT_UNKNOWN"
        elif timing["state"] == "EARLY_START_CONTINUES_THROUGH_ANCHOR_BREAK":
            hypothesis = "EARLY_START_CONTINUATION_CANDIDATE"
        elif timing["state"] == "OLD_FOLLOWER_RESTART_AT_ANCHOR_BREAK":
            hypothesis = "OLD_FOLLOWER_RESTART_CANDIDATE"
        else:
            hypothesis = "RELATION_BACKGROUND_TIMING_UNRESOLVED"
        relations[child].append({"relation_id": rid,"child_code":child,"anchor_code":parent,
            "anchor_name":anchors[parent]["name"] if parent else None,
            "anchor_in_current_pool": parent in stocks if parent else False,
            "relation_types":edge["relation_types"],"connection_strength":strength,
            "recognition_state":hypothesis,"timing":timing,"discussion_evidence":texts,
            "alternative_anchors_or_explanations":edge.get("alternative_anchors_or_explanations",[]),
            "cancel_or_weaken_conditions":edge.get("cancel_or_weaken_conditions",[]),
            "author_claim_is_verified_role":False,"fund_causality_confirmed":False,
            "unique_parent_confirmed":False,"confirmed_catchup_dragon":False,
            "promotion_direction":None,"tradeability_verified":False,"training_role_label":None})
    rows = []
    for code,stock in sorted(stocks.items()):
        peers = sorted(k for k,s in stocks.items() if k != code and s["boards"] == stock["boards"])
        if stock.get("same_height_peers",peers) != peers:
            raise ValueError("complete same-height pool differs")
        edges = relations[code]
        parents = {e["anchor_code"] for e in edges if e["anchor_code"]}
        same_parent = sorted(c for c,es in relations.items() if c != code and any(e["anchor_code"] in parents for e in es))
        rows.append({"code":code,"name":stock["name"],"boards":stock["boards"],
            "same_height_peers":peers,"same_anchor_candidate_peers":same_parent,
            "relations":edges,"coverage":"CHECKED_RELATION_BACKGROUND" if edges else "UNKNOWN_IN_BOUNDED_CHECKED_SOURCES",
            "current_day_relation_body_count":sum(t["use"] == "SIGNAL_DAY_DISCUSSION" and t["source_section"] == "AUTHOR_BODY" for e in edges for t in e["discussion_evidence"]),
            "uncovered_is_no_catalyst":False,"unknown_is_negative_evidence":False,
            "promotion_direction":None,"confirmed_catchup_dragon":False,
            "resume_calendar_qualification_gap":stock.get("resume_calendar_qualification_gap",False),
            "next_session_paths":["COMMON_PROMOTION","COMMON_FAILURE","DIFFERENTIATION"]})
    return {"version":VERSION,"status":"RELATION_HYPOTHESES_REQUIRING_INDEPENDENT_REVIEW",
        "formed_at":original.now().isoformat(),"knowledge_cutoff":cutoff.isoformat(),
        "signal_date":pool["signal_date"],"target_date":pool["target_date"],"model_exposure":exposure,
        "pool_path":str(Path(pool_path).resolve()),"bank_path":str(Path(bank_path).resolve()),
        "pool_size":len(rows),"rows":rows,"anchors":bank["anchors"],
        "source_bindings":proof,"role_truth_created":False,"new_fits":0,
        "new_formal_forward_samples":0,"scores_or_directions_changed":False,
        "original_prediction_backfill_allowed":False,"blinding_verified":False}


def build(pool_path,bank_path,output_dir,cutoff=None,exposure="MODEL_OUTPUT_ALREADY_SEEN"):
    result = project(pool_path,bank_path,cutoff,exposure)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable relation output required")
    out.mkdir(parents=True)
    original.write_new(out/"relation_context.json",result)
    lines = ["# 完整候选补涨关系卡", "", "仅识别可复核关系假设；不自动生成晋级方向、资金因果或补涨龙真值。", "",
             "| 股票 | 关系识别与启动节点 | 同位与同父锚候选 |", "|---|---|---|"]
    for row in result["rows"]:
        descriptions = [f"{e['anchor_name'] or '父锚未知'}：{e['recognition_state']}；{e['connection_strength']}；本轮启动{e['timing'].get('current_streak_start')}；旧首板观察{e['timing'].get('prior_first_observed_board')}" for e in row["relations"]]
        lines.append(f"| {row['name']} {row['code']}（{row['boards']}板） | {'；'.join(descriptions) or '限定来源未覆盖，保持未知'} | 同位：{','.join(row['same_height_peers'])}；同锚：{','.join(row['same_anchor_candidate_peers'])} |")
    (out/"完整候选补涨关系卡.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def checked(facts):
    path = facts["supplement_relation_context_path"]
    original.check_sources(facts["source_bindings"])
    if facts["source_bindings"].get(path) != original.sha(path):
        raise ValueError("relation context is not source bound")
    saved = original.read(path)
    regenerated = project(saved["pool_path"],saved["bank_path"],saved["knowledge_cutoff"],saved["model_exposure"])
    if any(saved[k] != regenerated[k] for k in saved if k != "formed_at"):
        raise ValueError("saved relation context changed")
    expected,actual = original.unique_rows(saved["rows"]),original.unique_rows(facts["rows"])
    if set(expected) != set(actual):
        raise ValueError("full relation cohort required")
    for code,row in actual.items():
        if row.get("supplement_relation_context") != expected[code]:
            raise ValueError("row relation projection changed")
    return saved


def assessment_template(row):
    return {"reviewed":False,"reason":"","unknown_not_negative":True,
        "same_height_peers":row["same_height_peers"],"same_anchor_candidate_peers":row["same_anchor_candidate_peers"],
        "relations":[{"relation_id":e["relation_id"],"decision":None,"reason":"",
            "counterevidence":[],"cancel_conditions":[]} for e in row["relations"]],
        "role_identity_confirmed":False,"fund_causality_confirmed":False}


def validate_assessments(context,draft):
    expected,actual = original.unique_rows(context["rows"]),original.unique_rows(draft["rows"])
    if set(expected) != set(actual):
        raise ValueError("every original candidate needs supplement review")
    for code,fact in expected.items():
        assessment = actual[code].get("supplement_assessment",{})
        if assessment.get("reviewed") is not True or not str(assessment.get("reason","")).strip() or assessment.get("unknown_not_negative") is not True:
            raise ValueError("explicit supplement review or unknown reason required")
        if assessment.get("role_identity_confirmed") is not False or assessment.get("fund_causality_confirmed") is not False:
            raise ValueError("hypothesis review cannot certify role or funds causality")
        for key in ("same_height_peers","same_anchor_candidate_peers"):
            if assessment.get(key) != fact[key]:
                raise ValueError("complete competitors required in supplement review")
        decisions = assessment.get("relations",[])
        if len(decisions) != len({e.get("relation_id") for e in decisions}) or {e.get("relation_id") for e in decisions} != {e["relation_id"] for e in fact["relations"]}:
            raise ValueError("all alternative relation hypotheses require separate decisions")
        for decision in decisions:
            if decision.get("decision") not in DECISIONS or not str(decision.get("reason","")).strip():
                raise ValueError("relation needs ACCEPT/REJECT/PENDING and explicit reason")
            if not isinstance(decision.get("counterevidence"),list) or not isinstance(decision.get("cancel_conditions"),list):
                raise ValueError("relation counterevidence and cancellation fields required")
            if decision["decision"] == "ACCEPT" and (not decision["counterevidence"] or not decision["cancel_conditions"]):
                raise ValueError("accepted hypothesis needs opposing evidence and cancellation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pool","bank","output-dir"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--model-exposure",default="MODEL_OUTPUT_ALREADY_SEEN")
    args = p.parse_args()
    r = build(args.pool,args.bank,args.output_dir,exposure=args.model_exposure)
    print({"pool_size":r["pool_size"],"relations":sum(len(x["relations"]) for x in r["rows"]),"new_fits":0})
