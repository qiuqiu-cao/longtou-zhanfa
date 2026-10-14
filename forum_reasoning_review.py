"""Source-bound forum arguments -> independent expectations, without new scores.

This additive entry leaves the original review modules and their hashes intact.
Discussion presence, factual verification and a discretionary hypothesis are
separate fields; role-training labels are not required to make a judgement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path

import stock_catalyst_context as catalyst
import stock_judgement_review as original
import supplement_role_review as supplement


VERSION = "forum-reasoning-review-1"
DECISIONS = {"ACCEPT", "REJECT", "PENDING"}
EFFECTS = {"SUPPORT_PROMOTE", "SUPPORT_FAIL", "ROLE_ONLY", "NEUTRAL", "UNCERTAIN"}
AGREEMENTS = {"AGREEMENT", "PARTIAL", "DISAGREEMENT", "UNDETERMINED"}
FORMS = {"OPINION", "QUESTION", "HYPOTHESIS", "OBSERVATION"}
SCOPES = {"MARKET_OBSERVATION", "COMPANY_BUSINESS", "EVENT_STATUS", "DISCUSSION_PRESENCE", "ROLE_HYPOTHESIS", "FUNDS_CAUSALITY"}


def _key(entry):
    return (entry["source_id"], entry["source_section"], entry["offset"], entry["excerpt"])


def opinion_id(entry):
    encoded = json.dumps(_key(entry), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "opinion_" + hashlib.sha256(encoded).hexdigest()[:24]


def build(facts):
    """Build argument inputs from checked facts; never infer a forecast."""
    relation_context = supplement.checked(facts)
    inputs = facts["forum_preread_checked_input_paths"]
    pool = original.read(inputs["pool"])
    forum = original.read(facts["forum_preread_context_path"])
    _, sources, claims, _ = catalyst.validate_bank(inputs["bank"], pool.get("rows", pool.get("candidates")), original.stamp(forum["formed_at"]))
    claim_map = {}
    for claim in claims:
        claim_map.setdefault((claim["code"] if claim["scope"] == "STOCK" else None, *_key({**claim, "source_section": claim.get("source_section", "UNKNOWN_SECTION")})), []).append(claim)
    rows = []
    for fact in facts["rows"]:
        collected = {}
        for entry in fact["forum_preread_context"]["forum_evidence"]:
            item = deepcopy(entry)
            source = sources[item["source_id"]]
            checked_claims = claim_map.get((fact["code"], *_key(item)), [])
            item.update({"parent_post_published_at": item.get("published_at"), "text_path": source["text_path"],
                "reviewed_at": max((c["reviewed_at"] for c in checked_claims), key=original.stamp) if checked_claims else None,
                "available_at": max((c["available_at"] for c in checked_claims), key=original.stamp) if checked_claims else source["obtained_at"],
                "origins": ["FORUM_PREREAD"], "relation_ids": []})
            collected[_key(item)] = item
        for relation in fact["supplement_relation_context"]["relations"]:
            for entry in relation["discussion_evidence"]:
                item = deepcopy(entry)
                key = _key(item)
                if key in collected:
                    previous = collected[key]
                    for field in ("author", "url", "discussion_published_at"):
                        if previous.get(field) != item.get(field):
                            raise ValueError("same argument identity has conflicting author or publication")
                    available = max(original.stamp(previous["available_at"]), original.stamp(item["available_at"]))
                    origins, relations = previous["origins"], previous["relation_ids"]
                    previous.update(item)
                    previous.update(available_at=available.isoformat(), origins=origins, relation_ids=relations)
                    item = previous
                else:
                    item.update(origins=[], relation_ids=[])
                    collected[key] = item
                if "SUPPLEMENT_RELATION" not in item["origins"]:
                    item["origins"].append("SUPPLEMENT_RELATION")
                if relation["relation_id"] not in item["relation_ids"]:
                    item["relation_ids"].append(relation["relation_id"])
        opinions = []
        for key, item in sorted(collected.items()):
            item.update({"opinion_id": opinion_id(item), "source_contains_question": "?" in item["excerpt"] or "？" in item["excerpt"],
                "opinion_is_verified_market_fact": False, "author_popularity_is_accuracy_evidence": False,
                "promotion_direction_supplied": False})
            opinions.append(item)
        rows.append({"code": fact["code"], "name": fact["name"], "boards": fact["boards"],
            "coverage": fact["forum_preread_context"]["coverage"], "opinions": opinions,
            "unknown_is_negative_evidence": False, "unknown_is_positive_evidence": False})
    discussion_paths = {facts["forum_preread_context_path"], facts["supplement_relation_context_path"], inputs["bank"], relation_context["bank_path"]}
    for source in sources.values():
        if source["kind"] == "ORIGINAL_FORUM":
            discussion_paths.update(source[k] for k in ("text_path", "raw_path", "receipt_path") if source.get(k))
    relation_bank = original.read(relation_context["bank_path"])
    for source in relation_bank["sources"]:
        discussion_paths.update(source[k] for k in ("text_path", "raw_path", "receipt_path") if source.get(k))
    qualified_paths = {scope: set() for scope in SCOPES}
    for source in sources.values():
        scope = {"COMPANY_DISCLOSURE": "COMPANY_BUSINESS", "OFFICIAL_EVENT_RELEASE": "EVENT_STATUS", "ORIGINAL_FORUM": "DISCUSSION_PRESENCE"}.get(source["kind"])
        if scope:
            qualified_paths[scope].update(source[k] for k in ("text_path", "raw_path") if source.get(k))
    for source in relation_bank["sources"]:
        qualified_paths["DISCUSSION_PRESENCE"].update(source[k] for k in ("text_path", "raw_path") if source.get(k))
    for observation in relation_bank["temporal_facts"]:
        qualified_paths["MARKET_OBSERVATION"].update(observation[k] for k in ("source_path", "raw_path") if observation.get(k))
        if observation.get("ohlc_source", {}).get("path"):
            qualified_paths["MARKET_OBSERVATION"].add(observation["ohlc_source"]["path"])
    qualified_paths = {scope: sorted(str(Path(path).resolve()) for path in paths if str(Path(path).resolve()) in facts["source_bindings"]) for scope, paths in qualified_paths.items()}
    return {"version": VERSION, "status": "FORUM_ARGUMENT_INPUTS_NO_FORECAST", "formed_at": original.now().isoformat(),
        "signal_date": facts["signal_date"], "target_date": facts["target_date"], "pool_size": len(rows), "rows": rows,
        "source_bindings": facts["source_bindings"] | original.bind(__file__),
        "discussion_only_paths": sorted(discussion_paths),
        "verification_paths_by_scope": qualified_paths,
        "new_fits": 0, "promotion_directions_autofilled": False, "role_truth_created": False}


def draft_template(context):
    return {"rows": [{"code": row["code"], "name": row["name"], "boards": row["boards"],
        "forum_reasoning_assessment": {"reviewed": False, "synthesis": "", "agreement": None,
            "agreement_reason": "", "unknown_not_negative": True,
            "opinions": [{"opinion_id": item["opinion_id"], "author_claim": "", "author_reasoning": "",
                "testable_expectation": "", "author_claim_form": None, "author_call": "UNSPECIFIED",
                "author_expectation_target_date": None,
                "verification": {"status": None, "scope": None, "reason": "", "evidence_refs": []},
                "decision": None, "reason": "", "counterevidence": [], "independent_effect": None,
                "cancel_conditions": []} for item in row["opinions"]]}} for row in context["rows"]]}


def _text(value, message):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)


def _texts(value, message):
    if not isinstance(value, list) or any(not isinstance(s, str) or not s.strip() for s in value):
        raise ValueError(message)


def validate(context, draft, source_bindings=None):
    """Require an actual argument review; UNKNOWN does not prohibit a hypothesis."""
    if context.get("version") != VERSION:
        raise ValueError("unsupported forum reasoning schema")
    expected, actual = original.unique_rows(context["rows"]), original.unique_rows(draft["rows"])
    if set(expected) != set(actual):
        raise ValueError("every original candidate needs forum reasoning or explicit unknown")
    bindings = context["source_bindings"] | (source_bindings or {})
    original.check_sources(bindings)
    forum_files = {p["text_path"] for row in context["rows"] for p in row["opinions"]} | set(context.get("discussion_only_paths", []))
    for code, fact in expected.items():
        assessment = actual[code].get("forum_reasoning_assessment", {})
        if assessment.get("reviewed") is not True or assessment.get("unknown_not_negative") is not True:
            raise ValueError("all candidates need explicit forum reasoning review")
        _text(assessment.get("synthesis"), "independent synthesis or unknown explanation required")
        if assessment.get("agreement") not in AGREEMENTS:
            raise ValueError("explicit agreement or disagreement with forum arguments required")
        _text(assessment.get("agreement_reason"), "agreement requires independent evidence or unknown reason")
        opinions = assessment.get("opinions", [])
        if not isinstance(opinions, list) or any(not isinstance(p, dict) for p in opinions):
            raise ValueError("structured opinion decisions required")
        observed = {p.get("opinion_id"): p for p in opinions}
        wanted = {p["opinion_id"]: p for p in fact["opinions"]}
        if len(observed) != len(opinions) or set(observed) != set(wanted):
            raise ValueError("all checked arguments need separate decisions; no omission or duplication")
        if not wanted and assessment["agreement"] != "UNDETERMINED":
            raise ValueError("uncovered stocks cannot claim agreement with absent authors")
        for oid, opinion in observed.items():
            source = wanted[oid]
            for key in ("author_claim", "author_reasoning", "testable_expectation", "reason"):
                _text(opinion.get(key), "missing structured argument reasoning: " + key)
            if opinion.get("author_claim_form") not in FORMS or opinion.get("author_call") not in {"PROMOTE", "FAIL", "UNSPECIFIED"}:
                raise ValueError("author assertion form and actual forecast must be explicit")
            target = opinion.get("author_expectation_target_date")
            if target is not None:
                if not isinstance(target, str) or date.fromisoformat(target).isoformat() != target:
                    raise ValueError("author expectation target date must be an ISO date or null")
            if opinion["author_claim_form"] == "QUESTION" and (opinion["author_call"] != "UNSPECIFIED" or target is not None):
                raise ValueError("a question cannot inherit an explicit author forecast")
            if source.get("source_claim_scope", "").startswith("AUTHOR_ASKS_") and opinion["author_claim_form"] != "QUESTION":
                raise ValueError("a checked author question cannot be relabelled as a declaration")
            if opinion["author_call"] != "UNSPECIFIED" and target is None:
                raise ValueError("explicit author forecasts need their own target date")
            if source.get("use") != "SIGNAL_DAY_DISCUSSION" and target == context["target_date"]:
                d = date.fromisoformat(target)
                tokens = {target, f"{d.month}月{d.day}日", f"{d.month:02d}月{d.day:02d}日"}
                if not any(token in source["excerpt"] for token in tokens):
                    raise ValueError("old tomorrow cannot become a current author forecast")
            verification = opinion.get("verification", {})
            if verification.get("status") not in {"VERIFIED", "CONTRADICTED", "UNKNOWN"} or verification.get("scope") not in SCOPES:
                raise ValueError("separate factual verification status and scope required")
            _text(verification.get("reason"), "verification requires a reason")
            refs = verification.get("evidence_refs")
            _texts(refs, "verification references must be saved source paths")
            if not set(refs).issubset(bindings):
                raise ValueError("verification references are outside checked inputs")
            nonforum = [path for path in refs if path not in forum_files and Path(path).suffix.lower() not in {".py", ".pyc"}]
            if verification["status"] in {"VERIFIED", "CONTRADICTED"}:
                if not refs:
                    raise ValueError("verified or contradicted claims require bound evidence")
                if verification["scope"] != "DISCUSSION_PRESENCE" and not nonforum:
                    raise ValueError("discussion alone cannot verify a market, business, event or role claim")
                if verification["status"] == "VERIFIED" and verification["scope"] in {"ROLE_HYPOTHESIS", "FUNDS_CAUSALITY"}:
                    raise ValueError("this argument layer cannot certify role identity or funds causality")
                if verification["scope"] in {"MARKET_OBSERVATION", "COMPANY_BUSINESS", "EVENT_STATUS", "DISCUSSION_PRESENCE"} and not set(refs).intersection(context["verification_paths_by_scope"][verification["scope"]]):
                    raise ValueError("factual verification requires a qualified source for its scope")
            if opinion.get("decision") not in DECISIONS or opinion.get("independent_effect") not in EFFECTS:
                raise ValueError("each argument needs ACCEPT/REJECT/PENDING and independent effect")
            _texts(opinion.get("counterevidence"), "counterevidence must be explicit text entries")
            _texts(opinion.get("cancel_conditions"), "cancellation conditions must be explicit text entries")
            if opinion["decision"] == "PENDING" and opinion["independent_effect"] in {"SUPPORT_PROMOTE", "SUPPORT_FAIL"}:
                raise ValueError("pending arguments cannot become definite directional support")
            if opinion["decision"] == "REJECT" and opinion["independent_effect"] in {"SUPPORT_PROMOTE", "SUPPORT_FAIL", "ROLE_ONLY"}:
                raise ValueError("rejected arguments cannot supply accepted directional or role support")
            if opinion["decision"] == "ACCEPT" and (not opinion["counterevidence"] or not opinion["cancel_conditions"]):
                raise ValueError("accepted reasoning requires opposing evidence and cancellation")
    return actual


def attach(facts_path, output_dir, draft_path=None):
    """Append argument inputs to an existing exact full-pool facts file."""
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable forum reasoning output required")
    facts = original.read(facts_path)
    context = build(facts)
    context["source_bindings"].update(original.bind(facts_path))
    out.mkdir(parents=True)
    context_path = out / "forum_reasoning_context.json"
    original.write_new(context_path, context)
    bycode = original.unique_rows(context["rows"])
    merged = {**facts, "rows": [{**r, "forum_reasoning_context": bycode[r["code"]]} for r in facts["rows"]],
        "forum_reasoning_required": True, "forum_reasoning_base_facts_path": str(Path(facts_path).resolve()),
        "forum_reasoning_context_path": str(context_path.resolve()),
        "source_bindings": context["source_bindings"] | original.bind(context_path)}
    path = out / "facts_with_forum_reasoning.json"
    original.write_new(path, merged)
    draft = original.read(draft_path) if draft_path else supplement._draft(merged, path, merged["supplement_context_diagnostic_only"])
    if set(original.unique_rows(draft["rows"])) != set(bycode):
        raise ValueError("existing draft must retain the exact full pool")
    assessments = original.unique_rows(draft_template(context)["rows"])
    draft = deepcopy(draft)
    for row in draft["rows"]:
        row["forum_reasoning_assessment"] = assessments[row["code"]]["forum_reasoning_assessment"]
    draft.update({"facts_path": str(path.resolve()), "facts_sha256": original.sha(path), "forum_reasoning_required": True,
        "source_bindings": merged["source_bindings"] | original.bind(path), "author_review_completed": False})
    original.write_new(out / "author_draft_PENDING.json", draft)
    checked(merged)
    return merged


def checked(facts):
    if facts.get("forum_reasoning_required") is not True:
        raise ValueError("forum argument reasoning stage required")
    original.check_sources(facts["source_bindings"])
    base_path, context_path = facts["forum_reasoning_base_facts_path"], facts["forum_reasoning_context_path"]
    if any(facts["source_bindings"].get(path) != original.sha(path) for path in (base_path, context_path)):
        raise ValueError("forum reasoning must bind its exact base facts and argument context")
    base, saved = original.read(base_path), original.read(context_path)
    regenerated = build(base)
    for key in ("version", "status", "signal_date", "target_date", "pool_size", "rows", "new_fits", "promotion_directions_autofilled", "role_truth_created", "discussion_only_paths", "verification_paths_by_scope"):
        if saved.get(key) != regenerated.get(key):
            raise ValueError("saved forum reasoning does not match checked original discussion")
    expected, actual = original.unique_rows(base["rows"]), original.unique_rows(facts["rows"])
    inputs = original.unique_rows(saved["rows"])
    if set(expected) != set(actual):
        raise ValueError("forum reasoning lost original candidates")
    if any(facts.get(key) != value for key, value in base.items() if key not in {"rows", "source_bindings"}):
        raise ValueError("original dates, review mode or fact metadata changed")
    for code, row in actual.items():
        if {k: v for k, v in row.items() if k != "forum_reasoning_context"} != expected[code] or row.get("forum_reasoning_context") != inputs[code]:
            raise ValueError("original facts or argument row projection changed")
    supplement.checked(facts)
    return saved


def prepare(*args, **kwargs):
    """Run mandatory forum/role preparation and append the argument review."""
    if "output_dir" in kwargs:
        out = Path(kwargs["output_dir"])
        kwargs["output_dir"] = out / "supplement_review"
    else:
        args = list(args)
        out = Path(args[6])
        args[6] = out / "supplement_review"
    if out.exists():
        raise FileExistsError("new immutable forum reasoning output required")
    supplement.prepare(*args, **kwargs)
    return attach(out / "supplement_review" / "facts_with_supplement.json", out / "reasoning")


def _validate_paths(facts_path, draft_path):
    facts, draft = original.read(facts_path), original.read(draft_path)
    context = checked(facts)
    if draft.get("facts_path") != str(Path(facts_path).resolve()) or draft.get("facts_sha256") != original.sha(facts_path):
        raise ValueError("argument judgement must bind its exact facts")
    if draft.get("author_review_completed") is not True:
        raise ValueError("forum argument draft is pending actual author review")
    if draft.get("model_exposure") != facts["supplement_model_exposure"]:
        raise ValueError("actual model exposure must match the checked facts")
    validate(context, draft, facts["source_bindings"])
    return facts, draft


def freeze(facts_path, judgements_path, output):
    _validate_paths(facts_path, judgements_path)
    return supplement.freeze(facts_path, judgements_path, output)


def review_context(facts_path, judgements_path, output):
    _validate_paths(facts_path, judgements_path)
    return supplement.review_context(facts_path, judgements_path, output)


def compare(record, baseline, outcome_archive, output):
    saved = original.read(record)
    facts = original.read(saved["facts_path"])
    validate(checked(facts), saved, facts["source_bindings"])
    return supplement.compare(record, baseline, outcome_archive, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("attach")
    p.add_argument("--facts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--draft")
    p = sub.add_parser("prepare")
    for field in ("features", "theme-report", "mainline-report", "compatibility", "bank", "calendar", "output-dir", "pool", "relation-bank"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--model-exposure", default="MODEL_OUTPUT_ALREADY_SEEN")
    p.add_argument("--role-evidence")
    for name in ("freeze", "review-context"):
        p = sub.add_parser(name)
        for field in ("facts", "judgements", "output"):
            p.add_argument("--" + field, required=True)
    p = sub.add_parser("compare")
    for field in ("record", "baseline", "outcome-archive", "output"):
        p.add_argument("--" + field, required=True)
    args = parser.parse_args()
    if args.command == "attach":
        result = attach(args.facts, args.output_dir, args.draft)
    elif args.command == "prepare":
        result = prepare(features=args.features, theme_report=args.theme_report, mainline_report=args.mainline_report,
            compatibility=args.compatibility, bank=args.bank, calendar=args.calendar, output_dir=args.output_dir,
            pool=args.pool, relation_bank=args.relation_bank, exposure=args.model_exposure, role_evidence=args.role_evidence)
    elif args.command == "compare":
        result = compare(args.record, args.baseline, args.outcome_archive, args.output)
    else:
        result = globals()[args.command.replace("-", "_")](args.facts, args.judgements, args.output)
    print(json.dumps({"status": result.get("status"), "pool_size": result.get("pool_size", len(result.get("rows", [])))}, ensure_ascii=False))
