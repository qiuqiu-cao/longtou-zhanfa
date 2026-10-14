"""Source-bound per-stock catalysts and discretionary judgement validation.

An optional new entry around the unchanged independent judgement controller.
Forum narrative, business link, event truth, earnings and unknown coverage are
separate facts. No new scores, weights, fits, model selection or orders.
"""
import argparse
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import stock_judgement_review as original


VERSION = "stock-catalyst-context-1"
SOURCE_KINDS = {"COMPANY_DISCLOSURE", "OFFICIAL_EVENT_RELEASE", "ORIGINAL_FORUM", "ORIGINAL_FINANCIAL_REPORT"}
CLAIM_STATES = {"NARRATIVE": {"PRESENT"}, "BUSINESS_LINK": {"PRODUCT_LABEL_RELATION"},
    "EVENT_STATUS": {"UNCONFIRMED_CAUSE_NO_ONGOING_TRANSMISSION", "CONFIRMED_EVENT", "UNCONFIRMED_EVENT"},
    "EARNINGS_IMPACT": {"LIMITED_NO_MAJOR_EFFECT", "MATERIAL_CONFIRMED"}}
NEGATIVE_REASONS = {"PRICE_PATH_WEAKNESS", "PEER_COMPETITION_LOSS", "CYCLE_PRESSURE", "VERIFIED_EVENT_COUNTEREVIDENCE"}
FEATURE_SEMANTICS = {"turnover_vs_prev": "LAST_OBSERVED_RELAY_POOL_NOT_PREVIOUS_SESSION",
    "turnover_reference_date": "UNKNOWN_NOT_EXPORTED_BY_ORIGINAL_FEATURE_ARCHIVE",
    "signal_one_word": "LOW_TURNOVER_OHLC_PROXY_NOT_ACTUAL_ONE_WORD",
    "actual_one_word_daily": "INDEPENDENT_UNADJUSTED_OHLC_IDENTITY_SEPARATE_FROM_PROXY"}


def publication_upper(source):
    value = source["published_at"]
    if source["publication_precision"] == "DATE":
        return datetime.combine(datetime.fromisoformat(value).date()+timedelta(days=1), time(0), original.SHANGHAI)
    if source["publication_precision"] != "EXACT":
        raise ValueError("publication precision must be explicit DATE or EXACT")
    return original.stamp(value)


def validate_source(source, cutoff):
    kind = source["kind"]
    if kind not in SOURCE_KINDS or urlparse(source["url"]).scheme not in {"http", "https"}:
        raise ValueError("unsupported original source kind or URL")
    host = urlparse(source["url"]).hostname or ""
    primary_company = ("sse.com.cn", "szse.cn", "cninfo.com.cn", "epaper.cs.com.cn")
    if kind == "COMPANY_DISCLOSURE" and not any(host == h or host.endswith("."+h) for h in primary_company):
        raise ValueError("company-disclosure source must use checked original disclosure publishers")
    if kind == "OFFICIAL_EVENT_RELEASE" and not (host.endswith(".gov.cn") or any(host == h or host.endswith("."+h) for h in ("who.int", "un.org"))):
        raise ValueError("official event facts cannot be relabeled from forum or aggregator")
    obtained = original.stamp(source["obtained_at"])
    if obtained > cutoff:
        raise ValueError("source obtained after actual cutoff")
    upper = publication_upper(source)
    if source["publication_precision"] == "EXACT" and upper > obtained:
        raise ValueError("publication after claimed source capture")
    if source["publication_precision"] == "DATE" and upper.date()-timedelta(days=1) > obtained.astimezone(original.SHANGHAI).date():
        raise ValueError("publication day after source capture")
    proof = {source["text_path"]: source["text_sha256"], source["receipt_path"]: source["receipt_sha256"]}
    original.check_sources(proof)
    receipt = original.read(source["receipt_path"])
    if source["capture_kind"] == "PUBLIC_HTTP":
        proof[source["raw_path"]] = source["raw_sha256"]
        original.check_sources(proof)
        matches = [r for r in receipt.get("sources", receipt.get("http_requests", []))
                   if r.get("url") == source["url"]]
        if len(matches) != 1:
            raise ValueError("source not uniquely bound to original HTTP receipt")
        item = matches[0]
        if item.get("http_status", item.get("status")) != 200 or item.get("raw_path") != source["raw_path"]:
            raise ValueError("failed HTTP source cannot supply verified body")
        if item.get("raw_sha256", item.get("sha256")) != source["raw_sha256"]:
            raise ValueError("raw source differs from capture receipt")
        if item.get("captured_at", item.get("retrieved_at", item.get("retrieved"))) != source["obtained_at"]:
            raise ValueError("capture timestamp differs from original receipt")
        if item.get("page_reported_publication", item.get("published_at", item.get("published"))) != source["published_at"]:
            raise ValueError("publication label differs from capture receipt")
        if item["text_sha256"] != source["text_sha256"]:
            correction_path = source.get("hash_correction_path")
            if not correction_path:
                raise ValueError("text byte hash differs; explicit original correction required")
            correction = original.read(correction_path)
            proof.update(original.bind(correction_path))
            if correction["original_manifest_sha256"] != source["receipt_sha256"]:
                raise ValueError("hash correction belongs to another capture")
            fixes = [c for c in correction["corrections"] if c["path"] == source["text_path"]]
            if len(fixes) != 1 or fixes[0]["actual_file_bytes_sha256"] != source["text_sha256"] or fixes[0]["original_manifest_text_sha256"] != item["text_sha256"]:
                raise ValueError("hash correction does not reproduce saved text")
    elif source["capture_kind"] == "PUBLIC_WEB_TOOL":
        if receipt.get("source_kind") != "PUBLIC_WEB_TOOL_RESPONSE_NOT_RAW_HTTP" or receipt["capture_completed_at"] != source["obtained_at"]:
            raise ValueError("public web response clock/kind mismatch")
        response = receipt["response"]
        marker = source["web_response_marker"]
        if source["url"] not in marker or response.count(marker) != 1:
            raise ValueError("web extraction must select one exact source")
        segment = response[response.index(marker):].split("\n--------------------------------------------------------------------------------", 1)[0]
        extracted = []
        for line in segment.splitlines():
            match = re.match(r"L\d+:\s?(.*)", line)
            if match:
                extracted.append(match.group(1))
        if "\n".join(extracted) != Path(source["text_path"]).read_text(encoding="utf-8"):
            raise ValueError("web excerpt differs from saved original tool response")
    else:
        raise ValueError("capture kind cannot relabel failed HTTP as success")
    return proof


def validate_claim_kind(claim, source):
    kind = claim["kind"]
    if kind not in CLAIM_STATES or claim["state"] not in CLAIM_STATES[kind]:
        raise ValueError("unsupported claim kind/state")
    if claim["scope"] not in {"STOCK", "EVENT"}:
        raise ValueError("unsupported claim scope")
    if kind in {"BUSINESS_LINK", "EARNINGS_IMPACT"} and source["kind"] != "COMPANY_DISCLOSURE":
        raise ValueError("business/earnings require primary company disclosure")
    if kind == "EVENT_STATUS" and source["kind"] != "OFFICIAL_EVENT_RELEASE":
        raise ValueError("event truth requires original official release")
    if (kind == "EVENT_STATUS") != (claim["scope"] == "EVENT"):
        raise ValueError("event truth and stock facts must stay separate")


def validate_bank(bank_path, candidates, cutoff):
    bank = original.read(bank_path)
    if bank["version"] != VERSION or original.stamp(bank["formed_at"]) > cutoff:
        raise ValueError("checked excerpt bank unavailable at cutoff")
    expected = original.unique_rows(candidates)
    sources = {s["source_id"]: s for s in bank["sources"]}
    if len(sources) != len(bank["sources"]):
        raise ValueError("duplicate source ID")
    proof = original.bind(bank_path)
    for source in sources.values():
        proof.update(validate_source(source, cutoff))
    texts = {key: Path(source["text_path"]).read_text(encoding="utf-8") for key, source in sources.items()}
    claims = []
    for claim in bank["claims"]:
        source = sources[claim["source_id"]]
        validate_claim_kind(claim, source)
        if original.stamp(claim["reviewed_at"]) > original.stamp(bank["formed_at"]):
            raise ValueError("claim review later than complete bank")
        if original.stamp(claim["reviewed_at"]) < original.stamp(source["obtained_at"]):
            raise ValueError("claim reviewed before original capture")
        excerpt, offset = claim["excerpt"], claim["offset"]
        text = texts[claim["source_id"]]
        if not excerpt or type(offset) is not int or offset < 0 or text[offset:offset+len(excerpt)] != excerpt:
            raise ValueError("claim is not exact saved author text")
        if claim["scope"] == "STOCK":
            if claim["code"] not in expected:
                raise ValueError("claim stock outside complete original pool")
            entity = claim["entity_anchor"]
            entity_offset = claim["entity_offset"]
            if entity not in {claim["code"], expected[claim["code"]]["name"]} or text[entity_offset:entity_offset+len(entity)] != entity:
                raise ValueError("stock-to-source anchor is absent or ambiguous")
        elif not claim.get("event_id"):
            raise ValueError("event claim requires an explicit event node")
        claims.append({**claim, "available_at": max(original.stamp(source["obtained_at"]), original.stamp(claim["reviewed_at"])).isoformat(),
                       "published_upper": publication_upper(source).isoformat(), "url": source["url"]})
    return bank, sources, claims, proof


def build_rows(candidates, claims, sources, signal_cutoff, original_cutoff):
    cutoff, old = original.stamp(signal_cutoff), original.stamp(original_cutoff)
    expected = original.unique_rows(candidates)
    usable = []
    for claim in claims:
        validate_claim_kind(claim, sources[claim["source_id"]])
        if claim["scope"] == "STOCK" and claim["code"] not in expected:
            raise ValueError("stock claim outside original pool")
        if original.stamp(claim["available_at"]) <= cutoff:
            usable.append(claim)
    rows = []
    for code, candidate in sorted(expected.items()):
        stock = [c for c in usable if c["scope"] == "STOCK" and c["code"] == code]
        events = {c["event_id"] for c in stock if c.get("event_id")}
        linked = stock + [c for c in usable if c["scope"] == "EVENT" and c["event_id"] in events]
        business = any(c["kind"] == "BUSINESS_LINK" for c in stock)
        narrative = any(c["kind"] == "NARRATIVE" for c in stock)
        earnings = {c["state"] for c in stock if c["kind"] == "EARNINGS_IMPACT"}
        impact = "CONFLICT" if len(earnings) > 1 else "LIMITED" if "LIMITED_NO_MAJOR_EFFECT" in earnings else "MATERIAL" if "MATERIAL_CONFIRMED" in earnings else "UNKNOWN"
        rows.append({"code": code, "name": candidate["name"], "boards": candidate["boards"],
            "actual_one_word_daily": candidate.get("actual_one_word_daily"),
            "feature_semantics": FEATURE_SEMANTICS.copy(),
            "narrative_presence": "PRESENT_IN_CHECKED_SOURCE" if narrative else "UNKNOWN",
            "business_link": "PRIMARY_SOURCE_CONFIRMED" if business else "UNKNOWN",
            "event_states": sorted({c["state"] for c in linked if c["kind"] == "EVENT_STATUS"}),
            "earnings_impact": impact, "coverage": "LIMITED_CHECKED" if stock else "NOT_CHECKED",
            "unknown_is_negative_evidence": False, "role_identity_confirmed": False,
            "evidence": [{**c, "usable_at_original_cutoff": original.stamp(c["available_at"]) <= old and original.stamp(c.get("published_upper", c["available_at"])) <= old}
                         for c in linked]})
    return rows


def build_context(features, bank_path, compatibility, output):
    from model_upgrade_compat import runtime
    moment = original.now()
    inputs, _ = runtime(compatibility)
    data = inputs.load_feature_archive(features, moment.isoformat())
    manifest_path = Path(features)/"manifest.json"
    manifest = original.read(manifest_path)
    bank, sources, claims, proof = validate_bank(bank_path, data["relay"], moment)
    if (bank["signal_date"], bank["target_date"]) != (data["signal_date"], data["target_date"]):
        raise ValueError("bank and source-bound signal/target differ")
    cutoff = original.now().isoformat()
    rows = build_rows(data["relay"], claims, sources, cutoff, manifest["knowledge_cutoff"])
    signal_close = datetime.combine(datetime.fromisoformat(data["signal_date"]).date(), time(15), original.SHANGHAI)
    for row in rows:
        for claim in row["evidence"]:
            claim["available_by_signal_close"] = original.stamp(claim["available_at"]) <= signal_close and original.stamp(claim["published_upper"]) <= signal_close
    for filename in [Path(__file__), original.__file__, Path(features)/"features.json", manifest_path, compatibility, *manifest["source_hashes"]]:
        proof.update(original.bind(filename))
    context = {"version": VERSION, "status": "SOURCE_BOUND_CATALYST_FACTS_NO_MODEL_SCORE_CHANGE",
        "formed_at": cutoff, "knowledge_cutoff": cutoff, "signal_date": data["signal_date"], "target_date": data["target_date"],
        "feature_archive": str(Path(features).resolve()), "original_feature_cutoff": manifest["knowledge_cutoff"],
        "scope": data["scope"], "pool_size": len(rows), "rows": rows, "source_bindings": proof,
        "numeric_training_eligible": False,
        "training_gate": "No complete same-semantics historical source-time training set; do not fill unknown with zero or restart stopped collection.",
        "new_mature_forward_samples": 0, "model_score_rank_or_call_changed": False,
        "automatic_primary_identity_or_entry_allowed": False,
        "source_review_scope": "Only supplied checked original excerpts; unmentioned stock remains unknown, not no catalyst."}
    original.write_new(output, context)
    return context


def validate_catalyst_assessments(context, draft):
    expected, supplied = original.unique_rows(context["rows"]), original.unique_rows(draft["rows"])
    if set(expected) != set(supplied):
        raise ValueError("catalyst assessments require complete original pool")
    for code, row in supplied.items():
        a = row.get("catalyst_assessment", {})
        fact = expected[code]
        if a.get("narrative_presence") != fact["narrative_presence"] or a.get("earnings_impact") != fact["earnings_impact"]:
            raise ValueError("assessment cannot invent or erase checked catalyst/earnings facts")
        if a.get("event_and_business_reviewed") is not True:
            raise ValueError("explicit event and business counterevidence review required")
        reasons = a.get("negative_reason_codes")
        if not isinstance(reasons, list) or not set(reasons).issubset(NEGATIVE_REASONS):
            raise ValueError("unknown coverage, no catalyst or model-negative cannot be negative evidence")
        if row["promotion_judgement"] == "FAIL" and not reasons:
            raise ValueError("negative judgement requires an independent counterevidence category")
        if a.get("turnover_comparison_claim") == "PREVIOUS_SESSION":
            raise ValueError("original turnover reference date is unknown; last observed is not previous session")
    return supplied


def prepare(features, theme_report, mainline_report, compatibility, bank_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    base_path = out/"original_fact_projection.json"
    base = original.prepare(features, theme_report, mainline_report, compatibility, base_path)
    context_path = out/"catalyst_context.json"
    context = build_context(features, bank_path, compatibility, context_path)
    facts = {**base, "prepared_at": original.now().isoformat(), "catalyst_context_path": str(context_path.resolve()),
        "catalyst_context_sha256": original.sha(context_path),
        "source_bindings": base["source_bindings"] | context["source_bindings"] | original.bind(base_path) | original.bind(context_path)}
    indexed = {c["code"]: c for c in context["rows"]}
    facts["rows"] = [{**c, "catalyst_context": indexed[c["code"]], "feature_semantics": FEATURE_SEMANTICS.copy()} for c in base["rows"]]
    output = out/"facts_with_catalysts.json"
    original.write_new(output, facts)
    return facts


def checked_context(facts):
    path = facts["catalyst_context_path"]
    if original.sha(path) != facts["catalyst_context_sha256"]:
        raise ValueError("original catalyst context changed")
    context = original.read(path)
    if context["version"] != VERSION or context["signal_date"] != facts["signal_date"] or context["target_date"] != facts["target_date"]:
        raise ValueError("context and original fact date/schema differ")
    original.check_sources(context["source_bindings"])
    if original.stamp(context["formed_at"]) > original.stamp(facts["prepared_at"]):
        raise ValueError("context formed after complete augmented facts")
    return context


def freeze(facts_path, judgement_path, output):
    facts, draft = original.read(facts_path), original.read(judgement_path)
    context = checked_context(facts)
    validate_catalyst_assessments(context, draft)
    return original.freeze(facts_path, judgement_path, output)


def compare(record_path, baseline_path, outcome_archive, output):
    record = original.read(record_path)
    facts = original.read(record["facts_path"])
    validate_catalyst_assessments(checked_context(facts), record)
    return original.compare(record_path, baseline_path, outcome_archive, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    for name in ("features", "bank", "compatibility", "output"):
        p.add_argument("--"+name, required=True)
    p = sub.add_parser("prepare")
    for name in ("features", "theme-report", "mainline-report", "compatibility", "bank", "output-dir"):
        p.add_argument("--"+name, required=True)
    p = sub.add_parser("freeze")
    for name in ("facts", "judgements", "output"):
        p.add_argument("--"+name, required=True)
    p = sub.add_parser("compare")
    for name in ("record", "baseline", "outcome-archive", "output"):
        p.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_context(args.features, args.bank, args.compatibility, args.output)
    elif args.command == "prepare":
        result = prepare(args.features, args.theme_report, args.mainline_report, args.compatibility, args.bank, args.output_dir)
    elif args.command == "freeze":
        result = freeze(args.facts, args.judgements, args.output)
    else:
        result = compare(args.record, args.baseline, args.outcome_archive, args.output)
    print(json.dumps({"status": result.get("status", "CATALYST_AUGMENTED_FACTS"), "signal_date": result["signal_date"],
        "target_date": result["target_date"], "pool_size": result.get("pool_size"), "model_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
