"""Typed, source-bound stock/topic reading; mentions are not trading causes.

Semantic categories and relationships are actual author-review assertions.
Exact spans, entity/topic anchors, source kinds, full coverage and clocks are
checked here. Neither co-occurrence nor a reviewed narrative certifies funds,
leadership, current market consensus or complete theme membership.
"""
import argparse
import hashlib
import json
from pathlib import Path

import leader_role_engine as e
import stock_catalyst_context as c
import stock_judgement_review as original

VERSION = "stock-narrative-review-1"
TYPES = {"STOCK_MENTION", "PRICE_BOARD_SUMMARY", "EXPLICIT_TRADE_NARRATIVE",
         "COMPANY_MARKET_DISCUSSION", "UNKNOWN"}


def digest(claim):
    return hashlib.sha256(json.dumps(claim, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_annotations(bank, reviews, sources, candidates, cutoff):
    if reviews.get("version") != VERSION or e.stamp(reviews["formed_at"]) > cutoff:
        raise ValueError("typed review schema or actual formation is invalid")
    if not nonempty(reviews.get("reviewer")):
        raise ValueError("actual semantic reviewer required")
    required = {i for i, claim in enumerate(bank["claims"]) if claim["kind"] == "NARRATIVE"}
    entries, bindings, topic_definitions = {}, {}, {}
    texts = {sid: Path(s["text_path"]).read_text(encoding="utf-8") for sid, s in sources.items()}
    for review in reviews["reviews"]:
        index = review.get("claim_index")
        if type(index) is not int or index not in required or index in entries:
            raise ValueError("complete unique original narrative claim indices required")
        claim = bank["claims"][index]
        if review.get("claim_sha256") != digest(claim):
            raise ValueError("typed annotation differs from original claim")
        kind = review.get("content_type")
        if kind not in TYPES or not nonempty(review.get("reason")):
            raise ValueError("explicit content type and review reason required")
        source = sources[claim["source_id"]]
        when = e.stamp(review["reviewed_at"])
        if when < max(e.stamp(source["obtained_at"]), e.stamp(bank["formed_at"])) or when > e.stamp(reviews["formed_at"]):
            raise ValueError("typed review must follow actual acquisition and original bank, never backfill")
        code = claim["code"]
        if code not in candidates or review.get("code") != code:
            raise ValueError("typed review changed original stock")
        entry = {**review, "source_id": claim["source_id"], "url": source["url"],
                 "source_published_at": source["published_at"], "source_obtained_at": source["obtained_at"],
                 "original_claim": claim, "fund_causality_confirmed": False,
                 "business_link_confirmed_by_narrative": False, "role_identity_confirmed": False}
        if kind != "UNKNOWN":
            span = review.get("source_span", {})
            text, start = span.get("text"), span.get("offset")
            body = texts[claim["source_id"]]
            if not nonempty(text) or type(start) is not int or start < 0 or body[start:start+len(text)] != text:
                raise ValueError("typed classification must bind an exact saved body span")
            entity = claim["entity_anchor"]
            if kind != "COMPANY_MARKET_DISCUSSION" and entity not in text:
                raise ValueError("classification span must identify this stock, not a neighboring theme")
        else:
            if review.get("topic_id") or review.get("topic_anchor"):
                raise ValueError("unknown annotation cannot supply a guessed topic")
        if kind in {"EXPLICIT_TRADE_NARRATIVE", "COMPANY_MARKET_DISCUSSION"}:
            if not all(nonempty(review.get(k)) for k in ("topic_id", "topic_label", "topic_anchor", "relationship_review")):
                raise ValueError("explicit stock-topic semantic relationship and exact anchor required")
            if review["topic_anchor"] not in review["source_span"]["text"]:
                raise ValueError("topic anchor absent from the stock relationship span")
            if review.get("topic_level") not in {"PARENT", "FOCUS", "NAME_NARRATIVE"}:
                raise ValueError("parent topic, focus branch and name narrative must remain separate")
            definition = (review["topic_label"], review["topic_level"])
            previous = topic_definitions.setdefault(review["topic_id"], definition)
            if previous != definition:
                raise ValueError("same topic_id cannot silently change label or granularity")
            if kind == "EXPLICIT_TRADE_NARRATIVE" and source["kind"] not in {"ORIGINAL_FORUM", "ORIGINAL_FINANCIAL_REPORT"}:
                raise ValueError("company/event source cannot be relabeled a trading narrative")
            if kind == "COMPANY_MARKET_DISCUSSION" and source["kind"] != "COMPANY_DISCLOSURE":
                raise ValueError("company market warning requires actual company disclosure")
            entry["relationship_status"] = "REVIEWED_AUTHOR_REPORTED_LINK_NOT_VERIFIED_FUND_CAUSE"
        elif review.get("topic_id") or review.get("topic_anchor"):
            raise ValueError("stock mentions and price summaries cannot silently acquire a topic")
        entries[index] = entry
        bindings |= e.bind(source["text_path"])
    if set(entries) != required:
        raise ValueError("every original narrative claim must remain explicitly classified or unknown")
    return entries, bindings


def _build(context_path, bank_path, review_path, formed):
    context = e.read(context_path)
    moment = e.stamp(formed)
    if context.get("version") != c.VERSION or e.stamp(context["formed_at"]) > moment or moment > e.stamp(e.now()):
        raise ValueError("original catalyst context or current review clock invalid")
    e.verify(context["source_bindings"])
    bank_path = str(Path(bank_path).resolve())
    if context["source_bindings"].get(bank_path) != e.sha(bank_path):
        raise ValueError("typed review must use the original bound catalyst bank")
    candidates = original.unique_rows(context["rows"])
    bank, sources, _, proof = c.validate_bank(bank_path, list(candidates.values()), moment)
    if (bank["signal_date"], bank["target_date"]) != (context["signal_date"], context["target_date"]):
        raise ValueError("bank and catalyst context dates differ")
    if review_path is None:
        annotations = {i: {"claim_index": i, "code": claim["code"], "content_type": "UNKNOWN",
                          "reason": "NO_TYPED_SEMANTIC_REVIEW", "original_claim": claim,
                          "source_id": claim["source_id"], "reviewed_at": None}
                       for i, claim in enumerate(bank["claims"]) if claim["kind"] == "NARRATIVE"}
    else:
        reviews = e.read(review_path)
        if reviews.get("bank_path") != bank_path or reviews.get("bank_sha256") != e.sha(bank_path):
            raise ValueError("typed semantic review is not bound to the exact original bank")
        annotations, extra = validate_annotations(bank, reviews, sources, candidates, moment)
        proof |= extra | e.bind(review_path)
    rows = []
    for code, original_row in sorted(candidates.items()):
        entries = [entry for entry in annotations.values() if entry["code"] == code]
        trade = [entry for entry in entries if entry["content_type"] == "EXPLICIT_TRADE_NARRATIVE"]
        current = [entry for entry in trade if str(entry["source_published_at"])[:10] == context["signal_date"]]
        rows.append({"code": code, "name": original_row["name"], "boards": original_row["boards"],
                     "original_untyped_narrative_presence": original_row["narrative_presence"],
                     "original_flag_does_not_certify_trade_topic": True,
                     "content_counts": {kind: sum(x["content_type"] == kind for x in entries) for kind in sorted(TYPES)},
                     "explicit_trade_narratives": trade, "signal_day_trade_narratives": current,
                     "current_trade_narrative_presence": "PRESENT_IN_LIMITED_REVIEWED_SOURCE" if current else "UNKNOWN",
                     "signal_day_publication_does_not_certify_intraday_topic_activity": True,
                     "prior_discussion_does_not_certify_current_activity": True,
                     "company_market_discussions": [x for x in entries if x["content_type"] == "COMPANY_MARKET_DISCUSSION"],
                     "all_original_narrative_annotations": entries,
                     "coverage": "TYPED_ALL_SUPPLIED_CLAIMS" if entries and review_path else "NO_TYPED_COVERAGE",
                     "unknown_is_negative_evidence": False, "full_theme_membership_confirmed": False,
                     "fund_causality_confirmed": False, "role_identity_confirmed": False})
    proof |= context["source_bindings"] | e.bind(context_path) | e.bind(__file__)
    if review_path: proof |= e.bind(review_path)
    topic_groups = {}
    for row in rows:
        for entry in row["signal_day_trade_narratives"]:
            topic = topic_groups.setdefault(entry["topic_id"], {
                "topic_id": entry["topic_id"], "topic_label": entry["topic_label"],
                "topic_level": entry["topic_level"], "members": {},
                "scope": "EXPLICIT_SOURCE_LINKS_WITHIN_ORIGINAL_RELAY_COHORT_ONLY",
                "complete_theme_membership": False, "first_board_and_failed_board_denominators": "UNKNOWN_OUTSIDE_THIS_CONTEXT",
                "mainline_or_leader_identity_confirmed": False})
            topic["members"][row["code"]] = {"name": row["name"], "boards": row["boards"],
                                            "source_id": entry["source_id"]}
    for group in topic_groups.values():
        group["observed_board_distribution"] = {str(board): sum(r["boards"] == board for r in group["members"].values())
                                                for board in sorted({r["boards"] for r in group["members"].values()})}
    return {"version": VERSION, "formed_at": formed, "signal_date": context["signal_date"],
            "target_date": context["target_date"], "context_path": str(Path(context_path).resolve()),
            "context_sha256": e.sha(context_path), "bank_path": bank_path, "bank_sha256": e.sha(bank_path),
            "review_path": str(Path(review_path).resolve()) if review_path else None,
            "review_sha256": e.sha(review_path) if review_path else None,
            "pool_size": len(rows), "original_claim_count": len(bank["claims"]),
            "original_narrative_count": len(annotations), "rows": rows, "source_bindings": proof,
            "signal_day_source_topic_groups": list(topic_groups.values()),
            "status": "CURRENT_SOURCE_CONTENT_REVIEW_NOT_ORIGINAL_FORECAST", "prediction_registered": False,
            "numeric_fit_or_score_change": False, "new_mature_forward_samples": 0,
            "historical_original_time_training_eligible": False,
            "semantic_review_is_human_assertion_not_auto_market_fact": True}


def build(context_path, bank_path, review_path, output):
    result = _build(context_path, bank_path, review_path, e.now())
    e.write_new(output, result)
    return result


def checked(facts):
    path = facts.get("narrative_review_context_path")
    if path is None:
        return None
    e.verify({path: facts["narrative_review_context_sha256"]})
    saved = e.read(path)
    e.verify(saved["source_bindings"])
    if saved != _build(saved["context_path"], saved["bank_path"], saved["review_path"], saved["formed_at"]):
        raise ValueError("typed narrative projection differs from original source reconstruction")
    if (saved["signal_date"], saved["target_date"]) != (facts["signal_date"], facts["target_date"]):
        raise ValueError("typed narrative dates changed")
    if saved["context_path"] != facts["catalyst_context_path"] or e.stamp(saved["formed_at"]) > e.stamp(facts["prepared_at"]):
        raise ValueError("typed narrative context or formation differs from original facts")
    bycode = original.unique_rows(saved["rows"])
    actual = original.unique_rows(facts["rows"])
    if set(bycode) != set(actual) or any(row.get("reviewed_narrative") != bycode[code] for code, row in actual.items()):
        raise ValueError("typed narrative display changed or omitted an original candidate")
    return saved


def validate_assessments(context, draft):
    expected, actual = original.unique_rows(context["rows"]), original.unique_rows(draft["rows"])
    if set(expected) != set(actual):
        raise ValueError("typed narrative judgement must preserve complete original candidates")
    for code, row in actual.items():
        value = row.get("narrative_type_assessment", {})
        if value.get("current_trade_narrative_presence") != expected[code]["current_trade_narrative_presence"]:
            raise ValueError("mention/summary cannot become current trading narrative")
        if value.get("content_types_reviewed") is not True or value.get("unknown_not_negative") is not True:
            raise ValueError("typed coverage must be reviewed; unknown is not negative evidence")
        if not nonempty(value.get("reason")) or value.get("fund_causality_confirmed") is not False:
            raise ValueError("explicit interpretation and no automatic funding cause required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalyst-context", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--reviews")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.catalyst_context, args.bank, args.reviews, args.output)
    print(json.dumps({key: result[key] for key in ("status", "pool_size", "original_narrative_count", "prediction_registered")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
