"""Full-pool discretionary stock judgements, kept separate from model scores.

prepare masks model output; freeze preserves an actual pre-auction judgement;
compare reconstructs mature outcomes through the original source-bound adapter.
No weights, model fitting, ranking changes, orders or production registration.
"""
import argparse
from datetime import datetime, time, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


VERSION = "independent-stock-judgement-1"
SHANGHAI = ZoneInfo("Asia/Shanghai")
EXPOSURES = {"MODEL_OUTPUT_ALREADY_SEEN", "MODEL_NOT_REVIEWED_SELF_ATTESTED"}
CALLS = {"PROMOTE", "FAIL", "UNKNOWN"}
PRICE_FIELDS = ("turnover", "signal_amount", "amplitude", "turnover_vs_prev",
                "price", "signal_one_word")
LEDGER = Path(__file__).resolve().parent / "data" / "independent_stock_judgement" / "registry.jsonl"


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("an actual timezone is required")
    return parsed


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {str(path): sha(path)}


def check_sources(sources):
    for filename, wanted in sources.items():
        if sha(filename) != wanted:
            raise ValueError("source changed: " + filename)


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Existing forecasts, failures and judgements cannot be silently overwritten.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def registrations():
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if LEDGER.exists() else []


def unique_rows(rows):
    out = {}
    for row in rows:
        code = row["code"]
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit() or code in out:
            raise ValueError("duplicate or invalid candidate code")
        out[code] = row
    return out


def pre_auction(signal, target, sessions, moment):
    if signal not in sessions or target not in sessions or sessions.index(target) != sessions.index(signal) + 1:
        raise ValueError("exact next official trading session required")
    close = datetime.combine(datetime.fromisoformat(signal).date(), time(15), SHANGHAI)
    auction = datetime.combine(datetime.fromisoformat(target).date(), time(9, 15), SHANGHAI)
    if not close < moment < auction:
        raise ValueError("judgement must actually form after signal close and before target auction")


def prepare(features, theme_report, mainline_report, compatibility, output):
    from model_upgrade_compat import runtime
    cutoff = now().isoformat()
    inputs, _ = runtime(compatibility)
    data = inputs.load_feature_archive(features, cutoff)
    manifest_path = Path(features) / "manifest.json"
    manifest = read(manifest_path)
    theme, mainline = read(theme_report), read(mainline_report)
    signal, target = data["signal_date"], data["target_date"]
    pre_auction(signal, target, data["calendar"]["sessions"], stamp(cutoff))
    if theme["signal_date"] != signal or mainline["signal_date"] != signal:
        raise ValueError("theme/mainline date mismatch")
    for context in (theme, mainline):
        if stamp(context["knowledge_cutoff"]) > stamp(cutoff) or stamp(context["generated_at"]) > stamp(cutoff):
            raise ValueError("context was unavailable at actual preparation")
        check_sources(context["source_hashes"])
    check_sources(manifest["source_hashes"])
    relay = unique_rows(data["relay"])
    complete = unique_rows([c for c in data["original_pool"]["rows"] if c["boards"] >= 2])
    if set(relay) != set(complete):
        raise ValueError("feature archive lost original relay candidates")
    closing = unique_rows([c for c in theme["all_candidates"] if c["pool_kind"] == "limit_up"])
    if set(closing) != {c["code"] for c in data["original_pool"]["rows"]}:
        raise ValueError("theme closing pool differs from original signal pool")
    sources = {}
    for filename in [manifest_path, Path(features)/"features.json", theme_report, mainline_report,
                     compatibility, Path(__file__), *manifest["source_hashes"],
                     *theme["source_hashes"], *mainline["source_hashes"]]:
        sources.update(bind(filename))
    rows = []
    for code, c in sorted(relay.items()):
        t = closing[code]
        industry = t.get("industry_provider")
        rows.append({"code": code, "name": c["name"], "boards": c["boards"],
            "target_boards": c["boards"] + 1,
            "actual_one_word_daily": c.get("actual_one_word_daily"),
            "daily_price_volume": {k: c["features"].get(k) for k in PRICE_FIELDS},
            "low_turnover_proxy_is_not_actual_one_word": True,
            "provider_industry": industry,
            "category_status": t.get("category_status", "UNKNOWN"),
            "active_trading_theme_verified": t.get("active_trading_theme_verified", False),
            "same_height_peers": sorted(k for k, other in relay.items() if k != code and other["boards"] == c["boards"]),
            "provider_industry_peers_not_active_theme_identity": sorted(k for k, other in closing.items()
                if k != code and industry and other.get("industry_provider") == industry),
            "intraday_absorption": "UNKNOWN_NO_MINUTE_OR_ORDER_BOOK_EVIDENCE"})
    # Only fact fields are exported; neither model calls, scores nor model order.
    facts = {"version": VERSION, "prepared_at": cutoff, "signal_date": signal, "target_date": target,
        "feature_archive": str(Path(features).resolve()), "compatibility_policy": str(Path(compatibility).resolve()),
        "theme_report": str(Path(theme_report).resolve()), "mainline_report": str(Path(mainline_report).resolve()),
        "sessions": data["calendar"]["sessions"], "source_bindings": sources,
        "pool_size": len(rows), "scope": data["scope"], "rows": rows,
        "closing_and_failed_pool_counts": theme["pool_counts"],
        "mainline_status": mainline["status"], "mainline": mainline.get("mainline"),
        "mainline_unknowns": mainline.get("unresolved", []),
        "narratives": [{k: c.get(k) for k in ("theme_name", "stance", "excerpt", "url", "published_at", "obtained_at")}
                       for c in mainline.get("narrative_claims", [])],
        "model_scores_masked": True, "production_prediction_changed": False}
    write_new(output, facts)
    return facts


def validate_fact_projection(facts):
    feature_path = Path(facts["feature_archive"])/"features.json"
    if facts["source_bindings"].get(str(feature_path.resolve())) != sha(feature_path):
        raise ValueError("original feature file is not bound")
    original = read(feature_path)
    expected, actual = unique_rows(original["relay"]), unique_rows(facts["rows"])
    if set(actual) != set(expected) or facts["pool_size"] != len(expected):
        raise ValueError("masked facts lost original candidates")
    if any(facts[k] != original[k] for k in ("signal_date", "target_date")):
        raise ValueError("masked dates differ from original signal")
    if facts["sessions"] != original["calendar"]["sessions"]:
        raise ValueError("official sessions changed")
    for code, c in expected.items():
        row = actual[code]
        peers = sorted(k for k, other in expected.items() if k != code and other["boards"] == c["boards"])
        if any(row[k] != c.get(k) for k in ("name", "boards", "actual_one_word_daily")) or row["same_height_peers"] != peers:
            raise ValueError("stock identity or same-height competition changed")
        if row["daily_price_volume"] != {k: c["features"].get(k) for k in PRICE_FIELDS}:
            raise ValueError("observed price/volume changed")


def validate_judgements(facts, draft):
    expected = unique_rows(facts["rows"])
    actual = unique_rows(draft["rows"])
    if set(expected) != set(actual):
        raise ValueError("all original candidates, including unknowns, are required")
    if draft.get("model_exposure") not in EXPOSURES:
        raise ValueError("declare whether original model output has already been seen")
    for code, row in actual.items():
        if row.get("promotion_judgement") not in CALLS:
            raise ValueError("explicit PROMOTE/FAIL/UNKNOWN required")
        for key in ("role_hypothesis", "relative_thesis", "support", "rebuttal", "cancel_condition"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("missing independent reasoning: " + code + " " + key)
        if not isinstance(row.get("missing"), list):
            raise ValueError("explicit missing evidence list required")
        peers = row.get("comparison_peers")
        if not isinstance(peers, list) or len(peers) != len(set(peers)) or code in peers or not set(peers).issubset(expected):
            raise ValueError("comparison peer must be another original candidate")
        if not set(expected[code]["same_height_peers"]).issubset(peers):
            raise ValueError("same-height opponents cannot be silently dropped")
        refs = row.get("fact_references")
        if not isinstance(refs, list) or not refs:
            raise ValueError("bind reasoning to saved facts or source files")
        if not set(refs).issubset(facts["source_bindings"]):
            raise ValueError("reasoning reference outside checked input scope")
        confirm = row.get("opening_confirmation", {})
        if not all(isinstance(confirm.get(k), str) and confirm[k].strip() for k in ("09:25", "09:35", "10:00")):
            raise ValueError("conditional opening observations required; values remain pending")
    return actual


def freeze(facts_path, judgement_path, output):
    facts, draft = read(facts_path), read(judgement_path)
    if facts["version"] != VERSION:
        raise ValueError("unsupported fact schema")
    check_sources(facts["source_bindings"])
    validate_fact_projection(facts)
    moment = now()
    if stamp(facts["prepared_at"]) > moment:
        raise ValueError("facts are from the future")
    pre_auction(facts["signal_date"], facts["target_date"], facts["sessions"], moment)
    rows = validate_judgements(facts, draft)
    if any((r["signal_date"], r["target_date"]) == (facts["signal_date"], facts["target_date"]) for r in registrations()):
        raise ValueError("same signal/target already registered; preserve original judgement")
    record = {"version": VERSION, "status": "INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE",
        "formed_at": moment.isoformat(), "knowledge_cutoff": moment.isoformat(),
        "signal_date": facts["signal_date"], "target_date": facts["target_date"],
        "facts_path": str(Path(facts_path).resolve()), "facts_sha256": sha(facts_path),
        "source_bindings": facts["source_bindings"] | bind(facts_path) | bind(judgement_path),
        "feature_archive": facts["feature_archive"], "compatibility_policy": facts["compatibility_policy"],
        "model_exposure": draft["model_exposure"],
        "blinding_verified": False, "role_identity_verified_by_this_tool": False,
        "pool_size": len(rows), "rows": [rows[k] for k in sorted(rows)],
        "original_fixed_shadow_or_production_registration": False,
        "original_report_score_rank_or_top5_changed": False,
        "evaluation_rule": "Exact same full pool and source-reconstructed promotion labels; unknowns and abstentions retained; no probability/Brier; one day cannot authorize deployment.",
        "new_mature_samples": 0}
    write_new(output, record)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"signal_date": record["signal_date"], "target_date": record["target_date"],
            "formed_at": record["formed_at"], "path": str(Path(output).resolve()), "sha256": sha(output),
            "pool_size": record["pool_size"], "model_exposure": record["model_exposure"],
            "original_production_registration": False}, ensure_ascii=False) + "\n")
    return record


def metrics(calls, labels):
    if set(calls) != set(labels):
        raise ValueError("metrics require the entire same pool")
    if any(v is not None and type(v) is not bool for v in list(calls.values()) + list(labels.values())):
        raise ValueError("boolean or explicit unknown required")
    known = {k: v for k, v in labels.items() if v is not None}
    decided = {k: calls[k] for k in known if calls[k] is not None}
    tp = sum(calls[k] is True and v is True for k, v in known.items())
    tn = sum(calls[k] is False and v is False for k, v in known.items())
    fp = sum(calls[k] is True and v is False for k, v in known.items())
    fn = sum(calls[k] is False and v is True for k, v in known.items())
    positives = sum(v is True for v in known.values())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0.0 if positives else None)
    return {"registered": len(labels), "known_labels": len(known), "unknown_labels": len(labels)-len(known),
        "decided_known": len(decided), "abstentions_with_known_label": len(known)-len(decided),
        "positive_abstentions": sum(v is True and calls[k] is None for k, v in known.items()),
        "prediction_coverage": sum(v is not None for v in calls.values())/len(labels) if labels else None,
        "label_coverage": len(known)/len(labels) if labels else None,
        "scoring_coverage": len(decided)/len(labels) if labels else None,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": (tp+tn)/len(decided) if decided else None,
        "correct_fraction_of_all_known_labels": (tp+tn)/len(known) if known else None,
        "Precision": precision, "complete_positive_Recall": recall, "F1": f1,
        "Brier": None, "Brier_reason": "NO_CALIBRATED_PROBABILITY"}


def compare(record_path, baseline_path, outcome_archive, output):
    from model_upgrade_compat import runtime
    record, baseline = read(record_path), read(baseline_path)
    if record["version"] != VERSION or record["status"] != "INDEPENDENT_PREAUCTION_JUDGEMENT_NOT_PRODUCTION_FREEZE":
        raise ValueError("original independent record required")
    receipt = [r for r in registrations() if r["path"] == str(Path(record_path).resolve())]
    if len(receipt) != 1 or receipt[0]["sha256"] != sha(record_path):
        raise ValueError("independent registration missing or original judgement changed")
    check_sources(record["source_bindings"])
    facts = read(record["facts_path"])
    if sha(record["facts_path"]) != record["facts_sha256"]:
        raise ValueError("original fact record changed")
    validate_fact_projection(facts)
    validate_judgements(facts, record)
    pre_auction(record["signal_date"], record["target_date"], facts["sessions"], stamp(record["formed_at"]))
    if any(baseline[k] != record[k] for k in ("signal_date", "target_date")):
        raise ValueError("baseline dates differ")
    pre_auction(record["signal_date"], record["target_date"], facts["sessions"], stamp(baseline["generated_at"]))
    if stamp(baseline["knowledge_cutoff"]) > stamp(baseline["generated_at"]):
        raise ValueError("baseline knowledge cutoff is later than its actual formation")
    if baseline["feature_sha256"] != sha(Path(record["feature_archive"])/"features.json"):
        raise ValueError("baseline uses different source inputs")
    check_sources(baseline["source_hashes"])
    base = unique_rows(baseline["relay"]["candidates"])
    human = unique_rows(record["rows"])
    if set(base) != set(human):
        raise ValueError("baseline and human candidate pools differ")
    inputs, _ = runtime(record["compatibility_policy"])
    outcomes = inputs.build_outcomes(record, outcome_archive)
    labels = outcomes["candidates"]
    human_calls = {k: {"PROMOTE": True, "FAIL": False, "UNKNOWN": None}[v["promotion_judgement"]] for k, v in human.items()}
    base_calls = {k: v["predicted_positive"] for k, v in base.items()}
    if any(type(v) is not bool for v in base_calls.values()):
        raise ValueError("original baseline boolean call required")
    combined = {"independent": metrics(human_calls, labels), "BASE": metrics(base_calls, labels)}
    breakdown = {}
    for field in ("boards", "actual_one_word_daily"):
        groups = {}
        for c in facts["rows"]:
            groups.setdefault(str(c.get(field)), []).append(c["code"])
        breakdown[field] = {g: {arm: metrics({k: calls[k] for k in codes}, {k: labels[k] for k in codes})
            for arm, calls in (("independent", human_calls), ("BASE", base_calls))} for g, codes in groups.items()}
    result = {"version": VERSION, "status": "SAME_POOL_SOURCE_BOUND_COMPARISON_ONLY",
        "formed_at": now().isoformat(), "signal_date": record["signal_date"], "target_date": record["target_date"],
        "metrics": combined, "breakdown": breakdown,
        "rows": [{"code": k, "independent_call": human_calls[k], "BASE_call": base_calls[k],
                  "actual_promoted": labels[k]} for k in sorted(human)],
        "source_bindings": record["source_bindings"] | bind(record_path) | bind(baseline_path) | outcomes["sources"],
        "model_exposure": record["model_exposure"], "blinding_verified": False,
        "new_samples_in_original_model_forward_ledger": 0, "automatic_model_or_top5_change_allowed": False,
        "comparability": "FULL_COVERAGE" if all(m["scoring_coverage"] == 1 for m in combined.values()) else "PARTIAL_NO_WINNER_CLAIM",
        "deployment_decision": "NO_AUTOMATIC_DEPLOYMENT_FROM_THIS_SINGLE_DAY_OR_UNBLINDED_RECORD"}
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("features", "theme-report", "mainline-report", "compatibility", "output"):
        p.add_argument("--"+name, required=True)
    p = sub.add_parser("freeze")
    for name in ("facts", "judgements", "output"):
        p.add_argument("--"+name, required=True)
    p = sub.add_parser("compare")
    for name in ("record", "baseline", "outcome-archive", "output"):
        p.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.features, args.theme_report, args.mainline_report, args.compatibility, args.output)
    elif args.command == "freeze":
        result = freeze(args.facts, args.judgements, args.output)
    else:
        result = compare(args.record, args.baseline, args.outcome_archive, args.output)
    print(json.dumps({"status": result.get("status", "MODEL_MASKED_FULL_POOL_FACTS"),
        "signal_date": result["signal_date"], "target_date": result["target_date"],
        "pool_size": result.get("pool_size"), "output": str(Path(args.output).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
