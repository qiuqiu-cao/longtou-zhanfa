"""Optional role layer around unchanged catalyst/discretionary production APIs."""
import argparse
import json
from pathlib import Path

import leader_role_engine as e
import leader_role_research as research
import stock_catalyst_context as catalyst
import stock_judgement_review as original
import stock_short_term_context as short_term
import stock_expectation_review as expectation


def prepare(features, theme_report, mainline_report, compatibility, bank, calendar, output_dir, role_evidence=None, history_datasets=(), mainline_history=()):
    out = Path(output_dir)
    facts = catalyst.prepare(features, theme_report, mainline_report, compatibility, bank, out/"catalysts")
    dataset = e.from_archive(features, calendar, theme_report, out/"role_dataset.original.json")
    dataset = e.augment(out/"role_dataset.original.json", out/"role_dataset.json", evidence_path=role_evidence)
    roles = e.analyse(dataset, "STRICT")
    e.write_new(out/"role_facts.json", roles)
    short_context_path = out/"short_term_context.json"
    short_context = short_term.build(out/"role_dataset.json", facts["catalyst_context_path"], short_context_path, "STRICT", history_datasets, [*mainline_history, mainline_report])
    packet_path = out/"blind_identity_packet.json"
    e.write_new(packet_path, research.masked_packet(dataset))
    e.write_new(out/"identity_labels_PENDING.json", research.label_template(dataset, packet_path))
    merged = dict(facts)
    bycode = {r["code"]: r for r in roles["rows"]}
    short_bycode = {r["code"]: r for r in short_context["rows"]}
    merged["rows"] = [{**r, "leader_role_evidence": bycode[r["code"]],
                       "short_term_context": short_bycode[r["code"]]} for r in facts["rows"]]
    merged["role_dataset_path"] = str((out/"role_dataset.json").resolve())
    merged["role_dataset_sha256"] = e.sha(out/"role_dataset.json")
    merged["short_term_context_path"] = str(short_context_path.resolve())
    merged["short_term_context_sha256"] = e.sha(short_context_path)
    merged["source_bindings"] = facts["source_bindings"] | e.bind(out/"role_dataset.json") | e.bind(out/"role_facts.json") | short_context["source_bindings"] | e.bind(short_context_path) | e.bind(expectation.__file__)
    merged["prepared_at"] = e.now()
    merged["role_identity_verified_by_this_tool"] = False
    e.write_new(out/"facts_with_roles.json", merged)
    return merged


def check_role_facts(facts):
    e.verify({facts["role_dataset_path"]: facts["role_dataset_sha256"]})
    d = e.load_dataset(facts["role_dataset_path"])
    roles = e.analyse(d, "STRICT")
    bycode = {r["code"]: r for r in roles["rows"]}
    if d["rows"] and {r["signal_date"] for r in d["rows"]} != {facts["signal_date"]}:
        raise ValueError("role facts date mismatch")
    for row in facts["rows"]:
        if row["leader_role_evidence"] != bycode[row["code"]]:
            raise ValueError("role evidence projection changed")
    return roles


def freeze(facts_path, judgements_path, output):
    facts, draft = e.read(facts_path), e.read(judgements_path)
    roles = check_role_facts(facts)
    short_context = short_term.checked_projection(facts, required=True)
    short_term.validate_assessments(short_context, draft)
    expectation.validate_plans(facts, draft)
    rows = original.unique_rows(draft["rows"])
    expected = {r["code"]: r for r in facts["rows"]}
    if set(rows) != set(expected):
        raise ValueError("full original judgement cohort required")
    for code, row in rows.items():
        states = row.get("role_assessment", {})
        if set(states) != set(e.ROLES):
            raise ValueError("eleven role assessments including unknown required")
        for role, value in states.items():
            if value.get("state") not in ("CANDIDATE", "EVIDENCE_SUPPORTED", "COMPETING", "INVALIDATED", "UNKNOWN"):
                raise ValueError("unsupported role state")
            evidence = expected[code]["leader_role_evidence"]["roles"][role]
            if evidence["missing"] and value["state"] != "UNKNOWN":
                raise ValueError("missing required role facts cannot be certified")
            if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                raise ValueError("each role needs explicit evidence or unknown reason")
    # Original independent registry, score masking and all catalyst gates remain intact.
    return catalyst.freeze(facts_path, judgements_path, output)


def compare(record, baseline, outcome_archive, output):
    r = e.read(record)
    facts = e.read(r["facts_path"])
    check_role_facts(facts)
    short_context = short_term.checked_projection(facts)
    if short_context is not None:
        short_term.validate_assessments(short_context, r)
    return catalyst.compare(record, baseline, outcome_archive, output)


def confirm(record_path, confirmation_path, output):
    rec, c = e.read(record_path), e.read(confirmation_path)
    if c["signal_date"] != rec["signal_date"] or c["target_date"] != rec["target_date"]:
        raise ValueError("confirmation date mismatch")
    when = e.stamp(c["observed_at"])
    if when.astimezone(original.SHANGHAI).date().isoformat() != c["target_date"] or when > e.stamp(e.now()) or when <= e.stamp(rec["formed_at"]):
        raise ValueError("confirmation must be an actual later target observation")
    receipt = [r for r in original.registrations() if r["path"] == str(Path(record_path).resolve())]
    if len(receipt) != 1 or receipt[0]["sha256"] != e.sha(record_path):
        raise ValueError("independent registration missing or original judgement changed")
    e.verify(rec["source_bindings"])
    if e.sha(rec["facts_path"]) != rec["facts_sha256"]:
        raise ValueError("original fact record changed")
    reviewed = expectation.evaluate_confirmation(rec, c)
    result = {"status": "APPENDED_ROLE_CONFIRMATION_NOT_ORIGINAL_JUDGEMENT_REVISION", "formed_at": e.now(),
        "record_path": str(Path(record_path).resolve()), "record_sha256": e.sha(record_path),
        "confirmation": c, "expectation_review": reviewed,
        "source_bindings": c["source_bindings"] | e.bind(record_path) | e.bind(confirmation_path) | e.bind(expectation.__file__),
        "original_prediction_changed": False}
    e.write_new(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    for k in ("features", "theme-report", "mainline-report", "compatibility", "bank", "calendar", "output-dir"):
        a.add_argument("--"+k, required=True)
    a.add_argument("--role-evidence")
    a.add_argument("--history-dataset", action="append", default=[])
    a.add_argument("--mainline-history", action="append", default=[])
    a = sub.add_parser("freeze")
    for k in ("facts", "judgements", "output"): a.add_argument("--"+k, required=True)
    a = sub.add_parser("compare")
    for k in ("record", "baseline", "outcome-archive", "output"): a.add_argument("--"+k, required=True)
    a = sub.add_parser("confirm")
    for k in ("record", "confirmation", "output"): a.add_argument("--"+k, required=True)
    a = p.parse_args()
    if a.command == "prepare": r = prepare(a.features, a.theme_report, a.mainline_report, a.compatibility, a.bank, a.calendar, a.output_dir, a.role_evidence, a.history_dataset, a.mainline_history)
    elif a.command == "freeze": r = freeze(a.facts, a.judgements, a.output)
    elif a.command == "compare": r = compare(a.record, a.baseline, a.outcome_archive, a.output)
    else: r = confirm(a.record, a.confirmation, a.output)
    print(json.dumps({"status": r.get("status", "ROLE_AUGMENTED_FACTS"), "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
