"""Frozen expectation paths and source-bound observations, never trading scores.

This checks saved source fields and explicit predicates. It does not certify
provider authenticity, market causality, leader identity or executable fills.
Observations within a declared window can describe a path; they do not certify
synchronized snapshots, absorption or a funding explanation.
"""
import math
import json
from pathlib import Path

import leader_role_engine as e
import stock_judgement_review as original


VERSION = "stock-expectation-review-1"
PATHS = ("normal", "stronger", "weaker")
SCOPES = {"STOCK", "THEME", "PEERS", "MARKET"}
OPERATORS = {"EQ", "NE", "GT", "GE", "LT", "LE"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _codes(value):
    if (not isinstance(value, list) or not value
            or any(not isinstance(x, str) or len(x) != 6 or not x.isdigit() for x in value)
            or len(value) != len(set(value))):
        raise ValueError("explicit unique comparison codes required")
    return set(value)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _equal(a, b):
    # A boolean source field must not silently compare equal to an integer.
    return (type(a) is type(b) or _number(a) and _number(b)) and a == b


def _scope_registry(facts, row):
    role = row.get("leader_role_evidence", {})
    groups = role.get("source_groups", {})
    qualified = {group for value in role.get("roles", {}).values() for group in value.get("supporting_groups", [])}
    pool = groups.get("pool", {}).get("value", {}) if "pool" in qualified else {}
    theme = groups.get("theme", {}).get("value", {}) if "theme" in qualified else {}
    scope_id = role.get("scope_id")
    return {
        "STOCK": (scope_id, {row["code"]}),
        "PEERS": (scope_id, set(role.get("peer_codes", [])) if "pool" in qualified else set()),
        "THEME": (theme.get("comparison_scope_id"), set(theme.get("members", {})) - {row["code"]}),
        "MARKET": (scope_id, set(pool.get("codes", [])) - {row["code"]}),
    }


def _plans(row):
    return row.get("short_term_assessment", {}).get("confirmation_plan", [])


def _validate_plan(plan, code, target, formed, registry=None):
    if (not _text(plan.get("checkpoint_id")) or plan.get("status") != "PENDING"
            or not _text(plan.get("requirement"))):
        raise ValueError("unique checkpoint_id, PENDING and observable requirement required")
    if plan.get("baseline_kind") != "AUTHOR_PREDECLARED":
        raise ValueError("expectation is an author predeclared baseline, not verified market consensus")
    planned = e.stamp(plan["planned_at"])
    window = plan.get("observation_window", {})
    start, end = e.stamp(window.get("start_at")), e.stamp(window.get("end_at"))
    if (not start <= planned <= end or start <= formed
            or any(t.astimezone(original.SHANGHAI).date().isoformat() != target for t in (start, planned, end))):
        raise ValueError("explicit next-session observation window required")
    paths = plan.get("paths", {})
    if set(paths) != set(PATHS):
        raise ValueError("normal, stronger and weaker paths must remain separate")
    ids = set()
    for name, path in paths.items():
        if path.get("state") not in {"DEFINED", "UNKNOWN"} or not _text(path.get("reason")):
            raise ValueError("defined path or explicit abstention reason required")
        conditions = path.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError("explicit path conditions required")
        if path["state"] == "UNKNOWN":
            if conditions:
                raise ValueError("unknown path cannot contain guessed conditions")
            continue
        if not conditions:
            raise ValueError("defined path needs observable conditions")
        scopes = set()
        for condition in conditions:
            cid = condition.get("condition_id")
            if not _text(cid) or cid in ids:
                raise ValueError("unique condition_id required within checkpoint")
            ids.add(cid)
            scope = condition.get("scope")
            if (scope not in SCOPES or not _text(condition.get("metric"))
                    or not _text(condition.get("scope_id")) or not _text(condition.get("unit"))
                    or condition.get("operator") not in OPERATORS or "expected" not in condition):
                raise ValueError("explicit metric, scope, operator, expected and unit required")
            codes = _codes(condition.get("comparison_codes"))
            if scope == "STOCK" and codes != {code} or scope != "STOCK" and code in codes:
                raise ValueError("stock and external comparison ranges must remain separate")
            if registry is not None:
                scope_id, allowed = registry[scope]
                if condition["scope_id"] != scope_id or codes != allowed:
                    raise ValueError("comparison range must equal the bound original scope")
            if condition["operator"] not in {"EQ", "NE"} and not _number(condition["expected"]):
                raise ValueError("ordered condition requires finite numeric expected value")
            if condition["expected"] is None or isinstance(condition["expected"], (list, dict)):
                raise ValueError("observable expected value must be an explicit scalar")
            if isinstance(condition["expected"], float) and not math.isfinite(condition["expected"]):
                raise ValueError("nonfinite expected values prohibited")
            scopes.add(scope)
        if name == "normal" and "STOCK" not in scopes:
            raise ValueError("normal path needs an observable stock condition")
        if name in {"stronger", "weaker"} and ("STOCK" not in scopes or not scopes - {"STOCK"}):
            raise ValueError("stronger/weaker requires stock and an external comparison; high open alone is insufficient")
    return plan


def validate_plans(facts, draft):
    """New freeze requires complete plans; old records are never backfilled."""
    expected, supplied = original.unique_rows(facts["rows"]), original.unique_rows(draft["rows"])
    if set(expected) != set(supplied):
        raise ValueError("expectation review requires the complete original pool")
    formed = e.stamp(facts["prepared_at"])
    for code, row in supplied.items():
        plans = _plans(row)
        if not isinstance(plans, list) or not plans:
            raise ValueError("new freeze requires explicit pending expectation checkpoints")
        ids = set()
        for plan in plans:
            _validate_plan(plan, code, facts["target_date"], formed, _scope_registry(facts, expected[code]))
            if plan["checkpoint_id"] in ids:
                raise ValueError("duplicate checkpoint_id")
            ids.add(plan["checkpoint_id"])
    return supplied


def _pointer(document, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("exact non-root JSON pointer required")
    value = document
    try:
        for token in pointer[1:].split("/"):
            if "~" in token.replace("~1", "").replace("~0", ""):
                raise ValueError("invalid JSON pointer escape")
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not token.isdigit() or str(int(token)) != token:
                    raise ValueError("exact nonnegative JSON array index required")
                value = value[int(token)]
            else:
                value = value[token]
    except (KeyError, IndexError, ValueError, TypeError):
        raise ValueError("JSON pointer does not resolve") from None
    return value


def _source_cell(proof, sources, rec, observed_upper, current, window=None, condition=None, legacy_code=None):
    if set(proof) - {"condition_id", "source_path", "source_sha256", "pointer", "raw_value"}:
        raise ValueError("observation may contain only source proof and original condition_id")
    path = str(Path(proof["source_path"]).resolve())
    if sources.get(path) != proof.get("source_sha256"):
        raise ValueError("observation source must have a bound sha256")
    raw = _pointer(e.read(path), proof.get("pointer"))
    if (not isinstance(raw, dict)
            or json.dumps(raw, sort_keys=True, allow_nan=False) != json.dumps(proof.get("raw_value"), sort_keys=True, allow_nan=False)):
        raise ValueError("raw observation cell differs from exact source value")
    for key in ("metric", "scope_id", "unit"):
        if not _text(raw.get(key)):
            raise ValueError("source cell must bind metric, scope and unit")
    if raw.get("scope") not in SCOPES or "value" not in raw:
        raise ValueError("source cell has no observable scope/value")
    codes = _codes(raw.get("comparison_codes"))
    observed, available = e.stamp(raw["observed_at"]), e.stamp(raw["available_at"])
    if (observed.astimezone(original.SHANGHAI).date().isoformat() != rec["target_date"]
            or not e.stamp(rec["formed_at"]) < observed <= observed_upper <= current
            or not observed <= available <= current):
        raise ValueError("actual observation and availability times must be source-bound and not future")
    if window and not e.stamp(window["start_at"]) <= observed <= e.stamp(window["end_at"]):
        raise ValueError("observation outside the original checkpoint window")
    if condition is not None:
        if any(raw.get(k) != condition.get(k) for k in ("metric", "scope", "scope_id", "unit")) or codes != set(condition["comparison_codes"]):
            raise ValueError("observation changed original metric, unit or comparison range")
    elif legacy_code is not None:
        cohort = {r["code"] for r in rec["rows"]}
        if (raw["scope"] == "STOCK" and codes != {legacy_code}
                or raw["scope"] != "STOCK" and (legacy_code in codes or not codes <= cohort)):
            raise ValueError("legacy observation changed original code range")
    return raw


def _evaluate(condition, value):
    expected, op = condition["expected"], condition["operator"]
    if value is None or isinstance(value, (dict, list)) or isinstance(value, float) and not math.isfinite(value):
        return "UNKNOWN"
    if op in {"EQ", "NE"}:
        # Type mismatch is unavailable semantics, not evidence of failure.
        if type(value) is not type(expected) and not (_number(value) and _number(expected)):
            return "UNKNOWN"
        passed = _equal(value, expected)
        return "PASS" if (passed if op == "EQ" else not passed) else "FAIL"
    if not _number(value):
        return "UNKNOWN"
    passed = {"GT": value > expected, "GE": value >= expected,
              "LT": value < expected, "LE": value <= expected}[op]
    return "PASS" if passed else "FAIL"


def evaluate_confirmation(rec, confirmation, current=None):
    """Only immutable plans from rec may be evaluated; inputs contain proofs."""
    current = e.stamp(current or e.now())
    observed_upper = e.stamp(confirmation["observed_at"])
    if (confirmation["signal_date"] != rec["signal_date"] or confirmation["target_date"] != rec["target_date"]
            or observed_upper.astimezone(original.SHANGHAI).date().isoformat() != rec["target_date"]
            or not e.stamp(rec["formed_at"]) < observed_upper <= current):
        raise ValueError("confirmation requires an actual later target observation, never a future time")
    sources = confirmation.get("source_bindings")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("nonempty actual observation sources required")
    e.verify(sources)
    records, observed_rows = original.unique_rows(rec["rows"]), original.unique_rows(confirmation["rows"])
    if not set(observed_rows) <= set(records):
        raise ValueError("confirmation changed original candidate pool")
    output, proof_count, observed_codes = [], 0, set()
    for code, row in observed_rows.items():
        row_proof_start = proof_count
        if set(row) - {"code", "checkpoints", "observations"}:
            raise ValueError("confirmation cannot add or revise expectations")
        plans = _plans(records[code])
        has_new = any("checkpoint_id" in p or "paths" in p for p in plans)
        if not has_new:
            if row.get("checkpoints"):
                raise ValueError("legacy records cannot acquire retrospective expectations")
            proofs = row.get("observations", [])
            if not proofs:
                raise ValueError("legacy confirmation requires actual source observations")
            cells = [_source_cell(p, sources, rec, observed_upper, current, legacy_code=code) for p in proofs]
            proof_count += len(proofs)
            observed_codes.add(code)
            output.append({"code": code, "status": "LEGACY_OBSERVATION_ONLY_NO_PREDECLARED_EXPECTATION", "observations": cells})
            continue
        if row.get("observations"):
            raise ValueError("new expectation records require original checkpoint references")
        indexed = {}
        for plan in plans:
            _validate_plan(plan, code, rec["target_date"], e.stamp(rec["formed_at"]))
            if plan["checkpoint_id"] in indexed:
                raise ValueError("duplicate frozen checkpoint_id")
            indexed[plan["checkpoint_id"]] = plan
        checkpoints = row.get("checkpoints", [])
        seen, evaluated = set(), []
        for checkpoint in checkpoints:
            if set(checkpoint) - {"checkpoint_id", "observations"}:
                raise ValueError("confirmation cannot revise frozen checkpoint")
            cid = checkpoint.get("checkpoint_id")
            if cid not in indexed or cid in seen:
                raise ValueError("unknown or duplicate checkpoint_id")
            seen.add(cid)
            plan = indexed[cid]
            if e.stamp(plan["observation_window"]["start_at"]) > current and not checkpoint.get("observations"):
                evaluated.append({"checkpoint_id": cid, "status": "PENDING", "reason": "TARGET_CHECKPOINT_NOT_YET_OBSERVED"})
                continue
            conditions = {c["condition_id"]: c for p in plan["paths"].values() for c in p["conditions"]}
            states = {k: "UNKNOWN" for k in conditions}
            used, unscored = set(), []
            for proof in checkpoint.get("observations", []):
                condition_id = proof.get("condition_id")
                if "condition_id" not in proof:
                    # Unknown paths can still receive actual observations.
                    # They never acquire a retrospective expected predicate.
                    raw = _source_cell(proof, sources, rec, observed_upper, current,
                                       plan["observation_window"], legacy_code=code)
                    unscored.append(raw)
                    proof_count += 1
                    continue
                if condition_id not in conditions or condition_id in used:
                    raise ValueError("unknown or duplicate condition_id")
                used.add(condition_id)
                raw = _source_cell(proof, sources, rec, observed_upper, current,
                                   plan["observation_window"], conditions[condition_id])
                proof_count += 1
                states[condition_id] = _evaluate(conditions[condition_id], raw["value"])
            path_states = {}
            for name, path in plan["paths"].items():
                values = [states[c["condition_id"]] for c in path["conditions"]]
                path_states[name] = ("UNKNOWN" if path["state"] == "UNKNOWN" else
                    "FAIL" if "FAIL" in values else "UNKNOWN" if "UNKNOWN" in values else "PASS")
            passed = [k for k, v in path_states.items() if v == "PASS"]
            status = ("AMBIGUOUS" if len(passed) > 1 else
                      "UNKNOWN" if "UNKNOWN" in path_states.values() else
                      passed[0].upper() if passed else "NO_PATH_MATCH")
            evaluated.append({"checkpoint_id": cid, "status": status, "path_states": path_states,
                "condition_states": states, "missing_condition_ids": sorted(set(conditions) - used),
                "unscored_source_observations": unscored})
        for cid in indexed.keys() - seen:
            plan = indexed[cid]
            pending = e.stamp(plan["observation_window"]["start_at"]) > current
            evaluated.append({"checkpoint_id": cid, "status": "PENDING" if pending else "UNKNOWN",
                              "reason": "NO_SOURCE_OBSERVATION_APPENDED"})
        output.append({"code": code, "status": "APPENDED_EXPECTATION_OBSERVATION", "checkpoints": evaluated})
        if proof_count > row_proof_start:
            observed_codes.add(code)
    if not proof_count:
        raise ValueError("actual confirmation needs at least one source-bound observation; pending plans remain unchanged")
    unobserved = sorted(set(records) - observed_codes)
    missing_rows = []
    for code in unobserved:
        plans = _plans(records[code])
        pending = bool(plans) and all("observation_window" in p
            and e.stamp(p["observation_window"]["start_at"]) > current for p in plans)
        missing_rows.append({"code": code, "status": "PENDING" if pending else "UNKNOWN",
                             "reason": "NO_SOURCE_OBSERVATION_APPENDED"})
    return {"version": VERSION, "rows": output, "original_prediction_changed": False,
            "registered_pool_size": len(records), "observed_codes": sorted(observed_codes),
            "unobserved_codes": unobserved, "unobserved_rows": missing_rows,
            "observation_coverage": len(observed_codes)/len(records),
            "expectation_basis": "AUTHOR_PREDECLARED_NOT_VERIFIED_MARKET_CONSENSUS",
            "automatic_causality_identity_or_entry_certified": False,
            "numeric_model_scores_changed": False}
