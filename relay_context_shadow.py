"""Separate, source-bound forward ledger and fixed E1/E2 acceptance."""
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import relay_context_research as r
import model_upgrade_shadow as old


def approved_research(protocol_path):
    p = r.load_protocol(protocol_path)
    root = Path(protocol_path).parent
    result = r.read(root / "report.json")
    if result["protocol_sha256"] != r.sha(protocol_path):
        raise ValueError("context comparison protocol binding changed")
    r.verify({str(root / "full_candidate_comparison.csv"): result["predictions_sha256"]})
    survivors = [arm for arm in r.ARMS if all(c[arm]["historical_screen_passed"] for c in result["comparisons"].values())]
    if survivors != result["survivors"] or not survivors:
        raise ValueError("no historical survivor; shadow model prediction is disabled")
    return p, result, survivors


def session_calendar(root, filename):
    calendar = old.load_verified_calendar(filename)
    first = root / "calendar_initial.json"
    if first.exists():
        parent = r.read(first)
        r.verify({parent["path"]: parent["sha256"]})
        old.validate_calendar_extension(old.load_verified_calendar(parent["path"]), calendar)
    return calendar


def predict(args):
    p, result, survivors = approved_research(args.protocol)
    root = Path(args.protocol).parent / "context_shadow"
    calendar = session_calendar(root, args.calendar)
    from model_upgrade_compat import runtime
    inputs, _ = runtime(args.compatibility)
    cutoff = r.now()
    current = inputs.load_feature_archive(args.features, cutoff)
    signal, target = current["signal_date"], current["target_date"]
    old.require_current_closed_signal(signal, calendar["sessions"])
    if old.next_official_session(calendar["sessions"], signal) != target:
        raise ValueError("exact target calendar conflict")
    if current["market"]["status"] != "verified":
        raise ValueError("incomplete current market environment; no eligible shadow signal")
    if datetime.fromisoformat(cutoff) < datetime.fromisoformat(p["formed_at"]):
        raise ValueError("protocol unavailable at actual cutoff")
    prior_date = calendar["sessions"][calendar["sessions"].index(signal) - 1] if calendar["sessions"].index(signal) else None
    prior = inputs.load_feature_archive(args.previous_features, cutoff) if args.previous_features else None
    if prior and prior["signal_date"] != prior_date:
        raise ValueError("previous feature archive is not exact previous official session")
    market = [{"date": signal, "data_status": current["market"]["status"], **current["market"]["features"]}]
    if prior:
        market.append({"date": prior_date, "data_status": prior["market"]["status"], **prior["market"]["features"]})
    rows = [{"date": signal, "code": c["code"], "name": c.get("name"), "boards": c["boards"],
             "actual_one_word_daily": c.get("actual_one_word_daily"), **c["features"]} for c in current["relay"]]
    if not rows:
        raise ValueError("no candidate group; preserve daily missingness in official window instead")
    full = r.context_features(pd.DataFrame(rows), pd.DataFrame(market), calendar["sessions"])
    records = old._read_chain(root / "registrations.jsonl")
    if any(x["record"]["signal_date"] == signal for x in records):
        raise ValueError("context signal already registered")
    models = {"BASE": r.read(Path(p["base_root"]) / "models/relay/BASE/manifest.json")}
    for arm in survivors:
        models[arm] = r.read(Path(args.protocol).parent / "research_estimators/2026Sep" / arm / "manifest.json")
    predictions, bindings = {}, {}
    for arm, manifest in models.items():
        filename = manifest["model_path"]
        model_sha = manifest.get("model_sha256", manifest.get("sha256"))
        r.verify({filename: model_sha})
        if manifest["training_max_label_date"] != "2026-08-31":
            raise ValueError("frozen shadow training cutoff changed")
        if old.stamp(manifest["formed_at"]) > old.stamp(cutoff):
            raise ValueError("model unavailable at real knowledge cutoff")
        if arm != "BASE" and manifest["protocol_sha256"] != r.sha(args.protocol):
            raise ValueError("challenge artifact protocol conflict")
        estimator = joblib.load(filename)
        predicted = r.history._predict(estimator, full, manifest["fields"], "relay")
        predictions[arm] = [{"code": row.code, "name": row.get("name"), "boards": int(row.boards),
            "raw_score": float(row.raw_score) if pd.notna(row.raw_score) else None,
            "prediction": bool(int(row.predicted)) if row.status == "PREDICTED_UNCALIBRATED" else None,
            "actual_one_word_daily": float(row.actual_one_word_daily) if pd.notna(row.actual_one_word_daily) else None}
            for _, row in predicted.iterrows()]
        bindings[arm] = {"path": filename, "sha256": model_sha, "fields": manifest["fields"],
                         "training_max_label_date": manifest["training_max_label_date"]}
    if records and records[0]["record"]["model_bindings"] != bindings:
        raise ValueError("frozen shadow model hash changed; no automatic migration")
    source_hashes = {str(Path(args.protocol).resolve()): r.sha(args.protocol),
        str(Path(args.compatibility).resolve()): r.sha(args.compatibility),
        str(Path(args.features).resolve() / "manifest.json"): r.sha(Path(args.features) / "manifest.json"),
        str(Path(args.features).resolve() / "features.json"): r.sha(Path(args.features) / "features.json"),
        str(Path(args.protocol).parent.resolve() / "report.json"): r.sha(Path(args.protocol).parent / "report.json")}
    if prior:
        for name in ("manifest.json", "features.json"):
            path = Path(args.previous_features).resolve() / name
            source_hashes[str(path)] = r.sha(path)
    record = {"version": r.VERSION, "formed_at": r.now(), "knowledge_cutoff": cutoff,
        "protocol_path": str(Path(args.protocol).resolve()), "signal_date": signal, "target_date": target,
        "feature_archive": str(Path(args.features).resolve()), "previous_feature_archive": str(Path(args.previous_features).resolve()) if prior else None,
        "compatibility": str(Path(args.compatibility).resolve()), "calendar_path": str(Path(args.calendar).resolve()),
        "calendar_sha256": r.sha(args.calendar), "original_codes": sorted(full.code.tolist()),
        "predictions": predictions, "model_bindings": bindings, "source_hashes": source_hashes,
        "context_features": r.safe(full[["code"] + r.ENV + r.INTERACTIONS].to_dict("records")),
        "scope": p["scope"], "calibrated_probability": False, "production_changed": False}
    r.write_new(args.output, record)
    old._append_chain(root / "registrations.jsonl", {"record": record, "registration_path": str(Path(args.output).resolve()),
        "registration_sha256": r.sha(args.output)})
    if not (root / "calendar_initial.json").exists():
        r.write_new(root / "calendar_initial.json", {"path": str(Path(args.calendar).resolve()), "sha256": r.sha(args.calendar)})
    return {"status": "SHADOW_REGISTERED_NO_PRODUCTION_CHANGE", "new_mature_forward_samples": 0}


def settle(args):
    r.load_protocol(args.protocol)
    root = Path(args.protocol).parent / "context_shadow"
    record = r.read(args.registration)
    registrations = old._read_chain(root / "registrations.jsonl")
    matches = [x for x in registrations if x["registration_path"] == str(Path(args.registration).resolve())]
    if (len(matches) != 1 or matches[0]["registration_sha256"] != r.sha(args.registration)
            or record["protocol_path"] != str(Path(args.protocol).resolve()) or record["compatibility"] != str(Path(args.compatibility).resolve())):
        raise ValueError("immutable original shadow registration conflict")
    r.verify(record["source_hashes"])
    calendar = old.load_verified_calendar(record["calendar_path"], record["calendar_sha256"])
    old.require_closed_target(record["target_date"], calendar["sessions"])
    from model_upgrade_compat import runtime
    inputs, _ = runtime(args.compatibility)
    outcomes = inputs.build_outcomes(record, args.archive)
    r.verify(outcomes["sources"])
    if set(outcomes["candidates"]) != set(record["original_codes"]):
        raise ValueError("source-bound full outcome cohort mismatch")
    ledger = root / "settlements.jsonl"
    previous = old._read_chain(ledger)
    if any(x["registration_sha256"] == r.sha(args.registration) and x["outcomes"] == outcomes for x in previous):
        return {"status": "UNCHANGED_SETTLEMENT"}
    old._append_chain(ledger, {"registration_sha256": r.sha(args.registration), "signal_date": record["signal_date"],
        "archive_path": str(Path(args.archive).resolve()), "outcomes": outcomes})
    return {"status": "SOURCE_BOUND_SETTLEMENT_APPENDED", "new_mature_forward_samples": len(record["original_codes"])}


def paired_day_rows(records, outcomes, grid):
    arms = list(records[0]["predictions"]) if records else []
    by_day = {row["signal_date"]: row for row in records}
    result, reasons = {arm: [] for arm in arms}, {}
    for day in grid:
        record = by_day.get(day)
        truth = outcomes.get(day, {})
        if not record:
            reasons[day] = "MISSING_REGISTRATION_OR_NO_CANDIDATES"
            continue
        for arm in arms:
            for row in record["predictions"][arm]:
                result[arm].append({**row, "signal_date": day, "target_date": record["target_date"],
                                    "truth": truth.get(row["code"]), "arm": arm})
        groups = [record["predictions"][arm] for arm in arms]
        if not truth or any(type(truth.get(c)) is not bool for c in record["original_codes"]):
            reasons[day] = "UNKNOWN_TARGET_LABELS"
        elif any(type(row.get("prediction")) is not bool or row.get("raw_score") is None for group in groups for row in group):
            reasons[day] = "ABSTENTION_OR_MISSING_SCORE"
        elif any(row.get("actual_one_word_daily") not in (0., 1.) for group in groups for row in group):
            reasons[day] = "UNKNOWN_TRUE_ONEWORD_IDENTITY"
        else:
            reasons[day] = "COMPLETE_PAIRED"
    return result, reasons


def paired_accuracy_interval(base_rows, challenge_rows, grid):
    """Stock-weighted Accuracy difference, resampling official-day blocks."""
    base = {(x["signal_date"], x["code"]): x for x in base_rows}
    totals = {day: [0, 0] for day in grid}
    for row in challenge_rows:
        key = row["signal_date"], row["code"]
        b = base[key]
        if type(row.get("truth")) is bool and type(row.get("prediction")) is bool and type(b.get("prediction")) is bool:
            totals[key[0]][0] += int(row["prediction"] == row["truth"]) - int(b["prediction"] == b["truth"])
            totals[key[0]][1] += 1
    data = np.asarray([totals[day] for day in grid], dtype=float)
    if not len(data) or not data[:, 1].sum():
        return None
    rng = np.random.default_rng(r.GATE["seed"])
    draws = []
    for _ in range(r.GATE["draws"]):
        starts = rng.integers(0, len(data), size=int(np.ceil(len(data) / 5)))
        indexes = np.concatenate([(start + np.arange(5)) % len(data) for start in starts])[:len(data)]
        total = data[indexes].sum(axis=0)
        if total[1]:
            draws.append(total[0] / total[1])
    return np.quantile(draws, r.GATE["tails"]).tolist() if draws else None


def assess(rows, reasons, grid, sessions):
    complete = {day for day in grid if reasons.get(day) == "COMPLETE_PAIRED"}
    fold = {"id": "forward120", "start": grid[0], "end": grid[-1]}
    full = {arm: r.score(group, sessions, fold) for arm, group in rows.items()}
    paired = {arm: [x for x in group if x["signal_date"] in complete] for arm, group in rows.items()}
    baseline = r.score(paired["BASE"], sessions, fold)
    results = {}
    for arm in rows:
        if arm == "BASE":
            continue
        cm = r.score(paired[arm], sessions, fold)
        delta = cm["Accuracy"] - baseline["Accuracy"] if cm["Accuracy"] is not None and baseline["Accuracy"] is not None else None
        interval = paired_accuracy_interval(paired["BASE"], paired[arm], grid)
        segment_deltas = []
        for start in (0, 40, 80):
            days = set(grid[start:start + 40])
            bm = r.metrics([x for x in paired["BASE"] if x["signal_date"] in days])
            qm = r.metrics([x for x in paired[arm] if x["signal_date"] in days])
            segment_deltas.append(qm["Accuracy"] - bm["Accuracy"] if bm["Accuracy"] is not None and qm["Accuracy"] is not None else None)
        checks = r.guards(baseline, cm)
        # Unknown labels/identity stay in the full registered coverage guards.
        checks.update({"full_" + key: value for key, value in r.guards(full["BASE"], full[arm]).items()})
        checks["three40"] = all(x is not None and x >= -1e-12 for x in segment_deltas)
        checks["120_official_days"] = len(grid) == 120
        checks["60_complete_paired_days"] = len(complete) >= 60
        checks["90_percent_coverage"] = full[arm]["coverage"] is not None and full[arm]["coverage"] >= .9
        checks["accuracy_plus3pp"] = delta is not None and delta >= .03
        checks["corrected_interval_lower_positive"] = interval is not None and interval[0] > 0
        results[arm] = {"paired_metrics": cm, "Accuracy_delta": delta, "interval": interval,
                       "three40_deltas": segment_deltas, "guards": checks, "passed": all(checks.values())}
    passed = [arm for arm, value in results.items() if value["passed"]]
    winner = sorted(passed, key=lambda arm: (-results[arm]["paired_metrics"]["Accuracy"], len(r.ARMS[arm])))[0] if passed else None
    return {"full_metrics": full, "paired_BASE": baseline, "complete_paired_days": len(complete),
            "reasons_by_official_day": reasons, "challenges": results, "winner": winner}


def review(args, monitor=False):
    p = r.load_protocol(args.protocol)
    root = Path(args.protocol).parent / "context_shadow"
    records = old._read_chain(root / "registrations.jsonl")
    if not records:
        result = {"status": "WAITING_FOR_ELIGIBLE_NEW_SIGNALS", "production_changed": False, "new_mature_forward_samples": 0}
        r.write_new(args.output, result)
        return result
    calendar = session_calendar(root, args.calendar)
    originals = []
    for entry in records:
        r.verify({entry["registration_path"]: entry["registration_sha256"]})
        record = entry["record"]
        r.verify(record["source_hashes"])
        originals.append(record)
    if any(x["model_bindings"] != originals[0]["model_bindings"] for x in originals):
        raise ValueError("model hash migration inside fixed window")
    if monitor:
        return monitor_node(args, originals, calendar, root)
    grid = old.official_window_grid(calendar["sessions"], originals[0]["signal_date"], 120)
    if len(grid) < 120:
        result = {"status": "WAITING_FOR_OFFICIAL_CALENDAR_EXTENSION", "covered_window_days": len(grid), "production_changed": False}
        r.write_new(args.output, result)
        return result
    target = old.next_official_session(calendar["sessions"], grid[-1])
    try:
        old.require_closed_target(target, calendar["sessions"])
    except ValueError:
        result = {"status": "WAITING_FOR_FIXED120_MATURITY", "production_changed": False}
        r.write_new(args.output, result)
        return result
    from model_upgrade_compat import runtime
    latest = {}
    for entry in old._read_chain(root / "settlements.jsonl"):
        source = entry["outcomes"]
        r.verify(source["sources"])
        record = next(x for x in originals if x["signal_date"] == entry["signal_date"])
        inputs, _ = runtime(record["compatibility"])
        rebuilt = inputs.build_outcomes(record, entry["archive_path"])
        if rebuilt["candidates"] != source["candidates"]:
            raise ValueError("outcomes do not reproduce raw sources")
        latest[entry["signal_date"]] = source["candidates"]
    rows, reasons = paired_day_rows(originals, latest, grid)
    result = {"status": "FIXED120_REVIEW", "formed_at": r.now(), **assess(rows, reasons, grid, calendar["sessions"]),
              "protocol_sha256": r.sha(args.protocol), "production_changed": False,
              "model_bindings": originals[0]["model_bindings"],
              "application_requirement": "Verified receipt plus explicit relay-only model entry and original V298 author-review/freeze gates; never amend frozen dates"}
    r.write_new(args.output, result)
    return result


def monitor_node(args, originals, calendar, root):
    activation = r.read(args.activation)
    winner = activation.get("winner")
    if (activation.get("protocol_sha256") != r.sha(args.protocol) or winner not in r.ARMS
            or not activation.get("challenges", {}).get(winner, {}).get("passed")
            or activation.get("model_bindings") != originals[0]["model_bindings"]):
        raise ValueError("verified successful context acceptance receipt required")
    if args.node < 60 or args.node % 60:
        raise ValueError("monitor node must be a fixed multiple of 60 official signals")
    grid = old.official_window_grid(calendar["sessions"], args.first_signal, args.node, require_full=True)[-60:]
    if args.first_signal < old.stamp(activation["formed_at"]).astimezone(old.SHANGHAI).date().isoformat():
        raise ValueError("post-activation monitoring cannot include earlier signals")
    old.require_closed_target(old.next_official_session(calendar["sessions"], grid[-1]), calendar["sessions"])
    latest = {}
    from model_upgrade_compat import runtime
    for entry in old._read_chain(root / "settlements.jsonl"):
        r.verify(entry["outcomes"]["sources"])
        record = next(x for x in originals if x["signal_date"] == entry["signal_date"])
        inputs, _ = runtime(record["compatibility"])
        rebuilt = inputs.build_outcomes(record, entry["archive_path"])
        if rebuilt["candidates"] != entry["outcomes"]["candidates"]:
            raise ValueError("monitor truth/source conflict; disable new module")
        latest[entry["signal_date"]] = rebuilt["candidates"]
    rows, reasons = paired_day_rows(originals, latest, grid)
    complete = {day for day in grid if reasons.get(day) == "COMPLETE_PAIRED"}
    baseline = r.metrics([x for x in rows["BASE"] if x["signal_date"] in complete])
    current = r.metrics([x for x in rows[winner] if x["signal_date"] in complete])
    accuracy_drop = baseline["Accuracy"] - current["Accuracy"] if baseline["Accuracy"] is not None and current["Accuracy"] is not None else None
    recall_drop = baseline["complete_positive_Recall"] - current["complete_positive_Recall"] if baseline["complete_positive_Recall"] is not None and current["complete_positive_Recall"] is not None else None
    required = (accuracy_drop is not None and accuracy_drop >= .05) or (recall_drop is not None and recall_drop >= .10)
    result = {"status": "ROLLBACK_REQUIRED" if required else "MONITOR_NO_ROLLBACK_TRIGGER",
        "formed_at": r.now(), "node": args.node, "window": grid, "BASE": baseline, "active": current,
        "Accuracy_drop": accuracy_drop, "Recall_drop": recall_drop, "rollback_required": required,
        "reasons_by_day": reasons, "production_changed": False,
        "scope": "Decision receipt for explicit relay-only production controller; no automatic parameter recovery"}
    r.write_new(args.output, result)
    return result


def run_command(args):
    if args.command == "predict":
        return predict(args)
    if args.command == "settle":
        return settle(args)
    return review(args, monitor=args.command == "monitor")
