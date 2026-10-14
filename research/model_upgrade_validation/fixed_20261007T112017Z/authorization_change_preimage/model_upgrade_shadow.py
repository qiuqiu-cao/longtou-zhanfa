"""Source-bound prospective model checks, independent of production V298.

The command interface never accepts a simulated clock. Synthetic tests may pass
an explicit clock to the pure validation helpers; production uses now_utc().
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from copy import deepcopy
import argparse
import hashlib
from importlib import metadata
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")


def now_utc():
    return datetime.now(timezone.utc)


def stamp(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid ISO timestamp") from exc
    else:
        raise ValueError("timestamp must include a timezone")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return result


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_sessions(sessions):
    if not isinstance(sessions, list) or not sessions or sessions != sorted(set(sessions)):
        raise ValueError("official sessions must be nonempty, unique and increasing")
    for value in sessions:
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid official session date") from exc
        if parsed.isoformat() != value:
            raise ValueError("official session date must use YYYY-MM-DD")
    return sessions


def load_verified_calendar(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("calendar hash mismatch")
    calendar = json.loads(path.read_text(encoding="utf-8"))
    if calendar.get("status") != "verified" or not calendar.get("source"):
        raise ValueError("official source-bound verified calendar required")
    validate_sessions(calendar.get("sessions"))
    scope = calendar.get("scope")
    if (not isinstance(scope, list) or len(scope) != 2
            or not scope[0] <= calendar["sessions"][0]
            or not calendar["sessions"][-1] <= scope[1]):
        raise ValueError("calendar coverage does not contain its official grid")
    return calendar


def next_official_session(sessions, signal):
    validate_sessions(sessions)
    if signal not in sessions:
        raise ValueError("signal is not an official trading session")
    pos = sessions.index(signal)
    if pos + 1 == len(sessions):
        raise ValueError("official calendar must be extended before assigning target")
    return sessions[pos + 1]


def require_current_closed_signal(signal, sessions, current=None):
    current = stamp(current if current is not None else now_utc()).astimezone(SHANGHAI)
    next_official_session(sessions, signal)
    if current.date().isoformat() != signal:
        raise ValueError("forward signal must be the true current Shanghai date")
    if current.time().replace(tzinfo=None) < time(15, 0):
        raise ValueError("signal session has not closed")
    return current


def require_closed_target(target, sessions, current=None):
    validate_sessions(sessions)
    if target not in sessions:
        raise ValueError("target is not an official trading session")
    current = stamp(current if current is not None else now_utc())
    close = datetime.combine(date.fromisoformat(target), time(15), SHANGHAI)
    if current < close:
        raise ValueError("target session has not closed")
    return current


def validate_cutoff(signal, cutoff, current=None):
    current = stamp(current if current is not None else now_utc())
    cutoff = stamp(cutoff)
    local = cutoff.astimezone(SHANGHAI)
    if local.date().isoformat() != signal or local.time().replace(tzinfo=None) < time(15):
        raise ValueError("knowledge cutoff must be on the closed signal session")
    if cutoff > current:
        raise ValueError("knowledge cutoff is in the future")
    return cutoff


def official_window_grid(sessions, first_signal, length=120, require_full=False):
    validate_sessions(sessions)
    if first_signal not in sessions:
        raise ValueError("first eligible signal is absent from official calendar")
    if not isinstance(length, int) or length <= 0:
        raise ValueError("invalid official window length")
    grid = sessions[sessions.index(first_signal):][:length]
    if require_full and len(grid) != length:
        raise ValueError("official calendar extension required for complete fixed node")
    return grid


def validate_calendar_extension(parent, extended):
    old = validate_sessions(parent["sessions"])
    new = validate_sessions(extended["sessions"])
    if new[:len(old)] != old:
        raise ValueError("calendar extension cannot change the original official prefix")
    if extended.get("status") != "verified" or not extended.get("source"):
        raise ValueError("calendar extension lacks official source verification")
    return extended


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value).decode("utf-8") + "\n")
    return path


def _read_chain(path):
    path = Path(path)
    if not path.exists():
        return []
    rows, previous = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("empty line in append-only audit chain")
        row = json.loads(line)
        entry_hash = row.pop("entry_hash", None)
        if row.get("previous_hash") != previous or hashlib.sha256(canonical(row)).hexdigest() != entry_hash:
            raise ValueError("append-only chain binding conflict")
        row["entry_hash"] = entry_hash
        rows.append(row)
        previous = entry_hash
    return rows


def _append_chain(path, event):
    path = Path(path)
    rows = _read_chain(path)
    event = {**deepcopy(event), "recorded_at": now_utc().isoformat(),
             "previous_hash": rows[-1]["entry_hash"] if rows else None}
    event["entry_hash"] = hashlib.sha256(canonical(event)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(event).decode("utf-8") + "\n")
    return event


def _source_check(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("nonempty source hash bindings required")
    for name, expected in sources.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError("source hash binding conflict: " + str(path))


def _protocol(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("version") != "model-upgrade-fixed-1":
        raise ValueError("unsupported frozen upgrade protocol")
    if stamp(protocol["recorded_at"]) > now_utc():
        raise ValueError("protocol formation is in the future")
    implementation_path = path.parent / "implementation_BEFORE_FITS.json"
    if not implementation_path.is_file():
        raise ValueError("pre-fit complete implementation hash binding missing")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    if implementation.get("protocol_sha256") != file_sha256(path):
        raise ValueError("pre-fit implementation protocol hash conflict")
    original_implementation = implementation
    correction_paths = [path.parent / "implementation_correction_AFTER_FITS.json",
                        path.parent / "implementation_correction_BEFORE_FORWARD.json"]
    if correction_paths[1].exists() and not correction_paths[0].is_file():
        raise ValueError("pre-forward correction requires its original post-fit correction chain")
    prior_formed = None
    for layer, correction_path in enumerate(correction_paths):
      if correction_path.exists():
        correction = json.loads(correction_path.read_text(encoding="utf-8"))
        if (correction.get("protocol_sha256") != file_sha256(path)
                or correction.get("original_implementation_sha256") != file_sha256(implementation_path)):
            raise ValueError("post-fit correction original receipt/protocol hash conflict")
        formed = stamp(correction["formed_at"])
        if formed < stamp(protocol["recorded_at"]) or formed > now_utc():
            raise ValueError("post-fit correction formation outside true time boundary")
        if layer == 1:
            if correction.get("prior_correction_sha256") != file_sha256(correction_paths[0]):
                raise ValueError("pre-forward prior correction hash binding conflict")
            if formed < prior_formed:
                raise ValueError("pre-forward correction formed before its original correction")
            registrations = [e for e in _read_chain(path.parent / "shadow" / "registrations.jsonl")
                             if e.get("event") == "REGISTRATION"]
            if registrations and formed > min(stamp(e["recorded_at"]) for e in registrations):
                raise ValueError("pre-forward correction cannot change an already started real registration window")
        if (type(correction.get("numeric_refits")) is not int or correction["numeric_refits"] != 0
                or correction.get("feature_formula_changed") is not False
                or correction.get("parameter_changed") is not False):
            raise ValueError("post-fit correction cannot change numeric fits, formulas or parameters")
        originals = original_implementation.get("sources", {})
        snapshots = correction.get("source_snapshots", {})
        corrected = correction.get("corrected_sources", {})
        expected_names = {"model_upgrade_research.py", "model_upgrade_history.py", "model_upgrade_inputs.py",
                          "model_upgrade_shadow.py", "model_upgrade_metrics.py", "model_upgrade_apply.py"}
        if (len(originals) != 6 or {Path(p).name for p in originals} != expected_names
                or set(snapshots) != set(originals) or set(corrected) != set(originals)):
            raise ValueError("post-fit correction must preserve the exact six-file implementation group")
        preservation = (path.parent / "used_sources_preserved_after_fit").resolve()
        allowed_changes = {"model_upgrade_research.py", "model_upgrade_inputs.py", "model_upgrade_shadow.py"}
        if layer == 1:
            allowed_changes.add("model_upgrade_metrics.py")
        for filename, original_hash in originals.items():
            original_path = Path(filename)
            if not original_path.is_absolute() or original_path.resolve().parent != Path(__file__).resolve().parent:
                raise ValueError("post-fit correction source path outside the original implementation")
            snapshot = snapshots[filename]
            saved_path = Path(snapshot["path"]).resolve()
            try:
                saved_path.relative_to(preservation)
            except ValueError as exc:
                raise ValueError("post-fit original source snapshot outside preservation directory") from exc
            if snapshot.get("sha256") != original_hash or file_sha256(saved_path) != original_hash:
                raise ValueError("post-fit original source snapshot hash conflict")
            if original_path.name not in allowed_changes and corrected[filename] != original_hash:
                raise ValueError("post-fit correction cannot alter history, metrics or apply implementation")
            if (not isinstance(corrected[filename], str) or len(corrected[filename]) != 64
                    or any(c not in "0123456789abcdefABCDEF" for c in corrected[filename])):
                raise ValueError("correction source hash is not a SHA256 binding")
        implementation = {**implementation, "sources": corrected}
        prior_formed = formed
    if any(p.exists() for p in correction_paths):
        _source_check(implementation["sources"])
    provenance_path = Path(protocol.get("provenance_path", path.parent / "provenance_BEFORE_FITS.json"))
    if not provenance_path.is_absolute():
        provenance_path = path.parent / provenance_path
    provenance_hash = protocol.get("provenance_sha256", implementation.get("provenance_sha256"))
    if provenance_hash is None or file_sha256(provenance_path) != provenance_hash:
        raise ValueError("pre-fit provenance receipt hash conflict")
    provenance_receipt = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance_receipt.get("protocol_sha256", file_sha256(path)) != file_sha256(path):
        raise ValueError("pre-fit provenance protocol hash conflict")
    provenance = provenance_receipt.get("provenance", provenance_receipt)
    bound = {}
    for container in (protocol, implementation):
      for key in ("input_hashes", "dependency_hashes", "source_hashes", "sources", "code_sha256", "protected_sha256"):
        if container.get(key):
            mapping = {}
            for p, h in container[key].items():
                candidates = [Path(p)] if Path(p).is_absolute() else [path.parent / p, Path(__file__).parent / p]
                located = next((v.resolve() for v in candidates if v.is_file()), candidates[0].resolve())
                mapping[str(located)] = h
            _source_check(mapping)
            bound.update(mapping)
    for name in ("model_upgrade_shadow.py", "model_upgrade_metrics.py", "model_upgrade_inputs.py"):
        current_code = Path(__file__).with_name(name).resolve()
        if bound.get(str(current_code)) != file_sha256(current_code):
            raise ValueError("current controller/metrics/input implementation hash binding conflict")
    for key in ("source_hashes", "dependency_hashes"):
        for p, h in provenance.get(key, {}).items():
            if bound.get(str(Path(p).resolve())) != h:
                raise ValueError("pre-fit provenance source/dependency not fully hash bound")
    calendar = load_verified_calendar(protocol["calendar_path"], protocol["calendar_sha256"])
    root = path.parent / "shadow"
    calendar = {**calendar, "_path": str(Path(protocol["calendar_path"]).resolve()),
                "_sha256": protocol["calendar_sha256"]}
    for entry in _read_chain(root / "calendar_extensions.jsonl"):
        extended = load_verified_calendar(entry["path"], entry["sha256"])
        validate_calendar_extension(calendar, extended)
        calendar = {**extended, "_path": entry["path"], "_sha256": entry["sha256"]}
    return path, protocol, calendar, root


def _owned(path, root):
    path = Path(path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("output must stay inside this protocol's dedicated shadow directory") from exc
    return path


def _models(path, protocol, module, models_dir, cutoff):
    config = protocol["modules"][module]
    if set(config["arms"]) != {"BASE", "C1", "C2"}:
        raise ValueError("the complete fixed BASE/C1/C2 group is required")
    result = {}
    for arm in ("BASE", "C1", "C2"):
        relative = Path(config["model_manifests"][arm])
        choices = [relative] if relative.is_absolute() else [path.parent / relative, Path(models_dir) / relative]
        if not relative.is_absolute():
            choices.append(Path(models_dir) / module / arm / "manifest.json")
        manifest_path = next((p.resolve() for p in choices if p.is_file()), None)
        if manifest_path is None:
            raise ValueError("frozen model manifest missing: " + arm)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["module"] != module or manifest["arm"] != arm or manifest["protocol_sha256"] != file_sha256(path):
            raise ValueError("model manifest belongs to another frozen protocol")
        formed = stamp(manifest["formed_at"])
        if formed < stamp(protocol["recorded_at"]) or formed > cutoff:
            raise ValueError("model formation outside actual knowledge boundary")
        frozen = stamp(manifest["frozen_at"])
        if frozen < formed or frozen > cutoff:
            raise ValueError("model freeze outside actual knowledge boundary")
        expected_classes = {"IMPROVE", "CONTINUE", "WORSEN"} if module == "market" else {"0", "1"}
        if set(manifest.get("classes", [])) != expected_classes:
            raise ValueError("frozen model class definition differs from approved target")
        versions = manifest.get("runtime_versions", {})
        if not {"numpy", "pandas", "scikit-learn", "joblib"} <= set(versions):
            raise ValueError("frozen model runtime dependency bindings incomplete")
        if manifest.get("python_version", sys.version) != sys.version:
            raise ValueError("frozen Python runtime version binding conflict")
        for package, expected in versions.items():
            try:
                installed = metadata.version(package)
            except metadata.PackageNotFoundError as exc:
                raise ValueError("frozen runtime dependency missing: " + package) from exc
            if installed != expected:
                raise ValueError("frozen runtime version binding conflict: " + package)
        if manifest["training_max_label_date"] >= protocol["freeze_cutoff"]:
            raise ValueError("frozen training labels cross the original cutoff")
        fields = manifest["fields"]
        if (not fields or len(fields) != len(set(fields)) or any(k in {"target", "target_date", "label_valid", "correct"}
                                                                 or k.startswith("next_") for k in fields)):
            raise ValueError("model field definition is duplicate or contains future truth")
        _source_check(manifest.get("source_hashes", manifest.get("sources", {})))
        if manifest.get("code_sha256"):
            _source_check(manifest["code_sha256"])
        model_path = Path(manifest["model_path"])
        if not model_path.is_absolute():
            model_path = manifest_path.parent / model_path
        model_hash = manifest.get("model_sha256", manifest.get("sha256"))
        if not model_path.is_file() or file_sha256(model_path) != model_hash:
            raise ValueError("frozen model hash binding conflict")
        result[arm] = {"path": str(manifest_path), "sha256": file_sha256(manifest_path),
                       "model_path": str(model_path.resolve()), "model_sha256": model_hash,
                       "manifest": manifest}
    return result


def _model_probabilities(binding, vectors):
    manifest = binding["manifest"]
    if any(set(manifest["fields"]) - set(v) for v in vectors):
        raise ValueError("feature projection misses frozen model fields")
    frame = pd.DataFrame(vectors)[manifest["fields"]].astype(float)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("infinite observed features")
    model = joblib.load(binding["model_path"])
    classes = [str(c) for c in model.classes_]
    if classes != [str(c) for c in manifest["classes"]]:
        raise ValueError("model class order differs from frozen manifest")
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if (probabilities.shape != (len(vectors), len(classes)) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)):
        raise ValueError("invalid raw model class scores")
    return classes, probabilities


def _predict_impl(protocol_path, module, features_path, models_dir, output):
    """Register today's complete unlabelled three-arm prediction; never fit."""
    from model_upgrade_inputs import load_feature_archive
    path, protocol, calendar, root = _protocol(protocol_path)
    if module not in protocol["modules"]:
        raise ValueError("unregistered upgrade module")
    current = now_utc()
    cutoff = current
    feature_pack = load_feature_archive(features_path, cutoff=cutoff.isoformat())
    signal = feature_pack["signal_date"]
    require_current_closed_signal(signal, calendar["sessions"], current)
    validate_cutoff(signal, cutoff, current)
    if stamp(feature_pack["available_at"]) > cutoff:
        raise ValueError("complete feature archive formed after knowledge cutoff")
    _source_check(feature_pack["sources"])
    models = _models(path, protocol, module, models_dir, cutoff)
    roster = feature_pack["relay"]
    codes = [str(r["code"]) for r in roster]
    if len(codes) != len(set(codes)) or any(len(c) != 6 or not c.isdigit() for c in codes):
        raise ValueError("complete candidate roster has duplicate or invalid securities codes")
    if any(not isinstance(r["boards"], int) or isinstance(r["boards"], bool) or r["boards"] < 2
           or r["actual_one_word_daily"] not in (0, 1, None) for r in roster):
        raise ValueError("candidate boards or observed one-word status invalid")
    index = root / "registrations.jsonl"
    if any(e.get("module") == module and e.get("signal_date") == signal for e in _read_chain(index)):
        raise ValueError("this module/signal is already frozen")
    output = _owned(output, root)
    if output.suffix != ".json":
        output = output / "registration.json"
    window_path = root / module / "window.json"
    if window_path.exists():
        window = json.loads(window_path.read_text(encoding="utf-8"))
        if window["protocol_sha256"] != file_sha256(path):
            raise ValueError("shadow window protocol hash conflict")
        if signal < window["first_signal_date"]:
            raise ValueError("cannot backfill dates before first eligible signal")
    else:
        window = {"module": module, "first_signal_date": signal,
                  "freeze_start": min(stamp(b["manifest"]["formed_at"]) for b in models.values()).isoformat(),
                  "protocol_sha256": file_sha256(path), "registered_at": current.isoformat()}
    arm_rows = {}
    suspended = _pointer(root)["modules"].get(module, {}).get("suspended", False)
    for arm, binding in models.items():
        halted = suspended and arm != "BASE"
        if module == "market":
            vector = feature_pack["market"]["features"]
            observed = not halted and any(vector.get(k) is not None for k in binding["manifest"]["fields"])
            if not observed:
                arm_rows[arm] = {"status": "UNKNOWN", "prediction": None, "raw_scores": {},
                                 "reason": "CHALLENGE_SUSPENDED" if halted else "ALL_OBSERVED_FEATURES_MISSING"}
                continue
            classes, raw = _model_probabilities(binding, [vector])
            arm_rows[arm] = {"status": "KNOWN", "prediction": classes[int(raw[0].argmax())],
                             "raw_scores": dict(zip(classes, raw[0].tolist()))}
        elif module == "relay":
            complete = not halted and bool(roster)
            vectors = [r["features"] for r in roster]
            observed = all(any(v.get(k) is not None for k in binding["manifest"]["fields"]) for v in vectors)
            raw = None
            if complete and observed:
                classes, raw = _model_probabilities(binding, vectors)
                if "1" not in classes:
                    raise ValueError("relay frozen class order lacks positive class")
                positive = classes.index("1")
            candidates = [{"code": r["code"], "name": r.get("name"), "boards": r["boards"],
                           "actual_one_word_daily": r["actual_one_word_daily"],
                           "raw_score": float(raw[i, positive]) if raw is not None else None,
                           "predicted_positive": bool(raw[i, positive] > .5) if raw is not None else None}
                          for i, r in enumerate(roster)]
            arm_rows[arm] = {"status": "KNOWN" if raw is not None else "UNKNOWN", "candidates": candidates,
                             "primary_eligible": (any(r["actual_one_word_daily"] == 0 for r in roster)
                                                  if raw is not None and all(r["actual_one_word_daily"] is not None for r in roster)
                                                  else None),
                             "ranking_rule": "raw_score descending; code ascending; no replacement",
                             "reason": None if raw is not None else "INCOMPLETE_ROSTER_OR_ONE_WORD_OR_FEATURE_EVIDENCE"}
        else:
            raise ValueError("unsupported target module")
    highest = None
    if roster and all(r["actual_one_word_daily"] is not None for r in roster):
        nonword = [r for r in roster if r["actual_one_word_daily"] == 0]
        if nonword:
            highest = sorted(nonword, key=lambda r: (-r["boards"], r["code"]))[0]["code"]
    record = {"signal_date": signal, "target_date": next_official_session(calendar["sessions"], signal),
              "module": module, "arms": arm_rows, "original_codes": sorted(codes),
              "naive": {"majority": (models["BASE"]["manifest"]["training_majority"].get("class")
                                       if isinstance(models["BASE"]["manifest"]["training_majority"], dict)
                                       else models["BASE"]["manifest"]["training_majority"]),
                        "persistence": feature_pack["market"].get("persistence"), "highest_nonword": highest},
              "forward": True, "knowledge_cutoff": cutoff.isoformat(), "registered_at": current.isoformat(),
              "protocol_path": str(path), "protocol_sha256": file_sha256(path),
              "implementation_sha256": file_sha256(path.parent / "implementation_BEFORE_FITS.json"),
              "feature_archive": str(Path(features_path).resolve()),
              "feature_projection_sha256": hashlib.sha256(canonical(feature_pack)).hexdigest(),
              "feature_sources": feature_pack["sources"], "model_bindings": models,
              "calendar_path": calendar["_path"], "calendar_sha256": calendar["_sha256"],
              "shadow_code_sha256": file_sha256(__file__), "score_calibrated": False,
              "production_model_changed": False}
    if "outcomes" in record:
        raise AssertionError("forward registration cannot contain outcomes")
    _write_new(output, record)
    eligible = all(row["status"] == "KNOWN" for row in arm_rows.values())
    if module == "market":
        eligible = eligible and feature_pack["market"]["status"] == "verified"
    else:
        eligible = (eligible and all(r["actual_one_word_daily"] is not None for r in roster)
                    and any(r["actual_one_word_daily"] == 0 for r in roster))
    if not window_path.exists() and eligible:
        _write_new(window_path, window)
    _append_chain(index, {"event": "REGISTRATION", "module": module, "signal_date": signal,
                          "path": str(output), "sha256": file_sha256(output),
                          "window_sha256": file_sha256(window_path) if window_path.exists() else None,
                          "first_window_eligible": eligible})
    return record


def _registered(registration):
    from model_upgrade_inputs import load_feature_archive
    registration = Path(registration).resolve()
    record = json.loads(registration.read_text(encoding="utf-8"))
    path, protocol, calendar, root = _protocol(record["protocol_path"])
    entries = [e for e in _read_chain(root / "registrations.jsonl") if e.get("path") == str(registration)]
    if len(entries) != 1 or entries[0]["sha256"] != file_sha256(registration):
        raise ValueError("registration missing from original append-only index or changed")
    if not record.get("forward") or record["protocol_sha256"] != file_sha256(path):
        raise ValueError("registration is not the frozen forward protocol")
    if record["shadow_code_sha256"] != file_sha256(__file__):
        raise ValueError("registration controller code hash conflict")
    if record["implementation_sha256"] != file_sha256(path.parent / "implementation_BEFORE_FITS.json"):
        raise ValueError("original pre-fit implementation receipt hash conflict")
    if _models(path, protocol, record["module"], path.parent / "models", stamp(record["knowledge_cutoff"])) != record["model_bindings"]:
        raise ValueError("original model definition hash binding conflict")
    _source_check(record["feature_sources"])
    _source_check({b["path"]: b["sha256"] for b in record["model_bindings"].values()})
    _source_check({b["model_path"]: b["model_sha256"] for b in record["model_bindings"].values()})
    for binding in record["model_bindings"].values():
        _source_check(binding["manifest"].get("source_hashes", binding["manifest"].get("sources", {})))
        if binding["manifest"].get("code_sha256"):
            _source_check(binding["manifest"]["code_sha256"])
    pack = load_feature_archive(record["feature_archive"], cutoff=record["knowledge_cutoff"])
    if hashlib.sha256(canonical(pack)).hexdigest() != record["feature_projection_sha256"]:
        raise ValueError("original feature reconstruction differs from registration")
    if record["target_date"] != next_official_session(calendar["sessions"], record["signal_date"]):
        raise ValueError("registration target differs from official successor")
    return record, path, protocol, calendar, root


def _settle_impl(registration, source_archive):
    """Derive outcomes from immutable raw-source adapters, never supplied booleans."""
    from model_upgrade_inputs import build_outcomes
    record, path, protocol, calendar, root = _registered(registration)
    require_closed_target(record["target_date"], calendar["sessions"])
    outcomes = build_outcomes(record, source_archive)
    _source_check(outcomes["sources"])
    supplied = outcomes["candidates"]
    if set(supplied) - set(record["original_codes"]):
        raise ValueError("outcome candidate range differs from original registration")
    values = {c: supplied.get(c) for c in record["original_codes"]}
    if any(v is not None and not isinstance(v, bool) for v in values.values()):
        raise ValueError("outcome bridge returned unsupported candidate truth")
    updated = deepcopy(record)
    updated["outcomes"] = {"market_target": outcomes["market_target"], "candidates": values}
    updated["target_matured"] = True
    updated["candidate_group_complete"] = bool(values) and all(v is not None for v in values.values())
    updated["outcome_sources"] = outcomes["sources"]
    updated["outcome_archive"] = str(Path(source_archive).resolve())
    if outcomes["market_target"] not in {None, "IMPROVE", "CONTINUE", "WORSEN"}:
        raise ValueError("outcome bridge returned unsupported market truth")
    ledger = root / "settlements.jsonl"
    revisions = [r for r in _read_chain(ledger) if r.get("module") == record["module"]
                 and r.get("signal_date") == record["signal_date"]]
    if revisions and revisions[-1]["record"] == updated:
        return revisions[-1]
    return _append_chain(ledger, {"event": "SETTLEMENT_REVISION", "module": record["module"],
                                  "signal_date": record["signal_date"], "revision": len(revisions) + 1,
                                  "registration_sha256": file_sha256(registration), "record": updated})


def bind_calendar(protocol_path, calendar_path):
    """Append a separately verified official extension, keeping the old calendar."""
    path, protocol, calendar, root = _protocol(protocol_path)
    new_path = Path(calendar_path).resolve()
    extended = load_verified_calendar(new_path)
    validate_calendar_extension(calendar, extended)
    if len(extended["sessions"]) <= len(calendar["sessions"]):
        raise ValueError("calendar extension must add official sessions")
    return _append_chain(root / "calendar_extensions.jsonl", {
        "event": "OFFICIAL_CALENDAR_EXTENSION", "path": str(new_path),
        "sha256": file_sha256(new_path), "parent_sha256": calendar["_sha256"],
        "protocol_sha256": file_sha256(path)})


def _pointer(root):
    file = root / "active_pointer.json"
    if not file.exists():
        return {"modules": {}}
    pointer = json.loads(file.read_text(encoding="utf-8"))
    chain = _read_chain(root / "control_audit.jsonl")
    if not chain or chain[-1]["new_pointer"] != pointer:
        raise ValueError("active pointer append-only audit hash binding conflict")
    return pointer


def _save_pointer(root, pointer, event):
    # This file belongs to the research protocol. Production V298 never reads it.
    old = _pointer(root)
    audit = _append_chain(root / "control_audit.jsonl", {
        "event": event, "old_pointer": old, "new_pointer": pointer})
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / ("active_pointer." + audit["entry_hash"] + ".tmp")
    _write_new(temporary, pointer)
    temporary.replace(root / "active_pointer.json")
    return audit


def rollback(protocol_path, module, reason):
    """Suspend the challenge in the dedicated shadow pointer; retain artifacts."""
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("version") != "model-upgrade-fixed-1" or module not in protocol["modules"]:
        raise ValueError("unknown frozen module")
    root = path.parent / "shadow"
    pointer = _pointer(root)
    previous = pointer["modules"].get(module, {})
    pointer["modules"][module] = {**previous, "active_arm": "BASE", "suspended": True,
                                   "suspended_at": now_utc().isoformat(), "reason": str(reason),
                                   "protocol_path": str(path), "protocol_sha256": file_sha256(path),
                                   "production_model_changed": False}
    return _save_pointer(root, pointer, "ROLLBACK_TO_BASE")


def _binding_failure(protocol_path, module, error):
    waiting = ("session has not closed", "calendar must be extended", "calendar extension required",
               "true current shanghai date", "already frozen", "no genuine forward registration",
               "was not effective at original signal")
    if not any(token in str(error).lower() for token in waiting):
        rollback(protocol_path, module, "SOURCE_OR_ARTIFACT_BINDING_CONFLICT: " + str(error))


def predict(protocol_path, module, features_path, models_dir, output):
    try:
        record = _predict_impl(protocol_path, module, features_path, models_dir, output)
    except (ValueError, OSError) as exc:
        _binding_failure(protocol_path, module, exc)
        raise
    return record


def settle(registration, source_archive):
    record = json.loads(Path(registration).read_text(encoding="utf-8"))
    try:
        return _settle_impl(registration, source_archive)
    except (ValueError, OSError) as exc:
        _binding_failure(record["protocol_path"], record["module"], exc)
        raise


def _closed_through(sessions):
    local = now_utc().astimezone(SHANGHAI)
    return max((d for d in sessions if d < local.date().isoformat()
                or (d == local.date().isoformat() and local.time().replace(tzinfo=None) >= time(15))),
               default=None)


def _review_inputs(protocol_path, module):
    from model_upgrade_inputs import build_outcomes
    path, protocol, calendar, root = _protocol(protocol_path)
    window_path = root / module / "window.json"
    if not window_path.exists():
        raise ValueError("no genuine forward registration starts this module's window")
    window = json.loads(window_path.read_text(encoding="utf-8"))
    if window["protocol_sha256"] != file_sha256(path):
        raise ValueError("fixed forward window protocol hash conflict")
    registrations = _read_chain(root / "registrations.jsonl")
    group = [e for e in registrations if e["module"] == module and e.get("window_sha256") is not None]
    if (not group or group[0]["window_sha256"] != file_sha256(window_path)
            or window["first_signal_date"] != group[0]["signal_date"]):
        raise ValueError("original first eligible official window hash conflict")
    revisions = _read_chain(root / "settlements.jsonl")
    for revision in revisions:
        _source_check(revision["record"]["outcome_sources"])
    latest = {(e["module"], e["signal_date"]): e for e in revisions}
    records, bindings = [], {str(path): file_sha256(path), str(window_path): file_sha256(window_path),
                             calendar["_path"]: calendar["_sha256"], str(Path(__file__).resolve()): file_sha256(__file__)}
    for index in registrations:
        if index["module"] != module:
            continue
        record, _, _, _, _ = _registered(index["path"])
        revision = latest.get((module, record["signal_date"]))
        if revision:
            if revision["registration_sha256"] != file_sha256(index["path"]):
                raise ValueError("settlement original registration hash conflict")
            updated = revision["record"]
            _source_check(updated["outcome_sources"])
            rebuilt = build_outcomes(record, updated["outcome_archive"])
            expected = {"market_target": rebuilt["market_target"],
                        "candidates": {c: rebuilt["candidates"].get(c) for c in record["original_codes"]}}
            if expected != updated["outcomes"] or rebuilt["sources"] != updated["outcome_sources"]:
                raise ValueError("settlement source reconstruction differs from bound revision")
            if any(updated.get(k) != v for k, v in record.items()):
                raise ValueError("settlement changed original registration fields")
            record = deepcopy(updated)
            bindings.update(updated["outcome_sources"])
        else:
            record = deepcopy(record)
            record["outcomes"] = {"market_target": None,
                                  "candidates": {c: None for c in record["original_codes"]}}
        closed = _closed_through(calendar["sessions"])
        record["target_matured"] = closed is not None and record["target_date"] <= closed
        if record["signal_date"] >= window["first_signal_date"]:
            records.append(record)
        bindings[index["path"]] = index["sha256"]
        bindings.update(record["feature_sources"])
        for binding in record["model_bindings"].values():
            bindings[binding["path"]] = binding["sha256"]
            bindings[binding["model_path"]] = binding["model_sha256"]
    for ledger in (root / "registrations.jsonl", root / "settlements.jsonl", root / "calendar_extensions.jsonl"):
        if ledger.exists():
            bindings[str(ledger)] = file_sha256(ledger)
    implementation_path = path.parent / "implementation_BEFORE_FITS.json"
    bindings[str(implementation_path)] = file_sha256(implementation_path)
    derived = {**deepcopy(protocol), "forward_start_date": window["first_signal_date"],
               "first_signal_date": window["first_signal_date"],
               "closed_through_date": _closed_through(calendar["sessions"])}
    return path, derived, calendar, root, records, bindings


def review_node(protocol_path, module, node=120, output=None):
    """Check one fixed mature node using rederived truths and the real clock."""
    from model_upgrade_metrics import acceptance, rollback_check
    if node not in (60, 120):
        raise ValueError("only the preregistered 60 and 120 official-session nodes exist")
    try:
        path, derived, calendar, root, records, bindings = _review_inputs(protocol_path, module)
        if node == 120:
            result = acceptance(module, records, derived, calendar["sessions"])
        else:
            active = _pointer(root)["modules"].get(module, {})
            derived["active_arm"] = active.get("active_arm", "BASE")
            anchor = active.get("effective_signal", derived["first_signal_date"])
            tail = calendar["sessions"][calendar["sessions"].index(anchor):]
            mature_count = sum(d <= (derived["closed_through_date"] or "") for d in tail[1:])
            nodes_path = root / module / "rollback_nodes.jsonl"
            tested = [e for e in _read_chain(nodes_path) if e["anchor"] == anchor
                      and e["active_arm"] == derived["active_arm"]]
            tested_numbers = {e["node_index"] for e in tested}
            pending = [i for i in range(1, mature_count // 60 + 1) if i not in tested_numbers]
            node_index = pending[0] if pending else None
            if node_index is not None:
                grid = tail[(node_index - 1) * 60:node_index * 60]
                derived["rollback_window_start"] = grid[0]
                result = rollback_check(module, [r for r in records if r["signal_date"] in grid], derived,
                                        calendar["sessions"])
            else:
                result = {"status": "ALREADY_CHECKED" if tested else "NOT_MATURE", "rollback": False,
                          "selected_arm": derived["active_arm"]}
            if result.get("rollback"):
                rollback(path, module, "FIXED_60_MATURE_NODE: " + json.dumps(result.get("triggers"), ensure_ascii=False))
    except (ValueError, OSError) as exc:
        _binding_failure(protocol_path, module, exc)
        raise
    receipt = {"module": module, "node": node, "reviewed_at": now_utc().isoformat(),
               "protocol_path": str(path), "protocol_sha256": file_sha256(path),
               "first_signal_date": derived["first_signal_date"], "input_hashes": bindings,
               "result": result, "production_model_changed": False}
    if node == 60 and node_index is not None:
        receipt.update(node_index=node_index, node_signal_dates=grid)
        _append_chain(nodes_path, {"event": "FIXED_60_NODE_REVIEW", "anchor": anchor,
                                   "active_arm": derived["active_arm"], "node_index": node_index,
                                   "receipt": receipt})
    if output is not None:
        _write_new(_owned(output, root), receipt)
    return receipt


def promote(protocol_path, module, receipt_path):
    """Advance only the independent research pointer after a recomputed 120 gate."""
    path, protocol, calendar, root = _protocol(protocol_path)
    receipt_path = _owned(receipt_path, root)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("node") != 120 or receipt.get("module") != module
            or receipt.get("protocol_sha256") != file_sha256(path)):
        raise ValueError("promotion receipt does not bind this fixed 120-session module")
    _receipt_binding_check(receipt)
    live = review_node(path, module, 120)["result"]
    chosen = live.get("selected_arm")
    if (not live.get("upgrade_eligible") or chosen not in {"C1", "C2"}
            or not live.get("gates", {}).get(chosen, {}).get("passed")
            or live != receipt["result"]):
        raise ValueError("promotion gate not passed by source-bound mature fixed evidence")
    previous = _pointer(root)["modules"].get(module, {})
    if (previous.get("active_arm") == chosen and previous.get("protocol_sha256") == file_sha256(path)
            and not previous.get("suspended")):
        return {"event": "ALREADY_ACTIVE", "pointer": previous, "production_model_changed": False}
    local_today = now_utc().astimezone(SHANGHAI).date().isoformat()
    effective = next((d for d in calendar["sessions"] if d > local_today), None)
    if effective is None:
        raise ValueError("official calendar extension required before future promotion")
    models = _models(path, protocol, module, path.parent / "models", now_utc())
    pointer = _pointer(root)
    pointer["modules"][module] = {"active_arm": chosen, "effective_signal": effective,
                                   "promoted_at": now_utc().isoformat(), "suspended": False,
                                   "model": models[chosen], "fallback": models["BASE"],
                                   "protocol_path": str(path), "protocol_sha256": file_sha256(path), "receipt": str(receipt_path),
                                   "receipt_sha256": file_sha256(receipt_path),
                                   "production_model_changed": False}
    return _save_pointer(root, pointer, "PROMOTE_RESEARCH_POINTER")


def _receipt_binding_check(receipt):
    """Preserve a historical node when its audited ledgers only append later rows."""
    append_names = {"registrations.jsonl", "settlements.jsonl", "calendar_extensions.jsonl"}
    protocol_path = Path(receipt["protocol_path"]).resolve()
    root = protocol_path.parent / "shadow"
    for name, expected in receipt["input_hashes"].items():
        path = Path(name)
        if path.parent.resolve() == root.resolve() and path.name in append_names:
            _read_chain(path)
            prefix, matched = b"", False
            for line in path.read_bytes().splitlines(keepends=True):
                prefix += line
                if hashlib.sha256(prefix).hexdigest() == expected:
                    matched = True
                    break
            if not matched:
                raise ValueError("receipt original ledger prefix hash binding conflict")
        else:
            _source_check({name: expected})


def _verify_active_impl(registration, active_path):
    """Read-only proof for applying one accepted arm to a genuinely new plan."""
    record, path, protocol, calendar, root = _registered(registration)
    active_path = _owned(active_path, root)
    if active_path != (root / "active_pointer.json").resolve():
        raise ValueError("active pointer must be this protocol's audited control file")
    pointer = _pointer(root)["modules"].get(record["module"], {"active_arm": "BASE"})
    proof = {"status": "UNCHANGED_BASE", "registration_path": str(Path(registration).resolve()),
             "registration_sha256": file_sha256(registration), "active_path": str(active_path),
             "active_sha256": file_sha256(active_path) if active_path.exists() else None,
             "protocol_path": str(path), "protocol_sha256": file_sha256(path),
             "verified_at": now_utc().isoformat(), "production_model_changed": False}
    if pointer["active_arm"] == "BASE":
        return record, pointer, proof
    arm = pointer["active_arm"]
    if arm not in {"C1", "C2"} or pointer.get("suspended"):
        raise ValueError("active challenge is suspended or not preregistered")
    require_current_closed_signal(record["signal_date"], calendar["sessions"])
    if (record["signal_date"] < pointer["effective_signal"]
            or stamp(pointer["promoted_at"]) > stamp(record["knowledge_cutoff"])):
        raise ValueError("accepted arm was not effective at original signal knowledge cutoff")
    if pointer["protocol_sha256"] != file_sha256(path):
        raise ValueError("active protocol hash conflict")
    receipt_path = _owned(pointer["receipt"], root)
    if file_sha256(receipt_path) != pointer["receipt_sha256"]:
        raise ValueError("active acceptance receipt hash conflict")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("node") != 120 or receipt.get("module") != record["module"]
            or receipt.get("protocol_sha256") != file_sha256(path)
            or stamp(receipt["reviewed_at"]) > stamp(record["knowledge_cutoff"])):
        raise ValueError("active receipt is outside original knowledge boundary")
    _receipt_binding_check(receipt)
    live = review_node(path, record["module"], 120)["result"]
    if (live != receipt["result"] or not live.get("upgrade_eligible") or live.get("selected_arm") != arm
            or not live.get("gates", {}).get(arm, {}).get("passed")):
        raise ValueError("active arm no longer matches source-bound full acceptance gates")
    if pointer["model"] != record["model_bindings"][arm] or pointer["fallback"] != record["model_bindings"]["BASE"]:
        raise ValueError("active model hash binding differs from the new registration")
    proof.update(status="VERIFIED_ACTIVE", receipt_path=str(receipt_path),
                 receipt_sha256=pointer["receipt_sha256"], verified_at=now_utc().isoformat())
    return record, pointer, proof


def verify_active_for_registration(registration, active_path):
    record = json.loads(Path(registration).read_text(encoding="utf-8"))
    try:
        return _verify_active_impl(registration, active_path)
    except (ValueError, OSError) as exc:
        _binding_failure(record["protocol_path"], record["module"], exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    for name in ("protocol", "module", "features", "models", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("settle")
    p.add_argument("--registration", required=True)
    p.add_argument("--archive", required=True)
    p = sub.add_parser("review")
    p.add_argument("--protocol", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--node", type=int, choices=(60, 120), default=120)
    p.add_argument("--output", required=True)
    p = sub.add_parser("promote")
    for name in ("protocol", "module", "receipt"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("rollback")
    for name in ("protocol", "module", "reason"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("bind-calendar")
    p.add_argument("--protocol", required=True)
    p.add_argument("--calendar", required=True)
    args = parser.parse_args(argv)
    if args.command == "predict":
        result = predict(args.protocol, args.module, args.features, args.models, args.output)
    elif args.command == "settle":
        result = settle(args.registration, args.archive)
    elif args.command == "review":
        result = review_node(args.protocol, args.module, args.node, args.output)
        decision = result.get("result", {})
        chosen = decision.get("selected_arm")
        if (args.node == 120 and decision.get("upgrade_eligible") and chosen in {"C1", "C2"}
                and decision.get("gates", {}).get(chosen, {}).get("passed")):
            result = {"review": result, "promotion": promote(args.protocol, args.module, args.output)}
    elif args.command == "promote":
        result = promote(args.protocol, args.module, args.receipt)
    elif args.command == "rollback":
        result = rollback(args.protocol, args.module, args.reason)
    else:
        result = bind_calendar(args.protocol, args.calendar)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
