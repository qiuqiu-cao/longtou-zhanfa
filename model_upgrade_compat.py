"""Explicit compatibility wrapper executing unchanged fixed-controller checks.

Recognizes one approved optional protected-runner extension. No global module
mutation, hash substitution, numeric refit, altered rule or retrospective plan.
"""
import argparse
import ast
import builtins
from datetime import datetime, timezone
import json
from pathlib import Path
from types import FunctionType, ModuleType

import model_upgrade_inputs as original_inputs
import model_upgrade_shadow as original_shadow
from model_upgrade_research import sha, read


def verify_policy(path):
    path = Path(path).resolve(); p = read(path)
    if (p["version"] != "fixed-shadow-approved-protected-extension-1"
            or p["exception_path"] != str(Path(__file__).with_name("hypothesis_bound_daily_runner_v298.py").resolve())
            or p["numerical_refits"] != 0 or p["feature_formula_changed"] is not False
            or p["parameter_changed"] is not False or p["original_protocol_or_forecast_changed"] is not False
            or datetime.fromisoformat(p["formed_at"]) > datetime.now(timezone.utc)):
        raise ValueError("unsupported compatibility scope/time")
    implementation_path = path.parent/"implementation_BEFORE_USE.json"
    implementation = read(implementation_path)
    if implementation["policy_sha256"] != sha(path):
        raise ValueError("compatibility policy changed after binding")
    bindings = p["original_numeric_code_hashes"] | p["authorization_receipts"] | implementation["sources"] | {
        p["protocol_path"]: p["protocol_sha256"], p["exception_path"]: p["current_sha256"],
        p["preserved_original_path"]: p["original_sha256"]}
    original_shadow._source_check(bindings)
    protocol = read(p["protocol_path"])
    if protocol["protected_sha256"].get(p["exception_path"]) != p["original_sha256"]:
        raise ValueError("preserved original is not the originally protected program")
    if not any(read(r).get("source_hashes", {}).get(p["exception_path"]) == p["current_sha256"] for r in p["authorization_receipts"]):
        raise ValueError("current runner lacks actual approved G1 source receipt")
    def definitions(filename):
        return {n.name: n for n in ast.parse(Path(filename).read_text(encoding="utf-8")).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    old, new = definitions(p["preserved_original_path"]), definitions(p["exception_path"])
    if set(old) != set(new) or any(ast.dump(old[n]) != ast.dump(new[n]) for n in old if n != "validate_extension"):
        raise ValueError("protected extension changed original non-extension definitions")
    validator = new["validate_extension"]
    validator.body = [n for n in validator.body if not isinstance(n, ast.If)]
    if ast.dump(validator) != ast.dump(old["validate_extension"]):
        raise ValueError("protected extension removed original validation")
    return p, {str(path): sha(path), str(implementation_path): sha(implementation_path),
               str(Path(__file__).resolve()): sha(__file__)}


def clone_module(module, overrides=None, importer=None):
    """Local namespaces only: the actual imported module remains untouched."""
    clone = ModuleType(module.__name__); clone.__dict__.update(module.__dict__)
    if importer:
        clone.__dict__["__builtins__"] = {**vars(builtins), "__import__": importer}
    for name, obj in module.__dict__.items():
        if isinstance(obj, FunctionType) and obj.__module__ == module.__name__:
            replacement = FunctionType(obj.__code__, clone.__dict__, obj.__name__, obj.__defaults__, obj.__closure__)
            replacement.__kwdefaults__ = obj.__kwdefaults__; replacement.__annotations__ = obj.__annotations__
            clone.__dict__[name] = replacement
    clone.__dict__.update(overrides or {})
    return clone


def runtime(policy_path):
    policy, proof = verify_policy(policy_path)
    def check_sources(sources):
        ordinary = {}
        for filename, expected in sources.items():
            if filename == policy["exception_path"] and expected == policy["original_sha256"]:
                original_shadow._source_check({policy["preserved_original_path"]: expected,
                    filename: policy["current_sha256"]})
            else:
                ordinary[filename] = expected
        original_shadow._source_check(ordinary)
    def verify_protocol(filename):
        if Path(filename).resolve() != Path(policy["protocol_path"]):
            raise ValueError("compatibility cannot migrate to another protocol")
        p = read(filename)
        for group in ("source_hashes", "protected_sha256"):
            check_sources(p[group])
        if (p["version"] != "model-upgrade-fixed-1" or p["freeze_cutoff"] != "2026-09-01"
                or p["forward_sessions"] != 120 or p["preferred_arm"] != "C1"):
            raise ValueError("original numeric protocol semantics changed")
        if read(Path(filename).parent/"implementation_BEFORE_FITS.json")["protocol_sha256"] != sha(filename):
            raise ValueError("original protocol implementation binding changed")
        return {**p, "protocol_path": str(Path(filename).resolve()), "protocol_sha256": sha(filename)}
    inputs = clone_module(original_inputs, {"verify_protocol": verify_protocol})
    build_outcomes = inputs.build_outcomes
    def bound_outcomes(*args, **kwargs):
        result = build_outcomes(*args, **kwargs)
        result["sources"] = result["sources"] | proof
        return result
    inputs.build_outcomes = bound_outcomes
    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "model_upgrade_inputs":
            return inputs
        return builtins.__import__(name, globals, locals, fromlist, level)
    shadow = clone_module(original_shadow, {"_source_check": check_sources}, importer)
    return inputs, shadow


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--compatibility", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    settle = sub.add_parser("settle"); settle.add_argument("--registration", required=True); settle.add_argument("--archive", required=True)
    predict = sub.add_parser("predict")
    for name in ("protocol", "module", "features", "models", "output"):
        predict.add_argument("--"+name, required=True)
    build = sub.add_parser("build-features")
    for name in ("protocol", "archive", "calendar", "output"):
        build.add_argument("--"+name, required=True)
    build.add_argument("--history-archive", action="append", default=[])
    review = sub.add_parser("review")
    for name in ("protocol", "module", "node", "output"):
        review.add_argument("--"+name, required=True, type=int if name == "node" else str)
    args = parser.parse_args(); inputs, shadow = runtime(args.compatibility)
    if args.command == "settle": result = shadow.settle(args.registration, args.archive)
    elif args.command == "predict": result = shadow.predict(args.protocol, args.module, args.features, args.models, args.output)
    elif args.command == "build-features": result = inputs.build_feature_archive(args.protocol, args.archive, args.calendar, args.output, args.history_archive)
    else: result = shadow.review(args.protocol, args.module, args.node, args.output)
    print(json.dumps({"status": "ORIGINAL_CONTROLLER_WITH_BOUND_COMPATIBILITY_COMPLETE",
        "command": args.command, "event": result.get("event"), "module": result.get("module")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
