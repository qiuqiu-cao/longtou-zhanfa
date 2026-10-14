"""Run the unchanged explicit production selector with bound G1 compatibility."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import model_champion_runtime as native_champion
import model_upgrade_compat as compat
from model_upgrade_research import read, sha


def runtime(policy_path, implementation_path):
    policy_path = Path(policy_path).resolve()
    implementation_path = Path(implementation_path).resolve()
    receipt = read(implementation_path)
    if (receipt.get("version") != "champion-bound-protected-extension-1"
            or receipt.get("compatibility_path") != str(policy_path)
            or receipt.get("compatibility_sha256") != sha(policy_path)
            or receipt.get("numeric_refits") != 0
            or receipt.get("selection_rules_changed") is not False
            or receipt.get("production_model_changed") is not False
            or datetime.fromisoformat(receipt["formed_at"]) > datetime.now(timezone.utc)):
        raise ValueError("unsupported champion compatibility receipt")
    native_champion.shadow._source_check(receipt["sources"])
    if receipt["sources"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError("champion adapter implementation binding missing")
    inputs, shadow = compat.runtime(policy_path)
    champion = compat.clone_module(native_champion, {
        "shadow": shadow, "load_feature_archive": inputs.load_feature_archive})
    return champion, {str(policy_path): sha(policy_path),
                      str(implementation_path): sha(implementation_path),
                      str(Path(__file__).resolve()): sha(__file__)}


def predict(policy_path, implementation_path, protocol, module, features, registration=None):
    champion, proof = runtime(policy_path, implementation_path)
    result = champion.predict(protocol, module, features, registration)
    return {**result, "compatibility_sources": proof,
            "compatibility_scope": "approved optional V298 cycle/G1 extension; unchanged original selector and all source/date/model/acceptance checks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("compatibility", "implementation", "protocol", "module", "features", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--registration")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path(args.protocol).resolve().parent
    if (output.exists() or output.suffix != ".json" or not output.parent.is_dir()
            or output.is_relative_to(root)
            or any(part.lower() == "expectation_registry" for part in output.parts)):
        raise ValueError("unique production JSON outside fixed protocol/registry in existing directory required")
    result = predict(args.compatibility, args.implementation, args.protocol,
                     args.module, args.features, args.registration)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "BOUND_ORIGINAL_CHAMPION_PREDICTION_COMPLETE",
                      "module": result["module"], "selected_arm": result["selected_arm"]}))


if __name__ == "__main__":
    main()
