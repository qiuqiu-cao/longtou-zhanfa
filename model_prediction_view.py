"""Source-bound original BASE view for diagnostics and fixed G1 inputs.

Copies explicit module outputs without inference or training. Candidate ranks
come from the native fixed ranking, including candidates outside the top five.
The view forms now; it cannot replace an earlier prediction or cutoff.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import model_champion_compat as champion_compat
from model_upgrade_research import read, sha
from next_session_expectation_contract_v271 import stamp


def relay_view(forecast, roster):
    """Retain original facts and calls, validate native order, add its ranks."""
    rows = deepcopy(forecast["candidates"])
    expected = {r["code"]: r for r in roster}
    if (len(expected) != len(roster) or len(rows) != len(expected)
            or len({r["code"] for r in rows}) != len(rows)
            or {r["code"] for r in rows} != set(expected)):
        raise ValueError("complete original candidate roster required")
    for row in rows:
        if any(row.get(k) != expected[row["code"]].get(k)
               for k in ("name", "boards", "actual_one_word_daily")):
            raise ValueError("original candidate fact conflict")
    if forecast["status"] == "KNOWN":
        for row in rows:
            score = row.get("raw_score")
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or not 0 <= score <= 1 or type(row.get("predicted_positive")) is not bool
                    or row["predicted_positive"] != (score > .5)):
                raise ValueError("original score/call definition conflict")
        ranking = [r["code"] for r in sorted(rows, key=lambda r: (-r["raw_score"], r["code"]))]
        if (forecast.get("ranking") != ranking
                or forecast.get("ranking_rule") != "raw_score descending; code ascending; no replacement"):
            raise ValueError("native fixed ranking conflict")
        positions = {code: i for i, code in enumerate(ranking, 1)}
    elif forecast["status"] == "UNKNOWN":
        if (forecast.get("ranking") != []
                or any(r.get("raw_score") is not None or r.get("predicted_positive") is not None for r in rows)):
            raise ValueError("unknown scores cannot have a ranking or call")
        positions = {}
    else:
        raise ValueError("unknown native relay status")
    for row in rows:
        rank = positions.get(row["code"])
        if "rank" in row and row["rank"] != rank:
            raise ValueError("original candidate rank conflict")
        row["rank"] = rank
    return {**deepcopy(forecast), "candidates": rows,
            "rank_source": "native complete ranking; score descending/code ascending"}


def project(market_path, relay_path, compatibility_path, implementation_path):
    """Rebuild original features and verify bindings without loading estimators."""
    paths = {"market": Path(market_path).resolve(), "relay": Path(relay_path).resolve()}
    modules = {key: read(path) for key, path in paths.items()}
    at = datetime.now(timezone.utc).isoformat()
    champion, compatibility_sources = champion_compat.runtime(compatibility_path, implementation_path)
    shadow = champion.shadow
    market, relay = modules["market"], modules["relay"]
    shared = ("signal_date", "target_date", "feature_archive", "feature_manifest_sha256",
              "feature_projection_sha256", "protocol_path", "protocol_sha256")
    if any(market.get(key) != relay.get(key) for key in shared):
        raise ValueError("original module date/feature/protocol scope differs")
    protocol_path, protocol, calendar, _ = shadow._protocol(relay["protocol_path"])
    if sha(protocol_path) != relay["protocol_sha256"]:
        raise ValueError("original protocol hash conflict")
    pack = champion.load_feature_archive(relay["feature_archive"], cutoff=at)
    projection_sha = hashlib.sha256(shadow.canonical(pack)).hexdigest()
    if (pack["signal_date"] != relay["signal_date"]
            or shadow.next_official_session(calendar["sessions"], pack["signal_date"]) != relay["target_date"]
            or pack["manifest_sha256"] != relay["feature_manifest_sha256"]
            or projection_sha != relay["feature_projection_sha256"]):
        raise ValueError("original feature/calendar projection conflict")
    sources = dict(compatibility_sources)
    # Bind current executable readers and raw inputs using their actual bytes.
    # Old protected runner versions are verified by the original compatibility
    # receipt, never represented as the current file's hash.
    shadow._source_check(pack["sources"])
    sources.update({str(Path(p).resolve()): sha(p) for p in pack["sources"]})
    for name, value in modules.items():
        if (value.get("version") != "explicit-module-champion-runtime-1"
                or value.get("module") != name or value.get("selected_arm") != "BASE"
                or value.get("production_model_changed") is not False
                or value.get("registry_mutated") is not False or value.get("score_calibrated") is not False):
            raise ValueError("original unchanged explicit BASE module output required")
        cutoff = stamp(value["knowledge_cutoff"])
        if (cutoff > stamp(at) or stamp(pack["available_at"]) > cutoff
                or value.get("feature_available_at") != pack["available_at"]):
            raise ValueError("original module actual availability/cutoff conflict")
        bindings = value.get("compatibility_sources")
        if bindings != compatibility_sources:
            raise ValueError("original module compatibility binding conflict")
        shadow._source_check(bindings)
        if value.get("runtime_code_sha256") != sha(Path(__file__).with_name("model_champion_runtime.py")):
            raise ValueError("original runtime source hash conflict")
        model = shadow._models(protocol_path, protocol, name, protocol_path.parent/"models", cutoff)["BASE"]
        if value.get("model_binding") != model:
            raise ValueError("original frozen BASE model binding conflict")
        sources.update({model["path"]: model["sha256"], model["model_path"]: model["model_sha256"]})
    result = {"status": "SOURCE_BOUND_ORIGINAL_BASE_VIEW", "signal_date": relay["signal_date"],
        "target_date": relay["target_date"], "generated_at": at, "knowledge_cutoff": at,
        "original_module_cutoffs": {k: v["knowledge_cutoff"] for k, v in modules.items()},
        "model_version": "fixed original BASE explicit modules unchanged",
        "market": deepcopy(market["forecast"]), "relay": relay_view(relay["forecast"], pack["relay"]),
        "model_bindings": {k: v["model_binding"] for k, v in modules.items()},
        "original_module_bindings": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
        "feature_archive": pack["feature_archive"],
        "feature_sha256": sha(Path(pack["feature_archive"])/"features.json"),
        "feature_manifest_sha256": pack["manifest_sha256"], "feature_projection_sha256": projection_sha,
        "production_model_changed": False, "new_numerical_fits": 0, "new_mature_forward_samples": 0,
        "score_calibrated": False, "prediction_effect_verified": False,
        "scope": "original score projection formed now; no new forecast, earlier freeze or cutoff replaced"}
    for path in (*paths.values(), protocol_path, Path(pack["feature_archive"])/"manifest.json",
                 Path(pack["feature_archive"])/"features.json", Path(__file__),
                 Path(__file__).with_name("model_champion_runtime.py")):
        sources[str(path.resolve())] = sha(path)
    result["source_hashes"] = sources
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("market-output", "relay-output", "compatibility", "implementation", "output"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    # Never overwrite the original inputs or write into protected experiment,
    # model or forecast registration directories.
    protocol_root = Path(read(args.relay_output)["protocol_path"]).resolve().parent
    if (output.exists() or output.suffix != ".json" or not output.parent.is_dir()
            or output.is_relative_to(protocol_root)
            or any(p.lower() == "expectation_registry" for p in output.parts)):
        raise ValueError("unique new JSON outside protocol/registry required")
    value = project(args.market_output, args.relay_output, args.compatibility, args.implementation)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": value["status"], "candidate_count": len(value["relay"]["candidates"]),
                      "ranking": value["relay"]["ranking"], "output_sha256": sha(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
