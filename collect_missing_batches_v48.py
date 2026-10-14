#!/usr/bin/env python3
"""Resume immutable BaoStock minute collection in bounded batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


VERSION = "v48-resumable-minute-collector-1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def successful_keys(output_root: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for manifest_path in sorted(output_root.glob("batch_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest.get("cases", []):
            if case.get("error_code") != "0" or case.get("rows") != 48:
                continue
            raw_path = manifest_path.parent / f"{case['code']}_{case['date']}.json"
            if not raw_path.exists() or sha256_file(raw_path) != case.get("raw_sha256"):
                raise ValueError(f"archive hash mismatch: {raw_path}")
            keys.add((case["code"], case["date"]))
    return keys


def requested_successful_keys(output_root: Path,
                              requested: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Do not let successes from an older request pack inflate current progress."""
    return successful_keys(output_root).intersection(requested)


def _write_progress(output_root: Path, total: int, returned: int,
                    attempts: list[dict[str, Any]], stopped_reason: str) -> dict[str, Any]:
    result = {
        "version": VERSION,
        "updated_at": datetime.now().astimezone().isoformat(),
        "requested": total,
        "returned_complete": returned,
        "remaining": total - returned,
        "batches_this_run": len(attempts),
        "attempts": attempts,
        "stopped_reason": stopped_reason,
        "data_claim": "48 provider rows only; not independently verified prices or fills",
    }
    path = output_root / "progress.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run(requests_path: Path, output_root: Path, batch_size: int, max_batches: int) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 40 or max_batches < 1:
        raise ValueError("batch_size must be 1..40 and max_batches positive")
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    requested = [(str(item[0]), str(item[1])) for item in requests]
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate request identities")
    output_root.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    requested_set = set(requested)
    prior = requested_successful_keys(output_root, requested_set)
    for _ in range(max_batches):
        returned = requested_successful_keys(output_root, requested_set)
        missing = [key for key in requested if key not in returned]
        if not missing:
            return _write_progress(output_root, len(requested), len(returned), attempts, "COMPLETE")
        batch = missing[:batch_size]
        existing_indices = [int(path.name.split("_")[-1]) for path in output_root.glob("batch_*")
                            if path.name.split("_")[-1].isdigit()]
        batch_index = max(existing_indices, default=0) + 1
        request_path = output_root / f"requests_{batch_index:04d}.json"
        batch_dir = output_root / f"batch_{batch_index:04d}"
        request_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve().parent / "collect_historical_5min_v33.py"),
                   "--cases", str(request_path.resolve()), "--output", str(batch_dir.resolve())]
        process = subprocess.run(command, capture_output=True, text=True, timeout=65)
        after = requested_successful_keys(output_root, requested_set)
        gained = len(after - returned)
        attempts.append({
            "batch": batch_index,
            "requested": len(batch),
            "new_complete": gained,
            "returncode": process.returncode,
            "request_sha256": sha256_file(request_path),
            "manifest": str((batch_dir / "manifest.json").resolve()) if (batch_dir / "manifest.json").exists() else None,
        })
        if process.returncode != 0 or gained == 0:
            return _write_progress(output_root, len(requested), len(after), attempts,
                                   "STOPPED_NO_PROGRESS_OR_WORKER_ERROR")
    returned = requested_successful_keys(output_root, requested_set)
    reason = "MAX_BATCHES_REACHED_WITH_PROGRESS" if len(returned) > len(prior) else "STOPPED_NO_PROGRESS"
    return _write_progress(output_root, len(requested), len(returned), attempts, reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.requests, args.output_root, args.batch_size, args.max_batches),
                     ensure_ascii=False, indent=2))
