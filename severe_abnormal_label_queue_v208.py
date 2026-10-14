"""Produce a verification queue for 100% severe-abnormal labels.

Scanned threshold crossings are hypotheses.  Only official publications can be
positive labels; absence of a publication is not a negative label unless a
complete candidate universe and a fully observed declared horizon are proven.
No calibrated probability is emitted by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "v208-severe-abnormal-label-queue-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(library: dict, readiness: dict) -> tuple[pd.DataFrame, dict]:
    events = pd.DataFrame(library.get("events", []))
    required = {"code", "event_date", "target_pct", "scanned", "official_confirmed",
                "training_label_ready", "evidence_available_at"}
    if events.empty or required - set(events):
        raise ValueError("event library is empty or missing label fields")
    events["code"] = events.code.astype(str).str.zfill(6)
    events = events[events.target_pct.eq(100)].copy()
    events["label_status"] = "UNVERIFIED_THRESHOLD_HYPOTHESIS"
    events.loc[events.official_confirmed.astype(bool), "label_status"] = "OFFICIAL_POSITIVE_NOT_TRAINING_READY"
    events.loc[events.training_label_ready.astype(bool), "label_status"] = "TRAINING_READY_POSITIVE"
    events["priority"] = (
        events.scanned.astype(bool).astype(int) * 10
        + events.official_confirmed.astype(bool).astype(int) * 100
        + events.training_label_ready.astype(bool).astype(int) * 1000
    )
    events["required_next_action"] = events.label_status.map({
        "UNVERIFIED_THRESHOLD_HYPOTHESIS": "RECONCILE_WITH_OFFICIAL_EXCHANGE_PUBLICATION",
        "OFFICIAL_POSITIVE_NOT_TRAINING_READY": "COMPLETE_RULE_WINDOW_SECURITY_STATUS_AND_FEATURE_TIMESTAMP",
        "TRAINING_READY_POSITIVE": "LOCK_LABEL_AND_PRESERVE_EVIDENCE_HASH",
    })
    queue = events.sort_values(["priority", "event_date", "code"], ascending=[False, True, True])
    failed = list(readiness.get("failed_gates", []))
    allowed = bool(readiness.get("calibrated_hit_probability_allowed"))
    report = {
        "version": VERSION, "status": "CALIBRATION_BLOCKED" if not allowed else "LABEL_GATE_PASSED",
        "target": "official_100pct_severe_abnormal_publication_within_declared_horizon",
        "candidate_events": int(len(queue)),
        "official_positive_events": int(queue.official_confirmed.astype(bool).sum()),
        "training_ready_positive_events": int(queue.training_label_ready.astype(bool).sum()),
        "training_ready_negative_events": 0,
        "failed_gates": failed,
        "calibrated_probability_emitted": False,
        "calibrated_probability_allowed": allowed,
        "negative_label_policy": (
            "No publication is not a negative. A negative requires a frozen full-market cohort, "
            "complete declared horizon, official-publication reconciliation and no right censoring."
        ),
    }
    return queue, report


def run(library_path, readiness_path, output_dir):
    sources = [Path(library_path), Path(readiness_path)]
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable severe-abnormal label queue")
    queue, report = build(
        json.loads(sources[0].read_text(encoding="utf-8")),
        json.loads(sources[1].read_text(encoding="utf-8")))
    target.mkdir(parents=True)
    output = target / "official_label_verification_queue.csv"
    queue.to_csv(output, index=False, encoding="utf-8-sig")
    report["source_hashes"] = {str(path.resolve()): _sha256(path) for path in sources}
    report["output_sha256"] = _sha256(output)
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.library, args.readiness, args.output), ensure_ascii=False, indent=2))
