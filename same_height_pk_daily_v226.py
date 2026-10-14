"""One-command daily V225 scoring plus V226 immutable snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from same_height_leader_pk_v225 import run as run_pk
from same_height_pk_forward_v226 import snapshot, verify


def run(args: argparse.Namespace) -> dict:
    report = run_pk(args.review, args.raw_limit_up, args.announcements,
                    args.announcement_manifest, args.output,
                    args.fulltext_review, args.fulltext_review_report)
    out = Path(args.output)
    frozen = snapshot(args.registry, str(out / "same_height_scores.csv"),
                      str(out / "report.json"))
    integrity = verify(args.registry)
    summary = {
        "version": "v226-same-height-pk-daily-1",
        "signal_date": report["signal_date"],
        "group_descriptive_leaders": report["group_descriptive_leaders"],
        "forward_freeze_id": frozen["freeze_id"],
        "registry_integrity": integrity["status"],
        "official_champion_claimed": False,
        "entry_authorized": False,
    }
    (out / "daily_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", required=True)
    ap.add_argument("--raw-limit-up", required=True)
    ap.add_argument("--announcements", required=True)
    ap.add_argument("--announcement-manifest", required=True)
    ap.add_argument("--fulltext-review")
    ap.add_argument("--fulltext-review-report")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--output", required=True)
    return ap


if __name__ == "__main__":
    ns = parser().parse_args()
    print(json.dumps(run(ns), ensure_ascii=False, indent=2))
