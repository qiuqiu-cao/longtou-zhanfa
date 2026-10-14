"""Run relay PK, role separation, 3/4-board expectation and forward snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from relay_leader_role_forward_v236 import snapshot as snapshot_roles
from relay_leader_role_forward_v236 import verify as verify_roles
from relay_leader_role_gate_v235 import run as run_roles
from relay_cross_theme_expectation_v239 import run as run_cross_theme
from relay_cross_theme_forward_v241 import snapshot as snapshot_cross_theme
from relay_cross_theme_forward_v241 import verify as verify_cross_theme
from relay_path_expectation_v242 import run as run_path_expectations
from relay_path_forward_v244 import snapshot as snapshot_path_expectations
from relay_path_forward_v244 import verify as verify_path_expectations
from same_height_leader_pk_v225 import run as run_same_height
from same_height_pk_forward_v226 import snapshot as snapshot_same_height
from same_height_pk_forward_v226 import verify as verify_same_height


VERSION = "v244-relay-leader-daily-1"


def run(ns: argparse.Namespace) -> dict:
    root = Path(ns.output)
    if root.exists():
        raise FileExistsError("refusing to overwrite immutable V236 daily package")
    same_height_dir = root / "v225_same_height"
    roles_dir = root / "v235_cross_height_roles"
    cross_theme_dir = root / "v239_cross_theme_expectation"
    path_expectation_dir = root / "v242_path_expectation"
    v225 = run_same_height(
        ns.review, ns.raw_limit_up, ns.announcements, ns.announcement_manifest,
        str(same_height_dir), ns.fulltext_review, ns.fulltext_review_report)
    v226 = snapshot_same_height(
        ns.same_height_registry, str(same_height_dir / "same_height_scores.csv"),
        str(same_height_dir / "report.json"))
    v235 = run_roles(
        str(same_height_dir / "same_height_scores.csv"),
        str(same_height_dir / "report.json"), ns.review, str(roles_dir))
    v236 = snapshot_roles(
        ns.role_registry, str(roles_dir / "relay_role_candidates.csv"),
        str(roles_dir / "report.json"))
    v239 = run_cross_theme(
        str(same_height_dir / "same_height_scores.csv"),
        str(same_height_dir / "report.json"),
        str(roles_dir / "relay_role_candidates.csv"),
        str(roles_dir / "report.json"), ns.review, str(cross_theme_dir))
    v241 = snapshot_cross_theme(
        ns.cross_theme_registry,
        str(cross_theme_dir / "three_four_board_pool.csv"),
        str(cross_theme_dir / "expectation_tree.csv"),
        str(cross_theme_dir / "report.json"))
    v242 = run_path_expectations(
        str(cross_theme_dir / "three_four_board_pool.csv"),
        str(cross_theme_dir / "expectation_tree.csv"),
        str(cross_theme_dir / "report.json"), ns.review,
        str(path_expectation_dir))
    v244 = snapshot_path_expectations(
        ns.path_expectation_registry,
        str(path_expectation_dir / "path_expectation_matrix.csv"),
        str(path_expectation_dir / "scenario_tree.csv"),
        str(path_expectation_dir / "report.json"))
    same_integrity = verify_same_height(ns.same_height_registry)
    role_integrity = verify_roles(ns.role_registry)
    cross_theme_integrity = verify_cross_theme(ns.cross_theme_registry)
    path_expectation_integrity = verify_path_expectations(ns.path_expectation_registry)
    summary = {
        "version": VERSION,
        "signal_date": v225["signal_date"],
        "same_height_descriptive_leaders": v225["group_descriptive_leaders"],
        "height_emotion_anchor": v235["height_emotion_anchor"],
        "close_relay_challenger": v235["close_relay_challenger"],
        "official_relay_primary": None,
        "three_four_board_primary_expectation": v239["primary_expected_winner"],
        "three_four_board_conditional_fallbacks": v239["conditional_fallback_winners"],
        "three_four_board_no_winner_scenario": v239["no_winner_scenario"],
        "path_expectation_status": v242["status"],
        "path_expectation_improvements": v242["improvements_over_v239"],
        "same_height_freeze_id": v226["freeze_id"],
        "relay_role_freeze_id": v236["freeze_id"],
        "cross_theme_freeze_id": v241["freeze_id"],
        "path_expectation_freeze_id": v244["freeze_id"],
        "same_height_registry_integrity": same_integrity["status"],
        "relay_role_registry_integrity": role_integrity["status"],
        "cross_theme_registry_integrity": cross_theme_integrity["status"],
        "path_expectation_registry_integrity": path_expectation_integrity["status"],
        "entry_authorized": False,
    }
    (root / "summary.json").write_text(
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
    ap.add_argument("--same-height-registry", required=True)
    ap.add_argument("--role-registry", required=True)
    ap.add_argument("--cross-theme-registry", required=True)
    ap.add_argument("--path-expectation-registry", required=True)
    ap.add_argument("--output", required=True)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
