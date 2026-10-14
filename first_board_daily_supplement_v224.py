"""Run the V221-V224 first-board supplement as one immutable daily package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from first_board_forward_settlement_v224 import run as run_settlement
from first_board_open_confirmation_v221 import run as run_confirmation
from first_board_selective_policy_v223 import run as run_policy
from first_board_theme_catalyst_v222 import run as run_theme
from first_board_multilane_identity_v229 import run as run_multilane


VERSION = "v224-first-board-daily-supplement-1"


def run(ns: argparse.Namespace) -> dict:
    root = Path(ns.output)
    if root.exists():
        raise FileExistsError("refusing to overwrite immutable daily supplement package")
    root.mkdir(parents=True)

    v221_path = root / "v221_open_confirmation.json"
    v221 = run_confirmation(ns.snapshot, ns.auction, ns.minutes, ns.theme_sync,
                            ns.decision_date, str(v221_path))
    v222 = run_theme(ns.theme_matrix, ns.theme_evidence, ns.theme_cutoff,
                     str(root / "v222_theme_catalyst"))
    theme_map = getattr(ns, "theme_map", None)
    symbolic_report = getattr(ns, "symbolic_report", None)
    if bool(theme_map) != bool(symbolic_report):
        raise ValueError("--theme-map and --symbolic-report must be supplied together")
    v229 = (run_multilane(ns.current_watchlist, theme_map, symbolic_report,
                          str(root / "v229_multilane_identity"),
                          getattr(ns, "driver_claims", None),
                          getattr(ns, "official_driver_times", None))
            if theme_map else None)
    v223 = run_policy(ns.history_rows, ns.promotion_predictions, ns.current_watchlist,
                      str(root / "v223_selective_policy"), ns.min_train_sessions)
    v224_path = root / "v224_forward_settlement.json"
    v224 = run_settlement(ns.snapshot, str(v224_path), ns.ledger, str(v221_path),
                          ns.promotion_pool, ns.daily_bars)
    recency_report = None
    recency_path = getattr(ns, "official_announcement_recency", None)
    if recency_path:
        recency_report = json.loads(Path(recency_path).read_text(encoding="utf-8-sig"))
        if recency_report.get("version") != "v233-official-announcement-recency-1":
            raise ValueError("official-announcement-recency report version mismatch")
        if recency_report.get("signal_date") != v221.get("signal_date"):
            raise ValueError("official-announcement-recency signal date mismatch")
    adjudication_report = None
    adjudication_path = getattr(ns, "official_driver_adjudication", None)
    if adjudication_path:
        adjudication_report = json.loads(
            Path(adjudication_path).read_text(encoding="utf-8-sig"))
        if adjudication_report.get("version") != "v234-official-driver-adjudication-1":
            raise ValueError("official-driver-adjudication report version mismatch")
        if adjudication_report.get("signal_date") != v221.get("signal_date"):
            raise ValueError("official-driver-adjudication signal date mismatch")
    summary = {
        "version": VERSION,
        "decision_date": ns.decision_date,
        "open_confirmation": v221.get("decision"),
        "verified_real_themes": v222.get("verified_real_theme"),
        "multilane_identity": ({
            "narrative_lane_candidate": v229.get("narrative_lane_candidate"),
            "symbolic_pure_buyable_pioneer": v229.get("symbolic_pure_buyable_pioneer"),
            "symbolic_dual_identity_candidate": v229.get("symbolic_dual_identity_candidate"),
            "official_first_board_primary": v229.get("official_first_board_primary"),
        } if v229 else {"status": "NOT_PROVIDED"}),
        "official_announcement_recency": ({
            "status": recency_report.get("status"),
            "mainboard_claim_rows": recency_report.get("mainboard_claim_rows"),
            "successful_official_index_queries": recency_report.get(
                "successful_official_index_queries"),
            "failed_official_index_queries": recency_report.get(
                "failed_official_index_queries"),
            "candidates_with_compatible_official_titles": recency_report.get(
                "candidates_with_compatible_official_titles"),
            "title_match_proves_claim": False,
        } if recency_report else {"status": "NOT_PROVIDED"}),
        "official_driver_adjudication": ({
            "status": adjudication_report.get("status"),
            "adjudicated_candidates": adjudication_report.get("adjudicated_candidates"),
            "verdict_counts": adjudication_report.get("verdict_counts"),
            "fully_supported_candidates": adjudication_report.get(
                "fully_supported_candidates"),
            "material_detail_conflicts": adjudication_report.get(
                "material_detail_conflicts"),
            "after_cutoff_unusable": adjudication_report.get(
                "after_cutoff_unusable"),
            "official_fulltext_match_is_buy_signal": False,
        } if adjudication_report else {"status": "NOT_PROVIDED"}),
        "selective_policy_current": v223.get("current_frozen_top1"),
        "forward_settlement": v224.get("status"),
        "entry_authorized": False,
        "package_status": (
            "DATA_COMPLETE_RESEARCH_ONLY"
            if v221.get("decision") != "DATA_UNAVAILABLE" and v224.get("status") == "COMPLETE"
            else "DATA_INCOMPLETE_RESEARCH_ONLY"
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--decision-date", required=True)
    ap.add_argument("--auction")
    ap.add_argument("--minutes")
    ap.add_argument("--theme-sync")
    ap.add_argument("--theme-matrix", required=True)
    ap.add_argument("--theme-map")
    ap.add_argument("--symbolic-report")
    ap.add_argument("--driver-claims")
    ap.add_argument("--official-driver-times")
    ap.add_argument("--official-announcement-recency")
    ap.add_argument("--official-driver-adjudication")
    ap.add_argument("--theme-evidence", required=True)
    ap.add_argument("--theme-cutoff", required=True)
    ap.add_argument("--history-rows", required=True)
    ap.add_argument("--promotion-predictions", required=True)
    ap.add_argument("--current-watchlist", required=True)
    ap.add_argument("--promotion-pool")
    ap.add_argument("--daily-bars")
    ap.add_argument("--min-train-sessions", type=int, default=10)
    ap.add_argument("--ledger", default="data/v224/settlement_ledger.jsonl")
    ap.add_argument("--output", required=True)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
