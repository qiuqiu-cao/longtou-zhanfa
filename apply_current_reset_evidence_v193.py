"""Apply V192 reset chronology without overstating regulatory completeness.

Only a complete official exchange chronology can set the generic
``reset_status`` field to ``verified`` for the V171 diagnostic contract.
Company-disclosure search coverage remains unavailable at that gate.  All
other V191 blockers remain unchanged, so this overlay cannot create an
official distance, candidate ranking, probability, or entry signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v193-current-reset-evidence-overlay-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_reset_evidence(
    distances: pd.DataFrame,
    reset: pd.DataFrame,
) -> pd.DataFrame:
    required_distance = {
        "signal_date", "code", "benchmark_status", "reset_status",
        "corporate_action_status", "security_eligibility_status",
        "regulatory_distance_status", "research_indicator_status",
        "provider_aligned_30_return_coverage", "ranking_allowed", "buy_signal",
    }
    required_reset = {
        "code", "reset_chronology_status", "official_exchange_chronology_complete",
        "severe_event_count_since_rule_effective", "latest_exchange_publication_date",
        "latest_reset_effective_session", "source_level", "query_coverage",
    }
    if required_distance - set(distances):
        raise ValueError("distance fields missing")
    if required_reset - set(reset):
        raise ValueError("reset evidence fields missing")
    left, right = distances.copy(), reset.copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    right["code"] = right["code"].astype(str).str.zfill(6)
    if left["code"].duplicated().any() or right["code"].duplicated().any():
        raise ValueError("duplicate candidate codes")
    if set(left["code"]) != set(right["code"]):
        raise ValueError("reset evidence candidate coverage mismatch")
    keep = [
        "code", "reset_chronology_status", "official_exchange_chronology_complete",
        "severe_event_count_since_rule_effective", "latest_exchange_publication_date",
        "latest_reset_effective_session", "source_level", "query_coverage",
    ]
    merged = left.drop(columns=["reset_status"]).merge(
        right[keep], on="code", how="left", validate="one_to_one")
    complete = merged["official_exchange_chronology_complete"].fillna(False).astype(bool)
    merged["reset_status"] = np.where(complete, "verified", "unavailable")
    merged["reset_evidence_detail"] = merged["reset_chronology_status"]
    merged["reset_evidence_source_level"] = merged["source_level"]
    merged["reset_evidence_query_coverage"] = merged["query_coverage"]
    merged["official_reset_evidence"] = complete
    merged["regulatory_distance_status"] = "unavailable"
    merged["ranking_allowed"] = False
    merged["buy_signal"] = False
    if "hundred_percent_path_potential" in merged:
        merged["hundred_percent_path_potential"] = None
    return merged.drop(columns=["source_level", "query_coverage"])


def v171_compat(frame: pd.DataFrame, available_at: str) -> pd.DataFrame:
    result = frame.copy()
    result["available_at"] = available_at
    result["coverage_complete"] = result[
        "provider_aligned_30_return_coverage"].fillna(False)
    result["indicator_status"] = np.where(
        result["research_indicator_status"].eq("inferred_unreconciled"),
        "inferred", "unavailable")
    result["security_status"] = result["security_eligibility_status"].fillna("unavailable")
    result["scenario_limit_boards_to_100"] = result[
        "flat_benchmark_rounded_limit_first_hit_step"]
    result["active_theme_path_status"] = "unavailable"
    result["competition_identity_status"] = "unavailable"
    result["competition_identity"] = "unavailable"
    result["theme_support_ex_self"] = np.nan
    result["same_height_rivals"] = np.nan
    columns = [
        "signal_date", "code", "available_at", "rule_id", "rule_status",
        "benchmark_code", "benchmark_status", "reset_status",
        "corporate_action_status", "security_status", "coverage_complete",
        "indicator_status", "distance_100_pp", "distance_200_pp",
        "scenario_limit_boards_to_100", "scenario_status",
        "active_theme_path_status", "competition_identity_status",
        "competition_identity", "theme_support_ex_self", "same_height_rivals",
    ]
    return result[columns]


def run(
    distances_path: str | Path,
    reset_path: str | Path,
    reset_report_path: str | Path,
    output_dir: str | Path,
) -> dict:
    distances_path, reset_path, reset_report_path = map(
        Path, (distances_path, reset_path, reset_report_path))
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable V193 archive")
    distances = pd.read_csv(distances_path, dtype={"code": str})
    reset = pd.read_csv(reset_path, dtype={"code": str})
    reset_report = json.loads(reset_report_path.read_text(encoding="utf-8"))
    result = apply_reset_evidence(distances, reset)
    signal_dates = result["signal_date"].astype(str).unique().tolist()
    if signal_dates != [str(reset_report.get("signal_date"))]:
        raise ValueError("reset report and distance signal dates differ")
    available_at = str(reset_report.get("generated_at_utc") or "")
    target.mkdir(parents=True, exist_ok=False)
    result_path = target / "candidate_partially_reconciled_distances.csv"
    compatible_path = target / "v171_partial_reset_evidence.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    v171_compat(result, available_at).to_csv(
        compatible_path, index=False, encoding="utf-8-sig")
    complete = result["official_reset_evidence"].fillna(False).astype(bool)
    report = {
        "version": VERSION,
        "status": "DIAGNOSTIC_ONLY",
        "signal_date": signal_dates[0],
        "candidate_count": int(len(result)),
        "official_reset_evidence_count": int(complete.sum()),
        "official_reset_evidence_missing_count": int((~complete).sum()),
        "official_regulatory_distance_count": 0,
        "verified_benchmark_count": int(result["benchmark_status"].astype(str).str.lower().isin(
            {"verified", "verified_official", "official_reconciled"}).sum()),
        "corporate_action_reconciled_count": int(result[
            "corporate_action_status"].astype(str).str.lower().isin(
                {"verified", "verified_official", "official_reconciled"}).sum()),
        "candidate_reranking_allowed": False,
        "buy_signal": False,
        "entry_authorized": False,
        "priority_candidates": result[result["code"].isin({"000993", "605058"})][[
            "code", "name", "reset_status", "reset_evidence_detail",
            "official_reset_evidence", "regulatory_distance_status",
            "distance_100_pp", "distance_200_pp",
        ]].replace({np.nan: None}).to_dict("records"),
        "governance": {
            "verified_reset_alone_makes_distance_official": False,
            "partial_company_index_absence_counts_as_verified": False,
            "distance_may_rank_candidates": False,
            "distance_may_authorize_entry": False,
        },
        "interpretation": (
            "The reset blocker is removed only for candidates with complete SSE "
            "exchange-session coverage. Benchmark authority, corporate actions, "
            "historical price-limit eligibility, theme identity, and competition "
            "remain unresolved, so all distances stay diagnostic-only."
        ),
        "source_hashes": {
            str(distances_path.resolve()): _sha256(distances_path),
            str(reset_path.resolve()): _sha256(reset_path),
            str(reset_report_path.resolve()): _sha256(reset_report_path),
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
        },
        "files": {
            result_path.name: {"path": str(result_path.resolve()), "sha256": _sha256(result_path)},
            compatible_path.name: {"path": str(compatible_path.resolve()), "sha256": _sha256(compatible_path)},
        },
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", required=True)
    parser.add_argument("--reset-evidence", required=True)
    parser.add_argument("--reset-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.distances, args.reset_evidence, args.reset_report, args.output,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
