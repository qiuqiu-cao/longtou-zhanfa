"""Build date-clustered same-height pairwise validation rows.

Pairs are bookkeeping views of one height group, not independent samples.
All metrics and readiness counts remain clustered by signal date.  The module
does not choose a champion and refuses to validate the five-factor challenger
when either member lacks point-in-time evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v210-same-height-pairwise-validation-1"
FACTORS = [
    "turnover_absorption_quality", "theme_purity", "sector_leadership",
    "competitor_elimination", "announcement_soft_risk",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = set(["signal_date", "code", "name", "boards", "net_return_B_plus_2",
                    "big_loss_target", "five_factor_point_in_time_complete"] + FACTORS)
    if required - set(predictions):
        raise ValueError(f"prediction fields missing: {sorted(required - set(predictions))}")
    if predictions.duplicated(["signal_date", "code"]).any():
        raise ValueError("candidate keys must be unique")
    rows = predictions.copy()
    rows["code"] = rows.code.astype(str).str.zfill(6)
    rows["boards"] = pd.to_numeric(rows.boards, errors="raise").astype(int)
    rows["height_group_id"] = rows.signal_date.astype(str) + ":H" + rows.boards.astype(str)
    pairs = []
    for group_id, group in rows.groupby("height_group_id", sort=True):
        ordered = group.sort_values("code")
        for (_, left), (_, right) in itertools.combinations(ordered.iterrows(), 2):
            left_return = pd.to_numeric(pd.Series([left.net_return_B_plus_2]), errors="coerce").iloc[0]
            right_return = pd.to_numeric(pd.Series([right.net_return_B_plus_2]), errors="coerce").iloc[0]
            item = {
                "signal_date": str(left.signal_date), "height_group_id": group_id,
                "boards": int(left.boards), "group_candidate_count": int(len(group)),
                "left_code": left.code, "left_name": left.get("name"),
                "right_code": right.code, "right_name": right.get("name"),
                "B_plus_2_return_difference_left_minus_right": (
                    float(left_return - right_return)
                    if pd.notna(left_return) and pd.notna(right_return) else np.nan),
                "B_plus_2_pair_outcome_available": bool(pd.notna(left_return) and pd.notna(right_return)),
                "left_big_loss_B_plus_1": left.big_loss_target,
                "right_big_loss_B_plus_1": right.big_loss_target,
                "five_factor_pair_complete": bool(
                    left.five_factor_point_in_time_complete
                    and right.five_factor_point_in_time_complete),
            }
            for factor in FACTORS:
                lval = pd.to_numeric(pd.Series([left[factor]]), errors="coerce").iloc[0]
                rval = pd.to_numeric(pd.Series([right[factor]]), errors="coerce").iloc[0]
                item[factor + "_difference_left_minus_right"] = (
                    float(lval - rval) if pd.notna(lval) and pd.notna(rval) else np.nan)
            pairs.append(item)
    pair_frame = pd.DataFrame(pairs)
    groups = rows.groupby("height_group_id", as_index=False).agg(
        signal_date=("signal_date", "first"), boards=("boards", "first"),
        candidate_count=("code", "size"),
        five_factor_complete_candidates=("five_factor_point_in_time_complete", "sum"),
        B_plus_2_known_candidates=("net_return_B_plus_2", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
    )
    groups["complete_five_factor_group"] = groups.five_factor_complete_candidates.eq(groups.candidate_count)
    multi = groups.candidate_count.gt(1)
    evaluable = groups.complete_five_factor_group & multi
    report = {
        "version": VERSION, "status": "DIAGNOSTIC_ONLY",
        "candidate_rows": int(len(rows)), "height_groups": int(len(groups)),
        "multi_candidate_height_groups": int(multi.sum()),
        "pair_rows_not_independent_samples": int(len(pair_frame)),
        "distinct_signal_dates": int(rows.signal_date.nunique()),
        "B_plus_2_outcome_known_pairs": int(pair_frame.B_plus_2_pair_outcome_available.sum()) if len(pair_frame) else 0,
        "five_factor_complete_pairs": int(pair_frame.five_factor_pair_complete.sum()) if len(pair_frame) else 0,
        "five_factor_evaluable_groups": int(evaluable.sum()),
        "minimum_date_clusters": 100,
        "same_height_five_factor_uplift_validated": False,
        "winner_claimed": False, "entry_authorized": False,
        "governance": {
            "pairs_count_as_independent_samples": False,
            "validation_cluster": "signal_date",
            "all_same_height_candidates_retained": True,
            "fast_seal_or_zero_open_selects_champion": False,
            "missing_factor_filled_as_zero": False,
        },
    }
    return pair_frame, groups, report


def run(predictions_path, output_dir):
    source, target = Path(predictions_path), Path(output_dir)
    if target.exists():
        raise FileExistsError("refusing to overwrite immutable pairwise validation")
    pairs, groups, report = build(pd.read_csv(source, dtype={"code": str}))
    target.mkdir(parents=True)
    pair_path, group_path = target / "same_height_pairs.csv", target / "height_group_summary.csv"
    pairs.to_csv(pair_path, index=False, encoding="utf-8-sig")
    groups.to_csv(group_path, index=False, encoding="utf-8-sig")
    report["source_sha256"] = _sha256(source)
    report["artifact_hashes"] = {pair_path.name: _sha256(pair_path), group_path.name: _sha256(group_path)}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.predictions, args.output), ensure_ascii=False, indent=2))
