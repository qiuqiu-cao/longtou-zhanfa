"""Audit fixed opening-absorption categories against B+2 close-mark proxies."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(frame):
    values = pd.to_numeric(frame.b2_net_close_mark_LABEL_ONLY, errors="coerce").dropna()
    return {"rows_with_feature": int(len(frame)), "b2_proxy_known": int(len(values)),
            "b2_mean_net_close_mark": float(values.mean()) if len(values) else None,
            "b2_positive_rate": float(values.gt(0).mean()) if len(values) else None,
            "b2_large_loss_rate": float(values.le(-.07).mean()) if len(values) else None,
            "actual_verified_fills": int(frame.actual_entry_verified.fillna(False).astype(bool).sum())
                                     if "actual_entry_verified" in frame else 0}


def audit(cohort, features, outcomes):
    for frame in (cohort, features, outcomes):
        frame["code"] = frame.code.astype(str).str.zfill(6)
    if cohort.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate cohort key")
    if features.duplicated(["code", "date"]).any() or outcomes.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate evidence or outcome key")
    joined = cohort.merge(features, left_on=["code", "entry_date"], right_on=["code", "date"],
                          how="left", validate="one_to_one", indicator="feature_join")
    joined = joined.merge(outcomes, on=["code", "signal_date"], how="left", validate="one_to_one")
    usable = joined.loc[joined.feature_usable.fillna(False)].copy()
    usable["absorption_path"] = "NEVER_BELOW_CUM_VWAP_BY_1000"
    below = usable.first_below_cum_vwap_at.notna()
    usable.loc[below, "absorption_path"] = "BELOW_NOT_RECLAIMED_BY_1000"
    usable.loc[below & usable.first_reclaim_cum_vwap_at.notna(), "absorption_path"] = "BELOW_THEN_RECLAIMED_BY_1000"
    groups = {}
    for board_group, part in usable.groupby("board_group"):
        groups[board_group] = {
            "coverage": {"cohort_rows": int(cohort.board_group.eq(board_group).sum()),
                         "usable_feature_rows": int(len(part)),
                         "coverage_rate": float(len(part)/cohort.board_group.eq(board_group).sum())},
            "above_vwap_1000": {str(bool(key)): summary(group)
                                for key, group in part.groupby("close_1000_above_cum_vwap")},
            "absorption_path": {str(key): summary(group)
                                for key, group in part.groupby("absorption_path")},
        }
    return {"version": "v138-opening-absorption-value-audit-1", "status": "DIAGNOSTIC_ONLY",
            "cohort_rows": int(len(cohort)), "feature_joined_rows": int(joined.feature_join.eq("both").sum()),
            "usable_feature_rows": int(len(usable)), "groups": groups,
            "target": "B+2 net close mark after hypothetical next-session daily price-path opportunity",
            "actual_verified_fills": int(usable.actual_entry_verified.fillna(False).astype(bool).sum()),
            "complete_trade_exit_tested": False, "threshold_tuning": False,
            "deployment_authorized": False,
            "limitations": ["historical minute coverage is incomplete and first-board coverage is selected",
                            "five-minute bars do not prove actual entry or queue position",
                            "B+2 close mark is not a frozen complete-trade exit"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(pd.read_csv(args.cohort, dtype={"code": str}),
                   pd.read_csv(args.features, dtype={"code": str}),
                   pd.read_csv(args.outcomes, dtype={"code": str}))
    report["source_hashes"] = {str(path.resolve()): sha256(path)
                               for path in (args.cohort, args.features, args.outcomes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"cohort_rows": report["cohort_rows"],
                      "usable_feature_rows": report["usable_feature_rows"],
                      "groups": report["groups"]}, ensure_ascii=False, indent=2))
