"""Profile raw evidence availability, not the existence of derived emotion labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


REQUIREMENTS = {
    "height_feedback": ["promotion_1to2", "promotion_2to3", "promotion_3to4", "space_feedback"],
    "loss_transmission": ["previous_limit_tail_loss", "failed_tail_loss", "down_limit_count"],
    "intraday_absorption": ["first_pullback_volume_ratio", "vwap_reclaim_at", "reclaim_volume_ratio"],
    "specific_theme_sync": ["verified_theme_sync_rate", "theme_roster_available_at"],
    "repair_close_quality": ["repair_first_confirmed_at", "repair_close_retention", "repair_volume_support"],
}


def profile(frame):
    dimensions = {}
    for dimension, fields in REQUIREMENTS.items():
        coverage = {}
        jointly_known = pd.Series(True, index=frame.index)
        for field in fields:
            present = frame[field].notna() if field in frame else pd.Series(False, index=frame.index)
            if field in frame:
                present &= frame[field].astype(str).str.strip().ne("")
            jointly_known &= present
            coverage[field] = {"column_exists": field in frame, "nonmissing": int(present.sum()),
                               "missing": int((~present).sum())}
        dimensions[dimension] = {"fields": coverage, "jointly_nonmissing": int(jointly_known.sum()),
                                  "source_verified_by_this_audit": False}
    return {"status": "DATA_COVERAGE_ONLY", "rows": len(frame), "dimensions": dimensions,
            "date_start": str(frame.date.min()) if "date" in frame and len(frame) else None,
            "date_end": str(frame.date.max()) if "date" in frame and len(frame) else None,
            "data_status_counts": frame.data_status.fillna("MISSING").value_counts().to_dict()
                                  if "data_status" in frame else {},
            "warning": "Nonmissing does not prove source accuracy, point-in-time availability, or complete universe.",
            "substitution_of_eod_for_intraday_allowed": False, "deployment_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile(pd.read_csv(args.input))
    result["input"] = str(args.input.resolve())
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"rows": result["rows"], "dimensions": {
        k: v["jointly_nonmissing"] for k, v in result["dimensions"].items()}}, ensure_ascii=False))
