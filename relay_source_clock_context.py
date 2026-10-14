"""Add clock semantics to immutable reported archives without widening validity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from reported_seal_archive_v275 import clock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock_context(value):
    text = "" if value is None else str(value).strip()
    if text.lower() in ("", "nan", "none", "nat"):
        return {"raw": text, "structural_seconds": None, "semantics": "MISSING"}
    if re.fullmatch(r"\d{1,6}\.0+", text):
        text = text.split(".")[0]
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
        parts = text.split(":")
        text = parts[0].zfill(2)+parts[1]+parts[2]
    if not re.fullmatch(r"\d{1,6}", text):
        return {"raw": str(value), "structural_seconds": None, "semantics": "MALFORMED_OR_UNSUPPORTED_PRECISION"}
    text = text.zfill(6)
    h, m, s = int(text[:2]), int(text[2:4]), int(text[4:])
    if h > 23 or m > 59 or s > 59:
        return {"raw": str(value), "structural_seconds": None, "semantics": "MALFORMED_CLOCK"}
    seconds = h*3600+m*60+s
    if 9*3600+25*60 < seconds < 9*3600+30*60:
        semantics = "AUCTION_ADJACENT_UNRESOLVED_NOT_CONTRACT_ACCEPTED"
    elif pd.notna(clock(value)):
        semantics = "ORIGINAL_CONTRACT_TIME_ACCEPTED_NO_QUEUE_OR_FILL_PROOF"
    else:
        semantics = "OUTSIDE_ORIGINAL_CONTRACT_SESSION"
    return {"raw": str(value), "structural_seconds": seconds, "semantics": semantics}


def annotate(frame):
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("complete archive must have unique date/code rows")
    result = frame.copy()
    original_columns = list(frame.columns)
    for field in ("first_seal", "last_seal"):
        contexts = [clock_context(value) for value in frame[field]]
        result[field+"_structural_seconds_context"] = [r["structural_seconds"] for r in contexts]
        result[field+"_semantics_context"] = [r["semantics"] for r in contexts]
    result["reported_clock_adjacent_unresolved"] = (
        result.first_seal_semantics_context.str.startswith("AUCTION_ADJACENT")
        | result.last_seal_semantics_context.str.startswith("AUCTION_ADJACENT"))
    result["context_authorizes_new_validity_or_training"] = False
    pd.testing.assert_frame_equal(result[original_columns], frame)
    return result


def build(archive, output):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("immutable clock semantics output exists")
    original_hash = sha(archive)
    frame = pd.read_csv(archive, dtype={"code": str}, float_precision="round_trip")
    annotated = annotate(frame)
    relay = annotated[annotated.boards.ge(2)]
    output.mkdir(parents=True, exist_ok=False)
    annotated.to_csv(output / "full_archive_clock_context.csv", index=False, encoding="utf-8-sig")
    unresolved = relay[relay.reported_clock_adjacent_unresolved]
    unresolved.to_csv(output / "complete_relay_adjacent_unknown.csv", index=False, encoding="utf-8-sig")
    report = {"formed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ADDITIVE_CLOCK_SEMANTICS_NOT_VALIDITY_CHANGE", "archive_rows": len(frame), "relay_rows": len(relay),
        "original_valid_relay_rows": int(relay.reported_seal_status.eq("PROVIDER_REPORTED_VALID_CLOCKS").sum()),
        "adjacent_unresolved_relay_rows": len(unresolved),
        "unknown_raw_first_clock_counts": unresolved.first_seal.astype(str).value_counts().to_dict(),
        "all_original_columns_unchanged": True, "fit_count": 0, "new_mature_forward_samples": 0,
        "training_authorized": False, "snapshot_or_queue_stability_verified": False,
        "interpretation": "A well-formed09:25<clock<09:30 rejected by the old gate is a provider timestamp semantics unknown, not proof of malformed data or actual sealed auction",
        "sources": {str(archive): original_hash, str(Path(__file__).resolve()): sha(__file__),
                    str(Path(__file__).with_name("reported_seal_archive_v275.py")): sha(Path(__file__).with_name("reported_seal_archive_v275.py"))}}
    assert sha(archive) == original_hash
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output), ensure_ascii=True))
