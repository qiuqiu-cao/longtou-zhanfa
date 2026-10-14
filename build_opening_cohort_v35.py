"""Build all archived same-date candidates, without reading future outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

DATES = {"2024-01-12": "2024-01-15", "2024-01-16": "2024-01-17",
         "2024-12-26": "2024-12-27", "2024-12-27": "2024-12-30"}


def select(frame, dates=None):
    dates = DATES if dates is None else dates
    required = ["code", "signal_date", "boards", "start_known"]
    selected = frame.loc[frame.signal_date.isin(dates), required].copy()
    if selected.duplicated(["code", "signal_date"]).any():
        raise ValueError("duplicate candidate identity")
    selected["entry_date"] = selected.signal_date.map(dates)
    selected["provider_code"] = selected.code.map(lambda c: ("sh." if c.startswith("6") else "sz.")+c)
    return selected.sort_values(["entry_date", "code"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = select(pd.read_csv(args.candidates, dtype={"code": str}))
    args.output.mkdir(parents=True, exist_ok=False)
    data.to_csv(args.output/"cohort.csv", index=False)
    for day, group in data.groupby("entry_date"):
        (args.output/(day+"_requests.json")).write_text(json.dumps(group[["provider_code", "entry_date"]].values.tolist(), indent=2), encoding="utf-8")
    report = dict(status="DIAGNOSTIC_ONLY", rows=len(data),
                  counts=data.groupby("entry_date").size().to_dict(),
                  source_sha256=hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
                  selection="all archived candidates on four case-linked signal dates; no future outcomes read",
                  limitations=["dates selected from known winners: NOT random or blind validation",
                               "archived historical universe remains subject to survivorship/status error",
                               "unknown episode starts retained and flagged; no 5-board ceiling"])
    (args.output/"manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
