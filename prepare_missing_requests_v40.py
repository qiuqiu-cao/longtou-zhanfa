"""Derive missing requests from the locked cohort and hash-checked archives."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort, dtype={"code": str})
    requested = list(cohort[["provider_code", "entry_date"]].itertuples(index=False, name=None))
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate locked requests")
    returned = set()
    for root in args.archive_root:
        for manifest_path in root.glob("minute_*/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for case in manifest["cases"]:
                if case["error_code"] != "0" or case["rows"] != 48:
                    continue
                path = manifest_path.parent/(case["code"]+"_"+case["date"]+".json")
                if hashlib.sha256(path.read_bytes()).hexdigest() != case["raw_sha256"]:
                    raise ValueError("archive hash mismatch")
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("error_code") != "0" or len(raw.get("rows", [])) != 48:
                    raise ValueError("raw record count or status disagrees with manifest")
                if raw["frequency"] != "5" or raw["code"] != case["code"] or raw["requested_date"] != case["date"]:
                    raise ValueError("archive identity mismatch")
                returned.add((case["code"], case["date"]))
    missing = [list(key) for key in requested if key not in returned]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(missing, handle, indent=2)
    print(json.dumps(dict(locked=len(requested), returned=len(set(requested)&returned), missing=len(missing),
                         note="returned means 48 provider rows, not validated prices or fills")))


if __name__ == "__main__":
    main()
