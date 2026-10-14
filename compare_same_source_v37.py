"""Compare archived provider daily and intraday OHLC without repairing prices."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from audit_historical_5min_v33 import inspect


def load_archive(folder):
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    result = {}
    for case in manifest["cases"]:
        if case["error_code"] != "0":
            continue
        path = folder/(case["code"]+"_"+case["date"]+".json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["raw_sha256"]:
            raise ValueError("archive hash mismatch")
        raw = json.loads(path.read_text(encoding="utf-8"))
        result[(case["code"], case["date"])] = raw
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--minute", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    daily, minute = load_archive(args.daily), load_archive(args.minute)
    rows = []
    for key, raw in minute.items():
        comparator = daily.get(key)
        if comparator is None or len(comparator["rows"]) != 1:
            rows.append(dict(code=key[0], date=key[1], status="MISSING_DAILY"))
            continue
        if comparator["frequency"] != "d" or comparator["adjustflag"] != "3":
            raise ValueError("unadjusted daily comparator required")
        values = dict(zip(comparator["fields"], comparator["rows"][0]))
        if values["date"] != key[1] or values["code"] != key[0]:
            raise ValueError("daily identity mismatch")
        result = inspect(raw, values)
        result.pop("opening_bars", None)
        result["daily_ohlc"] = {k: values[k] for k in ("open", "high", "low", "close")}
        rows.append(result)
    report = dict(status="DIAGNOSTIC_ONLY", rows=rows,
                  counts=dict(Counter(r["status"] for r in rows)),
                  conclusion="Same-provider comparison only; disagreement does not identify root cause or prove fills.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
