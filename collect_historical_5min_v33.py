"""Bounded anonymous BaoStock probe for four historical case-study sessions.

Raw provider records are archived as JSON, never promoted to confirmed fills.
The parent process bounds SDK calls that may otherwise wait indefinitely.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

CASES = [("sz.000017", "2024-01-15"), ("sz.000017", "2024-01-17"),
         ("sz.000759", "2024-12-27"), ("sz.000759", "2024-12-30")]
FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def worker(directory, cases=None, frequency="5"):
    started = time.monotonic()
    sys.path.insert(0, str(Path(__file__).resolve().parent / ".runtime/baostock"))
    import baostock as bs
    socket.setdefaulttimeout(8)
    login = bs.login()
    save(directory / "login_status.json", {"error_code": login.error_code, "error_msg": login.error_msg})
    if login.error_code != "0":
        return
    try:
        for code, day in (CASES if cases is None else cases):
            # Leave room for an in-flight query and logout before parent timeout.
            # Parent's hard limit remains necessary for a nonresponsive SDK.
            if time.monotonic() - started >= 35:
                save(directory / "worker_stop.json", dict(reason="GRACEFUL_TIME_BUDGET", elapsed_seconds=time.monotonic()-started))
                break
            fields = FIELDS if frequency == "5" else FIELDS.replace("time,", "")
            result = bs.query_history_k_data_plus(code, fields, start_date=day, end_date=day,
                                                  frequency=frequency, adjustflag="3")
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            save(directory / f"{code}_{day}.json", dict(
                code=code, requested_date=day, frequency=frequency, adjustflag="3",
                fields=result.fields, rows=rows, error_code=result.error_code,
                error_msg=result.error_msg, captured_at=datetime.now().astimezone().isoformat(),
                source="BaoStock anonymous history API", data_status="provider_observed_not_reconciled"))
    finally:
        bs.logout()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--frequency", choices=["5", "d"], default="5")
    parser.add_argument("--cases", type=Path, help="JSON list of [provider_code, date], selected before collection")
    args = parser.parse_args()
    cases = CASES if args.cases is None else json.loads(args.cases.read_text(encoding="utf-8"))
    import re
    if (not isinstance(cases, list) or not cases or
            any(not isinstance(c, list) and not isinstance(c, tuple) or len(c) != 2
                or not re.fullmatch(r"(?:sz|sh)\.\d{6}", str(c[0]))
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c[1])) for c in cases)):
        raise ValueError("invalid case list")
    if len({tuple(c) for c in cases}) != len(cases):
        raise ValueError("duplicate cases")
    if args.worker:
        worker(args.output, cases, args.frequency)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone().isoformat()
    try:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(args.output.resolve()), "--frequency", args.frequency]
        if args.cases is not None:
            command.extend(["--cases", str(args.cases.resolve())])
        process = subprocess.run(command,
                                 capture_output=True, timeout=55)
        state = "COMPLETED" if process.returncode == 0 else "WORKER_ERROR"
        stdout = process.stdout.decode("utf-8", errors="replace")
        stderr = process.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        state = "WORKER_TIMEOUT_TERMINATED"
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
    records = []
    for code, day in cases:
        path = args.output / f"{code}_{day}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(dict(code=code, date=day, rows=len(data["rows"]), error_code=data["error_code"],
                                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        else:
            records.append(dict(code=code, date=day, rows=None, error_code="NOT_RETURNED"))
    report = dict(status="DIAGNOSTIC_ONLY", started_at=started, completed_at=datetime.now().astimezone().isoformat(),
                  process_state=state, cases=records, worker_stdout=stdout, worker_stderr=stderr,
                  frequency=args.frequency,
                  data_complete=all(r["error_code"] == "0" and r["rows"] == (48 if args.frequency == "5" else 1) for r in records),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitations=["5-minute data is not 1-minute, auction, tick or queue data",
                               "no fill or trade label assigned; source must be reconciled",
                               "case selection and historical universe bias require separate audit"],
                  cases_sha256=None if args.cases is None else hashlib.sha256(args.cases.read_bytes()).hexdigest())
    stop_path = args.output / "worker_stop.json"
    report["graceful_stop"] = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None
    save(args.output / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
