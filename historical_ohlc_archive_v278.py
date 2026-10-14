"""Bounded retrospective, unadjusted Tencent day-bar archive for held-out pools."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
POOL = ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
START, END = "2026-08-20", "2026-09-30"


def parse(payload, symbol):
    block = (payload.get("data") or {}).get(symbol, {})
    values = block.get("day")
    if not isinstance(values, list) or not values:
        raise ValueError("unadjusted day field missing; adjusted bars not substituted")
    if any(not isinstance(r, list) or len(r) < 6 for r in values):
        raise ValueError("invalid day-bar schema")
    f = pd.DataFrame([r[:6] for r in values], columns=["date", "open", "close", "high", "low", "volume"])
    if f.date.duplicated().any() or not f.date.is_monotonic_increasing:
        raise ValueError("duplicate or unordered dates")
    f = f[f.date.between(START, END)].copy()
    for key in ["open", "close", "high", "low", "volume"]:
        f[key] = pd.to_numeric(f[key], errors="raise")
    valid = (f.low.gt(0) & f.high.ge(f.low) & f.open.between(f.low, f.high)
             & f.close.between(f.low, f.high) & f.volume.ge(0))
    if f.empty or not valid.all():
        raise ValueError("invalid OHLC bounds")
    f.insert(0, "symbol", symbol)
    return f


def verify_archive(directory):
    """Reproduce normalized bars from hash-bound raw responses before use."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report["source_pool_sha256"] != hashlib.sha256(POOL.read_bytes()).hexdigest():
        raise ValueError("daily archive pool universe mismatch")
    frames, seen = [], set()
    for entry in report["audit"]:
        symbol = entry["symbol"]
        if symbol in seen:
            raise ValueError("duplicate archived symbol")
        seen.add(symbol)
        if entry["status"] != "UNADJUSTED_DAY_AVAILABLE":
            continue
        payload = (directory / "raw" / (symbol+".json")).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("raw daily archive hash mismatch")
        frames.append(parse(json.loads(payload),symbol))
    observed = pd.read_csv(directory / "unadjusted_daily.csv")
    reconstructed = pd.concat(frames,ignore_index=True)
    keys = ["symbol","date"]
    pd.testing.assert_frame_equal(observed.sort_values(keys).reset_index(drop=True),
        reconstructed.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-10)
    return observed


def run(output, prior=None):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    raw_dir = target / "raw"; raw_dir.mkdir()
    pools = pd.read_csv(POOL, dtype={"code": str})
    codes = sorted(pools.loc[pools.boards.ge(2), "code"].unique())
    symbols = [("sh" if c.startswith("6") else "sz")+c for c in codes] + ["sh000002", "sz399107"]
    reused = {}
    prior_report = None
    if prior:
        prior = Path(prior)
        prior_report = json.loads((prior / "report.json").read_text(encoding="utf-8"))
        if prior_report["source_pool_sha256"] != hashlib.sha256(POOL.read_bytes()).hexdigest():
            raise ValueError("prior archive uses another pool universe")
        for entry in prior_report["audit"]:
            if entry.get("status") == "UNADJUSTED_DAY_AVAILABLE":
                payload = (prior / "raw" / (entry["symbol"]+".json")).read_bytes()
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError("prior raw archive hash mismatch")
                frame = parse(json.loads(payload), entry["symbol"])
                (raw_dir / (entry["symbol"]+".json")).write_bytes(payload)
                entry = dict(entry)
                if "timestamp_semantics" not in entry:
                    entry["requested_at_utc"] = entry.pop("captured_at_utc", None)
                    entry["captured_at_utc"] = None
                    entry["timestamp_semantics"] = "LEGACY_REQUEST_START_ONLY_RECEIPT_TIME_UNMEASURED"
                reused[entry["symbol"]] = (entry, frame)

    def fetch(symbol):
        captured = datetime.now(timezone.utc).isoformat()
        params = {"param": f"{symbol},day,{START},{END},100,"}
        entry = {"symbol": symbol, "requested_at_utc": captured, "captured_at_utc": None, "source": URL,
                 "parameters": params, "evidence_status": "RETROSPECTIVE_RECEIPT"}
        try:
            response = requests.get(URL, params=params, timeout=(5, 8))
            entry["http_status"] = response.status_code
            response.raise_for_status()
            payload = response.content
            entry["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
            entry["timestamp_semantics"] = "RESPONSE_RECEIVED"
            (raw_dir / (symbol+".json")).write_bytes(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            frame = parse(response.json(), symbol)
            entry.update(status="UNADJUSTED_DAY_AVAILABLE", rows=len(frame),
                         first_date=frame.date.min(), last_date=frame.date.max())
            return entry, frame
        except (requests.RequestException, ValueError, KeyError) as error:
            entry.update(status="UNAVAILABLE", error=f"{type(error).__name__}: {str(error)[:220]}")
            return entry, None

    with ThreadPoolExecutor(max_workers=6) as executor:
        fetched = dict(zip([s for s in symbols if s not in reused],
                           executor.map(fetch, [s for s in symbols if s not in reused])))
    results = [reused[s] if s in reused else fetched[s] for s in symbols]
    frames = [f for _, f in results if f is not None]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(target / "unadjusted_daily.csv", index=False, encoding="utf-8-sig")
    report = {"version": "v278-unadjusted-day-archive-2", "source_pool_sha256": hashlib.sha256(POOL.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "requested_symbols": len(symbols), "available_symbols": len(frames),
              "requested_date_range": [START, END], "audit": [a for a, _ in results],
              "status": "RETROSPECTIVE_RESEARCH_ONLY", "forward_evidence": False,
              "minute_path_or_actual_fill_available": False}
    if frames:
        report["daily_sha256"] = hashlib.sha256((target / "unadjusted_daily.csv").read_bytes()).hexdigest()
    if prior_report is not None:
        report["prior_archive"] = {"path": str(prior.resolve()),
            "manifest_sha256": hashlib.sha256((prior / "report.json").read_bytes()).hexdigest(),
            "reused_symbols": len(reused), "new_requests": len(symbols)-len(reused)}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: report[k] for k in ["status", "requested_symbols", "available_symbols"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior", help="reuse hash-verified successes; retry only previously unavailable symbols once")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.prior)))
