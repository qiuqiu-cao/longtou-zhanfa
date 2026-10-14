"""Bounded primary provider probes discovered in AKShare's public source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/v299/attention_source_probe_20261002_v1"
SOURCE = "https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock/stock_hot_rank_em.py"
KINDS = {"intraday": "getCurrentList", "profiles": "getHisProfileList",
    "keywords": "getHotStockRankList", "related": "getFollowStockRankList"}
APP = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38"}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records = []
    calls = [("discovery_source", "GET", SOURCE, None)]
    for kind, endpoint in KINDS.items():
        payload = {**APP, "srcSecurityCode": "SH600825"}
        if kind in {"intraday", "profiles"}:
            payload["marketType"] = ""
        if kind == "profiles":
            payload["yearType"] = "5"
        calls.append((kind, "POST", "https://emappdata.eastmoney.com/stockrank/" + endpoint, payload))
    for kind, method, url, payload in calls:
        record = {"kind": kind, "method": method, "url": url, "query": payload, "requested_at": now()}
        try:
            response = requests.request(method, url, json=payload, timeout=(5, 9))
            path = OUT / (kind + ".response")
            with path.open("xb") as handle:
                handle.write(response.content)
            record.update(available_at=now(), http_status=response.status_code, raw_path=str(path.resolve()),
                raw_sha256=hashlib.sha256(response.content).hexdigest(), final_url=response.url)
            response.raise_for_status()
            if method == "POST":
                value = response.json(); data = value.get("data")
                record.update(status="RAW_JSON_RECEIVED", response_keys=list(value),
                    rows=len(data) if isinstance(data, list) else None,
                    first=data[:1] if isinstance(data, list) else data,
                    last=data[-1:] if isinstance(data, list) else None)
            else:
                record["status"] = "PRIMARY_SOURCE_RECEIVED"
        except Exception as exc:
            record.update(status="UNAVAILABLE", available_at=now(), error=str(exc)[:500])
        records.append(record)
    report = {"recorded_at": now(), "requests": records,
        "source_hashes": {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "original_forecast_modified": False, "forum_consensus_verified": False}
    with (OUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(records, ensure_ascii=True))


if __name__ == "__main__":
    main()
