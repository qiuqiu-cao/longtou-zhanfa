"""Archive classification-index history with bounded network requests."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

SYMBOLS = {"sh000002": "上证A股指数", "sz399107": "深证A股指数"}


def collect(output, start, end):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"collected_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "requested_start": start, "requested_end": end, "indexes": {},
              "status": "DIAGNOSTIC_ONLY", "official_reconciled": False}
    for symbol, name in SYMBOLS.items():
        record = {"name": name, "errors": [], "status": "unavailable"}
        report["indexes"][symbol] = record
        for provider in ("eastmoney", "tencent"):
            try:
                if provider == "eastmoney":
                    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                    params = {"secid": ("1." if symbol.startswith("sh") else "0.") + symbol[2:],
                              "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57",
                              "klt": "101", "fqt": "0", "beg": start.replace("-", ""), "end": end.replace("-", "")}
                else:
                    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                    params = {"param": f"{symbol},day,{start},{end},1000,qfq"}
                response = requests.get(url, params=params, timeout=(8, 20))
                response.raise_for_status()
                raw = output / f"{symbol}_{provider}.json"
                raw.write_bytes(response.content)
                payload = response.json()
                if provider == "eastmoney":
                    data = payload.get("data") or {}
                    rows = [line.split(",") for line in data.get("klines", [])]
                    frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"])
                else:
                    data = payload.get("data", {}).get(symbol, {})
                    rows = data.get("day") or data.get("qfqday") or []
                    frame = pd.DataFrame([r[:6] for r in rows], columns=["date", "open", "close", "high", "low", "volume"])
                if frame.empty:
                    raise ValueError("provider returned no rows")
                frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
                frame = frame[frame.date.between(start, end)].sort_values("date")
                if frame.empty or frame.date.duplicated().any():
                    raise ValueError("empty or duplicate dates")
                frame["close"] = pd.to_numeric(frame.close, errors="raise")
                if not frame.close.gt(0).all() or not frame.close.map(math.isfinite).all():
                    raise ValueError("invalid close")
                frame["benchmark_return_pct"] = frame.close.pct_change(fill_method=None) * 100
                frame["symbol"], frame["source"] = symbol, provider
                frame["collected_at"] = report["collected_at"]
                frame["data_status"] = "provider_observed_not_official_reconciled"
                target = output / f"{symbol}.csv"
                frame.to_csv(target, index=False, encoding="utf-8-sig")
                record.update(status="provider_observed", provider=provider, rows=len(frame),
                              first_date=frame.date.min(), last_date=frame.date.max(),
                              raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                              csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                              calendar_completeness_verified=False,
                              request_url=response.url)
                break
            except Exception as exc:
                record["errors"].append({"provider": provider, "error": str(exc)[:500]})
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2023-11-01")
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.output, args.start, args.end), ensure_ascii=False, indent=2))
