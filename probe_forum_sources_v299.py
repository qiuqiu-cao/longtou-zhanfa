"""Two bounded public forum GETs; raw bytes and actual clocks are retained."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/v299/forum_source_probe_20261002_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    records = []
    for code in ["600825", "605303"]:
        url = "https://guba.eastmoney.com/list," + code + ".html"
        record = {"code": code, "url": url, "requested_at": now()}
        try:
            response = requests.get(url, timeout=(5, 10))
            path = OUT / (code + ".response")
            with path.open("xb") as handle:
                handle.write(response.content)
            record.update(available_at=now(), http_status=response.status_code,
                final_url=response.url, raw_path=str(path.resolve()), raw_sha256=hashlib.sha256(response.content).hexdigest(),
                response_date=response.headers.get("Date"), content_type=response.headers.get("Content-Type"))
            response.raise_for_status()
            text = response.content.decode("utf-8", errors="replace")
            record.update(status="RAW_HTML_RECEIVED", bytes=len(response.content),
                variable_names=re.findall(r"(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=", text)[:50],
                title=re.search(r"<title>(.*?)</title>", text, re.S).group(1) if "<title>" in text else None)
        except Exception as exc:
            record.update(status="UNAVAILABLE", available_at=now(), error=str(exc)[:500])
        records.append(record)
    report = {"recorded_at": now(), "requests": records,
        "source_hashes": {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "current_research_only": True, "original_forecast_modified": False}
    with (OUT / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(records, ensure_ascii=True))


if __name__ == "__main__":
    main()
