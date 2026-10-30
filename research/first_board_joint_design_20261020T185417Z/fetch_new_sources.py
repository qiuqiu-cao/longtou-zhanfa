"""One bounded read of distinct public sources before any model or target analysis."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import requests

OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
RAW.mkdir(exist_ok=True)
def now(): return datetime.now(timezone.utc).isoformat()

urls = {
    "kpl_documentation": "https://raw.githubusercontent.com/LowellLee/kpl/master/" + quote("KPL接口.md"),
    "alpha_tree": "https://api.github.com/repos/Shadowell/Alpha/git/trees/HEAD?recursive=1",
    "alpha_readme": "https://api.github.com/repos/Shadowell/Alpha/readme",
    "kpl_one_historical_day": "https://apphis.longhuvip.com/w1/api/index.php?Order=0&st=1000&a=DailyLimitPerformance&c=HisHomeDingPan&PhoneOSNew=1&VerSion=5.7.0.12&Index=0&PidType=1&apiv=w31&Type=4&Day=2024-10-15",
}
protocol = dict(at=now(), previous_goal_turn="PROGRESS:Gemini4 actually replayed completeCSV audit;now user explicitly requests joint training design before fitting", task="Joint Codex/Gemini4 discussion and independent primary-source inspection;only novel public historical sealing endpoint gets one bounded probe", fits=0,
    source_policy="No auth credential,real deviceID,login/paywall bypass,retry loops,installation or third-party code execution. Retain raw bytes and real clocks. No S+1 outcome analysis during data-source qualification.",
    sources=urls, forbidden_claims="Repository features are not validated performance;forum search snippets are not full original text;source name is not field verification",
    stop="Each source one request,15second timeout;non200/access denial ends that source. No model arm until joint proposal,distinct information qualification and fixed protocol are complete.")
p = OUT / "SOURCE_PROTOCOL_BEFORE_REQUESTS.json"
with p.open("x", encoding="utf-8") as f: json.dump(protocol, f, ensure_ascii=False, indent=2)

def fetch(item):
    name, url = item
    receipt = dict(name=name, url=url, requested_at=now(), attempts=1)
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent":"LongtouResearch/1.0 (public read-only source qualification)", "Accept":"application/vnd.github+json" if "api.github.com" in url else "*/*"})
        data = response.content
        path = RAW / (name + ".response")
        path.write_bytes(data)
        receipt.update(received_at=now(), http_status=response.status_code, raw_file=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), content_type=response.headers.get("Content-Type"))
        if name == "kpl_one_historical_day" and response.status_code == 200:
            try:
                payload = response.json()
                receipt["json_root_keys"] = list(payload) if isinstance(payload,dict) else type(payload).__name__
                receipt["request_date_not_response_date_certification"] = True
            except Exception as err: receipt["parse_error"] = type(err).__name__
    except Exception as err:
        receipt.update(received_at=now(), error=type(err).__name__, detail=str(err)[:500], no_automatic_retry=True)
    return receipt

with ThreadPoolExecutor(max_workers=4) as pool: receipts = list(pool.map(fetch, urls.items()))
(OUT / "SOURCE_RECEIPTS.json").write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding="utf-8")
for r in receipts:
    print(json.dumps({key:value for key,value in r.items() if key not in ("url","raw_file")},ensure_ascii=True))
