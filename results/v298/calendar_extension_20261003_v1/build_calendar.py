"""Archive official schedules and extend a separate runtime calendar; never alter plans."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(r"D:\Documents\ChatGPT\龙头战法")
PARENT = ROOT / "data/v279/exchange_calendar_audit_20261002_v1/calendar.json"
OUTPUT = ROOT / "data/v298/exchange_calendar_extension_20261003_v1"
SOURCES = {
    "sse_annual": "https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml",
    "sse_holiday": "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml",
    "szse_annual": "https://investor.szse.cn/disclosure/notice/general/t20251222_618087.html",
    "szse_holiday": "https://investor.szse.cn/disclosure/notice/general/t20260917_622911.html",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def fetch(item):
    key, url = item
    entry = {"key": key, "requested_url": url, "started_at": now(), "attempts": 1}
    body = b""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read()
            entry.update(http_status=response.status, response_url=response.url,
                         response_headers=dict(response.headers.items()))
        entry["status"] = "RESPONSE_ARCHIVED"
    except Exception as exc:
        entry.update(status="FETCH_FAILED", error=repr(exc))
        if hasattr(exc, "read"):
            body = exc.read()
        if hasattr(exc, "code"):
            entry["http_status"] = exc.code
    entry["received_at"] = now()
    raw = OUTPUT / (key + ".raw.html")
    with raw.open("xb") as handle:
        handle.write(body)
    entry.update(raw_path=str(raw.resolve()), raw_sha256=sha256(body).hexdigest(), bytes=len(body))
    write_new(OUTPUT / (key + ".request.json"), entry)
    parser = TextParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    text = re.sub(r"\s+", "", "".join(parser.parts))
    return key, entry, text


def parse_schedule(text):
    matches = re.findall(r"[一二三四五六七]）([^：]+)：(\d+)月(\d+)日[（(][^）)]+[）)]至(?:(\d+)月)?(\d+)日", text)
    required = {"元旦", "春节", "清明节", "劳动节", "端午节", "中秋节", "国庆节"}
    if len(matches) != 7 or {row[0] for row in matches} != required:
        raise ValueError("official annual holiday schedule could not be completely parsed")
    return [(name, date(2026, int(month), int(day)).isoformat(),
             date(2026, int(end_month or month), int(end_day)).isoformat())
            for name, month, day, end_month, end_day in matches]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = now()
    parent_bytes = PARENT.read_bytes()
    parent_hash = sha256(parent_bytes).hexdigest()
    parent = json.loads(parent_bytes)
    with ThreadPoolExecutor(max_workers=4) as executor:
        fetched = list(executor.map(fetch, SOURCES.items()))
    entries = {key: entry for key, entry, text in fetched}
    texts = {key: text for key, entry, text in fetched}
    manifest = {"version": "v298-official-runtime-calendar-extension-1", "started_at": started,
                "assembled_at": now(), "parent_calendar": str(PARENT), "parent_sha256": parent_hash,
                "sources": entries, "builder_path": str(Path(__file__).resolve()),
                "builder_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    write_new(OUTPUT / "source_manifest.json", manifest)
    try:
        for key in ("sse_annual", "sse_holiday"):
            if entries[key]["status"] != "RESPONSE_ARCHIVED" or entries[key].get("http_status") != 200:
                raise ValueError("required SSE official notice unavailable: " + key)
        schedule = parse_schedule(texts["sse_annual"])
        if "10月8日（星期四）起照常开市" not in texts["sse_holiday"] or "10月10日（星期六）为周末休市" not in texts["sse_holiday"]:
            raise ValueError("current official holiday notice does not confirm reopening/weekend boundary")
        crosschecks = {}
        for key in ("szse_annual", "szse_holiday"):
            if entries[key]["status"] != "RESPONSE_ARCHIVED":
                crosschecks[key] = "UNKNOWN_FETCH_FAILED_PRESERVED"
                continue
            if key == "szse_annual":
                if parse_schedule(texts[key]) != schedule:
                    raise ValueError("SSE and SZSE annual schedules conflict")
            elif "10月8日（星期四）起照常开市" not in texts[key] or "10月10日（星期六）为周末休市" not in texts[key]:
                raise ValueError("SZSE holiday notice conflicts with reopening/weekend boundary")
            crosschecks[key] = "CONSISTENT_OFFICIAL_NOTICE"
        full_2026 = []
        current = date(2026, 1, 1)
        end = date(2026, 12, 31)
        while current <= end:
            day = current.isoformat()
            if current.weekday() < 5 and not any(lo <= day <= hi for name, lo, hi in schedule):
                full_2026.append(day)
            current += timedelta(days=1)
        old_end = parent["scope"][1]
        inherited_2026 = [day for day in parent["sessions"] if "2026-01-01" <= day <= old_end]
        if inherited_2026 != [day for day in full_2026 if day <= old_end]:
            raise ValueError("new official reconstruction disagrees with inherited 2026 sessions")
        added = [day for day in full_2026 if day > old_end]
        sessions = parent["sessions"] + added
        if sessions != sorted(set(sessions)) or "2026-10-10" in sessions:
            raise ValueError("duplicate/session/weekend error")
        if sessions[sessions.index("2026-09-30") + 1] != "2026-10-08":
            raise ValueError("original exact next-session boundary changed")
        calendar = {"status": "verified", "source": parent["source"] + "; " + SOURCES["sse_holiday"],
                    "scope": [parent["scope"][0], "2026-12-31"], "sessions": sessions,
                    "verified_at": now(), "parent_calendar": str(PARENT), "parent_sha256": parent_hash,
                    "source_manifest": str((OUTPUT / "source_manifest.json").resolve()),
                    "source_manifest_sha256": sha256((OUTPUT / "source_manifest.json").read_bytes()).hexdigest(),
                    "official_2026_closures": schedule, "crosschecks": crosschecks,
                    "limitation": "Published exchange schedule only; earlier sessions inherited unchanged. Recheck new official exceptional closures before future runs; no coverage beyond 2026-12-31."}
        if sha256(PARENT.read_bytes()).hexdigest() != parent_hash:
            raise ValueError("parent changed during extension")
        write_new(OUTPUT / "calendar.json", calendar)
        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        report = {"status": "OFFICIAL_CALENDAR_EXTENDED_SEPARATELY", "completed_at": now(),
                  "calendar": str((OUTPUT / "calendar.json").resolve()), "scope": calendar["scope"],
                  "added_sessions": len(added), "first_added_session": added[0],
                  "today": today, "today_is_exchange_session": today in sessions,
                  "next_session_after_signal_20260930": "2026-10-08", "crosschecks": crosschecks,
                  "original_calendar_unchanged": True, "original_plan_modified": False,
                  "calendar_sha256": sha256((OUTPUT / "calendar.json").read_bytes()).hexdigest()}
        write_new(OUTPUT / "verification_report.json", report)
        print(json.dumps(report, ensure_ascii=False))
    except Exception as exc:
        write_new(OUTPUT / "calendar_build_failure.json", {"recorded_at": now(), "error": repr(exc),
                  "original_calendar_unchanged": sha256(PARENT.read_bytes()).hexdigest() == parent_hash})
        raise


if __name__ == "__main__":
    main()
