"""Correct the saved notice parser using the first responses, preserving the failed v1."""
from datetime import date
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(r"D:\Documents\ChatGPT\龙头战法")
PARENT_ARCHIVE = ROOT / "data/v298/exchange_calendar_extension_20261003_v1"
PARENT_MANIFEST = PARENT_ARCHIVE / "source_manifest.json"
BUILDER = ROOT / "results/v298/calendar_extension_20261003_v1/build_calendar.py"
manifest_bytes = PARENT_MANIFEST.read_bytes()
manifest = json.loads(manifest_bytes)
if sha256(BUILDER.read_bytes()).hexdigest() != manifest["builder_sha256"]:
    raise ValueError("original builder changed")
spec = importlib.util.spec_from_file_location("original_calendar_builder", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def archived_fetch(item):
    key, url = item
    entry = dict(manifest["sources"][key])
    if entry["requested_url"] != url:
        raise ValueError("archived official source URL changed")
    raw = Path(entry["raw_path"])
    body = raw.read_bytes()
    if sha256(body).hexdigest() != entry["raw_sha256"]:
        raise ValueError("original official response changed")
    parser = builder.TextParser()
    parser.feed(body.decode("utf-8"))
    entry["reconstructed_at"] = builder.now()
    entry["derivation"] = {
        "parent_manifest": str(PARENT_MANIFEST), "parent_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "parent_builder": str(BUILDER), "parent_builder_sha256": manifest["builder_sha256"],
        "parent_failure": str(PARENT_ARCHIVE / "calendar_build_failure.json"),
        "correction": "Require opening section parenthesis; weekday closing parenthesis previously captured intervening narrative as a holiday name.",
        "no_new_http_request": True,
    }
    return key, entry, re.sub(r"\s+", "", "".join(parser.parts))


def corrected_schedule(text):
    matches = re.findall(r"（[一二三四五六七]）([^：（）]+)：(\d+)月(\d+)日（[^）]+）至(\d+)月(\d+)日", text)
    required = {"元旦", "春节", "清明节", "劳动节", "端午节", "中秋节", "国庆节"}
    if len(matches) != 7 or {row[0] for row in matches} != required:
        raise ValueError("official annual holiday schedule could not be completely parsed")
    return [(name, date(2026, int(month), int(day)).isoformat(),
             date(2026, int(end_month), int(end_day)).isoformat())
            for name, month, day, end_month, end_day in matches]


if __name__ == "__main__":
    builder.OUTPUT = ROOT / "data/v298/exchange_calendar_extension_20261003_v2"
    builder.__file__ = str(Path(__file__).resolve())
    builder.fetch = archived_fetch
    builder.parse_schedule = corrected_schedule
    builder.main()
