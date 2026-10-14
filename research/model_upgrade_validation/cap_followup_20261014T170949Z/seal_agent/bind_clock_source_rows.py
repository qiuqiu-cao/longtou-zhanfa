"""Bind exact original CSV line excerpts for the26 contract-rejected clocks."""
from pathlib import Path
import csv
import hashlib
import io
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data/v275/reported_seal_archive_20261002_v2/reported_seal_rows.csv"

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def source_lookup(path):
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    result = {}
    previous_line = 1
    for index, row in enumerate(reader):
        start, end = starts[previous_line], starts[reader.line_num]
        code = row.get("代码", row.get("code"))
        result[str(code).zfill(6)] = {"row":row,"original_data_row_index0":index,"physical_line_start1":previous_line+1,"physical_line_end1":reader.line_num,"raw_byte_start0":start,"raw_byte_end0_exclusive":end,"exact_row_text":data[start:end].decode("utf-8"),"exact_row_sha256":sha_bytes(data[start:end])}
        previous_line = reader.line_num
    return {"source_sha256":sha_bytes(data),"source_encoding":"UTF-8 BOM if present; byte offsets include BOM","lookup":result}

rows = pd.read_csv(ARCHIVE,dtype={"code":str})
bad = rows[rows.boards.ge(2) & rows.reported_seal_status.ne("PROVIDER_REPORTED_VALID_CLOCKS")]
sources = {}
bindings = []
for r in bad.itertuples(index=False):
    path = ROOT / r.source_path
    if str(path) not in sources:
        sources[str(path)] = source_lookup(path)
    src = sources[str(path)]
    raw = src["lookup"][r.code]
    assert src["source_sha256"] == r.source_sha256
    assert int(raw["row"]["首次封板时间"]) == r.first_seal
    assert int(raw["row"]["最后封板时间"]) == r.last_seal
    manifest = path.parent / "manifest.json"
    item = {"candidate_id":f"{r.date}:{r.code}","source_path":str(path),"source_sha256":src["source_sha256"],"source_encoding":src["source_encoding"],"manifest_path":str(manifest),"manifest_sha256":sha_bytes(manifest.read_bytes()),"archive_path":str(ARCHIVE),"archive_sha256":sha_bytes(ARCHIVE.read_bytes()),"original_reported_seal_status":r.reported_seal_status,"original_acceptance_flag":False,"original_reported_first_seconds":None,"original_reported_last_seconds":None if pd.isna(r.reported_last_seconds) else r.reported_last_seconds,"original_first_seal_raw_field":raw["row"]["首次封板时间"],"original_last_seal_raw_field":raw["row"]["最后封板时间"],"original_bomb_count_raw_field":raw["row"]["炸板次数"],"raw_first_clock_hhmmss":str(int(r.first_seal)).zfill(6),"raw_last_clock_hhmmss":str(int(r.last_seal)).zfill(6),"captured_at_original":r.captured_at,"provider_event_vs_receipt_timestamp_semantics":"UNKNOWN","original_flags_and_training_unchanged":True,**{k:v for k,v in raw.items() if k != "row"}}
    bindings.append(item)
assert len(bindings)==26
(OUT / "all26_original_clock_source_rows.json").write_text(json.dumps({"scope":"Only26 relay rows excluded by original exact09:25:00 clock admission; all raw source excerpts/flags preserved; no new data or qualification promotion","source_rows":bindings},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({"bound_rows":len(bindings),"unique_sources":len(sources),"raw_hashes_verified":True,"output":str(OUT / "all26_original_clock_source_rows.json")},ensure_ascii=False))
