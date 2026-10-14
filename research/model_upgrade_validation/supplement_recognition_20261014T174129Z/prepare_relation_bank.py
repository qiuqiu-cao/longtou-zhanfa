"""Convert the saved, audited fixture into the recognizer's explicit schema."""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import stock_judgement_review as o
import supplement_relation_review as r

OUT = Path(__file__).resolve().parent
fixture_path = OUT/"evidence_agent"/"relation_fixture.json"
compact_path = OUT/"evidence_agent"/"compact_relation_bank.json"
fixture,compact = o.read(fixture_path),o.read(compact_path)
proof = r._proof(fixture["source_bindings"]) | r._proof(compact["source_bindings"])
proof |= o.bind(fixture_path) | o.bind(compact_path) | o.bind(__file__)
calendar = ROOT/"data"/"v298"/"exchange_calendar_extension_20261003_v2"/"calendar.json"
proof |= o.bind(calendar)
md_path = ROOT/"results"/"sunshine_jiuyang_relation_20261014T152814Z"/"阳光九阳补涨关系与候选比较.md"
md = md_path.read_text(encoding="utf-8-sig")
proof |= o.bind(md_path)


def text_binding(excerpt,value):
    return {"path":str(md_path),"sha256":o.sha(md_path),"offset":md.index(excerpt),
            "excerpt":excerpt,"declared_value":value,
            "scope":"EXPLICIT_ARCHIVED_REVIEW_INTERPRETATION_NOT_PRIOR_CUTOFF_MARKET_CERTIFICATION"}


temporal = []
for row in fixture["temporal_facts"]:
    if "ohlc_cents" not in row:
        continue
    source = {"path":row["source_path"],"sha256":proof[row["source_path"]],"json_pointer":row["source_json_pointer"]}
    board_source = row.get("board_source")
    if row["code"] == "001318" and row["date"] == "2026-10-13":
        board_source = text_binding("| 10月13日 | 连续板中断，收17.67 | 3板8.86 | 重新首板14.03 |",1)
    temporal.append({**row,"ohlc_source":source,"board_source":board_source})

pair_source = compact["explicit_pair_chronology_binding"]
timeline = r._pointer(o.read(pair_source["path"]),pair_source["json_pointer"])
events = []
for i,item in enumerate(timeline):
    # A persisted break state is not a new break on every subsequent day.
    if item["jiuyang_limit_streak_broken"] and (i == 0 or not timeline[i-1]["jiuyang_limit_streak_broken"]):
        events.append({"code":"002242","date":item["date"],"event":"STREAK_BREAK","value":True,
            "available_at":fixture["formed_at"],"source":{**pair_source,"json_pointer":f"/timeline/{i}/jiuyang_limit_streak_broken"}})
events.append({"code":"001318","date":"2026-09-30","event":"PRIOR_FIRST_OBSERVED_BOARD","value":True,
    "available_at":fixture["formed_at"],"source":text_binding("| 9月30日 | 3板 | 未涨停 | 首板13.04 |",True)})
anchors = compact["anchors"]
anchors[0]["identity_source"] = {**pair_source,"json_pointer":"/parent"}
line = Path(anchors[1]["name_code_binding"]["path"]).read_text(encoding="utf-8-sig").splitlines()[64]
anchors[1]["identity_source"] = {**anchors[1]["name_code_binding"],"offset":Path(anchors[1]["name_code_binding"]["path"]).read_text(encoding="utf-8-sig").index(line),
    "excerpt":line,"declared_value":{"code":"605388","name":"均瑶健康"}}
bank = {"version":r.BANK_VERSION,"formed_at":o.now().isoformat(),"signal_date":fixture["signal_date"],
    "target_date":fixture["target_date"],"calendar_path":str(calendar),"anchors_are_forecast_candidates":False,
    "anchors":anchors,"sources":fixture["sources"],"claims":fixture["claims"],"edges":compact["edges"],
    "temporal_facts":temporal,"events":events,"source_bindings":proof,
    "source_scope":"ALREADY_SEEN_RETROSPECTIVE_CURRENT_CONTEXT_NOT_ORIGINAL_FORECAST_OR_TRAINING_BACKFILL",
    "original_prediction_backfill_allowed":False,"new_data_requests":0,"new_fits":0}
o.check_sources(proof)
parser = argparse.ArgumentParser()
parser.add_argument("--output",default=str(OUT/"qualified_relation_bank.json"))
args = parser.parse_args()
o.write_new(args.output,bank)
print({"pool":15,"relations":len(bank["edges"]),"observations":len(temporal),"events":len(events),"new_requests":0})
