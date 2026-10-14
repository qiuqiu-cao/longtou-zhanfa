"""V141 forward registry: bind the frozen primary and exact candidate outcomes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

import absorption_forward_registry_v140 as old


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT/"data"/"v141"/"absorption_forward"
DEPENDENCIES = [ROOT/"absorption_forward_protocol_v139.py", ROOT/"live_decision_bridge_v1.py",
                ROOT/"public_minute_capture_v1.py", Path(__file__).resolve()]
GATES = old.GATES
EXECUTION_STATES = {"ACTUAL_FILL", "CONSERVATIVE_SIM_FILL", "NO_FILL",
                    "QUEUE_UNVERIFIABLE", "DATA_UNAVAILABLE", "NO_FROZEN_ENTRY"}


def initialize(registry):
    active, protocol_path = registry/"active_freeze.json", registry/"protocol.json"
    if active.exists():
        raise FileExistsError("active V141 freeze already exists")
    if not protocol_path.exists():
        raise FileNotFoundError("protocol.json required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_RESEARCH_ONLY":
        raise ValueError("invalid protocol")
    dependencies = [protocol_path, *DEPENDENCIES]
    manifest = {"version": "v141-absorption-forward-registry-1",
                "freeze_created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "assurance": "LOCAL_TAMPER_EVIDENT_ONLY_NO_EXTERNAL_TIMESTAMP",
                "dependency_hashes": {str(path.resolve()): old.digest(path) for path in dependencies},
                "entry_authorized": False, "actual_fill_claim": False,
                "primary_binding_required": True, "exact_candidate_settlement_required": True}
    manifest["freeze_id"] = hashlib.sha256(old.canonical(manifest).encode()).hexdigest()[:16]
    registry.mkdir(parents=True, exist_ok=True)
    sealed = registry/f"freeze_{manifest['freeze_id']}.json"
    sealed.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    active.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    old.append_event(registry/"ledger.jsonl", {"event_type":"FREEZE", "freeze_id":manifest["freeze_id"],
        "manifest_file":sealed.name, "manifest_sha256":old.digest(sealed), "entry_authorized":False})
    return manifest


def verify(registry):
    result = old.verify(registry)
    errors = list(result.get("errors", [])); settlements = set()
    for number, event in enumerate(old.read_events(registry/"ledger.jsonl"), 1):
        if event.get("event_type") != "OUTCOME_SETTLEMENT":
            continue
        trade_date = event.get("trade_date")
        if trade_date in settlements:
            errors.append(f"duplicate settlement: {trade_date}")
        settlements.add(trade_date)
        path = registry/event.get("settlement_file", "")
        if not path.exists() or old.digest(path) != event.get("settlement_sha256"):
            errors.append(f"settlement missing or changed at {number}")
    return {**result, "status":"PASS" if not errors else "FAIL", "errors":errors,
            "settlement_count":len(settlements)}


def _forward_time(manifest, decision_path, decision, recorded_at=None):
    now = (datetime.now(ZoneInfo("Asia/Shanghai")) if recorded_at is None else
           datetime.fromisoformat(str(recorded_at)).astimezone(ZoneInfo("Asia/Shanghai")))
    frozen = datetime.fromisoformat(manifest["freeze_created_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    trade, signal = str(decision.get("trade_date")), str(decision.get("signal_date"))
    modified = datetime.fromtimestamp(decision_path.stat().st_mtime, ZoneInfo("Asia/Shanghai"))
    if (now.strftime("%Y-%m-%d") != trade or now.strftime("%H:%M") < "10:00"
            or now <= frozen or signal >= trade or modified < frozen-timedelta(seconds=2)):
        raise ValueError("not a same-day post-10:00 forward decision")
    return now, signal, trade


def snapshot_decision(registry, decision_path, recorded_at=None):
    if verify(registry)["status"] != "PASS":
        raise ValueError("registry verification failed")
    manifest = json.loads((registry/"active_freeze.json").read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (decision.get("asof") != "10:00" or decision.get("direct_entry_authorized") is not False
            or decision.get("absorption_shadow_gate_live_effect") != "NONE_RESEARCH_ONLY"):
        raise ValueError("decision is not a non-authorizing 10:00 shadow record")
    now, signal_date, trade_date = _forward_time(manifest, decision_path, decision, recorded_at)
    primary = decision.get("primary_choice")
    primary_code = (str(primary.get("code")).zfill(6) if isinstance(primary, dict) and primary.get("code")
                    else str(primary).zfill(6) if isinstance(primary, str) else None)
    rows = list(decision.get("first_board_candidates", []))+list(decision.get("relay_candidates", []))
    codes = [str(row.get("code", "")).zfill(6) for row in rows]
    if not rows or len(codes) != len(set(codes)) or (primary_code and primary_code not in codes):
        raise ValueError("candidate set or primary binding invalid")
    candidates = []
    for row, code in zip(rows, codes):
        gate = row.get("absorption_shadow_gate_v139")
        if gate not in GATES:
            raise ValueError(f"missing gate: {code}")
        candidates.append({"code":code, "name":row.get("name"), "boards":row.get("boards"),
            "board_group":row.get("board_group"), "is_frozen_primary":code == primary_code,
            "minute_data_status":row.get("minute_data_status"),
            "close_to_vwap_proxy":row.get("close_to_vwap_proxy"), "shadow_gate":gate,
            "public_price_decision":row.get("public_price_decision"), "buyability":row.get("buyability"),
            "actual_fill_verified":False, "complete_trade_result":None})
    events = old.read_events(registry/"ledger.jsonl")
    if any(e.get("event_type")=="TEN_OCLOCK_SNAPSHOT" and e.get("trade_date")==trade_date for e in events):
        raise ValueError("duplicate trade-date snapshot")
    directory = registry/"snapshots"/manifest["freeze_id"]; directory.mkdir(parents=True, exist_ok=True)
    path = directory/f"{trade_date}_1000.json"
    document = {"freeze_id":manifest["freeze_id"], "signal_date":signal_date, "trade_date":trade_date,
        "asof":"10:00", "forward_recorded_at":now.isoformat(), "source_decision":str(decision_path.resolve()),
        "source_decision_sha256":old.digest(decision_path), "primary_code":primary_code,
        "candidates":candidates, "entry_authorized":False, "outcomes_available_at_snapshot":False}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
    event = old.append_event(registry/"ledger.jsonl", {"event_type":"TEN_OCLOCK_SNAPSHOT",
        "freeze_id":manifest["freeze_id"], "trade_date":trade_date, "asof":"10:00",
        "snapshot_file":str(path.relative_to(registry)), "snapshot_sha256":old.digest(path),
        "candidate_count":len(candidates), "primary_count":sum(x["is_frozen_primary"] for x in candidates),
        "entry_authorized":False})
    return {"event_hash":event["event_hash"], "snapshot":str(path.resolve()),
            "candidate_count":len(candidates), "primary_count":sum(x["is_frozen_primary"] for x in candidates),
            "entry_authorized":False}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def settle(registry, trade_date, outcomes_path, available_at):
    if verify(registry)["status"] != "PASS":
        raise ValueError("registry verification failed")
    now = datetime.fromisoformat(str(available_at)).astimezone(ZoneInfo("Asia/Shanghai"))
    if now.date() <= pd.Timestamp(trade_date).date():
        raise ValueError("settlement cannot be available on or before entry session")
    events = old.read_events(registry/"ledger.jsonl")
    snapshots = [e for e in events if e.get("event_type")=="TEN_OCLOCK_SNAPSHOT"
                 and e.get("trade_date")==trade_date]
    if len(snapshots) != 1 or any(e.get("event_type")=="OUTCOME_SETTLEMENT"
                                  and e.get("trade_date")==trade_date for e in events):
        raise ValueError("snapshot absent/duplicated or settlement already exists")
    snapshot_path = registry/snapshots[0]["snapshot_file"]
    if old.digest(snapshot_path) != snapshots[0]["snapshot_sha256"]:
        raise ValueError("snapshot hash mismatch")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    outcomes = pd.read_csv(outcomes_path, dtype={"code":str})
    outcomes["code"] = outcomes.code.str.zfill(6)
    expected = {row["code"]:row for row in snapshot["candidates"]}
    if set(outcomes.code) != set(expected) or outcomes.code.duplicated().any():
        raise ValueError("settlement must cover the exact frozen candidate set")
    required = {"code", "outcome_status", "b1_net_close_mark", "b2_net_close_mark",
                "b3_net_close_mark", "execution_state", "actual_fill_verified", "exit_complete",
                "entry_price", "exit_price", "net_return_after_costs", "provenance"}
    if required-set(outcomes):
        raise ValueError(f"missing settlement fields: {sorted(required-set(outcomes))}")
    records = []
    for raw in outcomes.to_dict("records"):
        candidate = expected[raw["code"]]; state = raw["execution_state"]
        if state not in EXECUTION_STATES or raw["outcome_status"] not in {"OBSERVED", "UNAVAILABLE"}:
            raise ValueError("invalid outcome or execution state")
        marks = [_number(raw[f"b{i}_net_close_mark"]) for i in (1,2,3)]
        if raw["outcome_status"] == "OBSERVED" and any(x is None for x in marks):
            raise ValueError("observed outcomes require all B+1/B+2/B+3 marks")
        if not candidate["is_frozen_primary"] and state != "NO_FROZEN_ENTRY":
            raise ValueError("non-primary candidate cannot become a strategy trade")
        actual = str(raw["actual_fill_verified"]).strip().lower() in {"true","1"}
        complete = str(raw["exit_complete"]).strip().lower() in {"true","1"}
        prices = [_number(raw["entry_price"]), _number(raw["exit_price"]),
                  _number(raw["net_return_after_costs"])]
        if state in {"ACTUAL_FILL","CONSERVATIVE_SIM_FILL"}:
            if not complete or any(x is None for x in prices) or prices[0] <= 0 or prices[1] <= 0:
                raise ValueError("filled settlement requires complete executable prices and net return")
            if (state=="ACTUAL_FILL") != actual:
                raise ValueError("actual fill flag contradicts execution state")
            if state=="ACTUAL_FILL" and raw["provenance"] != "BROKER_IMPORT":
                raise ValueError("actual fill requires broker import provenance")
            if state=="CONSERVATIVE_SIM_FILL" and raw["provenance"] != "FROZEN_SIMULATOR":
                raise ValueError("simulated fill requires frozen simulator provenance")
        elif actual or any(x is not None for x in prices):
            raise ValueError("unfilled candidate cannot carry trade prices or return")
        records.append({**candidate, "outcome_status":raw["outcome_status"],
            "b1_net_close_mark":marks[0], "b2_net_close_mark":marks[1], "b3_net_close_mark":marks[2],
            "execution_state":state, "actual_fill_verified":actual, "exit_complete":complete,
            "entry_price":prices[0], "exit_price":prices[1], "net_return_after_costs":prices[2],
            "provenance":raw["provenance"]})
    manifest = json.loads((registry/"active_freeze.json").read_text(encoding="utf-8"))
    directory=registry/"settlements"/manifest["freeze_id"]; directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{trade_date}.json"
    document={"freeze_id":manifest["freeze_id"],"trade_date":trade_date,
              "available_at":now.isoformat(),"outcomes_source":str(outcomes_path.resolve()),
              "outcomes_source_sha256":old.digest(outcomes_path),"records":records}
    with path.open("x",encoding="utf-8") as stream: json.dump(document,stream,ensure_ascii=False,indent=2)
    event=old.append_event(registry/"ledger.jsonl",{"event_type":"OUTCOME_SETTLEMENT",
        "freeze_id":manifest["freeze_id"],"trade_date":trade_date,
        "settlement_file":str(path.relative_to(registry)),"settlement_sha256":old.digest(path),
        "candidate_count":len(records),"completed_trade_count":sum(r["exit_complete"] for r in records),
        "actual_fill_count":sum(r["actual_fill_verified"] for r in records)})
    return {"event_hash":event["event_hash"],"settlement":str(path.resolve()),
            "candidate_count":len(records),"entry_authorized":False}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--registry",type=Path,default=DEFAULT_REGISTRY)
    sub=parser.add_subparsers(dest="command",required=True); sub.add_parser("init"); sub.add_parser("verify")
    snap=sub.add_parser("snapshot"); snap.add_argument("--decision",type=Path,required=True)
    settle_parser=sub.add_parser("settle"); settle_parser.add_argument("--trade-date",required=True)
    settle_parser.add_argument("--outcomes",type=Path,required=True); settle_parser.add_argument("--available-at",required=True)
    args=parser.parse_args()
    result=(initialize(args.registry) if args.command=="init" else verify(args.registry) if args.command=="verify"
            else snapshot_decision(args.registry,args.decision) if args.command=="snapshot"
            else settle(args.registry,args.trade_date,args.outcomes,args.available_at))
    print(json.dumps(result,ensure_ascii=False,indent=2))
