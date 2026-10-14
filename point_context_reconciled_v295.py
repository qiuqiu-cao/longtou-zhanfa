"""Correct opaque provider internal IDs without rewriting the first archive."""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import point_context_v295 as original
from daily_research_contract_v292 import stamp
from next_session_expectation_contract_v271 import digest

VERSION="v295-context-internal-id-reconciliation-1"


def latest(value,code,available_at):
    data=original.attention_data(value)
    if not isinstance(data,dict) or data.get("srcSecurityCode")!=original.symbol(code):
        raise ValueError("attention explicit security identity mismatch")
    if not isinstance(data.get("innerCode"),str) or not data["innerCode"]:
        raise ValueError("provider internal identifier missing")
    # srcSecurityCode explicitly names the exchange and ticker. innerCode is
    # an opaque provider identifier (e.g. gs80534114), not a second ticker.
    adjusted=deepcopy(value)
    adjusted["data"]["innerCode"]=code
    result=original.parse_latest(adjusted,code,available_at)
    result["provider_internal_identifier"]=data["innerCode"]
    result["security_identity_basis"]="EXPLICIT_SRC_SECURITY_CODE; internal identifier not interpreted as ticker"
    return result


def rebuild(parent,recorded_at):
    parent=Path(parent)
    value=deepcopy(original.reconstruct(parent))
    manifest=json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
    entries={(e["code"],e["kind"]):e for e in manifest["requests"]}
    corrected=[]
    for row in value["candidates"]:
        entry=entries[row["code"],"attention_latest"]
        if entry["status"]=="RAW_JSON_RECEIVED":
            try:
                item=latest(json.loads(Path(entry["raw_path"]).read_bytes()),row["code"],entry["available_at"])
                item["raw_first_observed_at"]=entry["available_at"]
                item["available_at"]=max([entry["available_at"],recorded_at],key=stamp)
                item["raw_sources"]=[entry["raw_path"]]
                if row["attention"] is None:corrected.append(row["code"])
                row["attention"]=item
                row["errors"].pop("attention",None)
            except (ValueError,KeyError,TypeError) as exc:
                row["attention"]=None;row["errors"]["attention"]=str(exc)
        # This new derived receipt may never enter a cutoff before its creation.
        for key in ["categories","attention_history"]:
            item=row[key]
            if item:
                item["raw_first_observed_at"]=item["available_at"]
                item["available_at"]=max([item["available_at"],recorded_at],key=stamp)
    value.update(version=VERSION,recorded_at=recorded_at,correction={"rule":"Explicit srcSecurityCode binds security; innerCode remains opaque",
        "codes_previously_rejected_by_internal_id_check":corrected,"parent_archive_preserved":True,"source_business_or_sentiment_facts_upgraded":False})
    return value


def reconcile(parent,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    recorded=original.now()
    value=rebuild(parent,recorded)
    original.write(output/"context.json",value)
    original.write(output/"manifest.json",{"version":VERSION,"parent_archive":str(Path(parent).resolve()),"recorded_at":recorded,
        "source_hashes":{str(Path(__file__).resolve()):original.sha(__file__),str((Path(parent)/"manifest.json").resolve()):original.sha(Path(parent)/"manifest.json"),
            str((Path(parent)/"context.json").resolve()):original.sha(Path(parent)/"context.json")},"artifact_sha256":original.sha(output/"context.json")})
    reconstruct(output)
    return {"status":"OPAQUE_INTERNAL_IDS_RECONCILED_WITH_PARENT_PRESERVED","codes":len(value["candidates"]),
        "attention_available":sum(r["attention"] is not None for r in value["candidates"]),"corrected_codes":value["correction"]["codes_previously_rejected_by_internal_id_check"]}


def reconstruct(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["version"]!=VERSION:raise ValueError("reconciled context required")
    for path,wanted in manifest["source_hashes"].items():
        if original.sha(path)!=wanted:raise ValueError("reconciled context source changed")
    if original.sha(directory/"context.json")!=manifest["artifact_sha256"]:raise ValueError("reconciled context artifact changed")
    value=rebuild(manifest["parent_archive"],manifest["recorded_at"])
    if digest(value)!=digest(json.loads((directory/"context.json").read_text(encoding="utf-8"))):raise ValueError("reconciled context differs from raw bytes")
    return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["parent","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    print(json.dumps(reconcile(args.parent,args.output),ensure_ascii=True))
