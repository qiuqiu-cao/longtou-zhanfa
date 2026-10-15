"""Extend the SAME state machine with saved complete September/October pools.

No requests, no historical forecast edits, no fabricated original-time receipts.
Later obtained material is retrospective context only.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
from public_outcome_sources_v293 import parse_pool, parse_daily
import supplement_identity_discovery as identity

FEATURE_PACKS = [
    "results/review_20261008_forecast_20261009_20261008T091618Z/model/features/features.json",
    "results/weekend_review_20261011T142844Z/model/features/features.json",
    "results/daily_review_20261012T105005Z/model/features_bound/features.json",
    "results/daily_review_20261013T074936Z/model/features/features.json",
    "results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json",
]
SPEC = b.ROOT / "research/model_upgrade_validation/supplement_discovery_20261015T063312Z/existing_pool_and_daily_evidence.json"


def build(output_dir=b.DEFAULT_RUN):
    out=Path(output_dir)
    from short_term_thinking_research import data_dir, merged, BASE_ROOT
    bundle=joblib.load(data_dir(out)/"dataset.joblib")
    spec=b.read(SPEC);b.verify(spec["source_bindings"])
    snapshots=identity.read_snapshots(spec,spec["source_bindings"],identity.o.stamp(b.now()))
    sources=dict(spec["source_bindings"])
    sources[str(SPEC.resolve())]=b.sha(SPEC)
    sources[str(Path(__file__).resolve())]=b.sha(__file__)
    current_rows=[];markets=[];pool_rows=[];daily_rows=[]
    seen_daily=set()
    # All original HTTP response clocks and SHA remain stored in their source
    # manifests. Request end-date slicing is repeated through the raw parser.
    for item in spec["pool_snapshots"]:
        day=item["date"]
        if item["kind"]=="EASTMONEY_CLOSING_POOL":
            rows=parse_pool(b.read(item["path"]),day,"limit_up")["rows"]
            for r in rows:
                pool_rows.append(dict(signal_date=day,code=r["code"],name=r["name"],board_height=r["boards"],
                    pool_assignment="CLOSING_LIMIT_UP",close=r["price_cents"]/100,industry=r.get("industry") or ""))
            manifest=b.read(item["manifest"])
            for request in manifest["requests"]:
                path=request.get("raw_path")
                if not path or not Path(path).is_file() or request.get("status") not in (None,"OK","SUCCESS") and not request.get("raw_sha256"):
                    continue
                if request.get("raw_sha256") and b.sha(path)!=request["raw_sha256"]:
                    raise ValueError("current raw HTTP hash changed")
                sources[str(Path(path).resolve())]=b.sha(path)
                if request.get("kind")=="failed_board" and request.get("date")==day:
                    try:
                        failures=parse_pool(b.read(path),day,"failed_board")["rows"]
                    except ValueError:
                        continue
                    prev=bundle["sessions"][bundle["sessions"].index(day)-1]
                    pp=snapshots.get(prev,{}).get("rows",{})
                    for r in failures:
                        height=int(pp[r["code"]]["boards"])+1 if r["code"] in pp else 1
                        pool_rows.append(dict(signal_date=day,code=r["code"],name=r["name"],board_height=height,
                            pool_assignment="FAILED_BOARD_OBSERVATION",close=r["price_cents"]/100,industry=r.get("industry") or ""))
                if request.get("kind")=="daily" and path not in seen_daily:
                    seen_daily.add(path)
                    params=request.get("parameters",{}).get("param","").split(",")
                    if len(params)!=6 or params[-1]!="":
                        continue
                    parsed=parse_daily(b.read(path),request["code"],params[2],params[3])
                    for date,r in parsed["days"].items():
                        daily_rows.append(dict(date=date,code=request["code"],**{k:r[k]/100 for k in ("open","high","low","close")},volume=r["volume_native"],volume_unit="PROVIDER_NATIVE_CURRENT"))
        else:
            alt=b.read(item["path"])
            normalized=identity.read_daily(spec,spec["source_bindings"],identity.o.stamp(b.now()))
            for r in alt["rows"]:
                if r["kind"]!="limit_up" or not r.get("mainboard_name_non_st_scope"):
                    continue
                daily=normalized.get(r["code"],{}).get("days",{}).get(day)
                reason=r.get("reason_secondary","")
                families=identity.families(reason)
                pool_rows.append(dict(signal_date=day,code=r["code"],name=r["name"],board_height=r["consecutive_boards"],pool_assignment="CLOSING_LIMIT_UP",
                    close=daily["close"]/100 if daily else np.nan,industry=families[0] if families else reason))
            for code,item in normalized.items():
                for date,r in item["days"].items():
                    daily_rows.append(dict(date=date,code=code,**{k:r[k]/100 for k in ("open","high","low","close")},volume=r["volume_native"],volume_unit="PROVIDER_NATIVE_CURRENT"))
    # The combined raw daily pack also includes stocks that failed after the
    # previous signal. It is useful for outcomes without survivor filtering.
    combined=b.ROOT/"results/score_update_20261014T1153Z/combined_daily.json"
    sources[str(combined.resolve())]=b.sha(combined)
    for code,item in b.read(combined).items():
        path=item["raw_path"];sources[str(Path(path).resolve())]=b.sha(path)
        parsed=parse_daily(b.read(path),code,"2026-08-31","2026-10-14")
        if parsed["days"]!=item["days"] or parsed["adjusted"] is not False:
            raise ValueError("combined current daily reconstruction mismatch")
        for date,r in parsed["days"].items():
            daily_rows.append(dict(date=date,code=code,**{k:r[k]/100 for k in ("open","high","low","close")},volume=r["volume_native"],volume_unit="PROVIDER_NATIVE_CURRENT"))
    pool=pd.DataFrame(pool_rows).drop_duplicates(["signal_date","code","pool_assignment"],keep="last")
    pool_index=pool[pool.pool_assignment.eq("CLOSING_LIMIT_UP")].set_index(["signal_date","code"])
    manifest=b.read(BASE_ROOT/"models/relay/BASE/manifest.json")
    if b.sha(manifest["model_path"])!=manifest["sha256"]:
        raise ValueError("original BASE model hash changed")
    base_model=joblib.load(manifest["model_path"])
    for relative in FEATURE_PACKS:
        path=b.ROOT/relative;pack=b.read(path);sources[str(path.resolve())]=b.sha(path)
        day=pack["signal_date"]
        expected={r["code"] for r in snapshots[day]["rows"].values() if r["boards"]>=2}
        if expected!={r["code"] for r in pack["relay"]}:
            raise ValueError("current complete candidate pool differs")
        market=dict(pack["market"]["features"],date=day)
        markets.append(market)
        for r in pack["relay"]:
            p=pool_index.loc[(day,r["code"])]
            current_rows.append(dict(r["features"],date=day,code=r["code"],name=r["name"],boards=r["boards"],
                actual_one_word_daily=r.get("actual_one_word_daily"),industry=p.get("industry",None),theme=None,
                target_date=pack["target_date"],target=np.nan,label_valid=False))
    current=pd.DataFrame(current_rows)
    current["base_score"]=base_model.predict_proba(current[manifest["fields"]])[:,list(base_model.classes_).index(1)]
    old=bundle["prepared"]["relay"]
    candidates=pd.concat([old,current],ignore_index=True)
    market=pd.concat([bundle["prepared"]["market"],pd.DataFrame(markets)],ignore_index=True).drop_duplicates("date",keep="last")
    attempts=pd.concat([bundle["attempts"],pool],ignore_index=True).drop_duplicates(["signal_date","code","pool_assignment"],keep="last")
    daily=pd.DataFrame(daily_rows)
    if len(daily):
        # Conflicting same-day observations stay flagged and unobserved. Newer
        # responses do not silently pick a convenient price.
        conflicts=daily.groupby(["date","code"])[["open","high","low","close"]].nunique().max(axis=1).gt(1)
        bad=set(conflicts[conflicts].index)
        daily=daily[[ (r.date,r.code) not in bad for r in daily.itertuples()]].drop_duplicates(["date","code"],keep="last")
    else:
        bad=set()
    # Use the same current provider's native volume on both dates of each ratio;
    # never divide shares in an old CSV by native hand units in a new response.
    oldbars=bundle["bars"].copy()
    current_codes=set(daily.code) if len(daily) else set()
    oldbars=oldbars[~(oldbars.code.isin(current_codes)&oldbars.date.ge("2026-08-31"))]
    bars=pd.concat([oldbars,daily],ignore_index=True).drop_duplicates(["date","code"],keep="last")
    features,edges=b.causal_features(candidates,market,attempts,bars,bundle["sessions"])
    recent=features[features.date.ge("2026-09-30")].copy()
    dynamic=current.merge(features.drop(columns=["name","boards"]),on=["date","code"],validate="one_to_one")
    bi={(r["date"],r["code"]):r for r in bars[bars.code.isin(set(current.code))].to_dict("records")}
    for i,r in dynamic.iterrows():
        nxt=snapshots.get(r.target_date)
        today=bi.get((r.date,r.code));after=bi.get((r.target_date,r.code))
        if not nxt or not today or not after:
            continue
        target_row=nxt["rows"].get(r.code)
        promoted=target_row is not None and target_row["boards"]==int(r.boards)+1
        try:
            price_promoted=b.cents(after["close"])==b.cap_cents(today["close"])
        except ValueError:
            continue
        if promoted!=price_promoted:
            continue
        dynamic.loc[i,["target","label_valid"]]=[int(promoted),True]
    # Independently archived author decisions are comparison columns only.
    opinions={}
    registry=b.ROOT/"data/independent_stock_judgement/registry.jsonl"
    sources[str(registry.resolve())]=b.sha(registry)
    for line in registry.read_text(encoding="utf-8-sig").splitlines():
        r=json.loads(line)
        if b.sha(r["path"])!=r["sha256"]:
            raise ValueError("independent original record hash changed")
        record=b.read(r["path"]);sources[str(Path(r["path"]).resolve())]=r["sha256"]
        for stock in record["rows"]:
            opinions[(record["signal_date"],stock["code"])]=stock["promotion_judgement"]
    recent["original_independent_judgement"]=[opinions.get((r.date,r.code),"NO_ARCHIVED_INDEPENDENT_CALL") for r in recent.itertuples()]
    dynamic["original_independent_judgement"]=[opinions.get((r.date,r.code),"NO_ARCHIVED_INDEPENDENT_CALL") for r in dynamic.itertuples()]
    sources.update(bundle["sources"])
    recent.to_csv(out/"20260930_to_20261014_all_roles.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([e for e in edges if e["date"]>="2026-09-30"]).to_csv(out/"20260930_to_20261014_all_parent_links.csv",index=False,encoding="utf-8-sig")
    # Store just current stocks' daily rows for the execution diagnostics, rather
    # than creating a second 2-million-row historical serialization.
    joblib.dump(dict(full=dynamic,bars=bars,features=recent,sessions=bundle["sessions"],sources=sources,
        snapshots=snapshots),out/"current_extension.joblib",compress=3)
    b.write_new(out/"current_extension_receipt.json",dict(formed_at=b.now(),signal_dates=sorted(dynamic.date.unique()),
        candidates=len(dynamic),recent_report_rows=len(recent),known_targets=int(dynamic.label_valid.sum()),
        conflicting_daily_keys=sorted(bad),source_bindings=sources,source_time_mode="RETROSPECTIVE_NOT_NEW_PREDICTION",
        dataset_path=str((out/"current_extension.joblib").resolve()),dataset_sha256=b.sha(out/"current_extension.joblib")))
    return dynamic


if __name__=="__main__":
    f=build();print(f.groupby("date").agg(candidates=("code","size"),known_targets=("label_valid","sum")).to_string())
