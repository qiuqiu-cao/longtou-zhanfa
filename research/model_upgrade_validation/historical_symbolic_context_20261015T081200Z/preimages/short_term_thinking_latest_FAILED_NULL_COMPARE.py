"""Source-bound continuation of the shared causal engine, without fitting.

The alternative closing page and inferred quote fields remain diagnostic.
Neither this module nor its output is an original V293 production source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import re

from bs4 import BeautifulSoup
import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b
import short_term_thinking_workbench as thinking
import historical_symbolic_context as symbolic
import model_upgrade_compat as compat
import public_outcome_sources_v293 as public

RUN = symbolic.RUN
SOURCE = RUN / "latest15_alternative_page_once"
DAY = "2026-10-15"
PRIOR = "2026-10-14"
TARGET = "2026-10-16"


def prior_adapter():
    path = b.ROOT / "results/score_update_20261014T1153Z/supplemental_score_update.py"
    spec = importlib.util.spec_from_file_location("archived14_alternative_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct():
    """Re-parse original bytes, preserving page/quote scope and receipt clocks."""
    receipt = b.read(SOURCE / "receipt.json")
    captured = b.read(SOURCE / "daily_manifest.json")
    b.verify(captured["source_bindings"])
    parsed = b.read(SOURCE / "parsed_complete_pools.json")
    html = SOURCE / "page.raw.html"
    if b.sha(html) != parsed["source_sha256"] or receipt["http_status"] != 200:
        raise ValueError("exact raw alternative page binding required")
    body = BeautifulSoup(html.read_bytes(), "html.parser")
    date_input = body.select_one("#limitUpDateInput")
    if date_input is None or date_input.get("value") != DAY:
        raise ValueError("alternative page exact selected date differs")
    tables = body.find_all("table")
    if len(tables) != 3 or [len(t.find_all("tr"))-1 for t in tables] != [36, 7, 26]:
        raise ValueError("complete raw alternative denominators changed")
    ladder = {}
    for group in body.select("[data-ladder-height]"):
        for anchor in group.select("article a"):
            code = re.search(r"/([0-9]{6})/", anchor["href"]).group(1)
            if code in ladder:
                raise ValueError("duplicate ladder stock")
            ladder[code] = int(group["data-ladder-height"])
    if ladder != parsed["ladder"]:
        raise ValueError("raw reported consecutive ladder changed")
    normalized = b.read(SOURCE / "normalized_daily.json")
    requests = {r["code"]: r for r in captured["requests"]}
    if len(requests) != len(captured["requested_codes"]):
        raise ValueError("one retained request per code required")
    daily = {}
    bindings = dict(captured["source_bindings"])
    for code, item in normalized.items():
        request = requests[code]
        if b.sha(item["raw_path"]) != request["raw_sha256"]:
            raise ValueError("daily raw hash changed")
        values = public.parse_daily(b.read(item["raw_path"]), code, "2026-09-18", DAY)
        if any(values[k] != item[k] for k in ["code", "days", "quote", "adjusted"]):
            raise ValueError("daily values differ from original raw response")
        daily[code] = item
    adapter = prior_adapter()
    # Reuse its existing field/amount checks with only the exact new signal date.
    quote_adapter = compat.clone_module(adapter, {"DAY": DAY})
    pools, audit, raw_rows, seen = {}, [], [], set()
    obtained = max([r["available_at"] for r in requests.values() if r.get("available_at")])
    for kind, table in zip(["limit_up", "failed_board", "limit_down"], tables):
        rows = []
        for tr in table.find_all("tr")[1:]:
            anchor = tr.find_all("td")[0].find("a")
            code = re.search(r"/([0-9]{6})/", anchor["href"]).group(1)
            name = anchor.get_text(strip=True)
            scope = code.startswith(public.MAINBOARD) and "ST" not in name.upper() and "退" not in name
            raw_rows.append((kind, code, name, float(tr["data-sort-turnover"]), bool(scope)))
            if not scope:
                continue
            if code in seen:
                raise ValueError("closing/failed/down pools overlap")
            seen.add(code)
            item = daily.get(code)
            bar = item.get("days", {}).get(DAY) if item else None
            page_turnover = float(tr["data-sort-turnover"])
            turnover, amount, quote_turnover = page_turnover, None, None
            if kind in {"limit_up", "failed_board"}:
                if item is None or bar is None:
                    raise ValueError("exact current OHLC missing from full price-input pool")
                amount, quote_turnover, bar = quote_adapter.quote_fields(code, item)
                if abs(turnover-quote_turnover) > .011:
                    turnover = None
                upper = item["quote"]["values"]["upper_candidate"]
                if (bar["close"] == upper) != (kind == "limit_up"):
                    raise ValueError("reported closing pool and exact upper price disagree")
                if kind == "failed_board" and bar["high"] != upper:
                    raise ValueError("reported failed board lacks exact daily upper touch")
            row = dict(code=code, name=name, price_cents=bar["close"] if bar else None,
                       turnover_pct=turnover, amount=amount, industry=None,
                       source_semantics="ALTERNATIVE_PAGE_PLUS_QUOTE_FIELDS_INFERRED_CROSSCHECKED")
            if kind == "limit_up":
                row["boards"] = ladder.get(code, 1)
            rows.append(row)
            audit.append(dict(code=code, kind=kind, turnover_page=page_turnover,
                turnover_quote_rounded=quote_turnover, turnover_selected=turnover,
                amount_cny_inferred=amount, exact_day_bar=bar,
                business_or_narrative_from_AI_page_accepted_as_fact=False))
        pools[(DAY, kind)] = dict(date=DAY, kind=kind, rows=rows, available_at=obtained,
            raw_path=str(html.resolve()), capture_complete_within_alternative_page=True,
            exchange_wide_completeness_verified=False, exchange_security_status_verified=False)
    expected = [(r["kind"],r["code"],r["name"],r["turnover_pct"],r["scope"]) for r in parsed["rows"]]
    if raw_rows != expected:
        raise ValueError("saved pool projection differs from complete raw rows")
    if [len(pools[(DAY,k)]["rows"]) for k in ["limit_up","failed_board","limit_down"]] != [34,5,25]:
        raise ValueError("mainboard alternative denominator changed")
    prior = b.read(b.ROOT / "results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json")
    feedback = []
    for member in prior["original_pool"]["rows"]:
        code = member["code"]
        before = daily.get(code,{}).get("days",{}).get(PRIOR)
        after = daily.get(code,{}).get("days",{}).get(DAY)
        if not before or not after or before["close"] != member["price_cents"]:
            raise ValueError("previous full closing cohort exact feedback unavailable")
        feedback.append(dict(code=code, name=member["name"], price_cents=after["close"],
            pct=(after["close"]/before["close"]-1)*100, derived_from_exact_day_prices=True))
    pools[(DAY,"previous_limit")] = dict(date=DAY,kind="DERIVED_PREVIOUS_CLOSING_COHORT_FEEDBACK_NOT_EM_API",
        rows=feedback,available_at=obtained,raw_path=str((SOURCE/"normalized_daily.json").resolve()))
    for path in [html,SOURCE/"receipt.json",SOURCE/"daily_manifest.json",SOURCE/"parsed_complete_pools.json",SOURCE/"normalized_daily.json",Path(__file__),Path(adapter.__file__)]:
        bindings[str(path.resolve())] = b.sha(path)
    manifest = dict(version="ALTERNATIVE_PUBLIC_PAGE_DAILY_DIAGNOSTIC_INPUT_NOT_V293",
        signal_date=PRIOR,target_date=DAY,recorded_at=obtained,requests=[],source_hashes=bindings,
        original_raw_provider_capture_recovered=False,original_production_scoring_gate_passed=False)
    return manifest,pools,daily,[],audit


def build(output_dir):
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("new immutable current source/context directory required")
    manifest,pools,daily,failures,audit = reconstruct()
    out.mkdir(parents=True)
    b.write_new(out/"alternative_quote_pool_audit.json",dict(formed_at=b.now(),rows=audit,source_bindings=manifest["source_hashes"]))
    alt = out/"alternative_source_projection"
    b.write_new(alt/"manifest.json",manifest)
    b.write_new(alt/"normalized.json",dict(kind="EXPLICIT_DIAGNOSTIC_NOT_EM_JSON",pools={"/".join(k):v for k,v in pools.items()}))
    adapter = prior_adapter()
    inputs,_ = compat.runtime(adapter.POLICY)
    oldmanifest = b.read(adapter.OLD/"model/features/manifest.json")
    history = oldmanifest["history_archives"]+[oldmanifest["source_archive"],str(adapter.ALT.resolve())]
    def reader(path):
        if Path(path).resolve() == alt.resolve():
            return reconstruct()[:4]
        if Path(path).resolve() == adapter.ALT.resolve():
            return adapter.rebuild_alternative()[:4]
        return public.reconstruct(path)
    diagnostic = compat.clone_module(inputs,{"reconstruct":reader})
    pack = diagnostic._derive(adapter.PROTOCOL,alt,adapter.CALENDAR,history,b.now())
    if (pack["signal_date"],pack["target_date"]) != (DAY,TARGET) or len(pack["relay"]) != 6:
        raise ValueError("exact new date and complete six-candidate relay roster required")
    pack["source_status"] = "ALTERNATIVE_SOURCE_DIAGNOSTIC_NOT_PRODUCTION"
    path = out/"features_BEFORE_MODEL_DISPLAY.json"
    b.write_new(path,pack)
    engine = joblib.load(b.DEFAULT_RUN/"scope_correction/engine_context.joblib")
    candidates = pd.DataFrame([dict(r["features"],date=DAY,code=r["code"],name=r["name"],boards=r["boards"],industry=None,theme=None) for r in pack["relay"]])
    c = pd.concat([engine["prepared"]["relay"].query("date < @DAY"),candidates],ignore_index=True)
    m = pd.concat([engine["prepared"]["market"].query("date < @DAY"),pd.DataFrame([dict(pack["market"]["features"],date=DAY)])],ignore_index=True)
    previous = {r["code"]:r["boards"] for r in b.read(b.ROOT/"results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json")["original_pool"]["rows"]}
    extra = []
    for kind in ["limit_up","failed_board"]:
        for r in pools[(DAY,kind)]["rows"]:
            extra.append(dict(signal_date=DAY,code=r["code"],name=r["name"],board_height=r["boards"] if kind=="limit_up" else previous.get(r["code"],0)+1,
                pool_assignment="CLOSING_LIMIT_UP" if kind=="limit_up" else "FAILED_BOARD_OBSERVATION",close=r["price_cents"]/100,industry=""))
    a = pd.concat([engine["attempts"],pd.DataFrame(extra)],ignore_index=True)
    a.attrs["failed_scope_complete_dates"] = sorted(set(engine["attempts"].attrs["failed_scope_complete_dates"])|{DAY})
    bars = pd.DataFrame([dict(date=date,code=code,**{k:r[k]/100 for k in ["open","high","low","close"]},volume=r["volume_native"],volume_unit="PROVIDER_NATIVE_CURRENT")
        for code,item in daily.items() for date,r in item["days"].items()])
    oldbars = engine["bars"]
    common = oldbars.merge(bars,on=["date","code"],suffixes=("_old","_new"),validate="one_to_one")
    conflicts = common[[k+"_old" for k in ["open","high","low","close"]]].to_numpy() != common[[k+"_new" for k in ["open","high","low","close"]]].to_numpy()
    if conflicts.any():
        raise ValueError("existing/new current OHLC conflict; preserve as unresolved before using")
    # Use a whole native-volume interval for new names; all existing OHLC agree.
    first = bars.groupby("code").date.min().to_dict()
    keep = [r.code not in first or r.date < first[r.code] for r in oldbars.itertuples()]
    allbars = pd.concat([oldbars[keep],bars],ignore_index=True)
    features,edges = b.causal_features(c,m,a,allbars,engine["sessions"])
    current = features[features.date.eq(DAY)].copy()
    sources = engine["sources"]|manifest["source_hashes"]|{str(path.resolve()):b.sha(path)}
    capsule = dict(version=b.VERSION,prepared={"relay":c,"market":m},features=features,attempts=a,bars=allbars,sessions=engine["sessions"],sources=sources)
    bundle = out/"engine_context_WITH_CURRENT15.joblib"
    joblib.dump(capsule,bundle,compress=3)
    # Invoke the exact live workbench calculation, then compare to batch output.
    live = thinking.context(path,bundle)
    pd.testing.assert_frame_equal(pd.DataFrame(live["rows"])[current.columns].reset_index(drop=True),current.reset_index(drop=True),check_dtype=False)
    b.write_new(out/"causal_context_SCORES_MASKED.json",live)
    sym = symbolic.context_for_review(path,RUN/"qualified_final/symbolic_context.joblib",live)
    b.write_new(out/"symbolic_context_SCORES_MASKED.json",sym)
    current.to_csv(out/"complete_current6_causal_roles.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([e for e in edges if e["date"]==DAY]).to_csv(out/"complete_current6_parent_candidates.csv",index=False,encoding="utf-8-sig")
    # Outcome append is deliberately separate from all frozen training labels.
    old = b.read(b.ROOT/"results/score_update_20261014T1153Z/current29_feature_pack_BEFORE_INFERENCE.json")
    today = {r["code"]:r for r in pools[(DAY,"limit_up")]["rows"]}
    outcomes = []
    for stock in old["relay"]:
        code = stock["code"];before=daily[code]["days"][PRIOR];after=daily[code]["days"][DAY]
        ladder_promoted=code in today and today[code]["boards"]==stock["boards"]+1
        price_promoted=after["close"]==b.cap_cents(before["close"]/100)
        valid=ladder_promoted==price_promoted
        outcomes.append(dict(signal_date=PRIOR,target_date=DAY,code=code,name=stock["name"],signal_boards=stock["boards"],
            target_reported_boards=today.get(code,{}).get("boards"),promoted=int(ladder_promoted) if valid else None,
            price_and_ladder_agree=valid,signal_price_cents=before["close"],target_price_cents=after["close"],
            original_suspension_streak_gap=code in {"002813","002860"},formal_maturity_added=False))
    b.write_new(out/"previous14_full15_outcomes_APPEND_ONLY.json",dict(formed_at=b.now(),rows=outcomes,source_bindings=manifest["source_hashes"],original_labels_changed=False,new_formal_maturity=0))
    summary = dict(formed_at=b.now(),signal_date=DAY,target_date=TARGET,raw_counts=[36,7,26],mainboard_counts=[34,5,25],relay_count=len(current),
        previous_full_pool_feedback_count=34,previous14_outcomes_known=sum(r["promoted"] is not None for r in outcomes),
        previous14_promoted=sum(r["promoted"]==1 for r in outcomes),raw_daily_codes=len(daily),daily_latest_missing=[c for c,v in daily.items() if DAY not in v["days"]],
        historical_live_causal_rows_exact=True,common_OHLC_checked=len(common),common_OHLC_conflicts=0,
        model_outputs_displayed=False,new_fits=0,new_formal_forecasts=0,new_formal_maturity=0,actual_fills=0,
        source_policy="Alternative provider complete returned page; inferred quote units crosschecked, not original V293 or exchange certification",source_bindings=sources)
    b.write_new(out/"source_context_receipt.json",summary)
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",required=True);args=p.parse_args()
    result=build(args.output_dir);print(b.clean({k:v for k,v in result.items() if k!="source_bindings"}))
