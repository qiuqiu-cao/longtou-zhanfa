"""Append signal-input semantics and exact turnover references; no score edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from leader_system_v3 import parse_clock, safe_float
from model_upgrade_research import read, sha
from relay_error_analysis import replay_feature_archive, verify_sources
from public_outcome_sources_v293 import reconstruct


def legacy_clock_proxy(first, last, bombs, turnover_pct):
    """Original V3 rule, including its acceptance of missing bombs/turnover."""
    first, last = parse_clock(first), parse_clock(last)
    bombs, turnover = safe_float(bombs), safe_float(turnover_pct)
    return float(not math.isnan(first) and first <= 565.02
                 and not math.isnan(last) and last <= 565.02
                 and (math.isnan(bombs) or bombs == 0)
                 and (math.isnan(turnover) or turnover < 3))


def turnover_references(history, closing_by_day, signal, static_last_day, predecessor):
    """Reproduce the original last observed relay-pool dictionary, not OHLC."""
    days = sorted(d for d in closing_by_day if static_last_day < d <= signal)
    if not days or days[-1] != signal:
        raise ValueError("original current closing pool unavailable")
    prior = {}
    for r in history[history.date.lt(days[0])].sort_values(["date", "code"]).itertuples():
        if pd.notna(r.turnover):
            prior[r.code] = (r.date, float(r.turnover)*100)
    result = []
    for day in days:
        codes = [r["code"] for r in closing_by_day[day] if r["boards"] >= 2]
        if len(codes) != len(set(codes)):
            raise ValueError("duplicate original relay stock")
        for r in closing_by_day[day]:
            if r["boards"] < 2:
                continue
            ref_day, ref_pct = prior.get(r["code"], (None, None))
            current = safe_float(r.get("turnover_pct"))
            current = current if math.isfinite(current) and current >= 0 else None
            ratio = current/ref_pct if current is not None and ref_pct is not None and ref_pct > 0 else None
            if day == signal:
                result.append({"code": r["code"], "turnover_reference_date": ref_day,
                    "current_turnover_pct": current, "reference_turnover_pct": ref_pct,
                    "reproduced_turnover_vs_prev": ratio,
                    "reference_is_previous_official_session": None if ref_day is None else ref_day == predecessor,
                    "definition": "LAST_OBSERVED_RELAY_POOL_TURNOVER_NOT_PREVIOUS_SESSION"})
            if current is not None:
                prior[r["code"]] = (day, current)
    return result


def build(feature_archive, base_path, source_receipt, reader_policy, output):
    base, receipt = read(base_path), read(source_receipt)
    output = Path(output).resolve()
    if (output.exists() or output.suffix != ".json" or not output.parent.is_dir()
            or any(p.lower() == "expectation_registry" for p in output.parts)
            or output.is_relative_to(Path(source_receipt).resolve().parent)):
        raise ValueError("unique semantic JSON outside original research/registry required")
    verify_sources({receipt["prepared_path"]: receipt["prepared_sha256"]})
    verify_sources(receipt["sources"])
    prepared = joblib.load(receipt["prepared_path"])
    pack = replay_feature_archive(feature_archive, base["knowledge_cutoff"], reader_policy)
    if (str(Path(base["feature_archive"]).resolve()) != str(Path(feature_archive).resolve())
            or (pack["signal_date"], pack["target_date"]) != (base["signal_date"], base["target_date"])):
        raise ValueError("original BASE feature/date scope conflict")
    manifest = read(Path(feature_archive)/"manifest.json")
    calendar = read(manifest["calendar_path"])
    signal = pack["signal_date"]
    predecessor = calendar["sessions"][calendar["sessions"].index(signal)-1]
    pools, raw_clock_rows, sources = {}, {}, {}
    archives = list(dict.fromkeys(manifest["history_archives"]+[manifest["source_archive"]]))
    for archive in archives:
        m, rebuilt, _, _ = reconstruct(archive)
        for (day, kind), value in rebuilt.items():
            if kind == "limit_up":
                if day in pools and pools[day] != value["rows"]:
                    raise ValueError("original overlapping closing pools conflict")
                pools[day] = value["rows"]
        for r in m["requests"]:
            if r["kind"] != "limit_up" or r.get("date") != signal or not r.get("raw_path"):
                continue
            raw = read(r["raw_path"])
            if str(raw.get("data", {}).get("qdate")) != signal.replace("-", ""):
                continue
            for item in raw["data"]["pool"]:
                if item["c"] in raw_clock_rows and raw_clock_rows[item["c"]] != item:
                    raise ValueError("original raw clock revisions conflict")
                raw_clock_rows[item["c"]] = item
            sources[str(Path(r["raw_path"]).resolve())] = sha(r["raw_path"])
    # Use the same static market boundary as the original feature generator.
    from model_upgrade_inputs import OLD
    market_path = OLD/"combined_market_input.csv"
    market = pd.read_csv(market_path, usecols=["date"])
    latest = market.loc[market.date.lt(signal), "date"].max()
    rows = turnover_references(prepared["relay"], pools, signal, latest, predecessor)
    facts = {r["code"]: r for r in pack["relay"]}
    calls = {r["code"]: r for r in base["relay"]["candidates"]}
    if {r["code"] for r in rows} != set(facts) or set(facts) != set(calls):
        raise ValueError("complete original signal candidate scope conflict")
    for row in rows:
        f = facts[row["code"]]
        value = f["features"].get("turnover_vs_prev")
        replayed = row["reproduced_turnover_vs_prev"]
        if (value is None) != (replayed is None) or (value is not None and not math.isclose(value, replayed, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("turnover reference does not reproduce original feature")
        raw = raw_clock_rows.get(row["code"])
        old = legacy_clock_proxy(raw.get("fbt"), raw.get("lbt"), raw.get("zbc"), raw.get("hs")) if raw else None
        row.update(name=f["name"], boards=f["boards"], original_raw_score=calls[row["code"]]["raw_score"],
            original_predicted_positive=calls[row["code"]]["predicted_positive"],
            actual_one_word_daily=f["actual_one_word_daily"], original_signal_one_word=f["features"].get("signal_one_word"),
            original_clock_low_turnover_proxy=old,
            proxy_definition_agrees=None if old is None or f["features"].get("signal_one_word") is None else old == f["features"]["signal_one_word"],
            missing_fields=[k for k in prepared["relay_fields"] if f["features"].get(k) is None])
    formed = datetime.now(timezone.utc).isoformat()
    for path in (source_receipt, receipt["prepared_path"], base_path, reader_policy, market_path,
                 Path(feature_archive)/"manifest.json", Path(feature_archive)/"features.json", __file__):
        sources[str(Path(path).resolve())] = sha(path)
    result = {"status": "SOURCE_BOUND_CURRENT_INPUT_SEMANTICS_ONLY", "formed_at": formed,
        "signal_date": signal, "target_date": pack["target_date"], "original_cutoff": base["knowledge_cutoff"],
        "rows": rows, "source_bindings": sources,
        "proxy_note": "Historical V3 uses first/last seal clock, bombs and turnover; September/current input uses OHLC one-word and turnover. Neither certifies leader identity.",
        "tail_loss_note": "Current input requires complete exact prior closing feedback before Q10; missing remains unknown.",
        "turnover_unit": "model fraction = provider percent / 100; ratio dimensionless",
        "earlier_frozen_metadata_replaced": False, "score_or_call_changed": False,
        "new_fits": 0, "new_mature_forward_samples": 0, "accuracy_gain_verified": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-archive", "base", "source-receipt", "reader-policy", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    result = build(args.feature_archive, args.base, args.source_receipt, args.reader_policy, args.output)
    print(json.dumps({"status": result["status"], "candidate_count": len(result["rows"]),
                      "proxy_differences": sum(r["proxy_definition_agrees"] is False for r in result["rows"])}))


if __name__ == "__main__":
    main()
