"""Build causal 09:35-10:00 absorption features from archived five-minute bars.

Bars are completed-bar evidence. They do not identify tick order, queue priority,
accepted fills, auction cancels, or events inside a five-minute interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_payload(raw: dict) -> tuple[str, pd.DataFrame]:
    required = {"code", "requested_date", "fields", "rows", "error_code"}
    if required - set(raw) or raw["error_code"] != "0":
        raise ValueError("invalid or unsuccessful archive payload")
    frame = pd.DataFrame(raw["rows"], columns=raw["fields"])
    fields = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
    if not set(fields) <= set(frame) or len(frame) != 48:
        raise ValueError("incomplete five-minute archive")
    frame = frame[fields].copy()
    day, code = str(raw["requested_date"]), str(raw["code"])
    expected = [day.replace("-", "") + t.strftime("%H%M%S") + "000"
                for start, end in (("09:35", "11:30"), ("13:05", "15:00"))
                for t in pd.date_range(day + " " + start, day + " " + end, freq="5min")]
    if (not frame.date.astype(str).eq(day).all() or not frame.code.astype(str).eq(code).all()
            or not frame.adjustflag.astype(str).eq("3").all()
            or frame.time.astype(str).tolist() != expected):
        raise ValueError("archive identity or timestamp coverage mismatch")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(frame[numeric]).all().all()
            or not frame[["open", "high", "low", "close"]].gt(0).all().all()
            or not frame[["volume", "amount"]].ge(0).all().all()
            or not frame.high.ge(frame[["open", "close", "low"]].max(axis=1)).all()
            or not frame.low.le(frame[["open", "close", "high"]].min(axis=1)).all()):
        raise ValueError("invalid OHLCV")
    canonical = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), frame


def load_union(roots: list[Path]):
    payloads, fingerprints, sources = {}, {}, defaultdict(list)
    manifests = []
    invalid_archives = []
    for root in roots:
        manifest_paths = [*root.glob("batch_*/manifest.json"), *root.glob("minute_*/manifest.json")]
        if (root / "manifest.json").exists():
            manifest_paths.append(root / "manifest.json")
        for manifest_path in sorted(set(manifest_paths)):
            manifests.append({"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for case in manifest.get("cases", []):
                if case.get("error_code") != "0":
                    continue
                raw_path = manifest_path.parent / f"{case['code']}_{case['date']}.json"
                if not raw_path.exists() or sha256(raw_path) != case.get("raw_sha256"):
                    raise ValueError(f"raw hash mismatch: {raw_path}")
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                try:
                    fingerprint, frame = normalized_payload(raw)
                except ValueError as exc:
                    invalid_archives.append({"path": str(raw_path.resolve()),
                                             "manifest_code": str(case.get("code")),
                                             "manifest_date": str(case.get("date")),
                                             "reason": str(exc)})
                    continue
                key = (str(raw["code"]), str(raw["requested_date"]))
                if key in fingerprints and fingerprints[key] != fingerprint:
                    raise ValueError(f"conflicting duplicate archive: {key}")
                fingerprints[key], payloads[key] = fingerprint, frame
                sources[key].append(str(raw_path.resolve()))
    return payloads, sources, manifests, invalid_archives


def reconcile_daily(frame: pd.DataFrame, daily: pd.Series | None) -> str:
    if daily is None:
        return "DAILY_COMPARATOR_UNAVAILABLE"
    aggregate = {"open": float(frame.iloc[0].open), "high": float(frame.high.max()),
                 "low": float(frame.low.min()), "close": float(frame.iloc[-1].close)}
    if any(pd.isna(daily.get(k)) for k in aggregate):
        return "DAILY_COMPARATOR_UNAVAILABLE"
    exact = all(abs(value-float(daily[key])) <= 1e-6 for key, value in aggregate.items())
    endpoint = (abs(aggregate["open"]-float(daily["open"])) <= 1e-6
                and abs(aggregate["close"]-float(daily["close"])) <= 1e-6)
    contained = (aggregate["high"] <= float(daily["high"])+1e-6
                 and aggregate["low"] >= float(daily["low"])-1e-6)
    return ("DAILY_OHLC_MATCH_ONLY" if exact else "DAILY_OHLC_RANGE_CONSISTENT"
            if endpoint and contained else "DAILY_OHLC_CONTRADICTION")


def features(code: str, day: str, frame: pd.DataFrame, source_count: int,
             daily: pd.Series | None = None) -> dict:
    opening = frame.loc[frame.time.astype(str).le(day.replace("-", "") + "100000000")].copy()
    if len(opening) != 6:
        raise ValueError("09:35-10:00 completed-bar window must have six bars")
    opening["cum_vwap"] = opening.amount.cumsum() / opening.volume.cumsum().replace(0, np.nan)
    below = opening.index[(opening.index != opening.index[0]) & opening.close.lt(opening.cum_vwap)]
    event_index = below[0] if len(below) else None
    reclaim_index = None
    if event_index is not None:
        later = opening.loc[opening.index > event_index]
        reclaimed = later.index[later.close.ge(later.cum_vwap)]
        reclaim_index = reclaimed[0] if len(reclaimed) else None
    prior_volume = (opening.loc[opening.index < event_index, "volume"].median()
                    if event_index is not None else np.nan)
    event_volume = opening.loc[event_index, "volume"] if event_index is not None else np.nan
    reclaim_volume = opening.loc[reclaim_index, "volume"] if reclaim_index is not None else np.nan
    last = opening.iloc[-1]
    reconciliation = reconcile_daily(frame, daily)
    return {
        "code": code.split(".")[-1], "date": day, "archive_source_count": source_count,
        "window_first_completed_bar": "09:35", "window_last_completed_bar": "10:00",
        "first_below_cum_vwap_at": str(opening.loc[event_index, "time"]) if event_index is not None else None,
        "first_below_volume_vs_prior_median": (float(event_volume/prior_volume)
                                                if event_index is not None and prior_volume > 0 else None),
        "first_reclaim_cum_vwap_at": str(opening.loc[reclaim_index, "time"]) if reclaim_index is not None else None,
        "reclaim_volume_vs_below_volume": (float(reclaim_volume/event_volume)
                                            if reclaim_index is not None and event_volume > 0 else None),
        "close_1000_above_cum_vwap": bool(last.close >= last.cum_vwap) if pd.notna(last.cum_vwap) else None,
        "close_1000_vs_cum_vwap": float(last.close/last.cum_vwap-1) if pd.notna(last.cum_vwap) else None,
        "close_1000_vs_close_0935": float(last.close/opening.iloc[0].close-1),
        "window_min_vs_close_0935": float(opening.low.min()/opening.iloc[0].close-1),
        "completed_bar_only": True, "auction_verified": False, "tick_order_verified": False,
        "queue_or_fill_verified": False, "daily_reconciliation_status": reconciliation,
        "feature_usable": reconciliation in {"DAILY_OHLC_MATCH_ONLY", "DAILY_OHLC_RANGE_CONSISTENT"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", action="append", type=Path, required=True)
    parser.add_argument("--daily", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable feature output")
    payloads, sources, manifests, invalid_archives = load_union(args.archive_root)
    daily_lookup = {}
    if args.daily:
        daily_frame = pd.read_csv(args.daily, dtype={"code": str})
        if daily_frame.duplicated(["code", "date"]).any():
            raise ValueError("duplicated daily comparator keys")
        daily_lookup = {(str(row["code"]).zfill(6), str(row["date"])): row
                        for row in daily_frame.to_dict("records")}
    rows = [features(code, day, frame, len(sources[(code, day)]),
                     daily_lookup.get((code.split(".")[-1], day)))
            for (code, day), frame in sorted(payloads.items())]
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / "opening_absorption_features.csv", index=False,
                              encoding="utf-8-sig")
    report = {"version": "v138-opening-absorption-features-1", "status": "DIAGNOSTIC_ONLY",
              "unique_archives": len(rows), "duplicate_source_references": sum(len(v)-1 for v in sources.values()),
              "invalid_archives_excluded": len(invalid_archives), "invalid_archive_details": invalid_archives,
              "date_start": min((d for _, d in payloads), default=None),
              "date_end": max((d for _, d in payloads), default=None), "manifests": manifests,
              "daily_comparator": str(args.daily.resolve()) if args.daily else None,
              "daily_comparator_sha256": sha256(args.daily) if args.daily else None,
              "daily_reconciliation_status_counts": (pd.Series([r["daily_reconciliation_status"] for r in rows])
                                                       .value_counts().to_dict()),
              "limitations": ["provider bars begin at 09:35", "no 09:25 auction evidence",
                              "no ordering inside a five-minute bar", "no queue or actual-fill proof"],
              "deployment_authorized": False}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "unique_archives", "duplicate_source_references", "invalid_archives_excluded",
                                             "date_start", "date_end")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
