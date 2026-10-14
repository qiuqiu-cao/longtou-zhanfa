"""Outcome contract for uninterrupted leader relay to severe-deviation targets.

This module consumes separately verified, point-in-time regulatory indicators.
It does not reconstruct exchange rules or turn a target hit into trading profit.
Future path columns are LABELS ONLY and must never enter live feature matrices.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

VERSION = "v16-target-contract-1"
TARGETS = {"100": ("deviation_10d_pct", 100.0),
           "200": ("deviation_30d_pct", 200.0)}


def build_labels(observations: pd.DataFrame, path: pd.DataFrame,
                 calendar: list[str], horizon: int = 30) -> pd.DataFrame:
    """First future close hitting target while every relay day closes limit-up.

    The next trading day is the proposed entry day. A broken board, including
    that day, terminates the relay. Missing sessions/data censor unresolved
    outcomes; they are never filled with zero. Already-hit signals are excluded.
    `episode_id` identifies the same stock's continuous run across signal dates.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = {"code", "signal_date", "episode_id"}
    if not required.issubset(observations):
        raise ValueError(f"observations require {sorted(required)}")
    required_path = {"code", "date", "is_limit_up", "indicator_status", *[v[0] for v in TARGETS.values()]}
    if not required_path.issubset(path):
        raise ValueError(f"path requires {sorted(required_path)}")
    sessions = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in calendar]
    if sessions != sorted(set(sessions)):
        raise ValueError("calendar must be sorted unique trading sessions")
    bars = path.copy()
    bars["code"] = bars["code"].astype(str).str.zfill(6)
    bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
    if bars.duplicated(["code", "date"]).any():
        raise ValueError("duplicate code/date in path")
    indexed = bars.set_index(["code", "date"])
    results = []
    for obs in observations.to_dict("records"):
        code = str(obs["code"]).zfill(6)
        signal = pd.Timestamp(obs["signal_date"]).strftime("%Y-%m-%d")
        if signal not in sessions:
            raise ValueError("signal_date missing from trading calendar")
        if pd.isna(obs["episode_id"]) or not str(obs["episode_id"]).strip():
            raise ValueError("episode_id required for cluster-aware evaluation")
        result = {**obs, "code": code, "signal_date": signal,
                  "contract_version": VERSION, "horizon_sessions": horizon,
                  "label_only": True}
        values = {key: None for key in TARGETS}
        status = "CENSORED_HISTORY"
        initial = indexed.loc[(code, signal)] if (code, signal) in indexed.index else None
        valid_initial = initial is not None and initial["indicator_status"] == "verified"
        if valid_initial:
            valid_initial = all(math.isfinite(float(pd.to_numeric(initial[col], errors="coerce")))
                                for col, _ in TARGETS.values())
        if valid_initial:
            already = any(float(initial[col]) >= threshold for col, threshold in TARGETS.values())
            status = "EXCLUDED_ALREADY_HIT" if already else "CENSORED_FUTURE"
            if not already:
                start = sessions.index(signal) + 1
                future = sessions[start:start + horizon]
                for step, date in enumerate(future, 1):
                    key = (code, date)
                    if key not in indexed.index:
                        status = "CENSORED_MISSING_SESSION"
                        break
                    row = indexed.loc[key]
                    limit_up = pd.to_numeric(row["is_limit_up"], errors="coerce")
                    if limit_up not in (0, 1):
                        status = "CENSORED_BOARD_STATUS"
                        break
                    if limit_up == 0:
                        values = {k: 0 if v is None else v for k, v in values.items()}
                        status = "COMPLETE_BROKEN_BOARD"
                        break
                    if row["indicator_status"] != "verified" or any(
                        not math.isfinite(float(pd.to_numeric(row[col], errors="coerce")))
                        for col, _ in TARGETS.values()
                    ):
                        status = "CENSORED_INDICATORS"
                        break
                    for target, (col, threshold) in TARGETS.items():
                        if values[target] is None and float(row[col]) >= threshold:
                            values[target] = 1
                            result[f"target_{target}_first_hit_date"] = date
                            result[f"target_{target}_sessions_to_hit"] = step
                    if all(v == 1 for v in values.values()):
                        status = "COMPLETE_BOTH_HIT"
                        break
                else:
                    if len(future) == horizon:
                        values = {k: 0 if v is None else v for k, v in values.items()}
                        status = "COMPLETE_HORIZON"
        result.update({f"target_{k}_label": v for k, v in values.items()})
        result["target_either_label"] = (1 if 1 in values.values() else
                                          0 if all(v == 0 for v in values.values()) else None)
        result["outcome_status"] = status
        results.append(result)
    return pd.DataFrame(results)


def evaluate_locked_signals(signals: pd.DataFrame, target: str = "either") -> dict:
    """Audit supplied blind signals; no threshold search or fitting here.

    Earliest selected signal per episode is evaluated so a winning seven-board
    run cannot masquerade as several independent winning leaders.
    """
    if target not in {*TARGETS, "either"}:
        raise ValueError("target must be 100, 200 or either")
    label = f"target_{target}_label"
    required = {"code", "signal_date", "episode_id", "selected", "split", label}
    if not required.issubset(signals):
        raise ValueError(f"signals require {sorted(required)}")
    selected = pd.to_numeric(signals["selected"], errors="coerce")
    if not selected.isin([0, 1]).all():
        raise ValueError("selected must be explicit 0/1")
    chosen = signals[selected.eq(1)].sort_values("signal_date")
    if not chosen["split"].eq("blind").all():
        raise ValueError("selected evaluation rows must be blind")
    if chosen[["code", "episode_id", "signal_date"]].isna().any().any():
        raise ValueError("signal identifiers must be complete")
    chosen = chosen.drop_duplicates(["code", "episode_id"], keep="first")
    labels = pd.to_numeric(chosen[label], errors="coerce")
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("labels must be 0, 1 or missing")
    n, wins = int(labels.notna().sum()), int(labels.eq(1).sum())
    censored = int(labels.isna().sum())
    precision = wins / n if n else None
    lower = None
    if n:
        z = 1.959963984540054
        center = precision + z*z/(2*n)
        spread = z * math.sqrt(precision*(1-precision)/n + z*z/(4*n*n))
        lower = (center-spread)/(1+z*z/n)
    empirical_pass = n >= 100 and censored == 0 and precision >= 0.8
    return {"contract_version": VERSION, "target": target,
            "independent_episodes": n, "hits": wins, "censored_episodes": censored,
            "precision": precision, "wilson_95_lower": lower,
            "empirical_80pct_gate": empirical_pass,
            "status": "DIAGNOSTIC_ONLY",
            "not_proven": ["immutable pre-signal model lock", "execution and profit win rate",
                           "forward paper validation", "independence across market cycles"],
            "note": "Target-hit precision is not profit win rate or a guarantee."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build = subs.add_parser("label")
    build.add_argument("--observations", type=Path, required=True)
    build.add_argument("--path", type=Path, required=True)
    build.add_argument("--calendar", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    audit = subs.add_parser("audit")
    audit.add_argument("--signals", type=Path, required=True)
    audit.add_argument("--target", choices=["100", "200", "either"], default="either")
    args = parser.parse_args()
    if args.command == "label":
        labels = build_labels(pd.read_csv(args.observations, dtype={"code": str}),
                              pd.read_csv(args.path, dtype={"code": str}),
                              pd.read_csv(args.calendar)["date"].tolist())
        if args.output.exists():
            raise FileExistsError("refusing to overwrite existing labels")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(args.output, index=False, encoding="utf-8-sig")
    else:
        print(json.dumps(evaluate_locked_signals(pd.read_csv(args.signals, dtype={"code": str}),
                                                 args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
