"""Training-only, censored remaining-relay duration baseline; not holding advice.

An episode contributes once at each board-height landmark. A broken board is
an event, not a sample to discard. Missing future quotes are right censored.
This OHLC-derived survival curve does not predict executable exit time or PnL.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def duration_rows(observations, episodes, calendar, exclusive_cutoff):
    if calendar != sorted(set(calendar)) or exclusive_cutoff not in calendar:
        raise ValueError("sorted unique calendar and in-calendar cutoff required")
    positions = {d: i for i, d in enumerate(calendar)}
    end_position = positions[exclusive_cutoff] - 1
    metadata = episodes.set_index("episode_id")
    if not metadata.index.is_unique:
        raise ValueError("duplicate episode")
    selected = observations[observations.signal_date < exclusive_cutoff].copy()
    if selected.duplicated(["episode_id", "boards"]).any():
        raise ValueError("duplicate height landmark within episode")
    rows = []
    excluded_unknown = 0
    for obs in selected.to_dict("records"):
        if str(obs["start_known"]).lower() not in {"true", "1"}:
            excluded_unknown += 1
            continue
        ep = metadata.loc[obs["episode_id"]]
        if ep.code != obs["code"]:
            raise ValueError("episode code mismatch")
        start = positions[obs["signal_date"]]
        last = min(positions[ep.last_limit_date], end_position)
        terminal = ep.terminal_date
        broken = (ep.termination == "BROKEN_BOARD" and pd.notna(terminal)
                  and terminal < exclusive_cutoff)
        if broken:
            stop = positions[terminal]
            if stop != positions[ep.last_limit_date] + 1:
                raise ValueError("broken-board event must follow last observed limit session")
        else:
            stop = last
        if stop < start or (broken and stop == start):
            raise ValueError("outcome precedes landmark")
        rows.append(dict(code=obs["code"], episode_id=obs["episode_id"],
                         signal_date=obs["signal_date"], boards=int(obs["boards"]),
                         duration_sessions=stop-start, broken_observed=broken,
                         censor_reason=None if broken else "training_boundary_or_incomplete_path",
                         label_only=True))
    return pd.DataFrame(rows), excluded_unknown


def survival_curve(rows, horizon=10):
    if horizon < 1:
        raise ValueError("positive horizon required")
    if rows.empty:
        return dict(episodes=0, curve=[], restricted_mean_future_limit_sessions=None)
    if rows.episode_id.duplicated().any():
        raise ValueError("curve requires one landmark per episode")
    survival, curve = 1.0, []
    for step in range(1, horizon + 1):
        at_risk = int((rows.duration_sessions >= step).sum())
        events = int(((rows.duration_sessions == step) & rows.broken_observed).sum())
        if at_risk:
            survival *= 1 - events / at_risk
        elif survival != 0:
            survival = None
        curve.append(dict(future_session=step, at_risk=at_risk, broken_events=events,
                          relay_survival_estimate=survival))
    supported = all(row["relay_survival_estimate"] is not None for row in curve)
    return dict(episodes=len(rows), broken_episodes=int(rows.broken_observed.sum()),
                censored_episodes=int((~rows.broken_observed).sum()), curve=curve,
                restricted_mean_future_limit_sessions=(sum(r["relay_survival_estimate"] for r in curve)
                                                       if supported else None))


def run(observations_path, episodes_path, calendar_path, validation_start, embargo, output):
    calendar = pd.read_csv(calendar_path).date.tolist()
    if embargo < 1 or validation_start not in calendar or calendar.index(validation_start) < embargo:
        raise ValueError("invalid validation boundary or embargo")
    cutoff = calendar[calendar.index(validation_start)-embargo]
    observations = pd.read_csv(observations_path, dtype={"code": str})
    episodes = pd.read_csv(episodes_path, dtype={"code": str})
    rows, excluded = duration_rows(observations, episodes, calendar, cutoff)
    report = dict(status="DIAGNOSTIC_ONLY", validation_start=validation_start,
                  embargo_sessions=embargo, training_exclusive_cutoff=cutoff,
                  excluded_unknown_start_rows=excluded, train_landmarks=len(rows),
                  per_height={str(height): survival_curve(group) for height, group in rows.groupby("boards")},
                  expected_holding_sessions=None, actual_profit_win_rate=None,
                  target_100_probability=None, target_200_probability=None,
                  limitations=["OHLC relay duration is not holding duration or tradable profit",
                               "not leader-selected; all known-start candidates at each landmark",
                               "censoring may be informative; survival estimates not calibrated forecasts",
                               "current-universe survivorship and historical status evidence limitations",
                               "no validation or blind outcome used to fit this descriptive baseline",
                               "different heights overlap episodes and are not independent samples"],
                  source_hashes={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in [observations_path, episodes_path, calendar_path]})
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    rows.to_csv(directory / "duration_labels_TRAIN_ONLY.csv", index=False, encoding="utf-8-sig")
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "per_height"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--embargo", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.observations, args.episodes, args.calendar,
                         args.validation_start, args.embargo, args.output), ensure_ascii=False, indent=2))
