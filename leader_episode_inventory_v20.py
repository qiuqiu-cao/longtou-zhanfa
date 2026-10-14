"""Build candidate denominators independent of future threshold success.

Daily board recognition is a rounded-price proxy; security status is consumed
from the input evidence layer, never inferred from today's stock name.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def true(value):
    return str(value).lower() in {"true", "1"}


def inventory_one(frame, calendar, min_boards=2):
    if min_boards not in (1, 2):
        raise ValueError("min_boards must be 1 (full limit-up pool) or 2 (legacy relay)")
    frame = frame.sort_values("date").copy()
    if frame.date.duplicated().any():
        raise ValueError("duplicate stock date")
    positions = {date: i for i, date in enumerate(calendar)}
    observations, episodes, active = [], [], None
    previous_position, previous_known_nonlimit = None, False

    def finish(reason, terminal_date):
        nonlocal active
        if active is not None:
            episodes.append({**active, "termination": reason, "terminal_date": terminal_date,
                             "label_only": True})
            active = None

    for row in frame.itertuples(index=False):
        if row.date not in positions:
            raise ValueError("stock date absent from benchmark calendar")
        position = positions[row.date]
        consecutive = previous_position is not None and position == previous_position + 1
        if not consecutive:
            finish("CENSORED_GAP", row.date)
            previous_known_nonlimit = False
        previous_position = position
        flags_known = all(str(v).lower() in {"true", "false", "0", "1"}
                          for v in (row.status_verified, row.st, row.new_listing))
        eligible = flags_known and true(row.status_verified) and not true(row.st) and not true(row.new_listing)
        numeric = np.isfinite(row.close) and np.isfinite(row.pre_close) and row.close > 0 and row.pre_close > 0
        if not eligible or not numeric:
            finish("CENSORED_STATUS_OR_PRICE", row.date)
            previous_known_nonlimit = False
            continue
        limit = np.floor(row.pre_close * 110 + 0.5 + 1e-8) / 100
        is_limit = abs(row.close - limit) < 1e-6
        if not is_limit:
            finish("BROKEN_BOARD", row.date)
            previous_known_nonlimit = True
            continue
        if active is None:
            active = dict(code=row.code, episode_id=f"{row.code}:{row.date}", start_date=row.date,
                          start_known=previous_known_nonlimit, observed_boards=0, last_limit_date=row.date)
        active["observed_boards"] += 1
        active["last_limit_date"] = row.date
        previous_known_nonlimit = False
        if active["observed_boards"] >= min_boards:
            observations.append(dict(code=row.code, signal_date=row.date, episode_id=active["episode_id"],
                                     boards=active["observed_boards"], start_known=active["start_known"],
                                     candidate_scope="mainboard_nonst_status_evidenced",
                                     board_status="inferred_rounded_price", valid_identity=None,
                                     target_label=None))
    finish("CENSORED_END_OF_DATA", None)
    return observations, episodes


def run(bars, calendar_path, output, min_boards=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(bars, usecols=["date", "code", "close", "pre_close", "status_verified", "st", "new_listing"],
                        dtype={"code": str})
    frame = frame[frame.code.str.match(r"^(00|60)")]
    calendar = pd.read_csv(calendar_path).date.tolist()
    if calendar != sorted(set(calendar)):
        raise ValueError("invalid calendar")
    all_observations, all_episodes = [], []
    for _, group in frame.groupby("code", sort=True):
        observations, episodes = inventory_one(group, calendar, min_boards=min_boards)
        all_observations.extend(observations)
        all_episodes.extend(episodes)
    obs, eps = pd.DataFrame(all_observations), pd.DataFrame(all_episodes)
    if min_boards == 1 and len(obs):
        # Unknown starts can be higher boards: never label them first boards.
        obs["board_group"] = np.where(~obs.start_known, "UNKNOWN_LEFT_CENSORED",
                                      np.where(obs.boards.eq(1), "FIRST_BOARD", "CONSECUTIVE_BOARD"))
        obs["observed_boards_lower_bound"] = obs["boards"]
        obs["boards"] = obs["boards"].where(obs.start_known).astype("Int64")
    obs.to_csv(output / "candidate_observations.csv", index=False, encoding="utf-8-sig")
    eps.to_csv(output / "episode_outcomes_LABEL_ONLY.csv", index=False, encoding="utf-8-sig")
    report = dict(status="DIAGNOSTIC_ONLY", min_boards=min_boards, input_rows=len(frame), codes=frame.code.nunique(),
                  first_date=frame.date.min(), last_date=frame.date.max(), candidate_rows=len(obs),
                  candidate_episodes=obs.episode_id.nunique() if len(obs) else 0,
                  left_censored_candidate_rows=int((~obs.start_known).sum()) if len(obs) else 0,
                  verified_target_labels=0,
                  limitations=["input historical status evidence not reaudited here", "rounded daily board proxy",
                               "no intraday buyability", "no leader identity classification yet",
                               "incomplete/delisted universe coverage", "benchmark calendar not independently verified"],
                  source_hashes={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [bars, calendar_path]})
    if min_boards == 1:
        report["board_group_counts"] = obs.board_group.value_counts().to_dict() if len(obs) else {}
        report["historical_reconstruction_only"] = True
        report["original_point_in_time_capture_available"] = False
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-boards", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.bars, args.calendar, args.output, args.min_boards), ensure_ascii=False, indent=2))
