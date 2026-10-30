"""Earlier name-linked anchors and fixed prior cohorts, including today's losers.

Relations are deterministic name proxies, not verified market theme membership.
No future target or future parent selection is accepted by this feature engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from historical_symbolic_context import tokens
from first_board_ladder_emotion_features import reject_targets

KEYS = ["signal_date", "code"]
FIELDS = ["prior_name_anchor_present", "name_anchor_current_mean_return",
          "name_anchor_current_limit_survival", "name_prev_followers_current_mean_return",
          "name_prev_followers_current_loss7_share"]
ANCHOR_MIN_HEIGHT = 3
LOSS_CUTOFF = -.07


def build(candidates, closing_names, states, sessions):
    for frame in (candidates, closing_names, states):
        reject_targets(frame)
    if candidates.duplicated(KEYS).any() or closing_names.duplicated(KEYS).any() or states.duplicated(["p", "code"]).any():
        raise ValueError("Duplicate candidate or source identity")
    positions = {day: p for p, day in enumerate(sessions)}
    candidate = candidates.copy()
    candidate["_order"] = np.arange(len(candidate))
    named_groups = closing_names.groupby("p", sort=False)
    state_groups = states.groupby("p", sort=False)
    output = []
    for day, today in candidate.groupby("signal_date", sort=False):
        p = positions[day]
        previous = named_groups.get_group(p-1) if p-1 in named_groups.indices else closing_names.iloc[:0]
        previous_price = state_groups.get_group(p-1) if p-1 in state_groups.indices else states.iloc[:0]
        observed_previous = bool((previous_price.eligible & previous_price.valid_price).any())
        names_complete = bool(previous.name_ok.all())
        complete = observed_previous and names_complete
        token_groups = {}
        token_anchors = {}
        heights = dict(zip(previous.code, previous.height))
        if complete:
            for row in previous.itertuples(index=False):
                for token in tokens(row.historical_name):
                    token_groups.setdefault(token, set()).add(row.code)
                    if row.height >= ANCHOR_MIN_HEIGHT:
                        token_anchors.setdefault(token, set()).add(row.code)
        prices = state_groups.get_group(p) if p in state_groups.indices else states.iloc[:0]
        current = {}
        for row in prices.itertuples(index=False):
            known = bool(row.eligible and row.valid_price and row.regular_price)
            current[row.code] = (known, float(row.day_return), bool(row.limit))

        def feedback(codes):
            values = [current.get(code, (False, np.nan, False)) for code in sorted(codes)]
            unknown = sum(not x[0] for x in values)
            if not values or unknown:
                return np.nan, np.nan, np.nan, unknown
            return (float(np.mean([x[1] for x in values])), float(np.mean([x[2] for x in values])),
                    float(np.mean([x[1] <= LOSS_CUTOFF for x in values])), 0)

        for row in today.to_dict("records"):
            known = complete and bool(row["name_ok"])
            matched = sorted(token for token in tokens(row["historical_name"])
                             if token_anchors.get(token)) if known else []
            anchors = set().union(*(token_anchors[token] for token in matched)) if matched else set()
            members = set().union(*(token_groups[token] for token in matched)) if matched else set()
            anchors.discard(row["code"])
            followers = {code for code in members if heights[code] in (1, 2)}
            followers.discard(row["code"])
            ar, alive, _, au = feedback(anchors)
            fr, _, loss, fu = feedback(followers)
            record = {key: row[key] for key in KEYS}
            record.update(_order=row["_order"], **dict(zip(FIELDS, [float(bool(anchors)) if known else np.nan,
                ar if known else np.nan, alive if known else np.nan, fr if known else np.nan, loss if known else np.nan])))
            record.update(previous_source_observed_DIAGNOSTIC=observed_previous,
                previous_pool_names_complete_DIAGNOSTIC=names_complete,
                previous_pool_members_DIAGNOSTIC=len(previous),
                previous_pool_name_unknown_DIAGNOSTIC=int((~previous.name_ok).sum()),
                candidate_historical_name_DIAGNOSTIC=row["historical_name"],
                matched_tokens_DIAGNOSTIC="|".join(matched),
                anchor_codes_DIAGNOSTIC="|".join(sorted(anchors)),
                follower_codes_DIAGNOSTIC="|".join(sorted(followers)),
                anchor_members_DIAGNOSTIC=len(anchors), anchor_unknown_DIAGNOSTIC=au,
                follower_members_DIAGNOSTIC=len(followers), follower_unknown_DIAGNOSTIC=fu)
            output.append(record)
    result = pd.DataFrame(output).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    assert result[KEYS].equals(candidates[KEYS].reset_index(drop=True))
    return result
