"""Causal theme lifecycle and candidate relative status within theme lines.

Strictly no future targets, labels, or hindsight dragon identities.
Uses only qualified closing cohorts, historical dated names, causal price states,
and qualified day S seal facts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from historical_symbolic_context import tokens
from first_board_ladder_emotion_features import reject_targets

KEYS = ["signal_date", "code"]

LIFECYCLE_FIELDS = [
    "name_anchor_broken_age_sessions",
    "name_anchor_decay_from_peak",
    "name_fixed_followers_return_acceleration",
]

STATUS_FIELDS = [
    "theme_candidate_cohort_size",
    "theme_first_seal_rank_in_group",
    "theme_is_earliest_sealer",
    "theme_close_seal_rank_in_group",
]

ALL_FIELDS = LIFECYCLE_FIELDS + STATUS_FIELDS
ANCHOR_MIN_HEIGHT = 3
ANCHOR_MEMORY_SESSIONS = 3  # [p-3, p-1]


def build(candidates, closing_names, seal_facts, states, sessions):
    for frame in (candidates, closing_names, seal_facts, states):
        reject_targets(frame)
    if candidates.duplicated(KEYS).any() or closing_names.duplicated(KEYS).any():
        raise ValueError("Duplicate candidate or closing identity")
    if seal_facts.duplicated(KEYS).any() or states.duplicated(["p", "code"]).any():
        raise ValueError("Duplicate seal fact or state identity")

    positions = {day: p for p, day in enumerate(sessions)}
    candidate = candidates.copy()
    candidate["_order"] = np.arange(len(candidate))

    # Merge seal facts for day S candidates
    seal_cols = ["signal_date", "code", "reported_first_seal_trading_minutes", "reported_close_seal_log_amount_ratio"]
    candidate = candidate.merge(seal_facts[seal_cols], on=KEYS, how="left", validate="one_to_one")

    named_groups = closing_names.groupby("p", sort=False)
    state_groups = states.groupby("p", sort=False)
    output = []

    # Fast lookup for states: (p, code) -> (known, return, limit, height)
    # Pre-index state_groups for rapid retrieval
    state_dict = {}
    for p, grp in state_groups:
        cur = {}
        for r in grp.itertuples(index=False):
            known = bool(r.eligible and r.valid_price and r.regular_price)
            cur[r.code] = (known, float(r.day_return), bool(r.limit), int(r.height))
        state_dict[p] = cur

    for day, today in candidate.groupby("signal_date", sort=False):
        p = positions[day]

        # Check source observation for prior 3 sessions
        prior_ok = True
        for lag in range(1, ANCHOR_MEMORY_SESSIONS + 1):
            cur_p = p - lag
            if cur_p not in named_groups.indices or cur_p not in state_dict:
                prior_ok = False
                break
            prev_names = named_groups.get_group(cur_p)
            if not prev_names.name_ok.all():
                prior_ok = False
                break

        # Discover anchors in [p-3, p-1] by token
        # token -> list of (code, peak_height, last_limit_p)
        token_anchor_records = {}
        # Also discover followers at p-1: token -> list of codes with height in (1, 2)
        token_followers_p1 = {}

        if prior_ok:
            for lag in range(ANCHOR_MEMORY_SESSIONS, 0, -1):
                cur_p = p - lag
                prev_closing = named_groups.get_group(cur_p)
                for r in prev_closing.itertuples(index=False):
                    if r.height >= ANCHOR_MIN_HEIGHT:
                        for token in tokens(r.historical_name):
                            token_anchor_records.setdefault(token, {})[r.code] = max(
                                token_anchor_records.get(token, {}).get(r.code, 0), r.height
                            )
            # Followers at p-1
            p1_closing = named_groups.get_group(p - 1)
            for r in p1_closing.itertuples(index=False):
                if r.height in (1, 2):
                    for token in tokens(r.historical_name):
                        token_followers_p1.setdefault(token, set()).add(r.code)

        # Precompute per-candidate active anchors and matched tokens
        cand_records = today.to_dict("records")
        cand_matched_tokens = {}
        cand_anchors = {}
        cand_followers = {}

        for row in cand_records:
            code = row["code"]
            known = prior_ok and bool(row["name_ok"])
            if known:
                my_tokens = sorted(t for t in tokens(row["historical_name"]) if t in token_anchor_records)
                cand_matched_tokens[code] = my_tokens
                # Union of anchor codes
                anc = set()
                for t in my_tokens:
                    anc.update(token_anchor_records[t].keys())
                anc.discard(code)
                cand_anchors[code] = anc

                # Union of followers at p-1
                foll = set()
                for t in my_tokens:
                    if t in token_followers_p1:
                        foll.update(token_followers_p1[t])
                foll.discard(code)
                cand_followers[code] = foll
            else:
                cand_matched_tokens[code] = []
                cand_anchors[code] = set()
                cand_followers[code] = set()

        # Day S current state lookup
        p_states = state_dict.get(p, {})
        p1_states = state_dict.get(p - 1, {})

        # Compute Direction 1: Lifecycle per candidate
        lifecycle_results = {}
        for row in cand_records:
            code = row["code"]
            anchors = cand_anchors[code]
            followers = cand_followers[code]

            if not anchors:
                lifecycle_results[code] = (np.nan, np.nan, np.nan)
            else:
                # 1. Broken age & 2. Peak decay
                # For each anchor code, find its status at p-1 and peak height
                broken_ages = []
                decays = []
                for a_code in anchors:
                    # peak height
                    peak_h = max(token_anchor_records[t].get(a_code, 0)
                                 for t in cand_matched_tokens[code] if a_code in token_anchor_records.get(t, {}))
                    # status at p-1
                    st_p1 = p1_states.get(a_code, (False, np.nan, False, 0))
                    if st_p1[2] and st_p1[3] >= ANCHOR_MIN_HEIGHT:  # limit at p-1
                        b_age = 0
                        decay = max(0, peak_h - st_p1[3])
                    else:
                        # broke at p-1 or p-2
                        st_p2 = state_dict.get(p - 2, {}).get(a_code, (False, np.nan, False, 0))
                        if st_p2[2] and st_p2[3] >= ANCHOR_MIN_HEIGHT:
                            b_age = 1  # broke at p-1
                        else:
                            b_age = 2  # broke at p-2
                        decay = peak_h  # dropped to 0 height
                    broken_ages.append(b_age)
                    decays.append(decay)

                min_b_age = float(min(broken_ages)) if broken_ages else np.nan
                mean_decay = float(np.mean(decays)) if decays else np.nan

                # 3. Followers return acceleration: return(p) - return(p-1) on fixed cohort
                if followers:
                    vals_p = [p_states.get(fc, (False, np.nan, False, 0))[1] for fc in followers]
                    vals_p1 = [p1_states.get(fc, (False, np.nan, False, 0))[1] for fc in followers]
                    if all(np.isfinite(vals_p)) and all(np.isfinite(vals_p1)):
                        accel = float(np.mean(vals_p) - np.mean(vals_p1))
                    else:
                        accel = np.nan
                else:
                    accel = np.nan

                lifecycle_results[code] = (min_b_age, mean_decay, accel)

        # Compute Direction 2: Relative Status among today's first-board candidates
        # Group candidates by anchor token
        token_cand_map = {}
        for row in cand_records:
            code = row["code"]
            for t in cand_matched_tokens[code]:
                token_cand_map.setdefault(t, []).append(row)

        status_results = {}
        for row in cand_records:
            code = row["code"]
            matched = cand_matched_tokens[code]
            if not matched:
                status_results[code] = (np.nan, np.nan, np.nan, np.nan)
                continue

            # Cohort of all candidates sharing ANY of my matched anchor tokens
            peer_codes = set()
            for t in matched:
                for peer in token_cand_map.get(t, []):
                    peer_codes.add(peer["code"])

            N = len(peer_codes)
            cohort_size = float(N)

            if N == 1:
                # Sole candidate in this anchored theme
                status_results[code] = (cohort_size, np.nan, 1.0, np.nan)
            else:
                peer_rows = [r for r in cand_records if r["code"] in peer_codes]
                # Sort by first seal time (ascending)
                seals = [(r["code"], r["reported_first_seal_trading_minutes"]) for r in peer_rows]
                # Rank of this code
                seals.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else 999999, x[0]))
                seal_rank_idx = [x[0] for x in seals].index(code)
                seal_rel_rank = float(seal_rank_idx / (N - 1))
                is_earliest = 1.0 if seal_rank_idx == 0 else 0.0

                # Sort by close seal ratio (descending: larger is stronger)
                close_seals = [(r["code"], r["reported_close_seal_log_amount_ratio"]) for r in peer_rows]
                close_seals.sort(key=lambda x: (-x[1] if np.isfinite(x[1]) else 999999, x[0]))
                close_rank_idx = [x[0] for x in close_seals].index(code)
                close_rel_rank = float(close_rank_idx / (N - 1))

                status_results[code] = (cohort_size, seal_rel_rank, is_earliest, close_rel_rank)

        # Assemble output rows
        for row in cand_records:
            code = row["code"]
            rec = {k: row[k] for k in KEYS}
            rec["_order"] = row["_order"]
            lc = lifecycle_results[code]
            st = status_results[code]
            rec["name_anchor_broken_age_sessions"] = lc[0]
            rec["name_anchor_decay_from_peak"] = lc[1]
            rec["name_fixed_followers_return_acceleration"] = lc[2]
            rec["theme_candidate_cohort_size"] = st[0]
            rec["theme_first_seal_rank_in_group"] = st[1]
            rec["theme_is_earliest_sealer"] = st[2]
            rec["theme_close_seal_rank_in_group"] = st[3]
            output.append(rec)

    res = pd.DataFrame(output).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    assert res[KEYS].equals(candidates[KEYS].reset_index(drop=True))
    assert len(res) == len(candidates)
    return res
