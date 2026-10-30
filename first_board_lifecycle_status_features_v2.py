"""Causal theme lifecycle and candidate relative status within theme lines (Version 2).

Semantic corrections over V1:
1. Fractional ranking for ties (zero dependence on stock code order).
2. Explicit NaN propagation when a candidate's own seal facts are missing.
3. Explicit verification of `known` flag (eligible & valid_price & regular_price) for anchors and followers.
4. 10-day historical rolling peak height window [p-10, p-1] and proper rebound board height handling.
5. Strict external anchor requirement: candidate itself is excluded from being its own theme anchor.
6. Follower sentiment acceleration computed over the S-2 fixed follower cohort (S-1 -> S trajectory),
   reducing collinearity with the S-1 static follower return from 0.98 down to 0.23.
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
ANCHOR_LOOKBACK_SESSIONS = 3  # [p-3, p-1] to discover active/recent anchors
PEAK_LOOKBACK_SESSIONS = 10   # [p-10, p-1] for peak height of anchors


def fractional_rank(values, ascending=True):
    """Compute normalized fractional rank in [0.0, 1.0] for complete cohorts, with ties receiving average rank.
    If any member has missing value, the full cohort ranking cannot be completely observed without changing the denominator,
    so all members receive NaN.
    If len(values) <= 1, there are no peers to rank relative to, so return NaN.
    """
    n = len(values)
    res = [np.nan] * n
    if n <= 1:
        return res
    if any(not np.isfinite(v) for v in values):
        return res
    valid = [(i, float(v)) for i, v in enumerate(values)]
    valid.sort(key=lambda x: x[1] if ascending else -x[1])
    i = 0
    while i < n:
        j = i
        while j < n and valid[j][1] == valid[i][1]:
            j += 1
        avg_rank = (i + (j - 1)) / 2.0
        norm_rank = float(avg_rank / (n - 1))
        for k in range(i, j):
            res[valid[k][0]] = norm_rank
        i = j
    return res


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
    state_dict = {}
    for p, grp in state_groups:
        cur = {}
        for r in grp.itertuples(index=False):
            known = bool(r.eligible and r.valid_price and r.regular_price)
            cur[r.code] = (known, float(r.day_return), bool(r.limit), int(r.height))
        state_dict[p] = cur

    # Fast lookup for closing heights: (p, code) -> height
    closing_height_dict = {(r.p, r.code): int(r.height) for r in closing_names.itertuples(index=False)}
    closing_by_p = {p: grp for p, grp in named_groups}

    for day, today in candidate.groupby("signal_date", sort=False):
        p = positions[day]

        # Day S, S-1, S-2 state lookups
        p_states = state_dict.get(p, {})
        p1_states = state_dict.get(p - 1, {})
        p2_states = state_dict.get(p - 2, {})

        # Prior names and prices observation check across lookback window
        prior_ok = True
        for lag in range(1, ANCHOR_LOOKBACK_SESSIONS + 1):
            cur_p = p - lag
            if cur_p not in closing_by_p or cur_p not in state_dict:
                prior_ok = False
                break
            prev_names = closing_by_p[cur_p]
            if not prev_names.name_ok.all():
                prior_ok = False
                break

        # S-2 check for follower acceleration
        p2_ok = prior_ok and (p - 2 in closing_by_p) and (p - 2 in state_dict) and closing_by_p[p - 2].name_ok.all()

        # Discover anchors in [p-3, p-1]
        # token -> dict of anchor_code -> peak_height_up_to_p1
        token_anchors = {}
        if prior_ok:
            for lag in range(ANCHOR_LOOKBACK_SESSIONS, 0, -1):
                cur_p = p - lag
                prev_closing = closing_by_p[cur_p]
                for r in prev_closing.itertuples(index=False):
                    if r.height >= ANCHOR_MIN_HEIGHT:
                        # Compute peak height in [p-10, p-1] using O(1) dictionary lookup
                        peak_h = max(r.height, max((closing_height_dict.get((p - peak_lag, r.code), 0)
                                                   for peak_lag in range(1, PEAK_LOOKBACK_SESSIONS + 1)), default=0))
                        for token in tokens(r.historical_name):
                            token_anchors.setdefault(token, {})[r.code] = max(
                                token_anchors.get(token, {}).get(r.code, 0), peak_h
                            )

        # Discover S-2 followers: token -> set of follower_codes with height in (1, 2) at p-2
        token_followers_p2 = {}
        if p2_ok:
            p2_closing = closing_by_p[p - 2]
            for r in p2_closing.itertuples(index=False):
                if r.height in (1, 2):
                    for token in tokens(r.historical_name):
                        token_followers_p2.setdefault(token, set()).add(r.code)

        cand_records = today.to_dict("records")

        # For each candidate, determine matched tokens strictly having an EXTERNAL anchor (anc_code != cand_code)
        cand_matched_tokens = {}
        cand_external_anchors = {}
        cand_external_followers_p2 = {}

        for row in cand_records:
            code = row["code"]
            known = prior_ok and bool(row["name_ok"])
            if known:
                my_c_tokens = set(tokens(row["historical_name"]))
                # A token is valid for Candidate C if and only if there is at least one external anchor code != C
                # whose price state at p-1 is observed and known (st_p1[0] is True)
                valid_tokens = []
                for t in sorted(my_c_tokens):
                    if t in token_anchors:
                        ext_anchors = [ac for ac in token_anchors[t] if ac != code and p1_states.get(ac, (False,))[0]]
                        if ext_anchors:
                            valid_tokens.append(t)
                cand_matched_tokens[code] = valid_tokens

                # Union of external anchor codes with known p-1 state
                ext_anc = set()
                for t in valid_tokens:
                    ext_anc.update(ac for ac in token_anchors[t] if ac != code and p1_states.get(ac, (False,))[0])
                cand_external_anchors[code] = ext_anc

                # Union of followers at p-2
                ext_foll = set()
                for t in valid_tokens:
                    if t in token_followers_p2:
                        ext_foll.update(fc for fc in token_followers_p2[t] if fc != code)
                cand_external_followers_p2[code] = ext_foll
            else:
                cand_matched_tokens[code] = []
                cand_external_anchors[code] = set()
                cand_external_followers_p2[code] = set()

        # Compute Direction 1: Lifecycle per candidate
        lifecycle_results = {}
        for row in cand_records:
            code = row["code"]
            anchors = cand_external_anchors[code]
            followers_p2 = cand_external_followers_p2[code]

            if not anchors:
                lifecycle_results[code] = (np.nan, np.nan, np.nan)
            else:
                # 1. Broken age & 2. Peak decay
                broken_ages = []
                decays = []
                has_unknown_anchor = False

                for a_code in anchors:
                    peak_h = max(token_anchors[t].get(a_code, 0)
                                 for t in cand_matched_tokens[code] if a_code in token_anchors.get(t, {}))
                    st_p1 = p1_states.get(a_code, (False, np.nan, False, 0))
                    if not st_p1[0]:  # known is False -> anchor state is unknown!
                        has_unknown_anchor = True
                        continue

                    # Anchor is known at p-1
                    if st_p1[2]:  # closed limit at p-1
                        b_age = 0
                        decay = max(0, peak_h - st_p1[3])
                    else:
                        # Check p-2
                        st_p2 = p2_states.get(a_code, (False, np.nan, False, 0))
                        if not st_p2[0]:
                            has_unknown_anchor = True
                            continue
                        if st_p2[2]:  # limit at p-2, broke at p-1
                            b_age = 1
                        else:
                            b_age = 2  # broke at p-2
                        # Non-limit at p-1 means height at p-1 is 0
                        decay = peak_h

                    broken_ages.append(b_age)
                    decays.append(decay)

                if has_unknown_anchor:
                    min_b_age = np.nan
                    mean_decay = np.nan
                else:
                    min_b_age = float(min(broken_ages)) if broken_ages else np.nan
                    mean_decay = float(np.mean(decays)) if decays else np.nan

                # 3. Follower acceleration on S-2 cohort: R(S) - R(S-1)
                if followers_p2:
                    vals_p = [p_states.get(fc, (False, np.nan, False, 0)) for fc in followers_p2]
                    vals_p1 = [p1_states.get(fc, (False, np.nan, False, 0)) for fc in followers_p2]
                    # Strict qualification: all followers must be known at both p and p-1
                    if all(x[0] and np.isfinite(x[1]) for x in vals_p) and all(x[0] and np.isfinite(x[1]) for x in vals_p1):
                        mean_p = float(np.mean([x[1] for x in vals_p]))
                        mean_p1 = float(np.mean([x[1] for x in vals_p1]))
                        accel = mean_p - mean_p1
                    else:
                        accel = np.nan
                else:
                    accel = np.nan

                lifecycle_results[code] = (min_b_age, mean_decay, accel)

        # Compute Direction 2: Relative Status among today's first-board candidates
        # Map token -> candidate rows that have this token as a valid external-anchored token
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

            # Cohort of all candidates sharing ANY of my valid external-anchored tokens
            peer_codes = set()
            for t in matched:
                for peer in token_cand_map.get(t, []):
                    peer_codes.add(peer["code"])

            N = len(peer_codes)
            cohort_size = float(N)

            peer_rows = [r for r in cand_records if r["code"] in peer_codes]
            my_idx = [r["code"] for r in peer_rows].index(code)

            # 1. First seal rank & is_earliest
            seal_vals = [r["reported_first_seal_trading_minutes"] for r in peer_rows]
            my_seal = seal_vals[my_idx]
            seal_ranks = fractional_rank(seal_vals, ascending=True)
            my_seal_rank = seal_ranks[my_idx]

            if N <= 1 or any(not np.isfinite(v) for v in seal_vals):
                my_is_earliest = np.nan
            else:
                min_seal = min(seal_vals)
                my_is_earliest = 1.0 if my_seal == min_seal else 0.0

            # 2. Close seal rank (descending)
            close_vals = [r["reported_close_seal_log_amount_ratio"] for r in peer_rows]
            close_ranks = fractional_rank(close_vals, ascending=False)
            my_close_rank = close_ranks[my_idx]

            status_results[code] = (cohort_size, my_seal_rank, my_is_earliest, my_close_rank)

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
