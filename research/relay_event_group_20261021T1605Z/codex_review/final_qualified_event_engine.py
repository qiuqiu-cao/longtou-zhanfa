"""
Final Corrected Event-Anchored Cohort & Relative Status Engine
=============================================================
Incorporates the 3 final source qualification fixes before freeze:
1. Exact Price, Volume & Limit Logic:
   - Uses short_term_thinking_dataset_builder.cents.
   - Requires positive volume (>0), positive amount (>0), positive cents, and OHLC consistency.
   - Strict 110% rounded price limit check. Mismatch between price and pool assignment marked UNKNOWN.
   - Non-mainboard or ST regime observations marked UNKNOWN.
2. Interval Observed & Historical Eligibility:
   - DatedNames resolves interval_observed and historical_eligible.
   - Known eligible (interval_observed=1, historical_eligible=True) forms known cohort members.
   - Known ineligible (historical_eligible=False) excluded from mainboard cohort.
   - Unresolved stocks (interval_observed=0 or historical_eligible=None) tracked in unresolved_roster.
   - If unresolved_roster is non-empty, denominator N has uncertainty -> damage/repair/share ratios set to NaN.
   - Candidate on S-1: if interval_observed=0 or historical_eligible is None, matched is set to NaN.
3. Normalized Observations & Complete Ledger:
   - Outputs MEMBER_DAILY_OBSERVATIONS.csv (normalized daily status).
   - Outputs EVENT_GROUP_LEDGER.csv with full 9,245 rows, explicit bounds [k/N, (k+u)/N], and unresolved lists.
   - Causal prefix invariance verified with physical input slicing.
   - Strictly 0 model fits, 0 network requests.
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import time
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
import numpy as np
import joblib

ROOT = r"d:\Documents\ChatGPT\龙头战法"
sys.path.insert(0, ROOT)

from historical_symbolic_context import DatedNames, tokens
import short_term_thinking_dataset_builder as builder

DATASET_PATH = os.path.join(ROOT, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\historical_final\dataset.joblib")
MASTER_PATH = os.path.join(ROOT, r"data\v4\security_master_intervals.csv")
OUT_DIR = os.path.join(ROOT, r"research\relay_event_group_20261021T1605Z\qualified_final")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def clean_json(v):
    if isinstance(v, dict):
        return {str(k): clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return None
        return round(float(v), 6)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v

def cents(value):
    try:
        ans = builder.cents(value)
        return ans if (ans is not None and ans > 0) else None
    except (ValueError, TypeError, ArithmeticError):
        return None

def limit_cents(previous_cents, multiple_str):
    if previous_cents is None:
        return None
    try:
        return int((Decimal(previous_cents) * Decimal(multiple_str)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    except Exception:
        return None

def build_corrected_features_and_ledger(sessions, attempts, bars, features, master_df, cutoff_date=None):
    if cutoff_date is not None:
        sessions = [s for s in sessions if s <= cutoff_date]
        features = features[features['date'] <= cutoff_date].copy()
        attempts = attempts[attempts['signal_date'] <= cutoff_date].copy()
        bars = bars[bars['date'] <= cutoff_date].copy()
        master_df = master_df[master_df['valid_from'] <= cutoff_date].copy()

    s_idx = {s: i for i, s in enumerate(sessions)}
    dn = DatedNames(master_df)

    # 1. Pre-index closing limit-ups and failed board attempts
    active_by_day = defaultdict(set)
    failed_board_attempts = set()
    closing_limit_ups = defaultdict(dict) # day -> {code: board_height}

    attempts_clean = attempts.copy()
    attempts_clean['code'] = attempts_clean.code.astype(str).str.zfill(6)

    for r in attempts_clean[attempts_clean.pool_assignment.isin(['CLOSING_LIMIT_UP', 'FAILED_BOARD_OBSERVATION'])].itertuples():
        day, c = r.signal_date, r.code
        if day in s_idx:
            active_by_day[day].add(c)
            if r.pool_assignment == 'CLOSING_LIMIT_UP':
                closing_limit_ups[day][c] = int(r.board_height)
            elif r.pool_assignment == 'FAILED_BOARD_OBSERVATION':
                failed_board_attempts.add((day, c))

    features_clean = features.copy()
    features_clean['code'] = features_clean.code.astype(str).str.zfill(6)
    for r in features_clean.itertuples():
        day, c = r.date, r.code
        if day in s_idx and c not in closing_limit_ups[day]:
            closing_limit_ups[day][c] = int(r.boards)
            active_by_day[day].add(c)

    # 2. Process bars and qualify daily observation states
    bar_obs = {}
    normalized_obs_rows = []

    for r in bars.itertuples():
        day, c = r.date, str(r.code).zfill(6)
        if day not in s_idx:
            continue

        res = dn.resolve(c, day, getattr(r, 'name', ''))
        nm = res['name']
        obs = res['interval_observed']
        elig = res['historical_eligible']

        vol = pd.to_numeric(getattr(r, 'volume', None), errors='coerce')
        amt = pd.to_numeric(getattr(r, 'amount', None), errors='coerce')

        o = cents(getattr(r, 'open', None))
        h = cents(getattr(r, 'high', None))
        l = cents(getattr(r, 'low', None))
        cl = cents(getattr(r, 'close', None))
        pc = cents(getattr(r, 'pre_close', None))

        unknown_reason = None
        bar_valid = True

        if obs != 1:
            bar_valid = False
            unknown_reason = 'INTERVAL_NOT_VERIFIED'
        elif elig is not True:
            bar_valid = False
            unknown_reason = 'INELIGIBLE_REGIME'
        elif vol is None or not np.isfinite(vol) or not (vol > 0):
            bar_valid = False
            unknown_reason = 'ZERO_OR_MISSING_VOLUME'
        elif amt is None or not np.isfinite(amt) or not (amt > 0):
            bar_valid = False
            unknown_reason = 'ZERO_OR_MISSING_AMOUNT'
        elif any(v is None for v in (o, h, l, cl, pc)):
            bar_valid = False
            unknown_reason = 'INVALID_PRICE_CENTS'
        elif not (l <= o <= h and l <= cl <= h and l <= h):
            bar_valid = False
            unknown_reason = 'OHLC_INCONSISTENT'

        upper = limit_cents(pc, '1.10') if pc else None
        lower = limit_cents(pc, '0.90') if pc else None

        if bar_valid:
            if upper is not None and h > upper:
                bar_valid = False
                unknown_reason = 'PRICE_EXCEEDS_UPPER_LIMIT'
            elif lower is not None and l < lower:
                bar_valid = False
                unknown_reason = 'PRICE_BELOW_LOWER_LIMIT'

        is_limit_up = None
        if bar_valid:
            in_pool = (c in closing_limit_ups.get(day, {}))
            at_upper = (cl == upper)
            if at_upper and in_pool:
                is_limit_up = True
            elif (not at_upper) and (not in_pool):
                is_limit_up = False
            else:
                # Contradiction between price and pool assignment
                is_limit_up = None
                bar_valid = False
                unknown_reason = 'PRICE_POOL_CONTRADICTION'

        obs_dict = {
            'date': day,
            'code': c,
            'name': nm,
            'interval_observed': obs,
            'historical_eligible': elig,
            'bar_valid': int(bar_valid),
            'volume': float(vol) if (vol is not None and np.isfinite(vol)) else None,
            'amount': float(amt) if (amt is not None and np.isfinite(amt)) else None,
            'close_cents': cl,
            'open_cents': o,
            'high_cents': h,
            'low_cents': l,
            'pre_close_cents': pc,
            'upper_limit_cents': upper,
            'lower_limit_cents': lower,
            'is_limit_up': int(is_limit_up) if is_limit_up is not None else None,
            'unknown_reason': unknown_reason
        }
        bar_obs[(day, c)] = obs_dict
        if cutoff_date is None:
            normalized_obs_rows.append(obs_dict)

    # 3. Name token cache with eligibility check
    # key: (code, day) -> (name, interval_observed, historical_eligible, filtered_tokens)
    name_token_cache = {}
    def resolve_tokens_and_elig(c, day):
        key = (c, day)
        if key not in name_token_cache:
            res = dn.resolve(c, day, '')
            nm = res['name']
            obs = res['interval_observed']
            elig = res['historical_eligible']
            if obs == 1 and elig is True:
                all_toks = tokens(nm)
                filtered = {t for t in all_toks if t.startswith('H2:') or t.startswith('P:')}
            else:
                filtered = set()
            name_token_cache[key] = (nm, obs, elig, filtered)
        return name_token_cache[key]

    # 4. Strict 2-to-3 Anchor Emergence Events E
    b2_pairs = set()
    for day, c_dict in closing_limit_ups.items():
        for c, h in c_dict.items():
            if h == 2:
                b2_pairs.add((day, c))

    anchor_events = []
    seen_anchors = set()
    unresolved_anchor_events = []

    for day, c_dict in sorted(closing_limit_ups.items()):
        if day not in s_idx:
            continue
        idx = s_idx[day]
        if idx == 0:
            continue
        prev_day = sessions[idx - 1]

        for c, h in sorted(c_dict.items()):
            if h == 3:
                if (prev_day, c) not in b2_pairs:
                    continue
                if (day, c) in seen_anchors:
                    continue
                seen_anchors.add((day, c))

                a_name, a_obs, a_elig, a_toks = resolve_tokens_and_elig(c, day)
                # A missing name or price cannot establish that no competing
                # external anchor exists. Keep this source uncertainty separate.
                a_daily = bar_obs.get((day, c))
                a_previous = bar_obs.get((prev_day, c))
                if a_elig is False:
                    continue
                if not (a_obs == 1 and a_elig is True) or not (
                    a_daily and a_daily['bar_valid'] == 1 and a_daily['is_limit_up'] == 1 and
                    a_previous and a_previous['bar_valid'] == 1 and a_previous['is_limit_up'] == 1):
                    unresolved_anchor_events.append({'E': day, 'e_idx': idx, 'anchor': c})
                    continue
                if not (a_obs == 1 and a_elig is True) or not a_toks:
                    continue

                # Pool of active attempts in [E-4, E]
                window_days = [sessions[k] for k in range(max(0, idx - 4), idx + 1)]
                pool_stocks = set()
                for wd in window_days:
                    pool_stocks.update(active_by_day.get(wd, set()))
                pool_stocks.discard(c)

                # Segment pool into known members, known ineligibles, and unresolved
                for t in sorted(a_toks):
                    is_p = 1 if t.startswith('P:') else 0
                    known_members = set()
                    unresolved_members = set()

                    for m in pool_stocks:
                        m_name, m_obs, m_elig, m_toks = resolve_tokens_and_elig(m, day)
                        if m_elig is False:
                            continue # Known ineligible (e.g. ST or ChiNext)
                        elif m_obs == 1 and m_elig is True:
                            if t in m_toks:
                                known_members.add(m)
                        else:
                            # Unknown name or unverified interval
                            unresolved_members.add(m)

                    anchor_events.append({
                        'E': day,
                        'e_idx': idx,
                        'anchor': c,
                        'anchor_name': a_name,
                        'token': t,
                        'is_p': is_p,
                        'known_cohort': known_members,
                        'unresolved_cohort': unresolved_members
                    })

    # 5. Process all 9,245 candidate rows on day S
    feature_rows = []
    ledger_rows = []

    for r in features_clean.itertuples():
        day = r.date
        code = r.code
        s_pos = s_idx[day]

        # Candidate name resolved as of S-1
        prev_session = sessions[s_pos - 1] if s_pos > 0 else day
        cand_name, cand_obs, cand_elig, cand_tokens = resolve_tokens_and_elig(code, prev_session)

        # Check candidate qualification on S-1
        cand_qualified = (cand_obs == 1 and cand_elig is True)

        # Search eligible anchors in [S-10, S-1]
        eligible = []
        if cand_qualified:
            for ev in anchor_events:
                a_code = ev['anchor']
                if a_code == code:
                    continue
                lag = s_pos - ev['e_idx']
                if not (1 <= lag <= 10):
                    continue
                t = ev['token']
                if (t in cand_tokens) or (code in ev['known_cohort']):
                    sort_key = (ev['is_p'], lag, a_code, t)
                    eligible.append((sort_key, ev, lag))

        if not cand_qualified:
            # Candidate qualification is unknown or unverified
            # Matched cannot be confirmed false -> matched is NaN
            best_ev = None
            cand_unverified = True
        elif not eligible:
            best_ev = None
            cand_unverified = False
        else:
            eligible.sort(key=lambda x: x[0])
            best_ev, best_lag = eligible[0][1], eligible[0][2]
            cand_unverified = False

        if cand_unverified:
            # Candidate S-1 name unverified
            feature_rows.append({
                'date': day,
                'code': code,
                'event_cohort_matched': np.nan,
                'event_cohort_size': 0,
                'event_cohort_primary_member': 0,
                'event_cohort_damage_ratio': np.nan,
                'event_cohort_repair_ratio': np.nan,
                'event_anchor_status': -1,
                'event_cohort_amount_share_delta': np.nan
            })
            ledger_rows.append({
                'date': day,
                'code': code,
                'cand_name': cand_name,
                'cand_interval_observed': cand_obs,
                'cand_historical_eligible': cand_elig,
                'boards': int(r.boards),
                'matched': None,
                'anchor': None,
                'anchor_name': None,
                'E': None,
                'e_lag': None,
                'token': None,
                'N_known': 0,
                'N_unresolved': 0,
                'is_primary_member': 0,
                'damage_ratio': None,
                'damage_k': 0,
                'damage_u': 0,
                'damage_lower': None,
                'damage_upper': None,
                'repair_ratio': None,
                'repair_k': 0,
                'repair_u': 0,
                'repair_lower': None,
                'repair_upper': None,
                'anchor_status_today': -1,
                'amount_share_delta': None,
                'members_known': [],
                'members_unresolved': []
            })
        elif best_ev is None:
            # Verified candidate, but no matching cohort found
            feature_rows.append({
                'date': day,
                'code': code,
                'event_cohort_matched': 0,
                'event_cohort_size': 0,
                'event_cohort_primary_member': 0,
                'event_cohort_damage_ratio': np.nan,
                'event_cohort_repair_ratio': np.nan,
                'event_anchor_status': -1,
                'event_cohort_amount_share_delta': np.nan
            })
            ledger_rows.append({
                'date': day,
                'code': code,
                'cand_name': cand_name,
                'cand_interval_observed': cand_obs,
                'cand_historical_eligible': cand_elig,
                'boards': int(r.boards),
                'matched': 0,
                'anchor': None,
                'anchor_name': None,
                'E': None,
                'e_lag': None,
                'token': None,
                'N_known': 0,
                'N_unresolved': 0,
                'is_primary_member': 0,
                'damage_ratio': None,
                'damage_k': 0,
                'damage_u': 0,
                'damage_lower': None,
                'damage_upper': None,
                'repair_ratio': None,
                'repair_k': 0,
                'repair_u': 0,
                'repair_lower': None,
                'repair_upper': None,
                'anchor_status_today': -1,
                'amount_share_delta': None,
                'members_known': [],
                'members_unresolved': []
            })
        else:
            a_code = best_ev['anchor']
            a_name = best_ev['anchor_name']
            token = best_ev['token']
            E_day = best_ev['E']

            known_roster = sorted([m for m in best_ev['known_cohort'] if m != code and m != a_code])
            unresolved_roster = sorted([m for m in best_ev['unresolved_cohort'] if m != code and m != a_code])

            is_prim = 1 if (code in best_ev['known_cohort']) else 0
            N_known = len(known_roster)
            N_unresolved = len(unresolved_roster)

            if N_known == 0 and N_unresolved == 0:
                # Structurally empty cohort
                feature_rows.append({
                    'date': day,
                    'code': code,
                    'event_cohort_matched': 0,
                    'event_cohort_size': 0,
                    'event_cohort_primary_member': 0,
                    'event_cohort_damage_ratio': np.nan,
                    'event_cohort_repair_ratio': np.nan,
                    'event_anchor_status': -1,
                    'event_cohort_amount_share_delta': np.nan
                })
                ledger_rows.append({
                    'date': day,
                    'code': code,
                    'cand_name': cand_name,
                    'cand_interval_observed': cand_obs,
                    'cand_historical_eligible': cand_elig,
                    'boards': int(r.boards),
                    'matched': 0,
                    'anchor': a_code,
                    'anchor_name': a_name,
                    'E': E_day,
                    'e_lag': best_lag,
                    'token': token,
                    'N_known': 0,
                    'N_unresolved': 0,
                    'is_primary_member': 0,
                    'damage_ratio': None,
                    'damage_k': 0,
                    'damage_u': 0,
                    'damage_lower': None,
                    'damage_upper': None,
                    'repair_ratio': None,
                    'repair_k': 0,
                    'repair_u': 0,
                    'repair_lower': None,
                    'repair_upper': None,
                    'anchor_status_today': -1,
                    'amount_share_delta': None,
                    'members_known': [],
                    'members_unresolved': []
                })
            else:
                # Anchor status on day S
                a_obs_s = bar_obs.get((day, a_code))
                if a_obs_s is None or a_obs_s['bar_valid'] != 1:
                    a_status = -1
                elif a_obs_s['is_limit_up'] == 1:
                    a_status = 2 # Limit up
                else:
                    blown = (day, a_code) in failed_board_attempts
                    is_ld = (a_obs_s['lower_limit_cents'] is not None and a_obs_s['close_cents'] == a_obs_s['lower_limit_cents'])
                    if blown or is_ld:
                        a_status = 0 # Severe
                    else:
                        a_status = 1 # Traded/normal correction

                # Member observation tracking on day S and S-1
                k_damaged = 0
                u_damaged = N_unresolved # Unresolved members add to unknown
                k_repaired = 0
                u_repaired = N_unresolved
                prev_day = sessions[s_pos - 1] if s_pos > 0 else None

                for m in known_roster:
                    m_obs_s = bar_obs.get((day, m))
                    if m_obs_s is None or m_obs_s['bar_valid'] != 1 or m_obs_s['is_limit_up'] is None:
                        u_damaged += 1
                    elif m_obs_s['is_limit_up'] == 0:
                        k_damaged += 1

                    # Repair check
                    if prev_day is None:
                        u_repaired += 1
                    else:
                        m_obs_prev = bar_obs.get((prev_day, m))
                        if (m_obs_s is None or m_obs_s['bar_valid'] != 1 or m_obs_s['is_limit_up'] is None or
                            m_obs_prev is None or m_obs_prev['bar_valid'] != 1 or m_obs_prev['is_limit_up'] is None):
                            u_repaired += 1
                        else:
                            if m_obs_prev['is_limit_up'] == 0 and m_obs_s['is_limit_up'] == 1:
                                k_repaired += 1

                N_total_possible = N_known + N_unresolved
                if N_total_possible > 0:
                    dmg_lower = round(k_damaged / N_total_possible, 6)
                    dmg_upper = round((k_damaged + u_damaged) / N_total_possible, 6)
                    rep_lower = round(k_repaired / N_total_possible, 6)
                    rep_upper = round((k_repaired + u_repaired) / N_total_possible, 6)
                else:
                    dmg_lower = dmg_upper = rep_lower = rep_upper = None

                # Damage & repair ratio: strictly NaN if any unknown exists
                if u_damaged > 0 or N_known == 0:
                    damage_ratio = np.nan
                else:
                    damage_ratio = round(k_damaged / N_known, 6)

                if u_repaired > 0 or N_known == 0:
                    repair_ratio = np.nan
                else:
                    repair_ratio = round(k_repaired / N_known, 6)

                # Amount share delta on (known_roster U {candidate})
                # Requires N_unresolved == 0 and all positive valid amounts
                amt_delta = np.nan
                if N_unresolved == 0 and N_known > 0 and prev_day:
                    c_obs_s = bar_obs.get((day, code))
                    c_obs_prev = bar_obs.get((prev_day, code))
                    if (c_obs_s and c_obs_s['bar_valid'] == 1 and c_obs_s['amount'] and c_obs_s['amount'] > 0 and
                        c_obs_prev and c_obs_prev['bar_valid'] == 1 and c_obs_prev['amount'] and c_obs_prev['amount'] > 0):
                        
                        tot_s = c_obs_s['amount']
                        tot_prev = c_obs_prev['amount']
                        amt_valid = True

                        for m in known_roster:
                            m_s = bar_obs.get((day, m))
                            m_p = bar_obs.get((prev_day, m))
                            if (m_s and m_s['bar_valid'] == 1 and m_s['amount'] and m_s['amount'] > 0 and
                                m_p and m_p['bar_valid'] == 1 and m_p['amount'] and m_p['amount'] > 0):
                                tot_s += m_s['amount']
                                tot_prev += m_p['amount']
                            else:
                                amt_valid = False
                                break

                        if amt_valid and tot_s > 0 and tot_prev > 0:
                            share_s = c_obs_s['amount'] / tot_s
                            share_p = c_obs_prev['amount'] / tot_prev
                            amt_delta = round(share_s - share_p, 6)

                feature_rows.append({
                    'date': day,
                    'code': code,
                    'event_cohort_matched': 1,
                    'event_cohort_size': N_known,
                    'event_cohort_primary_member': is_prim,
                    'event_cohort_damage_ratio': damage_ratio,
                    'event_cohort_repair_ratio': repair_ratio,
                    'event_anchor_status': a_status,
                    'event_cohort_amount_share_delta': amt_delta
                })
                ledger_rows.append({
                    'date': day,
                    'code': code,
                    'cand_name': cand_name,
                    'cand_interval_observed': cand_obs,
                    'cand_historical_eligible': cand_elig,
                    'boards': int(r.boards),
                    'matched': 1,
                    'anchor': a_code,
                    'anchor_name': a_name,
                    'E': E_day,
                    'e_lag': best_lag,
                    'token': token,
                    'N_known': N_known,
                    'N_unresolved': N_unresolved,
                    'is_primary_member': is_prim,
                    'damage_ratio': float(damage_ratio) if np.isfinite(damage_ratio) else None,
                    'damage_k': k_damaged,
                    'damage_u': u_damaged,
                    'damage_lower': dmg_lower,
                    'damage_upper': dmg_upper,
                    'repair_ratio': float(repair_ratio) if np.isfinite(repair_ratio) else None,
                    'repair_k': k_repaired,
                    'repair_u': u_repaired,
                    'repair_lower': rep_lower,
                    'repair_upper': rep_upper,
                    'anchor_status_today': a_status,
                    'amount_share_delta': float(amt_delta) if np.isfinite(amt_delta) else None,
                    'members_known': known_roster,
                    'members_unresolved': unresolved_roster
                })

    # Complete the already requested source-uncertainty handling. This changes
    # no token, economic hypothesis, lookback, feature or performance gate.
    group_fields = ['event_cohort_matched', 'event_cohort_size', 'event_cohort_primary_member',
                    'event_cohort_damage_ratio', 'event_cohort_repair_ratio',
                    'event_anchor_status', 'event_cohort_amount_share_delta']
    for f, ledger in zip(feature_rows, ledger_rows):
        uncertain = [a for a in unresolved_anchor_events
                     if a['anchor'] != f['code'] and 1 <= s_idx[f['date']] - a['e_idx'] <= 10]
        ledger['unresolved_anchor_events'] = uncertain
        ledger['denominator_lower_N'] = ledger['N_known']
        ledger['denominator_upper_N'] = ledger['N_known'] + ledger['N_unresolved']
        if ledger['N_unresolved']:
            # A lower known member count is not the exact frozen group size.
            f['event_cohort_size'] = np.nan
            if not ledger['N_known']:
                f['event_cohort_matched'] = np.nan
        if ledger['matched'] is None or uncertain:
            for field in group_fields:
                f[field] = np.nan
            ledger['source_qualification'] = 'UNKNOWN_CANDIDATE_OR_EXTERNAL_ANCHOR'
        else:
            ledger['source_qualification'] = 'KNOWN_ANCHOR_UNRESOLVED_MEMBERS' if ledger['N_unresolved'] else 'QUALIFIED_ARCHIVED_UNIVERSE'
    df_feat = pd.DataFrame(feature_rows)
    return df_feat, ledger_rows, normalized_obs_rows

def main():
    start_time = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Final Qualified Event-Anchored Cohort Engine")
    print("=" * 60)
    print(f"Loading dataset: {DATASET_PATH}")
    print(f"Loading master: {MASTER_PATH}")
    print(f"Dataset SHA: {sha256_file(DATASET_PATH)}")
    print(f"Master SHA: {sha256_file(MASTER_PATH)}")

    d = joblib.load(DATASET_PATH)
    sessions = d['sessions']
    attempts = d['attempts']
    bars = d['bars']
    features = d['features']
    relay = d['prepared']['relay']
    from security_master_v4 import validate_master
    master = validate_master(pd.read_csv(MASTER_PATH, dtype=str, encoding='utf-8-sig')).fillna('')

    # 1. Build full features, ledger, and normalized observations
    print("\n[Step 1] Building full 9,245 rows with strict qualification...")
    df_feat, ledger_rows, normalized_obs = build_corrected_features_and_ledger(sessions, attempts, bars, features, master)
    assert len(df_feat) == 9245
    assert not df_feat.duplicated(['date', 'code']).any()

    merged = pd.merge(df_feat, relay[['date', 'code', 'target', 'boards']], on=['date', 'code'], how='inner')
    assert len(merged) == 9245

    feat_csv_path = os.path.join(OUT_DIR, "event_group_features_9245.csv")
    merged.to_csv(feat_csv_path, index=False)
    feat_sha = sha256_file(feat_csv_path)
    print(f"Saved: {feat_csv_path} (SHA: {feat_sha})")

    # Save normalized member observations
    obs_csv_path = os.path.join(OUT_DIR, "MEMBER_DAILY_OBSERVATIONS.csv")
    pd.DataFrame(normalized_obs).to_csv(obs_csv_path, index=False)
    obs_sha = sha256_file(obs_csv_path)
    print(f"Saved: {obs_csv_path} (SHA: {obs_sha})")

    # Save complete ledger
    ledger_csv_path = os.path.join(OUT_DIR, "EVENT_GROUP_LEDGER.csv")
    ledger_flat = []
    for r in ledger_rows:
        ledger_flat.append({
            'date': r['date'],
            'code': r['code'],
            'cand_name': r['cand_name'],
            'cand_interval_observed': r['cand_interval_observed'],
            'cand_historical_eligible': r['cand_historical_eligible'],
            'boards': r['boards'],
            'matched': r['matched'],
            'anchor': r['anchor'],
            'anchor_name': r['anchor_name'],
            'E': r['E'],
            'e_lag': r['e_lag'],
            'token': r['token'],
            'N_known': r['N_known'],
            'N_unresolved': r['N_unresolved'],
            'is_primary_member': r['is_primary_member'],
            'damage_ratio': r['damage_ratio'],
            'damage_k': r['damage_k'],
            'damage_u': r['damage_u'],
            'damage_lower': r['damage_lower'],
            'damage_upper': r['damage_upper'],
            'repair_ratio': r['repair_ratio'],
            'repair_k': r['repair_k'],
            'repair_u': r['repair_u'],
            'repair_lower': r['repair_lower'],
            'repair_upper': r['repair_upper'],
            'anchor_status_today': r['anchor_status_today'],
            'amount_share_delta': r['amount_share_delta'],
            'known_count': len(r['members_known']),
            'unresolved_count': len(r['members_unresolved'])
        })
    pd.DataFrame(ledger_flat).to_csv(ledger_csv_path, index=False)
    ledger_sha = sha256_file(ledger_csv_path)
    print(f"Saved: {ledger_csv_path} (SHA: {ledger_sha})")

    # Preserve the actual known/unresolved member codes and source reasons;
    # counts alone cannot certify a fixed-member historical ledger.
    ledger_jsonl_path = os.path.join(OUT_DIR, 'EVENT_GROUP_LEDGER_DETAILS.jsonl')
    with open(ledger_jsonl_path, 'x', encoding='utf-8') as handle:
        for row in ledger_rows:
            handle.write(json.dumps(clean_json(row), ensure_ascii=False) + '\n')

    # 2. Strict Causal Prefix Invariance Check
    print("\n[Step 2] Verifying causal prefix invariance with raw sliced subsets...")
    prefix_results = {}
    for cutoff in ['2025-06-30', '2025-12-31']:
        df_cut, _, _ = build_corrected_features_and_ledger(sessions, attempts, bars, features, master, cutoff_date=cutoff)
        sub_full = df_feat[df_feat['date'] <= cutoff].copy().reset_index(drop=True)
        assert len(df_cut) == len(sub_full)
        mismatches = 0
        for col in df_cut.columns:
            s1 = df_cut[col]
            s2 = sub_full[col]
            diff = ~(s1.eq(s2) | (s1.isna() & s2.isna()))
            mismatches += int(diff.sum())
        assert mismatches == 0, f"Mismatches at {cutoff}: {mismatches}"
        prefix_results[cutoff] = {
            "rows": len(df_cut),
            "mismatches": 0,
            "status": "STRICTLY_VERIFIED_INVARIANT"
        }
        print(f" - {cutoff}: {len(df_cut)} rows, 0 mismatches. Strictly verified.")

    prefix_path = os.path.join(OUT_DIR, "PREFIX_INVARIANCE_CHECK.json")
    with open(prefix_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(prefix_results), f, ensure_ascii=False, indent=2)

    # 3. Exact Coverage & Distribution Metrics
    print("\n[Step 3] Computing exact coverage metrics...")
    total_candidates = len(merged)
    matched_candidates = int((merged['event_cohort_matched'] == 1).sum())
    unmatched_candidates = int((merged['event_cohort_matched'] == 0).sum())
    cand_unverified_count = int(merged['event_cohort_matched'].isna().sum())
    primary_members = int(merged['event_cohort_primary_member'].sum())

    known_target_sub = merged[merged.target.notna()]
    total_known_targets = len(known_target_sub)
    known_positives = int((known_target_sub.target == 1).sum())
    known_negatives = int((known_target_sub.target == 0).sum())

    matched_sub = known_target_sub[known_target_sub['event_cohort_matched'] == 1]
    unmatched_sub = known_target_sub[known_target_sub['event_cohort_matched'] == 0]

    matched_pos = int((matched_sub.target == 1).sum())
    matched_known_total = len(matched_sub)
    unmatched_pos = int((unmatched_sub.target == 1).sum())
    unmatched_known_total = len(unmatched_sub)

    coverage = {
        "total_candidates": total_candidates,
        "matched_candidates": matched_candidates,
        "matched_percentage": round(matched_candidates / total_candidates, 6),
        "unmatched_candidates": unmatched_candidates,
        "candidate_unverified_qualification": cand_unverified_count,
        "primary_member_candidates": primary_members,
        "diffusion_candidates": matched_candidates - primary_members,
        "total_known_targets": total_known_targets,
        "total_unknown_targets": total_candidates - total_known_targets,
        "total_known_positives": known_positives,
        "total_known_negatives": known_negatives,
        "matched_known_total": matched_known_total,
        "matched_known_positives": matched_pos,
        "matched_known_hit_rate": round(matched_pos / matched_known_total, 6) if matched_known_total else None,
        "unmatched_known_total": unmatched_known_total,
        "unmatched_known_positives": unmatched_pos,
        "unmatched_known_hit_rate": round(unmatched_pos / unmatched_known_total, 6) if unmatched_known_total else None,
        "observed_counts": {
            "damage_ratio_notna": int(merged['event_cohort_damage_ratio'].notna().sum()),
            "repair_ratio_notna": int(merged['event_cohort_repair_ratio'].notna().sum()),
            "amount_share_delta_notna": int(merged['event_cohort_amount_share_delta'].notna().sum()),
            "anchor_status_valid": int((merged['event_anchor_status'] >= 0).sum())
        }
    }

    coverage_path = os.path.join(OUT_DIR, "EVENT_GROUP_COVERAGE.json")
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(coverage), f, ensure_ascii=False, indent=2)

    receipt = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FINAL_SOURCE_FIX_COMPLETED_ZERO_FITS",
        "fits_executed": 0,
        "http_requests_executed": 0,
        "files": {
            "features_csv": feat_csv_path,
            "features_sha256": feat_sha,
            "ledger_csv": ledger_csv_path,
            "ledger_sha256": ledger_sha,
            "observations_csv": obs_csv_path,
            "observations_sha256": obs_sha,
            "prefix_invariance": prefix_path,
            "coverage": coverage_path
        },
        "coverage_summary": coverage
    }
    receipt_path = os.path.join(OUT_DIR, "CORRECTED_RECEIPT.json")
    with open(receipt_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(receipt), f, ensure_ascii=False, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print(f"\nExecution finished in {elapsed}s.")
    print(f"Receipt saved: {receipt_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
