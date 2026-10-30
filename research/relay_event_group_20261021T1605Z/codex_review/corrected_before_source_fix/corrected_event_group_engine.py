"""
Corrected Event-Anchored Cohort & Relative Status Engine (Finite Zero-Fit)
==========================================================================
Implements the finite corrected event-anchored narrative cohort mechanism:
1. DatedNames:
   - Candidate names resolved as of S-1.
   - Cohort members and anchor names resolved as of E.
2. Strict Anchor Emergence Event E:
   - Official trading day E.
   - Anchor stock had board_height == 2 on sessions[e_idx - 1] (closing limit up).
   - Anchor stock had board_height == 3 on E with pool_assignment == 'CLOSING_LIMIT_UP'.
3. Tokens & Stopwords:
   - Only H2 (2-character tokens) and P (phonetic token families like 阳洋羊扬, 华花, 龙隆, etc.).
   - Explicitly excludes H1 (single-character tokens) to eliminate 144-stock market breadth explosions.
4. Search Window & Anchor Selection:
   - E in [S-10, S-1] (1 <= lag <= 10 official trading days).
   - Candidate self-anchoring strictly excluded (anchor_code != candidate_code).
   - Selection priority: (token_is_P, e_lag, anchor_code, token).
5. Fixed Cohort G_E & Tracking:
   - Cohort formed from active attempts in [E-4, E] sharing the token at date E.
   - Evaluation roster M = G_E \ {anchor, candidate}. Cohort size N = |M|.
   - If N == 0: matched=0, size=0, primary=0, anchor_status=-1, ratios=NaN.
   - If N > 0: matched=1, size=N, primary = (candidate in G_E).
   - damage_ratio: k_damaged / N where damaged means did not close limit-up on day S.
     If any member in M is unobserved in bars on day S, damage_ratio is NaN (records [k/N, (k+u)/N]).
   - repair_ratio: k_repaired / N where repaired means observed on S-1 (not limit-up) and closed limit-up on S.
     If any member in M is unobserved on S or S-1, repair_ratio is NaN.
   - amount_share_delta: candidate share in (M U {candidate}) on day S minus day S-1.
     If any amount is missing or <= 0 on S or S-1, amount_share_delta is NaN.
   - anchor_status: 2 (limit-up), 0 (blown/limit-down), 1 (broken/traded), -1 (unobserved/unmatched).
6. Complete Roster Ledger for all 9,245 rows.
7. Prefix Invariance Check with strictly sliced input subsets.
8. Strictly 0 model fits, 0 network requests.
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

DATASET_PATH = os.path.join(ROOT, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\historical_final\dataset.joblib")
MASTER_PATH = os.path.join(ROOT, r"data\v4\security_master_intervals.csv")
OUT_DIR = os.path.join(ROOT, r"research\relay_event_group_20261021T1605Z\corrected")

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

def _cent(val):
    if val is None or pd.isna(val):
        return None
    try:
        d = Decimal(str(val)) * 100
        return int(d)
    except Exception:
        return None

def _limit_down_cents(pre_close_cents, code, name):
    if pre_close_cents is None:
        return None
    code_str = str(code).zfill(6)
    name_str = str(name)
    if "ST" in name_str or "*ST" in name_str:
        mult = Decimal("0.95")
    elif code_str.startswith("30") or code_str.startswith("688"):
        mult = Decimal("0.80")
    else:
        mult = Decimal("0.90")
    return int((Decimal(pre_close_cents) * mult).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def build_corrected_features_and_ledger(sessions, attempts, bars, features, master_df, cutoff_date=None):
    if cutoff_date is not None:
        sessions = [s for s in sessions if s <= cutoff_date]
        features = features[features['date'] <= cutoff_date].copy()
        attempts = attempts[attempts['signal_date'] <= cutoff_date].copy()
        bars = bars[bars['date'] <= cutoff_date].copy()
        master_df = master_df[master_df['valid_from'] <= cutoff_date].copy()

    s_idx = {s: i for i, s in enumerate(sessions)}
    dn = DatedNames(master_df)

    # 1. Pre-index bars by (date, code)
    # Store: close_cents, pre_close_cents, amount, name
    bar_info = {}
    for r in bars.itertuples():
        c = str(r.code).zfill(6)
        cl_cents = _cent(r.close)
        pr_cents = _cent(r.pre_close)
        amt = float(r.amount) if (r.amount is not None and pd.notnull(r.amount) and np.isfinite(r.amount) and r.amount > 0) else None
        bar_info[(r.date, c)] = {
            'close_cents': cl_cents,
            'pre_close_cents': pr_cents,
            'amount': amt,
            'name': getattr(r, 'name', '')
        }

    # 2. Pre-index closing limit-ups and failed board attempts
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

    # Name token cache to avoid redundant tokenization: (code, date) -> (name, tokens_set)
    name_token_cache = {}
    def get_stock_name_and_tokens(c, day):
        key = (c, day)
        if key not in name_token_cache:
            res = dn.resolve(c, day, '')
            nm = res['name']
            all_toks = tokens(nm)
            # Only H2 and P tokens, strictly exclude H1
            filtered = {t for t in all_toks if t.startswith('H2:') or t.startswith('P:')}
            name_token_cache[key] = (nm, filtered)
        return name_token_cache[key]

    # 3. Identify all strict 2-to-3 anchor events E
    # Condition: on sessions[e_idx - 1], board_height == 2 in closing_limit_ups.
    # On E, board_height == 3 in closing_limit_ups.
    b2_pairs = set()
    for day, c_dict in closing_limit_ups.items():
        for c, h in c_dict.items():
            if h == 2:
                b2_pairs.add((day, c))

    anchor_events = [] # list of dicts: {'E': day, 'e_idx': idx, 'anchor': c, 'token': t, 'is_P': int, 'cohort': set_of_member_codes}
    seen_anchors = set()

    for day, c_dict in sorted(closing_limit_ups.items()):
        if day not in s_idx:
            continue
        idx = s_idx[day]
        if idx == 0:
            continue
        prev_day = sessions[idx - 1]

        for c, h in sorted(c_dict.items()):
            if h == 3:
                # Check strict adjacent 2-to-3
                if (prev_day, c) not in b2_pairs:
                    continue
                if (day, c) in seen_anchors:
                    continue
                seen_anchors.add((day, c))

                a_name, a_toks = get_stock_name_and_tokens(c, day)
                if not a_toks:
                    continue

                # Pool of active attempts in [E-4, E]
                window_days = [sessions[k] for k in range(max(0, idx - 4), idx + 1)]
                pool_stocks = set()
                for wd in window_days:
                    pool_stocks.update(active_by_day.get(wd, set()))
                pool_stocks.discard(c)

                # For each token of the anchor, build cohort of co-active stocks
                for t in sorted(a_toks):
                    is_p = 1 if t.startswith('P:') else 0
                    cohort_members = set()
                    for m in pool_stocks:
                        _, m_toks = get_stock_name_and_tokens(m, day)
                        if t in m_toks:
                            cohort_members.add(m)
                    
                    anchor_events.append({
                        'E': day,
                        'e_idx': idx,
                        'anchor': c,
                        'anchor_name': a_name,
                        'token': t,
                        'is_p': is_p,
                        'cohort': cohort_members # members excluding anchor
                    })

    # Group anchor events by (E, token) or index by lag for fast lookup
    # 4. Process all 9,245 candidate rows on day S
    feature_rows = []
    ledger_rows = []

    for r in features_clean.itertuples():
        day = r.date
        code = r.code
        s_pos = s_idx[day]

        # Candidate name resolved as of S-1
        prev_session = sessions[s_pos - 1] if s_pos > 0 else day
        cand_name, cand_tokens = get_stock_name_and_tokens(code, prev_session)

        # Find eligible anchor events:
        # Conditions:
        # 1. anchor != code (no self-anchoring)
        # 2. 1 <= lag <= 10 where lag = s_pos - e_idx (E in [S-10, S-1])
        # 3. token in cand_tokens OR code in ev['cohort']
        eligible = []
        for ev in anchor_events:
            a_code = ev['anchor']
            if a_code == code:
                continue
            lag = s_pos - ev['e_idx']
            if not (1 <= lag <= 10):
                continue
            t = ev['token']
            if (t in cand_tokens) or (code in ev['cohort']):
                # Priority: (is_p, lag, anchor_code, token)
                sort_key = (ev['is_p'], lag, a_code, t)
                eligible.append((sort_key, ev, lag))

        if not eligible:
            # No eligible event group
            best_ev = None
        else:
            eligible.sort(key=lambda x: x[0])
            best_ev, best_lag = eligible[0][1], eligible[0][2]

        if best_ev is None:
            # Unmatched
            feat_dict = {
                'date': day,
                'code': code,
                'event_cohort_matched': 0,
                'event_cohort_size': 0,
                'event_cohort_primary_member': 0,
                'event_cohort_damage_ratio': np.nan,
                'event_cohort_repair_ratio': np.nan,
                'event_anchor_status': -1,
                'event_cohort_amount_share_delta': np.nan
            }
            ledger_dict = {
                'date': day,
                'code': code,
                'cand_name': cand_name,
                'boards': int(r.boards),
                'matched': 0,
                'anchor': None,
                'anchor_name': None,
                'E': None,
                'e_lag': None,
                'token': None,
                'cohort_size_N': 0,
                'is_primary_member': 0,
                'damage_ratio': None,
                'damage_k': 0,
                'damage_u': 0,
                'repair_ratio': None,
                'repair_k': 0,
                'repair_u': 0,
                'anchor_status_today': -1,
                'amount_share_delta': None,
                'roster_members': []
            }
        else:
            a_code = best_ev['anchor']
            a_name = best_ev['anchor_name']
            token = best_ev['token']
            E_day = best_ev['E']

            # Fixed members M = cohort \ {anchor, candidate}
            raw_cohort = best_ev['cohort']
            is_prim = 1 if (code in raw_cohort) else 0
            roster = sorted([m for m in raw_cohort if m != code and m != a_code])
            N = len(roster)

            if N == 0:
                # Structure empty cohort (no other peers)
                feat_dict = {
                    'date': day,
                    'code': code,
                    'event_cohort_matched': 0,
                    'event_cohort_size': 0,
                    'event_cohort_primary_member': 0,
                    'event_cohort_damage_ratio': np.nan,
                    'event_cohort_repair_ratio': np.nan,
                    'event_anchor_status': -1,
                    'event_cohort_amount_share_delta': np.nan
                }
                ledger_dict = {
                    'date': day,
                    'code': code,
                    'cand_name': cand_name,
                    'boards': int(r.boards),
                    'matched': 0,
                    'anchor': a_code,
                    'anchor_name': a_name,
                    'E': E_day,
                    'e_lag': best_lag,
                    'token': token,
                    'cohort_size_N': 0,
                    'is_primary_member': 0,
                    'damage_ratio': None,
                    'damage_k': 0,
                    'damage_u': 0,
                    'repair_ratio': None,
                    'repair_k': 0,
                    'repair_u': 0,
                    'anchor_status_today': -1,
                    'amount_share_delta': None,
                    'roster_members': []
                }
            else:
                # Anchor status on day S
                a_bar = bar_info.get((day, a_code))
                if a_bar is None or a_bar['close_cents'] is None:
                    a_status = -1
                elif a_code in closing_limit_ups.get(day, {}):
                    a_status = 2 # Limit-up
                else:
                    blown = (day, a_code) in failed_board_attempts
                    is_ld = False
                    if a_bar['pre_close_cents'] is not None:
                        flr = _limit_down_cents(a_bar['pre_close_cents'], a_code, a_bar['name'])
                        if flr is not None and a_bar['close_cents'] <= flr:
                            is_ld = True
                    if blown or is_ld:
                        a_status = 0 # Severe
                    else:
                        a_status = 1 # Broken/traded

                # Member daily tracking on day S and S-1
                k_damaged = 0
                u_damaged = 0
                k_repaired = 0
                u_repaired = 0
                prev_day = sessions[s_pos - 1] if s_pos > 0 else None

                roster_details = []
                for m in roster:
                    m_bar_s = bar_info.get((day, m))
                    m_limit_s = (m in closing_limit_ups.get(day, {}))
                    m_s_obs = (m_bar_s is not None and m_bar_s['close_cents'] is not None)

                    if not m_s_obs:
                        u_damaged += 1
                        m_dmg_state = "UNKNOWN"
                    elif not m_limit_s:
                        k_damaged += 1
                        m_dmg_state = "DAMAGED"
                    else:
                        m_dmg_state = "LIMIT_UP"

                    # Repair tracking
                    m_rep_state = "NOT_REPAIRED"
                    if prev_day is None:
                        u_repaired += 1
                        m_rep_state = "UNKNOWN"
                    else:
                        m_bar_prev = bar_info.get((prev_day, m))
                        m_prev_obs = (m_bar_prev is not None and m_bar_prev['close_cents'] is not None)
                        if not m_s_obs or not m_prev_obs:
                            u_repaired += 1
                            m_rep_state = "UNKNOWN"
                        else:
                            was_damaged_prev = (m not in closing_limit_ups.get(prev_day, {}))
                            if was_damaged_prev and m_limit_s:
                                k_repaired += 1
                                m_rep_state = "REPAIRED"

                    roster_details.append({
                        'code': m,
                        'damage_state': m_dmg_state,
                        'repair_state': m_rep_state
                    })

                # If any member in M has unknown observation on S, damage_ratio is NaN
                if u_damaged > 0:
                    damage_ratio = np.nan
                else:
                    damage_ratio = round(k_damaged / N, 6)

                # If any member in M has unknown observation on S or S-1, repair_ratio is NaN
                if u_repaired > 0:
                    repair_ratio = np.nan
                else:
                    repair_ratio = round(k_repaired / N, 6)

                # Amount share delta on (M U {candidate})
                # Check amounts on day S
                amt_cand_s = bar_info.get((day, code), {}).get('amount')
                amt_cand_prev = bar_info.get((prev_day, code), {}).get('amount') if prev_day else None

                amt_s_valid = True if (amt_cand_s is not None and amt_cand_s > 0) else False
                amt_prev_valid = True if (amt_cand_prev is not None and amt_cand_prev > 0) else False

                total_amt_s = amt_cand_s if amt_s_valid else 0
                total_amt_prev = amt_cand_prev if amt_prev_valid else 0

                for m in roster:
                    m_amt_s = bar_info.get((day, m), {}).get('amount')
                    if m_amt_s is None or m_amt_s <= 0:
                        amt_s_valid = False
                    else:
                        total_amt_s += m_amt_s

                    if prev_day:
                        m_amt_prev = bar_info.get((prev_day, m), {}).get('amount')
                        if m_amt_prev is None or m_amt_prev <= 0:
                            amt_prev_valid = False
                        else:
                            total_amt_prev += m_amt_prev
                    else:
                        amt_prev_valid = False

                if amt_s_valid and amt_prev_valid and total_amt_s > 0 and total_amt_prev > 0:
                    share_s = amt_cand_s / total_amt_s
                    share_prev = amt_cand_prev / total_amt_prev
                    amount_share_delta = round(share_s - share_prev, 6)
                else:
                    amount_share_delta = np.nan

                feat_dict = {
                    'date': day,
                    'code': code,
                    'event_cohort_matched': 1,
                    'event_cohort_size': N,
                    'event_cohort_primary_member': is_prim,
                    'event_cohort_damage_ratio': damage_ratio,
                    'event_cohort_repair_ratio': repair_ratio,
                    'event_anchor_status': a_status,
                    'event_cohort_amount_share_delta': amount_share_delta
                }
                ledger_dict = {
                    'date': day,
                    'code': code,
                    'cand_name': cand_name,
                    'boards': int(r.boards),
                    'matched': 1,
                    'anchor': a_code,
                    'anchor_name': a_name,
                    'E': E_day,
                    'e_lag': best_lag,
                    'token': token,
                    'cohort_size_N': N,
                    'is_primary_member': is_prim,
                    'damage_ratio': float(damage_ratio) if np.isfinite(damage_ratio) else None,
                    'damage_k': k_damaged,
                    'damage_u': u_damaged,
                    'repair_ratio': float(repair_ratio) if np.isfinite(repair_ratio) else None,
                    'repair_k': k_repaired,
                    'repair_u': u_repaired,
                    'anchor_status_today': a_status,
                    'amount_share_delta': float(amount_share_delta) if np.isfinite(amount_share_delta) else None,
                    'roster_members': roster_details
                }

        feature_rows.append(feat_dict)
        ledger_rows.append(ledger_dict)

    df_feat = pd.DataFrame(feature_rows)
    return df_feat, ledger_rows

def main():
    start_time = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Finite Corrected Event-Anchored Cohort Engine")
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
    master = pd.read_csv(MASTER_PATH, dtype=str, encoding='utf-8-sig')

    # 1. Build full features and ledger
    print("\n[Step 1] Building full 9,245 rows features and complete ledger...")
    df_feat, ledger_rows = build_corrected_features_and_ledger(sessions, attempts, bars, features, master)
    assert len(df_feat) == 9245
    assert not df_feat.duplicated(['date', 'code']).any()

    merged = pd.merge(df_feat, relay[['date', 'code', 'target', 'boards']], on=['date', 'code'], how='inner')
    assert len(merged) == 9245

    feat_csv_path = os.path.join(OUT_DIR, "event_group_features_9245.csv")
    merged.to_csv(feat_csv_path, index=False)
    feat_sha = sha256_file(feat_csv_path)
    print(f"Saved: {feat_csv_path} (SHA: {feat_sha})")

    # Save complete ledger
    ledger_csv_path = os.path.join(OUT_DIR, "EVENT_GROUP_LEDGER.csv")
    # Flat summary for fast CSV querying
    ledger_flat = []
    for r in ledger_rows:
        ledger_flat.append({
            'date': r['date'],
            'code': r['code'],
            'cand_name': r['cand_name'],
            'boards': r['boards'],
            'matched': r['matched'],
            'anchor': r['anchor'],
            'anchor_name': r['anchor_name'],
            'E': r['E'],
            'e_lag': r['e_lag'],
            'token': r['token'],
            'cohort_size_N': r['cohort_size_N'],
            'is_primary_member': r['is_primary_member'],
            'damage_ratio': r['damage_ratio'],
            'damage_k': r['damage_k'],
            'damage_u': r['damage_u'],
            'repair_ratio': r['repair_ratio'],
            'repair_k': r['repair_k'],
            'repair_u': r['repair_u'],
            'anchor_status_today': r['anchor_status_today'],
            'amount_share_delta': r['amount_share_delta'],
            'roster_count': len(r['roster_members'])
        })
    pd.DataFrame(ledger_flat).to_csv(ledger_csv_path, index=False)
    ledger_sha = sha256_file(ledger_csv_path)
    print(f"Saved: {ledger_csv_path} (SHA: {ledger_sha})")

    # Save detailed JSONL ledger with member rosters
    ledger_jsonl_path = os.path.join(OUT_DIR, "EVENT_GROUP_LEDGER_DETAILS.jsonl")
    with open(ledger_jsonl_path, 'w', encoding='utf-8') as f:
        for r in ledger_rows:
            f.write(json.dumps(clean_json(r), ensure_ascii=False) + '\n')
    print(f"Saved: {ledger_jsonl_path}")

    # 2. Strict Causal Prefix Invariance Check
    print("\n[Step 2] Verifying causal prefix invariance with raw sliced subsets...")
    prefix_results = {}
    for cutoff in ['2025-06-30', '2025-12-31']:
        df_cut, _ = build_corrected_features_and_ledger(sessions, attempts, bars, features, master, cutoff_date=cutoff)
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
    matched_candidates = int(merged['event_cohort_matched'].sum())
    primary_members = int(merged['event_cohort_primary_member'].sum())

    # Exact known positive / negative targets
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
        "status": "FINITE_CORRECTION_COMPLETED_ZERO_FITS",
        "fits_executed": 0,
        "http_requests_executed": 0,
        "files": {
            "features_csv": feat_csv_path,
            "features_sha256": feat_sha,
            "ledger_csv": ledger_csv_path,
            "ledger_sha256": ledger_sha,
            "ledger_jsonl": ledger_jsonl_path,
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
