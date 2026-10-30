"""
P1 Technical Supplement V3: Two Middle-Tier Events & Bounded Cohorts
===================================================================
Fulfills the finite supplement required by Codex review:
1. Separation of Two Middle-Tier Feedback Events:
   - Event 1: Non-promotion (不再收盘涨停), k_nonpromoted.
   - Event 2: Severe damage (触板未封 [FAILED_BOARD_OBSERVATION] 或收盘跌停/大跌 [close/pre_close <= 0.915]), k_severe.
2. Complete Cohort Accounting:
   - N: Total middle-tier members (boards 3 and 4) at S-1.
   - n_obs: Observed members with valid trading price on day S.
   - u_unk: Missing/unobserved members on day S.
   - Explicit lower and upper bounds: [k/N, (k+u)/N].
   - Structural void (N == 0) yields None/null, never 0.0%.
3. Full Individual Relative Context:
   - same_height_amount_rank_pct: Candidate's own percentile rank in cohort on day S.
   - same_height_amount_rank_delta: Shift from day S-1 to day S.
4. Causal Prefix Invariance verified at 2025-06-30 and 2025-12-31.
5. Strictly 0 fits, 0 network requests.
"""

import os
import sys
import json
import math
import hashlib
import time
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
import numpy as np
import joblib
from scipy.stats import rankdata

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

def build_p1_features_v3(sessions, attempts, bars, features, cutoff_date=None):
    if cutoff_date is not None:
        sessions = [s for s in sessions if s <= cutoff_date]
        features = features[features['date'] <= cutoff_date].copy()
        attempts = attempts[attempts['signal_date'] <= cutoff_date].copy()
        bars = bars[bars['date'] <= cutoff_date].copy()

    s_idx = {s: i for i, s in enumerate(sessions)}
    
    # 0. Track valid price bars and blown board attempts
    bars_valid = set()
    bars_amt = {}
    bars_ret = {}
    for r in bars.itertuples():
        c = str(r.code).zfill(6)
        if r.close is not None and pd.notnull(r.close) and r.close > 0:
            bars_valid.add((r.date, c))
            bars_amt[(r.date, c)] = r.amount
            if r.pre_close is not None and pd.notnull(r.pre_close) and r.pre_close > 0:
                bars_ret[(r.date, c)] = r.close / r.pre_close

    failed_board_attempts = set()
    for r in attempts[attempts['pool_assignment'] == 'FAILED_BOARD_OBSERVATION'].itertuples():
        failed_board_attempts.add((r.signal_date, str(r.code).zfill(6)))

    # Closing limit-up pool
    closing_pool = defaultdict(dict)
    for r in attempts[attempts['pool_assignment'] == 'CLOSING_LIMIT_UP'].itertuples():
        day = r.signal_date
        if day not in s_idx:
            continue
        code = str(r.code).zfill(6)
        closing_pool[day][code] = {
            'boards': int(r.board_height),
            'amount': bars_amt.get((day, code), None)
        }

    for r in features.itertuples():
        day = r.date
        if day not in s_idx:
            continue
        code = str(r.code).zfill(6)
        if code not in closing_pool[day]:
            closing_pool[day][code] = {
                'boards': int(r.boards),
                'amount': bars_amt.get((day, code), None)
            }

    # 1. Compute Anchor Break Event E(j)
    E = {}
    for i in range(1, len(sessions)):
        day_j = sessions[i]
        day_prev = sessions[i - 1]
        pool_prev = closing_pool.get(day_prev, {})
        pool_j = closing_pool.get(day_j, {})
        anchors_prev = {c: v for c, v in pool_prev.items() if v['boards'] >= 4}
        if not anchors_prev:
            E[day_j] = 'NO_ANCHOR'
            continue
            
        max_h = max(v['boards'] for v in anchors_prev.values())
        top_anchors = {c: v for c, v in anchors_prev.items() if v['boards'] == max_h}
        
        has_break = False
        has_unknown = False
        
        for a in top_anchors:
            if a in pool_j and pool_j[a]['boards'] >= max_h + 1:
                pass # Promoted
            elif (day_j, a) in bars_valid:
                has_break = True # Traded with valid price, but did not advance
            else:
                has_unknown = True # Price / quote missing
                
        if has_break:
            E[day_j] = 1
        elif has_unknown:
            E[day_j] = 'UNKNOWN'
        else:
            E[day_j] = 0

    # 2. Precompute percentile_cache for same_height_amount_rank
    percentile_cache = {}
    for day, pool in closing_pool.items():
        by_board = defaultdict(list)
        for code, info in pool.items():
            by_board[info['boards']].append((code, info['amount']))
            
        day_pcts = {}
        for b, items in by_board.items():
            n = len(items)
            if n == 1:
                day_pcts[(b, items[0][0])] = 0.5
            else:
                has_missing = any(amt is None or pd.isna(amt) or not np.isfinite(amt) for _, amt in items)
                if has_missing:
                    for c, _ in items:
                        day_pcts[(b, c)] = None
                else:
                    codes = [c for c, _ in items]
                    amts = np.array([amt for _, amt in items], dtype=float)
                    ranks = rankdata(amts, method='average')
                    pcts = (ranks - 0.5) / n
                    for c, p in zip(codes, pcts):
                        day_pcts[(b, c)] = float(p)
                        
        percentile_cache[day] = day_pcts

    # 3. Compute row-level features
    rows = []
    for r in features.itertuples():
        day = r.date
        code = str(r.code).zfill(6)
        b = int(r.boards)
        idx = s_idx[day]
        
        # A. Anchor break event and age
        if idx < 4:
            brk_age = 'UNKNOWN'
            brk_today = 'UNKNOWN'
        else:
            brk_today = E.get(day, 'UNKNOWN')
            found_age = None
            for lag in range(5):
                sess = sessions[idx - lag]
                e_val = E.get(sess, 'UNKNOWN')
                if e_val == 1:
                    found_age = str(lag)
                    break
                elif e_val == 'UNKNOWN':
                    found_age = 'UNKNOWN'
                    break
            if found_age is None:
                found_age = 'NO_BREAK_IN_WINDOW'
            brk_age = found_age
            
        # B. Middle tier void at day S
        p_s = closing_pool.get(day, {})
        count_3 = sum(1 for v in p_s.values() if v['boards'] == 3)
        count_4 = sum(1 for v in p_s.values() if v['boards'] == 4)
        mid_void = 1 if (count_3 == 0 and count_4 == 0) else 0
        
        # C. Middle tier two feedback events from S-1
        if idx < 1:
            mid_N = 0
            mid_n = 0
            mid_u = 0
            mid_nonprom_k = None
            mid_nonprom_ratio = None
            mid_nonprom_lb = None
            mid_nonprom_ub = None
            mid_sev_k = None
            mid_sev_ratio = None
            mid_sev_lb = None
            mid_sev_ub = None
        else:
            day_prev = sessions[idx - 1]
            p_prev = closing_pool.get(day_prev, {})
            mid_cohort = {c: v for c, v in p_prev.items() if v['boards'] in (3, 4)}
            mid_N = len(mid_cohort)
            if mid_N == 0:
                mid_n = 0
                mid_u = 0
                mid_nonprom_k = None
                mid_nonprom_ratio = None
                mid_nonprom_lb = None
                mid_nonprom_ub = None
                mid_sev_k = None
                mid_sev_ratio = None
                mid_sev_lb = None
                mid_sev_ub = None
            else:
                n_obs = 0
                u_unk = 0
                k_nonprom = 0
                k_sev = 0
                for c, v in mid_cohort.items():
                    if (day, c) not in bars_valid:
                        u_unk += 1
                        continue
                    n_obs += 1
                    # Event 1: Non-promoted
                    promoted = (c in p_s and p_s[c]['boards'] >= v['boards'] + 1)
                    if not promoted:
                        k_nonprom += 1
                        # Event 2: Severe damage (blown board or near limit-down <= -8.5%)
                        blown = (day, c) in failed_board_attempts
                        ret = bars_ret.get((day, c), 1.0)
                        limit_down = (ret <= 0.915)
                        if blown or limit_down:
                            k_sev += 1
                            
                mid_n = n_obs
                mid_u = u_unk
                if n_obs == 0:
                    mid_nonprom_k = None
                    mid_nonprom_ratio = None
                    mid_nonprom_lb = None
                    mid_nonprom_ub = None
                    mid_sev_k = None
                    mid_sev_ratio = None
                    mid_sev_lb = None
                    mid_sev_ub = None
                else:
                    mid_nonprom_k = k_nonprom
                    mid_nonprom_ratio = round(k_nonprom / n_obs, 4)
                    mid_nonprom_lb = round(k_nonprom / mid_N, 4)
                    mid_nonprom_ub = round((k_nonprom + u_unk) / mid_N, 4)
                    
                    mid_sev_k = k_sev
                    mid_sev_ratio = round(k_sev / n_obs, 4)
                    mid_sev_lb = round(k_sev / mid_N, 4)
                    mid_sev_ub = round((k_sev + u_unk) / mid_N, 4)
                    
        # D. Same height amount rank pct and delta
        pct_curr = percentile_cache.get(day, {}).get((b, code), None)
        pct_curr_val = round(float(pct_curr), 4) if pct_curr is not None else np.nan
        
        if pct_curr is None or idx < 1:
            delta_amt_rank = np.nan
        else:
            day_prev = sessions[idx - 1]
            pct_prev = percentile_cache.get(day_prev, {}).get((b - 1, code), None)
            if pct_prev is None:
                delta_amt_rank = np.nan
            else:
                delta_amt_rank = round(float(pct_curr - pct_prev), 4)
                
        rows.append({
            'date': day,
            'code': code,
            'boards': b,
            'anchor_break_today': brk_today,
            'anchor_break_age_sessions': brk_age,
            'mid_tier_void': mid_void,
            'mid_tier_cohort_size': mid_N,
            'mid_tier_observed_count': mid_n,
            'mid_tier_unknown_count': mid_u,
            'mid_tier_nonpromoted_count': mid_nonprom_k,
            'mid_tier_nonpromoted_ratio': mid_nonprom_ratio,
            'mid_tier_nonpromoted_lower_bound': mid_nonprom_lb,
            'mid_tier_nonpromoted_upper_bound': mid_nonprom_ub,
            'mid_tier_severe_damaged_count': mid_sev_k,
            'mid_tier_severe_damaged_ratio': mid_sev_ratio,
            'mid_tier_severe_damaged_lower_bound': mid_sev_lb,
            'mid_tier_severe_damaged_upper_bound': mid_sev_ub,
            'same_height_amount_rank_pct': pct_curr_val,
            'same_height_amount_rank_delta': delta_amt_rank
        })
        
    return pd.DataFrame(rows)

def main():
    start_time = time.time()
    root = r"d:\Documents\ChatGPT\龙头战法"
    dataset_path = os.path.join(root, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\historical_final\dataset.joblib")
    out_dir = os.path.join(root, r"research\relay_joint_research_20261021T1142Z\p1_correction_v2")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading dataset for V3 supplement...")
    d = joblib.load(dataset_path)
    sessions = d['sessions']
    attempts = d['attempts']
    bars = d['bars']
    features = d['features']
    relay = d['prepared']['relay']
    
    # 1. Build supplemented V3 features
    print("Building supplemented features across 9,245 rows...")
    df_v3 = build_p1_features_v3(sessions, attempts, bars, features)
    assert len(df_v3) == 9245
    assert not df_v3.duplicated(['date', 'code']).any()
    
    # Merge targets
    relay_sub = relay[['date', 'code', 'target', 'promoted_next_day', 'next_open_gap', 'large_loss', 'net_return']]
    features_sub = features[['date', 'code', 'daily_one_word', 'name', 'name_cluster_members', 'theme_first_support', 'theme_relay_support']]
    merged_sub = pd.merge(relay_sub, features_sub, on=['date', 'code'], how='inner')
    merged = pd.merge(df_v3, merged_sub, on=['date', 'code'], how='inner')
    assert len(merged) == 9245
    
    v3_csv_path = os.path.join(out_dir, "p1_features_9245_v3.csv")
    merged.to_csv(v3_csv_path, index=False)
    v3_sha = sha256_file(v3_csv_path)
    print(f"Saved: {v3_csv_path} (SHA: {v3_sha})")
    
    # 2. Check prefix invariance
    print("Verifying prefix invariance at 2025-06-30 and 2025-12-31...")
    prefix_results = {}
    for cutoff in ['2025-06-30', '2025-12-31']:
        df_cut = build_p1_features_v3(sessions, attempts, bars, features, cutoff_date=cutoff)
        sub_full = df_v3[df_v3['date'] <= cutoff].copy().reset_index(drop=True)
        assert len(df_cut) == len(sub_full)
        mismatches = 0
        for col in df_cut.columns:
            s1 = df_cut[col]
            s2 = sub_full[col]
            if col == 'anchor_break_today':
                s1, s2 = s1.astype(str), s2.astype(str)
            diff = ~(s1.eq(s2) | (s1.isna() & s2.isna()))
            mismatches += int(diff.sum())
        assert mismatches == 0, f"Mismatches at {cutoff}: {mismatches}"
        prefix_results[cutoff] = {"rows": len(df_cut), "mismatches": 0, "status": "VERIFIED"}
        print(f" - {cutoff}: {len(df_cut)} rows, 0 mismatches.")
        
    # 3. Generate 3D Condition Cell summary
    print("Summarizing 3D Condition Cells (Age x Board x DailyOneWord)...")
    cells = []
    for (age, b, noword), g in merged.groupby(['anchor_break_age_sessions', 'boards', 'daily_one_word'], dropna=False):
        cells.append({
            'anchor_break_age_sessions': str(age),
            'boards': int(b) if pd.notnull(b) else None,
            'daily_one_word': float(noword) if pd.notnull(noword) else None,
            'N': len(g),
            'known': int(g.target.notna().sum()),
            'unknown': int(g.target.isna().sum()),
            'hits': int(g.target.eq(1).sum()),
            'hit_rate': round(float(g.target.mean()), 4) if g.target.notna().sum() > 0 else None
        })
    cells_json_path = os.path.join(out_dir, "P1_3D_CONDITION_CELLS.json")
    with open(cells_json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(cells), f, ensure_ascii=False, indent=2)
    print(f"Saved: {cells_json_path}")
    
    # 4. Generate Receipt
    receipt = {
        "status": "P1_SUPPLEMENT_V3_COMPLETED",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "fits": 0,
        "http_requests": 0,
        "supplement_summary": {
            "middle_tier_event_1": "mid_tier_nonpromoted_ratio (did not close limit up)",
            "middle_tier_event_2": "mid_tier_severe_damaged_ratio (failed board observation or close <= 0.915 pre_close)",
            "cohort_bounds": "Explicit N, n_obs, u_unk, lower bound k/N, upper bound (k+u)/N",
            "empty_cohort_handling": "N=0 produces None, not 0.0%",
            "individual_features": "same_height_amount_rank_pct (day S cohort percentile) and same_height_amount_rank_delta (day S - day S-1)"
        },
        "prefix_invariance": prefix_results,
        "features_csv": "p1_features_9245_v3.csv",
        "features_sha256": v3_sha
    }
    receipt_json_path = os.path.join(out_dir, "P1_SUPPLEMENT_V3_RECEIPT.json")
    with open(receipt_json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(receipt), f, ensure_ascii=False, indent=2)
    print(f"Saved: {receipt_json_path}")
    print("V3 Supplement Finished Successfully!")

if __name__ == '__main__':
    main()
