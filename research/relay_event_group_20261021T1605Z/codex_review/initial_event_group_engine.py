"""
Event-Anchored Cohort & Relative Status Engine (Zero-Fit Exploration)
=====================================================================
Implements the event-anchored narrative cohort mechanism:
1. Anchor Emergence Event E: An anchor stock reaching board >= 3 at date E (E < S, within [S-10, S-1]).
2. Fixed Cohort Formation G_E: Union of all stocks having CLOSING_LIMIT_UP or FAILED_BOARD_OBSERVATION
   in [E-4, E] that share distinctive semantic tokens (with stopword filtering and phonetic mapping).
   Cohort is strictly fixed at E close; members do NOT leave the denominator upon failure.
3. Tracking from E to S:
   - cohort_size (N)
   - cohort_damage_ratio: proportion of cohort members not closing limit-up on day S.
   - cohort_repair_ratio: proportion of cohort members damaged on S-1 showing recovery/bounce on day S.
   - is_cohort_primary_member: 1 if candidate was in G_E at E (co-starter), 0 if diffusion.
   - anchor_status_today: state of anchor on day S (2=limit-up, 1=broken, 0=severe damage, -1=unobserved).
4. Strictly 0 model fits, 0 network requests.
"""

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
DATASET_PATH = os.path.join(ROOT, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\historical_final\dataset.joblib")
OUT_DIR = os.path.join(ROOT, r"research\relay_event_group_20261021T1605Z")

STOPWORDS = {
    '中国', '中华', '股份', '集团', '控股', '科技', '发展', '产业', '投资', '实业',
    '能源', '电气', '精密', '重工', '新', '中', '国', '华', '天', '金', '大', '通',
    '东', '南', '西', '北', '有限', '公司', '智能', '信息', '通信', '材料', '电子',
    '环境', '装备', '生物', '商旅', '食品', '动力', '工程', '建工', '重机', '机械'
}

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

def extract_tokens(name):
    if not isinstance(name, str):
        return set()
    cleaned = name
    for sw in ['股份', '集团', '控股', '科技', '发展', '有限', '公司']:
        cleaned = cleaned.replace(sw, '')
    tokens = set()
    n = len(cleaned)
    for i in range(n - 1):
        bg = cleaned[i:i+2]
        if bg not in STOPWORDS:
            tokens.add(bg)
    # Phonetic mapping for homophone characters like 阳/洋 (e.g. 海洋王, 九阳, 百洋)
    if '阳' in cleaned or '洋' in cleaned:
        tokens.add('阳_洋')
    if '东方' in cleaned:
        tokens.add('东方')
    return tokens

def build_event_cohort_features(sessions, attempts, bars, features, cutoff_date=None):
    if cutoff_date is not None:
        sessions = [s for s in sessions if s <= cutoff_date]
        features = features[features['date'] <= cutoff_date].copy()
        attempts = attempts[attempts['signal_date'] <= cutoff_date].copy()
        bars = bars[bars['date'] <= cutoff_date].copy()

    s_idx = {s: i for i, s in enumerate(sessions)}
    
    # 0. Build stock names and tokens
    code_to_name = {}
    bar_info = {}
    for r in bars.itertuples():
        c = str(r.code).zfill(6)
        if hasattr(r, 'name') and pd.notnull(r.name) and c not in code_to_name:
            code_to_name[c] = r.name
        cl_cents = _cent(r.close)
        pr_cents = _cent(r.pre_close)
        amt = r.amount if (r.amount is not None and pd.notnull(r.amount) and np.isfinite(r.amount)) else None
        bar_info[(r.date, c)] = {
            'close_cents': cl_cents,
            'pre_close_cents': pr_cents,
            'amount': amt,
            'name': getattr(r, 'name', '')
        }

    for r in attempts.itertuples():
        c = str(r.code).zfill(6)
        if hasattr(r, 'name') and pd.notnull(r.name) and c not in code_to_name:
            code_to_name[c] = r.name

    stock_tokens = {c: extract_tokens(name) for c, name in code_to_name.items()}

    # Active attempts by date
    active_by_day = defaultdict(set)
    failed_board_attempts = set()
    closing_limit_ups = defaultdict(dict)
    
    for r in attempts[attempts.pool_assignment.isin(['CLOSING_LIMIT_UP', 'FAILED_BOARD_OBSERVATION'])].itertuples():
        day, c = r.signal_date, str(r.code).zfill(6)
        if day in s_idx:
            active_by_day[day].add(c)
            if r.pool_assignment == 'CLOSING_LIMIT_UP':
                closing_limit_ups[day][c] = int(r.board_height)
            elif r.pool_assignment == 'FAILED_BOARD_OBSERVATION':
                failed_board_attempts.add((day, c))

    for r in features.itertuples():
        day, c = r.date, str(r.code).zfill(6)
        if day in s_idx and c not in closing_limit_ups[day]:
            closing_limit_ups[day][c] = int(r.boards)
            active_by_day[day].add(c)

    # 1. Identify all Anchor Emergence Events E: first time reaching board >= 3
    emerged_anchors = [] # list of (E, e_idx, anchor_code, tokens)
    seen_anchor_streak = set()
    for r in attempts[(attempts.pool_assignment == 'CLOSING_LIMIT_UP') & (attempts.board_height == 3)].itertuples():
        day, c = r.signal_date, str(r.code).zfill(6)
        if day not in s_idx:
            continue
        if (day, c) in seen_anchor_streak:
            continue
        seen_anchor_streak.add((day, c))
        t = stock_tokens.get(c, set())
        if t:
            emerged_anchors.append((day, s_idx[day], c, t))

    # Precompute anchor cohorts G_E
    anchor_cohorts = {} # (E, anchor_code) -> set of member codes (excluding anchor itself)
    for E, e_idx, a_code, a_tokens in emerged_anchors:
        if (E, a_code) in anchor_cohorts:
            continue
        window_days = [sessions[k] for k in range(max(0, e_idx - 4), e_idx + 1)]
        pool_stocks = set()
        for wd in window_days:
            pool_stocks.update(active_by_day.get(wd, set()))
            
        members = set()
        for m in pool_stocks:
            if m == a_code:
                continue
            m_tokens = stock_tokens.get(m, set())
            if a_tokens & m_tokens:
                members.add(m)
        anchor_cohorts[(E, a_code)] = members

    # 2. Build row features for each candidate on day S
    rows = []
    for r in features.itertuples():
        day = r.date
        code = str(r.code).zfill(6)
        b = int(r.boards)
        s_pos = s_idx[day]
        cand_tokens = stock_tokens.get(code, set())
        
        # Find the best matched active anchor event E < S
        # Condition: E in [S-10, S-1], anchor shares tokens with candidate (or candidate was in anchor cohort)
        best_anchor = None
        best_cohort = None
        best_lag = 999
        
        for E, e_idx, a_code, a_tokens in emerged_anchors:
            lag = s_pos - e_idx
            if 1 <= lag <= 10:
                cohort = anchor_cohorts.get((E, a_code), set())
                # Match condition: candidate shares token with anchor OR candidate is in cohort
                shares_token = bool(cand_tokens & a_tokens)
                in_cohort = (code in cohort)
                if shares_token or in_cohort:
                    if lag < best_lag:
                        best_lag = lag
                        best_anchor = (E, a_code)
                        best_cohort = cohort
                        
        if best_anchor is None or not best_cohort:
            # Isolated line or no matched cohort
            has_cohort = 0
            cohort_size = 0
            is_prim_member = 0
            cohort_dmg_ratio = None
            cohort_rep_ratio = None
            anchor_status = -1
        else:
            has_cohort = 1
            E_date, a_code = best_anchor
            # Filter cohort to exclude candidate itself
            cohort_members = [m for m in best_cohort if m != code]
            cohort_size = len(cohort_members)
            is_prim_member = 1 if (code in best_cohort) else 0
            
            if cohort_size == 0:
                cohort_dmg_ratio = None
                cohort_rep_ratio = None
            else:
                # Track cohort on day S
                n_obs = 0
                k_damaged = 0
                k_repaired = 0
                prev_day = sessions[s_pos - 1] if s_pos > 0 else None
                
                for m in cohort_members:
                    info = bar_info.get((day, m))
                    if info is None or info['close_cents'] is None:
                        continue
                    n_obs += 1
                    # Was m closing limit-up on day S?
                    is_limit_s = (m in closing_limit_ups.get(day, {}))
                    if not is_limit_s:
                        k_damaged += 1
                        
                    # Did m repair on day S? (Damaged on S-1, but closed limit-up or had positive bounce > 3% on day S)
                    if prev_day is not None:
                        was_damaged_prev = (m not in closing_limit_ups.get(prev_day, {}))
                        if was_damaged_prev:
                            ret_s = None
                            if info['pre_close_cents'] and info['pre_close_cents'] > 0:
                                ret_s = (info['close_cents'] - info['pre_close_cents']) / info['pre_close_cents']
                            if is_limit_s or (ret_s is not None and ret_s >= 0.03):
                                k_repaired += 1
                                
                if n_obs == 0:
                    cohort_dmg_ratio = None
                    cohort_rep_ratio = None
                else:
                    cohort_dmg_ratio = round(k_damaged / n_obs, 4)
                    cohort_rep_ratio = round(k_repaired / n_obs, 4)
                    
            # Check anchor status on day S
            a_info = bar_info.get((day, a_code))
            if a_info is None or a_info['close_cents'] is None:
                anchor_status = -1 # Unobserved
            elif a_code in closing_limit_ups.get(day, {}):
                anchor_status = 2 # Closing limit up
            else:
                blown = (day, a_code) in failed_board_attempts
                is_ld = False
                if a_info['pre_close_cents'] is not None:
                    floor_cents = _limit_down_cents(a_info['pre_close_cents'], a_code, a_info['name'])
                    if floor_cents is not None and a_info['close_cents'] <= floor_cents:
                        is_ld = True
                if blown or is_ld:
                    anchor_status = 0 # Severe damage
                else:
                    anchor_status = 1 # Broken / regular correction
                    
        rows.append({
            'date': day,
            'code': code,
            'event_cohort_matched': has_cohort,
            'event_cohort_size': cohort_size,
            'event_cohort_primary_member': is_prim_member,
            'event_cohort_damage_ratio': cohort_dmg_ratio,
            'event_cohort_repair_ratio': cohort_rep_ratio,
            'event_anchor_status': anchor_status
        })
        
    return pd.DataFrame(rows)

def main():
    start_time = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Event-Anchored Cohort Construction (Zero-Fit Exploration)")
    print("=" * 60)
    print(f"Loading dataset: {DATASET_PATH}")
    dataset_sha = sha256_file(DATASET_PATH)
    print(f"Dataset SHA256: {dataset_sha}")
    
    d = joblib.load(DATASET_PATH)
    sessions = d['sessions']
    attempts = d['attempts']
    bars = d['bars']
    features = d['features']
    relay = d['prepared']['relay']
    
    # 1. Build event cohort features across all 9,245 rows
    print("\n[Step 1] Building event cohort features for all 9,245 rows...")
    df_cohort = build_event_cohort_features(sessions, attempts, bars, features)
    assert len(df_cohort) == 9245
    assert not df_cohort.duplicated(['date', 'code']).any()
    
    # Merge with target
    merged = pd.merge(df_cohort, relay[['date', 'code', 'target', 'boards']], on=['date', 'code'], how='inner')
    assert len(merged) == 9245
    
    feat_csv_path = os.path.join(OUT_DIR, "event_group_features_9245.csv")
    merged.to_csv(feat_csv_path, index=False)
    print(f"Saved: {feat_csv_path} (SHA: {sha256_file(feat_csv_path)})")
    
    # 2. Check Prefix Invariance
    print("\n[Step 2] Verifying causal prefix invariance at 2025-06-30 and 2025-12-31...")
    prefix_results = {}
    for cutoff in ['2025-06-30', '2025-12-31']:
        df_cut = build_event_cohort_features(sessions, attempts, bars, features, cutoff_date=cutoff)
        sub_full = df_cohort[df_cohort['date'] <= cutoff].copy().reset_index(drop=True)
        assert len(df_cut) == len(sub_full)
        mismatches = 0
        for col in df_cut.columns:
            s1 = df_cut[col]
            s2 = sub_full[col]
            diff = ~(s1.eq(s2) | (s1.isna() & s2.isna()))
            mismatches += int(diff.sum())
        assert mismatches == 0, f"Mismatches at {cutoff}: {mismatches}"
        prefix_results[cutoff] = {"rows": len(df_cut), "mismatches": 0, "status": "VERIFIED_INVARIANT"}
        print(f" - {cutoff}: {len(df_cut)} rows, 0 mismatches. Verified.")
        
    prefix_path = os.path.join(OUT_DIR, "PREFIX_INVARIANCE_CHECK.json")
    with open(prefix_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(prefix_results), f, ensure_ascii=False, indent=2)
        
    # 3. Compute Coverage Metrics
    print("\n[Step 3] Computing cohort coverage metrics...")
    total_candidates = len(merged)
    matched_candidates = int(merged['event_cohort_matched'].sum())
    primary_members = int(merged['event_cohort_primary_member'].sum())
    multi_member_cohorts = int((merged['event_cohort_size'] >= 2).sum())
    
    coverage = {
        "total_candidates": total_candidates,
        "matched_candidates": matched_candidates,
        "matched_percentage": round(matched_candidates / total_candidates, 4),
        "primary_member_candidates": primary_members,
        "diffusion_candidates": matched_candidates - primary_members,
        "multi_member_cohort_candidates": multi_member_cohorts,
        "damage_ratio_observed_count": int(merged['event_cohort_damage_ratio'].notna().sum()),
        "repair_ratio_observed_count": int(merged['event_cohort_repair_ratio'].notna().sum()),
        "hit_rate_matched_vs_unmatched": {
            "matched_known_hit_rate": round(float(merged[merged['event_cohort_matched'] == 1].target.mean()), 4),
            "unmatched_known_hit_rate": round(float(merged[merged['event_cohort_matched'] == 0].target.mean()), 4)
        }
    }
    coverage_path = os.path.join(OUT_DIR, "EVENT_GROUP_COVERAGE.json")
    with open(coverage_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(coverage), f, ensure_ascii=False, indent=2)
    print(f"Coverage: {coverage['matched_percentage']*100:.2f}% matched ({matched_candidates}/{total_candidates}).")
    print(f"Saved: {coverage_path}")
    
    # 4. Generate Cohort Ledger Sample (showing actual rosters including failures)
    print("\n[Step 4] Generating Cohort Roster Ledger...")
    sample_ledger = []
    # Find sample dates where interesting multi-member cohorts exist
    sample_sub = merged[(merged['event_cohort_size'] >= 2) & merged['event_cohort_damage_ratio'].notna()].head(10)
    for r in sample_sub.itertuples():
        sample_ledger.append({
            "signal_date": r.date,
            "candidate_code": r.code,
            "candidate_boards": int(r.boards),
            "candidate_target": int(r.target) if pd.notnull(r.target) else None,
            "cohort_size": int(r.event_cohort_size),
            "is_primary_member": int(r.event_cohort_primary_member),
            "cohort_damage_ratio": float(r.event_cohort_damage_ratio) if pd.notnull(r.event_cohort_damage_ratio) else None,
            "cohort_repair_ratio": float(r.event_cohort_repair_ratio) if pd.notnull(r.event_cohort_repair_ratio) else None,
            "anchor_status_today": int(r.event_anchor_status)
        })
    ledger_path = os.path.join(OUT_DIR, "EVENT_GROUP_LEDGER.json")
    with open(ledger_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(sample_ledger), f, ensure_ascii=False, indent=2)
    print(f"Saved: {ledger_path}")
    print("=" * 60)
    print(f"Completed in {round(time.time() - start_time, 2)}s. Zero fits, zero network requests.")
    print("=" * 60)

if __name__ == '__main__':
    main()
