"""
Arm A 3-Anchor Pre-Validation Engine (Executable but Unexecuted)
==============================================================
Implements Codex's minimal experimental protocol:
1. Algorithmic Baseline Control: ST_DUAL (82 features, HistGradientBoostingClassifier).
2. Added Non-Redundant Path Interaction Features (5 features):
   - anchor_break_age_code (deterministic integer: 0..4, 5=NO_BREAK, -1=UNKNOWN)
   - mid_tier_void (0, 1)
   - mid_tier_nonpromoted_ratio (exact non-promotion rate from S-1, float or NaN)
   - mid_tier_severe_damaged_ratio (blown board or exact penny-rounded limit down, float or NaN)
   - same_height_amount_rank_delta (day-over-day amount rank percentile shift)
   (Note: same_height_amount_rank_pct dropped to avoid redundancy with signal_amount_same_height_percentile).
3. 3 Fixed Anchor Cutoffs:
   - 2025-07-01: 252 signal days, 4062 train rows -> 2025-07 test window (23 dates, 320 rows)
   - 2026-01-01: 252 signal days, 3243 train rows -> 2026-01 test window (20 dates, 323 rows)
   - 2026-09-01: 252 signal days, 3059 train rows -> 2026-09 test window (16 dates, 182 rows)
4. Exact Penny-Precision Limit-Down Calculation (ROUND_HALF_UP).
5. Pre-check Mode (default): Verifies schemas, loads saved ST_DUAL models, runs 0-fit control inference.
6. Execution Mode: Only run when explicitly instructed by Codex freeze!
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
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = r"d:\Documents\ChatGPT\龙头战法"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from short_term_thinking_preprocessing import EmptyTrainingColumnGuard
DATASET_PATH = os.path.join(ROOT, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\historical_final\dataset.joblib")
PREDICTIONS_PATH = os.path.join(ROOT, r"research\model_upgrade_validation\fixed_20261007T112017Z\relay_history\predictions.csv")
ST_DUAL_DIR = os.path.join(ROOT, r"research\model_upgrade_validation\short_term_thinking_20261015T071000Z\execution_after_binning_repair")
OUT_DIR = os.path.join(ROOT, r"research\relay_joint_research_20261021T1142Z\p1_correction_v2")

HGB_PARAMS = dict(
    learning_rate=0.06,
    max_iter=120,
    max_leaf_nodes=7,
    max_depth=3,
    min_samples_leaf=25,
    l2_regularization=3.0,
    early_stopping=False,
    random_state=20261015
)

AGE_MAPPING = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    'NO_BREAK_IN_WINDOW': 5, 'UNKNOWN': -1
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

def build_path_interaction_features(sessions, attempts, bars, features):
    s_idx = {s: i for i, s in enumerate(sessions)}
    
    # Track bars with exact penny cents
    bar_info = {}
    for r in bars.itertuples():
        c = str(r.code).zfill(6)
        cl_cents = _cent(r.close)
        pr_cents = _cent(r.pre_close)
        amt = r.amount if (r.amount is not None and pd.notnull(r.amount) and np.isfinite(r.amount)) else None
        bar_info[(r.date, c)] = {
            'close_cents': cl_cents,
            'pre_close_cents': pr_cents,
            'amount': amt,
            'name': getattr(r, 'name', '')
        }

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
            'amount': bar_info.get((day, code), {}).get('amount')
        }

    for r in features.itertuples():
        day = r.date
        if day not in s_idx:
            continue
        code = str(r.code).zfill(6)
        if code not in closing_pool[day]:
            closing_pool[day][code] = {
                'boards': int(r.boards),
                'amount': bar_info.get((day, code), {}).get('amount')
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
                pass
            elif (day_j, a) in bar_info and bar_info[(day_j, a)]['close_cents'] is not None:
                has_break = True
            else:
                has_unknown = True
                
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
        else:
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
        age_code = AGE_MAPPING.get(brk_age, -1)
        
        # B. Middle tier void at day S
        p_s = closing_pool.get(day, {})
        count_3 = sum(1 for v in p_s.values() if v['boards'] == 3)
        count_4 = sum(1 for v in p_s.values() if v['boards'] == 4)
        mid_void = 1 if (count_3 == 0 and count_4 == 0) else 0
        
        # C. Middle tier feedback from S-1 (Penny Precision)
        if idx < 1:
            mid_nonprom_ratio = None
            mid_sev_ratio = None
        else:
            day_prev = sessions[idx - 1]
            p_prev = closing_pool.get(day_prev, {})
            mid_cohort = {c: v for c, v in p_prev.items() if v['boards'] in (3, 4)}
            mid_N = len(mid_cohort)
            if mid_N == 0:
                mid_nonprom_ratio = None
                mid_sev_ratio = None
            else:
                n_obs = 0
                k_nonprom = 0
                k_sev = 0
                for c, v in mid_cohort.items():
                    info = bar_info.get((day, c))
                    if info is None or info['close_cents'] is None:
                        continue
                    n_obs += 1
                    # Event 1: non-promotion
                    promoted = (c in p_s and p_s[c]['boards'] >= v['boards'] + 1)
                    if not promoted:
                        k_nonprom += 1
                        # Event 2: severe damage
                        blown = (day, c) in failed_board_attempts
                        is_limit_down = False
                        if info['pre_close_cents'] is not None:
                            floor_cents = _limit_down_cents(info['pre_close_cents'], c, info['name'])
                            if floor_cents is not None and info['close_cents'] <= floor_cents:
                                is_limit_down = True
                        if blown or is_limit_down:
                            k_sev += 1
                            
                if n_obs == 0:
                    mid_nonprom_ratio = None
                    mid_sev_ratio = None
                else:
                    mid_nonprom_ratio = round(k_nonprom / n_obs, 4)
                    mid_sev_ratio = round(k_sev / n_obs, 4)
                    
        # D. Same height amount rank delta
        pct_curr = percentile_cache.get(day, {}).get((b, code), None)
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
            'anchor_break_age_code': age_code,
            'mid_tier_void': mid_void,
            'mid_tier_nonpromoted_ratio': mid_nonprom_ratio,
            'mid_tier_severe_damaged_ratio': mid_sev_ratio,
            'same_height_amount_rank_delta': delta_amt_rank
        })
        
    return pd.DataFrame(rows)

def training_weights(train):
    if train.empty or train.target.nunique() != 2:
        raise ValueError("both training target classes required")
    w = 1 / train.groupby("date").date.transform("size")
    mass = w.groupby(train.target).sum()
    w = w * train.target.map({k: .5 / v for k, v in mass.items()})
    return w * train.date.nunique()

def run_precheck():
    start_time = time.time()
    print("=" * 60)
    print("Running Arm A 3-Anchor Pre-Flight Verification (0 FITS)")
    print("=" * 60)
    
    # 1. Verify Dataset
    print(f"Loading dataset: {DATASET_PATH}")
    dataset_sha = sha256_file(DATASET_PATH)
    print(f"Dataset SHA256: {dataset_sha}")
    d = joblib.load(DATASET_PATH)
    sessions = d['sessions']
    attempts = d['attempts']
    bars = d['bars']
    features = d['features']
    relay = d['prepared']['relay']
    
    # 2. Verify ST_DUAL features
    m7 = joblib.load(os.path.join(ST_DUAL_DIR, "2025-07_ST_DUAL_model.joblib"))
    st_cols = m7['features']
    assert len(st_cols) == 82, f"Expected 82 features, got {len(st_cols)}"
    print(f"Verified ST_DUAL feature schema: 82 features.")
    
    # 3. Assemble merged dataset
    print("Merging relay prepared data and features...")
    merged_base = pd.merge(relay, features, on=['date', 'code'], how='inner', suffixes=('', '_feat'))
    assert len(merged_base) == 9245
    
    # 4. Build 5 non-redundant path interaction features
    print("Building 5 non-redundant path interaction features...")
    df_path = build_path_interaction_features(sessions, attempts, bars, features)
    assert len(df_path) == 9245
    
    merged = pd.merge(merged_base, df_path, on=['date', 'code'], how='inner')
    assert len(merged) == 9245
    
    added_cols = [
        'anchor_break_age_code',
        'mid_tier_void',
        'mid_tier_nonpromoted_ratio',
        'mid_tier_severe_damaged_ratio',
        'same_height_amount_rank_delta'
    ]
    all_arm_a_cols = st_cols + added_cols
    assert len(all_arm_a_cols) == 87, f"Expected 87 features, got {len(all_arm_a_cols)}"
    for col in all_arm_a_cols:
        assert col in merged.columns, f"Feature missing: {col}"
    print(f"All 87 features verified on merged table.")
    
    # 5. Verify the 3 anchor training sets and row counts
    anchors = [
        ('2025-07', '2025-07-01', 4062, 23, 320),
        ('2026-01', '2026-01-01', 3243, 20, 323),
        ('2026-09', '2026-09-01', 3059, 16, 182)
    ]
    anchor_checks = {}
    for month, cutoff, exp_train_rows, exp_dates, exp_test_rows in anchors:
        valid = merged.label_valid.fillna(False).astype(bool) & merged.target.notna()
        known = merged[valid & merged.date.lt(cutoff) & merged.target_date.lt(cutoff)].copy()
        dates = sorted(known.date.unique())[-252:]
        train = known[known.date.isin(dates)].copy()
        assert len(train) == exp_train_rows, f"Train row mismatch at {month}: {len(train)} vs {exp_train_rows}"
        assert train.date.nunique() == 252, f"Train signal dates != 252: {train.date.nunique()}"
        
        test = merged[merged.date.str.startswith(month)].copy()
        assert len(test) == exp_test_rows, f"Test row mismatch at {month}: {len(test)} vs {exp_test_rows}"
        assert test.date.nunique() == exp_dates, f"Test dates mismatch at {month}: {test.date.nunique()} vs {exp_dates}"
        
        anchor_checks[month] = {
            "cutoff": cutoff,
            "train_rows": len(train),
            "train_signal_days": train.date.nunique(),
            "test_rows": len(test),
            "test_dates": test.date.nunique(),
            "status": "VERIFIED_MATCHES_BASELINE"
        }
        print(f" - Anchor {month} ({cutoff}): train={len(train)} rows (252 days), test={len(test)} rows ({test.date.nunique()} days). Verified.")

    # 6. Run 0-fit Control Inference using saved ST_DUAL models
    print("\nRunning 0-fit control inference on the 3 saved ST_DUAL models...")
    st_dual_results = {}
    total_st_hits = 0
    total_nonword_dates = 0
    
    for month, cutoff, _, _, _ in anchors:
        model_path = os.path.join(ST_DUAL_DIR, f"{month}_ST_DUAL_model.joblib")
        saved_obj = joblib.load(model_path)
        model = saved_obj['model']
        
        test = merged[merged.date.str.startswith(month)].copy()
        test['score'] = model.predict_proba(test[st_cols])[:, 1]
        
        nonword = test[test.daily_one_word.eq(0)].copy()
        top1 = nonword.sort_values(['date', 'score', 'code'], ascending=[True, False, True]).groupby('date').head(1)
        hits = int(top1.target.eq(1).sum())
        dates_cnt = int(top1.date.nunique())
        
        total_st_hits += hits
        total_nonword_dates += dates_cnt
        st_dual_results[month] = {
            "nonword_dates": dates_cnt,
            "top1_hits": hits,
            "hit_rate": round(hits / dates_cnt, 4)
        }
        print(f" - {month} ST_DUAL Re-infer: {hits} / {dates_cnt} ({hits/dates_cnt*100:.2f}%)")
        
    print(f"Total ST_DUAL Control across 3 anchors: {total_st_hits} / {total_nonword_dates} ({total_st_hits/total_nonword_dates*100:.2f}%)")
    
    # 7. Write Pre-Check Receipt
    receipt = {
        "status": "ARM_A_3ANCHOR_PRECHECK_COMPLETED",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "fits": 0,
        "http_requests": 0,
        "dataset_sha256": dataset_sha,
        "anchor_checks": anchor_checks,
        "feature_whitelist_verified": {
            "base_st_dual_count": len(st_cols),
            "added_path_interaction_count": len(added_cols),
            "total_arm_a_count": len(all_arm_a_cols)
        },
        "st_dual_control_baseline": {
            "by_month": st_dual_results,
            "total_hits": total_st_hits,
            "total_dates": total_nonword_dates,
            "baseline_hit_rate": round(total_st_hits / total_nonword_dates, 4)
        },
        "acceptance_criteria": {
            "required_arm_a_hits": 29,
            "required_arm_a_dates": 59,
            "required_minimum_rate": 0.4915,
            "net_gain_days": 3
        },
        "execution_readiness": "READY_FOR_TRAINING_UPON_CODEX_FREEZE_INSTRUCTION"
    }
    
    receipt_path = os.path.join(OUT_DIR, "ARM_A_3ANCHOR_PRECHECK_RECEIPT.json")
    with open(receipt_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(receipt), f, ensure_ascii=False, indent=2)
    print(f"Saved: {receipt_path}")
    print("=" * 60)
    print("Pre-flight check passed with 0 fits! Ready for Codex review.")
    print("=" * 60)

def run_execute():
    start_time = time.time()
    print("=" * 60)
    print("Executing Arm A 3-Anchor Training (EXACTLY 3 FITS)")
    print("=" * 60)
    d = joblib.load(DATASET_PATH)
    sessions = d['sessions']
    attempts = d['attempts']
    bars = d['bars']
    features = d['features']
    relay = d['prepared']['relay']
    
    m7 = joblib.load(os.path.join(ST_DUAL_DIR, "2025-07_ST_DUAL_model.joblib"))
    st_cols = m7['features']
    
    merged_base = pd.merge(relay, features, on=['date', 'code'], how='inner', suffixes=('', '_feat'))
    df_path = build_path_interaction_features(sessions, attempts, bars, features)
    merged = pd.merge(merged_base, df_path, on=['date', 'code'], how='inner')
    
    added_cols = [
        'anchor_break_age_code',
        'mid_tier_void',
        'mid_tier_nonpromoted_ratio',
        'mid_tier_severe_damaged_ratio',
        'same_height_amount_rank_delta'
    ]
    all_arm_a_cols = st_cols + added_cols
    
    anchors = [
        ('2025-07', '2025-07-01', 4062, 23, 320),
        ('2026-01', '2026-01-01', 3243, 20, 323),
        ('2026-09', '2026-09-01', 3059, 16, 182)
    ]
    
    fitted_models = {}
    predictions_by_month = {}
    eval_results = {}
    total_hits = 0
    total_dates = 0
    
    for month, cutoff, exp_train_rows, exp_dates, exp_test_rows in anchors:
        print(f"\nFitting Arm A for anchor {month} ({cutoff})...")
        valid = merged.label_valid.fillna(False).astype(bool) & merged.target.notna()
        known = merged[valid & merged.date.lt(cutoff) & merged.target_date.lt(cutoff)].copy()
        dates = sorted(known.date.unique())[-252:]
        train = known[known.date.isin(dates)].copy()
        
        w = training_weights(train)
        model = Pipeline([
            ('empty_training_columns', EmptyTrainingColumnGuard()),
            ('hgb', HistGradientBoostingClassifier(**HGB_PARAMS))
        ])
        model.fit(train[all_arm_a_cols], train.target.astype(int), hgb__sample_weight=w)
        fitted_models[month] = model
        
        test = merged[merged.date.str.startswith(month)].copy()
        test['score'] = model.predict_proba(test[all_arm_a_cols])[:, 1]
        test['arm'] = 'ST_A_PATH_INTERACTION'
        predictions_by_month[month] = test
        
        # Evaluate Top-1
        nonword = test[test.daily_one_word.eq(0)].copy()
        top1 = nonword.sort_values(['date', 'score', 'code'], ascending=[True, False, True]).groupby('date').head(1)
        hits = int(top1.target.eq(1).sum())
        dates_cnt = int(top1.date.nunique())
        total_hits += hits
        total_dates += dates_cnt
        eval_results[month] = {
            "nonword_dates": dates_cnt,
            "top1_hits": hits,
            "hit_rate": round(hits / dates_cnt, 4)
        }
        print(f" - {month} Arm A Top-1: {hits} / {dates_cnt} ({hits/dates_cnt*100:.2f}%)")
        
        # Save prediction CSV
        out_pred = os.path.join(OUT_DIR, f"{month}_ARM_A_predictions.csv")
        test[['date', 'code', 'score', 'target', 'daily_one_word', 'arm']].to_csv(out_pred, index=False)
        
    print(f"\nTotal Arm A across 3 anchors: {total_hits} / {total_dates} ({total_hits/total_dates*100:.2f}%)")
    pass_gate = (total_hits >= 29)
    print(f"Acceptance Gate (>= 29 hits): {'PASSED' if pass_gate else 'FAILED'}")
    
    summary = {
        "status": "ARM_A_3ANCHOR_EVALUATION_COMPLETED",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "fits": 3,
        "http_requests": 0,
        "results_by_month": eval_results,
        "total_hits": total_hits,
        "total_dates": total_dates,
        "hit_rate": round(total_hits / total_dates, 4),
        "acceptance_gate_passed": pass_gate
    }
    with open(os.path.join(OUT_DIR, "ARM_A_3ANCHOR_EVALUATION.json"), 'w', encoding='utf-8') as f:
        json.dump(clean_json(summary), f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        run_execute()
    else:
        run_precheck()
