"""Read existing artifacts; never fit models or modify Gemini's files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'research/model_upgrade_validation/dragon_3to5_breakthrough_20261027T064500Z'
DATA = ROOT / 'data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.strftime('%Y-%m-%d')
    return x

def metrics(g, target='target_5'):
    z = g[g[target].notna()].copy()
    return {'n': len(z), 'positive': int(z[target].sum()),
            'rate': float(z[target].mean()),
            'auc': float(roc_auc_score(z[target], z.pred_prob)),
            'average_precision': float(average_precision_score(z[target], z.pred_prob)),
            'brier': float(brier_score_loss(z[target], z.pred_prob))}

def daily_summary(g):
    ordered = g.sort_values(['date', 'pred_prob', 'code'], ascending=[True, False, True])
    top = ordered.groupby('date').head(1).copy()
    base = g.groupby('date').target_5.mean()
    return {'days': len(top), 'rows': len(g), 'top1_hits': int(top.target_5.sum()),
            'top1_rate': float(top.target_5.mean()),
            'random_one_per_day_expected_hits': float(base.sum()),
            'random_one_per_day_expected_rate': float(base.mean()),
            'row_weighted_base_rate': float(g.target_5.mean()),
            'top1_four_hits': int(top.target_4_original.sum()),
            'top1_four_rate': float(top.target_4_original.mean()),
            'top1_one_word_signal_n': int(top.one_word_proxy.eq(1).sum()),
            'top1_one_word_signal_hits': int(top.loc[top.one_word_proxy.eq(1), 'target_5'].sum()),
            'no_winner_days': int(g.groupby('date').target_5.sum().eq(0).sum()),
            'with_winner_days': int(g.groupby('date').target_5.sum().gt(0).sum())}

def main():
    original_files = list(SRC.glob('*')) + [DATA, ROOT/'train_and_backtest_dragon_3to5.py',
        ROOT/'compare_two_arms.py', ROOT/'run_two_arm_experiment.py']
    before = {str(p.relative_to(ROOT)): sha(p) for p in original_files if p.is_file()}
    df = pd.read_csv(DATA, dtype={'code': str}, low_memory=False)
    df['code'] = df.code.str.zfill(6)
    assert not df.duplicated(['date', 'code']).any()
    b = df[df.boards.eq(3)].copy()
    b['target_5_original'] = b.terminal_board.ge(5).astype(int)
    b['target_4_original'] = b.terminal_board.ge(4).astype(int)
    cal = json.loads((ROOT/'data/v279/exchange_calendar_audit_20261002_v1/calendar.json').read_text(encoding='utf-8'))['sessions']
    nxt = dict(zip(cal, cal[1:]))
    keyed = df.set_index(['date', 'code'])
    rows = []
    for r in b.itertuples():
        d1 = nxt.get(r.date)
        d2 = nxt.get(d1)
        y4 = np.nan
        y5 = np.nan
        known_on = None
        if r.trade_date == d1 and pd.notna(r.promoted_next_day):
            y4 = r.promoted_next_day
            if y4 == 0:
                y5 = 0
                known_on = d1
            elif (d1, r.code) in keyed.index:
                n1 = keyed.loc[(d1, r.code)]
                if n1.boards == 4 and n1.trade_date == d2 and pd.notna(n1.promoted_next_day):
                    y5 = n1.promoted_next_day
                    known_on = d2
        rows.append({'date': r.date, 'code': r.code, 'target_4_exact': y4,
                     'target_5_exact': y5, 'label_known_on': known_on,
                     'terminal_failure_known_on': nxt.get(r.terminal_date) if pd.notna(r.terminal_date) else None})
    b = b.merge(pd.DataFrame(rows), on=['date', 'code'], validate='one_to_one')
    o = pd.read_csv(SRC/'oof_predictions.csv', dtype={'code': str})
    o['code'] = o.code.str.zfill(6)
    o = o.merge(b.drop(columns=['name', 'boards', 'terminal_board']), on=['date', 'code'], validate='one_to_one')
    assert o.target_5.eq(o.target_5_original).all()
    comp = o.groupby('date').filter(lambda g: len(g) >= 2)
    top = comp.sort_values(['date', 'pred_prob', 'code'], ascending=[True, False, True]).groupby('date').head(1).copy()
    tiers = o.groupby('score_tier').target_5.agg(['count', 'sum', 'mean']).reset_index()
    years = {str(y): daily_summary(g) for y, g in comp.groupby(comp.date.str[:4])}
    folds = []
    for end, start, finish in [('2024-08-31','2024-09-01','2024-12-31'),
        ('2024-12-31','2025-01-01','2025-05-31'), ('2025-05-31','2025-06-01','2025-11-30'),
        ('2025-11-30','2025-12-01','2026-08-31')]:
        train = b[b.date.le(end)]
        test = b[b.date.between(start, finish)]
        late = train[train.label_known_on.ge(start) | train.target_5_exact.isna()]
        terminal_late = train[train.terminal_failure_known_on.ge(start)]
        terminal_late.assign(fold_start=start).to_csv(OUT/f'terminal_labels_not_mature_{start}.csv', index=False, encoding='utf-8-sig')
        changes = {}
        for col in ['price', 'float_cap_log']:
            a, z = float(train[col].median()), float(b[col].median())
            changes[col] = {'training_median': a, 'all_period_median': z,
                'training_flags_changed': int(train[col].le(a).ne(train[col].le(z)).sum()),
                'test_flags_changed': int(test[col].le(a).ne(test[col].le(z)).sum())}
        folds.append({'test_start': start, 'train_n': len(train), 'test_n': len(test),
            'training_two_session_labels_not_mature_or_unknown': len(late),
            'terminal_labels_requiring_test_period': len(terminal_late),
            'median_leakage': changes})
    missing = b[b.terminal_board.isna()].copy()
    missing.to_csv(OUT/'all_censored_terminal_rows.csv', index=False, encoding='utf-8-sig')
    disagreement = b[(b.target_5_exact.notna() & b.target_5_exact.ne(b.target_5_original)) |
                    (b.target_4_exact.notna() & b.target_4_exact.ne(b.target_4_original))]
    disagreement.to_csv(OUT/'exact_horizon_label_disagreements.csv', index=False, encoding='utf-8-sig')
    o.to_csv(OUT/'saved_scores_with_label_audit.csv', index=False, encoding='utf-8-sig')
    top.to_csv(OUT/'competitive_top1_recomputed.csv', index=False, encoding='utf-8-sig')
    fields = json.loads((SRC/'manifest.json').read_text(encoding='utf-8'))['features']
    original_feature_fields = [f for f in fields if f in b]
    coverage = {f: {'non_null': int(b[f].notna().sum()), 'unique': int(b[f].nunique()),
        'zero_count': int(b[f].eq(0).sum())} for f in original_feature_fields}
    base_manifest_path = ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/manifest.json'
    bm = json.loads(base_manifest_path.read_text(encoding='utf-8'))
    base_model = base_manifest_path.parent/'model.joblib'
    # Score saved full-data model only; no fitting. Verifies the preprocessing needed at inference.
    import importlib.util
    import joblib
    spec = importlib.util.spec_from_file_location('gemini_trainer_readonly', ROOT/'train_and_backtest_dragon_3to5.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engineered, feature_cols = module.prepare_features(df)
    final_model = joblib.load(SRC/'model.joblib')
    final_scores = final_model.predict_proba(engineered[feature_cols])[:, 1]
    base_pipeline = joblib.load(base_model)
    base_input = b.copy()
    for col in ['price', 'signal_amount', 'turnover', 'turnover_vs_prev']:
        base_input[col+'_same_height_percentile'] = base_input.groupby(['date','boards'])[col].rank(pct=True)
    base_input['signal_one_word'] = base_input.one_word_proxy
    base_input['log_signal_amount'] = np.log1p(base_input.signal_amount.clip(lower=0))
    base_input['base_score_retrospective'] = base_pipeline.predict_proba(base_input[list(base_pipeline.feature_names_in_)])[:,1]
    base_o = o[['date','code']].merge(base_input, on=['date','code'], validate='one_to_one')
    base_comp = base_o.groupby('date').filter(lambda g: len(g)>=2)
    base_top = base_comp.sort_values(['date','base_score_retrospective','code'], ascending=[True,False,True]).groupby('date').head(1)
    full_model_diagnostic = {'loaded': True, 'features': len(final_model.feature_names_in_),
        'scored_n': len(final_scores), 'score_min': float(final_scores.min()), 'score_max': float(final_scores.max()),
        'warning': 'Full-data fitted scores are not out-of-sample; no performance metric is reported.'}
    report = {'audit_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'READ_EXISTING_ARTIFACTS_NO_FIT_NO_OVERWRITE',
        'data': {'total_3board_n': len(b), 'dates': b.date.nunique(), 'first': b.date.min(), 'last': b.date.max(),
            'terminal_missing': len(missing), 'terminal_label_status': b.terminal_height_label_status.value_counts(dropna=False).to_dict(),
            'exact_5_unknown': int(b.target_5_exact.isna().sum()),
            'exact_5_positive': int(b.target_5_exact.eq(1).sum()),
            'exact_5_disagreements': int((b.target_5_exact.notna() & b.target_5_exact.ne(b.target_5_original)).sum()),
            'exact_4_disagreements': int((b.target_4_exact.notna() & b.target_4_exact.ne(b.target_4_original)).sum())},
        'saved_oof_metrics_original_labels': metrics(o), 'all_signal_days_original_labels': daily_summary(o),
        'competitive_days_original_labels': daily_summary(comp),
        'top2_individual_hits': int(comp.sort_values(['date','pred_prob','code'],ascending=[True,False,True]).groupby('date').head(2).target_5.sum()),
        'competitive_top1': {'failed_5': int(top.target_5.eq(0).sum()),
            'failed_3to4_original': int(top.target_4_original.eq(0).sum()),
            'reached_4_failed_5_original': int((top.target_4_original.eq(1) & top.target_5.eq(0)).sum()),
            'next_open_nonpositive_known_n': int(top.next_open_gap.le(0).sum()),
            'next_open_unknown_n': int(top.next_open_gap.isna().sum()),
            'next_trade_one_word_n': int(top.trade_one_word_up.eq(True).sum())},
        'tiers_original_labels': tiers.to_dict(orient='records'), 'years_original_labels': years,
        'folds': folds, 'feature_coverage': coverage,
        'additional_data_coverage': {f: {'non_null': int(b[f].notna().sum()), 'unique': int(b[f].nunique())}
            for f in ['theme','theme_status','theme_support','first_seal_minute','last_seal_minute','bomb_count','sealed_fund_float_cap']},
        'base': {'training_max_label_date': bm['training_max_label_date'], 'formed_at': bm['formed_at'],
            'actual_sha256': sha(base_model), 'expected_sha256': bm['sha256'],
            'hash_matches': sha(base_model) == bm['sha256'],
            'retrospective_top1_hits_recomputed': int(base_top.target_5_original.sum()),
            'retrospective_competitive_days': len(base_top),
            'comparison_warning': 'Frozen through Aug 2026 BASE is not historical OOS for Sept 2024-Aug 2026.'},
        'final_model_load': full_model_diagnostic,
        'source_hashes_before': before}
    after = {str(p.relative_to(ROOT)): sha(p) for p in original_files if p.is_file()}
    report['original_files_unchanged_during_audit'] = before == after
    manifest = json.loads((SRC/'manifest.json').read_text(encoding='utf-8'))
    report['checks'] = {
        'oof_rows_match_manifest': len(o) == manifest['backtest_evaluation']['out_of_sample_samples'],
        'auc_matches_manifest': abs(report['saved_oof_metrics_original_labels']['auc'] - manifest['backtest_evaluation']['auc_roc']) < 1e-12,
        'top1_rate_matches_manifest': abs(report['competitive_days_original_labels']['top1_rate'] - manifest['backtest_evaluation']['head_to_head_competition']['top1_pick_5board_rate']) < 1e-12,
        'base_hash_matches': report['base']['hash_matches'],
        'source_files_preserved': report['original_files_unchanged_during_audit'],
    }
    assert all(report['checks'].values()), report['checks']
    (OUT/'audit.json').write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(clean({k:v for k,v in report.items() if k not in ['source_hashes_before','feature_coverage','additional_data_coverage']}), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
