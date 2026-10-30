"""Audit round-two labels and existing scores. No training or source writes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SRC = ROOT/'research/model_upgrade_validation/dragon_stage_leader_20261027T071500Z'
DATA = ROOT/'data/v16/point_in_time_rebuild_cutoff_20260831_v2/leader_candidates.csv'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def label(g, tb):
    if len(g) == 0:
        return (0,0,0,0)
    space = int(g.daily_max_flag.max())
    unique = int(g.unique_daily_max_flag.max())
    dragon = int(space and tb >= 5 and
        (g.turnover.mean() >= .04 or g.amplitude.max() >= 2.5) and g.consecutive_count_audit.mean() >= 3)
    return (space,unique,int(g.daily_max_flag.sum()),dragon)

def counts(g, flag):
    k = g[flag].notna()
    return {'n': len(g), 'known': int(k.sum()), 'unknown': int((~k).sum()),
        'hits': int(g.loc[k, flag].sum()),
        'rate_known': float(g.loc[k,flag].mean()) if k.any() else None}

def stats(g):
    return {f: counts(g,f) for f in ['is_market_space_leader','is_emotion_dragon_proxy','is_3to5',
        'bounded_space_including_signal','strict_future_space','strict_dragon_proxy',
        'already_space_on_signal','bounded_dragon_proxy']}

def main():
    originals = [p for p in SRC.iterdir() if p.is_file()] + [DATA,ROOT/'audit_three_board_traps.py']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in originals}
    df = pd.read_csv(DATA,dtype={'code':str},low_memory=False)
    df.code = df.code.str.zfill(6)
    ds = df.groupby('date').boards.agg(['max','size']).rename(columns={'max':'max_board_audit','size':'consecutive_count_audit'})
    df = df.merge(ds,left_on='date',right_index=True,validate='many_to_one').copy()
    df['daily_max_flag'] = df.boards.eq(df.max_board_audit).astype(int)
    nm = df.groupby('date').daily_max_flag.sum()
    df['unique_daily_max_flag'] = (df.daily_max_flag.eq(1) & df.date.map(nm).eq(1)).astype(int)
    assert not df.duplicated(['date','code']).any()
    keyed = df.set_index(['date','code'])
    grouped = {c:g.sort_values('date') for c,g in df.groupby('code')}
    cal = json.loads((ROOT/'data/v279/exchange_calendar_audit_20261002_v1/calendar.json').read_text(encoding='utf-8'))['sessions']
    nxt = dict(zip(cal,cal[1:]))
    stage = pd.read_csv(SRC/'stage_leader_dataset.csv',dtype={'code':str},low_memory=False)
    stage.code = stage.code.str.zfill(6)
    oof = pd.read_csv(SRC/'oof_round2_predictions.csv',dtype={'code':str})
    oof.code = oof.code.str.zfill(6)
    assert not stage.duplicated(['date','code']).any()
    assert not oof.duplicated(['date','code']).any()
    records = []
    folds = [('2024-08-31','2024-09-01'),('2024-12-31','2025-01-01'),
        ('2025-05-31','2025-06-01'),('2025-11-30','2025-12-01')]
    late_records = []
    for r in stage.itertuples():
        same = grouped[r.code]
        actual = keyed.loc[(r.date,r.code)]
        # Exact reproduction of Gemini's unbounded label selection.
        wrong = same[(same.date >= r.date) & (same.boards >= 3) & (same.boards <= r.terminal_board)]
        lwrong = label(wrong,r.terminal_board)
        assert lwrong == (r.is_market_space_leader,r.is_unique_space_leader,r.space_leader_days,r.is_emotion_dragon_proxy)
        bounded = wrong[wrong.date <= r.terminal_date]
        lbound = label(bounded,r.terminal_board)
        # Follow ONLY exact adjacent official-session promotions from this three-board event.
        path = []
        d = r.date
        expected = 3
        completed = False
        failure_known_on = None
        for _ in range(150):
            if (d,r.code) not in keyed.index:
                break
            q = keyed.loc[(d,r.code)]
            if q.boards != expected:
                break
            path.append({**q.to_dict(),'date':d,'code':r.code})
            dnext = nxt.get(d)
            if q.trade_date != dnext or pd.isna(q.promoted_next_day):
                break
            if q.promoted_next_day == 0:
                completed = True
                failure_known_on = dnext
                break
            d = dnext
            expected += 1
        chain = pd.DataFrame(path)
        future = chain[chain.date > r.date]
        future_success = bool(len(future) and future.daily_max_flag.max())
        include_success = bool(chain.daily_max_flag.max())
        strict_space = int(future_success) if future_success or completed else np.nan
        strict_inc = int(include_success) if include_success or completed else np.nan
        strict_dragon = label(chain,int(chain.boards.max()))[3] if completed else np.nan
        records.append({'date':r.date,'code':r.code,'name':r.name,
            'original_terminal':actual.terminal_board,'round2_terminal':r.terminal_board,
            'original_terminal_date':actual.terminal_date,'round2_terminal_date':r.terminal_date,
            'original_censored':int(actual.terminal_height_censored),
            'wrong_selected_rows':len(wrong),'outside_current_run_rows':int(wrong.date.gt(r.terminal_date).sum()),
            'last_label_observation':wrong.date.max(),'bounded_space_including_signal':lbound[0],
            'bounded_unique_space':lbound[1],'bounded_space_days':lbound[2],'bounded_dragon_proxy':lbound[3],
            'strict_chain_completed':completed,'strict_chain_max':int(chain.boards.max()),
            'strict_chain_last_date':chain.date.max(),'failure_known_on':failure_known_on,
            'strict_space_including_signal':strict_inc,'strict_future_space':strict_space,
            'strict_dragon_proxy':strict_dragon,'already_space_on_signal':int(actual.daily_max_flag),
            'already_unique_on_signal':int(actual.unique_daily_max_flag)})
        for end,start in folds:
            if r.date <= end and r.terminal_date <= end:
                seen = wrong[wrong.date <= end]
                lseen = label(seen,r.terminal_board)
                late_records.append({'fold_start':start,'date':r.date,'code':r.code,'name':r.name,
                    'label_uses_observations_after_train_end':bool(wrong.date.gt(end).any()),
                    'space_target_changes_if_future_removed':lseen[0] != lwrong[0],
                    'dragon_target_changes_if_future_removed':lseen[3] != lwrong[3],
                    'current_streak_failure_not_yet_known':bool(failure_known_on and failure_known_on > end)})
    diag = pd.DataFrame(records)
    diag.to_csv(OUT/'all1851_label_diagnostics.csv',index=False,encoding='utf-8-sig')
    audit = oof.merge(diag,on=['date','code','name'],validate='one_to_one')
    audit.to_csv(OUT/'saved_predictions_with_label_diagnostics.csv',index=False,encoding='utf-8-sig')
    comp = audit.groupby('date').filter(lambda g:len(g)>=2)
    ordered = comp.sort_values(['date','score_composite_dragon','code'],ascending=[True,False,True])
    top = ordered.groupby('date').head(1).copy()
    rec = top[top.score_composite_dragon.ge(.4)]
    simple = comp.sort_values(['date','height_gap','code']) if 'height_gap' in comp else None
    late = pd.DataFrame(late_records)
    late.to_csv(OUT/'training_label_future_observations.csv',index=False,encoding='utf-8-sig')
    training = late.groupby('fold_start').agg(
        n=('date','size'),label_reads_beyond_train=('label_uses_observations_after_train_end','sum'),
        space_labels_changed=('space_target_changes_if_future_removed','sum'),
        dragon_labels_changed=('dragon_target_changes_if_future_removed','sum'),
        failure_not_known=('current_streak_failure_not_yet_known','sum')).reset_index()
    raw_labels = stage[['date','code','is_market_space_leader','is_emotion_dragon_proxy']].merge(diag,on=['date','code'],validate='one_to_one')
    changes = {f:int(raw_labels[f].ne(raw_labels[t]).sum()) for f,t in [
        ('is_market_space_leader','bounded_space_including_signal'),
        ('is_emotion_dragon_proxy','bounded_dragon_proxy')]}
    changed = raw_labels[raw_labels.is_market_space_leader.ne(raw_labels.bounded_space_including_signal) |
        raw_labels.is_emotion_dragon_proxy.ne(raw_labels.bounded_dragon_proxy)]
    changed.to_csv(OUT/'labels_changed_by_current_run_bound.csv',index=False,encoding='utf-8-sig')
    top.to_csv(OUT/'top1_diagnostics.csv',index=False,encoding='utf-8-sig')
    # Original daily audit tags: inspect only; no invocation with fabricated inputs.
    daily = pd.read_csv(SRC/'daily_competition_round2.csv',dtype={'top1_code':str})
    kept = daily[(~daily.top1_audit_veto)&daily.top1_one_word_proxy.eq(0)]
    sizes = comp.groupby('date').size()
    matched_sizes = comp[comp.base_historical_score.notna()].groupby('date').size()
    matched_2 = matched_sizes[matched_sizes.ge(2)]
    all_tops = audit.sort_values(['date','score_composite_dragon','code'],ascending=[True,False,True]).groupby('date').head(1)
    base_manifest_path = ROOT/'research/model_upgrade_validation/fixed_20261007T112017Z/models/relay/BASE/manifest.json'
    bm=json.loads(base_manifest_path.read_text(encoding='utf-8'))
    base_path=base_manifest_path.parent/'model.joblib'
    # Check artifact existence/loadability only. Do not report fitted-sample scores as validation.
    import joblib
    models={}
    for f in SRC.glob('model*.joblib'):
        m=joblib.load(f)
        models[f.name]={'loaded':True,'feature_count':int(m.n_features_in_),'sha256':sha(f)}
    report={'audit_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'NO_FIT_NO_SOURCE_OVERWRITE', 'source_sha256':hashes,
        'raw_original_scores_top1':stats(top),'raw_original_scores_selected_040':stats(rec),
        'all_signal_days_top1_diagnostics':stats(all_tops),
        'original_metrics_recomputed':{'competitive_days':len(top),'selected_days':len(rec),
            'selected_coverage':len(rec)/len(top),
            'all_competitive_space_positive_rows':int(comp.is_market_space_leader.sum()),
            'all_competitive_dragon_positive_rows':int(comp.is_emotion_dragon_proxy.sum())},
        'label_bug':{'stage_rows':len(stage),'oof_rows':len(oof),
            'all_rows_with_outside_current_run_observations':int(diag.outside_current_run_rows.gt(0).sum()),
            'changed_labels_by_same_run_upper_bound':changes,
            'oof_space_label_changes':int(audit.is_market_space_leader.ne(audit.bounded_space_including_signal).sum()),
            'oof_dragon_label_changes':int(audit.is_emotion_dragon_proxy.ne(audit.bounded_dragon_proxy).sum()),
            'raw_top1_space_hits_already_high_at_signal':int(top.loc[top.is_market_space_leader.eq(1),'already_space_on_signal'].sum()),
            'selected_space_hits_already_high_at_signal':int(rec.loc[rec.is_market_space_leader.eq(1),'already_space_on_signal'].sum())},
        'training_label_leakage_by_fold':training.to_dict(orient='records'),
        'original_cutoff_rows':diag[diag.original_censored.eq(1)].to_dict(orient='records'),
        'saved_audit_tags':{'vetoed':int(daily.top1_audit_veto.sum()),
            'claimed_tradable_n':len(kept),'claimed_tradable_space_hits':int(kept.top1_is_space_leader.sum()),
            'claimed_tradable_dragon_hits':int(kept.top1_is_emotion_dragon.sum()),
            'claimed_tradable_base_below_50':int(kept.top1_base_score.lt(.5).sum()),
            'claimed_tradable_base_unknown':int(kept.top1_base_score.isna().sum()),
            'all_tag_counts':daily.top1_audit_tags.value_counts().to_dict()},
        'base_match_coverage':{'competitive_rows':len(comp),
            'matched_rows':int(comp.base_historical_score.notna().sum()),
            'matched_days_2plus':len(matched_2),
            'complete_days_2plus':int(matched_2.eq(sizes.reindex(matched_2.index)).sum()),
            'partial_days_2plus':int(matched_2.lt(sizes.reindex(matched_2.index)).sum())},
        'models':models,'base_hash_matches':sha(base_path)==bm['sha256'],
        'original_data_board_min':float(df.boards.min()),
        'fitted_new_models':0,'new_formal_predictions':0,'new_trades':0}
    after={str(p.relative_to(ROOT)):sha(p) for p in originals}
    report['all_original_files_unchanged']=hashes==after
    manifest=json.loads((SRC/'round2_comprehensive_metrics.json').read_text(encoding='utf-8'))
    report['checks']={'preserved_originals':report['all_original_files_unchanged'],
        'base_hash_matches':report['base_hash_matches'],
        'top1_space_rate_matches_saved':round(top.is_market_space_leader.mean(),4)==manifest['head_to_head_competition']['top1_space_rate'],
        'top1_dragon_rate_matches_saved':round(top.is_emotion_dragon_proxy.mean(),4)==manifest['head_to_head_competition']['top1_dragon_rate'],
        'selective_space_rate_matches_saved':round(rec.is_market_space_leader.mean(),4)==manifest['selective_recommendation']['rec_space_rate']}
    assert all(report['checks'].values()),report['checks']
    def default(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    report=clean(report)
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','original_cutoff_rows','models']},ensure_ascii=False,indent=2,default=default))

if __name__=='__main__':
    main()
