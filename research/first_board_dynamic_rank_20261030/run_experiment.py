"""Prepare -> genuinely peer-frozen protocol -> one fixed monthly replay."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from dynamic_features import build, FIELDS, THEME, ANCHOR, KEYS
from pairwise_residual import choose, fit, score, TARGET
from theme_sources import ROOT, sha, load_provider, provider_inventory

HERE = Path(__file__).resolve().parent
PREP = HERE / 'prepared_v2'
DATA = ROOT / 'research/first_board_top3_40_20261030/results'
S1 = ROOT / 'research/first_second_stage_20261020_run1'
STATES = ROOT / 'research/gemini_codex_joint_theme_feedback_design_20261021T074448Z/CAUSAL_PRICE_STATES_WITHOUT_TARGETS.joblib'
ARMS = {'A_DYNAMIC_THEME': THEME, 'B_THEME_PLUS_PRIOR_ANCHOR': FIELDS}


def read(path, **kwargs):
    return pd.read_csv(path, dtype={'code': str}, float_precision='round_trip', **kwargs)


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def clock(): return datetime.now(timezone.utc).isoformat()


def maturity_window(pool, labels, sessions, first_prediction_day):
    position = {day: i for i, day in enumerate(sessions)}
    cutoff_position = position[first_prediction_day] - 1
    cutoff = sessions[cutoff_position]
    # Eligibility of a signal date is calendar-only, never chosen by its outcomes.
    dates = sorted(d for d in pool.signal_date.unique() if position[d] + 1 <= cutoff_position)[-252:]
    train_pool = pool[pool.signal_date.isin(dates)].copy()
    known = labels[labels.signal_date.isin(dates) & labels.label_end.notna()
                   & labels.label_end.le(cutoff) & labels[TARGET].isin([0, 1])].copy()
    return train_pool, known, cutoff, dates


def prepare():
    out = PREP; out.mkdir(exist_ok=False)
    pool_path = DATA / 'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv'
    x = read(pool_path)
    if len(x) != 7254 or x.signal_date.nunique() != 286 or x.duplicated(KEYS).any():
        raise ValueError('original evaluation pool changed')
    provider, bindings = load_provider()
    state = joblib.load(STATES)
    features = build(x[KEYS], provider, state['states'], state['sessions'])
    features.to_csv(out / 'FULL_POOL_THEME_FEATURES_NO_LABELS.csv', index=False, encoding='utf-8-sig')
    exported = provider.copy(); exported['themes'] = exported.themes.map(lambda v: json.dumps(v, ensure_ascii=False))
    exported.to_csv(out / 'PROVIDER_REPORTED_THEMES_NO_LABELS.csv', index=False, encoding='utf-8-sig')
    monthly = []
    for month, group in x.groupby(x.signal_date.str[:7]):
        prediction_path = S1 / f'{month}_S1_PREDICTIONS_BEFORE_METRICS.csv'
        metadata_path = S1 / 'models' / f'{month}_S1_STAGE_CYCLE_GENE_full_BEFORE_FIT.json'
        members_path = S1 / 'models' / f'{month}_S1_STAGE_CYCLE_GENE_full_members.csv'
        q = read(prediction_path, usecols=KEYS+['model_score'])
        joined = group.merge(q, on=KEYS, how='left', validate='one_to_one')
        if not joined.s1_score.equals(joined.model_score): raise ValueError('S1 monthly score mismatch')
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        members = read(members_path, usecols=['signal_date','label_end'])
        first = group.signal_date.min(); i = state['sessions'].index(first)
        cutoff = state['sessions'][i-1]
        if not (members.label_end.notna().all() and members.label_end.le(cutoff).all()
                and members.signal_date.lt(first).all() and members.label_end.max() == meta['max_label_end']):
            raise ValueError('S1 original monthly training maturity failed')
        monthly.append({'month': month, 'matched_rows': len(group), 'score_max_error': 0,
            'training_label_end_max': meta['max_label_end'], 'prediction_cutoff': cutoff,
            'original_model_actual_fit_at': meta['started_at'], 'original_monthly_historical_rebuild_not_live': True})
        for p in [prediction_path, metadata_path, members_path]: bindings[str(p.relative_to(ROOT))] = sha(p)
    for p in [pool_path, DATA/'OUTCOMES_DIAGNOSTIC_ONLY.csv', S1/'S1_ALL_PREDICTIONS.csv', STATES]:
        bindings[str(p.relative_to(ROOT))] = sha(p)
    save_json(out/'INPUT_BINDINGS.json', bindings)
    save_json(out/'S1_MONTHLY_BINDING_CHECK.json', monthly)
    save_json(out/'SESSIONS.json', state['sessions'])
    save_json(out/'PREPARATION.json', {'created_at': clock(), 'candidate_rows': len(x),
        'candidate_dates': int(x.signal_date.nunique()), 'provider': provider_inventory(provider),
        'candidate_context_counts': features.context_status.value_counts().to_dict(),
        'feature_observed_counts': features[FIELDS].notna().sum().to_dict(),
        'source_qualification': 'POSTHOC_PROVIDER_THEME_EXPLORATION_ONLY', 'market_fits': 0,
        'independent_blind_test': False})
    print((out/'PREPARATION.json').read_text(encoding='utf-8'), flush=True)


def verify_freeze():
    frozen_path = HERE/'PROTOCOL.json'
    protocol = json.loads(frozen_path.read_text(encoding='utf-8'))
    for name in ['CODEX_FREEZE_ACK.json', 'TERRA_FREEZE_ACK.json']:
        ack = json.loads((HERE/name).read_text(encoding='utf-8'))
        if ack['status'] != 'AGREED_EXPLORATORY_PROTOCOL' or ack['protocol_sha256'] != sha(frozen_path):
            raise ValueError(f'actual peer agreement missing or changed: {name}')
    for path, digest in protocol['bindings'].items():
        if sha(ROOT/path) != digest: raise ValueError(f'frozen binding changed: {path}')
    if protocol['arms'] != ARMS: raise ValueError('arm definitions differ')
    return protocol


def evaluator():
    path = ROOT/'research/first_board_math_20261030/review_gemini_cascade.py'
    spec = importlib.util.spec_from_file_location('dynamic_ranking_evaluator', path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj.evaluate


def slots(keys):
    keys = keys.copy(); keys['slot'] = keys.groupby('signal_date').cumcount()+1
    return keys


def bootstrap(baseline, challenger, dates):
    def day_counts(p):
        return p.groupby('signal_date')[TARGET].agg(hits=lambda a: a.eq(1).sum(), known='count').reindex(dates).fillna(0).to_numpy(float)
    b, c = day_counts(baseline), day_counts(challenger)
    rng = np.random.default_rng(20261030); values = []
    n = len(dates); block = min(25, n)
    for _ in range(5000):
        starts = rng.integers(0, n-block+1, size=(n+block-1)//block)
        indices = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
        bh, bk = b[indices].sum(axis=0); ch, ck = c[indices].sum(axis=0)
        values.append([(ch/ck-bh/bk)*100, (ch-bh)/(3*n)*100])
    intervals = np.quantile(values, [.025,.975], axis=0)
    return {'known_precision_difference_pp_95': intervals[:,0].tolist(),
            'all_slots_confirmed_difference_pp_95': intervals[:,1].tolist(),
            'block_sessions': block, 'samples': 5000, 'seed': 20261030,
            'descriptive_not_selection_adjusted_not_forward_validation': True}


def run():
    protocol = verify_freeze()
    out = HERE/'results'; out.mkdir(exist_ok=False)
    save_json(out/'RUN_STARTED_BEFORE_FITS.json', {'created_at': clock(), 'protocol_sha256': sha(HERE/'PROTOCOL.json')})
    x = read(DATA/'FULL_POOL_EXISTING_EVIDENCE_NO_LABELS.csv')
    features = read(PREP/'FULL_POOL_THEME_FEATURES_NO_LABELS.csv')
    panel = x.merge(features[KEYS+FIELDS], on=KEYS, validate='one_to_one')
    labels = read(DATA/'OUTCOMES_DIAGNOSTIC_ONLY.csv')
    maturity = read(S1/'S1_ALL_PREDICTIONS.csv', usecols=KEYS+['label_end', TARGET])
    labels = labels.merge(maturity.rename(columns={TARGET:'saved_S1_target'}),on=KEYS,validate='one_to_one')
    if not (labels[TARGET].eq(labels.saved_S1_target) | (labels[TARGET].isna() & labels.saved_S1_target.isna())).all():
        raise ValueError('outcome source disagreement')
    sessions = json.loads((PREP/'SESSIONS.json').read_text())
    dates = sorted(panel.signal_date.unique())
    selections = {'S1_BASELINE': slots(choose(panel)), 'STRICT_NEW_INFORMATION_GATE_S1_FALLBACK': slots(choose(panel))}
    models, predictions = [], []
    for arm, fields in ARMS.items():
        arm_picks = []
        for month, future in panel.groupby(panel.signal_date.str[:7]):
            train_pool, train_labels, cutoff, train_dates = maturity_window(panel, labels, sessions, future.signal_date.min())
            if len(train_dates) < 60:
                model = {'status': 'FALLBACK', 'reason': 'FEWER_THAN_60_MATURE_SIGNAL_DAYS', 'fields': fields}
            else:
                model = fit(train_pool, train_labels, fields)
            model.update(arm=arm, month=month, cutoff=cutoff, training_dates=train_dates,
                training_candidates=len(train_pool), mature_known_training_labels=len(train_labels),
                max_training_label_end=(train_labels.label_end.max() if len(train_labels) else None), created_at=clock())
            save_json(out/f'{arm}_{month}_MODEL.json', model); models.append(model)
            # Member files retain all training candidates, including unknown labels.
            train_pool[KEYS].merge(train_labels[KEYS+[TARGET,'label_end']],on=KEYS,how='left',validate='one_to_one').to_csv(
                out/f'{arm}_{month}_TRAIN_MEMBERS.csv',index=False,encoding='utf-8-sig')
            for day, daily in future.groupby('signal_date'):
                daily = daily.copy()
                daily['research_score'], daily['residual'], status = score(daily, model)
                daily['arm'] = arm; daily['score_status'] = status
                predictions.append(daily[KEYS+['arm','s1_score','research_score','residual','score_status']])
                arm_picks.append(slots(choose(daily, 'research_score')))
        selections[arm] = pd.concat(arm_picks,ignore_index=True)
    pd.concat(predictions,ignore_index=True).to_csv(out/'ALL_CANDIDATE_SCORES_NO_LABELS.csv',index=False,encoding='utf-8-sig')
    for arm, picks in selections.items():
        picks.to_csv(out/f'{arm}_PICKS_BEFORE_OUTCOME_JOIN.csv',index=False,encoding='utf-8-sig')
    save_json(out/'SELECTIONS_COMPLETE_BEFORE_EVALUATION.json', {'created_at':clock(), 'rows_per_arm':{k:len(v) for k,v in selections.items()}})
    pool = x.merge(labels.drop(columns='saved_S1_target'),on=KEYS,validate='one_to_one')
    evaluate = evaluator(); results = {}; detailed = {}; monthly = []; ledgers = []
    for arm, picks in selections.items():
        metric, detail = evaluate(pool,picks,dates); results[arm] = metric; detailed[arm] = detail
        detail.to_csv(out/f'{arm}_SELECTIONS_WITH_OUTCOMES.csv',index=False,encoding='utf-8-sig')
        for month in sorted(set(d[:7] for d in dates)):
            ds = [d for d in dates if d.startswith(month)]
            report,_ = evaluate(pool[pool.signal_date.isin(ds)],picks[picks.signal_date.isin(ds)],ds)
            monthly.append({'arm':arm,'month':month,**{k:report[k] for k in ['dates','slots','hits','known','unknown','known_precision','reach4_hits','reach4_known']}})
        joined = detail.merge(pool[KEYS+['one_price']],on=KEYS,validate='one_to_one')
        subset = joined[joined.one_price.eq(0)]
        results[arm]['signal_day_non_one_price'] = {'selected':len(subset),'known':int(subset[TARGET].notna().sum()),
            'hits':int(subset[TARGET].eq(1).sum()),'unknown':int(subset[TARGET].isna().sum())}
    base = detailed['S1_BASELINE']; swaps = {}; uncertainty = {}
    for arm in ARMS:
        changed_dates=[]; rows=[]; alternative=detailed[arm]
        for day in dates:
            b=base[base.signal_date.eq(day)]; c=alternative[alternative.signal_date.eq(day)]
            removed=b[~b.code.isin(c.code)].sort_values('slot'); added=c[~c.code.isin(b.code)].sort_values('slot')
            if not len(added): continue
            changed_dates.append(day)
            for (_,old),(_,new) in zip(removed.iterrows(),added.iterrows()):
                rows.append({'arm':arm,'signal_date':day,'displaced_code':old.code,'promoted_code':new.code,
                    'displaced_promotion':old[TARGET],'promoted_promotion':new[TARGET],
                    'displaced_reach4':old['reach4_LABEL_ONLY'],'promoted_reach4':new['reach4_LABEL_ONLY'],
                    'pairing_note':'rank-ordered bookkeeping only, not a causal replacement explanation'})
        ledger=pd.DataFrame(rows); ledgers.append(ledger)
        counts={}
        for direction in ['promoted','displaced']:
            v=ledger[f'{direction}_promotion'] if len(ledger) else pd.Series(dtype=float)
            counts[direction]={'hits':int(v.eq(1).sum()),'failures':int(v.eq(0).sum()),'unknown':int(v.isna().sum())}
        swaps[arm]={'changed_membership_dates':len(changed_dates),'swaps':len(ledger),**counts}
        if changed_dates:
            changed_pool=pool[pool.signal_date.isin(changed_dates)]
            swaps[arm]['changed_dates_baseline']=evaluate(changed_pool,selections['S1_BASELINE'][selections['S1_BASELINE'].signal_date.isin(changed_dates)],changed_dates)[0]
            swaps[arm]['changed_dates_challenger']=evaluate(changed_pool,selections[arm][selections[arm].signal_date.isin(changed_dates)],changed_dates)[0]
        uncertainty[arm]=bootstrap(base,alternative,dates)
    pd.DataFrame(monthly).to_csv(out/'MONTHLY_COMPARISON.csv',index=False,encoding='utf-8-sig')
    pd.concat(ledgers,ignore_index=True).to_csv(out/'BIDIRECTIONAL_SWAP_LEDGER.csv',index=False,encoding='utf-8-sig')
    protected=json.loads((ROOT/'research/first_board_dragon_upgrade_20261029/DELETION_RECEIPT.json').read_text(encoding='utf-8'))['protected_sha256']
    bad=[p for p,digest in protected.items() if not Path(p).is_file() or sha(p)!=digest]
    if bad: raise ValueError('protected model or evidence changed')
    save_json(out/'REPORT.json', {'created_at':clock(),'results':results,'swaps':swaps,'paired_block_intervals':uncertainty,
        'fits':sum(m['status']=='FIT' for m in models),'fallback_models':sum(m['status']!='FIT' for m in models),
        'source_status':'RETROSPECTIVE_PROVIDER_THEMES_NOT_PIT_VERIFIED','strict_new_information_rows':0,
        'original_s1_and_height_weights_changed':False,'production_deployment':False,
        'independent_blind_test':False,'validated_predictive_improvement':False,
        'protected_checked':len(protected),'protected_mismatches':bad,
        'protocol_sha256':sha(HERE/'PROTOCOL.json')})
    print(json.dumps({arm:{k:v[k] for k in ['hits','known','unknown','known_precision','reach4_hits']} for arm,v in results.items()}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run'])
    action=parser.parse_args().action
    prepare() if action=='prepare' else run()
