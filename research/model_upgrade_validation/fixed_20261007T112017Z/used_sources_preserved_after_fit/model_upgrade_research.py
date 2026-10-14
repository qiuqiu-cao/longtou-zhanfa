"""Approved fixed market/relay research and source-bound shadow lifecycle CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'results/v278/september_falsification_20261002_v6'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_protocol(path):
    path = Path(path).resolve()
    p = read(path)
    for group in ('source_hashes', 'protected_sha256'):
        for filename, expected in p[group].items():
            if not Path(filename).is_file() or sha(filename) != expected:
                raise ValueError('frozen input/source changed: ' + filename)
    if (p['version'] != 'model-upgrade-fixed-1' or p['freeze_cutoff'] != '2026-09-01'
            or p['forward_sessions'] != 120 or p['preferred_arm'] != 'C1'):
        raise ValueError('unsupported fixed upgrade protocol')
    p['protocol_path'] = str(path)
    p['protocol_sha256'] = sha(path)
    implementation = path.parent/'implementation_BEFORE_FITS.json'
    if implementation.exists() and read(implementation)['protocol_sha256'] != p['protocol_sha256']:
        raise ValueError('frozen protocol changed after implementation binding')
    return p


def initialize(output):
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('new immutable research workspace required')
    original = read(OLD / 'report.json')
    sources = {}
    for name, expected in original['source_hashes'].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError('original source no longer matches V278: ' + name)
        sources[str(path)] = expected
    for path in [OLD / 'report.json', OLD / 'combined_market_input.csv',
                 OLD / 'market_predictions_LABEL_ONLY.csv', OLD / 'relay_predictions_LABEL_ONLY.csv']:
        sources[str(path)] = sha(path)
    for name in ['next_session_forecast_lab_v271.py', 'relay_promotion_holdout_v256.py',
                 'emotion_three_regime_challenger_v86.py', 'market_expectation_extensions_v272.py',
                 'reported_seal_archive_v275.py']:
        sources[str(ROOT/name)] = sha(ROOT/name)
    protected_names = ['data/v271/expectation_registry/2026-09-30_2c74dc406eef7dcf.json',
                       'hypothesis_bound_daily_runner_v298.py', 'public_outcome_sources_v293.py',
                       'public_outcome_bridge_v293.py', 'research/theme_same_height/protocol.json',
                       'research/theme_same_height/comparison_specification.json',
                       'market_direction_research.py']
    calendar = ROOT / 'data/v279/exchange_calendar_audit_20261002_v1/calendar.json'
    sources[str(calendar)] = sha(calendar)
    p = {'version': 'model-upgrade-fixed-1', 'recorded_at': now(),
         'human_authorization': 'Implement approved market then relay research, fixed shadow validation and evidence-gated module upgrade',
         'freeze_cutoff': '2026-09-01', 'maximum_training_label_date': '2026-08-31',
         'calendar_path': str(calendar), 'calendar_sha256': sha(calendar),
         'calendar_extension_policy': 'new officially verified prefix-preserving archive; never overwrite original calendar or infer future holidays',
         'folds': [{'id': '2025H2', 'start': '2025-07-01', 'end': '2025-12-31'},
                   {'id': '2026JanAug', 'start': '2026-01-01', 'end': '2026-08-31'},
                   {'id': '2026Sep', 'start': '2026-09-01', 'end': '2026-09-30'}],
         'modules': {m: {'arms': {a: a for a in ('BASE', 'C1', 'C2')},
                         'model_manifests': {a: f'models/{m}/{a}/manifest.json' for a in ('BASE', 'C1', 'C2')}}
                     for m in ('market', 'relay')},
         'recipes': {'market': {'kind': 'forest', 'window': None, 'fields': 46,
                      'n_estimators': 160, 'max_depth': 5, 'min_samples_leaf': 15,
                      'max_features': .75, 'random_state': 274, 'n_jobs': 2},
                     'relay': {'kind': 'logistic', 'window': 252, 'window_unit': 'known-label signal dates',
                      'fields': 29, 'C': .25, 'max_iter': 2500, 'random_state': 271,
                      'sample_weight': 'inverse training candidate rows per signal date'}},
         'method_definitions': {'market_C1': '(yesterday_premium<=0) AND (promotion_mid - exact previous official session <0)',
              'market_C2': 'C1 plus (yesterday_premium<=0) AND (failed_rate - exact previous official session >0)',
              'relay_C1': '(signal_amount_same_height_percentile-0.5)*log1p(same_height_rivals)',
              'relay_C2': 'C1 plus (turnover_same_height_percentile-0.5)*log1p(same_height_rivals)',
              'market_target': 'same formula next official session emotion_quality_score minus signal quality: >0.05 IMPROVE, <-0.05 WORSEN, otherwise CONTINUE; missing UNKNOWN',
              'relay_target': 'independent cent-exact unadjusted price and explicit cap reconciled with complete next official closing pool: boards current+1',
              'promotion_mid': 'unchanged original previous closing pool 2-4 boards promotion rate',
              'rank_policy': 'raw_score descending, code ascending (current V280); negative rank first remains negative',
              'missing': 'NaN; original training-only imputation; no future/unknown-to-zero conversion'},
         'forward_sessions': 120, 'min_market_pairs': 108, 'min_class_support': 20,
         'min_relay_pairs': 60, 'min_coverage': .9, 'min_improvement': .03,
         'bootstrap': {'draws': 2000, 'seed': 20261007, 'block_sessions': 5, 'tails': [.00625, .99375]},
         'preferred_arm': 'C1', 'stability_blocks': [40, 40, 40],
         'rollback': {'sessions': 60, 'primary_drop': .05, 'recall_drop': .10},
         'signal_rule': 'true current Asia/Shanghai official closed signal, target exact next officially known session; first actual eligible signal after protocol and all module models formed',
         'reproduction_score_tolerance': 1e-8, 'budget_minutes_per_batch': 30,
         'history_is_pristine_blind': False, 'historical_results_allow_upgrade': False,
         'historical_winners_change_shadow_arms': False, 'shadow_refitting_allowed': False,
         'production_entry': 'hypothesis_bound_daily_runner_v298.py freeze/tick',
         'calibrated_probability_allowed': False, 'orders_enabled': False,
         'source_hashes': sources, 'protected_sha256': {str(ROOT/n): sha(ROOT/n) for n in protected_names},
         'original_model_selection': original['selection_logs']}
    out.mkdir(parents=True)
    write_new(out / 'protocol_BEFORE_EFFECTS.json', p)
    return {'protocol': str(out / 'protocol_BEFORE_EFFECTS.json'), 'status': 'PREDECLARED_BEFORE_REPRODUCTION_AND_CHALLENGER_EFFECTS'}


def diagnosis(protocol_path):
    p = verify_protocol(protocol_path)
    out = Path(protocol_path).resolve().parent / 'diagnosis'
    if out.exists():
        raise FileExistsError('diagnosis already recorded')
    import numpy as np
    import pandas as pd
    market = pd.read_csv(OLD / 'market_predictions_LABEL_ONLY.csv')
    relay = pd.read_csv(OLD / 'relay_predictions_LABEL_ONLY.csv', dtype={'code': str})
    market['diagnostic_error'] = np.where(market.label_valid & market.target.notna(), market.predicted.ne(market.target), None)
    relay['diagnostic_error'] = np.where(relay.label_valid & relay.target.notna(), relay.predicted.ne(relay.target), None)
    out.mkdir()
    market.to_csv(out / 'original_market_all_dates.csv', index=False)
    relay.to_csv(out / 'original_relay_all_candidates.csv', index=False)
    report = {'recorded_at': now(), 'protocol_sha256': p['protocol_sha256'],
              'status': 'SAVED_PREDICTION_DIAGNOSIS_ONLY', 'market_observed_dates': len(market),
              'market_known_labels': int((market.label_valid & market.target.notna()).sum()),
              'relay_rows': len(relay), 'relay_dates': int(relay.date.nunique()),
              'features_and_challengers_changed_by_diagnosis': False,
              'new_forward_samples': 0, 'limits': 'Known historical errors; no new-model scoring and no target replacement'}
    write_new(out / 'report.json', report)
    return report


def saved_error_analysis(protocol_path):
    """Describe the original stored errors; never alter features or formulas."""
    import pandas as pd
    from model_upgrade_inputs import clean
    from model_upgrade_metrics import classification_metrics,binary_metrics
    p=verify_protocol(protocol_path)
    market=pd.read_csv(OLD/'market_predictions_LABEL_ONLY.csv')
    relay=pd.read_csv(OLD/'relay_predictions_LABEL_ONLY.csv',dtype={'code':str})
    known=relay.label_valid & relay.target.notna()
    by_board={}
    for height,rows in relay.groupby('boards'):
        by_board[str(int(height))]=binary_metrics(rows.target.where(rows.label_valid).tolist(),rows.predicted.tolist())
    value=clean({'status':'ORIGINAL_SAVED_ERRORS_NO_RESELECTION','recorded_at':now(),
        'protocol_sha256':p['protocol_sha256'],
        'market':classification_metrics(market.target.where(market.label_valid).tolist(),market.predicted.tolist(),
                                      ['IMPROVE','CONTINUE','WORSEN']),
        'market_errors':market[market.label_valid & market.target.notna() & market.predicted.ne(market.target)]
            [['date','predicted','target']].to_dict('records'),
        'relay':binary_metrics(relay.target.where(relay.label_valid).tolist(),relay.predicted.tolist()),
        'relay_by_board':by_board,
        'relay_errors':relay[known & relay.predicted.ne(relay.target)]
            [['date','code','boards','predicted','target','same_height_rivals','signal_amount_same_height_percentile',
              'turnover_same_height_percentile','actual_one_word_daily']].to_dict('records'),
        'interpretation':['Market September has only 8 valid next-day labels and no CONTINUE truth; 75% cannot establish three-class stability.',
                          'Relay misses 36 of 55 positive labels; raising overall accuracy by predicting failures is insufficient.',
                          'Errors describe stored cohorts only; interaction formulas, folds and future arms remain fixed.'],
        'new_forward_samples':0})
    destination=Path(protocol_path).resolve().parent/'diagnosis/saved_error_analysis.json'
    write_new(destination,value)
    return {'status':value['status'],'analysis_path':str(destination)}


def implementation_receipt(out,prepared,protocol):
    paths = [ROOT / n for n in ['model_upgrade_research.py', 'model_upgrade_history.py',
             'model_upgrade_inputs.py', 'model_upgrade_shadow.py', 'model_upgrade_metrics.py',
             'model_upgrade_apply.py']]
    path = Path(out) / 'implementation_BEFORE_FITS.json'
    hashes = {str(p): sha(p) for p in paths}
    provenance_path=Path(out)/'provenance_BEFORE_FITS.json'
    if not provenance_path.exists():
        write_new(provenance_path,{'recorded_at':now(),'protocol_sha256':protocol['protocol_sha256'],
                                 'provenance':prepared['provenance']})
    else:
        saved_provenance=read(provenance_path)['provenance']
        current={k:v for k,v in prepared['provenance'].items() if k!='prepared_at'}
        saved={k:v for k,v in saved_provenance.items() if k!='prepared_at'}
        if current!=saved:
            raise ValueError('prepared provenance changed since first fit')
        prepared['provenance']['prepared_at']=saved_provenance['prepared_at']
    protocol['provenance_path']=str(provenance_path.resolve())
    protocol['provenance_sha256']=sha(provenance_path)
    protocol['input_hashes']=prepared['provenance']['source_hashes']
    protocol['dependency_hashes']=prepared['provenance']['dependency_hashes']
    if path.exists():
        saved = read(path)
        if saved['sources'] != hashes:
            raise ValueError('implementation changed since fixed real research; retain old receipt and repair explicitly')
        return saved
    value = {'recorded_at': now(), 'sources': hashes, 'protocol_sha256':protocol['protocol_sha256'],
             'provenance_sha256':sha(provenance_path),
             'source_hashes':protocol['input_hashes'],'dependency_hashes':protocol['dependency_hashes'],
             'protected_sha256':protocol['protected_sha256'],'new_forward_samples': 0}
    write_new(path, value)
    return value


def historical_records(module,predictions,prepared,fold,fit_logs,use_legacy_proxy=False):
    """Translate saved outputs on the entire fixed official diagnostic grid."""
    import numpy as np
    import pandas as pd
    sessions=prepared['provenance']['sessions']
    successor=dict(zip(sessions[:-1],sessions[1:]))
    predecessor=dict(zip(sessions[1:],sessions[:-1]))
    majority=next(r['training_majority']['class'] for r in fit_logs if r['fold']['id']==fold['id'] and r['arm']=='BASE')
    f=predictions[predictions.fold.eq(fold['id'])]
    market=prepared['market'].set_index('date')
    result=[]
    finite=lambda v: pd.notna(v) and np.isfinite(v)
    for day in [d for d in sessions if fold['start']<=d<=fold['end']]:
        group=f[f.date.eq(day)]
        record={'module':module,'signal_date':day,'target_date':successor.get(day),
                'forward':False,'arms':{},'outcomes':{},'naive':{}}
        if module=='market':
            truth=None
            for arm in ('BASE','C1','C2'):
                rows=group[group.arm.eq(arm)]
                if len(rows)>1: raise ValueError('duplicate saved market output')
                r=rows.iloc[0] if len(rows) else None
                known=r is not None and r['status']=='PREDICTED_UNCALIBRATED'
                record['arms'][arm]={'status':'KNOWN' if known else 'UNKNOWN',
                    'prediction':str(r.predicted) if known else None,
                    'raw_scores':{c:float(r.get('raw_score_'+c)) for c in ('IMPROVE','CONTINUE','WORSEN')}
                    if known else None}
                if r is not None and bool(r.label_valid) and pd.notna(r.target): truth=str(r.target)
            record['outcomes']={'market_target':truth}
            previous=predecessor.get(day)
            persistence=None
            if day in market.index and previous in market.index:
                a,b=market.loc[day,'emotion_quality_score'],market.loc[previous,'emotion_quality_score']
                if finite(a) and finite(b):
                    delta=a-b
                    persistence='IMPROVE' if delta>.05 else 'WORSEN' if delta<-.05 else 'CONTINUE'
            record['naive']={'majority':majority,'persistence':persistence}
        else:
            base=group[group.arm.eq('BASE')]
            if len(base):
                codes=sorted(base.code.astype(str).tolist())
                if len(codes)!=len(set(codes)): raise ValueError('duplicate saved original candidate')
                record['original_codes']=codes
                facts={}
                truths={}
                for arm in ('BASE','C1','C2'):
                    rows=group[group.arm.eq(arm)]
                    if sorted(rows.code.astype(str))!=codes: raise ValueError('historical challenger lost full original group')
                    candidates=[]
                    for r in rows.to_dict('records'):
                        word=r.get('one_word_proxy') if use_legacy_proxy else r.get('actual_one_word_daily')
                        word=int(word) if finite(word) and word in (0,1) else None
                        known=r['status']=='PREDICTED_UNCALIBRATED'
                        candidates.append({'code':str(r['code']),'boards':int(r['boards']),
                            'actual_one_word_daily':word,
                            'raw_score':float(r['raw_score']) if known else None,
                            'predicted_positive':bool(r['predicted']) if known else None})
                        facts[str(r['code'])]=(int(r['boards']),word)
                        truths[str(r['code'])]=bool(r['target']) if bool(r['label_valid']) and finite(r['target']) else None
                    record['arms'][arm]={'status':'KNOWN','candidates':candidates}
                record['outcomes']={'candidates':truths}
                nonword=[c for c in codes if facts[c][1]==0]
                record['naive']={'highest_nonword':sorted(nonword,key=lambda c:(-facts[c][0],c))[0]
                    if nonword and all(facts[c][1] is not None for c in codes) else None}
        result.append(record)
    return result


def history_summary(module,out,prepared):
    import pandas as pd
    from model_upgrade_inputs import clean
    from model_upgrade_metrics import summarize_records,_comparison
    directory=Path(out)/(module+'_history')
    report=read(directory/'report.json')
    if sha(directory/'predictions.csv')!=report['predictions_sha256']: raise ValueError('history output changed')
    predictions=pd.read_csv(directory/'predictions.csv',dtype={'code':str})
    summaries={}
    for fold in report['folds']:
        records=historical_records(module,predictions,prepared,fold,report['fit_logs'])
        summary=summarize_records(module,records,prepared['provenance']['sessions'])
        summary['paired_comparisons']={a:_comparison(module,records,a) for a in ('C1','C2')}
        check_rows=predictions[predictions.fold.eq(fold['id'])]
        independent=[]
        for arm in ('BASE','C1','C2'):
            rows=check_rows[check_rows.arm.eq(arm)]
            scored=rows[rows.label_valid & rows.target.notna() & rows.predicted.notna()]
            metrics=summary['arms'][arm] if module=='market' else summary['arms'][arm]['all_candidate']
            labels=('IMPROVE','CONTINUE','WORSEN') if module=='market' else (0,1)
            matrix={str(y):{str(p):int((scored.target.eq(y)&scored.predicted.eq(p)).sum()) for p in labels} for y in labels}
            correct=int(scored.target.eq(scored.predicted).sum())
            if matrix!=metrics['confusion_matrix'] or correct!=metrics['correct'] or len(scored)!=metrics['scored_rows']:
                raise ValueError('independent confusion/scored-row reconstruction mismatch')
            if module=='relay':
                raw=rows[rows.label_valid & rows.target.notna() & rows.raw_score.notna()]
                brier=float(((raw.raw_score-raw.target)**2).mean()) if len(raw) else None
                claimed=summary['arms'][arm]['raw_scores']['raw_score_Brier']
                if (brier is None)!=(claimed is None) or brier is not None and abs(brier-claimed)>1e-12:
                    raise ValueError('independent raw-score Brier mismatch')
            independent.append({'arm':arm,'confusion_and_scored_count_match':True,
                                'candidate_raw_Brier_match':True if module=='relay' else None})
        # Independent paired reconstruction reads the saved prediction table,
        # without reusing the metric controller's per-day derivation.
        for identity in ([None,'actual_one_word_daily','one_word_proxy'] if module=='relay' else [None]):
            if module=='relay' and identity is None: continue
            values={a:{} for a in ('BASE','C1','C2')}
            for arm in values:
                arm_rows=check_rows[check_rows.arm.eq(arm)]
                if module=='market':
                    values[arm]={r.date:float(r.predicted==r.target) for r in arm_rows.itertuples()
                        if r.label_valid and pd.notna(r.target) and pd.notna(r.predicted)}
                else:
                    for day,group in arm_rows.groupby('date'):
                        if (identity not in group or group[identity].isna().any() or not group.label_valid.all()
                            or group.target.isna().any()): continue
                        nonword=group[group[identity].eq(0)]
                        if not len(nonword) or nonword.raw_score.isna().any(): continue
                        first=nonword.sort_values(['raw_score','code'],ascending=[False,True]).iloc[0]
                        values[arm][day]=float(first.target)
            for arm in ('C1','C2'):
                days=sorted(set(values['BASE'])&set(values[arm]))
                delta=sum(values[arm][d]-values['BASE'][d] for d in days)/len(days) if days else None
                comparison=(summary['paired_comparisons'][arm] if module=='market' or identity=='actual_one_word_daily'
                            else None)
                if comparison is not None:
                    if len(days)!=comparison['paired_dates'] or (delta is None)!=(comparison['primary_difference'] is None) or delta is not None and abs(delta-comparison['primary_difference'])>1e-12:
                        raise ValueError('independent same-date primary comparison mismatch')
                independent.append({'arm':arm,'identity':identity,'paired_dates':len(days),'primary_difference':delta})
        summary['independent_recomputation']=independent
        if module=='relay':
            proxy=historical_records(module,predictions,prepared,fold,report['fit_logs'],True)
            summary['legacy_one_word_proxy_diagnostic']=summarize_records(module,proxy,prepared['provenance']['sessions'])
            summary['legacy_one_word_proxy_diagnostic']['identity_definition']='Original turnover<3 proxy; NOT independent OHLC true-one-price validation'
            summary['legacy_one_word_proxy_diagnostic']['paired_comparisons']={a:_comparison(module,proxy,a) for a in ('C1','C2')}
        summaries[fold['id']]=summary
    value=clean({'status':'HISTORICAL_DIAGNOSTIC_NO_CORE_UPGRADE','module':module,'formed_at':now(),
                 'prediction_sha256':report['predictions_sha256'],'folds':summaries,'new_forward_samples':0,
                 'selection_changed':False,'future_arms':['BASE','C1','C2']})
    write_new(directory/'metrics.json',value)
    return {'metrics_path':str(directory/'metrics.json'),'status':value['status']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init'); init.add_argument('--output', required=True)
    for name in ('diagnose', 'diagnose-errors', 'reproduce', 'research', 'freeze'):
        sub = commands.add_parser(name); sub.add_argument('--protocol', required=True)
        if name in ('research', 'freeze'): sub.add_argument('--module', required=True, choices=['market', 'relay'])
    build = commands.add_parser('build-features')
    build.add_argument('--protocol', required=True); build.add_argument('--archive', required=True)
    build.add_argument('--output', required=True); build.add_argument('--calendar', required=True)
    build.add_argument('--history-archive', action='append', default=[])
    # Lifecycle commands are dispatched without weakening their native checks.
    life = commands.add_parser('shadow'); life.add_argument('arguments', nargs=argparse.REMAINDER)
    apply = commands.add_parser('apply-active'); apply.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    started=now()
    if args.command == 'init': result = initialize(args.output)
    elif args.command == 'shadow':
        from model_upgrade_shadow import main as shadow_main
        return shadow_main(args.arguments)
    elif args.command == 'apply-active':
        from model_upgrade_apply import main as apply_main
        return apply_main(args.arguments)
    else:
        protocol = verify_protocol(args.protocol)
        out = Path(args.protocol).resolve().parent
        if args.command == 'diagnose': result = diagnosis(args.protocol)
        elif args.command == 'diagnose-errors': result = saved_error_analysis(args.protocol)
        elif args.command == 'build-features':
            from model_upgrade_inputs import build_feature_archive
            result = build_feature_archive(args.protocol, args.archive, args.calendar, args.output, args.history_archive)
        else:
            from model_upgrade_history import prepare_datasets, reproduce, evaluate_module, freeze_module
            prepared = prepare_datasets()
            implementation_receipt(out,prepared,protocol)
            if args.command == 'reproduce': result = reproduce(out, protocol, prepared=prepared)
            else:
                if not (out/'reproduction/report.json').exists() or read(out/'reproduction/report.json').get('status') not in ('PASS', 'REPRODUCED', 'REPRODUCTION_PASS'):
                    raise ValueError('successful original reproduction required before challenger research')
                if args.module == 'relay' and not (out/'market_history/report.json').exists():
                    raise ValueError('market research must complete before relay research')
                if args.command == 'research':
                    result = evaluate_module(args.module, out, protocol, prepared=prepared)
                    result.pop('predictions',None)
                    result.update(history_summary(args.module,out,prepared))
                else: result = freeze_module(args.module, out/'models', protocol, prepared=prepared)
            verify_protocol(args.protocol)
        with (out/'execution_log.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'command':args.command,'module':getattr(args,'module',None),
                'started_at':started,'completed_at':now(),'new_forward_samples':0,
                'status':result.get('status','ARTIFACTS_SAVED'),'budget_minutes':30,
                'numeric_reselection':False,'platform_or_automation_changed':False},ensure_ascii=False)+'\n')
    printable={k:v for k,v in result.items() if k not in ('manifests','fit_logs','predictions')}
    if args.command=='reproduce':
        printable['modules']={m:{k:v for k,v in data.items() if k!='manifest'} for m,data in result['modules'].items()}
    if args.command=='freeze': printable['frozen_arms']=list(result['manifests'])
    print(json.dumps(printable, ensure_ascii=True, default=str))


if __name__ == '__main__':
    raise SystemExit(main())
