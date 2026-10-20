"""Verified, isolated source-qualification attachment; never edits old forecasts.

No fit, request, threshold tuning, or pairwise-to-direction conversion occurs.
Historical recomputation with late obtained sources is not an original forecast.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import short_term_thinking_dataset_builder as b

VERSION = 'isolated-raw-qualification-repair-v2-explicit-metric-applicability'
ABSOLUTE_B_HEADS = frozenset({'CORRECTED_CAUSAL', 'CORRECTED_CAUSAL_COST1P5'})
RAW_FIELDS = ('open', 'close', 'high', 'low', 'volume')
SHAPE_FIELDS = ('open', 'high', 'low', 'close', 'pre_close', 'volume')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _bindings(mapping):
    for path, expected in mapping.items():
        if b.sha(path) != expected:
            raise ValueError('source hash mismatch: ' + str(path))


def _finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _known(value):
    if value is True or isinstance(value, np.bool_) and value:
        return True
    if isinstance(value, str) and value.lower() == 'true':
        return True
    return False


def validate_reference_date(signal, day, sessions):
    if signal not in sessions:
        raise ValueError('signal date outside bound official calendar')
    position=sessions.index(signal)
    previous=sessions[position-1] if position else None
    if day not in {signal,previous} or day>signal:
        raise ValueError('nonadjacent or future signal-day source')
    return previous


def classify_current(current, previous):
    """Same eligibility order as the bound daily(); no relative-volume output."""
    if not current:
        return None, 'UNKNOWN_MISSING_DAILY'
    suspended=current.get('is_suspended')
    if (isinstance(suspended,(bool,np.bool_)) and bool(suspended)) or str(current.get('trade_status', '')).upper() == 'SUSPENDED':
        return None, 'UNKNOWN_SUSPENDED_EXPLICIT'
    pc = _finite(previous.get('close')) if previous else None
    shape = b.price_state(current)
    if pc is None or pc <= 0 or shape['one_word'] is None:
        return None, 'UNKNOWN_ADJACENT_REFERENCE_OR_SHAPE'
    explicit = _finite(current.get('pre_close'))
    if explicit is not None and abs(explicit-pc) > .011:
        return None, 'UNKNOWN_REFERENCE_CONFLICT'
    volume = _finite(current.get('volume'))
    if volume is None or volume <= 0:
        return None, 'UNKNOWN_NONTRADING_OR_VOLUME'
    upper = b.cap_cents(pc)
    if b.cents(current['high']) > upper:
        return None, 'UNKNOWN_10PCT_SCOPE_CONFLICT'
    if b.cents(current['close']) != upper:
        return None, 'OBSERVED_NONCLOSING_10PCT_PRICE_PATH'
    return shape['one_word'], 'VERIFIED_10PCT_CLOSING_PRICE_PATH_PROXY'


def apply_missing_fields(original, proposals, source_rows):
    """Field-level validation against already verified raw rows, missing only."""
    result = {key: dict(value) for key, value in original.items()}
    for proposal in proposals:
        key = (proposal['date'], proposal['code'])
        field = proposal['field']
        if field not in RAW_FIELDS:
            raise ValueError('unsupported patch field')
        existing = _finite(result.get(key, {}).get(field))
        value = _finite(proposal['proposed_value'])
        if value is None or value <= 0:
            raise ValueError('nonpositive or missing proposed raw value')
        # Repeated references to one field are allowed only if values agree.
        original_value = _finite(original.get(key, {}).get(field))
        if original_value is not None:
            raise ValueError('candidate would overwrite nonmissing original field')
        if existing is not None and existing != value:
            raise ValueError('conflicting repeated candidate value')
        source_paths = proposal.get('source_paths') or []
        if not source_paths:
            raise ValueError('proposal has no source reference')
        for path in source_paths:
            source_key = (proposal['signal_date'], proposal['date'], proposal['code'], str(Path(path).resolve()))
            raw = source_rows.get(source_key)
            if raw is None or _finite(raw.get(field)) != value:
                raise ValueError('unreferenced source or candidate/raw mismatch')
        result.setdefault(key, {})[field] = value
    return result


def load_qualified_rows(audit_dir):
    """Recheck exact raw rows and official references before applying candidates."""
    directory = Path(audit_dir).resolve()
    receipt_path = directory/'source_audit_delivery_receipt.json'
    receipt = _read(receipt_path)
    _bindings(receipt['all_source_audit_output_bindings'])
    summary = _read(directory/'raw_repair_candidate_summary.json')
    exact = _read(directory/'exact_daily_qualification_summary.json')
    _bindings(summary['source_bindings'])
    _bindings(exact['source_bindings'])
    all_bindings = summary['source_bindings'] | exact['source_bindings']
    def one_named(name):
        matches = [Path(p) for p in all_bindings if Path(p).name == name]
        if len(matches) != 1:
            raise ValueError('unique bound source required: '+name)
        return matches[0]
    artifact = joblib.load(one_named('review_scope_context.joblib'))
    engine_path = str(Path(artifact['engine_path']).resolve())
    if engine_path not in all_bindings:
        raise ValueError('engine has no verified hash binding')
    engine = joblib.load(engine_path)
    price_context = _read(one_named('complete9245_provisional_price_context_DELIVERY.json'))
    source_scope = {(x['date'], x['code']): x.get('sources', []) for x in price_context['rows']}
    sessions = list(engine['sessions'])
    indices = {day:i for i,day in enumerate(sessions)}
    unknown = artifact['rows'][artifact['rows'].current_price_shape.isna()].copy()
    if len(unknown) != 244:
        raise ValueError('unexpected audited unknown denominator')
    keys = set(zip(unknown.date, unknown.code))
    needed = keys | {(sessions[indices[d]-1], c) for d,c in keys if indices[d]}
    columns = [x for x in SHAPE_FIELDS + ('is_suspended', 'trade_status') if x in engine['bars']]
    subset = pd.DataFrame(sorted(needed), columns=['date','code']).merge(engine['bars'][['date','code']+columns], on=['date','code'], how='inner', validate='one_to_one')
    originals = {(r[0], r[1]):dict(zip(columns,r[2:])) for r in subset.itertuples(index=False,name=None)}
    pools = engine['attempts'][engine['attempts'].pool_assignment.eq('CLOSING_LIMIT_UP')]
    fallback = {(r.signal_date,r.code):dict(close=r.close) for r in pools[['signal_date','code','close']].itertuples(index=False)}
    evidence = _read(directory/'raw_same_and_adjacent_evidence.json')['rows']
    proposals = _read(directory/'missing_field_candidate_patches_NO_WRITEBACK.json')['rows']
    raw_cache = {}; verified_sources = {}; per_signal = defaultdict(list)
    for row in evidence:
        signal, day, code = row['signal_date'], row['date'], row['code']
        if (signal,code) not in keys or signal not in indices:
            raise ValueError('unbound original signal/code')
        previous = validate_reference_date(signal,day,sessions)
        path = str(Path(row['raw_path']).resolve())
        candidates = [x for x in source_scope.get((signal,code), []) if x.get('raw_path') and str(Path(x['raw_path']).resolve()) == path]
        if len(candidates) != 1:
            raise ValueError('raw response outside original signal source scope')
        source = candidates[0]
        params = source.get('request', {}).get('param','').split(',')
        symbol = ('sh' if code.startswith('6') else 'sz')+code
        if len(params)!=6 or params[0]!=symbol or params[1]!='day' or params[-1]!='' or not params[2]<=day<=params[3]:
            raise ValueError('raw request symbol/unadjusted/date scope mismatch')
        if path not in all_bindings or row['raw_sha256']!=all_bindings[path] or source['raw_sha256']!=all_bindings[path]:
            raise ValueError('raw hash not bound by verified audit')
        if row.get('available_at_actual') != source.get('available_at'):
            raise ValueError('actual receipt clock changed')
        if path not in raw_cache:
            block = (_read(path).get('data') or {}).get(symbol) or {}
            raw_days = block.get('day')
            if not isinstance(raw_days,list):
                raise ValueError('unadjusted raw day array required')
            dates = [str(x[0]) for x in raw_days]
            if dates!=sorted(set(dates)):
                raise ValueError('duplicate or unordered raw daily dates')
            raw_cache[path] = {str(x[0]):(i,x) for i,x in enumerate(raw_days)}
        if day not in raw_cache[path]:
            raise ValueError('raw exact day missing')
        index, cells = raw_cache[path][day]
        if len(cells)<6 or row['raw_pointer']!=f'data.{symbol}.day[{index}]':
            raise ValueError('raw field pointer changed')
        raw = {k:float(cells[i]) for k,i in [('open',1),('close',2),('high',3),('low',4),('volume',5)]}
        if b.price_state(raw)['one_word'] is None or _finite(raw['volume']) is None or raw['volume']<0:
            raise ValueError('invalid raw OHLC or volume')
        if any(_finite(row[k])!=raw[k] for k in RAW_FIELDS):
            raise ValueError('raw row and audited values differ')
        verified_sources[(signal,day,code,path)] = raw
        per_signal[(signal,code)].append(row)
    proposals_by_signal = defaultdict(list)
    for proposal in proposals:
        key = (proposal['signal_date'],proposal['code'])
        if key not in keys:
            raise ValueError('patch outside audited unknown scope')
        previous=validate_reference_date(key[0],proposal['date'],sessions)
        proposals_by_signal[key].append(proposal)
    qualified=[]
    for row in unknown.itertuples(index=False):
        key=(row.date,row.code);prev=sessions[indices[row.date]-1] if indices[row.date] else None
        sources=per_signal[key];conflicts=[]
        for day in [row.date,prev]:
            items=[x for x in sources if x['date']==day]
            for field in RAW_FIELDS:
                values={float(x[field]) for x in items}
                if len(values)>1:
                    conflicts.append('RAW_SOURCE_CONFLICT:'+day+':'+field)
                original=_finite(originals.get((day,row.code),{}).get(field))
                if field!='volume' and original is not None and values and values!={original}:
                    conflicts.append('EXISTING_PRICE_CONFLICT:'+day+':'+field)
        patched=apply_missing_fields(originals,proposals_by_signal[key],verified_sources)
        current=patched.get(key);previous=patched.get((prev,row.code)) or fallback.get((prev,row.code))
        shape,status=classify_current(current,previous)
        if conflicts:
            shape=None;status='UNKNOWN_RAW_OR_EXISTING_PRICE_CONFLICT'
        clocks=[x.get('available_at_actual') for x in sources]
        qualified.append(dict(date=row.date,code=row.code,verified_current_price_shape=shape,
            qualification_status=status,qualification_source_paths=sorted({x['raw_path'] for x in sources}),
            qualification_field_count=len(proposals_by_signal[key]),qualification_source_clock_values=clocks,
            qualification_source='VERIFIED_ORIGINAL_RAW_SAME_AND_ADJACENT_SESSION_LATE_RESEARCH' if shape is not None else 'UNRESOLVED_OR_OBSERVED_NONCLOSING_PRICE_PATH',
            qualification_PIT_eligible=False,qualification_late_data=bool(sources),qualification_receipt_time_unknown=any(x is None for x in clocks),
            qualification_volume_unit='PROVIDER_NATIVE_POSITIVE_VOLUME_ONLY_NO_RELATIVE_VOLUME_UPDATE',
            qualification_conflicts=conflicts,relative_volume_features_changed=False,original_writeback=False))
    qualified=pd.DataFrame(qualified)
    if qualified.verified_current_price_shape.notna().sum()!=47:
        raise ValueError('replay no longer reproduces all47 validated qualifications')
    metadata=dict(version=VERSION,source_bindings=all_bindings|{str(receipt_path):b.sha(receipt_path)},all_audited_unknowns=244,recovered_shapes=47,remaining_unknown_shapes=197,
        official_calendar_in_bound_engine=True,new_fit_count=0,original_forecast_success_claim=False)
    return qualified,metadata


def attach_qualifications(predictions, qualifications):
    """Preserve every original column; attach verified shape and research calls."""
    required={'date','code','arm','current_price_shape','call'}
    if not required<=set(predictions):
        raise ValueError('prediction identity, original shape and call required')
    if predictions.duplicated(['date','code','arm']).any():
        raise ValueError('duplicate dated prediction identity')
    reserved={'verified_current_price_shape','recovered_B_call','source_repair_applied','qualification_status'}
    if reserved & set(predictions):
        raise ValueError('refuse to overwrite an existing qualification attachment')
    out=predictions.copy()
    if qualifications.duplicated(['date','code']).any():
        raise ValueError('duplicate qualification identity')
    lookup=qualifications.set_index(['date','code']).to_dict('index')
    columns=[x for x in qualifications.columns if x not in ['date','code']]
    attachments=[]
    for row in predictions.to_dict('records'):
        original_shape=_finite(row['current_price_shape']);entry=lookup.get((row['date'],row['code']))
        if original_shape is not None and original_shape not in [0,1]:
            raise ValueError('invalid original price shape')
        if entry and original_shape is not None:
            raise ValueError('qualification attempted to overwrite nonmissing original shape')
        values={name:(entry.get(name) if entry else None) for name in columns}
        if not entry:
            values.update(verified_current_price_shape=original_shape,
                qualification_status='ORIGINAL_SHAPE_RETAINED' if original_shape is not None else 'UNKNOWN_NO_VERIFIED_SOURCE_REFERENCE',
                qualification_source='UNCHANGED_ORIGINAL_PREDICTION',qualification_PIT_eligible=False,
                qualification_late_data=False,qualification_receipt_time_unknown=False)
        repaired=entry is not None and _finite(entry.get('verified_current_price_shape')) is not None
        recovered=row['call'];reason='ORIGINAL_CALL_RETAINED'
        if repaired and values['verified_current_price_shape']==0 and row['arm'] in ABSOLUTE_B_HEADS and row['call']=='UNKNOWN':
            raw=_finite(row.get('b_score'))
            if raw is not None and 0<=raw<=1:
                recovered='PROMOTE' if raw>.5 else 'FAIL'
                reason='SAME_SIGNAL_EXISTING_ABSOLUTE_B_SCORE_STRICT_GT_0P5_RETROSPECTIVE'
            else:
                reason='UNKNOWN_NO_VALID_EXISTING_ABSOLUTE_B_SCORE'
        elif repaired and row['arm'] not in ABSOLUTE_B_HEADS:
            reason='NO_DIRECTION_CONVERSION_FOR_PAIRWISE_OR_REFERENCE_HEAD'
        values.update(source_repair_applied=bool(repaired),recovered_B_call=recovered,recovered_B_call_basis=reason,
            qualification_is_retrospective_recomputation=True)
        attachments.append(values)
    attached=pd.DataFrame(attachments,index=out.index)
    for column in attached:
        if column in out:
            raise ValueError('refuse to overwrite original column: '+column)
        out[column]=attached[column]
    pd.testing.assert_frame_equal(out[predictions.columns],predictions)
    return out


def _metrics(frame,call_field):
    valid=frame.target.notna() & frame.label_valid.map(_known)
    known=frame[valid];positive=known.target.eq(1);p=known[call_field].eq('PROMOTE');n=known[call_field].eq('FAIL')
    tp=int((positive&p).sum());fp=int((~positive&p).sum());positives=int(positive.sum());missed=positives-tp
    return dict(candidates=len(frame),known=len(known),unknown_outcomes=len(frame)-len(known),positives=positives,
        tp=tp,fp=fp,Recall=tp/positives if positives else None,Precision=tp/(tp+fp) if tp+fp else None,
        F1=2*tp/(2*tp+fp+missed) if 2*tp+fp+missed else None,
        unknown_calls=int(frame[call_field].eq('UNKNOWN').sum()),known_abstentions=int((~p&~n).sum()),
        decision_coverage=float((p|n).mean()) if len(known) else None)


def evaluate_attachment(frame):
    if not {'target','label_valid','daily_one_word'}<=set(frame):
        raise ValueError('targets and frozen nonword identity required for evaluation')
    identity='frozen_original_nonword_identity' if 'frozen_original_nonword_identity' in frame else 'daily_one_word'
    periods={'2025H2':('2025-07-01','2025-12-31'),'2026JanAug':('2026-01-01','2026-08-31'),
        '2026Sep':('2026-09-01','2026-09-30'),'2026Oct':('2026-10-01','2026-10-31'),'ALL_BOUND_DATES':(frame.date.min(),frame.date.max())}
    for month in sorted(frame.date.str[:7].unique()):
        periods['MONTH_'+month]=(month+'-01',month+'-31')
    records=[]
    for period,(first,last) in periods.items():
        group=frame[frame.date.ge(first)&frame.date.le(last)&frame[identity].eq(0)]
        for arm,rows in group.groupby('arm',sort=True):
            for mode,field in [('ORIGINAL_FROZEN_CALL','call'),('SOURCE_REPAIRED_RETROSPECTIVE','recovered_B_call')]:
                metrics=_metrics(rows,field)
                applicability='NO_ABSOLUTE_EVENT_HEAD' if str(arm).startswith('PAIRWISE') else 'ABSOLUTE_EVENT_HEAD'
                if applicability=='NO_ABSOLUTE_EVENT_HEAD':
                    # Unknown calls describe the absence of an event head.
                    # Zero recall would falsely grade a ranking-only score.
                    for name in ['tp','fp','Recall','Precision','F1']:
                        metrics[name]=None
                records.append(dict(period=period,arm=arm,mode=mode,scope='SAME_ORIGINAL_FROZEN_NONWORD_POOL',
                    metric_applicability=applicability,**metrics))
    return pd.DataFrame(records)


def run(predictions_path,audit_dir,output_dir):
    output=Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('new immutable output directory required')
    predictions_path=Path(predictions_path).resolve()
    predictions=pd.read_csv(predictions_path,dtype={'code':str})
    qualifications,metadata=load_qualified_rows(audit_dir)
    attached=attach_qualifications(predictions,qualifications)
    metrics=evaluate_attachment(attached)
    output.mkdir(parents=True)
    for name,frame in [('all_predictions_with_isolated_qualification.csv',attached),('all244_verified_qualification_rows.csv',qualifications),
        ('all_repaired_prediction_rows.csv',attached[attached.source_repair_applied]),('same_pool_before_after_nonword_metrics.csv',metrics)]:
        safe=frame.copy()
        for column in safe:
            if safe[column].map(lambda x:isinstance(x,(list,dict))).any():
                safe[column]=safe[column].map(lambda x:json.dumps(b.clean(x),ensure_ascii=False) if isinstance(x,(list,dict)) else x)
        safe.to_csv(output/name,index=False,encoding='utf-8-sig')
    meta=dict(metadata,formed_at=b.now(),predictions_path=str(predictions_path),predictions_sha256=b.sha(predictions_path),
        implementation_path=str(Path(__file__).resolve()),implementation_sha256=b.sha(__file__),
        input_prediction_rows=len(predictions),all_original_columns_unchanged=True,
        repaired_prediction_rows=int(attached.source_repair_applied.sum()),
        changed_research_calls=int(attached.call.ne(attached.recovered_B_call).sum()),absolute_head_allowlist=sorted(ABSOLUTE_B_HEADS),
        pairwise_to_direction_conversion=False,scores_changed=False,original_files_changed=False,
        outcome_scope='Historical late-source recomputation, not original prediction success',
        output_bindings={str(path):b.sha(path) for path in output.glob('*.csv')})
    b.write_new(output/'source_qualification_repair_receipt.json',meta)
    print(json.dumps({k:meta[k] for k in ['input_prediction_rows','recovered_shapes','remaining_unknown_shapes','repaired_prediction_rows','changed_research_calls']},ensure_ascii=False))
    return attached,metrics,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--audit-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    run(args.predictions,args.audit_dir,args.output_dir)


if __name__=='__main__':
    main()
