"""One immutable, event-causal pullback/personality relay experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[2]
sys.path.insert(0, str(PROJECT))
from current_model_diagnostics import RELAY_FIELDS, FOLDS
import model_upgrade_history as history

BUNDLE = PROJECT / 'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/engine_context_WITH_CURRENT15.joblib'
ROLL = PROJECT / 'research/model_upgrade_validation/recency_walkforward_20261013T134351Z/full_candidate_predictions.csv'
OLD_ROOT = PROJECT / 'research/model_upgrade_validation/fixed_20261007T112017Z'
BASE = OLD_ROOT / 'relay_history/predictions.csv'
NEW = json.loads((RUN / 'protocol_BEFORE_FITS.json').read_text(encoding='utf-8'))['features']
FIELDS = RELAY_FIELDS + NEW


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def write(name, obj):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def read(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Frozen input changed: ' + path)


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def build_features(signals, bars, attempts, sessions):
    """Receives ONLY date/code/boards signal fields and historical OHLCV, no y."""
    assert list(signals.columns) == ['date', 'code', 'boards', 'price']
    day_index = {d: i for i, d in enumerate(sessions)}
    useful = set(signals.code)
    z = bars[bars.code.isin(useful) & bars.date.isin(day_index)].copy()
    z['day'] = z.date.map(day_index)
    # Only a descriptive market-relative launch return, not order-flow leadership.
    all_ret = bars[['date', 'close', 'pre_close']].copy()
    all_ret['return'] = all_ret.close / all_ret.pre_close - 1
    market = all_ret[all_ret['return'].between(-.31, .31)].groupby('date')['return'].median().to_dict()
    a = attempts[attempts.pool_assignment.eq('CLOSING_LIMIT_UP') & attempts.code.isin(useful)].copy()
    capkeys = set(zip(a.signal_date, a.code))
    columns = ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'float_cap']
    lookup = {}
    source_anchor_n = 0
    for code, group in z.groupby('code', sort=False):
        p = group.set_index('day').reindex(range(len(sessions)))
        v = p[columns].to_numpy(dtype=float)
        o, h, l, c, pc, vol, cap = v.T
        observed = np.isfinite(v[:, :5]).all(axis=1) & (l > 0) & (pc > 0) & (l <= np.minimum(o, c) + .00001) & (h + .00001 >= np.maximum(o, c))
        cent = np.rint(c * 100)
        pc_cent = np.rint(pc * 100)
        limit_cent = np.floor((pc_cent * 110 + 50) / 100)
        exact_cap = observed & (cent == limit_cent) & (np.rint(h * 100) == limit_cent)
        listed = np.array([(d, code) in capkeys for d in sessions])
        anchors = exact_cap & listed
        source_anchor_n += int(anchors.sum())
        contiguous = observed.copy()
        contiguous[1:] &= observed[:-1] & (np.abs(pc[1:] - c[:-1]) < .0051)
        lookup[code] = (v, observed, contiguous, exact_cap, anchors)
    output = []
    for row in signals.itertuples(index=False):
        d, code, boards, price = row
        t = day_index[d]
        start = t - int(boards) + 1
        out = dict(date=d, code=code, **{k: np.nan for k in NEW})
        out.update(launch_date=sessions[start] if start >= 0 else None,
                   anchor_date=None, feature_latest_date=d, history_latest_date=None,
                   pullback_status='UNOBSERVED', current_run_verified=False)
        if start < 0 or code not in lookup:
            output.append(out)
            continue
        v, observed, contiguous, iscap, anchors = lookup[code]
        o, h, l, c, pc, volume, cap = v.T
        out['history_latest_date'] = sessions[start - 1] if start else None
        price_matched = observed[t] and np.isfinite(price) and abs(c[t] - price) < .0051
        if price_matched:
            out['signal_log_float_cap'] = np.log(cap[t]) if np.isfinite(cap[t]) and cap[t] > 0 else np.nan
        current_ok = price_matched and observed[start:t+1].all() and iscap[start:t+1].all() and contiguous[start+1:t+1].all()
        # Starting this run requires a verified non-cap preceding session.
        current_ok = current_ok and start > 0 and observed[start-1] and contiguous[start] and not iscap[start-1]
        out['current_run_verified'] = bool(current_ok)
        left = max(0, start-63)
        idx = np.arange(left, start)
        out['history63_coverage'] = float(observed[idx].sum() / 63)
        capids = idx[anchors[idx]]
        out['history63_limit_count'] = float(len(capids)) if observed[idx].any() else np.nan
        matured = capids[(capids+1 < start)]
        matured = matured[contiguous[matured+1]]
        if len(matured):
            follow = c[matured+1] / c[matured] - 1
            out['history63_link_rate'] = float(iscap[matured+1].mean())
            out['history63_follow_return'] = float(follow.mean())
            out['history63_follow_loss_rate'] = float((follow <= -.05).mean())
        run, maximum = 0, 0
        for j in idx:
            run = run+1 if anchors[j] and (run == 0 or contiguous[j]) else int(anchors[j])
            maximum = max(maximum, run)
        out['history63_max_run'] = float(maximum) if observed[idx].any() else np.nan
        if not current_ok:
            out['pullback_status'] = 'CURRENT_RUN_OR_ADJACENCY_GAP'
            output.append(out)
            continue
        # No current-run board is allowed to become the earlier pullback anchor.
        prior = np.arange(max(0, start-21), start-1)
        qualified = prior[anchors[prior]]
        if not len(qualified):
            complete = start >= 21 and observed[start-21:start].all() and contiguous[start-20:start].all()
            out['pullback_has_anchor'] = 0. if complete else np.nan
            out['pullback_status'] = 'NO_PRIOR_CAP_IN_COMPLETE_WINDOW' if complete else 'PRIOR_WINDOW_INCOMPLETE'
            output.append(out)
            continue
        anchor = int(qualified[-1])
        pull = np.arange(anchor+1, start)
        # Another cap in the gap would be a different run; all gap bars required.
        complete = observed[anchor:start+1].all() and contiguous[anchor+1:start+1].all() and not iscap[pull].any()
        if not complete:
            out['pullback_status'] = 'ANCHOR_TO_LAUNCH_PATH_GAP'
            output.append(out)
            continue
        low = float(l[pull].min())
        # Require observable pullback, otherwise retain a distinct non-pullback state.
        if low >= c[anchor]:
            out['pullback_has_anchor'] = 0.
            out['pullback_status'] = 'PRIOR_CAP_BUT_NO_PRICE_PULLBACK'
            output.append(out)
            continue
        out.update(anchor_date=sessions[anchor], pullback_status='OBSERVED_CAP_PULLBACK_RELAUNCH',
                   pullback_has_anchor=1., pullback_days=float(len(pull)),
                   pullback_min_return=ratio(low, c[anchor])-1,
                   pullback_low_vs_anchor_low=ratio(low, l[anchor])-1,
                   pullback_last_close_return=ratio(c[start-1], c[anchor])-1,
                   relaunch_reclaim_return=ratio(c[start], c[anchor])-1,
                   signal_vs_anchor_close=ratio(c[t], c[anchor])-1,
                   relaunch_excess_return=(ratio(c[start], pc[start])-1) - market.get(sessions[start], np.nan),
                   relaunch_close_position=ratio(c[start]-l[start], h[start]-l[start]),
                   relaunch_range=ratio(h[start]-l[start], pc[start]))
        # Do not compare unknown or mixed units; >100x adjacent shifts are quarantined.
        vv = volume[anchor:start+1]
        units_ok = np.isfinite(vv).all() and (vv > 0).all() and not ((vv[1:]/vv[:-1] > 100) | (vv[:-1]/vv[1:] > 100)).any()
        if units_ok:
            out.update(pullback_mean_volume_ratio=ratio(volume[pull].mean(), volume[anchor]),
                       pullback_min_volume_ratio=ratio(volume[pull].min(), volume[anchor]),
                       pullback_last_volume_ratio=ratio(volume[start-1], volume[anchor]),
                       relaunch_volume_ratio=ratio(volume[start], volume[pull].mean()))
        output.append(out)
    result = pd.DataFrame(output)
    if result.duplicated(['date', 'code']).any() or len(result) != len(signals):
        raise ValueError('feature denominator changed')
    return result, {'qualified_archived_anchors_for_relay_codes': source_anchor_n,
                    'descriptive_market_median_universe': 'archived daily universe with survivorship bias',
                    'historical_intent_or_minute_leadership_truth': 0}


def prepare():
    if (RUN/'implementation_BEFORE_FITS.json').exists():
        raise ValueError('Already prepared; no rewriting')
    b = joblib.load(BUNDLE)
    full = b['prepared']['relay'].copy()
    original_count = len(full)
    saved = pd.read_csv(ROLL, dtype={'code': str}, low_memory=False).query("arm == 'ROLL252'").copy()
    base = pd.read_csv(BASE, dtype={'code': str}, low_memory=False).query("arm == 'BASE'").copy()
    if len(saved) != 3707 or len(base) != 3707:
        raise ValueError('Original complete evaluation denominator changed')
    key = ['date','code']
    check = full.merge(base[key+RELAY_FIELDS+['target','target_date','label_valid']], on=key, how='inner', suffixes=('', '_original'), validate='one_to_one')
    if len(check) != 3707:
        raise ValueError('Missing original evaluation candidate')
    for k in RELAY_FIELDS + ['target','target_date','label_valid']:
        pd.testing.assert_series_equal(check[k].reset_index(drop=True), check[k+'_original'].reset_index(drop=True), check_names=False, check_dtype=False, atol=1e-12, rtol=1e-12)
    features, meta = build_features(full[key+['boards','price']], b['bars'], b['attempts'], b['sessions'])
    one = b['features'][key+['daily_one_word']].copy()
    full = full.merge(features, on=key, validate='one_to_one').merge(one, on=key, validate='one_to_one')
    assert len(full) == original_count
    # The feature-builder receives neither target nor any trade/exit future field.
    path = RUN/'prepared_FEATURES_BEFORE_FITS.joblib'
    joblib.dump({'full': full, 'sessions': b['sessions'], 'saved_roll': saved, 'saved_base': base}, path, compress=3)
    features.to_csv(RUN/'causal_features_ALL_ROWS.csv',index=False,encoding='utf-8-sig')
    months = [str(m) for m in pd.period_range('2025-07','2026-09',freq='M')]
    plans = []
    # Original ROLL252 training digest is an independent check of all 29 fields and y.
    recency = ROLL.parent
    old_impl = json.loads((recency/'implementation_BEFORE_FITS.json').read_text(encoding='utf-8'))
    for month in months:
        cutoff = month+'-01'
        train = history.training_rows(full, cutoff, 'relay')
        test = full[full.date.str[:7].eq(month)]
        old = next(c for c in old_impl['checks'] if c['month'] == month)
        digest = history._frame_digest(train, RELAY_FIELDS)
        if digest != old['training_digest'] or len(train) != old['training_n'] or len(test) != old['test_n']:
            raise ValueError('Original training membership/fields/labels differ: '+month)
        if train.target_date.ge(cutoff).any() or train.date.nunique()!=252:
            raise ValueError('Immature or wrong training window')
        plans.append(dict(month=month, cutoff=cutoff, train_n=len(train), test_n=len(test),
                          old_training_digest=digest, new_training_digest=history._frame_digest(train,FIELDS),
                          train_max_target=train.target_date.max(), train_signal_days=252))
    source_check=[]
    for p, expected in b['sources'].items():
        path=Path(p)
        actual=sha(path) if path.exists() else None
        source_check.append(dict(path=p, expected=expected, actual=actual, matched=actual==expected))
    write('inherited_source_binding_audit.json', {'checked_at':now(),'rows':source_check,
          'matched':sum(r['matched'] for r in source_check),'total':len(source_check),
          'qualification':'Existing frozen retrospective reconstruction; audit does not backdate source availability or certify original HTTP clocks.'})
    protected = [OLD_ROOT/'models/relay/BASE/model.joblib', OLD_ROOT/'models/relay/BASE/manifest.json',
                 OLD_ROOT/'protocol_BEFORE_EFFECTS.json',
                 PROJECT/'data/independent_stock_judgement/registry.jsonl',
                 PROJECT/'data/independent_stock_judgement/settlement_ledger.jsonl']
    bindings = {str(p):sha(p) for p in [BUNDLE,ROLL,BASE,Path(__file__),RUN/'protocol_BEFORE_FITS.json',RUN/'forum_hypotheses.md',RUN/'forum_web_read.txt',RUN/'prepared_FEATURES_BEFORE_FITS.joblib',*protected]}
    write('implementation_BEFORE_FITS.json',dict(formed_at=now(),bindings=bindings,training_plans=plans,
          feature_fields=FIELDS,source_meta=meta,full_n=len(full),
          feature_coverage=features[NEW].notna().sum().to_dict(),
          pullback_status=features.pullback_status.value_counts().to_dict(),
          inherited_source_total=len(source_check),inherited_source_matched=sum(r['matched'] for r in source_check),
          runtime={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']},
          fits=0,production_refits=0,new_forward_labels=0,deployment=False))
    print(json.dumps({'prepared':len(full),'status':features.pullback_status.value_counts().to_dict(),'plans':len(plans),'source_matched':sum(r['matched'] for r in source_check),'source_total':len(source_check)}),flush=True)


def metrics(frame, score):
    valid = frame.label_valid.fillna(False).astype(bool) & frame.target.notna() & frame[score].notna()
    f=frame[valid];y=f.target.astype(int).to_numpy(); pred=f[score].to_numpy()>.5
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    return dict(total=len(frame),known=len(f),unknown=len(frame)-len(f),TP=tp,FP=fp,FN=fn,TN=tn,
                Accuracy=(tp+tn)/len(f) if len(f) else None,Precision=tp/(tp+fp) if tp+fp else None,
                Recall=tp/(tp+fn) if tp+fn else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def top(frame,score,k):
    chosen=frame.sort_values(['date',score,'code'],ascending=[True,False,True],kind='stable').groupby('date',sort=False).head(k)
    valid=chosen.label_valid.fillna(False).astype(bool)&chosen.target.notna()
    return dict(selected=len(chosen),known=int(valid.sum()),unknown=int((~valid).sum()),hits=int(chosen.loc[valid,'target'].sum()),days=chosen.date.nunique())


def bootstrap(frame, column):
    known=frame[frame.label_valid.fillna(False).astype(bool)&frame.target.notna()].copy()
    known['pred']=known[column].gt(.5)
    known['base']=known.roll_score.gt(.5)
    x=[]
    for _,g in known.groupby('date',sort=True):
        row=[]
        for p in ['pred','base']:
            row.extend([int((g.target.eq(1)&g[p]).sum()),int((g.target.eq(0)&g[p]).sum()),int((g.target.eq(1)&~g[p]).sum())])
        x.append(row)
    if len(x)<5:return {'lower':None,'upper':None,'days':len(x)}
    x=np.asarray(x);n=len(x);rng=np.random.default_rng(20261020)
    starts=rng.integers(0,n,size=(2000,(n+4)//5));idx=((starts[:,:,None]+np.arange(5))%n).reshape(2000,-1)[:,:n]
    counts=x[idx].sum(axis=1)
    den1=2*counts[:,0]+counts[:,1]+counts[:,2];den0=2*counts[:,3]+counts[:,4]+counts[:,5]
    ok=(den1>0)&(den0>0);delta=2*counts[ok,0]/den1[ok]-2*counts[ok,3]/den0[ok]
    return dict(lower=float(np.quantile(delta,.025)),upper=float(np.quantile(delta,.975)),days=n,draws=2000,undefined=int((~ok).sum()),method='paired circular signal-date blocks, block=5; descriptive repeated retrospective research, no selection-adjusted guarantee')


def run():
    e=read('implementation_BEFORE_FITS.json');verify(e['bindings'])
    if (RUN/'attempt_STARTED.json').exists():raise ValueError('Already attempted; do not refit')
    write('attempt_STARTED.json',dict(started_at=now(),maximum_fits=15))
    b=joblib.load(RUN/'prepared_FEATURES_BEFORE_FITS.joblib');full=b['full'];saved=b['saved_roll'];base=b['saved_base']
    outputs=[]
    with threadpool_limits(limits=2):
        for plan in e['training_plans']:
            if now() >= read('protocol_BEFORE_FITS.json')['deadline_utc']:raise ValueError('Batch deadline reached')
            train=history.training_rows(full,plan['cutoff'],'relay')
            test=full[full.date.str[:7].eq(plan['month'])].copy()
            if history._frame_digest(train,FIELDS)!=plan['new_training_digest']:raise ValueError('Frozen train drift')
            prefix=plan['month']
            write(prefix+'_fit_STARTED.json',dict(**plan,started_at=now(),fit_attempt=1))
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.25,max_iter=2500,random_state=271))
            weights=1/train.groupby('date').date.transform('size')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model.fit(train[FIELDS],train.target.astype(int),logisticregression__sample_weight=weights)
                # Persist all candidate scores and model before looking at evaluation.
                test['new_score']=model.predict_proba(test[FIELDS])[:,list(model.classes_).index(1)]
                test.to_csv(RUN/(prefix+'_predictions_BEFORE_VALIDATION.csv'),index=False,encoding='utf-8-sig')
                joblib.dump(dict(model=model,fields=FIELDS,cutoff=plan['cutoff'],training_digest=plan['new_training_digest']),RUN/(prefix+'_PULLBACK_PERSONALITY.joblib'),compress=3)
                outputs.append(test)
                write(prefix+'_fit_COMPLETED.json',dict(completed_at=now(),prediction_sha=sha(RUN/(prefix+'_predictions_BEFORE_VALIDATION.csv')),model_sha=sha(RUN/(prefix+'_PULLBACK_PERSONALITY.joblib'))))
                print(json.dumps({'completed_month':prefix,'train':len(train),'test':len(test)}),flush=True)
            except Exception as ex:
                write(prefix+'_fit_FAILED.json',dict(failed_at=now(),error=repr(ex),no_retry=True))
                raise
    pred=pd.concat(outputs,ignore_index=True)
    q=saved.rename(columns={'signal_date':'date','raw_score':'roll_score'})[['date','code','target_date','truth','roll_score']]
    pred=pred.merge(q,on=['date','code'],validate='one_to_one',suffixes=('','_saved'))
    if len(pred)!=3707 or not pred.target_date.eq(pred.target_date_saved).all():raise ValueError('Baseline comparison denominator/target date mismatch')
    v=pred.label_valid.fillna(False).astype(bool)&pred.target.notna()
    truth=pred.truth.astype('boolean')
    if not (pred.loc[v,'target'].astype(bool).to_numpy()==truth.loc[v].to_numpy()).all():raise ValueError('Baseline truth mismatch')
    pred=pred.merge(base[['date','code','raw_score']].rename(columns={'raw_score':'base_score_saved'}),on=['date','code'],validate='one_to_one')
    pred.to_csv(RUN/'full_paired_predictions.csv',index=False,encoding='utf-8-sig')
    report={'formed_at':now(),'fits':15,'baseline_fits':0,'production_refits':0,'full_n':len(pred),'folds':{}}
    for fold in FOLDS:
        f=pred[pred.date.between(fold['start'],fold['end'])]
        groups={'all':f,'actual_nonword':f[f.daily_one_word.eq(0)],'actual_oneword':f[f.daily_one_word.eq(1)],'identity_unknown':f[f.daily_one_word.isna()],
                'cap_pullback':f[f.pullback_has_anchor.eq(1)],'no_pullback':f[f.pullback_has_anchor.eq(0)],'pullback_unknown':f[f.pullback_has_anchor.isna()]}
        result={}
        for name,g in groups.items():
            result[name]={arm:dict(metrics=metrics(g,col),top={str(k):top(g,col,k) for k in (1,2,3)}) for arm,col in [('ROLL252','roll_score'),('PULLBACK_PERSONALITY','new_score'),('FROZEN_BASE','base_score_saved')]}
        result['nonword_F1_delta95']=bootstrap(groups['actual_nonword'],'new_score')
        report['folds'][fold['id']]=result
    folds=report['folds'];sep=folds['2026Sep']['actual_nonword'];old=sep['ROLL252']['metrics'];new=sep['PULLBACK_PERSONALITY']['metrics']
    checks=dict(nonword_F1_all_folds=all(v['actual_nonword']['PULLBACK_PERSONALITY']['metrics']['F1']>=v['actual_nonword']['ROLL252']['metrics']['F1'] for v in folds.values()),
                sep_recall_plus10pp=new['Recall']>=old['Recall']+.10,
                sep_precision_not_less5pp=new['Precision'] is not None and old['Precision'] is not None and new['Precision']>=old['Precision']-.05,
                sep_nonword_top2_not_worse=sep['PULLBACK_PERSONALITY']['top']['2']['hits']>=sep['ROLL252']['top']['2']['hits'])
    report['fixed_screen']=checks;report['screen_passed']=all(checks.values())
    report['statistically_supported']=folds['2026Sep']['nonword_F1_delta95']['lower'] is not None and folds['2026Sep']['nonword_F1_delta95']['lower']>0
    report['deployed']=False;report['new_forward_labels']=0;report['calibrated_probability']=False
    write('results.json',report)
    print(json.dumps({'fits':15,'screen':checks,'passed':report['screen_passed'],'sep_nonword':{'ROLL252':old,'NEW':new},'sep_F1_delta95':folds['2026Sep']['nonword_F1_delta95']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    try:
        (prepare if args.action=='prepare' else run)()
    except Exception as ex:
        target='failure_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json'
        write(target,dict(at=now(),action=args.action,error=repr(ex),fits_started=len(list(RUN.glob('*_fit_STARTED.json'))),completed=len(list(RUN.glob('*_fit_COMPLETED.json')))))
        raise
