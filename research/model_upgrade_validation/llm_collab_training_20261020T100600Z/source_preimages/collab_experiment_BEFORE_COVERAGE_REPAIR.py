"""Isolated collaboration research; immutable stages, no production writes.

The only fitted estimator is an eight-feature head on OLD, pre-September,
outcome-qualified independent calls. New labels are inaccessible to the public
review packet and are revealed only after all live-review decisions are bound.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/model_upgrade_validation/llm_joint_replay_20261019T172800Z'
BUNDLE = ROOT / 'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_prepared_v2/engine_context_WITH_CURRENT15.joblib'
CALENDAR = ROOT / 'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'
SEED = 'actual-two-llm-evidence-and-learned-arbitration-20261020-v1'
STRATA = [('SepEarly', '2026-09-01', '2026-09-10'),
          ('SepMiddle', '2026-09-11', '2026-09-20'),
          ('SepLate', '2026-09-21', '2026-09-30')]
FEATURES = ['codex_vote', 'gemini_vote', 'codex_unknown', 'gemini_unknown',
            'one_price', 'one_price_unknown', 'codex_vote_x_one_price', 'gemini_vote_x_one_price']
VOTES = {'YES': 1., 'NO': -1., 'UNKNOWN': 0.}


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)): return x.item()
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def number(value):
    try: return float(value) if pd.notna(value) and np.isfinite(float(value)) else None
    except (ValueError, TypeError): return None
def old_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, OLD / filename)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(OLD)); spec.loader.exec_module(module)
    return module


def vector(codex, gemini, one_price):
    c, g = VOTES[codex], VOTES[gemini]
    known = number(one_price) in (0., 1.)
    w = float(one_price) if known else 0.
    return [c, g, float(codex == 'UNKNOWN'), float(gemini == 'UNKNOWN'),
            w, float(not known), c*w, g*w]


def prepare():
    """Date-only selection and explicit signal projection. No test outcomes."""
    protocol = {
        'created_at_utc': now(), 'experiment': 'ACTUAL_CODEX_GEMINI38_COLLAB_TRAINING',
        'status': 'CODEX_FIXED_PROPOSAL_AWAITING_GEMINI_AGREEMENT',
        'human_constraints': 'Preserve ALL old models and results; no adoption, deployment or production switch without subsequent user decision.',
        'hypothesis': 'Evidence-specific changes and a learned arbitration head can reduce induced errors, harmful withdrawals and Top3 substitution losses while preserving complementary detection.',
        'main_event': 'Next adjacent official trading session close limit-up; exact-cent 10% unadjusted price corroborated by valid archived promotion target. Missing qualification remains UNKNOWN.',
        'secondary_event': 'Intraday touch, separate from close limit-up.',
        'baseline': 'Codex independent; Gemini independent is a prespecified peer reference.',
        'challenges': ['A_evidence_audited_joint', 'B_fixed_learned_arbitration'],
        'references': ['Old formal BASE if saved and same-pool available', 'Saved ROLL252, separate from BASE', 'Always NO', 'Height Top3'],
        'selection': {'seed': SEED, 'strata': STRATA, 'days_per_stratum': 4,
                      'rule': 'Lowest SHA256(seed:stratum:date), excluding every date in previous 12-day sample; complete archived relay pool, no label/score/performance/size filter.'},
        'training': {'source': str(OLD/'complete_122_revealed_all_arms.csv'),
                     'signal_date_before': '2026-09-01', 'rows': 'all pre-September old rows with strict_close known; unknown labels retained in exclusions',
                     'chronology': 'Every included training target_date strictly before every test signal date.',
                     'features': FEATURES, 'model': 'sklearn LogisticRegression',
                     'params': {'C': .25, 'penalty': 'l2', 'solver': 'lbfgs', 'fit_intercept': True, 'max_iter': 1000, 'tol': .0001, 'class_weight': None},
                     'weights': 'Each known training row receives (known rows / training days) / known rows on its day; mean sample weight is one, dates have equal total weight.',
                     'fits_allowed': 1, 'search': False, 'decision': 'YES iff raw score > 0.5; NO otherwise, except both reviewers UNKNOWN -> UNKNOWN.',
                     'ranking': 'Raw learned score descending, then equal average of both independent full-pool ranks, then fixed anonymous-ID hash. No forced shape slots.',
                     'probability': 'Uncalibrated internal score, never advertised as validated event probability.'},
        'consultation': {'sequence': ['Both reviewers independently write all rows and full rank before viewing the other.', 'Codex proposes each change using concrete evidence and counterevidence.', 'Gemini audits or counters each proposal once.', 'Codex binds acceptance/rejection with reasons; no forced consensus.', 'Freeze all rows, Top3, head output and all discussion hashes; only then reveal.'],
                         'change_card_fields': ['case', 'id', 'from', 'to', 'original_support', 'signal_evidence', 'counterevidence', 'alternative_explanation', 'change_cost', 'new_fact_or_reinterpretation', 'resolution'],
                         'top3_change': 'Name entrants and displaced stocks, compare both on identical available evidence. UNKNOWN withdrawal requires its own reason and opportunity cost.',
                         'unresolved': 'Preserve Codex baseline call and explicitly record unresolved disagreement; do not silently force UNKNOWN or YES.',
                         'prohibited_shortcuts': ['No word=YES or word=NO rule', 'No posthoc 2.2/2.5x volume cutoff', 'No fixed word/nonword/low-board slots', 'No theme, queue or money-flow invention', 'No one-field missingness ban on all predictions']},
        'metrics': ['Complete Top3 hits, misses and unknown slots', 'Full-known correct fraction (UNKNOWN is unanswered)', 'Judged Accuracy, coverage, Precision, full Recall, F1', 'Shape, board and stratum partitions', 'All correction/induced-error/resolution/withdrawal changes and Top3 displacement', '10000 fixed-seed day-cluster paired intervals'],
        'success_rule': 'Diagnostic candidate only if complete Top3 confirmed hits and full-known correct fraction both exceed Codex independent; Precision and non-one-price F1 not lower. Unknown labels and coverage reported separately. Even a diagnostic win is not stable/forward proof and never permits automatic adoption.',
        'budget_minutes': 30,
        'end_rule': 'One fit and one reveal; no effect-driven parameter change, date replacement or refit. If the actual Gemini service remains unavailable, do not claim joint completion and do not reveal the new test.',
        'limits': ['Previously researched historical domain and late-acquired sources: NOT independently certified blind or genuine forward evidence.', 'No qualified historical theme/forum/announcement text: price-volume/emotion/competition scope only.', 'Only 8 available old training days; small fit may be unstable, all same-date stocks correlated.', 'September-only holdout is not cross-cycle validation; retain pool/survivor and supplier biases.', 'No trading fills, fees or returns are inferred from this event-prediction experiment.'],
        'method_sources': ['https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'],
    }
    write(HERE/'protocol_BEFORE_TEST.json', protocol)
    oldmap = read(OLD/'PRIVATE_mapping_DO_NOT_READ_BEFORE_JOINT_FREEZE.json')
    excluded = {r['date'] for r in oldmap}
    data = joblib.load(BUNDLE)
    sf = ['boards','turnover','amplitude','first_seal_minute','last_seal_minute','bomb_count','signal_amount','signal_float_cap']
    mf = ['limit_count','consecutive_count','max_board','failed_count','down_limit_count','failed_rate','seal_rate','yesterday_premium','emotion_state']
    bf = ['date','code','open','high','low','close','pre_close','volume','amount','turnover']
    relay = data['prepared']['relay'][['date','code','name','price']+sf].copy()
    relay['date'] = relay.date.astype(str).str[:10]; relay['code'] = relay.code.astype(str).str.zfill(6)
    selected = []
    for phase, start, end in STRATA:
        dates = set(relay.loc[relay.date.between(start,end),'date']) - excluded
        assert len(dates) >= 4, (phase, len(dates))
        selected += [(phase,d) for d in sorted(dates,key=lambda d:digest(SEED+':'+phase+':'+d))[:4]]
    selected.sort(key=lambda v: digest(SEED+':case:'+v[1]))
    market = data['prepared']['market'][['date']+mf].copy(); market['date'] = market.date.astype(str).str[:10]
    market = market.sort_values('date')
    bars = data['bars'][bf].copy()
    bars['date'] = bars.date.astype(str).str[:10]; bars['code'] = bars.code.astype(str).str.zfill(6)
    codes = set(relay.loc[relay.date.isin([d for _,d in selected]),'code'])
    bars = bars.loc[bars.code.isin(codes)].sort_values(['code','date'])
    assert not bars.duplicated(['date','code']).any()
    groups = {c:g for c,g in bars.groupby('code')}
    features = data['features'][['date','code','daily_one_word','relative_volume','verified_turnover_boards','first_open_after_one_word']].copy()
    features['date'] = features.date.astype(str).str[:10]; features['code'] = features.code.astype(str).str.zfill(6)
    features = features.set_index(['date','code']); assert features.index.is_unique
    sessions = sorted(str(d)[:10] for d in data['sessions']); session_index = {d:i for i,d in enumerate(sessions)}
    notes = ('日期/代码/名称/旧模型分数/测试结果遮蔽；仅信号日及之前字段。turnover_pct及历史turnover为百分数；市场failed_rate/seal_rate/premium为比例。'
             '封板分钟从09:30起墙钟分钟，午休不扣。one_price为当日OHLC同价形态；null未知不是0，不等于买得到。'
             'observed_nonword_count是本轮已核非一价节点数，不是连续换手天数；过去5条可能跨交易日，须读lag_sessions，0才是T。'
             '题材、公告、论坛与真实盘口均未知；成交额/流通市值仅池内分位，不证明核心和资金因果。Top3为相对排序，不必全部YES。')
    packet = {'experiment':protocol['experiment'], 'event':protocol['main_event'], 'notes':notes, 'outcomes_present':False, 'days':[]}
    mapping = []
    for i,(phase,day) in enumerate(selected,1):
        case = f'N{i:02}'
        rows = relay.loc[relay.date.eq(day)].copy()
        rows['_order'] = rows.code.map(lambda c: digest(SEED+':'+day+':'+c)); rows = rows.sort_values('_order')
        assert not rows.code.duplicated().any()
        for col in ['signal_amount','signal_float_cap']:
            rows[col+'_pctile'] = pd.to_numeric(rows[col],errors='coerce').rank(pct=True)*100
        m = market.loc[market.date.le(day)].tail(3)
        stockrows = []
        for j,(_,r) in enumerate(rows.iterrows(),1):
            sid = f'S{j:02}'; f = features.loc[(day,r.code)]
            hist = groups.get(r.code,pd.DataFrame(columns=bf))
            hist = hist.loc[hist.date.le(day)].tail(5)
            past = []
            for _,h in hist.iterrows():
                pre = number(h.pre_close)
                hh = {k: (100*(number(h[col])/pre-1) if pre and number(h[col]) is not None else None)
                      for k,col in [('o','open'),('h','high'),('l','low'),('c','close')]}
                hh.update(turnover=number(h.turnover), lag_sessions=session_index[day]-session_index[h.date]); past.append(hh)
            item = {'id':sid,'boards':number(r.boards),'one_price':number(f.daily_one_word),
                    'turnover_pct':number(r.turnover)*100 if number(r.turnover) is not None else None,
                    'amplitude_pct':number(r.amplitude),'first_seal_min':number(r.first_seal_minute),
                    'last_seal_min':number(r.last_seal_minute),'reported_open_count':number(r.bomb_count),
                    'amount_pctile':number(r.signal_amount_pctile),'float_cap_pctile':number(r.signal_float_cap_pctile),
                    'volume_vs_previous':number(f.relative_volume),'observed_nonword_count':number(f.verified_turnover_boards),
                    'first_open_after_one_price':number(f.first_open_after_one_word), 'past5':past}
            stockrows.append(item)
            mapping.append({'case':case,'id':sid,'period':phase,'date':day,'code':r.code,'name':r['name'],'pool_price':number(r.price)})
        packet['days'].append({'case':case,'complete_pool_count':len(rows),
                               'past3_market_old_to_new':[{k:clean(r[k]) for k in mf} for _,r in m.iterrows()], 'stocks':stockrows})
    write(HERE/'PRIVATE_test_mapping_UNREAD_UNTIL_FREEZE.json', mapping)
    write(HERE/'review_packet_SIGNAL_ONLY.json', packet)
    lines = ['# 新样本完整池：只含信号资料', '', notes, '']
    cols = ['id','boards','one_price','turnover_pct','amplitude_pct','first_seal_min','last_seal_min','reported_open_count','amount_pctile','float_cap_pctile','volume_vs_previous','observed_nonword_count','first_open_after_one_price']
    def fmt(v): return '?' if v is None else str(round(v,4)) if isinstance(v,float) else str(v)
    for d in packet['days']:
        lines += ['## '+d['case']+' 完整池 '+str(d['complete_pool_count']), '', json.dumps(d['past3_market_old_to_new'],ensure_ascii=False), '', '|'+ '|'.join(cols)+'|', '|'+ '|'.join(['---']*len(cols))+'|']
        lines += ['|'+'|'.join(fmt(r[k]) for k in cols)+'|' for r in d['stocks']]
        lines += ['', '逐股过去5条(O/H/L/C%、换手%、官方交易日lag)：']
        for r in d['stocks']: lines.append(r['id']+' '+json.dumps(clean(r['past5']),ensure_ascii=False,separators=(',',':')))
        lines.append('')
    with (HERE/'review_packet_SIGNAL_ONLY.md').open('x',encoding='utf-8') as f: f.write('\n'.join(lines))
    schema = {'reviewer':'FILL_ACTUAL_REVIEWER', 'packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY.json'),
              'created_at_utc':'ACTUAL_CLOCK', 'outcomes_seen':False,
              'days':[{'case':d['case'],'ranking':[r['id'] for r in d['stocks']], 'top3':[],
                       'market_reason':'AUTHOR_WRITE', 'stocks':[{'id':r['id'],'decision':'UNFILLED', 'reason':'AUTHOR_WRITE_SUPPORT_AND_COUNTEREVIDENCE'} for r in d['stocks']]} for d in packet['days']]}
    write(HERE/'decision_SCHEMA_NOT_PREDICTION.json',schema)
    protected = [ROOT/'data/independent_stock_judgement/registry.jsonl', ROOT/'data/independent_stock_judgement/settlement_ledger.jsonl', ROOT/'model_champion_runtime.py']
    oldfiles = [p for p in OLD.iterdir() if p.is_file()]
    write(HERE/'preparation_receipt.json', {'created_at_utc':now(), 'source_sha256':sha(BUNDLE),
          'protocol_sha256':sha(HERE/'protocol_BEFORE_TEST.json'),'packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY.json'),
          'mapping_sha256':sha(HERE/'PRIVATE_test_mapping_UNREAD_UNTIL_FREEZE.json'),
          'days':len(selected),'rows':len(mapping), 'old_dates_excluded':True, 'test_outcomes_read':False,
          'protected_hashes':{str(p):sha(p) for p in protected+oldfiles if p.exists()}})
    print(json.dumps({'days':len(selected),'rows':len(mapping),'sizes':{d['case']:d['complete_pool_count'] for d in packet['days']}, 'test_outcomes_read':False}))


def train():
    agreement = read(HERE/'final_design_agreement_BEFORE_FIT.json')
    assert agreement['real_gemini_review_received'] and agreement['old_models_preserved']
    assert agreement['protocol_sha256'] == sha(HERE/'protocol_BEFORE_TEST.json')
    # An attempt is locked before deserializing any fitting inputs.
    write(HERE/'fit_attempt_ONCE.json', {'created_at_utc':now(),'allowed_fits':1,'test_label_reads':0})
    df = pd.read_csv(OLD/'complete_122_revealed_all_arms.csv',dtype={'date':str,'target_date':str,'code':str})
    old_train = df.loc[df.date.lt('2026-09-01')].copy()
    known = old_train.loc[old_train.strict_close.notna()].copy()
    assert known.strict_close.isin([0,1]).all() and len(known.strict_close.unique())==2
    test_dates = [r['date'] for r in read(HERE/'PRIVATE_test_mapping_UNREAD_UNTIL_FREEZE.json')]
    assert known.target_date.notna().all() and known.target_date.max() < min(test_dates)
    x = np.array([vector(r.codex_independent,r.gemini_independent,r.one_price) for _,r in known.iterrows()])
    y = known.strict_close.astype(int).to_numpy()
    counts = known.groupby('date').size(); ndays=len(counts)
    weights = known.date.map((len(known)/ndays)/counts).to_numpy()
    assert np.isclose(weights.sum(),len(known))
    write(HERE/'fit_inputs_BEFORE_OPTIMIZATION.json', {'created_at_utc':now(), 'features':FEATURES,
          'members':[{'case':r['case'],'id':r['id'],'date':r['date'],'target_date':r['target_date'],'label':int(r['strict_close']),'weight':float(w),'x':xx.tolist()} for (_,r),w,xx in zip(known.iterrows(),weights,x)],
          'excluded_unknown_preSeptember':old_train.loc[old_train.strict_close.isna(),['case','id','date']].to_dict('records'),
          'excluded_September_rows':int((~df.date.lt('2026-09-01')).sum()),'latest_train_target':known.target_date.max(), 'earliest_test_signal':min(test_dates)})
    model = LogisticRegression(C=.25,penalty='l2',solver='lbfgs',fit_intercept=True,max_iter=1000,tol=.0001,class_weight=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y,sample_weight=weights)
    artifact = HERE/'B_learned_arbitration_RESEARCH_ONLY.joblib'
    assert not artifact.exists(); joblib.dump({'model':model,'features':FEATURES,'protocol_sha256':sha(HERE/'protocol_BEFORE_TEST.json'),'production_allowed':False},artifact)
    loaded = joblib.load(artifact)['model']; scores = model.predict_proba(x)[:,1]
    assert np.allclose(loaded.predict_proba(x)[:,1], scores, atol=0,rtol=0)
    pred=scores>.5
    write(HERE/'fit_receipt_BEFORE_NEW_TEST_LABELS.json', {'created_at_utc':now(), 'fits':1,'rows':len(known),'days':ndays,
          'model_sha256':sha(artifact),'coefficient_by_feature':dict(zip(FEATURES,model.coef_[0].tolist())),
          'intercept':model.intercept_.tolist(), 'in_sample_correct':int((pred==y).sum()),'in_sample_denominator':len(y),
          'in_sample_is_not_validation':True,'latest_train_target':known.target_date.max(), 'earliest_test_signal':min(test_dates),
          'save_load_exact':True,'test_labels_read':False,'old_models_retrained':0,'adoption_requires_user_decision':True})
    print(json.dumps({'fits':1,'train_rows':len(known),'train_days':ndays,'latest_train_target':known.target_date.max(),'test_labels_read':False}))


def validate(path):
    p = read(path); packet = read(HERE/'review_packet_SIGNAL_ONLY.json')
    assert p['packet_sha256']==sha(HERE/'review_packet_SIGNAL_ONLY.json') and p['outcomes_seen'] is False
    assert len(p['days'])==len(packet['days'])
    lookup = {d['case']:d for d in p['days']}; assert len(lookup)==len(packet['days'])
    for d in packet['days']:
        v=lookup[d['case']]; ids={r['id'] for r in d['stocks']}
        assert {r['id'] for r in v['stocks']}==ids and len(v['stocks'])==len(ids)
        assert set(v['ranking'])==ids and len(v['ranking'])==len(ids)
        assert v['top3']==v['ranking'][:min(3,len(ids))]
        assert all(r['decision'] in VOTES and len(r['reason'])>=8 for r in v['stocks'])
    return p


def predict_head():
    c=validate(HERE/'codex_independent_BEFORE_GEMINI.json')
    g=validate(HERE/'gemini_first_received_BEFORE_DISCUSSION.json')
    model=joblib.load(HERE/'B_learned_arbitration_RESEARCH_ONLY.joblib')['model']
    packet=read(HERE/'review_packet_SIGNAL_ONLY.json'); gd={d['case']:d for d in g['days']}; pdays={d['case']:d for d in packet['days']}
    output={'reviewer':'FIXED_TRAINED_HEAD_NOT_LIVE_LLM','packet_sha256':sha(HERE/'review_packet_SIGNAL_ONLY.json'),'created_at_utc':now(),'outcomes_seen':False,'days':[]}
    for d in c['days']:
        gg=gd[d['case']]; cs={r['id']:r for r in d['stocks']}; gs={r['id']:r for r in gg['stocks']}
        ps={r['id']:r for r in pdays[d['case']]['stocks']}; ids=list(cs)
        xx=np.array([vector(cs[s]['decision'],gs[s]['decision'],ps[s]['one_price']) for s in ids])
        scores=dict(zip(ids,model.predict_proba(xx)[:,1]))
        ranking=sorted(ids,key=lambda s:(-scores[s],(d['ranking'].index(s)+gg['ranking'].index(s))/2,digest(SEED+':tie:'+d['case']+':'+s)))
        rows=[]
        for sid in ids:
            both_unknown=cs[sid]['decision']==gs[sid]['decision']=='UNKNOWN'
            dec='UNKNOWN' if both_unknown else 'YES' if scores[sid]>.5 else 'NO'
            rows.append({'id':sid,'decision':dec,'raw_uncalibrated_score':float(scores[sid]),'reason':'固定旧样本融合头；双方独立投票与一价交互；双弃判保留UNKNOWN；分数未校准。'})
        output['days'].append({'case':d['case'],'ranking':ranking,'top3':ranking[:3],'market_reason':'Fixed fitted vote/shape fusion; no new-label inputs.', 'stocks':rows})
    write(HERE/'B_head_predictions_BEFORE_LABELS.json', output)
    validate(HERE/'B_head_predictions_BEFORE_LABELS.json')
    print(json.dumps({'head_predictions':sum(len(d['stocks']) for d in output['days']),'test_outcome_reads':0}))


if __name__=='__main__':
    {'prepare':prepare,'train':train,'predict-head':predict_head}[sys.argv[1]]()
