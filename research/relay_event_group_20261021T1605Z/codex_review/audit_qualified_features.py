"""Independent fixed-member/unknown/price-state checks; zero model fits."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import json
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / 'qualified_final'
sys.path.insert(0, str(ROOT))
from historical_symbolic_context import DatedNames, tokens
from security_master_v4 import validate_master
import short_term_thinking_dataset_builder as b

data = joblib.load(ROOT / 'research/model_upgrade_validation/short_term_thinking_20261015T071000Z/historical_final/dataset.joblib')
sessions = data['sessions']; pos = {d:i for i,d in enumerate(sessions)}
master = validate_master(pd.read_csv(ROOT / 'data/v4/security_master_intervals.csv', dtype=str)).fillna('')
names = DatedNames(master)
features = pd.read_csv(OUT / 'event_group_features_9245.csv', dtype={'code':str})
ledger = [json.loads(line) for line in (OUT / 'EVENT_GROUP_LEDGER_DETAILS.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(features) == len(ledger) == 9245
assert not features.duplicated(['date','code']).any()
assert set(zip(features.date, features.code)) == set(zip(data['features'].date, data['features'].code))
assert int(features.target.isna().sum()) == 67
assert not any('target' in row for row in ledger)
fi = features.set_index(['date','code'])
active = defaultdict(set); closing = defaultdict(dict)
for row in data['attempts'].itertuples():
    if row.pool_assignment in ('CLOSING_LIMIT_UP','FAILED_BOARD_OBSERVATION'):
        active[row.signal_date].add(row.code)
    if row.pool_assignment == 'CLOSING_LIMIT_UP': closing[row.signal_date][row.code] = int(row.board_height)
for row in data['features'].itertuples():
    active[row.date].add(row.code)
    closing[row.date][row.code] = int(row.boards)
cache = {}
def group(E, anchor, token):
    key = E, anchor, token
    if key in cache: return cache[key]
    pool = set().union(*(active.get(day,set()) for day in sessions[max(0,pos[E]-4):pos[E]+1]))
    known = set(); unknown = set()
    for code in pool - {anchor}:
        n = names.resolve(code,E,'')
        if n['historical_eligible'] is False: continue
        if n['interval_observed'] == 1 and n['historical_eligible'] is True:
            if token in tokens(n['name']): known.add(code)
        else: unknown.add(code)
    cache[key] = known, unknown
    return known, unknown

needed = set()
for row in ledger:
    if row['E']:
        S,c,E,a = row['date'],row['code'],row['E'],row['anchor']
        for day in (S,sessions[pos[S]-1],E,sessions[pos[E]-1]):
            for code in set(row['members_known']) | {c,a}: needed.add((day,code))
obs = {}
for chunk in pd.read_csv(OUT / 'MEMBER_DAILY_OBSERVATIONS.csv',dtype={'code':str},chunksize=100000):
    sub = chunk.loc[pd.MultiIndex.from_frame(chunk[['date','code']]).isin(needed)]
    for row in sub.to_dict('records'): obs[(row['date'],row['code'])] = row

def state(day, code):
    r = obs.get((day, code))
    if not r or r['bar_valid'] != 1: return None
    assert np.isfinite(r['volume']) and r['volume'] > 0
    assert np.isfinite(r['amount']) and r['amount'] > 0
    assert r['interval_observed'] == 1 and r['historical_eligible'] == True
    pc = int(r['pre_close_cents'])
    upper = int((Decimal(pc)*Decimal('1.10')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
    lower = int((Decimal(pc)*Decimal('.90')).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
    assert lower <= r['low_cents'] <= r['close_cents'] <= r['high_cents'] <= upper
    assert r['low_cents'] <= r['open_cents'] <= r['high_cents']
    assert r['is_limit_up'] == int(r['close_cents'] == upper)
    return r

checks = defaultdict(int)
fields = ['event_cohort_matched','event_cohort_size','event_cohort_primary_member',
          'event_cohort_damage_ratio','event_cohort_repair_ratio','event_anchor_status','event_cohort_amount_share_delta']
for r in ledger:
    f = fi.loc[(r['date'],r['code'])]
    if r['source_qualification'] == 'UNKNOWN_CANDIDATE_OR_EXTERNAL_ANCHOR':
        assert f[fields].isna().all()
        checks['source_unknown_preserved'] += 1
    if not r['E']: continue
    S,c,E,a,t = r['date'],r['code'],r['E'],r['anchor'],r['token']
    assert a != c and 1 <= pos[S]-pos[E] <= 10
    assert t.startswith(('H2:','P:'))
    assert closing[E][a] == 3 and closing[sessions[pos[E]-1]][a] == 2
    known,unknown = group(E,a,t)
    active_at_E = any(c in active.get(day,set()) for day in sessions[max(0,pos[E]-4):pos[E]+1])
    c_at_E = names.resolve(c,E,'')
    if active_at_E and c_at_E['historical_eligible'] is None and r['source_qualification'] != 'UNKNOWN_CANDIDATE_OR_EXTERNAL_ANCHOR':
        assert pd.isna(f['event_cohort_primary_member']), (S,c,E,'unknown candidate E membership')
    known = known-{c,a}; unknown=unknown-{c,a}
    assert known == set(r['members_known']) and unknown == set(r['members_unresolved'])
    assert len(known) == r['N_known'] and len(unknown) == r['N_unresolved']
    assert r['denominator_lower_N'] == len(known)
    assert r['denominator_upper_N'] == len(known)+len(unknown)
    if len(known)+len(unknown) == 0: continue
    k=0; u=len(unknown); kr=0; ur=len(unknown)
    prior=sessions[pos[S]-1]
    for member in known:
        v=state(S,member); p=state(prior,member)
        if v is None: u+=1
        elif v['is_limit_up']==0: k+=1
        if v is None or p is None: ur+=1
        elif v['is_limit_up']==1 and p['is_limit_up']==0: kr+=1
    assert (k,u,kr,ur)==(r['damage_k'],r['damage_u'],r['repair_k'],r['repair_u'])
    if r['source_qualification'] != 'UNKNOWN_CANDIDATE_OR_EXTERNAL_ANCHOR':
        for field, wins, missing in [('event_cohort_damage_ratio',k,u),('event_cohort_repair_ratio',kr,ur)]:
            if missing or not known: assert pd.isna(f[field])
            else: assert abs(f[field]-wins/len(known))<=.000001
        if unknown: assert pd.isna(f['event_cohort_size']) and pd.isna(f['event_cohort_amount_share_delta'])
        if pd.notna(f.event_cohort_amount_share_delta):
            now=[state(S,x) for x in known|{c}]; old=[state(prior,x) for x in known|{c}]
            assert all(v is not None for v in now+old)
            expected=state(S,c)['amount']/sum(x['amount'] for x in now)-state(prior,c)['amount']/sum(x['amount'] for x in old)
            assert abs(expected-f.event_cohort_amount_share_delta)<=.000001
    checks['event_rosters_recomputed'] += 1

receipt=dict(at_utc=datetime.now(timezone.utc).isoformat(),fits=0,rows=9245,
    checks=dict(checks),distinct_event_groups=len(cache),price_observations_checked=len(obs),
    limitations=['Name proxies are not verified trading themes or fund identities.',
                 'Dated interval reconstruction does not establish original-time acquisition.',
                 'Prefix comparison does not alone prove every possible absence of leakage.'],
    sources={str(OUT/p):hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in ['event_group_features_9245.csv','EVENT_GROUP_LEDGER_DETAILS.jsonl','PREFIX_INVARIANCE_CHECK.json']})
with (OUT/'CODEX_CONSTRUCTION_AUDIT.json').open('x',encoding='utf-8') as handle:
    json.dump(receipt,handle,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=True))
