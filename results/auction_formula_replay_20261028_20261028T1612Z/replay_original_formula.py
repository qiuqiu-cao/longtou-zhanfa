"""Retrospective replay of the exported primary formula, no training or orders."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import re
import shutil

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
PREV = ROOT / 'results/daily_review_20261027T0928Z/facts'
TODAY = ROOT / 'results/daily_review_20261028T1020Z/facts'
DATE = '2026-10-28'
FORMULA = ROOT / 'templates/同花顺公式_龙头竞价非一字选股_v2026.txt'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def price(value):
    return float(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))

def conjunction(values):
    if any(x is False for x in values):
        return False
    return None if any(x is None for x in values) else True

previous_facts = load(PREV / 'RESOLVED_FACTS_v2_SCORES_MASKED.json')
previous_daily = load(PREV / 'normalized_daily.json')
today_daily = load(TODAY / 'normalized_daily.json')
previous_pool = [r for r in previous_facts['rows'] if r['kind'] == 'limit_up' and r.get('scope') is True]
assert len({r['code'] for r in previous_pool}) == len(previous_pool)

quote_text = (OUT / 'raw/quotes_current_crosscheck.response').read_bytes().decode('gb18030', errors='replace')
quotes = {}
for symbol, body in re.findall(r'v_(\w+)="([^"]*)"', quote_text):
    fields = body.split('~')
    if len(fields) > 48:
        quotes[fields[2]] = dict(name=fields[1], close=float(fields[3]), previous_close=float(fields[4]),
            open=float(fields[5]), observed_at=fields[30], high=float(fields[33]), low=float(fields[34]),
            upper=float(fields[47]), lower=float(fields[48]))

choice = OUT / 'raw/xiangyang_0926_choice.response'
choice_text = BeautifulSoup(choice.read_bytes(), 'html.parser').get_text(' ', strip=True)
assert '北京时间10月28日09:26分' in choice_text
assert '09:25:00 16.96 57228' in choice_text
assert '成交9705.87万元' in choice_text
(OUT / 'raw/xiangyang_0926_choice_extracted.txt').write_text(choice_text, encoding='utf-8')
auction = {'000678': dict(price=16.96, volume_lots=57228, amount_cny_displayed=97058700,
    amount_cny_price_times_volume=97058688,
    source_url='https://finance.eastmoney.com/a/202610283911131982.html',
    source_published_at='2026-10-28T09:26:00+08:00', event_time='2026-10-28T09:25:00+08:00',
    raw_sha256=digest(choice), precision='Article amount rounded to 0.01 ten-thousand CNY; price-times-volume crosschecks archived Tencent first point, not assumed from its 09:30 timestamp alone.')}

rows = []
for r in previous_pool:
    code = r['code']
    pb = previous_daily.get(code, {}).get('days', {})
    yesterday = pb.get('2026-10-27')
    before_yesterday = pb.get('2026-10-26')
    tb = today_daily.get(code, {}).get('days', {}).get(DATE)
    qq = quotes.get(code, {})
    open_price = tb['open'] / 100 if tb else qq.get('open')
    previous_close = yesterday['close'] / 100 if yesterday else qq.get('previous_close')
    yesterday_high = yesterday['high'] / 100 if yesterday else r.get('high')
    previous_amount = r.get('amount_cny') or r.get('primary_amount_cny')
    previous_turnover = r.get('turnover_pct')
    is20 = code.startswith(('300', '301', '688'))
    upper = price(previous_close * (1.20 if is20 else 1.10)) if previous_close else None
    previous_test_price = price(before_yesterday['close'] / 100 * (1.195 if is20 else 1.095)) if before_yesterday else None
    a = auction.get(code)
    ratio = a['amount_cny_displayed'] / previous_amount * 100 if a and previous_amount else None
    gap = (open_price / previous_close - 1) * 100 if open_price and previous_close else None
    check = dict(
        NOT_ST=not ('ST' in r['name'].upper() or r['name'].startswith('退')),
        YESTERDAY_ZT=(yesterday['close'] / 100 >= previous_test_price - .01 and yesterday['close'] == yesterday['high']) if yesterday and previous_test_price else None,
        NOT_ONE_WORD=(open_price < upper - .01) if open_price and upper else None,
        GOLDEN_GAP=(2 - 1e-9 <= gap <= 7 + 1e-9) if gap is not None else None,
        EXPLOSIVE_VOL=(ratio >= 5) if ratio is not None else None,
        BREAK_HIGH=(open_price >= yesterday_high) if open_price and yesterday_high else None,
        HEALTHY_CAPACITY=(previous_amount >= 300000000 and 8 <= previous_turnover <= 32) if previous_amount is not None and previous_turnover is not None else None)
    # OR volume ratio branch remains unknown where amount is absent; a true amount branch suffices.
    status = conjunction(check.values())
    non_auction = conjunction(v for k, v in check.items() if k != 'EXPLOSIVE_VOL')
    reasons = [k for k, v in check.items() if v is False]
    unknown = [k for k, v in check.items() if v is None]
    rows.append(dict(code=code, name=r['name'], previous_boards=r.get('boards'),
        previous_close=previous_close, previous_high=yesterday_high, previous_amount_cny=previous_amount,
        previous_turnover_pct=previous_turnover, open=open_price, gap_pct=gap, upper_reference=upper,
        auction_amount_cny= a['amount_cny_displayed'] if a else None, auction_amount_ratio_pct=ratio,
        original_formula_signal=status, static_conditions_signal=non_auction,
        reject_reasons=reasons, unknown_conditions=unknown, checks=check))

# Selection is complete before reading today's high/close as outcomes.
selected = [r['code'] for r in rows if r['original_formula_signal'] is True]
assert selected == ['000678'], selected
for row in rows:
    code = row['code']
    tb = today_daily.get(code, {}).get('days', {}).get(DATE)
    q = quotes.get(code, {})
    close = tb['close'] / 100 if tb else q.get('close')
    high = tb['high'] / 100 if tb else q.get('high')
    upper = row['upper_reference']
    row.update(close=close, high=high, closed_at_limit=(abs(close - upper) < .005) if close and upper else None,
        touched_limit=(high >= upper - .005) if high and upper else None,
        close_change_pct=(close / row['previous_close'] - 1) * 100 if close and row['previous_close'] else None)

def save_table(path, records):
    flattened = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in records]
    pd.DataFrame(flattened).to_csv(path, index=False, encoding='utf-8-sig')

save_table(OUT / 'complete_38_prior_limit_up_formula_replay.csv', rows)
save_table(OUT / 'confirmed_formula_matches.csv', [r for r in rows if r['original_formula_signal'] is True])
save_table(OUT / 'all_eight_continuing_limit_up_with_misses.csv', [r for r in rows if r['closed_at_limit'] is True])
snapshot = OUT / 'source_snapshot'
assert not snapshot.exists()
snapshot.mkdir()
sources = [FORMULA, PREV / 'RESOLVED_FACTS_v2_SCORES_MASKED.json', PREV / 'normalized_daily.json',
    TODAY / 'normalized_daily.json', TODAY / 'RESOLVED_FACTS_v2_SCORES_MASKED.json',
    ROOT / 'results/daily_review_20261028T1020Z/minutes/sz000678.response']
bindings = []
for source in sources:
    target = snapshot / source.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    bindings.append(dict(path=str(source), sha256=digest(source)))

result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), trade_date=DATE,
    run_type='RETROSPECTIVE_PYTHON_EQUIVALENT_PRIMARY_EXPORTED_FORMULA_NOT_THS_CLIENT_RUN_NOT_NEW_BLIND_PREDICTION',
    formula_sha256=digest(FORMULA), formula_condition_order=list(rows[0]['checks']),
    fits=0, actual_orders=0, prior_pool_rows=len(rows),
    definitive_matches=len(selected), definitive_non_matches=sum(r['original_formula_signal'] is False for r in rows),
    unknown_final_signals=sum(r['original_formula_signal'] is None for r in rows),
    static_pass_rows=sum(r['static_conditions_signal'] is True for r in rows),
    actual_continuing_limit_ups=sum(r['closed_at_limit'] is True for r in rows),
    matched_continuing_limit_ups=sum(r['closed_at_limit'] is True and r['original_formula_signal'] is True for r in rows),
    missing_outcomes=sum(r['closed_at_limit'] is None for r in rows),
    matches=[r for r in rows if r['original_formula_signal'] is True], auction_evidence=auction,
    provenance=bindings,
    limitations=[
        'Universe is the complete returned and reconciled archived mainboard prior-limit pool; exchange-wide completeness and historical security status are not certified.',
        'Python equivalent uses archived prior turnover percentage as the software REF(VOL,1)/CAPITAL*100 observation; the client CAPITAL field and compiled result were not independently run.',
        'Most excluded stocks have no verified auction amount/ratio, but at least one known false AND condition makes their final primary-formula signal false without inventing auction fields.',
        'Article was retrieved after market close. Its 09:25 facts corroborate archived data but are not an originally frozen 09:25 forecast.',
        'Daily price equality verifies close-at-limit price only, not fills or complete live order-flow. No claim of executable open-price return.',
        'This evaluates the primary relay formula only; it does not assert an all-market first-board formula scan or a trained-model probability.',
        'One matched and successful stock on one known day does not estimate future accuracy. Seven other same-pool continuing limit ups were missed.'])
(OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['provenance']}, ensure_ascii=False, indent=2))
