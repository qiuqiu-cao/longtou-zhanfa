"""Read-only retrospective return calculation from archived, pre-entry choices.

This file writes only to its own new results directory. It never registers a
prediction or a fill, alters a ranking, fits a model, or modifies a ledger.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
import csv
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CALENDAR = ROOT / 'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'
END = '2026-10-22'
URL = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'


def now():
    return datetime.now(timezone.utc).isoformat()


def save_new(name, obj):
    path = OUT / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)


def bind(path):
    p = ROOT / path
    return {'path': str(p.resolve()), 'sha256': sha256(p.read_bytes()).hexdigest()}


def one_md(directory, prefix=''):
    paths = sorted((ROOT / directory).glob(prefix + '*.md'))
    if len(paths) != 1:
        raise ValueError(f'Ambiguous source: {directory}: {paths}')
    return str(paths[0].relative_to(ROOT)).replace('\\', '/')


def choices():
    # These selections are fixed before any fresh outcome price requests.
    items = [
        ('2026-09-30', '2026-10-08', '605303', '园林股份',
         'results/review_20260930_forecast_20261008_20261006T082455Z/review.md',
         '2026-10-06T16:30:25.153663+08:00', 'HOLIDAY_PRE_ENTRY_APPENDIX_ORIGINAL_PICK_RETAINED'),
        ('2026-10-08', '2026-10-09', '600825', '新华传媒',
         'results/review_20261008_forecast_20261009_20261008T091618Z/review.md',
         '2026-10-08T17:23:07.626111+08:00', 'SAME_SIGNAL_DATE_REPORT'),
        ('2026-10-09', '2026-10-12', '600241', '时代万恒',
         one_md('results/weekend_review_20261011T142844Z', '复盘'),
         '2026-10-11T14:41:53.594346+00:00', 'SUNDAY_PRE_ENTRY_REPORT_NOT_FRIDAY_NIGHT'),
        ('2026-10-12', '2026-10-13', '605136', '丽人丽妆',
         one_md('results/daily_review_20261012T105005Z', '复盘'),
         '2026-10-12T11:01:02.157921+00:00', 'SAME_SIGNAL_DATE_REPORT'),
        ('2026-10-13', '2026-10-14', '603096', '新经典',
         one_md('results/closing_delivery_20261013T080630Z', '10月'),
         '2026-10-13T08:12:05.081068+00:00', 'SAME_SIGNAL_DATE_REPORT'),
        ('2026-10-14', '2026-10-15', '002343', '慈文传媒',
         one_md('results/daily_review_20261014T1118Z', '10月'),
         None, 'EXPLICIT_MANUAL_WATCH_PRIMARY_NO_VALID_BASE_IN_ORIGINAL_REPORT'),
        ('2026-10-15', '2026-10-16', '002343', '慈文传媒',
         one_md('results/daily_review_20261015T093600Z', '10月15日复盘'),
         '2026-10-15T09:45:38.732057+00:00', 'SAME_SIGNAL_DATE_REPORT'),
        ('2026-10-16', '2026-10-19', None, None, None, None,
         'NO_PRE_ENTRY_NIGHTLY_PICK_FOUND_DO_NOT_BACKFILL_WITH_LATER_RESEARCH'),
        ('2026-10-19', '2026-10-20', '003029', '吉大正元',
         one_md('results/joint_review_20261019T092500Z', '10月19日晚间'),
         '2026-10-19T22:33:00+08:00', 'EVENING_APPENDIX_PRIORITY_UNCHANGED'),
        ('2026-10-20', '2026-10-21', '603278', '大业股份',
         one_md('results/daily_review_20261020T101630Z', '10月20日复盘'),
         None, 'ORIGINAL_COMPLETE_REPORT_TOP1_LATER_PAIRWISE_EAST_ENGINE_PREFERENCE_SEPARATE'),
        ('2026-10-21', '2026-10-22', '002724', '海洋王',
         one_md('results/joint_review_20261021T084800Z', '10月21日'),
         '2026-10-21T09:05:12.369001+00:00', 'JOINT_REPORT_TOP1_EXIT_2026_10_23_PENDING'),
    ]
    sessions = json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions']
    rows = []
    for signal, buy, code, name, source, formed, status in items:
        assert sessions[sessions.index(signal)+1] == buy
        sell = sessions[sessions.index(buy)+1]
        record = dict(signal_date=signal, buy_date=buy, sell_date=sell,
                      code=code, name=name, source=bind(source) if source else None,
                      source_formation_time_as_reported=formed, provenance_status=status,
                      prior_close_decision_only=True, actual_order=False, actual_fill=False)
        rows.append(record)
    return rows


def freeze():
    records = choices()
    model_files = {
        '2026-09-30': 'results/review_20260930_forecast_20261008_20261006T082455Z/full_candidate_scores.csv',
        '2026-10-08': 'results/review_20261008_forecast_20261009_20261008T091618Z/model/base_predictions.json',
        '2026-10-09': 'results/weekend_review_20261011T142844Z/model/base_predictions.json',
        '2026-10-12': 'results/daily_review_20261012T105005Z/model/base_predictions.json',
        '2026-10-13': 'results/daily_review_20261013T074936Z/model/base_predictions.json',
        '2026-10-15': 'results/daily_review_20261015T093600Z/BASE_scores_AFTER_INDEPENDENT_REVIEW.json',
        '2026-10-19': 'results/joint_review_20261019T092500Z/BASE_scores_AFTER_INDEPENDENT.json',
        '2026-10-20': 'results/daily_review_20261020T101630Z/BASE_scores_AFTER_INDEPENDENT.json',
        '2026-10-21': 'results/joint_review_20261021T084800Z/BASE_scores_AFTER_INDEPENDENT.json',
    }
    protocol = {
        'frozen_at_utc': now(), 'asof_market_date': END,
        'month_allocation': 'BUY_DATE_IN_OCTOBER_2026',
        'entry': 'S+1 official trading session opening-auction price, proxied by unadjusted daily open',
        'exit': 'B+1 official trading session open, confirmed by user',
        'return_formula': 'sell_open / buy_open - 1',
        'selection': 'original nightly report summary top1; explicit manual watch primary on Oct14 separately labelled',
        'no_replacement': True,
        'full_slot_denominator': 11,
        'fees_main': 'gross price returns; no commission, tax, slippage or minimum fee deducted',
        'fills': 'hypothetical price proxy only; limit-up entries and limit-down exits flagged, never treated as actual fills',
        'unknowns': 'missing original choices or prices kept unknown, never zero return',
        'portfolio': 'equal-notional trade P&L; geometric chaining is an arithmetic illustration, not an executable same-auction full-capital portfolio',
        'protected_files': 'No production BASE, original forecasts, ledgers, schedules, or goals modified',
        'calendar': bind(str(CALENDAR.relative_to(ROOT))),
        'official_calendar_check': 'https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml',
        'records': records,
        'BASE_comparison_sources': {date: bind(path) for date, path in model_files.items()},
        'BASE_exceptions': {'2026-10-14': 'UNKNOWN_IN_ORIGINAL_REPORT_NOT_BACKFILLED',
                            '2026-10-16': 'NO_PRE_ENTRY_FORECAST_FOUND'},
        'late_pairwise_20': {'code': '600178', 'name': '东安动力',
                            'source': bind(one_md('results/daily_review_20261020T101630Z', '隔夜委托比较')),
                            'formation': '2026-10-20T15:03:33+00:00',
                            'scope': 'Separate late-evening pairwise preference; original full-pool Top1 is not overwritten'},
    }
    save_new('SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json', protocol)
    print(json.dumps({'frozen': True, 'path': str(OUT), 'slots': len(records),
                      'picks': [[r['signal_date'], r['name']] for r in records]}, ensure_ascii=False))


def fetch_all():
    protocol = json.loads((OUT / 'SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json').read_text(encoding='utf-8'))
    codes = sorted({r['code'] for r in protocol['records'] if r['code']} | {'600148', '600178'})
    raw = OUT / 'raw'
    raw.mkdir(exist_ok=False)
    save_new('PRICE_REQUEST_STARTED.json', {'at_utc': now(), 'codes': codes,
              'protocol_sha256': sha256((OUT / 'SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json').read_bytes()).hexdigest()})

    def fetch(code):
        symbol = ('sh' if code.startswith('6') else 'sz') + code
        params = {'param': f'{symbol},day,2026-09-29,{END},80,'}
        item = {'code': code, 'symbol': symbol, 'url': URL, 'params': params, 'requested_at_utc': now()}
        try:
            response = requests.get(URL, params=params, timeout=(5, 12))
            item.update(received_at_utc=now(), http_status=response.status_code,
                        resolved_url=response.url)
            payload = response.content
            path = raw / (code + '.json')
            path.write_bytes(payload)
            item.update(raw_path=str(path), sha256=sha256(payload).hexdigest())
            response.raise_for_status()
            block = response.json().get('data', {}).get(symbol, {})
            bars = block.get('day')
            if not isinstance(bars, list) or not bars:
                raise ValueError('Unadjusted day list missing; adjusted list not substituted')
            item.update(status='UNADJUSTED_DAILY_AVAILABLE', first_date=bars[0][0], last_date=bars[-1][0], rows=len(bars))
        except (requests.RequestException, ValueError, KeyError) as error:
            item.update(status='UNAVAILABLE', error=f'{type(error).__name__}: {str(error)[:280]}')
        return item

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, codes))
    save_new('PRICE_REQUEST_MANIFEST.json', {'formed_at_utc': now(), 'requests': receipts,
             'actual_fills': 0, 'source': 'Tencent unadjusted daily bars; retrospectively received'})
    for r in receipts:
        print(json.dumps(r, ensure_ascii=False))


def load_prices():
    manifest = json.loads((OUT / 'PRICE_REQUEST_MANIFEST.json').read_text(encoding='utf-8'))
    prices, sources = {}, {}
    for receipt in manifest['requests']:
        if receipt['status'] != 'UNADJUSTED_DAILY_AVAILABLE':
            continue
        path = Path(receipt['raw_path'])
        assert sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        prices[receipt['code']] = parse_bars(payload, receipt['code'])
        sources[receipt['code']] = receipt
    # One failed network request is covered by an existing, exact-date raw
    # archive. No retry or replacement of the selection takes place.
    fallback = ROOT / 'data/v293/outcome_review_20261014T1116Z/raw/605136_daily.json'
    prices['605136'] = parse_bars(json.loads(fallback.read_text(encoding='utf-8-sig')), '605136')
    sources['605136'] = {
        'code': '605136', 'status': 'EXISTING_UNADJUSTED_RAW_ARCHIVE_REUSED_AFTER_SINGLE_TIMEOUT',
        **bind(str(fallback.relative_to(ROOT))),
        'original_manifest': bind('data/v293/outcome_review_20261014T1116Z/manifest.json'),
        'fresh_request_failure_preserved': True,
    }
    return prices, sources


def parse_bars(payload, code):
    symbol = ('sh' if code.startswith('6') else 'sz') + code
    rows = payload['data'][symbol]['day']
    bars, previous = {}, None
    for row in rows:
        date = row[0]
        assert previous is None or date > previous
        previous = date
        op, cl, hi, lo, vol = [Decimal(x) for x in row[1:6]]
        assert lo > 0 and lo <= op <= hi and lo <= cl <= hi and vol >= 0
        assert all(x * 100 == (x * 100).to_integral_value() for x in [op, cl, hi, lo])
        bars[date] = dict(open=op, close=cl, high=hi, low=lo, volume_native=vol)
    return bars


def cents(value):
    return value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)


def trade(row, prices):
    item = dict(row)
    code = item.get('code')
    if not code:
        item['result_status'] = 'ORIGINAL_PICK_MISSING_UNKNOWN_NOT_CASH'
        return item
    bars = prices.get(code, {})
    buy = bars.get(item['buy_date'])
    sell = bars.get(item['sell_date'])
    if buy:
        item['entry_open'] = buy['open']
        item['entry_bar'] = buy
        item['entry_raw_json_pointer'] = f'/data/{("sh" if code.startswith("6") else "sz")+code}/day[date={item["buy_date"]}]'
        sessions = json.loads(CALENDAR.read_text(encoding='utf-8-sig'))['sessions']
        prev = sessions[sessions.index(item['buy_date'])-1]
        prior = bars.get(prev)
        if prior:
            upper = cents(prior['close'] * Decimal('1.10'))
            item['prior_close'] = prior['close']
            item['reference_upper'] = upper
            item['entry_open_at_reference_limit_up'] = buy['open'] == upper
            item['entry_whole_day_one_price_limit_up'] = len({buy[k] for k in ['open','close','high','low']}) == 1 and buy['open'] == upper
            item['limit_reference_scope'] = 'normal non-ST mainboard 10% from prior official-session close; reference only'
    if item['sell_date'] > END:
        item.update(result_status='EXIT_SESSION_NOT_YET_OCCURRED', gross_return=None)
        return item
    if not buy or not sell:
        item.update(result_status='PRICE_MISSING_UNKNOWN_NOT_ZERO', gross_return=None)
        return item
    item['exit_open'] = sell['open']
    item['exit_bar'] = sell
    item['exit_raw_json_pointer'] = f'/data/{("sh" if code.startswith("6") else "sz")+code}/day[date={item["sell_date"]}]'
    item['exit_open_at_reference_limit_down'] = sell['open'] == cents(buy['close'] * Decimal('.90'))
    item['gross_return'] = sell['open'] / buy['open'] - 1
    item['result_status'] = 'COMPLETED_PRICE_PROXY_NOT_CONFIRMED_FILL'
    item['fee_and_lot_scenario_10000'] = cost_scenario(buy['open'], sell['open'])
    return item


def cost_scenario(buy, sell):
    budget = Decimal('10000')
    shares = int(budget / (buy * 100)) * 100
    def fees(price, size, is_sell):
        notional = price * size
        return max(Decimal('5'), notional * Decimal('.0003')) + notional * (Decimal('.00001') + (Decimal('.0005') if is_sell else 0))
    while shares > 0 and buy * shares + fees(buy, shares, False) > budget:
        shares -= 100
    if shares <= 0:
        return {'status': 'BUDGET_BELOW_ONE_LOT'}
    paid = buy * shares + fees(buy, shares, False)
    received = sell * shares - fees(sell, shares, True)
    return dict(budget=budget, shares=shares, paid=paid, received=received,
                net_pnl=received-paid, return_on_budget=(received-paid)/budget,
                fees_assumed='commission 0.03% each side/minimum 5; transfer 0.001% each side; stamp 0.05% sell; zero slippage; illustrative only')


def metrics(rows):
    done = [r for r in rows if r.get('gross_return') is not None]
    values = [r['gross_return'] for r in done]
    if not values:
        return dict(completed_price_pairs=0)
    compound, peak, mdd = Decimal(1), Decimal(1), Decimal(0)
    for value in values:
        compound *= 1+value
        peak = max(peak, compound)
        mdd = min(mdd, compound/peak-1)
    positive = sum((v for v in values if v > 0), Decimal(0))
    negative = -sum((v for v in values if v < 0), Decimal(0))
    total = sum(values, Decimal(0))
    return dict(completed_price_pairs=len(done), win_count=sum(v>0 for v in values),
                loss_count=sum(v<0 for v in values), flat_count=sum(v==0 for v in values),
                win_fraction=Decimal(sum(v>0 for v in values))/len(values),
                equal_notional_return_sum=total, average_trade_return=total/len(values),
                equal_notional_pnl_per_10000_per_trade=total*10000,
                arithmetic_geometric_chain=compound-1,
                chain_max_drawdown=mdd,
                profit_factor=positive/negative if negative else None,
                best_trade_return=max(values), worst_trade_return=min(values),
                opened_at_limit_up=sum(r.get('entry_open_at_reference_limit_up',False) for r in done),
                entire_entry_day_one_price_limit_up=sum(r.get('entry_whole_day_one_price_limit_up',False) for r in done),
                exit_open_at_limit_down=sum(r.get('exit_open_at_reference_limit_down',False) for r in done),
                illustrative_fee_lot_pnl_per_10000_budget_per_trade=sum((r['fee_and_lot_scenario_10000'].get('net_pnl',Decimal(0)) for r in done),Decimal(0)))


def serializable(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {k:serializable(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [serializable(v) for v in obj]
    return obj


def base_rows(protocol, prices):
    original = {r['signal_date']:r for r in protocol['records']}
    rows = []
    for signal, source in protocol['BASE_comparison_sources'].items():
        p = Path(source['path'])
        assert sha256(p.read_bytes()).hexdigest() == source['sha256']
        if p.suffix == '.csv':
            candidates = list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))
            first = min(candidates, key=lambda r:int(r['original_order']))
        else:
            data = json.loads(p.read_text(encoding='utf-8-sig'))
            relay = data.get('relay',data)
            candidates = relay.get('candidates',relay.get('rows',[]))
            ranking = relay.get('ranking',[])
            first = next(r for r in candidates if r['code'] == ranking[0]) if ranking else max(candidates,key=lambda r:r['raw_score'])
        record = {**original[signal], 'code':first['code'], 'name':first['name'],
                  'source':source, 'stream':'ARCHIVED_BASE_TOP1_ONLY_NO_REFIT'}
        rows.append(trade(record,prices))
    return rows


def analyze():
    protocol = json.loads((OUT / 'SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json').read_text(encoding='utf-8'))
    for row in protocol['records']:
        if row['source']:
            assert sha256(Path(row['source']['path']).read_bytes()).hexdigest() == row['source']['sha256']
    prices, price_sources = load_prices()
    rows = [trade(r,prices) for r in protocol['records']]
    late = []
    for row in protocol['records']:
        if row['signal_date'] == '2026-10-20':
            row = {**row, **{k:protocol['late_pairwise_20'][k] for k in ['code','name','source']},
                   'provenance_status':'LATE_EVENING_PAIRWISE_PREFERENCE_SEPARATE_COMPARISON'}
        late.append(trade(row,prices))
    models = base_rows(protocol, prices)
    eligible = [r for r in rows if not r.get('entry_open_at_reference_limit_up',True)]
    same_evening = [r for r in rows if r['signal_date'].startswith('2026-10') and r['signal_date'] != '2026-10-09']
    result = dict(formed_at_utc=now(), asof_market_date=END,
                  status='RETROSPECTIVE_PRICE_PROXY_ONLY_NOT_REALIZED_TRADING_PNL',
                  protocol_sha256=sha256((OUT/'SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json').read_bytes()).hexdigest(),
                  rows=rows, original_report_metrics=metrics(rows),
                  october_signal_dates_only_metrics=metrics([r for r in rows if r['signal_date'].startswith('2026-10')]),
                  strict_same_signal_evening_only_metrics=metrics(same_evening),
                  late_evening_pairwise_variant_rows=late, late_evening_pairwise_variant_metrics=metrics(late),
                  BASE_only_rows=models, BASE_only_metrics=metrics(models),
                  open_below_limit_subset_rows=eligible, open_below_limit_subset_metrics=metrics(eligible),
                  price_sources=price_sources,
                  totals=dict(planned_buy_dates=11, recorded_choices=10, missing_original_choice=1,
                              completed_price_pairs=sum(r.get('gross_return') is not None for r in rows),
                              exit_not_yet_occurred=sum(r['result_status']=='EXIT_SESSION_NOT_YET_OCCURRED' for r in rows),
                              actual_order_or_fill_records=0),
                  interpretation='Return sums use equal notional per trade, not account equity. Chaining known rows is arithmetic only; a full same-auction reinvestment portfolio and a complete month are not established.')
    save_new('CALCULATED_RETURNS.json',serializable(result))
    save_new('PROVENANCE_ADDENDUM.json', {
        'formed_at_utc':now(),
        'oct14_completion_receipt':bind('results/daily_review_20261014T1118Z/completion_receipt.json'),
        'oct14_report_completed_at_utc':'2026-10-14T11:34:44.863617+00:00',
        'oct19_evening_receipt':bind('results/joint_review_20261019T092500Z/evening_delivery_receipt.json'),
        'oct19_evening_receipt_formed_at_utc':'2026-10-19T14:35:59.239656+00:00',
        'oct20_original_independent_freeze':bind('results/daily_review_20261020T101630Z/independent_freeze_receipt_BEFORE_MODEL.json'),
        'oct20_original_independent_frozen_at_utc':'2026-10-20T10:28:17.358169+00:00',
        'oct16_pick_search':'No signal_date=2026-10-16 forecast in results/ or data/ JSON/JSONL outside raw/features/notices; root October report directories enumerated. Later research scores do not establish an evening forecast.',
        'raw_prices_fallback':price_sources['605136'],
    })
    make_report(result)
    print(json.dumps(serializable({k:result[k] for k in ['totals','original_report_metrics','late_evening_pairwise_variant_metrics','BASE_only_metrics','open_below_limit_subset_metrics','strict_same_signal_evening_only_metrics']}),ensure_ascii=False,indent=2))


def pct(x):
    return f'{x*100:+.2f}%' if x is not None else '—'


def make_report(result):
    m = result['original_report_metrics']
    l = result['late_evening_pairwise_variant_metrics']
    s = result['open_below_limit_subset_metrics']
    lines = [
        '# 2026年10月首选：次日竞价买、再下一交易日开盘卖', '',
        f'统计截至北京时间2026年10月22日收盘；本次计算完成于{result["formed_at_utc"]}。卖出按开盘价，已由用户确认。', '',
        f'按原复盘摘要首位，已完成{m["completed_price_pairs"]}笔价格配对，{m["win_count"]}盈{m["loss_count"]}亏，胜率{m["win_fraction"]*100:.2f}%。每笔等额本金的收益率合计{pct(m["equal_notional_return_sum"])}，平均每笔{pct(m["average_trade_return"])}；每笔1万元名义本金对应累计毛利{m["equal_notional_pnl_per_10000_per_trade"]:.2f}元。', '',
        '**这是“假设能按两天开盘价成交”的价格收益测算。没有实际成交证据，主表未扣手续费、税和滑点，也不是账户月收益率。**', '',
        '| 原信号/成稿 | 首选 | 竞价买入日 | 买入开盘 | 卖出日 | 卖出开盘 | 毛收益 | 竞价成交限制 |',
        '|---|---|---|---:|---|---:|---:|---|',
    ]
    for r in result['rows']:
        signal = r['signal_date'][5:]
        if r['signal_date']=='2026-09-30':signal='09-30／10-06附页'
        if r['signal_date']=='2026-10-09':signal='10-09／10-11周末成稿'
        restriction = '竞价涨停，买到存疑' if r.get('entry_open_at_reference_limit_up') else '开盘未涨停；仍非实际成交'
        if r.get('entry_whole_day_one_price_limit_up'):restriction='竞价涨停且全天一字，买到存疑'
        if not r['code']:restriction='原首选缺失，保留未知'
        lines.append(f'| {signal} | {r.get("name") or "未找到原首选"} {r.get("code") or ""} | {r["buy_date"][5:]} | {r.get("entry_open", "—")} | {r["sell_date"][5:]} | {r.get("exit_open", "—")} | {pct(r.get("gross_return")) if r["result_status"]!="EXIT_SESSION_NOT_YET_OCCURRED" else "未到卖出日"} | {restriction} |')
    lines += ['',
        f'9笔已完成记录中，{m["opened_at_limit_up"]}笔竞价开在涨停价，其中{m["entire_entry_day_one_price_limit_up"]}笔买入日全天一价。不能把这些账面盈利视为可实现盈利；没有把盘中开板后的买入移作竞价买入。卖出日开盘跌停记录{m["exit_open_at_limit_down"]}笔。', '',
        '## 不同“首选”口径分别统计', '',
        '| 口径 | 完成笔数 | 盈／亏 | 每笔等额本金收益率合计 | 平均单笔 | 每笔1万元名义本金累计毛利 |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for title, key in [('原复盘摘要首位','original_report_metrics'),('20日晚追加偏东安，其余沿原首选','late_evening_pairwise_variant_metrics'),('只取原BASE原始第一名','BASE_only_metrics'),('只计开盘未涨停的已知子集','open_below_limit_subset_metrics'),('仅10月信号日，含11日周末稿','october_signal_dates_only_metrics'),('严格同一信号日晚间成稿，剔除长假／周末稿','strict_same_signal_evening_only_metrics')]:
        q=result[key]
        lines.append(f'| {title} | {q["completed_price_pairs"]} | {q["win_count"]}／{q["loss_count"]} | {pct(q["equal_notional_return_sum"])} | {pct(q["average_trade_return"])} | {q["equal_notional_pnl_per_10000_per_trade"]:.2f}元 |')
    lines += ['',
        f'20日原完整报告第一名为大业，21日13.19买、22日14.50卖，{pct(next(r for r in result["rows"] if r["signal_date"]=="2026-10-20")["gross_return"])}。20日23:03北京时间用户追问后的成对追加略偏东安：21日13.65买、22日12.90卖，{pct(next(r for r in result["late_evening_pairwise_variant_rows"] if r["signal_date"]=="2026-10-20")["gross_return"])}。两份原记录保持，原完整排名和预测不回写。', '',
        '19日独立摘要第一名为吉大，原BASE第一名为长春一东。BASE序列保留这个差别；14日原报告没有有效BASE输出，故BASE口径不补算后来研究分数。原复盘口径14日用明确的人工交易观察首选慈文，这不等于原策略已经确认可买。', '',
        '10月归属按买入日。8日买园林来自9月30日信号、国庆前瞻附页；12日买时代来自10月11日晚周末稿，不能称9日晚已经给出。严格同日晚上成稿的单独子集见上表。未找到16日晚首选，不能以研究重放或后来赢家填补。21日晚海洋王对应22日11.21买、23日卖，23日未发生，不纳入已完成收益。10月尚未结束，也不把未来日期算零收益。', '',
        f'开盘未涨停的已知子集4笔仍不保证成交；其收益率合计{pct(s["equal_notional_return_sum"])}、平均{pct(s["average_trade_return"])}。这个成交约束分层包含亏损股，不从盈利股中挑可买样本，也不递补其他股票。', '',
        '## 资金、费用与计算边界', '',
        '主统计公式为卖出日开盘价÷买入日开盘价−1。等额本金收益率合计用于累加各笔同额投入的盈亏，不能直接称账户月回报。用户未给账户本金、每日仓位和可用资金，因而不报告真实账户收益率。', '',
        f'把这9个已知收益纯数学连乘得到{pct(m["arithmetic_geometric_chain"])}，最大回撤{pct(m["chain_max_drawdown"])}，盈利因子Profit Factor={m["profit_factor"]:.3f}。该连乘跳过了未知记录，并假设每笔资金完全复用；每日09:25既卖旧仓又买新仓时，卖出资金能否赶上同次竞价买入未建立，因此只作算式展示，不作为可执行满仓复利月收益。', '',
        f'费用示例：每笔预留1万元、100股整数手、双边佣金万3且每笔最低5元、双边过户万0.1、卖出印花万5、滑点0，原9笔理论成交的累计净盈亏约{m["illustrative_fee_lot_pnl_per_10000_budget_per_trade"]:.2f}元。费用为本次假设，未核用户券商；仍假设能买到竞价涨停股，不认证可交易净收益。逐笔数量及费用算式见CALCULATED_RETURNS.json。', '',
        '## 来源与复核', '',
        '首选先从原报告冻结，再请求行情；冻结名单、协议、原报告和BASE输出哈希见[冻结名单](SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json)。原报告只读，未修改生产BASE、预测、账本、定时任务或平台目标。', '',
        '行情为腾讯不复权day数组，字段顺序日期／开盘／收盘／最高／最低／成交量；使用真实请求、收件时钟与原始响应SHA。10只本次成功，丽人一次超时，复用原14日准确日期原始响应，失败原件保留，未循环重试。日线开盘价用作09:25竞价成交价代理，不认证个人委托成交、排队位置或实际撮合量。', '',
        '官方交易日使用项目已存扩展日历，并在线核对[上交所2026年中秋、国庆休市公告](https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml)：10月1日至7日休市，8日起开市；周末顺延至下一官方交易日。', '',
        '全部逐笔结果与价格来源见[计算明细](CALCULATED_RETURNS.json)，请求时钟、URL及哈希见[行情收据](PRICE_REQUEST_MANIFEST.json)。补充来源与成稿时间见[来源追加](PROVENANCE_ADDENDUM.json)。原数据缺口和成交不确定性保留，本统计不构成新增预测或实际成交记录。', '',
    ]
    # Make all workspace links usable in the desktop, including links to the
    # exact archived report supporting each historical selection.
    lines += ['### 原始首选逐日出处','']
    for row in result['rows']:
        if row['source']:
            lines.append(f'- {row["signal_date"]} {row["name"]}：[原报告]({row["source"]["path"].replace(chr(92),"/")})。')
    text='\n'.join(lines)
    for filename in ['SELECTIONS_AND_PROTOCOL_BEFORE_PRICE_REQUESTS.json','CALCULATED_RETURNS.json','PRICE_REQUEST_MANIFEST.json','PROVENANCE_ADDENDUM.json']:
        text=text.replace(']('+filename+')',']('+str(OUT/filename).replace('\\','/')+')')
    with (OUT/'10月份首选竞价买入隔日开盘卖出统计.md').open('x',encoding='utf-8') as stream:
        stream.write(text)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    {'freeze': freeze, 'fetch': fetch_all, 'analyze': analyze}[sys.argv[1]]()
