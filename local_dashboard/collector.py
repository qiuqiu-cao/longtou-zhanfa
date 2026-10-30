"""Bounded free public-data collector. Appends evidence, never scores a model."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
import hashlib,json,math,re,sys,time
from zoneinfo import ZoneInfo
import requests
import data

ROOT=data.ROOT
sys.path.insert(0,str(ROOT))
import public_outcome_sources_v293 as pools
import public_live_capture_v1 as quotes

SH=ZoneInfo('Asia/Shanghai')
HOME=Path(__file__).resolve().parent/'.runtime'/'public_updates'
CALENDAR=ROOT/'data/v298/exchange_calendar_extension_20261003_v2/calendar.json'

def clock():return datetime.now(timezone.utc)
def iso():return clock().isoformat()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)
def new_json(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def session_context(at=None):
    at=(at or clock()).astimezone(SH);day=at.date().isoformat();cal=data.read(CALENDAR)
    sessions=cal['sessions'];start,end=cal['scope']
    if not start<=day<=end:return dict(today=day,state='CALENDAR_OUT_OF_SCOPE',target_day=None,active=False,next_session=None)
    trade=day in sessions;t=at.strftime('%H:%M:%S')
    prior=[d for d in sessions if d<day];future=[d for d in sessions if d>day]
    active=trade and ('09:20:00'<=t<='11:35:00' or '12:55:00'<=t<='15:10:00')
    target=day if trade and t>='09:20:00' else prior[-1] if prior else None
    state='TRADING_WINDOW' if active else 'SESSION_BREAK' if trade and '11:35:00'<t<'12:55:00' else 'AFTER_CLOSE' if trade and t>'15:10:00' else 'BEFORE_OPEN' if trade else 'NON_TRADING_DAY'
    return dict(today=day,target_day=target,state=state,active=active,
        next_session=day if trade and t<'09:20:00' else future[0] if future else None,
        calendar_source=cal['source'],calendar_sha256=data.digest(CALENDAR),shanghai_now=at.isoformat())

def capture_pool(kind,day,folder):
    url=pools.EM_BASE+pools.ENDPOINTS[kind]
    params=dict(ut='7eea3edcaed734bea9cbfc24409ed989',dpt='wz.ztzt',Pageindex='0',pagesize='10000',
        sort='fbt:asc' if kind in ['limit_up','failed_board'] else 'fund:asc',date=day.replace('-',''))
    rec=dict(kind=kind,source='东方财富公开涨跌停池',url=url,parameters=params,requested_at=iso(),attempts=1,requested_day=day)
    try:
        r=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0'},timeout=(4,7));raw=folder/(kind+'.response');raw.write_bytes(r.content)
        rec.update(received_at=iso(),http_status=r.status_code,raw_path=str(raw),sha256=data.digest(raw));r.raise_for_status()
        payload=r.json();native=(payload.get('data') or {});rec['source_day']=str(native.get('qdate'))
        parsed=pools.parse_pool(payload,day,kind)
        rec.update(status='EXACT_DATE_PROVIDER_POOL',source_time_precision='DATE_ONLY_NO_INTRADAY_CLOCK',
            provider_total=parsed['all_market_raw_count'],mainboard_rows=len(parsed['rows']))
        bycode={v['c']:v for v in native['pool']}
        rows=[dict(x,kind=kind,reported_float_cap_cny=bycode[x['code']].get('ltsz'),
            source_day=day,source_status='PUBLIC_POOL_NOT_DAILY_OR_EXECUTION_VERIFIED',
            first_seal_status='UNVERIFIED_PROVIDER_FIELD_PAST_CONFLICTS_RETAINED',
            eligible_for_model=False) for x in parsed['rows']]
        return rec,rows
    except Exception as e:
        rec.update(status='UNAVAILABLE_OR_DATE_REJECTED',completed_at=iso(),error=str(e)[:260]);return rec,[]

def parse_quotes(raw,expected_day,at=None):
    at=at or clock();text=raw.decode('gb18030',errors='replace');rows=[]
    for match in quotes.LINE_PATTERN.finditer(text):
        symbol=match.group('symbol');frame=quotes.parse_sina_response(match.group(0))
        if frame.empty:continue
        x=frame.iloc[0];stamp=x['quote_timestamp']
        if not stamp:continue
        observed=datetime.fromisoformat(stamp);last=float(x['last']);previous=float(x['previous_close'])
        if not math.isfinite(last) or not math.isfinite(previous) or last<=0 or previous<=0:continue
        high=float(x['high']);low=float(x['low'])
        if not all(math.isfinite(v) for v in [high,low]) or not 0<low<=last<=high:continue
        context=session_context(at)
        if observed.date().isoformat()!=expected_day:status='STALE_OTHER_SESSION'
        elif observed>at.astimezone(SH)+timedelta(seconds=30):status='INVALID_FUTURE_TIMESTAMP'
        elif context['active']:status=quotes.quote_freshness(stamp,at)[0]
        else:status='EXACT_SESSION_CLOSED_QUOTE_NOT_LIVE'
        rows.append(dict(symbol=symbol,code=symbol[-6:],name=str(x['provider_name']),last=last,
            previous_close=previous,change_pct=(last/previous-1)*100,
            quote_time=stamp,fetched_at=at.isoformat(),status=status,expected_day=expected_day,
            usable_for_current_session=status not in ['STALE_OTHER_SESSION','INVALID_FUTURE_TIMESTAMP'],
            live_order_flow_verified=False))
    return rows

def capture_quotes(symbols,day,folder):
    url=quotes.SINA_URL.format(symbols=','.join(symbols));rec=dict(kind='quote',source='新浪公开Level-1',url=url,requested_at=iso(),attempts=1,requested_symbols=symbols,requested_day=day)
    try:
        r=requests.get(url,headers=quotes.SINA_HEADERS,timeout=(4,7));raw=folder/'quotes.response';raw.write_bytes(r.content)
        at=clock();rec.update(received_at=at.isoformat(),http_status=r.status_code,raw_path=str(raw),sha256=data.digest(raw));r.raise_for_status()
        rows=parse_quotes(r.content,day,at);seen={r['symbol'] for r in rows};missing=sorted(set(symbols)-seen)
        valid=[r for r in rows if r['usable_for_current_session']]
        if not rows:raise ValueError('公开报价未返回可解析字段')
        rec.update(status='QUOTES_RECEIVED' if len(valid)==len(symbols) else 'PARTIAL_OR_STALE_QUOTES',
            returned=len(rows),current_session=len(valid),missing_symbols=missing)
        return rec,rows
    except Exception as e:
        rec.update(status='UNAVAILABLE_OR_INVALID',completed_at=iso(),error=str(e)[:260]);return rec,[]

class Collector:
    def __init__(self):
        self.stop_event=Event();self.lock=Lock();self.thread=None;self.last_run=0
        self.last=data.read(HOME/'latest.json',{})
        self.running=False;self.last_error=None
        self.close_attempted_day=None
    def start(self):
        self.thread=Thread(target=self.loop,name='public-archive-collector',daemon=True);self.thread.start()
    def stop(self):self.stop_event.set()
    def status(self):
        with self.lock:snapshot=dict(self.last);running=self.running;error=self.last_error
        context=session_context()
        # No previous-day packet may masquerade as a failed current-day attempt.
        mismatch=bool(snapshot and snapshot.get('requested_day')!=context['target_day'])
        if mismatch:snapshot=dict(status='CURRENT_SESSION_NOT_CAPTURED',requested_day=context['target_day'],
            pools=[],quotes=[],requests=[],previous_attempt_at=snapshot.get('completed_at'),
            previous_attempt_day=snapshot.get('requested_day'))
        return dict(enabled=True,running=running,last_worker_error=error,context=context,
            policy='服务器运行期间：交易时段每120秒一次；连续失败退避到300秒；午休/非交易日不循环请求；启动及交易日盘后各补采一次。',
            archive_review_date=data.catalog()['latest'],review_status='当日独立研判未自动生成；有新合格归档才更新模型及人工意见',
            data=snapshot,served_at=iso(),orders=0,model_runs=0,paid_calls=0)
    def collect(self):
        with self.lock:
            if self.running:return
            self.running=True
        folder=None
        try:
            context=session_context();day=context['target_day']
            if context['state']=='AFTER_CLOSE':self.close_attempted_day=day
            if not day:raise ValueError('交易日历不覆盖当前日期，不推测交易日')
            folder=HOME/'by_trade_day'/day/clock().strftime('%Y%m%dT%H%M%S_%fZ');folder.mkdir(parents=True,exist_ok=False)
            with ThreadPoolExecutor(max_workers=3) as pool:
                got=list(pool.map(lambda k:capture_pool(k,day,folder),['limit_up','failed_board','limit_down']))
            requests_list=[r for r,_ in got];poolrows=[x for _,rows in got for x in rows]
            codes={x['code'] for x in poolrows}
            latest=data.catalog()['latest']
            if latest:
                old=data.read(data.archives()[latest]/'facts/complete_enriched_facts_SCORES_MASKED.json')
                codes|={r['code'] for r in old['rows'] if r['kind']=='limit_up'}
            symbols=['sh000001','sz399001','sz399006']+[quotes.sina_symbol(c) for c in sorted(codes)]
            qr,prices=capture_quotes(symbols,day,folder);requests_list.append(qr)
            good=sum(r['status']=='EXACT_DATE_PROVIDER_POOL' for r in requests_list)
            status='SUCCESS_PUBLIC_DATA_ONLY' if good==3 and qr['status']=='QUOTES_RECEIVED' else 'PARTIAL_PUBLIC_DATA' if good or prices else 'FAILED'
            packet=dict(requested_day=day,local_observation_day=context['today'],context=context,status=status,
                completed_at=iso(),pools=poolrows,quotes=prices,requests=requests_list,
                actual_market_clock='池仅有交易日期；报价各行带行情时钟。抓取时钟不能替代行情时钟。',
                is_live=False,live_claim='公开来源时间可验证，但不保证实时延迟、完整盘口或交易所认证。',
                model_eligible=False,review_generated=False,new_fits=0,orders=0,paid_calls=0)
            new_json(folder/'receipt.json',packet)
            objective=dict(formed_at=iso(),trade_day=day,phase=context['state'],
                title='自动客观盘面快照；不是独立研判或模型预测',
                source_receipt=str(folder/'receipt.json'),source_sha256=data.digest(folder/'receipt.json'),
                counts={kind:len(rows) if rec['status']=='EXACT_DATE_PROVIDER_POOL' else None for kind,(rec,rows) in zip(['limit_up','failed_board','limit_down'],got)},
                mainboard_ladder=[dict(code=r['code'],name=r['name'],boards=r['boards']) for r in poolrows if r['kind']=='limit_up' and r.get('boards',0)>1],
                original_model_scores=None,independent_top3=None,auction_permissions=None,
                next_steps='完整价格/风险/模型输入、独立题材与关系审阅及冻结通过后才形成新预测。')
            new_json(folder/'客观盘面快照.json',objective)
            atomic(HOME/'latest.json',packet)
            with self.lock:self.last=packet;self.last_error=None
        except Exception as e:
            failed=dict(status='FAILED',requested_day=session_context()['target_day'],completed_at=iso(),error=str(e)[:300],pools=[],quotes=[],requests=[])
            if folder:new_json(folder/'worker_failure.json',failed)
            atomic(HOME/'latest.json',failed)
            with self.lock:self.last=failed;self.last_error=str(e)[:300]
        finally:
            with self.lock:self.running=False
            self.last_run=time.monotonic()
    def loop(self):
        self.collect() # one bounded bootstrap refresh, even on a weekend
        while not self.stop_event.wait(15):
            ctx=session_context();elapsed=time.monotonic()-self.last_run
            with self.lock:last=dict(self.last)
            failed=last.get('status') in ['FAILED','PARTIAL_PUBLIC_DATA']
            due=ctx['active'] and elapsed>=(300 if failed else 120)
            afterclose=ctx['state']=='AFTER_CLOSE' and self.close_attempted_day!=ctx['target_day']
            if due or afterclose:self.collect()

if __name__=='__main__':
    c=Collector();c.collect();s=c.status();print(json.dumps(dict(context=s['context'],status=s['data'].get('status'),pools=len(s['data'].get('pools',[])),quotes=len(s['data'].get('quotes',[])),requests=[{k:r.get(k) for k in ['source','status','source_day','requested_day','error']} for r in s['data'].get('requests',[])]),ensure_ascii=False))
