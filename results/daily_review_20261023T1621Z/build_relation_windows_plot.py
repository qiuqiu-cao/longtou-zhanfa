from pathlib import Path
from datetime import datetime, timezone,timedelta
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
OUT=Path(__file__).resolve().parent/'minutes';DAY='2026-10-23'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_paths=[OUT/'capture_receipt.json',OUT/'additional_capture_receipt.json'];records=[]
for p in source_paths:records.extend(read(p)['records'])
records=[r for r in records if r.get('source')!='EASTMONEY_MINUTE_OHLC' and r['status']!='UNAVAILABLE']
by_symbol={r['symbol']:r for r in records}
assert len(records)==len(by_symbol)
SYMBOLS=['sh000001','sz399006','sh603278','sz002724','sh603089','sh603677','sz002009','sh600215','sh603081','sh600825','sz002343','sz002242','sz002696','sz002185','sz002156']
def dt(t):return datetime.strptime(DAY+t,'%Y-%m-%d%H%M')
def in_session(t):return '0930'<=t<='1130' or '1300'<=t<='1500'
def same_session(a,b):return ('0930'<=a<='1130' and '0930'<=b<='1130') or ('1300'<=a<='1500' and '1300'<=b<='1500')
EVENTS=[dict(time='1020',name='沪指与创业板当日最低分钟价',event_type='EX_POST_OBSERVED_MINUTE_POINT_LOW',qualification='Both index lowest sampled minute points; not a contemporaneous forecast of reversal'),dict(time='1310',name='大业首次分钟收于涨停',event_type='MINUTE_POINT_FIRST_LIMIT_CLOSE',provider_second_timestamp='13:09:45',qualification='Minute bin 13:10; provider second-level stamp separately attributed'),dict(time='1426',name='海洋王首次分钟触板并收于涨停',event_type='CROSSCHECKED_MINUTE_OHLC_HIGH_AND_CLOSE',qualification='Eastmoney OHLC and Tencent point both 14:26; reject conflicting 09:53:51 pool fbt')]
result=[]
for event in EVENTS:
 center=event['time'];rowset=[]
 for symbol in SYMBOLS:
  r=by_symbol[symbol];pts={x['time']:x for x in r['rows']};samples={};base=pts.get(center)
  for offset in [-5,-3,-1,0,1,3,5]:
   time=(dt(center)+timedelta(minutes=offset)).strftime('%H%M');ok=in_session(time) and same_session(center,time);point=pts.get(time) if ok else None
   samples[str(offset)]=dict(offset_clock_minutes=offset,time=time,valid_same_trading_session=ok,raw_price=point['price'] if point else None,day_return_pct=(point['price']/r['previous_close']-1)*100 if point else None,cumulative_volume_native=point['cum_volume'] if point else None,cumulative_amount_cny=point.get('cum_amount') if point else None)
  windows={}
  for h in [1,3,5]:
   lo=samples[str(-h)]['raw_price'];hi=samples[str(h)]['raw_price'];mid=base['price'] if base else None
   windows[str(h)]=dict(from_time=samples[str(-h)]['time'],event_time=center,to_time=samples[str(h)]['time'],pre_event_return_pct=(mid/lo-1)*100 if mid and lo else None,post_event_return_pct=(hi/mid-1)*100 if hi and mid else None,symmetric_window_return_pct=(hi/lo-1)*100 if hi and lo else None)
  rowset.append(dict(symbol=symbol,name=r['name'],source='TENCENT_MINUTE_POINTS',quote_date=r['source_date'],previous_close=r['previous_close'],source_raw_path=r['path'],source_raw_sha256=r['sha256'],source_requested_at=r['requested_at'],source_received_at=r['received_at'],samples=samples,windows=windows,close_price=r['last_quote'],close_return_pct=(r['last_quote']/r['previous_close']-1)*100))
 result.append(dict(event=event,stocks=rowset))
obj=dict(formed_at=now(),signal_date=DAY,clock_timezone='Asia/Shanghai',minute_timestamp_semantics='Provider minute timestamp; not exchange-certified trade-by-trade time',price_basis='Raw Tencent prices, day return relative to each own previous close; no interpolation across lunch or missing minutes',events=result,source_receipt_bindings={str(p):sha(p) for p in source_paths},interpretation_limit='Timed co-movement and counterexamples only; no account-level capital migration or causal attribution',network_requests=0,models_seen_or_loaded=False)
if (OUT/'RELATION_WINDOWS.json').exists():
 old=read(OUT/'RELATION_WINDOWS.json');assert old['events']==obj['events'];obj=old
else:
 with (OUT/'RELATION_WINDOWS.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
# Plot exact provider minute points on real civil clock. Lunch is shaded and lines segmented.
fonts=[Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/simhei.ttf')]
font=next(x for x in fonts if x.exists());font_manager.fontManager.addfont(str(font));prop=font_manager.FontProperties(fname=str(font));plt.rcParams['font.family']=prop.get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(3,1,figsize=(15,14),sharex=True,constrained_layout=True)
fig.suptitle('2026年10月23日：指数修复、高低位接力与旧情绪反馈',fontsize=19,fontweight='bold')
groups=[('指数、半导体与高位锚：修复不是所有方向同步', ['sh000001','sz399006','sz002185','sz002156','sh603278','sz002724']),('机器人高低位：早封者与午后修复的先后', ['sh603278','sh603089','sh603677','sz002009','sh600215','sh603081']),('海洋王与旧情绪股：共振、滞后及未同步响应', ['sz002724','sh600825','sz002343','sz002242','sz002696'])]
palette=['#1f4e79','#19a974','#d97a00','#8952b0','#d94841','#5a6b73']
for ax,(title,symbols) in zip(axes,groups):
 for idx,symbol in enumerate(symbols):
  r=by_symbol[symbol]
  for j,(lo,hi) in enumerate([('0930','1130'),('1300','1500')]):
   pts=[x for x in r['rows'] if lo<=x['time']<=hi]
   ax.plot([dt(x['time']) for x in pts],[(x['price']/r['previous_close']-1)*100 for x in pts],label=r['name'] if j==0 else None,color=palette[idx],linewidth=1.7,alpha=.95)
 ax.axhline(0,color='#889099',linewidth=.8,alpha=.8)
 ax.axvspan(dt('1130'),dt('1300'),color='#eceff1',alpha=.85)
 for event in EVENTS:ax.axvline(dt(event['time']),color='#222',linestyle=':',linewidth=1,alpha=.7)
 ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=23);ax.set_ylabel('较昨收（%）');ax.grid(axis='y',alpha=.16);ax.legend(loc='upper left',ncol=3,fontsize=10,framealpha=.95)
 ax.text(dt('1215'),.02,'午休：不连线、不插值',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=9,color='#78828c')
 ax.spines[['top','right']].set_visible(False)
axes[0].text(dt('1020'),1.01,'10:20 指数最低分钟点',transform=axes[0].get_xaxis_transform(),ha='center',fontsize=9)
axes[0].text(dt('1310'),1.01,'13:10 大业封板分钟',transform=axes[0].get_xaxis_transform(),ha='center',fontsize=9)
axes[0].text(dt('1426'),1.01,'14:26 海洋封板分钟',transform=axes[0].get_xaxis_transform(),ha='center',fontsize=9)
ticks=['0930','1000','1020','1100','1130','1300','1330','1400','1426','1500']
axes[-1].set_xticks([dt(t) for t in ticks]);axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));axes[-1].set_xlabel('上海时间；同源原价计算，观察到的先后与相关性不证明同一资金迁移')
axes[-1].set_xlim(dt('0930'),dt('1500'))
fig.savefig(OUT/'情绪关系分时图.png',dpi=170,facecolor='white')
plt.close(fig)
# Compact evidence-first note with all requested comparators and three exact +/-5 windows.
lines=['# 10月23日情绪关系：三个节点的分钟核对','',f'形成于 {obj["formed_at"]}；仅使用本批已保存原文，未新增请求、未查看模型。图中午休断开；窗口均为上海时间连续交易时段的前后1/3/5个自然分钟，不跨午休。','', '下表为节点前5分钟至后5分钟的原价区间变化，不是个股全天涨跌，也不是资金流量。','', '| 对象 | 10:15→10:25 | 13:05→13:15 | 14:21→14:31 | 收盘较昨收 |','|---|---:|---:|---:|---:|']
for symbol in SYMBOLS:
 xs=[next(x for x in event['stocks'] if x['symbol']==symbol) for event in result]
 lines.append('| '+xs[0]['name']+' | '+' | '.join(f'{x["windows"]["5"]["symmetric_window_return_pct"]:+.2f}%' for x in xs)+' | '+f'{xs[-1]["close_return_pct"]:+.2f}%'+' |')
lines += ['', '## 可以支持的判断与反证','', '**10:20首先是指数拐点，不能概括成高位龙头一致带盘。** 沪指与创业板同为当日最低分钟点，随后5分钟回升；大业同期回落，海洋王仍在跌停价附近，正裕已封板，奇精经历小幅开合后再封。说明指数修复、已经获认可的低位品种与高位接力是不同路径。该节点是事后观察到的低点，不能伪装成当时已知的确定买点。','', '**13:10附近出现机器人扩散，但不是全部低位跟随大业才启动。** 大业13:10分钟收于涨停，供应商秒级首次封板记录为13:09:45，二者分开归属。正裕和奇精此前早已封板；派斯林与大丰在午后走强，大丰随后幅度突出。同期创业板也强烈回升，因此还存在广泛风险偏好修复、共同题材催化的解释。海洋王在该窗口仍未响应，慈文也不是同步走强，不能据此确认统一资金调度。','', '**14:26更像高位及旧情绪补修复，仍不是全市场共同再次上冲。** 海洋王此时首次分钟触板并收于涨停，九阳、慈文、百洋及新华在前后5分钟有不同幅度修复；但沪指同期略弱，华天基本横住，通富仅小幅变化。旧情绪响应有先后，也未全部封板。需分别看之后能否维持，不能据这一窗口把半导体与旧情绪固化为永久互斥。','', '## 次日使用边界','', '把“早盘逆势已获认可”和“午后普涨补修复”分开核对竞价及开盘承接：前者若弱开并持续跌破竞价区间，是认可衰减的反证；后者需观察高位锚、同题材队列和指数能否同时延续，不能只凭昨日地天或收盘涨停升级。正裕、奇精等早封品种不能因午后锚封板而倒写成它的跟风；海洋封板也不能单独证明九阳/百洋恢复龙头身份。具体参与排序仍需风险核验、完整竞争及独立审阅。','', '东财涨停池海洋09:53:51、中广09:55:21与分时冲突。海洋应采用14:26分钟事件；中广13:44已触板、13:59首次分钟收于涨停。两股东财/Tencent共同241个分钟收盘价一致；这是分钟级交叉核验，不能认证逐笔队列、真实持仓成本、成交机会或账户间资金迁移。','', '完整原价、时钟、来源哈希及前后1/3/5分钟各自变化见 `RELATION_WINDOWS.json`；各股分钟完整终盘走势见 `情绪关系分时图.png`。']
(OUT/'三个情绪节点_关系与反证.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(json=str(OUT/'RELATION_WINDOWS.json'),plot=str(OUT/'情绪关系分时图.png'),note=str(OUT/'三个情绪节点_关系与反证.md'),rows=3*len(SYMBOLS),all_windows_same_session=all(s['valid_same_trading_session'] for e in result for r in e['stocks'] for s in r['samples'].values())),ensure_ascii=False))
