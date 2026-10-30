from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

OUT=Path(__file__).resolve().parent
fonts={f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family']=next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts),'DejaVu Sans')
plt.rcParams['axes.unicode_minus']=False
series={}
qa={}
for symbol,name in [('sz002185','华天科技'),('sh600825','新华传媒')]:
    rows=json.loads((OUT/(symbol+'_eastmoney_parsed.json')).read_text(encoding='utf-8'))
    series[name]={r['time'][11:]:r['close'] for r in rows}
    tx=json.loads((OUT/('tencent_'+symbol+'.response')).read_text(encoding='utf-8'))['data'][symbol]['data']
    txprices={r.split()[0]:float(r.split()[1]) for r in tx['data']}
    qa[symbol]={'date':tx['date'],'rows':len(rows),'minute_close_mismatch':sum(abs(r['close']-txprices[r['time'][11:].replace(':','')])>1e-8 for r in rows)}
    assert qa[symbol]['date']=='20261022' and qa[symbol]['rows']==241 and qa[symbol]['minute_close_mismatch']==0
    original=json.loads((OUT.parent/'facts'/'raw'/(symbol[2:]+'_daily.json')).read_text(encoding='utf-8'))['data'][symbol]['day']
    daily=next(r for r in original if r[0]=='2026-10-22')
    minute_ohlc=[rows[0]['open'],max(r['high'] for r in rows),min(r['low'] for r in rows),rows[-1]['close']]
    daily_ohlc=[float(daily[i]) for i in [1,3,4,2]]
    assert minute_ohlc==daily_ohlc
    qa[symbol]['minute_daily_ohlc_equal']=True
    qa[symbol]['minute_volume_minus_daily_hands']=sum(r['volume'] for r in rows)-float(daily[5])
for rec in json.loads((OUT/'peer_tencent_capture_receipt.json').read_text(encoding='utf-8'))['sources']:
    d=json.loads(Path(rec['raw_path']).read_text(encoding='utf-8'))['data'][rec['symbol']]
    assert d['data']['date']=='20261022'
    series[rec['name']]={r.split()[0][:2]+':'+r.split()[0][2:]:float(r.split()[1]) for r in d['data']['data']}
qa['window_returns_1354_to_1403_pct']={name:(v['14:03']/v['13:54']-1)*100 for name,v in series.items()}
qa['scope']='Retrospective descriptive case, minute evidence, no account-flow causality or predictive validation.'
(OUT/'analysis_verification.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')

fig,axes=plt.subplots(2,1,figsize=(10.6,7.2),sharex=True)
sets=[(['华天科技','新华传媒'],['#2166ac','#c43c39']),(['长电科技','通富微电','粤传媒','百洋股份','上证指数','创业板指'],['#348abd','#4e9d65','#b55d60','#c78b32','#777777','#aaaaaa'])]
for ax,(names,colors) in zip(axes,sets):
    for name,color in zip(names,colors):
        v=series[name]
        keys=sorted(k for k in v if '13:45'<=k<='14:10')
        ax.plot([datetime.fromisoformat('2026-10-22 '+k) for k in keys],[(v[k]/v['13:54']-1)*100 for k in keys],label=name,color=color,lw=1.3 if name in ['上证指数','创业板指'] else 2,ls='--' if name in ['上证指数','创业板指'] else '-')
    ax.axhline(0,color='#d0d0d0',lw=.8)
    ax.axvline(datetime.fromisoformat('2026-10-22 13:58'),color='#2166ac',ls=':',lw=1.4)
    ax.axvline(datetime.fromisoformat('2026-10-22 14:00'),color='#c43c39',ls=':',lw=1.4)
    ax.set_ylabel('相对13:54变化（%）')
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',ncol=3,frameon=False,fontsize=9)
axes[0].set_title('华天封板与新华跳水：分钟时序核对（2026-10-22）',loc='left',fontsize=14,pad=15)
axes[0].text(.99,.96,'蓝色竖线：13:58封板所在分钟\n红色竖线：14:00新华下跌加速',transform=axes[0].transAxes,ha='right',va='top',fontsize=9)
axes[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,5)))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[1].set_xlabel('上海时间；曲线为分钟收价，不能确认秒级先后或同一账户资金迁移')
fig.text(.08,.018,'来源：东财／腾讯公开分钟；13:54为比较基准。新资料盘后取得，不回填原预测。',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.04,1,1])
fig.savefig(OUT/'华天新华_分时对照.png',dpi=160,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'verified':True,'plot':str(OUT/'华天新华_分时对照.png'),'qa':qa},ensure_ascii=True))
