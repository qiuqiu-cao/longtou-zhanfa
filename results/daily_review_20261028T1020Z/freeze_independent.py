from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv
O=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
facts=json.loads((O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json').read_text(encoding='utf-8'))
# Author decisions recorded before loading today's BASE. Prior dates' scores/results are known.
relay={
'000678':('YES',1,'最高5板；09:32:15封板、成交8.26亿元、量较昨日约减半；机器人中低位扩散且本股未随多伦跌停。最强反证是明日财报、长期亏损、五板后兑现、尾盘队列增量及一年断板史未核。列优先条件观察，不凭低分淘汰；豁免尚待真实队列与历史风险证据。'),
'002708':('YES',2,'三板近一字T板，09:30封、全天振幅约0.87%，机器人轴承与PCB两条市场叙事有同伴；先发强于中后排。低换手3.71%有集中兑现风险，明日需更强竞价及真实换手；与襄阳可共同晋级，不能机械互斥。'),
'601956':('YES',3,'2板09:30:17封、5.30%换手，制造链低位先发，低价和相对封板强度优于晚封江顺；三季度亏损是已核风险，襄阳/光洋弱化会破坏补涨接力假设。'),
'001219':('YES',4,'3板全天一字、1.76%换手、成交0.55亿；延续封板倾向靠前但非主动可成交首选。控制权筹划不是完成转让；九百断板、桂发祥跌、阳光未晋级，消费广度弱，只保留一字先锋情绪观察。'),
'603037':('UNKNOWN',5,'2板10:14封、换手11%、成交3.71亿，有真实交换；但相对东贝后发，若机器人分歧中独立回封可重评，本次不因东贝取消自动切换。'),
'600715':('UNKNOWN',6,'2板10:52封，传媒分支掌阅/百纳可形成不同方向承接；新华/慈文未形成高度接力，过往股东减持已披露，不能冒称资金从机器人确定转来。'),
'001400':('UNKNOWN',7,'2板11:05首封、11:13末封，换手约20%、相邻日量比3.42，存在放量承接也有兑现；高价88.31与同位早封对手相比不优先。'),
'600828':('NO',8,'2板09:56首封、13:36回封，量比4.15，消费高位九百断板，竞价弱后盘中翻强不追溯原取消为成功；本次不把单日回封当消费主升。')}
first={
'003028':('YES',1,'早盘09:31首封09:43稳定、5.90%换手、流通18.32亿；相对低位小盘，机器人智能控制叙事有群体支持。季度收入增而归母净利降7.84%、前三季降50.16%，不能称利润高增；潜在兑现和题材分歧须确认。'),
'603897':('YES',2,'09:30早封、流通54.56亿、无开板报告；Q3归母利润同比52.38%已核，具业绩与制造链叠加。成交0.83亿和1.54%换手代表筹码尚未充分交换，明日不能一字排单。'),
'000670':('YES',3,'09:37首封10:50稳定、9.18%换手、流通58.76亿、成交5.34亿；作为芯片/存储分支与机器人形成分流对照，不能叫唯一主线。三次开板与金安/方正更强容量替代使其只适合确认后观察。'),
'002164':('UNKNOWN',4,'09:31早封、流通81.08亿；主营齿轮/减速器有市场题材映射，前三季利润增39.45%。但前一日冲高回落、今日量仅昨日0.38倍，有回本与获利兑现假设，不优于较低位新首板。'),
'603081':('UNKNOWN',5,'09:34封、5.64%换手，制造与具身智能群体支持；前期多日板路径有旧筹码，低位先手不等于新周期核心。'),
'603633':('UNKNOWN',6,'低价6.36、流通27.15亿、换手6.14%，09:38首封10:11稳定，两次开板；机器人连接器叙事待核实收入，低价不是必然龙头。'),
'002862':('UNKNOWN',7,'AI玩具分支11:09封、流通19.78亿换手6.29%；同主线正反馈有限，是替代方向观察而非强凑主选。'),
'002879':('UNKNOWN',8,'13:24封、流通20.50亿，管道机器人/液冷叙事多重；不如早封控制器直接，晚到启动需次日同伴验证。'),
'002696':('UNKNOWN',9,'尾盘14:49封、33.65%换手，旧名称抱团的高波动反包；有成交不等于早盘主动换手真龙，不满足本次尾盘加单豁免证据，不能作为首板早期潜龙主选。')}
rows=[]
for r in facts['rows']:
 c=r['code'];kind=r['kind'];rank=None
 if c in relay:call,rank,reason=relay[c]
 elif c in first:call,rank,reason=first[c]
 elif kind=='limit_up':
  call='UNKNOWN';reason=f"{r['name']}收盘价{r.get('close')}，首封{r.get('seal_first')}、末封{r.get('seal_last')}、换手{r.get('turnover_pct')}%；"
  if r.get('screen_status')=='EXCLUDED_CAP':reason+='超原首板流通市值上限，只留完整对照，不纳入首板潜龙主选。'
  elif r.get('screen_status')=='EXCLUDED_THIN_PROXY':reason+='命中既存小盘低换手/量能薄代理筛选，非认定真实庄股；不纳入主选。'
  elif r.get('qualification')=='CLOSING_PRICE_CONFIRMED_POOL_CONFLICT':reason+='收盘价与原炸板池冲突；只确认收价事件，封板时序不明，不作接力首选。'
  else:reason+='未证明早封、题材相对身位及持续性优于完整同日对手，保留有理由观察，不硬填断板。'
 else:
  call='UNKNOWN' if c in {'605488','603278','002724','600825','002343','603028','600838'} else 'NO'
  reason=f"{r['name']}在{kind}池，收盘价{r.get('close')}、涨跌{r.get('change_pct')}%；当前缺少强于同日封板者的独立连续承接证据，不列明日封板前三；NO是相对偏不封板判断，不把源未知补成事实。"
 rows.append({'code':c,'name':r['name'],'kind':kind,'boards':r.get('boards'),'next_close_limit_up_call':call,'priority_rank':rank,'reason':reason,
  'role_hypothesis':'HIGH_ANCHOR' if c=='000678' else 'SECTOR_PIONEER' if c in {'002708','001219'} else 'EARLY_POTENTIAL' if c in first else 'RELAY_CONTENDER' if c in relay else 'UNCONFIRMED',
  'source_fact_qualification':r.get('qualification'),'cancel_if':'09:25不达标或09:30后承接/同位/题材否证，取消不自动换股；不得用盘后结果改写本条',
  'forum_opinion_policy':'Accept relative-role and validation hypotheses only; reject specific account/certain flow/motivation assertions',
  'risk_unknowns':['verified 1430-to-close queue pair','exact 0925 amount/unmatched volume','full one-year failed-streak history']})
v={'formed_at':now(),'signal_date':'2026-10-28','target_date':'2026-10-29','model_output_seen':False,'prior_model_outputs_and_results_known':True,'full_pool_scope':'Returned mainboard non-ST provider/page union, not certified exchange completeness',
 'relay_top3':['000678','002708','601956'],'first_board_top3':['003028','603897','000670'],'pure_seal_continuation_observation':['001219','002708','000678'],
 'market_call':'Index breadth rebound with concentrated robot repair and residual mid/high failures; prefer core discrimination over universal acceleration',
 'rows':rows,'facts_sha256':sha(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json'),'model_changes':0,'new_fits':0,'orders':0}
p=O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json';write(p,v)
write(O/'INDEPENDENT_FREEZE_RECEIPT.json',{'formed_at':now(),'sha256':sha(p),'full_rows':len(rows),'models_seen_before_freeze':False,'original_prediction_files_modified':False})
with (O/'完整112只_独立判断_模型展示前.csv').open('x',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['code','name','kind','boards','next_close_limit_up_call','priority_rank','role_hypothesis','reason']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
print(json.dumps({'full_rows':len(rows),'relay_top3':v['relay_top3'],'first_top3':v['first_board_top3'],'sha256':sha(p)},ensure_ascii=False))
