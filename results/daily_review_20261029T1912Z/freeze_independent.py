from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,csv
O=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x',encoding='utf8') as s:json.dump(v,s,ensure_ascii=False,indent=2)
f=json.loads((O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json').read_text(encoding='utf8'))
relay={
'603131':('YES',1,'二板09:50:29首封且未报告再开板，换手11.12%、成交5.75亿；商业航天与焊接机器人双线，今日航天多只低位扩散，优于单纯承受昨日机器人高位兑现。反证：新消息方向一日游、宇晶及中国卫星次日失败；本股并非最早首封即可认证带动。'),
'601956':('YES',2,'三板全天一字、换手2.53%、成交1.06亿，只作封板延续与机器人情绪锚观察。襄阳断板说明一字强未转为高位助攻；前三季归母亏损1.14亿已核，开板承接未经过检验，三进四平低开永久取消本预案。'),
'600715':('UNKNOWN',3,'三板09:30:11首封、11:08:26末封，成交11.55亿/换手17.97%有容量和早封要件，影视有横店首板。最强反证是多次开板、传媒老高标未必响应、上半年亏损及今日题材扩散有限；一年断板风险与尾盘队列缺口，暂不认证真龙特赦。'),
'001219':('YES',4,'四板全天一字，消费高度锚、食品首板扩散；封单延续倾向靠前，1.57%换手代表集中兑现未检验。桂发祥跌停说明食品同题材不自动共赢；控制权筹划未完成，一字不排单。'),
'002862':('YES',5,'二板缩量一字、换手1.67%、成交0.36亿，AI玩具分支硬度观察；可与文投形成AI应用共振假设，但不同业务不认证同资金。小盘薄量代理排除于首板路线，连板仍仅看一字风向。'),
'002708':('UNKNOWN',6,'四板28.87亿/27.90%换手，具有容量但17次开板且末封15:00，尾盘薄封单约854万，不能等同加单锁仓。机器人与PCB今日多只失败，首板扩散不及航天；明天需明显修复，否则兑现。Q3预告10月30日披露，风险前置待核。'),
'002526':('UNKNOWN',7,'二板缩量一字，机器人低位身位，和东贝/实丰/沪工共同竞争；市场退潮下开板后的真实承接未知。作者称尾盘漏单仅线索，缺队列对不能事实认证撤单超过30%。'),
'603006':('UNKNOWN',8,'二板09:30早封且零报告开板，2.42%换手、0.72亿；筹码尚未充分交换，相对沪工缺少已核题材扩散优势，保留同位竞争观察不硬填断板。'),
'002172':('UNKNOWN',9,'二板18.60%换手、7.57亿，早盘多次开板后09:39稳封；名称叙事及医疗支持关系待核。襄阳断板、盛洋新首板的名称共存并不证明接力到本股，非默认补涨龙。'),
'002943':('NO',10,'二板13:00才首封、14:44再封，成交9.03亿、18.05%换手，后发且相邻日放量约3.89倍。与沪工相比盘面主动性更弱；航天/设备叙事不是否定，但本次偏不晋级，次日若反证成立追加新判断不改旧预测。')}
first={
'003032':('YES',1,'09:32:03早封零报告开板，流通25.75亿、换手8.52%；Q3营收+64.07%、归母利润4327.90万，扣非亦增，财报兑现与AI教育叠加。相对同日低价弱业绩旧股有新催化，实丰/文投若转弱则AI分支认可不足；新季报高增不保证连板。'),
'002565':('YES',2,'09:51首封09:56稳封、9.69亿、11.48%换手、流通86.28亿在原上限内；航天算力叙事与同日卫星群体对应，容量强于多数低位首板。10月21日已涨停，属于旧强股重起而非纯新周期；Q3亏损3732万与员工持股解锁是风险，不预设资金卖出。'),
'600592':('UNKNOWN',3,'09:30:17早封零开板、流通69.56亿，轴承低位与光洋仍有观察联系；Q3扣非+48.53%但归母-20.98%，不能写利润全面增长。成交2.47亿、换手3.58%薄于主动核心；若光洋/襄阳负反馈扩散不作补涨接力，列第三观察非已准买。'),
'603703':('UNKNOWN',4,'10:05早封零开板、流通45.07亿、换手8.26%，卫星通信群体线索；但Q3归母亏损1573万且现金流弱，相对传智财报支撑更差；不能因洋字自动认证新龙。'),
'600501':('UNKNOWN',5,'10:25首封、10:45末封、5.69亿换手8.07%，航天直接名称且在原市值内；Q3归母盈利3070万但前三季仍亏890万，不能把名称或单季扭亏直接认证唯一航天龙。'),
'000530':('UNKNOWN',6,'37.92%换手、18.05亿，10:36首封超过本特赦10:30门，14:05末封；有独立容量但旧周期高波动，不属首板低位新核心首选，不能用有成交直接认证锁仓。'),
'600261':('UNKNOWN',7,'09:32早封、换手5.68%、2.79亿；可能关联航天叙事与旧名称线，但真实题材身份和持续性需核，不把阳字自动加分。'),
'605338':('UNKNOWN',8,'09:59封、低换手2.16%、0.96亿，食品扩散受青岛高度影响；桂发祥跌停是直接反证，优于单靠食品概念仍须换手确认。'),
'603536':('UNKNOWN',9,'10:00首封、5.75%换手、1.40亿、流通25.11亿，食品低位观察；相对早封AI与航天少新催化，且旧消费亏钱效应未清。'),
'002086':('NO',10,'13:53首封、21.83%换手、8.37亿，旧海洋名称抱团反抽；不满足早封门且市场旧高位断板，偏不晋级，不作首板提前挖龙主选。')}
rows=[]
for r in f['rows']:
 c=r['code'];rank=None
 if c in relay:call,rank,reason=relay[c]
 elif c in first:call,rank,reason=first[c]
 elif r['kind']=='limit_up':
  call='UNKNOWN';reason=f"{r['name']}首封{r.get('seal_first')}、末封{r.get('seal_last')}、换手{r.get('turnover_pct'):.2f}%；"
  reason+='超原首板市值上限，仅完整对照。' if r.get('screen_status')=='EXCLUDED_CAP' else '未证实持续性及相对位置优于完整同日对手，不强填正向或淘汰。'
 else:
  call='UNKNOWN' if c in {'000678','003028','002164','001400','001270'} else 'NO'
  reason=f"{r['name']}今日{r['kind']}、收盘{r.get('close')}、涨跌{r.get('change_pct'):.2f}%；"+('仅保留断板/炸板修复观察，不能混入连板晋级候选。' if call=='UNKNOWN' else '相对封板池承接和抗跌不足，本次偏不封板；缺少的盘口和消息不当已核事实。')
 rows.append(dict(code=c,name=r['name'],kind=r['kind'],boards=r.get('boards'),next_close_limit_up_call=call,priority_rank=rank,reason=reason,
  qualification=r.get('qualification'),source_scope='EASTMONEY_RETURNED_UNION_NOT_EXCHANGE_CERTIFIED',
  cancel_if='09:25截图/同位题材否证或09:30及09:35后承接失效；取消不自动替补',probability='UNVALIDATED',
  unknowns=['full one-year failed-streak risk','timestamped 1430-to-close queue pair','future auction data']))
v=dict(formed_at=now(),signal_date='2026-10-29',target_date='2026-10-30',model_output_seen=False,prior_model_outputs_and_results_known=True,
 full_pool_scope='Complete returned mainboard non-ST three pools; alternative page timed out; not certified exchange-wide',
 relay_top3=['603131','601956','600715'],first_board_top3=['003032','002565','600592'],
 seal_continuation_observation=['001219','601956','002862'],market_call='退潮与中低位局部抱团并存，拒绝全面主升假设；航天延续和AI应用为待验替代方向',
 forum_sources=[dict(author='春风吹又生tjy',published_at='2026-10-29 15:54 Asia/Shanghai',file='context/forum_chunfeng_1029.txt',read_scope='正文至盘前持仓及操作；未采纳评论',
  accepted='分清硬一字与可买性、文投与横店/沪工与航天持续性验证',rejected='首封领先不直接证明带动；个人交易不证明真实队列；盘后文字不能算尾盘封单证据'),
  dict(author='我最贪婪',published_at='2026-10-29 16:14 Asia/Shanghai',file='context/forum_greedy_1029.txt',read_scope='正文，无评论',
  accepted='高位反馈与首板承接分层，低位题材次日分歧待验',rejected='6/61分母不移植到本次51只；固定40红线、70%概率、光洋开板30次及资金身份未核不照抄；两日6板失败不证明永久硬顶')],
 facts_sha256=sha(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json'),new_fits=0,orders=0,rows=rows)
p=O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json';write(p,v)
write(O/'INDEPENDENT_FREEZE_RECEIPT.json',dict(formed_at=now(),sha256=sha(p),full_rows=len(rows),models_seen_before_freeze=False,original_prediction_files_modified=False))
with (O/'完整102只_独立判断_模型展示前.csv').open('x',encoding='utf-8-sig',newline='') as s:
 keys=['code','name','kind','boards','next_close_limit_up_call','priority_rank','reason'];w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
print(json.dumps(dict(full_rows=len(rows),relay_top3=v['relay_top3'],first_top3=v['first_board_top3'],sha256=sha(p))))
