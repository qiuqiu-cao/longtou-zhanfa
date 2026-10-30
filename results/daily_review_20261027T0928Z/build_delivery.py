from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib,csv
O=Path(__file__).resolve().parent;R=O.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def new(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
f=read(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json');lookup={r['code']:r for r in f['rows']};own=read(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json');base=read(O/'BASE_scores_AFTER_INDEPENDENT.json');bm={r['code']:r for r in base['rows']}
relay=['600838','001219','000678'];first=['600828','603758','001400']
specs={
'600838':('L1',[13.98,14.24],[14.25,14.64],'人工偏晋级、BASE偏不晋级；仅观察，本次不设主动买入许可','青岛、桂发祥、阳光、茂业、广百，以及多伦/襄阳的风险反馈','唯一五板且09:30封稳、消费扩散，从前一日高度候选增加了可观察证据；最像情绪龙头的候选仍是它，不认证唯一总龙。','五板后的潜在浮盈兑现与抱团瓦解。外部新增接力可能为追求消费高度和相对抗跌，但早盘一封不证明所有消费由它带动。','09:30观察实际开盘成交和消费群体，不抢竞价或首个拉升；一字14.64只记录，不作为可买。','09:35后观察首次换手回踩是否收回均价、上攻回封14.64；这只是新证据，本卡仍仅观察，不因出现强封临时转买。','九百失守13.31且反抽失败，或食品/零售同时高开低走，取消人工偏强延续情景；本卡始终不提供新买许可。','人工倾向争取六板，但模型45.4低于阈值，保留真实冲突'),
'001219':('L2',[15.43,15.88],[15.89,16.48],'人工偏晋级、BASE偏不晋级；仅观察，本次不设主动买入许可','九百、桂发祥、阳光、茂业与广百；食品同位可共同晋级','昨日人工首板第一今天二板、09:38后稳定、换手7.77%，有消费群体支持，先发地位优于下午反复封的阳光。','涨停开盘后曾低至14.51，交换有所增加仍可能再次兑现；新增接力假设是争取消费低位高度，不能拿青岛业绩增长保证连板。','09:30观察是否保持真实承接和食品同位同步，直接一字16.48不追。','09:35后观察真实换手回封16.48、九百稳定且桂发/阳光至少有承接；仍只做验证，不把观察偷偷升级为买入。','低于15.43不按预设加速路径处理；失守14.98反抽失败、食品集体走弱则取消偏强延续假设。','人工倾向争取三板，与BASE27.2反向；今天成功不证明后续二进三必成'),
'000678':('L3',[16.49,16.97],[16.98,17.61],'人工偏晋级、BASE偏不晋级；仅观察，本次不设主动买入许可','多伦、赛福、光洋、国脉；反例正裕/林州/睿能/南方反馈','四板09:36稳定、20.57%换手后全天保持，正裕和林州跌停时未同跌；相比多伦14:49末封，盘中稳定性更早。','连续大换手可能是承接也可能是拥挤，不能据此断言吸筹。场外愿接要靠同位存活和本股分歧修复，主业概念强弱不是唯一判据。','09:30观察大换手后的溢价和机器人中位亏钱是否缓和，不能因昨天换手充分就直接承接高开。','09:35后观察缩量回踩不破承接区、再放量回封17.61；仍仅观察。若多伦也强且本股稳，是共振而非必须淘汰对手。','失守16.01并反抽失败、放量多次冲板不封、机器人前排开始传导跌停则取消延续假设。','人工倾向争取五板，但BASE27.7反向，不能包装成模型支持'),
'600828':('F1',[4.56,4.69],[4.70,4.92],'人工候选；季度全文未取得，补核前仅观察，补核后仍须实时条件确认','九百、青岛、广百、九牧王；新华百货/百大只作尾盘修复对照','零售09:40稳定，较广百13:28、新华百货14:55更早确认；流通77.39亿未超原上限。','1.57%换手意味着真实分歧尚未测试，明日补量是关键；新增接力可能为消费低位扩散，不能把大盘下跌自动当零售资金流入。','09:30不因高开直接买；须先补核现存季度全文及新重大风险。若未核到、无法核验竞价成交和未匹配量，继续只观察。','完成风险补核后，09:35后真实换手、回踩能收回开盘与成交均价，九百/食品不集体失守且广百仍有承接，再放量回封4.92才保留打板预案；一字不追。','补核失败不开放买入；低于4.56取消本加速买入预案，或失守4.47反抽失败、零售前排一起走弱、两次冲板未封则取消。','人工首板相对第一，偏向争取二板，低换手与风险全文缺口使买入资格仍待验证'),
'603758':('F2',[10.95,11.27],[11.28,11.80],'人工条件关注；无合格当日首板BASE','江顺、大明、东贝、襄阳/光洋；反例正裕、林州及昨日二板失败群体','09:33早封稳定、低开转强，流通46.47亿；早封和小盘地位优于午后反复的亚太/凯众等。','2.95%换手可能次日集中补量。已披露Q3利润大降是风险背景，不称业绩催化；不能用传感器业务逻辑直接替代市场承接。','09:30先看真实补量，不能因早封或弱转强名称就抢买；一字11.80无可成交预案。','09:35后缩量回踩收回开盘区和成交均价，制造同伴仍有承接、机器人前排未集体崩溃，再放量回封11.80才保留；大盘反抽不等于个股达标。','低于10.95取消本加速预案；跌破10.73反抽失败或后排亏钱继续传导、两次冲板未封则取消。','人工倾向争取二板，但情绪龙身份尚无验证，首板模型未输出'),
'001400':('F3',[81.89,84.29],[84.30,88.31],'人工条件关注；无合格当日首板BASE','秦安、大明、华新以及光学/液冷同线；不把所有制造都当助攻','昨日炸板后今天09:35反包稳定，流通25.86亿、换手5.84%；比同类下午才封者更有主动性。','反包后存在前日解套及当日浮盈兑现压力，高价不等于大市值但波动损失也不小；盘后Q3不能倒填今天早盘买盘动机。','09:30看反包溢价是否被真实承接，不抢高开；直接一字88.31不追。','09:35后回踩收回开盘及成交均价，早盘制造同伴没有明显掉队，再放量回封88.31才保留；若只是随科技容量反抽而独立地位下降则取消。','低于81.89取消本加速预案；失守80.28且反抽失败、连续冲板不封、制造群体亏钱扩大则取消。','人工倾向争取二板；本轮来自反包强度，不能据此认证唯一新龙'),
}
cards=[]
for c in relay+first:
 r=lookup[c];slot,exp,strong,gate,conf,reason,cp,a,b,cancel,pred=specs[c]
 limit=float((Decimal(str(r['close']))*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP));assert limit==r['next_upper']==strong[1]
 if slot=='L1':later=['若28日六板且消费继续有晋级，29日才保留高度进一步打开；若断板，不把冲高当连板成功。','30日看首次明显放量分歧能否修复及有无群体跟随，只有独立抱团则降低持续性判断。','11月3日五日窗只有在高度、承接、群体共振都持续时保留走强路径，不预设每天上涨。']
 elif slot=='L2':later=['若28日三板，29日重新判断同股三进四，不能用今天二板成功倒填首板龙头身份。','30日若消费高度和食品群体延续，可提高持续性假设；失去同位地位则停止外推。','11月3日五日窗不承诺连涨，随首次分歧后的承接独立结算累计/逐日与连板。']
 elif slot=='L3':later=['若28日五板，29日须再验高换手后的集中兑现和与多伦/赛福相对地位。','30日只有分歧后仍有群体支持才保留高度路线；板块补跌则失效。','11月3日五日窗保留条件性强势假设，不把同字同音延续当确定路径。']
 else:later=['若28日二板，29日再独立判断同股二进三；未晋级则本轮连续路径失效，未来反包另评。','30日三交易日窗，看是否从跟随变成相对领先、是否有同伴持续响应；未形成则不认证潜在龙头。','11月3日五日窗分别结算连续高度、累计收益和逐日上涨，不承诺五连板或每天上涨。']
 cards.append(dict(slot=slot,code=c,name=r['name'],reference_date='2026-10-27',target_date='2026-10-28',close=r['close'],limit_up_reference=limit,expected_price=exp,strong_price=strong,weak_price_below=exp[0],numerical_rule='人工未验证预案，不是BASE阈值或最优区间；低于区间只取消本预设买入路径，不把未来晋级标签强改为NO。',gate=gate,selection_reason=reason,confirmation_objects=conf,counterparty_hypotheses=cp,auction_actual='28日09:25尚未发生；真实竞价成交额/量、未匹配方向及数量、09:20后稳定性未知。需用有时钟来源核验，未匹配字段缺失不填0，09:15虚拟封单不算真实承接，缺信息只观察。',after0930=a,after0935=b,cancel=cancel,paths=[dict(date='2026-10-28',expectation=pred),dict(date='2026-10-29',expectation=later[0]),dict(date='2026-10-30',expectation=later[1]),dict(date='2026-11-02',expectation='周末后检查是否有新信息和主线变化，旧情景不能自动沿用。'),dict(date='2026-11-03',expectation=later[2])]))
new(O/'FINAL_SIX_EXECUTION_CARDS_20261028.json',dict(formed_at=now(),signal_date='2026-10-27',target_date='2026-10-28',judgement='完整83只独立留档后对照原保存BASE；包括分开记录的盘后公告，本次未取得Gemini新意见',independent_sha256=sha(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json'),relay_top3=relay,first_board_top3=first,relay_order_definition='人工相对情绪/晋级观察顺序。BASE全部负向，三只均仅观察，没有主动接力买入主选。',firstboard_BASE_status='NOT_OUTPUT_INPUT_NOT_QUALIFIED',cards=cards))
# Separate append-only settlement of previous final cards; never edit the old forecast or use it as a new training label source.
prior=O.parent/'daily_review_20261026T1323Z/FINAL_SIX_EXECUTION_CARDS_20261027.json';oldcards=read(prior);fb=read(O/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json');fbm={r['code']:r for r in fb['rows']};supp=read(O/'facts/PRIOR_MISSING_CLOSE_MINUTE_SUPPLEMENT.json');sm={r['code']:r for r in supp['rows']};settled=[]
for x in oldcards['cards']:
 y=fbm[x['code']];cl=y['close'];actual=y['closed_limit_up'];qualification='exact-day daily pair'
 if actual is None and x['code'] in sm:cl=sm[x['code']]['close'];actual=sm[x['code']]['closed_limit_up'];qualification='minute/quote closing-event supplement; daily OHLC still unknown'
 settled.append(dict(slot=x['slot'],code=x['code'],name=x['name'],old_gate=x['gate'],old_expected_price=x['expected_price'],old_strong_price=x['strong_price'],actual_close=cl,return_from_previous_close_pct=(cl/x['close']-1)*100 if cl else None,closed_limit_up=actual,daily_open_proxy=y.get('open_gap_pct'),actual_0925_unmatched='UNAVAILABLE',qualification=qualification,executed_trade=None,old_forecast_unchanged=True))
new(O/'facts/PRIOR_FINAL_SIX_SETTLEMENT_APPEND_ONLY.json',dict(formed_at=now(),source=str(prior),source_sha256=sha(prior),rows=settled,relay_watchlist_close_hits=sum(x['closed_limit_up'] for x in settled if x['slot'].startswith('L') and x['closed_limit_up'] is not None),first_watchlist_close_hits=sum(x['closed_limit_up'] for x in settled if x['slot'].startswith('F') and x['closed_limit_up'] is not None),first_top1_hits=1,warning='Single-date results, not a validated win rate. Observation-only names remain observation-only, price success is not an executed profitable trade.'))
out=[]
def add(s=''):out.append(s)
add('# 2026年10月27日复盘与10月28日预期')
add('下一官方交易日：2026年10月28日周三。形成于10月27日上海晚间，含分别记录的盘后公告；不是15:00之前已知的纯收盘封存信号。完整83只独立判断先留档，再加载原BASE，未重训；没有取得Gemini当前新意见，不冒称联合共识。')
add('来源范围：原非ST主板池36涨停、26炸板、21跌停。兰生/百大虽在炸板池，原池收盘价、未复权日线、15:00分钟与报价一致为涨停参考价9.04/12.62，另列收盘价格事件：38价格涨停=12连板+26首板、24炸板、21跌停，仍为83唯一股票。原池成员冲突、封板时序/队列未知保留，不擅称全交易所完整。155个日线请求152成功；大西洋、上海物贸、曙光日线缺失，其中大西洋分钟补收盘，仍不补造OHLC。完整事实和原响应均在facts/。')
add('\n## 1. 大盘：早盘冲高、午后风险收缩')
add('|指数|收盘|涨跌幅|\n|---|---:|---:|\n|沪指|3648.30|−1.33%|\n|深成指|11245.73|−2.50%|\n|创业板|2548.15|−3.41%|')
add('沪深成交原分钟累计合计14609.58亿元，较昨日16087.93亿元缩1478.35亿元，约−9.19%。三指数09:31分钟点均为当日观测高点，15:00均为观测低点；这是分钟点性质，不冒充逐笔真实日内最高低。早盘强势没有维持，14:30后继续下压。公开网页仍混盘中/昨日读数，收盘涨跌家数未通过核验，本次不填精确家数，也不拿网页盘中−0.09%当沪指收盘。指数原文在minutes/capture_receipt.json。')
add('10/28主情景：先观察弱修复/反抽能否成立，核心抱团接受分歧；即便指数反弹，也优先防冲高兑现。转强须见深创不再扩大跌幅、昨日中位跌停打开有承接、消费二板形成新的三板、九百/四板群体分歧后仍稳；转弱须见高位开板后无法回封、消费二板高开低走、今天反包股再被兑现。没有证据直接预测全面反转。')
add('\n## 2. 情绪、梯队与对手盘')
add('定性：弱市中高位抱团和消费扩散，伴随中位杀跌；不是全面升温。当前最像情绪龙头的候选仍是上海九百，但是否唯一总龙、能否穿越首次大分歧都未证明。消费二板青岛是更低位置的观察，襄阳是高换手机器人幸存者，三个角色不能混成一个胜率。')
add('|昨日完整梯队|今日反馈|含义|\n|---|---|---|\n|首板33|日线8/31，2未知；大西洋分钟补失败后为8/32已知、1未知上海物贸|消费更集中，不能报成无缺口的8/33|\n|二板8|0晋级，均已知；平均−0.63%|三板断层，中位容错差，不能因高标上涨普遍接力|\n|三板5|3晋级；正裕、林州跌停|同位淘汰剧烈，存活不等于所有板块情绪好|\n|四板1|九百晋级五板|高度回升与指数下降背离，抱团需防集中兑现|\n|昨日炸板18|2收盘涨停、3跌停，均值−2.94%|仅少数反包，有反例不能一概排除，也不应普遍抄炸板|\n|昨日跌停25|24日线已知、1未知，已知均值−5.19%|亏钱传导仍明显|')
add('完整梯队：五板上海九百；四板襄阳、赛福天、多伦；三板空缺；二板青岛、桂发祥、阳光、国脉、光洋、胜通、沪光，以及价格确认但池身份有冲突的兰生。胜通超原市值范围，兰生只补价格晋级事件，缺合格封板卡不列主选。')
add('|关系|已核事实|关系假设/反证|明日影响|\n|---|---|---|---|\n|指数↔投机|指数尾盘下跌，九百/青岛/襄阳价格仍封住|相对抱团；也可能是存量筹码紧，不代表全市场新增接力|只看分歧承接，不抢明早一致加速|\n|消费↔机器人|食品三只晋级，零售多首板；旧二板机器人全失败，而三只四板存活|局部偏好向消费扩散；机器人也有独立存活，不是永久互斥|消费列人工优先，仍检查机器人跌停是否拖累风险偏好|\n|容量科技↔传媒|14:00前后华天强于新华，之后尾盘两者都回落|可提出风格分化，也有共同指数卖压解释；不能断言同一资金搬家|延华失败不能因科技一拉升就单因解释，也不固定科技/传媒跷跷板|\n|海洋王↔洋字跟风|海洋王反包；大西洋/东方海洋炸、百洋收跌|老高标修复没有普遍传到跟风，名称只是叙事候选|不把名称当自动加分，明日需同伴真实回流|\n|同位竞争|襄阳/多伦/赛福共同四板，正裕/林州跌停|共同晋级与淘汰并存，不能预设只有一个赢家|前排稳定和失败者止跌共同检查|')
add('关键窗口已对齐前后1/3/5分钟：09:37观察正裕低位与襄阳回封，九百/青岛并未同跌；14:00华天约15.71、新华15.06附近，指数只短暂修复；14:53附近沪深创继续下压，大西洋14:54后从6.12滑落，收5.72，东方海洋亦回落，而九百/青岛/襄阳保持。支持亏钱分层、旧线跟风承接弱；也要考虑个股兑现和共同市场风险收缩，不证明确定资金因果。')
add('多伦日线最低9.66、末封14:49；分钟价格点全9.85只能说明点位采样，不能否认区间内打开。赛福全天OHLC一价但原末封15:00/页面09:25矛盾，不能编造真实队列。它今天换手10.31%，按原规则已不命中昨日薄流动性排除；不因昨日被排除就永久排除，也不把一字当可买。')
add('对手盘要落到可观察证据：九百五板后存在浮盈兑现假设，新买方可能追求高度和消费共振；青岛补量后仍有先发地位，但也可能在二进三集中兑现；襄阳连续高换手既可能释放抛压，也可能拥挤。茂业低换手首板要防次日集中补量；秦安低开早封尚缺首次分歧；江顺反包要面对前日解套与当日浮盈两种卖压。均不认证真实成本、吸筹控盘、特定资金身份或三板组。')
add('有限论坛预读：两篇26日原帖复核，27日新正文未取得，逐观点采纳/否决、作者时间和实取时钟见context/论坛预读与事实取舍.md；不把论坛26日看好延华当今天仍有效，也不照搬自报胜率/资金动机。')
add('\n## 3. 连板完整BASE')
add('真实身份：原保存29字段relay BASE，权重SHA256 8c734bd5…dc3af7；目标10/28收盘晋级，阈值0.5。分数为raw_score×100，未校准，不是概率。使用公开替代源、市场上下文13字段未知、包含兰生价格确认的池成员冲突，正式源门未过，故全部属于补充推理，不冒称新正式预测。')
add('|原排名|股票|板位→目标|分数/100|原方向|筛选/资格|\n|---|---|---|---:|---|---|')
for b in sorted(base['rows'],key=lambda x:x['rank']):
 r=lookup[b['code']];status='超原市值范围' if r.get('screen_status')=='EXCLUDED_CAP' else '池成员/封板资格冲突，仅观察' if r['code']=='600826' else '一价且末封字段冲突，仅观察' if r['code']=='603028' else '范围内，模型未转正'
 add(f"|{b['rank']}|{r['name']} {r['code']}|{r['boards']}→{r['boards']+1}|{b['score_display']:.1f}|{'偏晋级' if b['BASE_direction']=='YES' else '偏不晋级'}|{status}|")
add('BASE结论：全部低于原阈值，没有模型正向连板主选。九百相对第一不等于转正；人工对九百/青岛/襄阳偏晋级留档，但本次不据此开放主动买入。多伦今天虽然晋级，明天分数33.4及尾盘分歧已不同，不能沿用昨晚正向结论。')
add('\n## 4. 首板完整BASE')
add('当日首板BASE未输出。保存62字段STAGE_CYCLE_GENE latest_S1模型存在，但当前原全市场截面、60日连续历史、按日换手等资格不足；26只完整首板全部保留，不借旧分数/连板分数/人工排序填BASE。大洋、盾安、中谷超原93.27亿元流通市值范围；曙光缺日线/风险资格；百大价格确认而池成员冲突。')
add('|人工相对序|股票|首封→末封|换手%|流通亿元|筛选与BASE|\n|---|---|---|---:|---:|---|')
frank={r['code']:r.get('first_board_relative_rank') for r in own['rows']}
def tm(v):
 if v is None:return '未核'
 s=f'{int(v):06d}';return s[:2]+':'+s[2:4]
firstrows=sorted([r for r in f['rows'] if r['kind']=='limit_up' and r['boards']==1],key=lambda x:frank[x['code']])
for r in firstrows:
 status='超限' if r.get('screen_status')=='EXCLUDED_CAP' else '资格未知' if r['code']=='600303' else '池冲突' if r['code']=='600865' else '范围内'
 add(f"|{frank[r['code']]}|{r['name']} {r['code']}|{tm(r.get('seal_first'))}→{tm(r.get('seal_last'))}|{r['turnover_pct']:.2f}|{r.get('reported_float_cap_cny',0)/1e8:.2f}|{status}；BASE未输出|")
add('\n## 5. 最终连板前三：均仅观察')
add('|人工关注序|股票|为什么排前|BASE冲突与参与状态|\n|---|---|---|---|\n|1|上海九百|唯一五板、消费扩散，最像当前情绪龙头候选|原第1/45.4/偏不晋级；人工偏晋级但仅观察|\n|2|青岛食品|消费二板先发、09:38稳定、7.77%换手，位置低于高度股|原第8/27.2/偏不晋级；人工偏晋级但仅观察|\n|3|襄阳轴承|四板真实高换手后09:36稳定，较多伦尾盘才末封更早确认承接|原第7/27.7/偏不晋级；人工偏晋级但仅观察|')
add('不强凑可买三只。赛福一价/末封冲突、多伦尾盘分歧、光洋低换手且接近市值上限、桂发祥日内振幅与同位地位都单列保留，不自动替补买入。大业监管原事项未清、林州重大财务处罚事先告知继续不主动参与；普通异动公告或机器人营收小不能单独否决市场认可。')
add('\n## 6. 最终首板前三：人工相对顺序')
add('|顺序|股票|为什么选|反证与资格|\n|---|---|---|---|\n|1|茂业商业|09:40零售早封，承接消费扩散，较广百/新华/国芳更早稳定|仅1.57%换手，流通77.39亿未超原线但不算极小；季度全文请求失败，补核前仅观察|\n|2|秦安股份|09:33早封、低开转强、流通46.47亿，制造方向相对先行|2.95%换手待补量；Q3利润下降是已公开风险背景，不能称业绩催化，不认证传感器龙头|\n|3|江顺科技|昨日炸板今天09:35反包稳住，流通25.86亿、换手5.84%|反包兑现与高价波动；今晚季度报告不倒填今天早盘动因|')
add('东贝早封但目前相对身份不及秦安/江顺，已披露业绩转亏不抹去当日认可；九牧13:12末封、广百13:28、新华/国芳尾盘确认，消费同位可以共同晋级，只是本次相对次序靠后。深赛格一字0.76%换手未验承接，有担保代偿背景，不以封住就放行；大明流通8.42亿、换手11.04%仅比较，不认定资金身份。')
add('\n## 7. 六只10/28竞价、触发与取消')
add('数值是未经回测验证的人工情景区间，非最优阈值、BASE转正线或概率。低于符合预期下沿只取消本预设加速参与路径，不等于保证断板；弱转强若发生须另时点判断，不能盘后倒填。09:25真实成交/未匹配量未知，缺数据只观察。普通涨停限价委托不是条件触发单，一字不给虚假成交。')
for c in cards:
 add(f"\n### {c['slot']} {c['name']} {c['code']}")
 add(f"10/27收盘{c['close']:.2f}，10/28涨停参考{c['limit_up_reference']:.2f}。09:25符合{c['expected_price'][0]:.2f}—{c['expected_price'][1]:.2f}；强于{c['strong_price'][0]:.2f}—{c['strong_price'][1]:.2f}；低于{c['weak_price_below']:.2f}记弱。状态：{c['gate']}。")
 add('理由：'+c['selection_reason']+' 对手盘：'+c['counterparty_hypotheses'])
 add('确认对象：'+c['confirmation_objects']+'。 '+c['auction_actual'])
 add('09:30：'+c['after0930']+' 09:35后：'+c['after0935'])
 add('取消：'+c['cancel'])
 add('逐日路径：'+'；'.join(x['date']+' '+x['expectation'] for x in c['paths']))
add('这里没有账户、仓位和实际订单资料，不指定账户仓位或虚构成交。若人工条件参与，仅小规模试错，不加仓摊低；新买当日通常受A股T+1约束，买后破位不能假设可立即卖出。次一可卖交易日若高开跌回均价且不能收回、冲板未封且群体转弱，按原持仓规则减出；涨停卖出受阻/跌停无法卖出单列。止损/减仓条件不是确保成交的价格承诺。')
add('\n## 8. 上次预测结算与本次检验')
add('|10/26晚间原候选|今日收盘|晋级|原参与状态保留|\n|---|---:|---|---|\n|多伦|9.85|是|原条件关注；是否有真实入场/成交未知|\n|九百|13.31|是|原仅观察，不追认成可买成功|\n|正裕|11.77|否，跌停|原仅观察，独立偏晋级判断失败保留|\n|青岛|14.98|是|原人工首板第一；一字开后分歧不等于自动成交|\n|大西洋|5.72|否|分钟补收盘事件，原日线仍缺失|\n|延华|5.18|否|高开约8.61%后收跌约3%，不能把强竞价当晋级保证|')
add('人工首板Top1本日1/1、Top3为1/3；这是单日命中，不是已验证60%胜率，更不是交易净收益。连板关注名单2/3，但其中九百原仅观察，不能混为两笔可执行盈利。昨日原BASE正向赛福/多伦均晋级；原负向九百/襄阳漏判也保留，不能只展示正例。完整90只旧池结果及弃判未知都留档，未把旧预测改成今日赢家。')
add('本次可检验预测：九百/青岛/襄阳人工偏晋级、模型反向且最终仅观察；茂业/秦安/江顺人工争取二板，首板BASE未输出。10/28结算收盘涨停，10/29新评同股晋级，10/30三日窗口，11/3五日窗口；盘中触板、收盘涨停、累计上涨、逐日上涨、连续板及真实龙头身份分别结算。没有事后删弱日或缩小分母。')
add('\n## 9. 原文与边界')
add('行情原始HTTP、155次日线请求、主分钟39次/38成功、新增11次/11成功、第二域15次均失败，以及42只公告索引和11份公告全文在本目录。公开网页中盘中快照与昨日数据混杂，本次以带日期/时钟的原响应复核为主。公告供应商日期标28日但实取27日晚间者，首次公开时间仍未知；只作盘后预案，不解释今天早盘因果。')
add('有限股吧：[延边刺客1026](https://m.tgb.cn/a/2wd3CQePUzU)、[率土之滨非魔也10/26复盘](https://m.tgb.cn/a/2wdbgSIcLdH)。文件：[多伦风险提示](https://pdf.dfcfw.com/pdf/H2_AN202610261830935082_1.pdf)、[九百风险提示](https://pdf.dfcfw.com/pdf/H2_AN202610231830847770_1.pdf)、[光洋异动](https://pdf.dfcfw.com/pdf/H2_AN202610271831032908_1.pdf)、[江顺今晚季度报告](https://pdf.dfcfw.com/pdf/H2_AN202610271831025831_1.pdf)。论坛观点仅启发假设，发行人公告只说明文件事实，不认证涨跌原因。')
add('拟合0、生产BASE改动0、旧预测/账本/交易/定时任务改动0。未检验实时队列、真实竞价未匹配量、账户成交和收益；资料取得及说服力完善不等于预测能力已经提升。')
report=O/'20261027复盘与1028预期_完整双榜.md'
with report.open('x',encoding='utf-8') as w:w.write('\n\n'.join(out)+'\n')
with (O/'完整83只_独立判断_模型展示前.csv').open('x',encoding='utf-8-sig',newline='') as w:
 writer=csv.DictWriter(w,fieldnames=list(own['rows'][0]));writer.writeheader();writer.writerows(own['rows'])
protected=read(O/'PROTECTED_BEFORE.json');checks={p:sha(R/p)==h for p,h in protected.items()};assert all(checks.values()),checks
assert sha(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')==read(O/'INDEPENDENT_FREEZE_RECEIPT.json')['sha256']
assert len(base['rows'])==12 and len(firstrows)==26 and len(own['rows'])==83 and all(b['BASE_direction']=='NO' for b in base['rows'])
new(O/'DELIVERY_VERIFICATION.json',dict(formed_at=now(),report=str(report.relative_to(R)),report_sha256=sha(report),protected_files_unchanged=checks,full83_independent_unchanged=True,full12_BASE=True,first26_no_qualified_BASE=True,original_pool_membership_conflicts_retained=['600826','600865'],raw_daily_unknowns_retained=['600558','600822','600303'],final_relay_top3=relay,final_first_top3=first,relay_all_observation_only=True,maoye_risk_fulltext_pending=True,price_limits_verified=True,cards=6,new_fits=0,new_formal_predictions=0,actual_Gemini_opinion=False,live_auction_observations=0,orders=0))
print(json.dumps(dict(report=str(report),cards=6,all_negative_BASE=True,protected_files_unchanged=all(checks.values()),prior_first_top3='1/3'),ensure_ascii=False))
