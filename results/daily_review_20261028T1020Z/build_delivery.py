from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP,ROUND_CEILING,ROUND_FLOOR
import json,hashlib,csv
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
f=read(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json');d=read(O/'facts/normalized_daily.json');base=read(O/'BASE_scores_AFTER_INDEPENDENT.json');own=read(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json');receipt=read(O/'INDEPENDENT_FREEZE_RECEIPT.json')
assert sha(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')==receipt['sha256']
lut={r['code']:r for r in f['rows']};blut={r['code']:r for r in base['rows']}
first=[r for r in f['rows'] if r['kind']=='limit_up' and r['boards']==1]
qold=read(ROOT/'results/daily_review_20261027T0928Z/features/FIRSTBOARD_BASE_QUALIFICATION.json')
q={'formed_at':now(),'signal_date':'2026-10-28','target_date':'2026-10-29','status':'NOT_OUTPUT_CURRENT_FIRSTBOARD_BASE_INPUT_NOT_QUALIFIED',
 'complete_firstboard_rows':len(first),'candidate_codes':[r['code'] for r in first],
 'model_identity':qold['model_identity'],'model_path':qold['model_path'],'model_sha256':qold['model_sha256'],'target':qold['target'],
 'training_label_cutoff':qold['training_label_cutoff'],'feature_count':62,
 'reason':'Current OHLC capture covers Sep18-Oct28 only, lacks source-qualified complete 60-session price/gene and daily first/failed/relay/cycle history mapping required by the saved first-board model. Training imputation is not permission to supply invented inputs.',
 'scores_output':0,'new_fits':0,'source_gate_pass':False}
write(O/'features/FIRSTBOARD_BASE_QUALIFICATION.json',q)
params={
'000678':('L1',3,6,1000,['002708','601956','002164','600838'],'高标第一观察；财报及尾盘队列/历史风险证据待补，未认证完成BASE豁免，不预先授权新买入。','09:30若秒板不盲排；09:35后回踩缩量不破实际开盘价和成交均价、再放量回封19.37才考虑。前置证据未补齐则继续观察。','低于18.14取消该加速预案；<=17.61绝不低吸。放量跌破实际竞价价、连续冲板失败、光洋/东贝显著负反馈或财报新风险均取消。',
 ['10/29偏争六板，但必须验证财报和承接','10/30六进七若成功更需分歧换手，不预设缩量一字','11/02高度竞争和兑现压力上升，必须保持核心地位','11/04以阶段高标延续或断板再评为情景，不预测逐日必涨']),
'001219':('L2',7,9.5,300,['600838','002708','600828','002820'],'三板一字先锋、仅观察；无可成交承接证据不排缩量一字。','09:30一字18.13仅看风向；09:35后若打开需先看到真实卖压释放和回封、消费同伴修复，原一字观察状态不自动转买入。','<=16.48永久注销本次三进四买入预案；16.49至17.63低于昨日一字所需强度，本次不追盘中翻强。九百转大跌、食品群体走弱、反复炸板则取消延续倾向。',
 ['10/29倾向冲四板或继续一字，封板延续强于可参与性','10/30若四板成功，首个爆量开板为主要风险节点','11/02需消费扩散而非独苗，否则高度受限','11/04不认证消费总龙，控制权进展与题材持续性另验']),
'601956':('L3',3,6,300,['000678','002708','603037'],'二进三条件关注；BASE45.5偏不晋级，人工相对先发判断与模型分歧；本次不是三进四低分豁免。','09:30先看6.28以上竞价兑现承接；09:35后守实际开盘价和成交均价，缩量回踩后放量冲6.70或首次充分换手回封，且襄阳/光洋仍稳。','<=6.27取消本预案；高开后快速跌破竞价价、凯众独强而本股不响应、襄阳和光洋同时走弱或新公告硬风险取消。',
 ['10/29偏争三板，需低位先发得到确认','10/30若晋级三板，重新执行三板标签与同位竞争审查','11/02预期中位分歧，不预设追随高标一路晋级','11/04只有机器人扩散持续才保留补涨高度假设']),
'003028':('F1',2,5,200,['000678','002708','603897','002164'],'首板早期潜龙第一观察；无合格当日首板BASE，属独立判断。','09:30高开需有真实成交承接，不能只看买一；09:35后开盘区和成交均价不失守、回踩缩量后放量冲28.50或充分换手回封，机器人前排不能同步转弱。','<=26.42取消高开确认预案；竞价虚高后放量跌破开盘价、长城/宁波走强而本股持续掉队或利润下滑被市场负面定价则取消。',
 ['10/29倾向争二板，作为首板入围验证','10/30若二板成功需证明同批相对领先','11/02争三板时易分歧，不用小盘身份保证晋级','11/04只有持续扩散与自身承接成立才谈龙头潜力']),
'603897':('F2',2,5,150,['003028','002164','000678'],'首板第二观察；Q3利润增长已核，低成交低换手需补量，无首板BASE。','09:30若一字29.07不排；09:35后新增成交释放卖压且守实际开盘区，振邦/宁波至少保持承接，再看换手回封29.07。','<=26.95取消本预案；开板后补量失控、放量失守开盘区，或仅有财报故事而制造链集体回落则取消。',
 ['10/29倾向争二板，但更看补量质量','10/30若过二板，需由业绩首板转为市场认可的接力角色','11/02低换手后的集中兑现是主风险','11/04继续走强需题材高度与个股独立性验证']),
'000670':('F3',2,5,500,['002636','600601','002741','002185'],'首板第三观察；算力/芯片分支替代路径，无首板BASE；不把容量反弹直接当小票连板主升。','09:30观察芯片容量和本股相对强度；09:35后缩量回踩守竞价价、再放量冲7.79或回封，金安/方正不能同时炸板走弱。','<=7.22取消本预案；反复触板不封、量能放大却滞涨，或金安/方正转弱且本股无独立承接则取消。',
 ['10/29倾向争二板，但弱于机器人直接核心的确认度','10/30需芯片分支继续而非一天反弹','11/02若容量回落，预计分歧大于顺畅加速','11/04必须跨题材轮动仍领先，才能升级早期龙头假设'])}
cards=[]
for c,(slot,lo,hi,amt,peers,state,trigger,cancel,path) in params.items():
 r=lut[c];p=Decimal(str(r['close']));low=(p*(1+Decimal(str(lo))/100)).quantize(Decimal('.01'),rounding=ROUND_CEILING);high=(p*(1+Decimal(str(hi))/100)).quantize(Decimal('.01'),rounding=ROUND_FLOOR);cap=(p*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
 cards.append({'slot':slot,'code':c,'name':r['name'],'target_date':'2026-10-29','previous_close':float(p),'upper_reference':float(cap),'expected_gap_pct':[lo,hi],'expected_price':[float(low),float(high)],
  'stronger_than_expected':f'高于{high}至涨停参考{cap}；近一字只观察，越高不等于越可买','below_expectation':f'低于{low}取消本卡加速预案；三进四平开低开永久取消，不盘中翻红重启',
  'auction_amount_observation_min_cny':amt*10000,'threshold_origin':'THIS_REVIEW_HUMAN_SCENARIO_NOT_VALIDATED_OR_OPTIMAL',
  'auction_required':'09:25真实撮合成交额和成交量、未匹配量方向及手/股单位、09:20至25价格/买卖盘路径均需核验；无真实数据不得宣称达标。观察成交额门槛仅人工预案，满足不代表准买。',
  'actual_0925':None,'confirmation_peers':peers,'state':state,'after0930_0935':trigger,'cancel':cancel,'forward_1_2_3_5':path,'probability':'UNVALIDATED_NO_PERCENT_OUTPUT'})
write(O/'FINAL_SIX_EXECUTION_CARDS_20261029.json',{'formed_at':now(),'signal_date':'2026-10-28','target_date':'2026-10-29','cards':cards,
 'relay_final3':['000678','001219','601956'],'first_final3':['003028','603897','000670'],
 'independent_pre_model_relay3':['000678','002708','601956'],
 'final_change_reason':'光洋三板BASE43.0且当日成交3.58亿/换手3.71%，不满足最新容量特赦门，退出最终前三但保留独立偏晋级与题材风向观察。青岛补入仅观察席位，不当可买替代。襄阳五板不是三板通用低分否决对象，仍有尾盘队列及财报前置。',
 'orders_placed':0,'position_size':'No account/holdings supplied; no numeric account allocation inferred. T+1 stock cannot be sold same-day after purchase.'})
lines=['# 2026年10月28日复盘与10月29日预期预测','',
'全部时间为北京时间。事实采用10月28日收盘；盘后论坛与公告的实际取得时间另存。完整112只主板非ST返回联合池先独立审阅冻结，再读取今日BASE。既往模型分数及结果已知，不称完全盲审。',
'','## 1 大盘','',
'沪指3681.47涨0.91%，深成指11473.18涨2.02%，创业板2634.76涨3.40%。指数15:00分钟点与报道相符。媒体报告沪深成交1.63万亿元、较前日增加1675亿元、上涨超4100只。属于普涨修复；深创明显强于沪指，尚不能据一天反弹认定新主升。',
'','主情景：10月29日先检验修复后的兑现和题材分化，聚焦核心承接。沪指3670附近、今日低3652.37及创业板2604附近午后回撤区只作人工观察参照，不是已验证支撑位。指数继续走强但中位杀跌，仍降低接力预期；指数回落而襄阳/光洋稳定则支持独立抱团，仍不等于主力资金身份。',
'','## 2 情绪 梯队 题材与对手盘','',
'结论为强修复中的核心集中与中位淘汰并存。来源原涨停池72只、替代页69只、媒体71只存在口径/时点差异，不能混算。主板非ST返回联合池58封板，加福莱新材收盘价格涨停冲突附页1只，共59只收价涨停事件、48只炸板、5只跌停。该范围收价封板率59/107=55.14%，不认证交易所全覆盖。福莱原在炸板池，日线与15:00分钟均48.61=涨停参考，只确认价格事件，封单与末封时刻待核。',
'','昨日完整38只收价涨停样本今日均有价格反馈：首板26进5（19.23%）；二板8进2（25%）；四板3进1（33.33%）；五板九百未晋级。连板12只仅3只晋级，即25%。昨日4板多伦跌停、赛福先深水后反抽再回落，襄阳维持封板，证明同板位不是一起好。',
'','题材判断：机器人/轴承的襄阳5板—光洋3板—东贝/凯众2板—长城/振邦/宁波等首板形成层次；消费青岛强一字、茂业2板，但九百断板、桂发祥跌7.67%、阳光平收，不能把青岛独强当消费全面转强。芯片PCB容量金安/方正强，盈方/光华形成分支；传媒文投/掌阅强而新华、慈文未形成高度响应，因此不是全传媒共同主升。名称洋线百洋尾封、海洋王只修复、光洋早封，关系为假设而非同字自动加分。',
'','分钟对照：09:30赛福9.61低开、09:35约9.30，襄阳16.96开后09:35封17.61、光洋18.94持续，支持个股独立强弱差异；13:30至14:00创业板2630.23降2604.23、福莱48.61降47.80、赛福10.69降10.23，存在共同风险收缩解释，不能全归因同账户调仓。14:49至14:54百洋11.49至11.77，同时新华14.65至14.68、光洋和襄阳仍封，并非所有方向跷跷板互斥。节点前后1/3/5分钟已按同轴保存。',
'','对手盘：襄阳前几板形成较大账面浮盈，今天量约昨日0.50倍，新增接力可能来自高度稀缺及轴承群体认可，主要反证是缩量兑现、财报与后排失效；光洋与青岛低换手使卖压可能延后集中释放。只根据成交路径提出筹码假设，不声称真实账户持仓成本或卖出计划。',
'','当前最像情绪高标的独立候选是襄阳，题材中低位扩散的先发观察是光洋；二者可共同晋级，也可同时受市场分歧冲击。青岛是消费风向锚，不能与机器人永久设为互斥。',
'','尾盘加撤单审查：全池没有合格14:30—15:00同口径队列快照对。收盘单笔封单额不能算Seal_Delta，均保持未知。襄阳今日成交8.26亿元、早封符合容量和时点条件；但锁仓及完整一年连续跌停史待核，不能宣称特赦全条件已认证。用户所述27日尾盘加撤单作为线索记录，不倒造28日证据。',
'','三板专审：青岛为【三板-身位一字先锋】的形态观察，控制权转让仍筹划，待爆量分歧；光洋为近一字T板先锋观察而非15%换手真龙，3.71%换手和3.58亿成交不满足容量特赦，且一年历史前科/尾盘队列未完核。两股无已核尾盘偷袭红标签，未知不能填安全，也不因主营收入占比小直接否定市场认可。',
'','论坛只预读延边刺客1028（15:58）及1027（16:03）两篇正文，实际取得10月28日18:23左右，评论与图片未作为证据。采纳其“区分身位、群体持续性和替代方向”的问题；否决主控/敌对账户/固定玄学因果的断言。不复制作者分数、胜率或交易建议。','',
'## 3 连板完整BASE','',
'身份：生产冻结fixed_20261007T112017Z/relay/BASE，29字段，训练标签截止2026-08-31；目标是下一官方交易日收盘连续晋级。今日使用替代来源附层原公式推理，原模型未改。来源并非正式生产认证，历史27日归档缺部分行（37只限池），与完整事实38只分母不同；今日8只候选全部输出。这是来源受限的观察分，未校准，不能叫概率。','',
'| 排名 | 个股 | 板位 | 原分 | 原方向 | 独立判断与资格 |','|---|---|---:|---:|---|---|']
for r in sorted(base['rows'],key=lambda x:x['rank']):
 c=r['code'];ind=next(x for x in own['rows'] if x['code']==c)
 lines.append(f"| {r['rank']} | {r['name']} {c} | {lut[c]['boards']} | {r['score_display']:.1f} | {r['BASE_direction']} | 独立{ind['next_close_limit_up_call']}；{'三板低分且不满足容量特赦，仅风向观察' if c=='002708' else '未校准、替代源观察'} |")
lines+=['','模板新增潮汐/大龙模型未取得经本次资格核验的当日推理包，不给分、不套旧日期分、不照抄模板62.7%/翻倍等战绩。三板训练模型也不外推给五板、二板或首板。','',
'## 4 首板完整BASE','',
f'完整首板池{len(first)}只均已独立留档；原首板研究市值上限约93.27亿元及薄量风险代理保留。当前未形成合格62字段首板输入，首板BASE全部未输出；不能借连板分或人工优先序冒充。保存模型latest_S1，目标1进2，训练标签截止2026-08-28，身份与缺口见资格文件。','',
'| 首板 | 收价 | 流通市值亿元 | 原筛选 | 当日BASE |','|---|---:|---:|---|---|']
for r in sorted(first,key=lambda r:r.get('seal_first') or 999999):
 cap=r.get('reported_float_cap_cny');lines.append(f"| {r['name']} {r['code']} | {r['close']} | {cap/1e8:.2f}"+f" | {r['screen_status']} | 未输出 |" if cap else f"| {r['name']} {r['code']} | {r['close']} | 未知 | {r['screen_status']} | 未输出 |")
lines+=['','## 5 最终连板前三','',
'综合次序是襄阳轴承、青岛食品、东贝集团；均非无条件可以买。襄阳优先研究高度延续，财报与豁免证据未补齐前不发主动买入许可；青岛仅一字先锋观察；东贝二进三条件关注。独立冻结前本为襄阳、光洋、东贝；看BASE后依三板准入规则，光洋43.0且不满足最新容量条件退出最终前三，原独立偏晋级留存，青岛仅补观察席位，不代表自动改买。','',
'纯封板延续相对观察：青岛/光洋靠前，但一字难参与；高度和情绪带动独立第一观察为襄阳。最像龙头、次日封板和可买三者分开。东贝相对凯众更早封，凯众换手更足仍可共同晋级，不机械二选一。','',
'## 6 最终首板前三','',
'振邦智能、长城科技、盈方微。前两只分别是低位小盘控制器与早封业绩制造线，盈方是芯片分支替代路径。宁波东力保留观察：虽早封，但昨日冲高回落、今日量仅昨日0.38倍，更需核回本卖压。金安、方正、万向等大容量仅作板块锚，不为提高命中率加入首板潜龙主选。','',
'## 7 次日竞价与执行卡','',
'所有价格按10月28日收盘价和普通主板10%计算到分，未考虑未来除权/特殊交易安排，盘前须对券商涨跌停价。以下区间和成交额观察门槛均为本次人工预案，未验证最优；不承诺概率。09:25实际成交和未匹配量尚未发生，全部待核。','',
'| 代码 名称 | 收盘 | 涨停参考 | 09:25符合预期 | 人工成交额观察门槛 | 强于及低于预期 |','|---|---:|---:|---|---:|---|']
for c in cards:
 lines.append(f"| {c['slot']} {c['name']} {c['code']} | {c['previous_close']:.2f} | {c['upper_reference']:.2f} | {c['expected_price'][0]:.2f}—{c['expected_price'][1]:.2f} | {c['auction_amount_observation_min_cny']/10000:.0f}万元 | {c['stronger_than_expected']}；低于{c['expected_price'][0]:.2f}取消本预案 |")
for c in cards:
 lines+=['',f"### {c['slot']} {c['name']} {c['code']}",'',c['state'],'',
  '竞价核验：'+c['auction_required']+'确认对象：'+'、'.join(lut.get(v,{}).get('name',v) for v in c['confirmation_peers'])+'。',
  '',c['after0930_0935'],'','取消：'+c['cancel'],'','后续路径：'+'；'.join(c['forward_1_2_3_5'])+'。']
lines+=['','光洋额外观察卡：昨收18.94，四板参考20.83；由于三板近一字，竞价19.89—20.82（约+5%至涨停前）才保留强势观察，20.83一字不排。<=18.94按原规则永久取消该日三进四买入预案。它目前BASE43.0且无容量豁免资格，以上只验证强度，不授权追板。青岛/光洋一年内三板见顶后跌停前科与今日尾盘队列仍须补核。',
'','通用确认：09:20至09:25参考价和真实买盘不能持续下滑；09:25成交额门槛达标仍须核未匹配量方向、单位、盘口连续性及相对同伴。买一手数、委比或单一封单不能单独证明真抢筹。触发买入前先确认无新增停牌/减持/立案等硬风险，普通涨停限价单不冒充条件单。',
'','操作时点：不提前把未确认的隔夜单送入竞价。沪深09:20—09:25不接受撤单，09:25不能保证撤回已经申报或已成交委托；本报告“取消”优先指注销尚未触发/未送交易所的买入预案。已申报未成交按可撤时段及交易所确认处理，已成交不能靠撤单消除。当天买入A股受T+1限制，不能假设盘中止损可卖。任何主选取消均不自动换买下一名。',
'','后续交易日窗口：T+1 10/29、T+2 10/30、T+3 11/02、T+5 11/04。各股路径是条件预测，不是每天必涨的承诺；连续涨停、累计上涨、逐日上涨分别结算。本次不输出数值概率。','',
'## 8 昨日结果与来源保护','',
'昨日旧报告原连板摘要九百/青岛/襄阳，今日收价涨停为0/1/1；九百收涨5.11%但未六板，不能把上涨当晋级。旧首板茂业/秦安/江顺，今日涨停1/0/1；秦安收涨5.31%但未连板。茂业盘中回封不追溯推翻任何09:25已取消的方案；未取得实际成交及完整竞价单，不计算交易盈利。只追加事实，不改旧排名、旧方向或旧账本。',
'','行情原文、精确日线、49个分钟对照、64份公告索引、20份全文与两篇论坛原文均在本目录绑定真实取得时钟和哈希。媒体来源：[金十收评](https://flash.jin10.com/detail/20261028150034629800)、[财联社连板分析转载](https://finance.sina.com.cn/roll/2026-10-28/doc-iniwukpx9499564.shtml)；聚合池：[Toupage当日](https://toupage.com/stocks/limit-up?date=2026-10-28)。公告全文为发行人文件镜像，非用AI异动分析替代。','',
'撤单规则：[上交所2026年交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)、[深交所交易规则投教说明](https://www.szse.cn/www/investor/warning/t20190513_567105.html)。论坛：[延边刺客1028](https://m.tgb.cn/a/2wgkSCDy1bR)、[1027](https://m.tgb.cn/a/2weGjaPomt8)。仅其正文的相对关系思路用于提问，不把主控身份和资金动机当事实。']
report=O/'20261028复盘与1029预期预测_完整双榜.md'
report.write_text('\n'.join(lines),encoding='utf-8')
write(O/'FINAL_COMPARISON.json',{'formed_at':now(),'current_base_seen_after_freeze':True,'original_independent_sha':receipt['sha256'],'final_relay3':['000678','001219','601956'],'first3':['003028','603897','000670'],'final_relay_change':'三板光洋BASE43且不满足容量豁免，正式前三剔除，独立偏晋级不改；青岛补仅观察。','probabilities_validated':False,'new_fits':0,'new_orders':0})
verification={'formed_at':now(),'report_path':str(report),'report_sha256':sha(report),'full_rows':len(f['rows']),'firstboard_rows':len(first),'relay_rows':len(base['rows']),
 'independent_freeze_unchanged':sha(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')==receipt['sha256'],'model_loaded_only_after_freeze':True,'complete_relay_output':sorted(blut)==sorted(r['code'] for r in f['relay']),
 'first_model_output':0,'cards_complete':len(cards)==6,'source_hashes_valid':all(Path(p).exists() and sha(Path(p))==v for p,v in f['source_bindings'].items()),
 'protected_original_base':read(O/'BASE_verification.json')['protected_files_unchanged'],'fits':0,'orders':0,'old_predictions_modified':False,'seal_delta_available':False}
assert verification['independent_freeze_unchanged'] and verification['complete_relay_output'] and verification['cards_complete'] and verification['source_hashes_valid']
write(O/'DELIVERY_VERIFICATION.json',verification)
print(json.dumps({'verification':verification,'cards':[{k:c[k] for k in ['slot','code','name','previous_close','upper_reference','expected_price']} for c in cards]},ensure_ascii=False))
