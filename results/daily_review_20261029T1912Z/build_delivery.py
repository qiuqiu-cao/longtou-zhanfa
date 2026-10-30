from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP,ROUND_CEILING,ROUND_FLOOR
import json,csv,hashlib
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as s:json.dump(v,s,ensure_ascii=False,indent=2,allow_nan=False)
f=read(O/'facts/RESOLVED_FACTS_v2_SCORES_MASKED.json');own=read(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json');br=read(O/'BASE_scores_AFTER_INDEPENDENT.json');gr=read(O/'GEMINI_R7_SCORES.json');gate=read(O/'REGIME_GATE.json')
lut={r['code']:r for r in f['rows']};blut={r['code']:r for r in br['rows']};glut={r['code']:r for r in gr['rows']};ilut={r['code']:r for r in own['rows']}
first=[r for r in f['rows'] if r['kind']=='limit_up' and r['boards']==1]
q62=read(ROOT/'results/daily_review_20261028T1020Z/features/FIRSTBOARD_BASE_QUALIFICATION.json')
q62.update(formed_at=now(),signal_date='2026-10-29',target_date='2026-10-30',complete_firstboard_rows=len(first),candidate_codes=[r['code'] for r in first],
 reason='Current exact-day OHLC covers Sep18-Oct29; complete source-qualified 60-session price/gene and whole-market cycle inputs remain missing; no old-date score substituted.',scores_output=0,new_fits=0)
write(O/'features/FIRSTBOARD_BASE_QUALIFICATION.json',q62)
old44=ROOT/'research/first_board_leader_20261020T114600Z/first_board_leader_bundle.joblib'
write(O/'features/FIRSTBOARD_44_QUALIFICATION.json',dict(formed_at=now(),signal_date='2026-10-29',target_date='2026-10-30',model_identity='Retained 44-field first-board continuous-four-board height ranker',
 model_path=str(old44),model_sha256=sha(old44),status='NOT_SCORED_INPUT_NOT_QUALIFIED',complete_pool=28,scores_output=0,
 reason='Current normalized close bars lack full 60-session history, historical amounts for amount_ratio_5 and qualified whole-market median/breadth inputs. Bounded historical kline probe failed; no invented constants or all-column median fills.',new_fits=0))

params={
'001219':('L1',7,9.5,'仅消费高度和封板延续观察；一字不排队；控制权转让仍筹划。',['605338','603536','605179','002820'],
 '09:30一字只看风向；09:35若开板，本次仍以观察首个爆量承接为主，不因回封自动准买。',
 '竞价低于19.40取消本卡加速预期；平低开或食品跟随转弱，不追盘中反抽。',
 ['10月30日偏争五板，延续强于可参与性','11月2日若五板成功，首个爆量换手是核心风险','11月3日须有消费持续扩散，否则高度受限','11月5日只保留高标延续或断板再评情景，不预测逐日必涨']),
'601956':('L2',7,9.5,'【三板-身位一字先锋】形态观察；旧BASE与R7同向，但R7情绪闸门未过，本预案不放行三进四新买入。',['002708','002526','600592','000678'],
 '09:30及09:35均观察，不以临时高开或秒板绕过情绪闸门；开板检验真实承接，原卡不自动转成委托。',
 '09:25<=6.70永久取消本次三进四买入预案；低于7.17弱于一字加速预期；机器人高标继续负反馈则下调延续。',
 ['10月30日两模型偏争四板，但只作情绪观察','11月2日若四板成立，须检验能否突破旧高度约束','11月3日首次开板及累计兑现压力高，不预设缩量加速','11月5日能否成高标待多日同伴及承接验证']),
'603131':('L3',3,6,'独立题材判断与BASE21.8偏不晋级冲突；二进三分歧观察，非模型共同主选。',['600118','002565','603703','002943'],
 '09:30先核高开后的真实成交；09:35后守实际开盘区及成交均价，缩量回踩后放量首次回封17.55，且卫星/顺灏至少维持承接。',
 '竞价低于16.43取消本次强接力预案；>=8%高开或一字默认观察；放量跌破开盘区、两次冲板不封或航天同伴成片走弱取消。',
 ['10月30日人工偏争三板，BASE反对，需强确认','11月2日若三板成立，重新执行三板标签和完整同位审查','11月3日须在航天新线形成核心认可，否则按中位分歧','11月5日持续扩散才保留高度潜力，不用今天双概念保证连板']),
'003032':('F1',3,6,'首板人工第一观察；旧BASE62及44字段高度模型当日输入未过门，无首板模型分。',['600715','002862','603103'],
 '09:30若秒板不盲排；09:35后守实际开盘区与成交均价、回踩缩量再放量回封9.96，且AI应用同伴不出现严重负反馈。',
 '低于9.33取消本卡强接力预案；过热一字只观察；高开放量跌破开盘区、文投和实丰同时走弱且本股未体现独立性则取消。',
 ['10月30日人工偏争二板，先验证财报首板是否获接力','11月2日若二板成立，需证明同批领先而非一次性财报脉冲','11月3日争三板需重新评同位、标签和亏钱效应','11月5日只有AI应用扩散及本股承接持续才谈龙头潜力']),
'002565':('F2',2,5,'航天旧强股重新起板观察，非纯新周期；Q3亏损及旧筹码兑现须前置。',['603131','600118','603703','600501'],
 '09:30先看上海沪工/卫星方向确认；09:35后守开盘区及成交均价，回踩缩量再放量回封8.95；不因同伴涨停直接追涨。',
 '低于8.31取消本次加速预案；竞价虚高后回落、航天批量炸板、成交放大但价格滞涨取消；旧员工持股解锁不冒称实际卖出。',
 ['10月30日人工偏争二板，需航天延续','11月2日若二板成立，检验旧高点供给能否被吸收','11月3日高位旧筹码兑现与新题材认可竞争加剧','11月5日航天若一日游则失效；若成为持续主线再评高度']),
'600592':('F3',2,5,'首板第三仅观察；轴承低位关系待验，非已认证补涨龙。',['002708','000678','601956'],
 '09:30及09:35先观察光洋/襄阳修复是否持续；本次不因低位轴承概念直接下单。若形成新可核独立强度，再另记判断，不回改本卡。',
 '低于17.76取消本卡强势观察；光洋/襄阳继续走弱、本股仅冲高无承接，取消补涨假设；不自动买其他首板。',
 ['10月30日仅观察二板验证，未给正向买入结论','11月2日若二板成立且旧轴承高标修复，才重评相对地位','11月3日低换手后的集中卖压为主要反证','11月5日不能靠早封与龙字确认龙头身份'])}
cards=[]
for c,(slot,lo,hi,state,peers,trigger,cancel,path) in params.items():
 r=lut[c];p=Decimal(str(r['close']));low=(p*(1+Decimal(str(lo))/100)).quantize(Decimal('.01'),rounding=ROUND_CEILING);high=(p*(1+Decimal(str(hi))/100)).quantize(Decimal('.01'),rounding=ROUND_FLOOR);upper=(p*Decimal('1.1')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
 cards.append(dict(slot=slot,code=c,name=r['name'],signal_date='2026-10-29',target_date='2026-10-30',previous_close=float(p),upper_reference=float(upper),expected_gap_pct=[lo,hi],expected_price=[float(low),float(high)],
  stronger_than_expected=f'{high+Decimal(".01")}至{upper}，近一字只观察；沪工/首板高开超过8%不直接放行',
  auction_amount_reference_cny=round(r['amount_cny']*.01,2),threshold_origin='UNVALIDATED_HUMAN_SCENARIO_1_PERCENT_SIGNAL_AMOUNT_OBSERVATION_NOT_OPTIMAL_OR_BUY_PERMISSION',
  auction_check='核真实09:25撮合价、成交额/量及手股单位、未匹配量方向；与09:20后多时点价格及买卖盘比较，不把单张静态图解读成撤单全过程。参考成交额线未验证，过线仍须承接。',
  actual_auction=None,unmatched_volume=None,confirmation_peers=peers,state=state,after0930_0935=trigger,cancel=cancel,forward_1_2_3_5=path,probability='UNVALIDATED',
  order_protocol='截图终审后评估尚未触发的盘中条件；未提交订单；09:25已撮合成交不能撤回',history_and_1430_queue='UNAVAILABLE_KEEP_PENDING'))
write(O/'FINAL_SIX_EXECUTION_CARDS_20261030.json',dict(formed_at=now(),signal_date='2026-10-29',target_date='2026-10-30',cards=cards,
 relay_final3=['001219','601956','603131'],first_final3=['003032','002565','600592'],
 final_selection_basis='Combined observation, not new calibrated probability or model reranking. Qingdao continuation anchor; Dongbei native two-model leader with gate closed; Hugong independent dissent observer. Shifeng score61.9 retained in complete model table; thin one-price liquidity prevents treating it as an executable substitute.',
 pre_model_relay3=own['relay_top3'],pre_model_first3=own['first_board_top3'],orders=0,new_fits=0))

combined=[]
for r in f['rows']:
 c=r['code'];b=blut.get(c,{});g=glut.get(c,{})
 combined.append(dict(code=c,name=r['name'],kind=r['kind'],boards=r.get('boards'),close=r.get('close'),change_pct=r.get('change_pct'),seal_first=r.get('seal_first'),seal_last=r.get('seal_last'),
  amount_cny=r.get('amount_cny'),turnover_pct=r.get('turnover_pct'),float_cap_cny=r.get('reported_float_cap_cny'),screen_status=r.get('screen_status'),
  BASE_rank=b.get('rank'),BASE_score=b.get('score_display'),BASE_direction=b.get('BASE_direction'),Gemini_R7_capacity_score=g.get('score_arm3_capacity'),Gemini_R7_3to4_score=g.get('score_3to4'),
  Gemini_applicability='NATIVE_3_BOARD' if c in glut else 'NOT_APPLICABLE',firstboard_BASE62_score=None,firstboard_44_score=None,independent_call=ilut[c]['next_close_limit_up_call'],independent_reason=ilut[c]['reason']))
with (O/'完整102只_原BASE_Gemini与独立判断.csv').open('x',encoding='utf-8-sig',newline='') as s:
 w=csv.DictWriter(s,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)

# Same-clock node windows; no account-level flow claims.
minute=read(O/'minutes/capture_receipt.json');extra=read(O/'minutes/sz000678_additional_v2.json');allm=minute['records']+[extra]
mlut={r['symbol']:{x['time']:x['price'] for x in r.get('rows',[])} for r in allm}
def clock(hhmm,delta):
 n=int(hhmm[:2])*60+int(hhmm[2:])+delta
 return f'{n//60:02d}{n%60:02d}'
windows=[]
for node in ['0950','1030','1055','1108','1457']:
 for delta in [-5,-3,-1,0,1,3,5]:
  t=clock(node,delta);windows.append(dict(node=node,offset_minutes=delta,time=t,prices={s:mlut.get(s,{}).get(t) for s in ['sh000001','sz399006','sz000678','sz002708','sh601956','sh600715','sh603131','sz002565','sz003032']}))
write(O/'minutes/NODE_WINDOWS_1_3_5.json',dict(formed_at=now(),rows=windows,limitation='One-minute sampled transaction prices cannot reconstruct tick queues or account transfers. Outside available times remains null.'))

lines=['# 2026年10月29日行情复盘及10月30日竞价预案','',
 '信号为10月29日收盘，目标为10月30日。北京时间10月30日凌晨形成。收盘独立判断先留档，随后加载当日旧BASE和Gemini R7；既往分数与结果已知，不称全程盲审。盘后新公告独立追加，不改收盘预测。',
 '','大盘','',
 '沪指3651.43，跌0.82%；深成指11138.20，跌2.92%；创业板2517.59，跌4.45%。腾讯15:00分钟点与[博时市场点评](https://finance.sina.com.cn/jjxw/2026-10-29/doc-iniwwtuv8779993.shtml)一致。沪深成交约1.56至1.57万亿元为媒体不同四舍五入口径，较昨日缩量。全市场涨跌家数有口径差异，本次未取得独立全证券快照，不拿论坛约数做精确分母。',
 '','判断：昨日普涨修复后出现科技成长集中兑现，短线进入退潮中的局部抱团。10月30日主情景是先检验止跌和高低切换，不能因盘中反抽就上调为主升。创业板若持续失守今日尾盘区而跌停扩散，继续降低接力；沪指站回今日开盘3676.77、创业板稳定且中位不核才支持修复加强。以上是人工场景参考。',
 '','情绪 梯队与题材','',
 '东方财富当日返回涨停39、炸板25、跌停41；经主板非ST范围筛选为涨停38、炸板23、跌停41，共102只，均有精确当日日线交叉核验，无池价冲突。全市场口径封板率39/64=60.94%，主板返回池38/61=62.30%；两种分母分开。替代页超时，未认证交易所全覆盖。今日连板10只，4板2、3板2、2板6；首板28只。',
 '','| 昨日板位 | 完整样本 | 今日晋级 | 未知 | 含义 |','|---|---:|---:|---:|---|']
feedback=read(O/'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json')
for h,v in feedback['layers'].items():
 desc={'1':'一进二承接弱','2':'中位有局部接力','3':'旧三板双晋级，但光洋封板质量明显退化','5':'襄阳断板，高度受阻'}[h]
 lines.append(f"| {h}板 | {v['total']} | {v['promoted']}/{v['known']}（{v['promoted']/v['known']:.2%}） | {v['unknown']} | {desc} |")
lines += ['',
 '旧高位与新题材须分开看：襄阳+6.47%断板、光洋虽晋级但27.90%换手且15:00末封；东贝一字未带动襄阳守住六板。机器人存在封板与炸板两组，不能用两只强票认定全板块修复。消费青岛一字、食品首板扩散，但桂发祥跌停，表明高低位并非共赢。文投三板与横店首板是AI/影视替代方向假设；传媒老龙是否响应须另核。',
 '','航天卫星是明日优先验证的新方向，上海沪工二板、顺灏/盛洋/中国卫星/航天晨光等首板构成群体；[上海六部门卫星互联网行动方案](https://sheitc.sh.gov.cn/cyfz/20261026/96dc7fe4e44e4dcf90b6fcc411bef863.html)已核到官方正文，发布日期10月26日，不能写成29日才发布。题材共现支持资金关注假设，不证明各股收入兑现，也不证明从机器人同账户调仓。',
 '','同轴节点：沪工09:40约14.78，09:50约15.80、09:51附近到涨停；同时创业板从2575.03到2548.55，支持该股相对指数抗跌。10:30光洋20.83、襄阳19.36，11:00分别19.30和19.06，支持机器人高位共同卖压，不能说一股吸干另一股。文投10:30/11:00仍2.25/2.24，末封11:08:26；光洋14:57为20.82、15:00为20.83，只确认尾盘回封，不能确认主力追加封单。节点前后1/3/5分钟完整表另存。',
 '','对手盘：光洋经历约7.51倍昨日成交量及多次开板，可能同时存在新接力与旧浮盈兑现；不是全部换手都代表锁仓。沪工昨日成交持有者面临二板浮盈兑现，新买方可能选择其航天/机器人双线及今日相对抗跌；中国卫星/顺灏走弱且本股不响应会否证核心假设。传智的新增接力理由是财报与早封；顺灏有10月21日旧涨停及更高历史成交区的潜在供给；龙溪的轴承低位假设受光洋/襄阳影响。持仓成本和具体资金身份均未认证。',
 '','三板标签：东贝为【三板-身位一字先锋】形态观察，实际日线一字但R7的原始09:25:01时钟代理因首封09:25:02记0，保留该定义差异，不私改模型。文投仅为【三板-换手核心真龙】待核候选，容量与早封满足，但完整历史风险、尾盘加单及题材地位尚未获认证，不能给正式绿标。文投BASE22.6且无已核特赦证据，不进入最终前三。两股次日平低开永久取消本次三进四计划。',
 '','全池无已核14:30与收盘同口径封单对，Seal_Delta均未知；未认证任何一股尾盘加单特赦或撤单超过30%。论坛提到山东矿机漏单仅列线索。',
 '','限定预读两篇：春风吹又生tjy《10月29日涨停复盘》，15:54发布；我最贪婪《2026年10月29日超短复盘》，16:14发布。实际取到正文为北京时间10月30日03:11。评论不纳入作者正文。采纳分清硬一字与可买性、检查替代方向持续性的思路；不移植6/61分母、70%概率、账户动机和开板30次等未核数值。详见独立留档。',
 '','连板完整旧BASE及Gemini对照','',
 '旧BASE：fixed_20261007T112017Z/relay/BASE，29字段，训练标签截至2026年8月31日，目标为下一官方交易日收盘连续晋级。原权重和原0.5方向保持。本次为当日主池及腾讯价格、混合历史来源的附层观察分，正式来源门未过；分数未校准，不当概率。',
 '','| 原BASE排名 | 个股 | 板位 | BASE分 | 原方向 | Gemini R7容量分 | 独立判断 |','|---|---|---:|---:|---|---:|---|']
for r in sorted(br['rows'],key=lambda x:x['rank']):
 c=r['code'];g=glut.get(c);gs=f"{g['score_arm3_capacity']*100:.1f}" if g else '不适用'
 lines.append(f"| {r['rank']} | {r['name']} {c} | {r['boards']} | {r['score_display']:.1f} | {r['BASE_direction']} | {gs} | {ilut[c]['next_close_limit_up_call']} |")
lines += ['',
 f"Gemini R7：保留的第七轮容量龙模型，原生仅三板；采用fold4已存五个预测头，训练截止2025年11月30日。东贝容量综合{glut['601956']['score_arm3_capacity']*100:.1f}、三进四头{glut['601956']['score_3to4']*100:.1f}；文投分别{glut['600715']['score_arm3_capacity']*100:.1f}、{glut['600715']['score_3to4']*100:.1f}，均为未校准分。情绪闸门{gate['score']*100:.2f}低于55，通过状态为否；因此不把东贝两模型领先升级为本次三进四买入。只对原生范围解释闸门，不宣称已验证全市场择时。",
 '','拥挤字段缺失使用各头已存训练中位数，原始缺失与插补明细保留；题材特征按训练时旧行业映射，未把当前航天叙事偷偷塞入冻结模型。其他板位不强行评分。五头各444行保存样本重放最大差小于1e-12；此项仅证明加载复现，不是新预测业绩。',
 '','首板完整旧BASE资格','',
 '当日首板28只全部保留。原BASE62目标是一进二；旧44字段高度模型目标是首板后连续达到四板，两者不同。本次62字段全周期/基因输入和44字段60日/历史成交额/全市场背景输入未齐，历史K线补取失败，均未输出当日分数，也未使用被删除的54字段模型。下面完整首板顺序为独立人工观察，不能冒充BASE排名。原流通市值上限精确为9,326,859,896.52元，超限与薄量代理仍列完整对照。',
 '','| 首板 | 收盘 | 首封至末封 | 换手 | 流通市值亿元 | 原筛选 | 当日BASE62及44分 |','|---|---:|---|---:|---:|---|---|']
for r in sorted(first,key=lambda x:(ilut[x['code']]['priority_rank'] or 999,x.get('seal_first') or 999999)):
 lines.append(f"| {r['name']} {r['code']} | {r['close']:.2f} | {r['seal_first']}至{r['seal_last']} | {r['turnover_pct']:.2f}% | {r['reported_float_cap_cny']/1e8:.2f} | {r['screen_status']} | 未输出 |")
lines += ['',
 '最终连板前三 综合观察排序','',
 '1. 青岛食品：封板延续及消费高度锚，旧BASE第一；仅观察一字与首次爆量开板。控制权转让方案未定，不能当已完成收购。',
 '2. 东贝集团：三板中旧BASE和R7同向，作为机器人一字风向；R7闸门未过，本预案不新增三进四买入。',
 '3. 上海沪工：独立航天延续假设保留，BASE只有21.8，明确为模型分歧观察，须强竞价与真实回封；不称共同推荐。',
 '','实丰文化BASE61.9原第二完整保留，封板延续靠前；流通21.76亿、换手1.67%、成交0.36亿的一字未检验兑现，命中既存薄量代理，不能当可成交替补。光洋末封太晚且财报新风险，文投三板低分且无特赦，都退出本次最终前三。独立留档原文投第三在模型与盘后信息后撤销，原记录不回写。',
 '','最像当前高度锚是青岛，是否成为真正情绪总龙仍待换手和群体持续性；最有条件观察的分歧接力是沪工，属于低分反证路线。最高板、封板延续、可参与接力分别判断。',
 '','最终首板前三 人工观察排序','',
 '1. 传智教育：09:32早封、零报告开板、流通25.75亿；Q3归母4327.90万元、扣非4174.74万元，财报与AI教育是具体接力理由。反证是财报一次性脉冲、AI应用同伴失效。',
 '2. 顺灏股份：航天方向9.69亿容量、09:56前稳封；原市值范围内，但有旧涨停筹码与Q3亏损3732.03万元，须看到题材持续与承接。',
 '3. 龙溪股份：09:30早封、零报告开板；Q3扣非同比增长48.53%，归母却下降20.98%，仅作低位轴承观察。光洋/襄阳继续走弱则取消补涨假设，不用龙字或早封认证身份。',
 '','盛洋是航天/名称分支观察，Q3亏损1572.87万元；中国卫星、西部材料、一鸣食品等超原市值上限，保留对照，不为凑首板主选纳入。',
 '','六只逐股竞价执行卡 目标10月30日','',
 '下面价格区间及成交额观察线均为本次人工预案，未经胜率或最优阈值验证。符合预期不等于可以买。实际09:25尚未发生，撮合量与未匹配量均未知；待用户截图直读，不要求任何同花顺条件公式。涨停价为正常10%四舍五入参考，盘前仍核除权和券商正式限价。',
 '','| 个股 | 29日收盘 | 30日涨停参考 | 符合预期 | 更强至参考涨停 | 成交额人工观察线万元 | 当前状态 |','|---|---:|---:|---|---|---:|---|']
for c in cards:
 lines.append(f"| {c['name']} | {c['previous_close']:.2f} | {c['upper_reference']:.2f} | {c['expected_price'][0]:.2f}至{c['expected_price'][1]:.2f} | {c['stronger_than_expected']} | {c['auction_amount_reference_cny']/1e4:.1f} | {'分歧条件观察' if c['code']=='603131' else '人工条件关注' if c['code'] in {'003032','002565'} else '仅观察'} |")
for c in cards:
 lines += ['',f"{c['name']} {c['code']}：{c['state']}",f"确认对象：{'、'.join(lut[p]['name'] if p in lut else p for p in c['confirmation_peers'])}。{c['auction_check']}",f"盘中：{c['after0930_0935']}",f"取消：{c['cancel']}",f"后续路径：{'；'.join(c['forward_1_2_3_5'])}。"]
lines += ['',
 '盘后公告补充 已与收盘信号分开','',
 '实际取得北京时间10月30日03:17左右。光洋Q3归母107.70万元、同比下降89.88%，Q3扣非亏损3409.79万元，维持高位风险观察，不因大换手直接特赦。文投新增风险提示，前三季归母亏损3754.58万元、营收同比下降29.51%，叠加低分和闸门未过，本次不接三进四。青岛再次提示控制权方案价格等尚在协商，参与状态维持仅观察；沪工异动公告未披露新增重大事项，不把它当未来无风险保证。公司公告PDF及真实取得时钟在postclose/notices内。',
 '','[光洋三季报](https://pdf.dfcfw.com/pdf/H2_AN202610291831175567_1.pdf)；[文投风险提示](https://pdf.dfcfw.com/pdf/H2_AN202610291831180793_1.pdf)；[青岛风险提示](https://pdf.dfcfw.com/pdf/H2_AN202610291831173674_1.pdf)；[沪工异动公告](https://pdf.dfcfw.com/pdf/H2_AN202610291831179149_1.pdf)。上述为公司公告公开镜像，非交易所网站主机认证。',
 '','执行边界：09:25已撮合成交不可撤回，本卡的取消指不提交新的盘中买入或注销尚未触发的内部计划。没有下单。普通涨停限价委托不等同条件触发单；普通A股当日新买入不能当日卖出，盘中失效不代表可当天止损。主选取消不自动换下一只。',
 '','完整102只含未知与失败样本的CSV、原模型分与方向、六卡JSON、来源时钟、独立留档及保护哈希均在本目录；旧模型、旧预测和账本未改，新增拟合0。未来1/2/3/5交易日分别为10月30日、11月2日、11月3日、11月5日，尚未成熟，不给验证概率。']
report=O/'20261029复盘与1030预案_旧BASE与Gemini模型.md'
with report.open('x',encoding='utf8') as s:s.write('\n'.join(lines)+'\n')

# Revalidate all protected hashes actually captured before inference, including old predictions.
protected=read(O/'BASE_inference_BEFORE_LOAD.json')['protected_hashes']|read(O/'PROTOCOL_BEFORE_SCORES.json')['protected_and_source_hashes']
checks={p:sha(Path(p))==h for p,h in protected.items()};assert all(checks.values()),'Protected source changed'
assert sha(O/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')==read(O/'INDEPENDENT_FREEZE_RECEIPT.json')['sha256']
assert len(combined)==102 and len(first)==28 and len(br['rows'])==10 and len(gr['rows'])==2
assert all(r['qualification']=='EXACT_DAY_OHLC_OBSERVED' for r in f['rows'])
assert br['signal_date']=='2026-10-29' and br['target_date']=='2026-10-30'
assert gr['signal_date']=='2026-10-29' and gr['target_date']=='2026-10-30'
assert not gate['pass_gate']
write(O/'DELIVERY_VERIFICATION.json',dict(formed_at=now(),report=str(report),report_sha256=sha(report),full_rows=102,relay_rows=10,first_rows=28,
 firstboard_models_current_scores=0,gemini_native_three_board_rows=2,gemini_gate_pass=False,protected_hash_checks=checks,
 old44_model_sha256=sha(old44),original_prediction_and_ledgers_unchanged=True,all102_exact_ohlc=True,new_fits=0,orders=0,source_gate_pass=False,
 files={str(p):sha(p) for p in [report,O/'完整102只_原BASE_Gemini与独立判断.csv',O/'FINAL_SIX_EXECUTION_CARDS_20261030.json']}))
print(json.dumps(dict(report=str(report),rows=102,protected=len(checks),cards=[dict(name=c['name'],close=c['previous_close'],upper=c['upper_reference'],expected=c['expected_price']) for c in cards]),ensure_ascii=True))
