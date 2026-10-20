"""Readable review with complete denominators, original calls and failed study."""
from pathlib import Path
from datetime import datetime,timezone
import html,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import short_term_thinking_dataset_builder as b
OUT=Path(__file__).resolve().parent
def read(p):return b.read(OUT/p)
def label(s):return {'YES':'偏晋级','PROMOTE':'偏晋级','NO':'偏不晋级','FAIL':'偏不晋级','UNKNOWN':'未知／观察',None:'未提供逐股判断'}.get(s,s)
def direction(s):return {'YES':'累计偏上','NO':'累计偏下','UNKNOWN':'累计未知'}.get(s,s)
def table(headers,rows):
 def escape(x):return str(x).replace('|','／').replace('\n','；')
 return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(escape(x) for x in r)+'|' for r in rows])
def main():
 own=read('independent_full39_BEFORE_MODEL_AND_GEMINI.json');facts=read('facts/qualified_current_facts_SCORES_MASKED.json');base=read('BASE_scores_AFTER_INDEPENDENT.json');study=read('model_optimization/fixed_research_result.json')
 qa=read('verification_FINAL.json');old=read('settlement/15_to16_full6_APPEND_ONLY.json');old7=read('settlement/16_to19_full7_APPEND_ONLY.json');md=read('settlement/15_multiday_mature_asof19_APPEND_ONLY.json')
 vm={r['code']:r for r in own['rows']};fm={r['code']:r for r in facts['rows']};bm={r['code']:r for r in base['rows']};gemini_path=OUT/'gemini/gemini_independent_20261019_BEFORE_CODEX_MODEL.md'
 gm={'003029':'YES','002889':'YES','002085':'YES','600178':'YES','603197':'NO','603348':'UNKNOWN','600860':'YES','000576':'UNKNOWN','600148':'YES','002724':'NO','603278':'NO'}
 comparison=[]
 for r in sorted([r for r in own['rows'] if r['boards']>=2],key=lambda r:(r['priority_rank'] if r['priority_rank'] is not None else 99,r['code'])):
  c=r['code'];z=bm[c];comparison.append(dict(code=c,name=r['name'],boards=r['boards'],own_original=r['next_close_limit_up_call'],own_priority=r['priority_rank'],gemini_original=gm[c],gemini_postdiscussion='YES' if c=='603197' else gm[c],BASE_score=z['raw_score'],BASE_direction=z['BASE_direction'],BASE_rank=z['rank'],own_vs_BASE_conflict=z['conflict'],current_result='PENDING_2026-10-20'))
 pd.DataFrame(comparison).to_csv(OUT/'完整11只_双方原判断与BASE冲突.csv',index=False,encoding='utf-8-sig')
 # Correct one self-peer listing by an append, without touching the frozen view.
 peers=vm['600487']['complete_same_height_peers'];fixed=[c for c in peers if c!='600487']
 b.write_new(OUT/'minor_scope_corrections_APPEND_ONLY.json',dict(formed_at=b.now(),original_independent_sha256=b.sha(OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json'),corrections=[dict(code='600487',field='complete_same_height_peers',original=peers,corrected=fixed,reason='Remove the issuer itself; all27 other first-board peers retained; no direction/rank/horizon change'),dict(source='forum/ale_1019.raw.html',original_receipt_status='TEXT_CAPTURED_REQUIRES_AUTHOR_AND_TIME_REVIEW',corrected='HTTP200_EMPTY_BODY_NOT_QUALIFIED_ORIGINAL_HTTP_TEXT',other_source='Full article read in web-search rendered result; source publication/updates kept separate; not raw HTTP evidence')],all_directions_unchanged=True))
 full=[]
 for r in facts['rows']:
  c=r['code'];v=vm.get(c,{});full.append(dict(date='2026-10-19',code=c,name=r['name'],kind=r['kind'],boards=r['boards'],close=r['close'],turnover_pct=r['turnover_pct'],amount_cny=r['amount_cny'],one_price=r['one_price'],own_direction=v.get('next_close_limit_up_call'),own_priority=v.get('priority_rank'),role_hypothesis=v.get('role_hypothesis'),support=v.get('support'),rebuttal=v.get('counter_evidence'),cancel=v.get('cancel_conditions'),source_qualification=r['qualification']))
 conflict=read('facts/603337_source_conflict_APPEND_CORRECTION.json');full.append(dict(date='2026-10-19',code='603337',name='杰克科技',kind='SOURCE_CONFLICT_PRIMARY_FAILED_VS_CLOSE_AT_UPPER',close=conflict['daily']['close']/100,source_qualification='UNKNOWN_POOL_ASSIGNMENT_EXACT_DAILY_PRESENT',own_direction='UNKNOWN'))
 pd.DataFrame(full).to_csv(OUT/'完整80只_收盘池炸板跌停与来源冲突.csv',index=False,encoding='utf-8-sig')
 flat=[]
 for v in own['rows']:
  for h,x in v['horizons'].items():flat.append(dict(code=v['code'],name=v['name'],horizon=h,target_date=own['official_horizon_targets'][h],**x))
 pd.DataFrame(flat).to_csv(OUT/'完整39只_2_3_5日原预测含弃判.csv',index=False,encoding='utf-8-sig')
 top=sorted([v for v in own['rows'] if v['priority_rank']],key=lambda r:r['priority_rank'])
 top_table=table(['优先序','股票／代码','20日收盘涨停','21日：2日累计','22日：3日累计','26日：5日累计','路径与主要风险'],[[v['priority_rank'],v['name']+' '+v['code'],label(v['next_close_limit_up_call']),direction(v['horizons']['2']['cumulative_up_call']),direction(v['horizons']['3']['cumulative_up_call']),direction(v['horizons']['5']['cumulative_up_call']),{'003029':'先争4、5板，22日前后换手分歧；业务收入小，一字开板承接风险','603197':'先争3、4板再换手；特定飞行汽车订单未核，不由一价推断明日必开','002085':'容量换手推进，21日起可能震荡；需汽车／低空同组持续','002889':'大换手后争4板，后续剧震风险较高；公告索引本轮超时','600487':'业绩驱动趋势观察，20日封板未知；近百亿成交带来接力量要求'}[v['code']]] for v in top])
 lines=['# 10月19日联合复盘、20日优先候选与模型优化实测','',
 '本次实际通过 Antigravity 原生界面与 **Gemini 3.8 Flash High** 协作。Codex先独立留档完整39只收盘候选，再读取Gemini本批结论和BASE分数。双方排序、分歧与核验纠错分别保存。这里的方向是独立预测，**概率未验证**；没有把分数转成涨停概率。','',
 '## 先给下一交易日结论','',
 '**20日相对优先：①吉大正元003029；②保隆科技603197；③万丰奥威002085。另观察东方嘉盛002889、亨通光电600487。** 重点是新高度／低空汽车分支的前排延续，预计后排分化；不押老传媒和消费集体反包。','',top_table,'',
 '表内2／3／5日均指对应官方交易日收盘相对19日收盘的累计方向。逐日上涨、连续涨停另列，不能用累计涨回来算每天都涨。以上不是已经成交的持仓路径。','',
 '## 盘面：指数修复与短线分化并存','',
 '沪指收3730.92点、涨0.39%；深成指涨0.13%；创业板指跌0.94%。两市成交14375亿元，比上日少1179亿元；Wind口径两市及北交所3126只下跌。银行与午后机器人较强，农业、乳业及多只传媒股回落。该组合支持“缩量轮动、前排续强后排淘汰”的情景，不能据沪指翻红推断普涨修复。[收盘报道](https://www.thepaper.cn/newsDetail_forward_34280548)。','',
 '精确日期收盘页17:25快照：46封板、20炸板、22跌停；主板按简称非ST过滤后为39封板（11连板＋28首板）、18炸板、22跌停。完整返回页及HTTP字节、哈希、请求与收件时间已保存。158只相关股票的不复权日线和19日报价逐份重建核对。来源覆盖是供应商页面范围，不冒称交易所全市场完整认证。','',
 '东方财富原炸板源另有杰克科技603337，但它的日线／报价显示收盘34.46等于上限、替代页又未列出；单独保留为来源冲突。它有真实日线，不能称未采到，也不混入18只已一致核实炸板。页上“46封板＋20炸板”的66不能称66只收盘涨停。','',
 '今天3板仅吉大正元、东方嘉盛；2板9只为甘化科工、万丰奥威、海洋王、长春一东、东安动力、京城股份、保隆科技、大业股份、文灿股份。9只都保留，不把汽车、照明、氢储气、信息安全混成相同直接业务。','',
 '## 重点候选的支持、反证与取消条件','']
 for v in top:
  c=v['code'];r=fm[c];lines += [f"### {v['priority_rank']}. {v['name']}（{c}）",'',
   f"19日收盘{r['close']:.2f}元，{r['boards']}板；成交{r['amount_cny']/1e8:.3f}亿元，换手{r['turnover_pct']:.2f}%，振幅{r['amplitude_pct']:.2f}%。{('全日一价' if r['one_price'] else '全日非一价')}，原日线成交量相对16日为{r['volume_ratio_previous_official']:.3f}倍。",'',
   v['support'],'',v['counter_evidence'],'',f"**20日取消交易关注条件：** {v['cancel_conditions']}",'',
   table(['官方窗口','累计方向','逐日上涨判断','连续涨停判断','最可能路径'],[[own['official_horizon_targets'][h],direction(v['horizons'][h]['cumulative_up_call']),v['horizons'][h]['every_day_up_call'],v['horizons'][h]['consecutive_limit_ups_call'],v['horizons'][h]['path']] for h in ['2','3','5']]),'']
  if c=='003029':lines += ['吉大的AI安全、抗量子及低空安全技术储备由公司公告确认，但相关收入尚小、未规模化；半年报亏损。它的优先顺序来自情绪与价量判断，不来自虚构的业绩兑现。[发行人异常波动公告镜像](https://pdf.dfcfw.com/pdf/H2_AN202610181830660042_1.pdf)。','']
  if c=='600487':lines += ['亨通预告前三季归母净利润59.696207—63.696207亿元、同比增长151.27%—168.10%，未经审计。这是业绩催化，不是次日封板或未来收益保证。[发行人业绩预告镜像](https://pdf.dfcfw.com/pdf/H2_AN202610161830617754_1.pdf)。','']
 lines+=['交易层只列待确认条件：一价不代表能买到；非一价仍需实际竞价、开盘承接及回封证据。24.19／32.33／10.90／15.74／50.25是19日收盘参考位，不能叫均线。当天新买股票受T+1约束，不能把当日跌破参考位写成一定能卖出的止损。原YES预测不会因取消交易关注而从结算样本删除；本次没有新增实际或纸面成交证据。','',
 '## 双方独立意见与原模型冲突','',
 'Gemini原Top3为吉大正元、东方嘉盛、万丰奥威，观察东安动力、保隆科技；Codex原Top3为吉大、保隆、万丰。共同重点是吉大与万丰；最主要原始分歧是保隆：Codex认为前排续板更值得优先，Gemini原认为开板后偏断板。Gemini在看到Codex判断与事实纠错后，追加上修为偏换手分歧晋级。原NO保留，会后上修不是新的独立盲审共识，分别到期检验。','',
 table(['代码／简称','板数','Codex原判断','Gemini原判断','BASE分数（非概率）','BASE方向／名次'],[[r['code']+' '+r['name'],r['boards'],label(r['own_original']),label(r['gemini_original']),f"{r['BASE_score']*100:.2f}",label(r['BASE_direction'])+'／'+str(r['BASE_rank'])] for r in comparison]),'',
 'BASE在双方独立记录后才加载旧权重，未重训。它前3为长春一东、海洋王、京城，吉大第4；保隆、东方、万丰均低于原0.5方向线。本次保留这些冲突，没有看完模型再改写原独立排序。BASE来源为新日期替代页与原公式补充推断，原正式来源门未通过。','',
 'Gemini稿中“一字明天必开板”“须15%—20%换手才真封”“百洋跌停导致海洋王羊字溢价衰退”“19日收盘价是均线”等说法没有合格证据，不采纳为事实或执行门槛。公开页面所报封单不是已认证实时队列，不能据此称能买到或证明锁仓；原文与追加纠正分别保存。追加稿仍将1.066倍成交量写为成交额并称缩量，这不采纳：本批只核实日线原生量39519／37081＝1.06575，16日逐股成交额本批未知。也不将“亨通高于保隆”扩为全市场封单最大已核。Gemini自填标题时钟存在早于实际到达的矛盾，使用本批实际读到时刻作收据，不认证其自报精确成稿时刻。','',
 '## 旧判断到期结算：失误全部保留','',
 'Codex 15→16日原Top3慈文、百洋、富春收盘涨停 **0／3**；慈文盘中触板，不能算收盘封板。完整6只里独立明确判断4只，正确0／4，另2只弃判；实际正例冰山、新华文轩均未捕捉。同期BASE为2／6正确、Recall为0，不能只拿独立判断比它更差就说模型已达标。','',
 table(['股票','原独立判断','16日收盘','16日涨停价','触板','收盘涨停','15→16日涨幅'],[[r['code']+' '+r['name'],label(r['original_direction']),f"{r['target_close']:.2f}",f"{r['upper_price']:.2f}",r['actual_intraday_touch'],r['actual_close_limit_up'],f"{r['return_pct']:+.2f}%"] for r in old['rows']]),'',
 '15日另一份已看模型的2／3／5日情景也独立结算：至19日T+2，原Top3累计上涨只有慈文成立，百洋、富春未成立；不能用慈文后来反包修正16日收盘涨停误报。h3＝20日、h5＝22日还未成熟。完整6只现有30个已成熟事件，余36个保持待到期。','',
 'Gemini 16→19日原Top3新华传媒、读者传媒、冰山冷热收盘涨停 **0／3**。完整16日7只连板池，19日仅东方嘉盛、吉大正元晋级，2／7；未列东方是Top-k漏选，不能直接算二分类FN。上一份精确形成时刻仅有Gemini自报，未独立核实时钟；结果价格由本批原日线核验。','',
 table(['16日原池','供应商板数','19日收盘','19日最高','涨停参考价','收盘涨停'],[[r['code']+' '+r['name'],str(r['reported_signal_boards'])+('（跨停牌）' if r['suspension_gap'] else ''),r['close19'],r['high19'],r['upper19'],r['actual_close_limit_up']] for r in old7['rows']]),'',
 '新华传媒10月13—15日停牌、16日复牌；供应商“11板”跨停牌，不认定相邻官方交易日11连板。它19日最高15.78、涨停参考16.68，未触板。重组预案与9月复牌事件不能改写成10月15日新催化。[停牌核查结果与复牌公告镜像](https://pdf.dfcfw.com/pdf/H2_AN202610151830531978_1.pdf)。','',
 '## 经验与本轮实际模型优化','',
 '第一，先核停牌、真实相邻板数、公告原日期；老高标的叙事强度不能替代次日封板证据。第二，完整同位与首板扩散先于“跟风必死／同位只能活一只”的故事：本日东方、吉大可以共同晋级，16日老高标也能全部失败。第三，分开评价涨停、累计上涨、每天上涨、连续涨停与买后收益；第四，大成交可以是分歧或派发，须有后续承接才能区分，不从一天成交推断资金因果。','',
 '限定预读采用林曜01的16日原文、jl韭菜抄家的16日原文，以及阿乐19日早评的搜索工具正文。采用完整竞争、种子节点和可撤销推演的检查思路；不采纳名字／字母直接决定涨跌、先后封板即资金让位、特定产品供货等未经核实的因果。林曜正文写京城、结尾写京投，歧义不改；阿乐05:35正文与09:23／09:29／09:30更新分时标记，不能倒填。19日优先作者当前新正文覆盖没有取得，保留未知。[林曜原文](https://m.tgb.cn/a/2vWrtLGJL9K)、[相反观点原文](https://www.tgb.cn/a/2vWNJbv2XAs)。','',
 '实际新研究为 **DIRECT_CUMULATIVE_TARGETS**：原29字段、原Logistic及0.5判断线固定，只把训练事件分别定义为T+2／T+3／T+5收盘高于T收盘。2025年7月起每月只用月初前已成熟的252个信号日，不用9月样本反测9月。旧ROLL252冻结方向与排名仅作“把次日涨停判断外推”的代理比较，另有恒预测上涨及高度排名基准。不是次日涨停模型升级，也没有重新搜索权重、阈值或旧失败算法。','',
 '实际完成 **47次新研究训练**（45个历史月度头＋10月2日／3日头）；10月5日头因2026-09-10整日无合格终点标签、252日中仅251日可观察，保持严格资格并跳过。初次资格检查在拟合前停止，随后仅追加“跳过不合格窗口、其余独立窗口继续”的技术修复；失败、协议、原源码和修复收据均保留。未补价格、未把未知标签当0。','',
 table(['目标','时期','完整样本／已核／未知','直接目标Accuracy','次日判断外推Accuracy','恒预测上涨Accuracy','本段通过'],[[str(r['horizon'])+'日累计',r['fold'],f"{r['direct']['full']}／{r['direct']['known']}／{r['direct']['unknown']}",f"{r['direct']['accuracy']*100:.2f}%",f"{r['proxy_nextlimit_direction']['accuracy']*100:.2f}%",f"{r['always_up']['accuracy']*100:.2f}%",r['fold_pass']] for r in study['fold_metrics']]),'',
 '**结论：2日／3日目标有局部改进，但未稳定超过简单上涨基准；5日目标在2025下半年和2026年9月退化。三组均未通过整体验收，不部署，不称次日涨停准确率已提高。** 每个目标完整3707只次保留，合计11121行；10765个可核终点标签，356个未知。9月缺失较多，不能只报可见样本胜率。Top3完整906个选择槽与未知也保留。','',
 '2日目标全池Accuracy相对次日代理提高5.20个百分点，三目标校正块区间约为+1.14至+9.48个百分点；但相对恒预测上涨只提高1.74个百分点，区间约为−1.85至+5.41，且2025下半年相对代理增幅未达预登记3个百分点。因此不能笼统说所有局部差异均无统计证据，也不能据局部差异宣称模型通过。','',
 '新研究当前影子头对万丰T+2／T+3偏否、吉大T+3偏否，与独立累计偏上判断有冲突；保隆T+2／T+3偏上。10月T+5头保持未知。这些是未通过验收的研究方向，不覆盖独立预测，也不把0.48／0.61写成涨跌概率。','',
 'Gemini提出的父锚从属与新题材独立性假设已做输入资格核查：9315行中非空行业仅64行、题材0行；64行非空不自动等于历史行业已独立核真。原名称扩散与价格关系研究已经失败，不能换名重跑；本轮不拿名称或事后涨幅伪造行业真值。下一项有价值的输入是逐日有出处的业务／题材映射，以及当时可得的首板和炸板扩散、父锚状态；应先补来源并留档，再另做固定比较。','',
 '## 完整39只收盘候选审阅摘要','',
 table(['代码／简称','板数','独立次日判断','优先序','角色假设','主要依据／限制'],[[v['code']+' '+v['name'],v['boards'],label(v['next_close_limit_up_call']),v['priority_rank'] or '—',v['role_hypothesis'],v['support']] for v in sorted(own['rows'],key=lambda r:(r['priority_rank'] or 99,-r['boards'],r['code']))]),'',
 '每只完整反证、取消条件、完整同板位竞争者及2／3／5日弃判保存在原JSON和CSV；非Top5没有因缺证据而被删除。完整18炸板、22跌停及额外603337冲突见80只总表。','',
 '## 验证与文件','',
 f"核验通过：47个保存模型重新加载，{qa['saved_model_prediction_rows_verified']}行预测复现，最大分数差{qa['max_saved_score_delta']:.3g}；10765个多日价格标签独立重算；旧正式registry／ledger与原独立文件哈希保持。新39只前向记录对应429个事件全部尚未成熟，不能记新增正确结果。",'',
 '原正式模型、阈值、旧评分、正式账本与定时任务未改。本轮新增独立预测39只，新增正式登记0、正式成熟0、真实前向结果0；实际成交证据0，不能把无成交记录称空仓或零收益。','',
 '- [完整11只双方原判断与BASE冲突](完整11只_双方原判断与BASE冲突.csv)',
 '- [完整80只含首板、炸板、跌停与来源冲突](完整80只_收盘池炸板跌停与来源冲突.csv)',
 '- [完整39只2／3／5日预测与弃判](完整39只_2_3_5日原预测含弃判.csv)',
 '- [Codex原独立留档](independent_full39_BEFORE_MODEL_AND_GEMINI.json)',
 '- [Gemini原独立稿](gemini/gemini_independent_20261019_BEFORE_CODEX_MODEL.md)',
 '- [Gemini会后追加修正，原稿保留](gemini/gemini_joint_comparison_APPEND_ONLY.md)',
 '- [研究实际结果](model_optimization/fixed_research_result.json)',
 '- [最终核验](verification_FINAL.json)','']
 text='\n'.join(lines);(OUT/'10月19日联合复盘与20日预测_模型优化实测.md').write_text(text,encoding='utf-8')
 # A compact independent facts view keeps the raw source table easy to inspect.
 factshtml='<html lang="zh-CN"><meta charset="utf-8"><title>10月19日完整候选与模型冲突</title><style>body{font:16px system-ui;margin:30px;line-height:1.6;color:#13243b}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccd4df;text-align:left}th{background:#eaf1f8;position:sticky;top:0}h1{font-size:24px}.note{color:#506070}</style><h1>10月19日完整候选与20日原判断</h1><p class="note">概率未验证；这是完整替代来源观察池，已成交与真实前向结果均未取得。</p><h2>双方独立判断与BASE冲突</h2>'+pd.DataFrame(comparison).to_html(index=False,escape=True)+'<h2>完整80只与来源冲突</h2>'+pd.DataFrame(full).drop(columns=['support','rebuttal','cancel'],errors='ignore').to_html(index=False,escape=True)+'</html>'
 (OUT/'完整候选与模型冲突.html').write_text(factshtml,encoding='utf-8')
 b.write_new(OUT/'delivery_receipt.json',dict(formed_at=b.now(),status='JOINT_REVIEW_FORECAST_AND_FIXED_OPTIMIZATION_COMPLETED_NOT_MODEL_UPGRADE',top3=own['top3'],watch2=own['watch2'],new_independent_candidates=39,new_formal_predictions=0,new_research_fits=47,all_three_horizons_failed_acceptance=True,
  source_bindings={str((OUT/p).resolve()):b.sha(OUT/p) for p in ['10月19日联合复盘与20日预测_模型优化实测.md','完整11只_双方原判断与BASE冲突.csv','完整80只_收盘池炸板跌停与来源冲突.csv','完整39只_2_3_5日原预测含弃判.csv','verification_FINAL.json','gemini/gemini_independent_20261019_BEFORE_CODEX_MODEL.md']}))
 print(json.dumps(dict(report=str((OUT/'10月19日联合复盘与20日预测_模型优化实测.md').resolve()),bytes=len(text.encode()),full_source_rows=len(full),model_fits=47),ensure_ascii=False))
if __name__=='__main__':main()
