from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,html
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def metric(rows,call='old_call'):
    k=[r for r in rows if r['close_limit_up'] is not None];j=[r for r in k if r[call]!='UNKNOWN']
    tp=sum(r[call]=='YES' and r['close_limit_up']==1 for r in k);fp=sum(r[call]=='YES' and r['close_limit_up']==0 for r in k)
    tn=sum(r[call]=='NO' and r['close_limit_up']==0 for r in k);pos=sum(r['close_limit_up']==1 for r in k)
    def f(a,b):return a/b if b else None
    return dict(rows=len(rows),known=len(k),judged=len(j),correct=tp+tn,TP=tp,FP=fp,TN=tn,positive=pos,accuracy=f(tp+tn,len(j)),full_known_correct=f(tp+tn,len(k)),coverage=f(len(j),len(k)),precision=f(tp,tp+fp),recall=f(tp,pos),F1=f(2*tp,2*tp+fp+pos-tp))
def pct(v):return '未知' if v is None else f'{v*100:.2f}%'
def main():
    rec=read(OUT/'independent_full46_BEFORE_MODEL.json');receipt=read(OUT/'independent_freeze_receipt_BEFORE_MODEL.json')
    assert sha(OUT/'independent_full46_BEFORE_MODEL.json')==receipt['sha256']
    facts=read(OUT/'facts/qualified_full_facts_SCORES_MASKED.json');base=read(OUT/'BASE_scores_AFTER_INDEPENDENT.json')
    vm={r['code']:r for r in rec['rows']};fm={r['code']:r for r in facts['rows']};bm={r['code']:r for r in base['rows']}
    prior=read(OUT/'settlement_19to20_original.json');m=metric(prior['rows'])
    oldbase=read(ROOT/'results/joint_review_20261019T092500Z/BASE_scores_AFTER_INDEPENDENT.json')
    oldbm={r['code']:r for r in oldbase['rows']};common=[dict(r,BASE=oldbm[r['code']]['BASE_direction']) for r in prior['rows'] if r['code'] in oldbm]
    comparison={'original_full39':m,'same11_independent':metric(common),'same11_BASE':metric(common,'BASE')}
    write(OUT/'prior_forecast_metrics_COMPLETE_DENOMINATORS.json',comparison)
    top=sorted([r for r in rec['rows'] if r['priority_rank']],key=lambda r:r['priority_rank'])
    lines=['# 10月20日复盘与10月21日预期、预测','',
      '数据对应北京时间2026年10月20日收盘；本稿在20日晚实际取得并阅读公告后形成。下一个官方交易日为10月21日，T+2为22日、T+3为23日、T+5为27日。真实生成时间见JSON收据，不把21日署名但20日晚已公开的公告当成未来才知道的资料。','',
      '**我的独立涨停相对优先顺序：大业股份、福莱新材、吉大正元；补充观察海正药业、东安动力。全部为未校准的独立预测，概率未验证。**',
      '主预期是修复中的分化：机器人继续争取连板高度，端侧AI由模组、芯片向终端扩散；21日更可能先分歧、后少数个股修复，不把今日较多上涨家数外推为明日普涨加速。','',
      '## 一、重点候选与多日路径','',
      '多日“累计偏上”指对应日期收盘高于20日收盘，不表示窗口内每天上涨，也不表示连续涨停。明日收盘封住与盘中触板分别结算。','',
      '|独立序|股票/代码|今日板位/收盘|21日收盘涨停倾向|22日累计/路径|23日累计/路径|27日累计/路径|原BASE评分与方向|',
      '|---|---|---|---|---|---|---|---|']
    for r in top:
        b=bm[r['code']];hs=r['horizons']
        lines.append(f"|{r['priority_rank']}|{r['name']} {r['code']}|{r['boards']}板 / {r['signal_close']:.2f}|偏晋级；概率未验证|{hs['2']['path']}|{hs['3']['path']}|{hs['5']['path']}|{b['score_display']:.1f} / {'偏晋级' if b['BASE_direction']=='YES' else '偏断板'}|")
    lines+=['','## 二、今日市场事实与来源口径','',
      '沪指3735.37点（+0.12%）、深成指11801.29点（−0.14%）、创业板2739.20点（−0.26%）；沪深成交14561亿元，较前日增加186亿元。包括北交所在内，3774家上涨、1666家下跌。个股修复强于指数，午后回落及尾盘ETF放量收窄跌幅同时存在。[澎湃新闻原报道](https://m.thepaper.cn/newsDetail_forward_34289151)','',
      '本次实际原始响应：东方财富收盘涨停池54只、炸板池29只；替代页面52涨停/29炸板/6跌停。筛选主板、非ST后的两路并集为46只收盘候选、24只炸板、6只跌停，13只连板逐只通过相邻官方交易日日线检查。媒体56涨停/7跌停、午间57或60等口径另留，不相互混用。替代页面抓取时标注18:15。此处完整指返回来源范围完整，不认证全交易所证券状态完整。','',
      '100个代码逐股各请求一次，97个取得准确20日日线；主板收盘候选45只具备本轮相邻19→20完整价格，三峡新材因复牌导致19日参考缺失，完整保留UNKNOWN，不当断板。原始HTTP、真实时钟、SHA和字段位置见facts及context。Level2队列、完整逐笔订单流、实际成交不可核。','',
      '## 三、情绪与21日三种预案','',
      '- 基准：高位补量和部分首板兑现，机器人保留前排，端侧AI内部择强；指数以震荡对待。',
      '- 强于预期：大业延续，吉大换手后稳定回封，福莱/宁波/荣泰至少两只保持强势，新首板继续扩散且成交配合。',
      '- 弱于预期：大业、吉大反复开板难回封；福莱等2板多只失守20日收盘，端侧AI高开回落，则停止主动追价。','',
      '这是事前待检验路径，不是已经发生的竞价确认。没有竞价源就不填竞价分、真实封单与成交。','',
      '## 四、主线、分支与旧周期','',
      '机器人/物理AI是当前最值得延续观察的盘面集群：大业3板，福莱、宁波、荣泰等2板，爱仕达、泰尔、晶华、春光、华之杰等首板提供扩散背景。它们产品和业务不同；页面标签、媒体归类或股票共涨不能确认全部为同一真实产业因果链。大业、福莱等晚间公告没有认证新机器人订单。',
      '端侧AI看美格、移远、日海的模组集群，以及瑞芯微、博通、龙旗等芯片/终端子组；与机器人可以共振，也可能分化。医药另看海正业绩分支，不能套用昨日个别医药失败的负面结论。',
      '低空汽车昨日保隆/万丰未晋级，东安独自保持一价不能代表整组仍强。九阳、百洋的旧周期反包与海洋王的名称情绪连接只列假设：不因羊/Y字辈认证当前补涨龙，不把百洋饲料等同一般消费，更不据名称改原数值分。','',
      '## 五、完整13只连板竞争','',
      '|股票|板位|日线一价|换手%|成交量/19日|末封|独立方向|主要判断|','|---|---|---|---|---|---|---|---|']
    for f in sorted(facts['relay'],key=lambda r:(-r['boards'],r['code'])):
        r=vm[f['code']];t=str(f.get('seal_last','?')).zfill(6)
        lines.append(f"|{r['name']} {r['code']}|{r['boards']}|{'是' if f['one_price'] else '否'}|{f['turnover_pct']:.2f}|{f['volume_ratio_previous_official']:.3f}|{t[:2]}:{t[2:4]}:{t[4:]}|{r['next_close_limit_up_call']}|{r['support']} 反证：{r['counter_evidence']}|")
    lines+=['','全部同身位已逐只保留；允许共同晋级、共同失败和分化，不强制二选一。长春一东按14:56回封的最终3板结算，不采用午间未封状态。','',
      '## 六、晚间公告与股吧逐观点处理','',
      '已读公司原文：吉大正元重申AI/抗量子/低空安全业务尚未规模化、半年亏损；东安动力确认1—9月飞行汽车业务无收入；大业提示累计上涨与亏损风险；福莱提示异动及情绪风险；科森披露收购暂缓并有终止可能；京城子公司此前被立案；海洋王AI照明收入有限。公告日期和实际取得时间分开，原文镜像不冒称交易所直连。','',
      '- [吉大正元晚间公告](https://pdf.dfcfw.com/pdf/H2_AN202610201830720578_1.pdf)',
      '- [大业风险提示](https://pdf.dfcfw.com/pdf/H2_AN202610201830721608_1.pdf)',
      '- [福莱异动公告](https://pdf.dfcfw.com/pdf/H2_AN202610201830720542_1.pdf)',
      '- [东安业务澄清](https://pdf.dfcfw.com/pdf/H2_AN202610201830719487_1.pdf)',
      '- [科森异常波动及暂缓交易说明](https://pdf.dfcfw.com/pdf/H2_AN202610201830722310_1.pdf)','',
      '限定搜索四位优先作者未取得20日合格新正文，实际读取两篇相关19日原文作背景。率土之滨非魔也重视完整梯队可采纳；由封单厚、早一字直接推出保隆/万丰最强延续的推断被今天反例削弱。主升龙头空空龙的同位/板块层级观察可采纳；“识别千亿资金”及谁带谁不作资金因果证据，19日医药弱不能机械移植20日海正。作者名气、热度、自报收益不加分，评论不继承正文日期。完整逐股取舍在独立JSON，当前正文缺口不判没有催化。','',
      '## 七、重点逐股的反证、取消与交易限制','']
    for r in top:
        f=fm[r['code']]
        lines += [f"### {r['priority_rank']}. {r['name']} {r['code']}",'',f"依据：{r['support']}",f"反证：{r['counter_evidence']}",f"21日取消：{r['cancel_conditions']}",f"20日收盘{f['close']:.2f}；按主板正常10%口径计算21日涨停参考{f['next_upper']:.2f}，须以当日交易状态和交易所价格为准。",'']
    lines+=['## 八、原模型对照与冲突','',
      '以下是独立46只原稿及SHA留档之后才加载的旧BASE；未训练新模型，未改系数、阈值或生产选择。分数是100×原始内部值，不是涨停概率。市场特征仍partial且来源是替代补充，此次推理不冒称原正式生产来源门已经通过。','',
      '|原BASE名次|股票|原评分|原方向|独立方向|','|---|---|---|---|---|']
    for b in sorted(base['rows'],key=lambda r:r['rank']):lines.append(f"|{b['rank']}|{b['name']} {b['code']}|{b['score_display']:.1f}|{b['BASE_direction']}|{b['independent_direction']}|")
    lines+=['','原模型前5依次为大业72.5、东安71.8、福莱43.9、京城39.8、吉大38.2。只有前两数值偏晋级，均为日线一价；原模型非一价正向首选为空。独立更看好福莱、吉大、海正、宁波，理由及反证已事前保存；不把独立观点反写原分数，也不因原低分删除观察。','',
      '## 九、21日执行观察卡','',
      '先看09:25相对溢价与可核竞价成交，再看09:35是否守住开盘均价和20日收盘、同组是否共同走强，10:00仍反复触板不能封则取消追价。偏好平开至+5%后实际换手与回封确认；+7%以上冲高跌破均价放量回落不追。一字候选只看走势，不把排队写成成交。',
      '预测排序第一是大业；可参与性优先观察福莱，其次海正。京城有立案事项、科森有收购暂缓反证，均不列本次主动交易候选。新买入遵守T+1，不能假定买入当天触发止损就能卖；没有券商确认不报实际/纸面成交、收益、滑点或持仓。','',
      '## 十、昨天预测的完整结算','',
      f"原39只保留，38条可核、1条未知；原已判断{m['judged']}条正确{m['correct']}条（{pct(m['accuracy'])}），已知覆盖{pct(m['coverage'])}，完整已知正确占比{pct(m['full_known_correct'])}，Precision {pct(m['precision'])}、完整Recall {pct(m['recall'])}。弃判不计正确NO，未知价格不当断板。",'原前三吉大/保隆/万丰中仅吉大收盘涨停，Top3为1/3；保隆开盘约+8.20%但收盘−3.96%，万丰收盘−1.19%，显示开盘溢价不等于收盘晋级。大业原NO后来一字晋级属于漏判，福莱/东安/海正/海洋原弃判但晋级也完整保留。',
      f"只比较原完整11只连板同池：独立已判断Accuracy {pct(comparison['same11_independent']['accuracy'])}、覆盖{pct(comparison['same11_independent']['coverage'])}；旧BASE Accuracy {pct(comparison['same11_BASE']['accuracy'])}、覆盖{pct(comparison['same11_BASE']['coverage'])}。完整分母与旧晚间追加版分别保留，不能只选有利口径。",'本次是实际今天到期的独立附页结算，原正式账本新增0；不混入既有39登记/38已核/28正确窗口。2/3/5日事件按各原官方日期到期再核，不因一日晋级宣称多日预测成功。','',
      '## 十一、经验与事前检验安排','',
      '本次落实：当前完整池先复核，20日公司公告原文与19日股吧判断分栏，量倍数同时看前日一价低基数、绝对换手与最后封板，全部独立46只先留档再加载原BASE。修正没有把昨天大业NO、保隆/万丰YES改成事前正确，也没有因今天案例重训。',
      '下一日检验：①涨停Top3；②全部46只YES/NO/UNKNOWN及误报漏报；③13连板完整同位共同晋级/分化；④一价预测与可成交分开；⑤福莱缩量补量；⑥吉大高换手后5板；⑦大业新高后的卖压；⑧东安澄清后的情绪；⑨海正经营数据分支；⑩科森暂缓后的反应；⑪京城监管风险；⑫端侧AI首板兑现；⑬旧羊/Y线反包；⑭2/3/5日累计/逐日上涨/连续涨停分别到期；⑮竞价和开盘确认只追加真实时点，不改本稿。以上是本稿观察清单，不冒称已完成原V298正式15假设冻结。',
      '旧模型与原结果保留，采用与否由用户决定；本稿没有重启已取消的Gemini训练、没有修改平台目标、定时任务或下单。','',
      '[完整46只独立预测CSV](完整46只_独立方向与2_3_5日原预测.csv) · [原始独立稿](independent_full46_BEFORE_MODEL.json) · [完整首板/连板/炸板/跌停](facts/完整首板连板炸板跌停_遮分.csv) · [前日完整结算](prior_forecast_metrics_COMPLETE_DENOMINATORS.json)']
    report=OUT/'10月20日复盘与21日预期预测.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    # One complete, readable full-pool view; no automatic direction changes.
    table=[]
    for r in rec['rows']:
        b=bm.get(r['code']);table.append(dict(code=r['code'],name=r['name'],boards=r['boards'],independent=r['next_close_limit_up_call'],priority=r['priority_rank'],BASE_score=b['score_display'] if b else None,BASE_direction=b['BASE_direction'] if b else 'NOT_SCORED_FIRST_BOARD',support=r['support'],counter=r['counter_evidence'],cancel=r['cancel_conditions']))
    pd.DataFrame(table).to_csv(OUT/'完整46只_独立与原BASE对照.csv',encoding='utf-8-sig',index=False)
    independent_ok=sha(OUT/'independent_full46_BEFORE_MODEL.json')==receipt['sha256']
    sources_ok=all(Path(p).exists() and sha(p)==v for p,v in rec['source_bindings'].items())
    inference=read(OUT/'BASE_inference_BEFORE_LOAD.json');protected_ok=all(sha(p)==h for p,h in inference['protected_hashes'].items())
    assert independent_ok and sources_ok and protected_ok
    assert len(vm)==46 and len(bm)==13 and len({r['code'] for r in top})==5
    write(OUT/'completion_VERIFIED.json',dict(formed_at=now(),full_independent46=True,relay13=True,top5_frozen_before_model=True,independent_unchanged=independent_ok,source_hashes_verified=len(rec['source_bindings']),sources_unchanged=sources_ok,old_models_and_formal_files_unchanged=protected_ok,new_fits=0,orders=0,new_formal_predictions=0,new_formal_matured=0,report=str(report),report_sha256=sha(report),independent_formed_at=rec['formed_at'],BASE_formed_at=base['formed_at'],target_date='2026-10-21',horizons=rec['horizon_dates'],initial_independent_completeness_failure='Initial draft omitted002929; validator refused before saving. Added actual authored review before any current model output; no frozen prediction overwritten.',new_training_restarted=False))
    print(json.dumps({'report':str(report),'full_independent':46,'relay':13,'prior_metrics':comparison,'source_checks':sources_ok,'protected_unchanged':protected_ok,'fits':0},ensure_ascii=False))
if __name__=='__main__':main()
