"""Render the requested review and verify the full unchanged cohorts."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
import relay_cohort_output
import short_term_thinking_latest as latest

OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z'
RAW=PRIOR/'latest15_alternative_page_once'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,data):
    with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    views=read(OUT/'independent_forward_views_BEFORE_MODEL.json')
    before=read(OUT/'independent_archive_receipt_BEFORE_MODEL.json')
    if sha(OUT/'independent_forward_views_BEFORE_MODEL.json')!=before['independent_forward_sha256']:
        raise ValueError('original independent views changed')
    base=read(OUT/'BASE_scores_AFTER_INDEPENDENT_REVIEW.json')
    shadow=read(OUT/'ST_DUAL_shadow_scores_AFTER_INDEPENDENT_REVIEW.json')
    bmap={r['code']:r for r in base['rows']};smap={r['code']:r for r in shadow['rows']}
    manifest,pools,daily,failures,audit=latest.reconstruct()
    closing=pools[('2026-10-15','limit_up')]['rows']
    roster={r['code']:r['boards'] for r in closing if r['boards']>=2}
    if roster!={r['code']:r['boards'] for r in views['rows']} or len(roster)!=6:
        raise ValueError('full original six-stock denominator mismatch')
    for r in views['rows']:
        bar=daily[r['code']]['days']['2026-10-15']
        if bar['close']!=next(x['price_cents'] for x in closing if x['code']==r['code']):
            raise ValueError('current close mismatch')
        if bar['low']==bar['high']:
            raise ValueError('unexpected current one-price identity')
    cohort=relay_cohort_output.summarize([dict(code=r['code'],name=r['name'],boards=r['boards'],
        raw_score=bmap[r['code']]['raw_score'],model_positive=bmap[r['code']]['predicted_positive'],
        independent_call=r['promotion_judgement'],actual_one_word_daily=bmap[r['code']]['actual_one_word_daily'],
        missing_facts=r['missing']) for r in views['rows']])
    write(OUT/'complete_same_height_original_scores.json',dict(formed_at=now(),groups=cohort,
        threshold_unchanged=0.5,relative_rank_confirms_promotion=False))
    rawpool=read(RAW/'parsed_complete_pools.json')
    rows=[]
    for row in rawpool['rows']:
        rows.append(dict(signal_date='2026-10-15',code=row['code'],name=row['name'],pool=row['kind'],
            reported_boards=row['boards'],mainboard_name_non_st_scope=row['scope'],turnover_pct=row['turnover_pct'],
            source_scope='COMPLETE_RETURNED_ALTERNATIVE_PAGE_NOT_EXCHANGE_WIDE_CERTIFICATION',
            unverified_AI_summary_NOT_COMPANY_FACT=row['reason_summary'],source_row=row['row'],source_sha256=rawpool['source_sha256']))
    pd.DataFrame(rows).to_csv(OUT/'完整收盘首板炸板跌停池.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(PRIOR/'delivery_final/完整81只周期角色与原判断对照.csv',dtype={'code':str})
    old=old[old.date.eq('2026-10-14')].copy()
    if len(old)!=15 or old.latest_source_outcome.isna().any():raise ValueError('previous full15 outcomes incomplete')
    old['BASE_call']=old.BASE.gt(.5)
    old['BASE_correct']=old.BASE_call.eq(old.latest_source_outcome.eq(1))
    old[['date','code','name','boards','daily_one_word','original_independent_judgement','BASE','BASE_call',
         'latest_source_outcome','latest_source_status']].to_csv(OUT/'14至15日完整15只原判断与结果.csv',index=False,encoding='utf-8-sig')
    tp=int((old.BASE_call & old.latest_source_outcome.eq(1)).sum())
    positives=int(old.latest_source_outcome.eq(1).sum())
    if (tp,positives,int(old.BASE_correct.sum()))!=(1,4,10):raise ValueError('previous BASE metrics differ')
    call_zh={'PROMOTE':'偏晋级','FAIL':'偏断板','UNKNOWN':'观察／弃判'}
    ordered=sorted(views['rows'],key=lambda r:r['relative_watch_rank'])
    lines=[
        '# 10月15日收盘复盘与10月16日条件预测', '',
        f'实际成稿：{now()}（UTC）；交易日期及下述盘口时刻均用北京时间。独立观点先于模型显示留档，保留完整6只。', '',
        '我的基准预期：退潮中的高标抱团仍在，下一交易日先看分化能否转为局部修复，暂不按全面反转处理。晋级观察先慈文、百洋，2进3看富春；阳光和新华文轩待确认，冰山偏断板。', '',
        '数据来源为15日15:50收盘替代页、腾讯不复权日线及本次取得的公司公告PDF镜像；原行情接口本次4个池各一次请求仍失败。替代页36涨停、7炸板、26跌停；主板且名称非ST范围34／5／25，连板6只。这里的范围不是交易所全市场身份认证。没有Level-2、明日竞价、分钟回封稳定性或实际成交证据。', '',
        '沪指收3707.48点、跌1.59%，深成指跌2.43%、创业板指跌2.92%；下跌超过5100只。指数尾盘收窄跌幅与个股亏损扩散并存。金融、白酒的指数支撑不自动等于连板资金回流。来源：[新浪收评](https://city.sina.com.cn/city_finance/2026-10-15/detail-inivhtum8139063.shtml)、[港台收市报道](https://gbcode.rthk.hk/TuniS/news.rthk.hk/rthk/ch/component/k2/1874059-20261015.htm)。', '',
        '昨日15只连板仅慈文、百洋、冰山、阳光4只续板，晋级率4/15=26.67%；4板3只晋级2只，3板1只失败，2板11只晋级2只。路畅、星帅尔昨日供应商板数跨停牌的资格缺口原样保留，不能据此另造严格连续板分母。今日梯队5板2只、4板空缺、3板2只、2板2只，来源：[完整收盘池](https://toupage.com/stocks/limit-up)。', '',
        '上一批也有错误：独立补充审阅已判断6只中4只正确、覆盖6/15；新经典和超声电子偏晋级失败，阳光弃判后晋级。BASE全15只正确10只，但4个晋级仅抓住1个，慈文、冰山、阳光漏判；其中信号日非一字正例慈文、阳光都漏掉。分母不同，不把独立4/6写成优于模型10/15。这是追加来源结算，未增加正式成熟账本。', '',
        '股吧预读复用了已核的作者正文及单独定时的评论，共8条观点；本次限定所选作者搜索未取得15日合格当日正文。采纳九阳与百洋的周期候选关系，以及“先换手检验再看弱转强”的检查；作者问题不改写成定论，评论不认证资金因果，名字线不直接给买点。没有点名的股票继续保留。', '',
        '| 独立观察序 | 股票 | 晋级目标 | 独立倾向 | BASE原分数／方向 | 研究双通道分数／方向 |',
        '|---:|---|---|---|---|---|'
    ]
    for row in ordered:
        b=bmap[row['code']];s=smap[row['code']]
        lines.append(f"| {row['relative_watch_rank']} | {row['name']} {row['code']} | {row['boards']}→{row['boards']+1} | {row['lean']} | {100*b['raw_score']:.1f}／{call_zh[b['direction']]} | {100*s['channel_b_score']:.1f}／{call_zh[s['direction']]} |")
    lines += ['',
        '分数是未经校准的模型输出×100，不是晋级概率。BASE仍为原固定正式模型；本次分数使用替代来源，只作补充展示。双通道是未通过整体验收的既存研究模型，本次只加载固定权重观察，没有训练、换阈值或部署。独立序不是BASE原排序。', '',
        'BASE原全池顺序：慈文55.7、阳光51.9、富春45.7、百洋42.2、冰山34.1、新华文轩31.0；严格大于50的方向仍只有慈文、阳光。所有6只15日都存在真实日线价格区间，不能沿用14日一价身份。', '',
        '同位完整比较：5板慈文高于百洋13.58分；3板阳光高于冰山17.80分；2板富春高于新华文轩14.65分。富春虽为2板非一字相对首位，原BASE方向仍为负向，不能用相对第一覆盖0.5门。各组都保留共同晋级、共同失败和分化的可能。', '',
        '主要分歧：我保留百洋和富春的晋级倾向，BASE两者均负向；依据分别是完整周期节点与放量后封板、独立业绩事件与较低本轮高度，同时保留大换手及公告反证。我没有随BASE把阳光转正，因缩量后承接与乳业扩散仍缺。研究模型把冰山转正，我仍保留板块负反馈下的偏断板；这些分歧按原文件到期检验。', '']
    for row in ordered:
        lines += [f"**{row['name']}：{row['lean']}。** {row['role_hypothesis']}。",'',
            row['support'],'',row['relative_thesis'],'',
            '反证：'+row['rebuttal'],'',
            '取消或假设失效条件：'+row['cancel_condition'],'',
            '交易安排：'+row['tradeability']['plan'],'']
    lines += [
        '公告中特别需要保留的约束：慈文短剧归属收入约50万元、AIGC尚无收入；百洋鱼粉涨价同时提高饲料成本、鱼粉业务影响有限，且上半年亏损；阳光上半年收入及利润下降；新华文轩披露主业不涉及新技术新产业等重大变化。富春前三季度归母预告3.38—3.78亿元、扣非1.45—1.85亿元，未经审计，不能把归母同比全部当持续主业改善。六份PDF、原始索引、请求时钟和SHA均在notice_fulltext及bounded_sources中；精确首次披露时分未知，不倒填。', '',
        '明天的三条路径：基准为高标继续分化、局部修复；较强路径须两只5板至少一只稳定晋级，另一只无极端负反馈，昨日断板股亏损缓和并有新2板扩散；较弱路径是两只5板同时明显走弱、中位继续跌停，此时取消新增接力。3700及15日低点3674.82仅作为指数观察，不能独立产生个股买点。', '',
        '竞价到开盘：09:25核原始竞价及同位、同线反馈，当前数值待核；0%—5%只是优先观察区，5%—7%要求更强承接，超过7%不直接追，一字只观察。09:35核真实换手、均价与回封，10:00仍未转强或两次触板失败则取消当次新接力。白酒活跃不直接认证乳业共振，冰雪工程不直接认证交通订单，名称共现不直接认证父锚。', '',
        '执行以空仓等待确认为起点；若条件全部满足，默认单股5%—10%、总新增风险仓不超过20%，账户规模与已有持仓未知。入场前找有效分歧低点，从拟成交价到低点超过4%就放弃。A股新买T+1，不能承诺当天跌3%或4%即卖出。B为实际入场日，B+1先处理溢价，只有持续封板且同伴稳定才延长至B+2；跳空、跌停或卖出受阻按可成交事实记录，不补造买卖。', '',
        '本次已把完整6只前瞻倾向留档，但原正式来源门未过，不能走完原全11类角色及正式freeze登记。沿已有review-context完成了诊断审阅并单独保存条件观点，正式模型、旧预测、正式与独立账本均未修改；不是已正式验证的新样本。明日执行条件取消也不删除今日原方向，弃判继续保留。拟合0、正式预测新增0、实际与纸面成交新增0。', '',
        '附件：independent_forward_views_BEFORE_MODEL.json为原始独立条件判断；complete_same_height_original_scores.json为原同位分数；完整收盘首板炸板跌停池.csv保留69条原页面行与范围标记；14至15日完整15只原判断与结果.csv保留全15只结算。', ''
    ]
    md='\n'.join(lines)
    report=OUT/'10月15日复盘与16日预期预测.md'
    with report.open('x',encoding='utf-8') as stream:stream.write(md)
    # A readable compact HTML for the app, without external scripts or hidden scores.
    cards=[]
    for row in ordered:
        cards.append('<article><h2>'+html.escape(row['name']+' '+row['lean'])+'</h2><p>'+html.escape(row['support'])+'</p><p class="risk">反证：'+html.escape(row['rebuttal'])+'</p><p>取消：'+html.escape(row['cancel_condition'])+'</p></article>')
    table='<table><thead><tr><th>股票</th><th>目标</th><th>独立倾向</th><th>BASE分数</th></tr></thead><tbody>'
    for row in ordered:
        table+='<tr><td>'+html.escape(row['name'])+'</td><td>'+str(row['boards'])+'→'+str(row['boards']+1)+'</td><td>'+html.escape(row['lean'])+'</td><td>'+f"{100*bmap[row['code']]['raw_score']:.1f}"+'</td></tr>'
    table+='</tbody></table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>10月15日复盘与16日预期</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;background:#f4f6f8;color:#18222e;max-width:1080px;margin:36px auto;padding:0 24px}article,table{background:white;padding:22px;border-radius:12px;margin:18px 0}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #e1e5e8}h1{font-size:30px}h2{font-size:21px}.risk{color:#8a3d29}.tag{color:#496273}</style><h1>10月15日复盘 → 10月16日预期</h1><p>退潮中高标抱团，基准看分化后的局部修复。慈文、百洋优先，2进3看富春。</p><p class="tag">完整6只｜先存独立观点，再读分数｜替代来源条件预测，未入正式账本｜BASE分数不等于概率</p>'+table+''.join(cards)+'<p>未核明日竞价、Level-2与可成交队列。新买T+1，超过7%高开不直接追、一字只观察；未确认保持等待。</p><p>完整公告、同位差值、昨日全15只结果和执行规则见同目录Markdown。</p></html>'
    with (OUT/'10月15日复盘与16日预期预测.html').open('x',encoding='utf-8') as stream:stream.write(page)
    verification=read(OUT/'model_inference_verification.json')
    if any(sha(path)!=value for path,value in verification['protected_hashes'].items()):raise ValueError('protected hashes changed')
    artifacts=[report,OUT/'10月15日复盘与16日预期预测.html',OUT/'完整收盘首板炸板跌停池.csv',
               OUT/'14至15日完整15只原判断与结果.csv',OUT/'complete_same_height_original_scores.json']
    result=dict(formed_at=now(),status='REQUESTED_REVIEW_AND_FULL6_CONDITIONAL_PREDICTIONS_DELIVERED',
        signal_date='2026-10-15',target_date='2026-10-16',full_pool_count=6,
        original_raw_page_rows=len(rows),complete_original_source_return_counts=rawpool['raw_pool_counts'],
        source_bound_original_reconstruction_checks=len(audit),remaining_source_gaps=failures,
        independent_archived_before_model=True,current6_all_real_non_one_price_sessions=True,
        original_model_order_and_threshold_preserved=True,old15_full_outcomes_retained=True,
        protected_hashes_unchanged=True,formal_freeze_complete=False,new_fits=0,
        new_formal_predictions=0,new_actual_or_paper_fills=0,
        artifacts={str(path):sha(path) for path in artifacts})
    write(OUT/'completion_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','remaining_source_gaps']},ensure_ascii=False))


if __name__=='__main__':main()
