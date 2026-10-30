"""Read-only projection of archived research. Never loads a model or edits a source."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
CORRECTION_DATE = '2026-10-23'

def read(path, default=None):
    if not path.exists(): return default
    return json.loads(path.read_text(encoding='utf-8-sig'))

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def archives():
    result = {}
    for path in sorted((ROOT / 'results').glob('*/facts/complete_enriched_facts_SCORES_MASKED.json')):
        facts = read(path)
        if facts.get('signal_date') and (path.parent.parent / 'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json').exists():
            result[facts['signal_date']] = path.parent.parent
    return result

def catalog():
    dates = sorted(archives(), reverse=True)
    return dict(dates=dates, latest=dates[0] if dates else None, served_at=now(), mode='ARCHIVE_ONLY')

def report(day):
    folder = archives().get(day)
    if folder is None: raise KeyError(day)
    facts_path = folder / 'facts/complete_enriched_facts_SCORES_MASKED.json'
    facts = read(facts_path)
    own = read(folder / 'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json')
    base = read(folder / 'BASE_scores_AFTER_INDEPENDENT.json', {})
    first = read(folder / 'features/FIRSTBOARD_BASE_QUALIFICATION.json', {})
    feedback = read(folder / 'facts/COMPLETE_PRIOR_COHORT_FEEDBACK.json', {})
    pool = read(folder / 'facts/pool_capture_manifest.json', {})
    cards_doc = read(folder / 'FINAL_SIX_EXECUTION_CARDS_20261026.json', {})
    cards = {x['code']:dict(x) for x in cards_doc.get('cards', [])}
    independent = {r['code']:r for r in own['rows']}
    model = {r['code']:r for r in base.get('rows', [])}
    # Display correction is additive. Historical rank and directions stay in the archive.
    correction = read(Path(__file__).parent / 'display_correction.json', {}) if day == CORRECTION_DATE else {}
    relay_notes = {
        '603028':'仅模型榜首；一字且触发原薄盘筛选，不主动追入',
        '603933':'模型偏晋级；一字承接未检验，重组问询延期',
        '603280':'模型偏晋级；近一字，需首次真实换手确认',
        '600838':'模型偏晋级；一字、零售情绪锚，排队不等于成交',
        '002535':'主动主选剔除：财务处罚事先告知；仍保留模型原分',
        '603089':'人工分歧观察；撤回“连板第一主选”，不作为买入推荐',
        '603278':'仅情绪锚观察；监管事项未清，不买入',
        '002724':'地天后高位观察，次日兑现与再次承接待验',
        '003021':'超原市值范围，仅容量对照',
        '000678':'同位观察；高换手、多次开板压力',
        '603528':'同位观察；需证明相对主动性',
    } if correction else {}
    rows=[]
    for x in facts['rows']:
        code=x['code']; o=independent.get(code,{}); m=model.get(code)
        card=cards.get(code)
        if card and correction:
            card.pop('slot', None) # Withdraw old relay rank in the current execution view.
            if code=='603089':
                card['gate']='仅人工分歧观察；原第一主选推荐已撤回'
                card['after0930']='仅记录实时承接，不依据本张旧预案发出买入许可。'
                card['after0935']='人工异议尚无样本外优势证据，继续观察并另行结算。'
            elif code=='603278': card['gate']='仅情绪锚观察；监管事项未清，不买入'
            execution_display={
                '603280':('继续一字只观察；首次开板不抢。先检验持筹兑现能否被真实成交吸收。','09:35后，实际换手后重新回封，且正裕、睿能等同位没有批量负反馈，才保留原条件计划。'),
                '603677':('回踩卖压收缩、重新站回日均，且未被同批明显甩开；机器人二板群体保持承接，再观察有效上板。','若已有真实换手，重新回封且题材未转弱，继续观察；不能只因高标封住就放宽条件。'),
                '002009':('高开不直接追，先观察早盘筹码兑现及奇精等同批反馈。','分歧时不持续放量跌穿日均，重新收回开盘区间后上板；奇精与机器人前排维持正反馈。'),
                '603081':('09:35前不抢，等待午后快速拉升所产生的浮盈兑现。','真实换手后站稳日均及开盘区间，奇精、天奇未同时转弱，派斯林未显著强于它，再观察回封。'),
            }
            if code in execution_display:card['after0930'],card['after0935']=execution_display[code]
        rows.append(dict(code=code,name=x['name'],kind=x['kind'],boards=x.get('boards'),
            close=x.get('close'),change_pct=x.get('change_pct'),turnover_pct=x.get('turnover_pct'),
            cap_cny=x.get('reported_float_cap_cny'),screen=x.get('screen_status','非涨停筛选对象'),
            qualification=x.get('qualification'),one_price=x.get('one_price'),
            base=m,independent_call=o.get('next_close_limit_up_call'),
            first_rank=o.get('first_board_relative_rank'),support=o.get('support','待核'),
            counter=o.get('counter_evidence','待核'),participation=relay_notes.get(code),card=card))
    minutes=[]; indexes=[]; sources=[]
    for req in pool.get('requests',[]):
        sources.append(dict(label='涨跌停池 / '+str(req.get('kind')),url=req.get('url'),
            market_at=day+' 15:00 Asia/Shanghai',requested_at=req.get('requested_at'),
            fetched_at=req.get('available_at'),status=req.get('status'),sha256=req.get('raw_sha256')))
    for filename in ['minutes/capture_receipt.json','minutes/additional_capture_receipt.json']:
        document=read(folder/filename,{})
        for r in document.get('records',[]):
            if not r.get('rows') or not r.get('previous_close'): continue
            symbol=r.get('symbol','')
            item=dict(symbol=symbol,name=r.get('name'),previous_close=r['previous_close'],
                close=r.get('last_quote'),rows=r['rows'],fetched_at=r.get('received_at'),
                source_url=r.get('url'),source_date=r.get('source_date',document.get('date')),status=r.get('status'))
            minutes.append(item)
            if symbol in ['sh000001','sz399001','sz399006']: indexes.append(item)
    metadata_paths=[facts_path,folder/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json',folder/'BASE_scores_AFTER_INDEPENDENT.json']
    return dict(schema=1,day=day,target_day=facts.get('target_date'),served_at=now(),
        market_at=day+' 15:00 Asia/Shanghai',archive_formed_at=facts.get('formed_at'),
        mode='ARCHIVE_ONLY',status_text='归档行情 · 非实时 · 刷新只重读本地文件',
        summary=own.get('market','当前归档未提供市场判断'),
        independent_formed_at=own.get('formed_at'),base_formed_at=base.get('formed_at'),
        counts={k:sum(r['kind']==k for r in rows) for k in ['limit_up','failed_board','limit_down']} | {
            'first':sum(r['kind']=='limit_up' and r['boards']==1 for r in rows),
            'relay':sum(r['kind']=='limit_up' and r['boards']>=2 for r in rows),
            'unknown':sum(r['close'] is None for r in rows)},
        rows=rows,indexes=indexes,minutes=minutes,feedback=feedback.get('layers',{}),sources=sources,
        first_top3=own.get('first_board_top3',own.get('top3',[])),
        relay_model_top3=base.get('ranking',[])[:3],
        base_identity='原保存 BASE29 · 连板次日晋级 · 阈值 0.5 · 未校准分数',
        base_qualification='替代来源补充推理，正式来源门未过；不能称合格正式预测',
        first_qualification=first or dict(status='未输出当日首板BASE；该归档没有合格输入资格收据',gaps=['原始归档未提供当日合格首板BASE']),
        correction=correction,scope='返回源主板并集，非交易所全市场完备认证；原流通市值上限约93.27亿元及原薄盘筛选保留',
        issues=facts.get('seal_time_source_conflicts',[]),
        provenance=[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=digest(p)) for p in metadata_paths if p.exists()],
        training=dict(status='本网页未训练、未更换模型',new_fits=0,effect='预测改善尚未由本网页验证；首板完整日期Top1的60%目标仍待验收',blockers=['首板BASE当日输入资格缺口','人工覆盖BASE的优势未验证','公开报价已自动采集，但实时延迟、竞价未匹配量与订单队列仍未验证']),
        evidence_image=('/evidence/'+day+'/relationship.png') if (folder/'minutes/情绪关系分时图.png').exists() else None)
