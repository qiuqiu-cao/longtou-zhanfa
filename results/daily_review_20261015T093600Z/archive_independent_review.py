"""Author the complete six-stock review before displaying any current model output.

The inherited workbench is explicitly diagnostic. Its original forecast fields
stay null. The requested next-session views are separately timestamped as an
unregistered conditional supplement, with every original freeze gate retained.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import relay_review_workbench as workbench
import stock_judgement_review as original

OUT=Path(__file__).resolve().parent
PREP=ROOT/'research/model_upgrade_validation/workbench_latest_20261015T091200Z'
FACTS=PREP/'symbolic/facts_with_symbolic_context.json'
RAW=ROOT/'research/model_upgrade_validation/historical_symbolic_context_20261015T081200Z/latest15_alternative_page_once'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False)


# Decisions below are authored from the closing facts and issuer documents.
# They are neither a transformation of model scores nor filled from rule labels.
VIEWS={
 '002343':dict(call='PROMOTE',rank=1,lean='偏晋级6板',role='传媒换手高标候选、共同高度锚',
   thesis='同为5板先比较百洋；慈文日内振幅2.96%、成交6.82亿元、换手16.17%，以更窄价格波动完成涨停。新经典断连板而新华文轩、内蒙新华、读者传媒仍封板，传媒有跨板位同伴，允许慈文与百洋共同晋级。',
   support='5个相邻官方交易日连续收盘涨停已核，15日8.67至8.91元而收于8.91元；完整收盘池保留出版首板和新华文轩2板。采纳先检验换手再看弱转强的观点，本次方向来自逐股收盘比较。',
   rebuttal='公司公告明确AIGC尚无收入、《开标之战》归属收入约50万元、短剧及海外业务占比较小，半年报亏损。新经典断板、掌阅炸板和全市场普跌反驳传媒全面走强；日线不证明回封稳定及真实承接。',
   cancel='竞价高开后持续失守8.91元，且开盘5分钟不能重回均价；或新华文轩与出版首板同步转弱、两次触板不能封住，取消接力。',
   trade='第一观察；仅真实换手后重新转强且传媒同伴确认时考虑。高开超过7%不直接追，一字不记可买。',
   peers=['002696','601811'],anchor_accept=[],identity='UNKNOWN'),
 '002696':dict(call='PROMOTE',rank=2,lean='偏晋级6板，分歧风险较高',role='九阳周期提前启动的补涨候选、共同高度锚',
   thesis='同位慈文更有板块扩散；百洋保留自身5板辨识度和九阳周期先后关系。与阳光可共同晋级，也可能名称情绪退去后一同失败，不预设二选一。',
   support='9日首板、12日2板、13日九阳断连板时百洋3板、15日5板的节点可重建；15日最低9.75元、最终10.73元收涨停，经历此前一价后的大量换手。两只高标仍共同封住，支持保留延续倾向。',
   rebuttal='成交7.82亿元、换手22.37%，原生量为14日约22.47倍，振幅10.05%；这既可能释放分歧，也可能是末端兑现，不能单凭爆量认证承接。公司亏损且披露鱼粉业务收入占比11.04%、涨价同时增加饲料成本，盈利影响有限。',
   cancel='竞价明显弱于慈文且阳光同步走弱；开盘持续跌破10.73元后放量失守均价、5分钟不能收回，或两次触板失败，取消新接力。9.75元是15日低点反证，不是准许承受的止损距离。',
   trade='第二观察，必须见分歧后转强；不因补涨关系排一字，不以名字线替代盘口。',
   peers=['002343','001318'],anchor_accept=['002242'],identity='CYCLE_SUPPLEMENT_CANDIDATE'),
 '605189':dict(call='PROMOTE',rank=3,lean='偏晋级3板',role='业绩事件驱动的低位换手候选',
   thesis='与新华文轩同为2板，富春本轮高度较低且有单独业绩事件；新华文轩实际已14天10板。染织、染料与化工涨价不是同一行业，醋化失败仍作为涨价偏好受挫的交叉反证。',
   support='公司预告前三季度归母3.38至3.78亿元、扣非1.45至1.85亿元，原PDF已核。15日17.35至17.85元、收17.85元，成交7.24亿元、换手17.10%，由14日一价转为有价格区间的换手板。',
   rebuttal='预告未经审计，归母与扣非差额较大，不能把归母同比全部当持续主业盈利；原生量放大约6.83倍未认证真实承接。超声、宝鼎未续板说明业绩叙事并非普遍接力保护。',
   cancel='竞价失去溢价且开盘不能收复17.85元、出现放量跌破均价；或冲板两次不封，取消。17.35元是昨日低点，跌破则形态进一步破坏，不以此设置过宽入场风险。',
   trade='2进3优先观察；限非一字、开板换手后回封，业绩预增不等于买点。',
   peers=['601811'],anchor_accept=[],identity='UNKNOWN'),
 '001318':dict(call='UNKNOWN',rank=4,lean='积极观察，暂不押4板',role='九阳相关旧跟风重启候选',
   thesis='同位冰山放量更极端；阳光换手6.33%、量约前一官方交易日0.445倍，可能锁筹也可能未经分歧检验。与百洋同周期比较，但乳业与水产不视为同业务；旧跟风重启不能重写成新补涨龙。',
   support='13日至15日完整3板已核；16.30至16.97元、收涨停，存在真实价格区间而非一价；九阳断板当日重新首板的时序可核。百洋仍5板提供共同情绪观察点。',
   rebuttal='缺同日作者正文和完整乳业扩散证据。公司公告提示上半年收入下降8.13%、净利润下降22.48%，主营未变；缩量不能证明承接，名字同音不能单独产生晋级方向。',
   cancel='百洋失去强势且阳光放量跌破16.97元或均价、5分钟不能收回；或消费首板扩散消失，停止观察接力。',
   trade='等待放量分歧后回封；当前不列正向首选，未知不记预测断板。',
   peers=['000530','002696'],anchor_accept=['002242'],identity='OLD_FOLLOWER_RESTART'),
 '601811':dict(call='UNKNOWN',rank=5,lean='积极观察，暂不押3板',role='出版方向反复活跃的高位容量观察锚',
   thesis='同位富春本轮位置低；新华文轩虽连续2板但已14天10板，不按普通低位2进3评估。与慈文是板块共振竞争和支持，不认定其为慈文的补涨子股。',
   support='15日22.73元开、22.63元低、25.58元收涨停，成交15.67亿元；完整池保留内蒙新华和读者传媒，说明出版分支仍有封板同伴。',
   rebuttal='振幅12.69%、尾盘14:49封板，已长期高位反复，涨停不能认证次日溢价；公司公告称没有新技术新产业等业务变化及未披露重大事项。新华传媒15日日线缺失，不以缺失证明父锚崩溃。',
   cancel='慈文转弱且本股开盘不能收复25.58元、持续失守均价；或内蒙新华、读者传媒同步失去溢价，则取消转强观察。',
   trade='先看板块容量承接，回封后仍需风险距离合适；不按2板身份低估位置。',
   peers=['605189','002343'],anchor_accept=[],identity='RELATED_UNRESOLVED'),
 '000530':dict(call='FAIL',rank=6,lean='偏断板，反包除外另记观察',role='冰雪方向换手竞争者，尚未认证换手龙',
   thesis='同位阳光换手6.33%，冰山换手41.05%、金额16.97亿元，筹码交换更剧烈。哈空调、龙江交通未续板且收盘跌停，板块不能因冰山单股晋级就视作完整共振。',
   support='15日有价格区间且收涨停，但原生量约14日13.61倍，极端高换手与同方向两股负反馈同时存在；退潮下中位承接要求更高，偏向3进4失败。',
   rebuttal='本股确实逆势收涨停，可能成为独立活口并继续走强，不能仅凭放量或高换手判死刑；日内开板路径和液冷、储能分支真实跟随未核。',
   cancel='若次日竞价与阳光相比主动领先、开盘持续守住均价且放量回封，同时哈空调/龙江止住负反馈，则原偏断板假设失效，转为观察；不得回改本次方向。',
   trade='不列新接力首选；转强只触发重新观察，需保存新的确认时间。',
   peers=['001318'],anchor_accept=[],identity='UNKNOWN'),
}

OPINIONS={
 'opinion_83bf69c0ffa8a102e1b5769c':('旧消费段认为九阳、均瑶提供身位，但缩量承接未明，后排谨慎。','ACCEPT','ROLE_ONLY','采纳把身位和承接分开检查的框架；9月30日对后排的结论不移作10月16日判断。'),
 'opinion_c411f771d7bcb085bfb42037':('旧化工段提出第二个字Y的拼音连接，先看能否脱离气氛组。','PENDING','UNCERTAIN','只确认这条名称假设出现，不能把其他化工股票的关系移植为阳光的业务或资金因果。'),
 'opinion_983be17090052372fbb8bbfa':('旧文比较九阳、均瑶、襄阳与新华的名称及高低位联动。','ACCEPT','ROLE_ONLY','采纳完整竞争者和首板扩散检查，不继承旧评分、旧日晋级方向或唯一父龙结论。'),
 'opinion_5e39671456abb7359dd06f49':('慈文属于传媒伴飞，需要放量检验以后再看弱转强。','ACCEPT','NEUTRAL','15日已观察到16.17%换手且封板，但只有日线；采纳检查顺序，不认证盘中吸筹或16日一定晋级。'),
 'opinion_615005c65e583807a9d6975a':('主帖把三羊、九阳、百洋、圣阳并列为名称叙事。','ACCEPT','ROLE_ONLY','只采纳该名称叙事曾出现；结合独立时序保留九阳周期补涨候选，不认证四股资金链。'),
 'opinion_3881102d5c2dfd4f09cbdd21':('农业板块弱而百洋强，作者把相对独立视作主动性待检验。','ACCEPT','NEUTRAL','采纳相对表现检查，反对把板块越弱直接当越利好；15日放量和公告业绩风险仍须单独计入。'),
 'opinion_afb25256f83d606e24a62a4c':('作者询问远程、百洋是否属于Y连接及能否独立走出。','ACCEPT','ROLE_ONLY','这是问题而非确认九阳布置资金的结论；以原日期先后作角色候选，未写作者当前预测。'),
 'opinion_5a91f9088710f3769a037d7d':('评论者提出Y像羊、九阳布局的解释。','PENDING','UNCERTAIN','评论具有自己的21:09时间；未取得资金链证据，不能将评论提升为主帖断言或资金因果事实。'),
}


def main():
    if (OUT/'independent_forward_views_BEFORE_MODEL.json').exists():
        raise FileExistsError('independent forward views are immutable')
    facts=read(FACTS)
    original.check_sources(facts['source_bindings'])
    # Persist the bounded search outcome. No current-day author body is claimed.
    write(OUT/'forum_search_scope.json',dict(formed_at=now(),
        selected_authors=['延边刺客','毛竹下的橙子','林曜01','凯域竟知'],
        queries='限定2026-10-15及10.15的作者与候选检索；未找到可核当日相关正文，未扩全论坛',
        verified_current_day_bodies=0,existing_checked_opinion_count=8,
        use='Old author bodies and separately timed comments are background only; no current author forecast inferred.'))
    notice_manifest=read(OUT/'notice_fulltext/manifest.json')
    extra_paths=[OUT/'bounded_sources/manifest.json',OUT/'notice_fulltext/manifest.json',OUT/'forum_search_scope.json',Path(__file__)]
    extra_paths += [Path(row[key]) for row in notice_manifest['rows'] if row['status']=='PDF_TEXT_AND_FIRST_PAGE_SAVED'
                    for key in ['pdf_path','text_path']]
    bound=deepcopy(facts)
    bound['source_bindings'] |= {str(path.resolve()):sha(path) for path in extra_paths}
    bound['requested_review_supplement']=dict(formed_at=now(),formal_source_gate_passed=False,
        original_facts_path=str(FACTS),original_facts_sha256=sha(FACTS),
        issuer_fulltext_manifest=str(OUT/'notice_fulltext/manifest.json'),
        original_rows_unchanged=True,full_role_freeze_bypassed=False)
    write(OUT/'prospective_fact_bundle.json',bound)
    fact_path=OUT/'prospective_fact_bundle.json'
    original.check_sources(bound['source_bindings'])
    drafts=read(PREP/'symbolic/author_draft_PENDING.json')
    drafts['author_review_completed']=True
    drafts['status']='AUTHOR_COMPLETED_DIAGNOSTIC_REVIEW_NO_FORMAL_FORECAST'
    drafts['reviewed_at']=now()
    discovery=read(facts['identity_discovery_context_path'])
    discoveries={row['code']:row for row in discovery['rows']}
    for row in drafts['rows']:
        code=row['code'];view=VIEWS[code]
        row['forum_preread_assessment'].update(reviewed=True,reason=(
            '已逐条阅读限定旧正文及评论，未取到15日当日正文；旧讨论只作背景。' if row['forum_reasoning_assessment']['opinions']
            else '限定已核正文未覆盖本股；保留缺口，不把未点名当无催化、无身份或断板理由。'))
        supplement=row['supplement_assessment'];supplement.update(reviewed=True,reason=view['thesis'])
        for rel in supplement['relations']:
            rid=rel['relation_id']
            rel.update(decision='PENDING' if rid=='sunshine_junyao_alternative' else 'ACCEPT',
                       reason=('均瑶只保留替代背景，父子时序未完整闭合。' if rid=='sunshine_junyao_alternative' else
                               '仅采纳旧文和独立时序支持的角色候选；不认证唯一父龙、资金因果或次日方向。'),
                       counterevidence=[view['rebuttal']],cancel_conditions=[view['cancel']])
        reasoning=row['forum_reasoning_assessment'];reasoning.update(reviewed=True,synthesis=view['thesis'],
            agreement='PARTIAL' if reasoning['opinions'] else 'UNDETERMINED',
            agreement_reason='限定旧叙事只作背景；用本次完整同位、公告和日线独立判断，未知不作为负向证据。')
        for opinion in reasoning['opinions']:
            claim,decision,effect,reason=OPINIONS[opinion['opinion_id']]
            opinion.update(author_claim=claim,author_reasoning=claim,
                testable_expectation='检查下一官方交易日的相对强弱、实际换手及完整同伴反馈；不是原作者对10月16日的预测。',
                author_claim_form='QUESTION' if opinion['opinion_id']=='opinion_afb25256f83d606e24a62a4c' else 'HYPOTHESIS',
                author_call='UNSPECIFIED',author_expectation_target_date=None,
                verification=dict(status='UNKNOWN',scope='ROLE_HYPOTHESIS',reason='叙事出现已按原文核对；身份与资金因果没有独立真值，旧文不能证明当前盘口。',evidence_refs=[]),
                decision=decision,reason=reason,counterevidence=[view['rebuttal']],
                independent_effect=effect,cancel_conditions=[view['cancel']])
        disc=discoveries[code]
        rejected=[]
        for h in disc['anchor_hypotheses']:
            if h['anchor_code'] in view['anchor_accept']:continue
            reason='语义或名称关联不能认证直接父子及资金因果，完整时序仍待核。'
            decision='PENDING'
            if code=='601811' and h['anchor_code']=='600812':
                decision='REJECT';reason='仅名称含共同字不构成医药与出版的直接父子关系，无逐股业务或周期连接证据。'
            if code=='601811' and h['anchor_code']=='002343':
                decision='REJECT';reason='新华文轩已有较早多波高位记录，与慈文只保留传媒共振，不按更晚高标指定其父龙。'
            rejected.append(dict(anchor_code=h['anchor_code'],decision=decision,reason=reason))
        peers=sorted({c for p in disc['same_cycle_competitors'] for c in p['members'] if c!=code})
        row['identity_discovery_assessment'].update(reviewed=True,identity_call=view['identity'],
            accepted_anchor_codes=view['anchor_accept'],rejected_or_pending_anchors=rejected,
            support=view['support'],counterevidence=view['rebuttal'],cancel_condition=view['cancel'],same_cycle_competitors=peers)
    write(OUT/'completed_diagnostic_author_review.json',drafts)
    completed=workbench.review_context(FACTS,OUT/'completed_diagnostic_author_review.json',OUT/'completed_diagnostic_review_record.json')
    # A separate prospective decision file, explicitly outside the formal registry.
    forward_rows=[]
    notice_by_code={r['code']:r for r in notice_manifest['rows']}
    for source_row in facts['rows']:
        code=source_row['code'];view=VIEWS[code]
        refs=[str(RAW/'normalized_daily.json'),str(RAW/'page.raw.html'),notice_by_code[code]['text_path']]
        forward_rows.append(dict(code=code,name=source_row['name'],boards=source_row['boards'],
            promotion_judgement=view['call'],relative_watch_rank=view['rank'],lean=view['lean'],
            role_hypothesis=view['role'],relative_thesis=view['thesis'],support=view['support'],
            rebuttal=view['rebuttal'],cancel_condition=view['cancel'],comparison_peers=view['peers'],
            missing=['10月16日09:25竞价','分钟路径、真实盘口和可成交队列','10月15日所选作者当日合格正文',
                     '券商实际成交与账户持仓','截至本次截止后的新增公告'],fact_references=refs,
            opening_confirmation={'09:25':'核真实竞价强弱与同位同伴，超过7%不直接追、一字仅观察；数值当前待核。',
               '09:35':'核开盘后实际换手、均价及回封，同伴负反馈扩大则取消；数值当前待核。',
               '10:00':'仍未转强、两次触板失败或放量失守均价则取消当次新接力；不逆向补仓。'},
            tradeability=dict(status='UNCONFIRMED_CONDITIONAL_ONLY',plan=view['trade'],
               setup='主板连板换手后转强或回封',preferred_open_gap='0%至5%，只作执行过滤，不改变固定模型阈值',
               hot_open='5%至7%需更强承接；超过7%不直接追，一字不生成成交',
               weak_open='平低开仅在实际收复昨收及均价且同伴稳定后再观察',
               position='默认单股5%至10%、总新增风险仓不超过20%；未满足确认则0，不假定用户账户规模',
               stop='实际入场前确定有效分歧低点；从拟成交价到该点超过4%则放弃。新买A股T+1，当天不能按此价卖出。',
               first_target='B+1先处理溢价；冲高后失守均价或回封失败时减退，须有实际成交且可卖',
               second_target='仅持续封板且同伴稳定才延长至B+2，不预设固定收益',
               holding_period='假设实际入场B后1至2个官方交易日，不能从信号日直接计算卖出',
               gap_down_exit='已有可卖仓位若次日跳空破有效低点、不能收复，按可成交价格处理；跌停不能假设卖出',
               sector_leader_break='同线高标及容量锚同步失守均价并扩大负反馈，撤销新增接力')))
    forward=dict(version='requested-review-forward-supplement-1',status='PROSPECTIVE_CONDITIONAL_VIEWS_NOT_FORMAL_FREEZE',
        formed_at=now(),knowledge_cutoff=now(),signal_date='2026-10-15',target_date='2026-10-16',
        model_exposure='MODEL_NOT_REVIEWED_SELF_ATTESTED',blinding_independently_verified=False,
        author_review_completed=True,full_pool_count=6,facts_path=str(fact_path),facts_sha256=sha(fact_path),
        source_bindings=bound['source_bindings']|{str(fact_path):sha(fact_path)},
        formal_source_gate_passed=False,full_original_freeze_completed=False,
        original_11_role_and_freeze_guards_unchanged=True,formal_registry_modified=False,
        retrospective_rewrite_allowed=False,new_fits=0,new_formal_predictions=0,
        predictions_to_evaluate='All six exact next-session closing-limit outcomes; abstentions retained. Execution cancellations do not erase these original calls.',
        market_call='退潮中高标抱团；基准预期分化中局部修复，尚不预测全面反转',
        bull_confirmation='两只5板至少一只稳定晋级且另一只无极端负反馈，昨日断板亏损缓和并有新2板扩散',
        bear_confirmation='两只5板同时明显转弱、原失败中位继续跌停或涨停扩散收缩',
        rows=forward_rows)
    original.validate_judgements(bound,forward)
    write(OUT/'independent_forward_views_BEFORE_MODEL.json',forward)
    verify=dict(formed_at=now(),stock_count=6,all_sources_rechecked=True,
        diagnostic_validation=workbench.validate_completed(FACTS,completed),
        forward_reasoning_validation='ALL_SIX_ORIGINAL_JUDGEMENT_REASON_PEER_AND_CONFIRMATION_FIELDS_PASSED',
        independent_forward_sha256=sha(OUT/'independent_forward_views_BEFORE_MODEL.json'),
        before_current_model_display=True,new_model_loads=0,new_fits=0,new_formal_predictions=0,
        new_actual_or_paper_fills=0,formal_source_failure_manifest=str(OUT/'bounded_sources/manifest.json'))
    write(OUT/'independent_archive_receipt_BEFORE_MODEL.json',verify)
    print(json.dumps({k:v for k,v in verify.items() if k!='source_bindings'},ensure_ascii=False))


if __name__=='__main__':main()
