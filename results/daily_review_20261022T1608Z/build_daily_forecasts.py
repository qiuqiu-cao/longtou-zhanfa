from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json,hashlib
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
 with p.open('x',encoding='utf-8') as h:json.dump(x,h,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
facts=read(OUT/'facts/complete_enriched_facts_SCORES_MASKED.json');own=read(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json');rows={x['code']:x for x in facts['rows']}
paths={
'603028':[
('偏一进二；基准为小幅高开后承接，维持早盘相对领先才保留封住倾向','若不随大业失速且资金愿意承接，二板是第一步；只触板未封仍算晋级失败'),
('若23日收住二板，26日偏挑战三板；相对22日累计方向条件偏上','若23日未收住，26日转为震荡回落预期，不能把二板成功条件回填到23日预测'),
('三进四是辨识度检验，基准增加分歧，累计条件偏上而非必连板','须同线至少有持续接力、自己不退成后排；条件失效则转为兑现'),
('若已通过分歧，才保留低位补涨向核心演化；否则偏震荡','新中位接力没有补位或旧高标负反馈扩大，取消继续走高情景'),
('累计方向未知；只在先前逐日兑现且相对领跑时保留继续主升情景','不预测五日每天上涨，不认证已是情绪龙头')],
'603280':[
('偏一进二，但要比22日更早形成有效封板或回封','南方/山河机械分支需延续，不能只跟随大业瞬间上冲'),
('若二板成功且机械有新增承接，累计条件偏上，三板有挑战','若放量冲高但机械同伴退潮，偏套利兑现'),
('更可能分歧换手；能否到四板未知，累计方向取决于前两日承接','与赛福天可同时晋级，不预设二选一，也不保证同涨'),
('机械分支形成梯队时偏高位震荡后再走强，否则回落','由个股连续领先判断身份，不因设备/机器人标签自动升级'),
('累计方向未知，需再评估是否从跟随变为独立核心','五日连续涨停未验证，不预先认龙')],
'600261':[
('偏一进二但优先级第三；基准是照明支线有延续、开盘后需要承接','海洋王/星光不能同时严重负反馈；百洋和旧阳字大面是反证'),
('若首日二板且照明同线不掉队，累计条件偏上；否则偏震荡','不以名称发酵代替真实接力'),
('低位补涨身份分化，累计方向转为条件性未知','如果只剩自己独立拉升且同线退潮，不押四板'),
('偏震荡，除非已出现连续主动领先与新的低位扩散','若海洋王走弱而自己无独立承接，情景取消'),
('累计方向未知，不预测五天逐日上涨','不能把一次早封等同持续情绪认同')],
'603278':[
('偏五进六；更可能开盘后接受换手而非仍无分歧','监管函正文未取得，交易执行保持待核；预测延续不等于获准参与'),
('若六板成功，26日七板分歧进一步扩大，累计条件偏上','机器人低位需有二板承接，若中位继续崩溃则偏高位兑现'),
('高位能否跨过分歧未知，基准为高位震荡而非无条件八板','若首次断板后不能修复，不因空间最高继续上调预期'),
('累计未知，偏等待新梯队身份确认','大面积负反馈不收敛时不押旧高位反复加速'),
('累计未知，不预测连续五板或每天收涨','监管理由与实际交易资格必须持续核验')],
'002724':[
('六板判断未知；基准先分歧，须明显弱转强才能提高执行预期','五板尾盘回封后不能把微红开直接当超预期；名称线退潮反证重'),
('若23日有效修复且照明低位延续，累计条件偏上，否则偏回落','价量修复须有实际承接，不能靠封单故事'),
('偏高位分化，累计方向未知','九阳/百洋若继续亏钱而照明无独立支撑，取消强势情景'),
('累计未知，观察照明线是否独立于名称线存活','未证明独立时不把羊/阳/洋整线认可当依据'),
('累计未知，不押连续五日涨停','10月27日原公告计划披露三季报，事件另记，不预测财报结果')],
}
dates=['2026-10-23','2026-10-26','2026-10-27','2026-10-28','2026-10-29']
bands={
'603028':dict(expected='高开约2%—5%，参考7.92—8.15元',strong='高开超过5%至8%左右，参考8.16—8.38；更高开要防一致兑现',weak='低于约2%或低开，视为偏弱；须重新证明弱转强'),
'603280':dict(expected='高开约1%—4%，参考29.36—30.22元',strong='高开超过4%至7%左右，参考30.23—31.09',weak='接近平开或低开为弱于本次预期，不能按低价直接抄'),
'600261':dict(expected='高开约2%—5%，参考3.68—3.78元',strong='高开超过5%至8%左右，参考3.79—3.89',weak='平开附近或低开偏弱；海洋王/星光同步弱则取消积极执行'),
'603278':dict(expected='高开约4%—8%，参考15.10—15.67元；五板应有溢价',strong='超过8%接近涨停，参考15.68—15.96；不能因接近板价就追',weak='低于约4%视为弱于五进六预期，平低开尤其需防兑现'),
'002724':dict(expected='高开约1%—5%，参考11.33—11.77元，仅符合分歧修复观察',strong='超过5%至8%左右，参考11.78—12.11且竞价真实成交支持',weak='平开或低开偏弱，五板后的弱溢价不能套普通0—5%就认为可接'),
}
confirm={
'603028':['603278','603280','000678','002249','003021'],
'603280':['002097','603278','603028','002535'],
'600261':['002724','002076','603330','002242','002696'],
'603278':['603028','603280','003021','002249','002347','001239'],
'002724':['600261','002076','603330','002242','002696','600178']}
candidates=[]
for code in own['top3']+own['observations']:
 r=rows[code]
 candidates.append(dict(code=code,name=r['name'],role='FIRST_BOARD_EARLY_LEADER_FOCUS' if code in own['top3'] else 'HIGH_BOARD_OBSERVATION',signal_close=r['close'],next_session_limit_reference=r['next_upper'],reference_price_status='Derived 10 percent ROUND_HALF_UP from observed close; subject to next official reference and corporate/security status',probability='UNVALIDATED_INDEPENDENT_SCENARIO',
  days=[dict(date=day,baseline=path,condition=condition) for day,(path,condition) in zip(dates,paths[code])],
  auction=dict(target_date='2026-10-23',bands=bands[code],origin='THIS_REVIEW_HUMAN_OBSERVATION_UNVALIDATED_NOT_OPTIMAL_THRESHOLDS',
   required_verification='09:20—09:25带时点匹配价格/匹配成交与未匹配买卖量；09:25最终成交价额。无同刻历史不编最低额与同比，无逐笔不把余额变化叫撤单。竞价达档不自动允许买入。沪市09:20—09:25不接受撤单。',confirmation_codes=confirm[code]),
  execution=dict(gate='REGULATORY_LETTER_FULLTEXT_PENDING_NO_NEW_ENTRY_CLEARANCE' if code=='603278' else 'CONDITIONAL_SECURITY_AND_NOTICE_STATUS_RECHECK_NO_BLANKET_CLEARANCE',
   after0930='先观察真实价格交换，第一次回落不失守开盘价与实际成交均价；回收时量能增加且同线不破，再看首次上板或换手回封。不是开盘价买入指令。',
   after0935='09:35以后仍保持相对领先、首次回落收敛并重新上攻封住才保留打板观察；先确认后可能没有成交，不倒填买在封前。',
   cancellation='开盘后持续跌破开盘价与实际成交均价且无法回收、反复触板不能封、同线前排与中位同时恶化、或出现停牌/硬资格风险，取消本卡积极执行；原收盘预测仍完整保留。',
   orders='No orders placed. A limit-price order is not a conditional trigger; queue is not fill.',
   position='沿原账户规则，未通过既有保护门不新增交易许可；本卡不设新仓位授权。',
   exit='新买A股T+1，日内失效不能假设当天卖出。次一可卖日若接力失败且不能回收成交均价/原收盘，优先退出观察；真实成交、费用、未平仓及卖出受阻另记。')))
payload=dict(formed_at=now(),signal_date='2026-10-22',target_date='2026-10-23',dates=dates,full_pool_independent_path=str(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json'),full_pool_sha256=sha(OUT/'INDEPENDENT_FULL_POOL_BEFORE_MODEL.json'),
 event_definitions=dict(next_close_limit='Next official-session closing price at qualified upper limit, not just intraday touch',cumulative_up='Future horizon close above 2026-10-22 close',every_session_up='Every successor close strictly above previous close; not equivalent to cumulative up',consecutive_limit='Every successor closes at upper limit; unverified, not asserted'),
 daily_market=[dict(date=dates[0],forecast='分歧后局部修复优先，非全面反转。旧中位止跌、双五板不共同大面且低位有晋级才加强修复判断；否则仍退潮。'),dict(date=dates[1],forecast='验证23日新二板能否走三板，若只有一字和旧高标则修复质量不足；基准继续淘汰后排。'),dict(date=dates[2],forecast='验证新梯队能否产生持续核心，若前三日反馈有递进才偏继续走强，否则修复后再分歧。'),dict(date=dates[3],forecast='高度与亏钱效应再检验，基准加大分化，不预设普涨延续。'),dict(date=dates[4],forecast='若低位与中位接力不断档才保留新周期情景；否则偏轮动和兑现，方向未知。')],
 candidates=candidates,model_output_status='Independent complete-pool calls were archived before BASE; daily scenarios written after BASE, not retrospectively represented as blind.',new_fits=0,new_orders=0,production_mutation=False)
write(OUT/'MULTIDAY_PREDICTIONS_AND_EXECUTION_CARDS.json',payload)
print(json.dumps(dict(focus=[x['code'] for x in candidates],dates=dates,new_fits=0),ensure_ascii=True))
