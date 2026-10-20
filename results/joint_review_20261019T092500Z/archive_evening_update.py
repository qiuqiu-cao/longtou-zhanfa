"""Versioned evening findings; original afternoon predictions remain immutable."""
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import short_term_thinking_dataset_builder as b
OUT=Path(__file__).resolve().parent
def main():
 original=OUT/'independent_full39_BEFORE_MODEL_AND_GEMINI.json';own=b.read(original);late=deepcopy(own)
 source=OUT/'facts/evening_postdated20_currently_public';snapshot=OUT/'facts/evening_refresh_20261019T142830Z'
 proofs={str(p.resolve()):b.sha(p) for p in list((source/'notices').glob('*.pdf'))+list((source/'notices').glob('*.txt'))+[source/'notices/index_manifest.json',source/'delta_receipt.json',snapshot/'evening_delta_receipt.json',snapshot/'notices/002889_AN202610161830618206.pdf',snapshot/'notices/002889_AN202610161830618206.txt',original]}
 decisions=[
  dict(code='003029',previous='YES',latest='YES',scope='NO_NEW_ARTICLE_IN_BOUNDED_EVENING_SCOPE',reason='已取得的低收入未规模化、半年亏损反证保持，不把无新增索引视为风险清零。'),
  dict(code='603197',previous='YES',latest='YES',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='异常公告2026-080称经营正常、无应披露未披露重大事项；没有认证新的飞行汽车订单。激励对象3人离职、28000股按18.68回购注销是激励计划操作，不作二级市场增持新催化。'),
  dict(code='002085',previous='YES',latest='YES',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='2026-046称经营正常、无未披露重大事项；26日预约三季报，当前盈利数据未公开，不预测利好报表；h5仍为有条件累计偏上，财报窗口为重要反证。'),
  dict(code='002889',previous='YES',latest='UNKNOWN',scope='ISSUER_MIRROR_TEXT_VS_DATED_PRICE_DIRECTION_CONFLICT',reason='原超时缺口已取得镜像。2026-034正文写15/16日跌幅偏离23.61%，与本批不复权日线15/16连续涨停冲突，另一公告转载也同文。未判定哪方错，不从镜像改写日线，保留来源冲突并取消主动正向、降为观察；原YES不可删除。'),
  dict(code='600487',previous='UNKNOWN',latest='UNKNOWN',scope='NO_NEW_ARTICLE_IN_BOUNDED_EVENING_SCOPE',reason='业绩驱动的趋势观察保持，不将无索引新增记为可买确认。'),
  dict(code='600860',previous='UNKNOWN',latest='UNKNOWN',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='公告临2026-025披露半年亏损6263.39万元、子公司北洋天青9月18日因重组信息涉嫌虚假记载等被立案。新取得旧风险与当前披露分开；不将BASE第三转成接力许可，执行关注排除。'),
  dict(code='600148',previous='UNKNOWN',latest='UNKNOWN',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='公告称19日静态PE221.72、行业27.75，暂无未披露重大事项；不把一价等同锁仓或可买，保持未知。'),
  dict(code='600178',previous='UNKNOWN',latest='UNKNOWN',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='公告半年亏损893.47万元，基本面无重大变化，兵装注入等不作为已核催化，保持观察。'),
  dict(code='603348',previous='UNKNOWN',latest='UNKNOWN',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='公告半年归母亏损21590.20万元，无未披露重大事项；维持完整同位观察而非默认转正。'),
  dict(code='002724',previous='UNKNOWN',latest='UNKNOWN',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='AI+智慧照明收入有限、未对整体业绩构成重大影响；不使用羊字关系推断，维持观察。'),
  dict(code='603278',previous='NO',latest='NO',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='半年亏损6399.14万元、无未披露重大事项，与原后排大分歧反证并存，维持偏不晋级。'),
  dict(code='000576',previous='NO',latest='NO',scope='NEW_NOTICE_ACTUALLY_RECEIVED19_COVER20',reason='一般异常波动说明并不证明新增催化，原同位价量判断保持。')]
 for row in late['rows']:
  decision=next((d for d in decisions if d['code']==row['code']),None)
  if decision:row['evening_additional_evidence']=decision['reason']
  if row['code']=='002889':
   row.update(next_close_limit_up_call='UNKNOWN',next_intraday_touch_call='UNKNOWN',evening_direction_changed_from='YES')
   row['cancel_conditions']+='；公告方向与行情冲突未澄清前，仅观察，不视为正向交易许可。'
   for h in ['2','3','5']:
    row['horizons'][h]=dict(cumulative_up_call='UNKNOWN',every_day_up_call='UNKNOWN',consecutive_limit_ups_call='UNKNOWN',path='来源冲突补核后的晚间弃判；原下午全部路径保留，不以本版本消除原误报或漏报。')
  if row['code']=='002085':row['horizons']['5']['path']+='；26日预约三季报窗口，尚未知结果，不能推断财报利好。'
 late.update(formed_at=b.now(),version='EVENING_APPEND_AFTER_MODELS_AND_JOINT_DISCUSSION',model_exposure='MODEL_OUTPUT_ALREADY_SEEN',gemini_current_views_exposed=True,original_author_record=str(original.resolve()),original_sha256=b.sha(original),original_afternoon_formed_at=own['formed_at'],sources=own['sources']|proofs,new_fits=0,current_forecast_formal=False)
 b.write_new(OUT/'independent_full39_EVENING_APPEND_AFTER_MODEL.json',late)
 b.write_new(OUT/'evening_evidence_review_APPEND_ONLY.json',dict(formed_at=b.now(),signal_date='2026-10-19',target_date='2026-10-20',cover_date20_not_future_retrieval=True,source_acquisition_is19_evening=True,first_publication_times_not_invented=True,decisions=decisions,original_afternoon_predictions_preserved=True,top3_unchanged=late['top3'],watch2_unchanged=late['watch2'],only_direction_changed='002889 YES->UNKNOWN',sources=proofs,new_fits=0,new_formal_predictions=0,source_policy_change='Evening requests include next official cover date but only actually received material; no future data acquisition is manufactured.'))
 print(dict(formed_at=late['formed_at'],top3=late['top3'],direction_changed='002889 YES->UNKNOWN',new_fits=0))
if __name__=='__main__':main()
