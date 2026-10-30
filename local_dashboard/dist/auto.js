'use strict';
const collectionLabels={SUCCESS_PUBLIC_DATA_ONLY:'采集成功 · 仅事实',PARTIAL_PUBLIC_DATA:'部分成功 / 有缺口',FAILED:'本轮采集失败',CURRENT_SESSION_NOT_CAPTURED:'当前交易日尚未采集'};
const sessionLabels={TRADING_WINDOW:'交易时段',NON_TRADING_DAY:'非交易日',BEFORE_OPEN:'盘前',AFTER_CLOSE:'盘后',SESSION_BREAK:'午休',CALENDAR_OUT_OF_SCOPE:'交易日历超出范围'};
const sourceLabels={EXACT_DATE_PROVIDER_POOL:'日期通过 · 仅供应商范围',UNAVAILABLE_OR_DATE_REJECTED:'失败 / 日期拒收',QUOTES_RECEIVED:'报价已返回',PARTIAL_OR_STALE_QUOTES:'报价缺失 / 过期',UNAVAILABLE_OR_INVALID:'失败 / 无合格报价'};
const quoteLabels={EXACT_SESSION_CLOSED_QUOTE_NOT_LIVE:'对应交易日旧报价 · 非实时',STALE_OTHER_SESSION:'其他交易日旧报价 · 不可用于本日',INVALID_FUTURE_TIMESTAMP:'异常未来时钟 · 拒用',VERIFIED_PUBLIC_QUOTE_FRESH:'公开报价距抓取≤15秒 · 非盘口认证',DELAYED_PUBLIC_QUOTE:'公开报价延迟15—120秒',STALE_PUBLIC_QUOTE:'过期报价>120秒'};
let autoStatusBusy=false;
async function loadLiveStatus(){
 if(autoStatusBusy)return;autoStatusBusy=true;
 try{
  const s=await get('/api/live-status'),d=s.data||{},ctx=s.context||{},status=d.status||'CURRENT_SESSION_NOT_CAPTURED';
  $('liveState').className='pill '+(status==='SUCCESS_PUBLIC_DATA_ONLY'?'blue':status==='FAILED'?'danger':'archive');
  $('liveState').textContent=(s.running?'正在采集 · ':'')+(collectionLabels[status]||'等待首轮采集');
  $('liveSummary').textContent='上海日期 '+ctx.today+' · '+(sessionLabels[ctx.state]||ctx.state)+' · 本轮目标交易日 '+(ctx.target_day||'未知')+' · 下一交易日 '+(ctx.next_session||'待核')+'。'+(ctx.state==='NON_TRADING_DAY'?'休市期间没有当天成交行情，以下是最近交易日的重新抓取结果。':'公开来源不保证零延迟；逐行按源时间判断新旧。');
  const rows=d.pools||[];
  const count=k=>(d.requests||[]).some(x=>x.kind===k&&x.status==='EXACT_DATE_PROVIDER_POOL')?rows.filter(r=>r.kind===k).length:'未知';
  $('liveMetrics').innerHTML=[['涨停（主板）',count('limit_up')],['炸板（主板）',count('failed_board')],['跌停（主板）',count('limit_down')],['公开报价条数',(d.quotes||[]).length||'未取得']].map(([label,v])=>'<div><b>'+esc(v)+'</b><span>'+label+'</span></div>').join('');
  const clocks=(d.quotes||[]).map(x=>x.quote_time).sort(),quoteTime=clocks.length?time(clocks[0])+' — '+time(clocks.at(-1)):'未取得';
  $('liveSources').innerHTML=(d.requests||[]).map(r=>'<tr><td>'+esc(r.source)+'<br>'+esc(r.kind)+'</td><td>'+esc(r.requested_day)+'</td><td>'+esc(r.kind==='quote'?quoteTime:(r.source_day||'未知')+'（仅日期，无盘中时钟）')+'</td><td>'+esc(time(r.received_at||r.completed_at))+'</td><td>'+esc(sourceLabels[r.status]||r.status)+(r.error?'<br>'+esc(r.error):'')+'</td></tr>').join('')||'<tr><td colspan="5">本次目标日没有可用采集收据，旧日期结果不填入当前日。</td></tr>';
  $('livePolicy').textContent=s.policy+' 最近完成：'+time(d.completed_at)+'（上海）；页面每20秒只重读本机状态，不额外触发行情请求。';
  const poolRows=rows.map(r=>'<tr><td>'+esc(r.name)+'<br>'+esc(r.code)+'</td><td>'+esc(r.kind==='limit_up'?(r.boards===1?'首板':r.boards+'板'):r.kind==='failed_board'?'炸板':'跌停')+'</td><td>'+n(r.price_cents/100)+'</td><td>'+esc(r.source_day)+'</td><td>公开池，未核完整日线/原风险筛选；无模型分数</td></tr>').join('');
  const priceRows=(d.quotes||[]).map(r=>'<tr><td>'+esc(r.name)+'<br>'+esc(r.symbol)+'</td><td>'+n(r.last)+'</td><td class="'+sign(r.change_pct)+'">'+pct(r.change_pct)+'</td><td>'+esc(time(r.quote_time))+'</td><td>'+esc(quoteLabels[r.status]||r.status)+'</td></tr>').join('');
  $('liveQuotes').innerHTML='<h4>本轮主板事件池 · 不生成推荐排序</h4><div class="live-tables"><table><thead><tr><th>股票</th><th>类型</th><th>供应商价格</th><th>源交易日</th><th>资格</th></tr></thead><tbody>'+poolRows+'</tbody></table></div><h4>本轮公开报价 · 保留每股行情时钟</h4><div class="live-tables"><table><thead><tr><th>标的</th><th>报价</th><th>涨跌</th><th>源时钟（上海）</th><th>状态</th></tr></thead><tbody>'+priceRows+'</tbody></table></div>';
  const gap=s.archive_review_date!==ctx.target_day?'当日事实与旧复盘分开：当前采集目标 '+ctx.target_day+'；最新已留档复盘仍为 '+s.archive_review_date+'。':'当前采集与最新归档日期同为 '+ctx.target_day+'；下方所选日期以标题为准，重新抓到的报价不会改写原预测。';
  $('liveReviewGap').textContent=gap+' 自动每日研判尚未接通：缺完整合格模型输入、当日独立情绪与逐股审阅、公告/同位核验及冻结环节。无合格输入不出新BASE，不沿用旧排名当新预测。';
 }catch(e){
  $('liveState').className='pill danger';$('liveState').textContent='采集状态不可读';$('liveSummary').textContent='本机采集状态读取失败：'+e.message+'；未将上轮数据显示为本轮成功。';$('liveMetrics').innerHTML='';$('liveSources').innerHTML='';$('liveQuotes').innerHTML='';$('liveReviewGap').textContent='归档复盘仍在下方保留原日期；当前更新状态未知。';
 }finally{autoStatusBusy=false;}
}
loadLiveStatus();setInterval(()=>{if(!document.hidden)loadLiveStatus();},20000);
$('reload').addEventListener('click',loadLiveStatus);
