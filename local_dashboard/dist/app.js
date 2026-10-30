'use strict';
let report=null,filter='all';
const $=id=>document.getElementById(id);
const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=(v,d=2)=>typeof v==='number'?v.toFixed(d):'未知';
const pct=v=>typeof v==='number'?(v>0?'+':'')+v.toFixed(2)+'%':'未知';
const sign=v=>v>0?'positive':v<0?'negative':'';
const time=v=>{if(!v)return '未取得';const d=new Date(v);return Number.isNaN(+d)?String(v):d.toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});};
const call=v=>({YES:'偏晋级',NO:'偏不晋级',UNKNOWN:'未知 / 观察'}[v]||'未输出');
const screen=v=>({WITHIN_ORIGINAL_RESEARCH_SCREEN:'原筛选范围内',EXCLUDED_CAP:'超市值上限',EXCLUDED_THIN_PROXY:'原薄盘筛选排除',UNKNOWN_MISSING_EXACT_DAILY:'缺合格日线'}[v]||v||'待核');
const kind=r=>r.kind==='limit_up'?(r.boards>1?r.boards+'连板':'首板'):r.kind==='failed_board'?'炸板':'跌停';
async function get(url){const r=await fetch(url,{cache:'no-store'});const x=await r.json();if(!r.ok)throw new Error(x.error||r.status);return x;}
function spark(rows,base){const vals=rows.map(x=>x.price/base*100-100);let lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo)hi=lo+1;return `<svg class="sparkline" viewBox="0 0 250 36" preserveAspectRatio="none" aria-hidden="true"><path d="${vals.map((v,i)=>(i?'L':'M')+(i/(vals.length-1)*250).toFixed(1)+','+(33-(v-lo)/(hi-lo)*30).toFixed(1)).join(' ')}" fill="none" stroke="${vals.at(-1)>=0?'#db7777':'#52a58d'}" stroke-width="1.7"/></svg>`;}
async function load(day){$('loading').hidden=false;$('error').hidden=true;$('reload').disabled=true;try{report=await get('/api/report?date='+encodeURIComponent(day));render();$('content').hidden=false;}catch(e){$('content').hidden=true;$('error').hidden=false;$('error').textContent='归档读取失败：'+e.message+'。未用旧数据冒充本次结果。';}finally{$('loading').hidden=true;$('reload').disabled=false;}}
function render(){
 $('dayTitle').textContent=report.day;$('target').textContent='预案目标交易日 '+report.target_day+' · 独立留档后对照原模型 · 盘后资料单列时钟';
 $('clock').textContent='行情截至 '+report.market_at.replace(' Asia/Shanghai','（上海）');
 $('indexes').innerHTML=report.indexes.length?report.indexes.map(x=>{const p=(x.close/x.previous_close-1)*100;return `<article class="index-card"><div class="index-title">${esc(x.name)}<span>${esc(x.symbol)}</span></div><div class="index-value ${sign(p)}">${n(x.close)}<span class="index-delta">${pct(p)}</span></div>${spark(x.rows,x.previous_close)}</article>`;}).join(''):'<p class="muted">此归档没有已绑定指数分钟，保留未知。</p>';
 $('summary').textContent=report.summary;$('formation').textContent='人工判断形成：'+time(report.independent_formed_at)+'（上海）';
 $('scope').textContent=report.scope;$('relayCount').textContent=report.counts.relay;
 $('counts').innerHTML=[['first','首板'],['relay','连板'],['failed_board','炸板'],['limit_down','跌停']].map(([k,v])=>`<div><strong>${report.counts[k]}</strong><span>${v}</span></div>`).join('');
 $('feedback').innerHTML=Object.entries(report.feedback).map(([k,v])=>`<tr><td>${esc(k)}板</td><td><b>${v.promoted} / ${v.total}</b></td><td>${v.unknown}</td><td class="${sign(v.mean_return_pct)}">${pct(v.mean_return_pct)}</td><td>${v.loss7_count}</td></tr>`).join('')||'<tr><td colspan="5">缺少完整前池反馈</td></tr>';
 $('baseIdentity').textContent=report.base_identity+'。'+report.base_qualification+'。原分数、原方向和原排名保留。';
 const base=report.rows.filter(x=>x.base).sort((a,b)=>a.base.rank-b.base.rank);
 $('modelTop').innerHTML=base.slice(0,3).map(r=>`<article class="model-card"><span class="rank">0${r.base.rank}</span><strong>${esc(r.name)}</strong><span class="stock-code">${r.code} · ${r.boards}板</span><div class="model-score">${n(r.base.score_display,1)} <small style="font-size:11px;font-weight:400">分</small></div><p>原BASE排序 · ${call(r.base.BASE_direction)}</p></article>`).join('');
 $('baseRows').innerHTML=base.map(r=>`<tr><td>${r.base.rank}</td><td>${stockButton(r)}</td><td>${r.boards} → ${r.boards+1}</td><td class="score">${n(r.base.score_display,1)}</td><td><span class="pill ${r.base.BASE_direction==='YES'?'blue':'neutral'}">${call(r.base.BASE_direction)}</span></td><td class="table-note">${esc(r.participation||'见原资格与个股详情；非自动买入')}</td></tr>`).join('');
 const gaps=report.first_qualification.gaps||[];
 const gapNames={SOURCE_UNIVERSE_MISMATCH:'缺少与训练一致的全市场横截面，事件股池不能替代市场涨跌分布。',HISTORY_LENGTH_AND_SCOPE:'当前已绑定历史不足60个相邻交易日，长周期股性与上市状态无法完整核算。',HISTORICAL_FIELD_MISSING:'缺逐日换手、成交额及同口径历史梯队，不能用当前数值倒填过去。',EXACT_PRICE_MISSING:'中基健康缺本次合格日线；仍保留在完整首板池。',NO_CURRENT_QUALIFIED_FEATURE_BINDING:'当日完整62字段输入及原筛选资格尚未绑定，不能直接输出分数。'};
 $('firstGaps').innerHTML=(Array.isArray(gaps)?gaps:Object.keys(gaps)).map(x=>`<li>${esc(typeof x==='string'?x:gapNames[x.id]||'该来源资格待核，详情保留于原收据。')}</li>`).join('');
 $('firstModel').textContent=(report.first_qualification.model_identity?'已核真实保存的62字段首板研究模型；本日未推理':'当日首板BASE未输出')+' · 完整首板 '+report.counts.first+' 只';
 $('correction').innerHTML=report.correction.text?'<b>已追加纠正 · 正裕只作人工分歧观察</b>'+esc(report.correction.text):'历史归档：原判断保留。该日期不沿用其他日期的排名或竞价数值。';
 const relayCodes=report.correction.text?['603280','603089','603278']:[];
 $('relayWatch').innerHTML=relayCodes.map(c=>watchCard(report.rows.find(x=>x.code===c),false)).join('')||'<p class="muted">该日期未提供本网页可用的纠正后执行卡；请查原始留档。</p>';
 $('firstWatch').innerHTML=report.first_top3.map((c,i)=>watchCard(report.rows.find(x=>x.code===c),true,i+1)).join('');
 $('training').textContent=report.training.status+'；本次新增拟合 '+report.training.new_fits+' 次。'+report.training.effect;
 $('blockers').innerHTML=report.training.blockers.map(x=>'<li>'+esc(x)+'</li>').join('');
 $('sources').innerHTML=report.sources.map(x=>`<tr><td>${/^https:\/\//.test(x.url||'')?`<a href="${esc(x.url)}" target="_blank" rel="noopener noreferrer">${esc(x.label)} ↗</a>`:esc(x.label)}</td><td>${esc(x.market_at)}</td><td>${esc(time(x.fetched_at))}</td><td>${esc(x.status)} / 归档</td></tr>`).join('')+`<tr><td>腾讯分钟（绑定原响应）</td><td>${esc(report.day)} 09:30—15:00（上海）</td><td>${esc(time(report.minutes[0]?.fetched_at))}</td><td>归档；精确日期，非实时队列</td></tr>`;
 $('refreshTime').textContent='页面重读时间：'+time(report.served_at)+'（上海）；重读时间不等于行情时间。原始归档形成：'+time(report.archive_formed_at)+'（上海）。';
 $('provenance').textContent=JSON.stringify({异常:report.issues,源文件哈希:report.provenance,说明:'海洋与中广首次封板字段有来源冲突，已追加分钟核对；未知日线不填零。'},null,2);
 $('evidenceImage').innerHTML=report.evidence_image?`<h3>已归档的分时证据</h3><a href="${esc(report.evidence_image)}" target="_blank"><img src="${esc(report.evidence_image)}" loading="lazy" alt="指数、机器人高低位与旧情绪的归档分时对照，日期${esc(report.day)}"></a>`:'';
 drawChart();renderPool();
}
function stockButton(r){return `<button class="table-link" data-code="${esc(r.code)}"><span class="stock-name">${esc(r.name)}</span><span class="stock-code">${esc(r.code)}</span></button>`;}
function watchCard(r,first,rank){if(!r)return '';const card=r.card;const gate=card?.gate||'人工观察，缺少本网页可用执行卡';return `<article class="watch-card"><div class="card-top"><span class="pill ${r.code==='603089'?'danger':first?'blue':'neutral'}">${first?'人工观察 '+rank:r.code==='603089'?'与BASE分歧':r.code==='603278'?'情绪锚':'条件观察'}</span><span class="stock-code">${esc(r.code)}</span></div><h4>${esc(r.name)}</h4><span class="small muted">${kind(r)} · ${r.base?'BASE '+n(r.base.score_display,1)+'分 / '+call(r.base.BASE_direction):'首板BASE未输出'}</span><p class="reason">${esc(r.support)}</p><div class="watch-price"><span>收盘 <b>${n(r.close)}</b></span><span>涨停参考 <b>${n(card?.limit_up_reference)}</b></span></div><div class="gate">${esc(gate)}</div><button class="button" data-code="${r.code}">查看理由、竞价与取消条件 →</button></article>`;}
function renderPool(){if(!report)return;const q=$('search').value.trim().toLowerCase();const rows=report.rows.filter(r=>(filter==='all'||filter==='relay'&&r.kind==='limit_up'&&r.boards>1||filter==='first'&&r.kind==='limit_up'&&r.boards===1||filter===r.kind)&&(r.name.toLowerCase().includes(q)||r.code.includes(q)));
 $('poolCount').textContent='显示 '+rows.length+' / '+report.rows.length+' 只 · 日线未知 '+report.counts.unknown+' 只';
 $('poolRows').innerHTML=rows.map(r=>`<tr><td>${stockButton(r)}</td><td>${kind(r)}</td><td>${n(r.close)}</td><td class="${sign(r.change_pct)}">${pct(r.change_pct)}</td><td>${n(r.turnover_pct)}%</td><td>${typeof r.cap_cny==='number'?n(r.cap_cny/1e8)+'亿':'未知'}</td><td class="table-note">${esc(screen(r.screen))}${r.close===null?' · 缺合格日线':''}</td><td><button class="table-link" data-code="${r.code}">详情 ↗</button></td></tr>`).join('');$('noResults').hidden=rows.length>0;
}
function detail(code){const r=report.rows.find(x=>x.code===code);if(!r)return;const c=r.card;$('detailTitle').innerHTML='<h2>'+esc(r.name)+'</h2><span class="small muted">'+r.code+' · '+kind(r)+' · 归档 '+report.day+'</span>';
 let body=`<div class="detail-facts"><span>收盘<b>${n(r.close)}</b></span><span>原BASE分数<b>${r.base?n(r.base.score_display,1):'未输出'}</b></span><span>原BASE方向<b>${r.base?call(r.base.BASE_direction):'未输出'}</b></span><span>原筛选<b style="font-size:13px">${esc(screen(r.screen))}</b></span></div>`;
 if(r.participation)body+='<div class="correction">'+esc(r.participation)+'</div>';
 body+='<h3>人工支持理由</h3><p>'+esc(r.support)+'</p><h3>反证与对手盘</h3><p>'+esc(r.counter)+'</p><p class="small muted">原独立方向：'+call(r.independent_call)+'。这是历史留档，不能替代当前参与状态；人工异议未证明优于BASE。</p>';
 if(c){body+='<h3>竞价预案 · '+esc(c.target_date)+'</h3><div class="notice">'+esc(c.numerical_rule)+'。'+esc(c.auction_actual)+'</div><table><thead><tr><th>情景</th><th>价格 / 元</th></tr></thead><tbody><tr><td>符合预期</td><td>'+c.expected_price.map(v=>n(v)).join(' — ')+'</td></tr><tr><td>强于预期</td><td>'+c.strong_price.map(v=>n(v)).join(' — ')+'</td></tr><tr><td>低于预期</td><td>低于 '+n(c.weak_price_below)+'</td></tr></tbody></table><p><b>当前资格：</b>'+esc(c.gate)+'</p><p><b>确认对象：</b>'+esc(c.confirmation_objects)+'</p><h3>09:30起</h3><p>'+esc(c.after0930)+'</p><h3>09:35后</h3><p>'+esc(c.after0935)+'</p><h3>取消条件</h3><p>'+esc(c.cancel)+'</p><h3>后续路径预案</h3>'+c.paths.map(x=>'<p><b>'+esc(x.date)+'</b> '+esc(x.expectation)+'</p>').join('');}
 else body+='<div class="notice">本网页没有该股的合格当日执行卡，不生成买入价或用其他股票条件代替。</div>';
 body+='<p class="small muted">所有金额、价格为归档日期口径。没有实际订单或成交记录。A股T+1下，当日买入不能承诺当日止损。</p>';$('detailBody').innerHTML=body;$('detail').showModal();
}
function drawChart(){if(!report)return;const groups={anchors:['sh000001','sz399006','sh603278','sz002724'],low:['sh603089','sh603677','sz002009','sh603081'],old:['sz002724','sz002242','sh600825','sz002185']};const selected=groups[$('chartChoice').value].map(s=>report.minutes.find(x=>x.symbol===s)).filter(Boolean);if(!selected.length){$('chart').textContent='此归档没有对应分钟证据，未用其他日期替代。';$('legend').innerHTML='';$('chartSource').textContent='';return;}
 const colors=['#345dcb','#21a28d','#de7751','#a174b9'],W=1000,H=250,L=47,R=18,T=12,B=30;const index=t=>{const h=Number(t.slice(0,2)),m=Number(t.slice(2));const k=h*60+m;return k<=690?k-570:135+(k-780);};
 const all=selected.flatMap(s=>s.rows.map(r=>r.price/s.previous_close*100-100));let lo=Math.floor(Math.min(...all,0)/2)*2,hi=Math.ceil(Math.max(...all,0)/2)*2;if(hi-lo<2)hi=lo+2;const xx=v=>L+v/255*(W-L-R),yy=v=>H-B-(v-lo)/(hi-lo)*(H-B-T);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(report.day)}归档分时涨跌对照"><rect x="${xx(120)}" y="${T}" width="${xx(135)-xx(120)}" height="${H-B-T}" fill="#f0f3f7"/>`;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;svg+=`<line x1="${L}" x2="${W-R}" y1="${yy(v)}" y2="${yy(v)}" stroke="#e9edf4"/><text x="${L-8}" y="${yy(v)+4}" text-anchor="end" font-size="10" fill="#8a98aa">${v.toFixed(1)}%</text>`;}
 for(const [v,label] of [[0,'09:30'],[60,'10:30'],[120,'11:30'],[135,'13:00'],[195,'14:00'],[255,'15:00']])svg+=`<text x="${xx(v)}" y="${H-8}" text-anchor="middle" font-size="10" fill="#8a98aa">${label}</text>`;
 selected.forEach((s,i)=>{let path='',last=-1;s.rows.forEach(r=>{const t=index(r.time);if(t<0||t>255)return;path+=(last<=120&&t>=135||last<0?'M':'L')+xx(t).toFixed(1)+','+yy(r.price/s.previous_close*100-100).toFixed(1)+' ';last=t;});svg+=`<path d="${path}" fill="none" stroke="${colors[i]}" stroke-width="2"/>`;});
 $('chart').innerHTML=svg+'</svg>';$('legend').innerHTML=selected.map((s,i)=>`<span><i style="background:${colors[i]}"></i>${esc(s.name)}</span>`).join('');$('chartSource').textContent='来源：腾讯已存原响应；行情 '+report.day+'，取得 '+time(selected[0].fetched_at)+'（上海）。无Level-2、无真实成交认证。';
}
document.addEventListener('click',e=>{const button=e.target.closest('[data-code]');if(button)detail(button.dataset.code);const f=e.target.closest('[data-filter]');if(f){filter=f.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('selected',x===f));renderPool();}});
$('closeDetail').onclick=()=>$('detail').close();$('detail').addEventListener('click',e=>{if(e.target===$('detail')){const r=$('detail').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('detail').close();}});
$('search').addEventListener('input',renderPool);$('chartChoice').onchange=drawChart;$('date').onchange=()=>load($('date').value);
$('reload').onclick=async()=>{try{const c=await get('/api/catalog');const selected=$('date').value;$('date').innerHTML=c.dates.map(d=>`<option value="${d}">${d} 归档</option>`).join('');$('date').value=c.dates.includes(selected)?selected:c.latest;await load($('date').value);}catch(e){$('error').hidden=false;$('error').textContent='重读失败：'+e.message;}};
$('export').onclick=()=>{if(!report)return;const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));a.href=url;a.download='龙头战法_归档_'+report.day+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.querySelectorAll('.sidebar nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelectorAll('.sidebar nav a').forEach(x=>x.classList.remove('active'));a.classList.add('active');}));
(async()=>{try{const c=await get('/api/catalog');if(!c.latest)throw new Error('没有可用归档');$('date').innerHTML=c.dates.map(d=>`<option value="${d}">${d} 归档</option>`).join('');await load(c.latest);if(location.search){history.replaceState(null,'',location.pathname+location.hash);}}catch(e){$('loading').hidden=true;$('error').hidden=false;$('error').textContent='无法连接本地服务：'+e.message+'。请使用启动网页.cmd后显示的HTTP地址打开。';}})();
