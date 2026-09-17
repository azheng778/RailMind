/* Visual composition and client-side views; existing API owns all diagnoses. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const q = selector => document.querySelector(selector);
  const make = (tag, cls, html = '') => { const el = document.createElement(tag); el.className = cls; el.innerHTML = html; return el; };
  const state = { caps: [], kb: [], orders: [], capId: '', kbId: '', orderId: '', page: 1 };
  const domainNames = { composite_structure:'车体结构', underbody:'车底设备', pantograph:'受电弓', cabin:'车厢内部', lineside:'线路侧' };
  const statusNames = { DRAFT:'待确认', OPEN:'处理中', CLOSED:'已闭环', REJECTED:'已驳回' };
  const art = '/static/web/assets/train-night.png';
  const cover = c => c.domain === 'cabin' ? '/static/cabin_vlm/frames/w1_f1.jpg' : c.domain === 'pantograph' ? '/static/panto_samples/video_frame_0242.jpg' : c.domain === 'underbody' ? '/static/door_demo/demo_clean.jpg' : c.domain === 'lineside' ? '/static/fastener_demo/samples/sample_broken.jpg' : art;
  const metric = (value, label, id = '') => `<div class="metric"><b${id ? ` id="${id}"` : ''}>${esc(value)}</b><span>${esc(label)}</span></div>`;
  const definition = pairs => `<dl>${pairs.map(([k,v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v ?? '—')}</dd></div>`).join('')}</dl>`;
  const empty = text => `<div class="empty-state">${esc(text)}</div>`;
  const keyFor = c => [c.document_id,c.page,c.chapter].join('|');
  function detailBox(id, title) { const el = make('aside','panel detail-panel',`<h3>${title}</h3><div id="${id}">${empty('选择左侧条目查看详情')}</div>`); return el; }
  function input(id, placeholder) { return `<input id="${id}" type="search" aria-label="${placeholder}" placeholder="${placeholder}">`; }
  function select(id, label, values) { return `<select id="${id}" aria-label="${label}"><option value="">${label}</option>${values.map(([value,text])=>`<option value="${esc(value)}">${esc(text)}</option>`).join('')}</select>`; }
  function banner(id, title, subtitle) { const el = make('div','panel section-banner',`<div><h3>${title}</h3><p>${subtitle}</p></div><span class="banner-motto">智行千里 · 运维无忧</span>`); byId('sec-'+id).prepend(el); return el; }
  function navigate(section) { navToSec(section); }
  function button(label, fn, cls='btn') { const b=make('button',cls,esc(label)); b.type='button'; b.onclick=fn; return b; }

  // Keep the original page navigation, replacing only its icon treatment.
  const paths = [
    '<path d="m3 10 9-7 9 7M5 9v12h14V9M9 21v-7h6v7"/>',
    '<circle cx="12" cy="12" r="8"/><path d="M12 4V2m0 20v-2M4 12H2m20 0h-2M12 7v5l3 2"/>',
    '<rect x="4" y="3" width="16" height="14" rx="3"/><path d="M7 21l2-4m8 4-2-4M4 10h16M9 6h6M8 14h1m6 0h1"/>',
    '<rect x="3" y="5" width="18" height="15" rx="2"/><path d="M7 2v6m10-6v6M3 10h18M7 14h3m4 0h3M7 17h3"/>',
    '<path d="m8 3-4 18M16 3l4 18M7 6h10M6 11h12M5 17h14"/>',
    '<path d="m12 2 9 5v10l-9 5-9-5V7l9-5Zm0 10 9-5M12 12 3 7m9 5v10"/>',
    '<path d="M12 5C8 2 4 3 2 4v16c4-2 7-1 10 1 3-2 6-3 10-1V4c-2-1-6-2-10 1Zm0 0v16M5 8h4m-4 4h4m6-4h4m-4 4h4"/>',
    '<rect x="3" y="6" width="18" height="15" rx="2"/><path d="M8 6V3h8v3M3 12h18M10 12v3h4v-3"/>'
  ];
  document.querySelectorAll('nav button').forEach((b,i)=>{
    const label=b.textContent.replace(/^\S+\s*/, '');
    b.innerHTML=`<span class="nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24">${paths[i]}</svg></span><span>${esc(label)}</span>`;
    b.addEventListener('click',()=>{
      document.querySelectorAll('nav button').forEach(n=>n.setAttribute('aria-current',n===b?'page':'false'));
      window.scrollTo({top:0,behavior:'instant'});
    });
  });
  q('nav button.active').setAttribute('aria-current','page');
  q('nav').setAttribute('aria-label','平台主导航');
  new ResizeObserver(()=>document.documentElement.style.setProperty('--header-height',q('header').getBoundingClientRect().height+'px')).observe(q('header'));
  q('nav .foot').insertAdjacentHTML('beforeend','<div class="nav-brand">RailMind<small>AI for Safer Railways</small></div>');
  q('header .stat.ok').title='平台服务状态；各能力状态以能力中心为准';

  // Overview: primary three-column workspace, then health/inspection and totals.
  const consoleBar=q('.demo-bar');
  const consoleWrap=make('details','demo-console','<summary>演示控制台 · 展开运行诊断场景</summary>');
  consoleBar.before(consoleWrap); consoleWrap.append(consoleBar);
  const overview=q('#sec-overview'), columns=q('.g-main');
  const health=q('.g-main .health').closest('.panel');
  const lower=q('.g-mid'); const statistics=lower.children[0]; const inspection=lower.children[1];
  const secondary=make('div','overview-secondary'); secondary.append(health,inspection); lower.replaceWith(secondary);
  statistics.classList.add('overview-statistics'); secondary.after(statistics);
  health.querySelector('h3').append(button('详细监测 →',()=>navigate('running'),'btn mini tag'));
  inspection.querySelector('h3').append(button('查看详情 →',()=>navigate('station'),'btn mini tag'));
  const coverage=columns.children[1].firstElementChild;
  coverage.querySelector('svg').parentElement.replaceWith(make('div','train-scene','<div class="train-zones"><button data-go="station">受电弓系统<small>磨耗 / 电弧</small></button><button data-go="running">走行部<small>传动 / 冲击</small></button><button data-go="cabin">车厢内部<small>视频巡检</small></button><button data-go="lineside">轨道线路<small>异物 / 扣件</small></button></div>'));
  coverage.querySelector(':scope > .cap')?.remove();
  coverage.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>navigate(b.dataset.go));
  coverage.querySelector('h3').insertAdjacentHTML('beforeend','<span class="tag">多源融合监测</span>');
  q('.gauge-wrap').after(make('div','gauge-footer','<div><b data-mirror="route-mileage">—</b><span>累计里程 km</span></div><div><b data-mirror="route-speed">—</b><span>当前时速 km/h</span></div><div><b data-mirror="route-eta">—</b><span>距下一站</span></div>'));
  q('.gauge-wrap').parentElement.querySelector('.cap').textContent='演示遥测 · 经安全网关单向接入';
  [byId('wave-v'),byId('wave-f'),byId('run-v'),byId('run-i')].forEach(c=>{
    c.setAttribute('aria-label',c.id.includes('i')?'演示电流波形':'演示监测波形');
    c.after(make('div','wave-axis','<span>−60 s</span><span>−45 s</span><span>−30 s</span><span>−15 s</span><span>当前</span>'));
  });
  q('#chief-body .chief-empty').innerHTML='基于全量多源数据，进行故障分析、风险研判与运维建议，为列车运行提供辅助决策。<div class="chief-intro"><span class="ai-symbol">AI</span><span>多模态 · RAG · 专家经验</span></div>';
  const ask=make('button','ask-shortcut','自然语言提问 · 例如：受电弓电弧异常的可能原因？ <span>→</span>');
  ask.onclick=()=>{if(!byId('chat-win').classList.contains('open'))toggleChat();byId('chat-text').focus();};
  byId('kb-refs').parentElement.append(ask);
  byId('chat-fab').textContent='AI'; byId('chat-fab').setAttribute('role','button'); byId('chat-fab').tabIndex=0; byId('chat-fab').setAttribute('aria-label','打开运维助手');
  byId('chat-fab').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggleChat();}};

  banner('running','运行中监测','受电弓电气监测 · 传动系统健康 · SHM 结构冲击');
  banner('station','进站快检','图像采集与 AI 识别 · 检查门、车底部件及受电弓');
  q('#sec-station .section-banner > div').insertAdjacentHTML('beforeend','<div class="stepper"><span><b>01</b>进站识别</span><span><b>02</b>图像采集</span><span><b>03</b>智能检测</span><span><b>04</b>结果判定</span><span><b>05</b>生成记录</span></div>');
  q('#sec-station .section-banner > div').style.width='78%';
  const stationResult=make('div','panel inspection-result','<h3>检测结果与结论</h3><div id="station-result">'+empty('暂无快检结论，等待检测记录')+'</div>');
  q('#sec-station > .grid > div:last-child').prepend(stationResult);
  const stationGrid=q('#sec-station > .grid');stationGrid.classList.add('station-workspace');
  const stationRight=stationGrid.lastElementChild;
  const stationWear=stationRight.children[1],stationRecords=stationRight.children[2];
  stationGrid.firstElementChild.classList.add('station-images');stationWear.classList.add('station-wear');stationRecords.classList.add('station-records');
  stationGrid.append(stationResult,stationWear,stationRecords);stationRight.remove();
  stationGrid.querySelector('.station-images h3').textContent='检查门舱盖 · 图像对比与 AI 检测';
  stationGrid.querySelector('.station-images').insertAdjacentHTML('beforeend','<div class="ui-note">上图为已保存的正常 / 异常检测示例；右侧展示最近一次检测结论。</div>');
  banner('cabin','乘务员巡检 · 车厢内部实拍','视频回放与双窗口多模态分析 · G1 北京南 → 上海虹桥');
  const cabinGrid=q('#sec-cabin > .grid');cabinGrid.classList.add('cabin-workspace');
  const cabinRight=cabinGrid.lastElementChild,cabinWindows=cabinRight.children[0],cabinRecords=cabinRight.children[1];
  cabinGrid.append(cabinWindows,cabinRecords);cabinRight.remove();
  cabinRecords.classList.add('cabin-records');
  const cabinSummary=make('div','panel cabin-summary','<h3>车内巡检概况 <span class="tag">最近返回的巡检记录</span></h3><div id="cabin-summary">'+empty('等待巡检数据')+'</div>');cabinRecords.before(cabinSummary);
  banner('lineside','线路侧智能巡检','轨道异物识别 · 扣件缺陷检测 · 证据留存');
  const gallery=make('div','lineside-gallery',`<div class="panel"><h3>轨道异物检测 <span class="tag">YOLO26n</span></h3><div class="evidence-image"><img src="${art}" alt="高铁运营线路示意图"><span class="evidence-label">线路监测示意</span></div><p>鸟巢、漂浮物、气球及塑料袋等异物识别，结合面积占比判定限界风险。</p><div class="ui-note">检测结果与风险等级见下方实时记录。</div></div><div class="panel"><h3>扣件缺陷检测 <span class="tag">RFDD · YOLO26n</span></h3><div class="evidence-image"><img src="/static/fastener_demo/samples/sample_broken.jpg" alt="扣件缺陷检测样本"><span class="evidence-label">缺陷样本 · 非当前告警</span></div><p>识别扣件断裂、缺失等缺陷，检测信息不足时转入人工复核。</p><div class="ui-note">模型样本用于说明识别对象，实际状态以事件记录为准。</div></div>`);
  q('#sec-lineside > .panel.mb').after(gallery);

  // Capabilities: real registry data, selectable cards, filtering and detail.
  banner('caps','能力中心','汇聚行业 AI 能力，构建开放、可插拔的列车运维能力市场');
  const capMarket=q('#sec-caps > .panel.mb');
  const capToolbar=make('div','toolbar',input('cap-search','搜索能力名称、关键词、版本')+select('cap-domain','全部领域',Object.entries(domainNames))+select('cap-status','全部状态',[['online','在线'],['offline','离线']]));
  capToolbar.append(button('＋ 接入新能力',()=>registerDialog(),'btn primary')); capMarket.prepend(capToolbar);
  const capSplit=make('div','split-view');capMarket.before(capSplit);capSplit.append(capMarket,detailBox('cap-detail','能力详情'));
  const capStats=make('div','summary-strip');capStats.id='cap-summary';capSplit.before(capStats);
  ['cap-search','cap-domain','cap-status'].forEach(id=>byId(id).addEventListener('input',renderCapsView));

  // Knowledge: search and status filters operate on API-supplied excerpts.
  banner('kb','知识中心','沉淀专业知识 · 赋能智能运维');
  const kbTop=q('#sec-kb > .panel.mb');
  const kbToolbar=make('div','toolbar',input('kb-search','搜索文档、章节或关键词')+select('kb-status','全部状态',[['ACTIVE','已生效'],['DRAFT','草稿']]));kbTop.prepend(kbToolbar);
  const kbList=byId('kb-cards').parentElement;
  const kbSplit=make('div','split-view');kbList.before(kbSplit);kbSplit.append(kbList,detailBox('kb-detail','知识详情'));
  ['kb-search','kb-status'].forEach(id=>byId(id).addEventListener('input',renderKnowledge));

  // Workorders: missing reference completed using the same dashboard system.
  banner('wo','工单中心','从风险发现到人工确认，追踪每一次运维处置');
  const orderPanel=byId('wo-body').closest('.panel');
  const orderStats=make('div','summary-strip');orderStats.id='wo-summary';orderPanel.before(orderStats);
  const orderTools=make('div','toolbar',input('wo-search','搜索工单号、车辆或部件')+select('wo-status','全部状态',Object.entries(statusNames))+select('wo-risk','全部风险',[['HIGH','高风险'],['WARNING','中风险'],['OBSERVE','低风险'],['UNKNOWN','待复核']]));
  orderTools.append(button('导出当前结果',()=>exportOrders()));orderPanel.prepend(orderTools);
  const orderSplit=make('div','split-view');orderPanel.before(orderSplit);orderSplit.append(orderPanel,detailBox('wo-detail','工单详情与处置'));
  const pager=make('div','pagination');pager.id='wo-pagination';orderPanel.append(pager);
  ['wo-search','wo-status','wo-risk'].forEach(id=>byId(id).addEventListener('input',()=>{state.page=1;renderOrders();}));

  // Tables scroll inside their cards rather than widening the full application.
  document.querySelectorAll('.panel table').forEach(t=>{ const wrapper=make('div','table-scroll'); t.before(wrapper);wrapper.append(t); });
  document.querySelectorAll('img:not([alt])').forEach(img=>img.alt=img.closest('figure')?.querySelector('figcaption')?.textContent||'检测参考图');
  const dialog=make('dialog','ui-dialog');dialog.id='ui-dialog';document.body.append(dialog);
  function showDialog(title,body) { dialog.innerHTML='';dialog.append(button('关闭',()=>dialog.close(),'btn mini dialog-close'));dialog.insertAdjacentHTML('beforeend',`<h3>${esc(title)}</h3>${body}`);dialog.showModal(); }

  function renderCapsView() {
    const term=byId('cap-search').value.trim().toLowerCase(), domain=byId('cap-domain').value, online=byId('cap-status').value;
    const list=state.caps.filter(c=>(!term||JSON.stringify(c).toLowerCase().includes(term))&&(!domain||c.domain===domain)&&(!online||c.online===(online==='online')));
    if (!list.some(c=>c.capability_id===state.capId)) state.capId=list[0]?.capability_id||'';
    byId('caps-grid').innerHTML=list.length?list.map(c=>`<article class="capcard ${c.capability_id===state.capId?'selected':''}" role="button" tabindex="0" data-cap="${esc(c.capability_id)}"><h4>${esc(c.name)} <span class="mode mode-${esc(c.mode)}">${esc(modeCn(c.mode))}</span></h4><div class="cid">v${esc(c.version)} <span style="color:${c.online?'var(--ok)':'var(--danger)'}">● ${c.online?'在线':'离线'}</span></div><img class="cap-cover" src="${cover(c)}" alt="${esc(domainNames[c.domain]||c.domain)}能力示意"><div class="cap-summary">${esc(domainNames[c.domain]||c.domain)} · 支持 ${esc((c.input_types||[]).join(' / '))} 分析</div><div class="meta"><span>响应 ${esc(c.avg_latency_ms)} ms</span><span>调用 ${esc(c.call_count)} 次</span></div></article>`).join(''):empty('未找到符合条件的能力');
    byId('cap-summary').innerHTML=metric(state.caps.filter(c=>c.online).length,'在线能力 / 共 '+state.caps.length+' 项')+metric(state.caps.filter(c=>c.mode==='PRIMARY').length,'主用能力')+metric(state.caps.reduce((sum,c)=>sum+(c.call_count||0),0),'累计调用次数')+metric(state.caps.filter(c=>!c.online||c.error_rate>0).length,'离线或存在错误的能力');
    byId('caps-grid').querySelectorAll('[data-cap]').forEach(el=>{el.onclick=()=>{state.capId=el.dataset.cap;renderCapsView();};el.onkeydown=e=>{if(e.key==='Enter')el.click();};});
    const selected=list.find(c=>c.capability_id===state.capId)||list[0];
    if(!selected){byId('cap-detail').innerHTML=empty('没有可显示的能力');return;}
    state.capId=selected.capability_id;
    byId('cap-detail').innerHTML=`<img class="cap-cover" src="${cover(selected)}" alt="能力领域示意"><h4>${esc(selected.name)}</h4><p><span class="mode mode-${esc(selected.mode)}">${esc(modeCn(selected.mode))}</span> · ${selected.online?'在线运行':'当前离线'}</p>${definition([['能力 ID',selected.capability_id],['所属领域',domainNames[selected.domain]||selected.domain],['当前版本','v'+selected.version],['服务提供方',selected.provider],['输入数据',(selected.input_types||[]).join(' / ')],['适用场景',(selected.supported_scenes||[]).join(' / ')],['平均响应',selected.avg_latency_ms+' ms'],['错误率',(selected.error_rate*100).toFixed(1)+'%']])}<div class="detail-actions"></div>`;
    byId('cap-detail').querySelector('.detail-actions').append(button('查看完整描述',()=>showDialog('能力注册信息',`<pre>${esc(JSON.stringify(selected,null,2))}</pre>`)),button('运行模式',()=>modeDialog(selected),'btn primary'));
  }
  function modeDialog(c) {
    showDialog('调整能力运行模式',`<p>${esc(c.name)}</p><label>运行模式${select('mode-choice','选择模式',['PRIMARY','STANDBY','AUXILIARY','OBSERVE','DISABLED'].map(m=>[m,modeCn(m)]))}</label><p class="ui-note">保存后将更新平台的任务分派模式。</p><div id="mode-error" class="ui-error" role="alert"></div>`);
    byId('mode-choice').value=c.mode;
    dialog.append(button('保存模式',async()=>{
      try {const mode=byId('mode-choice').value;if(!mode)throw new Error('请选择运行模式');await api('/api/v1/capabilities/'+encodeURIComponent(c.capability_id)+'/mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});dialog.close();toast('能力模式已更新');refreshAll();} catch(e){byId('mode-error').textContent=e.message;}
    },'btn primary'));
  }
  function registerDialog() {
    showDialog('接入新能力','<p>填写外部能力的描述 JSON 和服务地址。成功注册后会出现在能力市场。</p><label>能力描述 JSON<textarea id="register-descriptor" placeholder="粘贴 capability 描述 JSON"></textarea></label><label>服务地址<input id="register-url" type="url" placeholder="http://127.0.0.1:8901"></label><div id="register-error" class="ui-error" role="alert"></div>');
    dialog.append(button('注册能力',async()=>{try{const descriptor=JSON.parse(byId('register-descriptor').value);const base_url=byId('register-url').value.trim();if(!/^https?:\/\//.test(base_url))throw new Error('请填写 http 或 https 服务地址');await api('/api/v1/capabilities/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({descriptor,base_url,mode:'AUXILIARY'})});dialog.close();toast('能力已注册');refreshAll();}catch(e){byId('register-error').textContent=e.message;}},'btn primary'));
  }

  function renderKnowledge() {
    const term=byId('kb-search').value.trim().toLowerCase(),status=byId('kb-status').value;
    const list=state.kb.filter(c=>(!term||[c.title,c.chapter,c.text,c.document_id].join(' ').toLowerCase().includes(term))&&(!status||c.status===status));
    const chosen=list.find(c=>keyFor(c)===state.kbId)||list[0];state.kbId=chosen?keyFor(chosen):'';
    byId('kb-cards').innerHTML=list.length?list.map((c,i)=>`<article class="kbcard ${keyFor(c)===state.kbId?'selected':''}" tabindex="0" role="button" data-kb="${i}"><b>《${esc(c.title)}》</b><div class="meta"><span class="mode mode-${c.status==='ACTIVE'?'PRIMARY':'AUXILIARY'}">${esc(c.status)}</span>　v${esc(c.version)} · P${esc(c.page)}　${esc(c.chapter)}</div><p>${esc(c.text)}…</p></article>`).join(''):empty('未检索到匹配条目，试试其他关键词');
    byId('kb-cards').querySelectorAll('[data-kb]').forEach(el=>{el.onclick=()=>{state.kbId=keyFor(list[Number(el.dataset.kb)]);renderKnowledge();};el.onkeydown=e=>{if(e.key==='Enter')el.click();};});
    if(!chosen){byId('kb-detail').innerHTML=empty('没有匹配的知识内容');return;}
    byId('kb-detail').innerHTML=`<h4>《${esc(chosen.title)}》</h4>${definition([['文档编号',chosen.document_id],['版本','v'+chosen.version],['生效状态',chosen.status],['页码','P'+chosen.page],['章节',chosen.chapter]])}<h4>内容摘要</h4><p>${esc(chosen.text)}…</p><div class="ui-note">此处展示知识库接口提供的摘要；处置应结合完整规程与当前诊断证据。</div><div class="detail-actions"></div>`;
    byId('kb-detail').querySelector('.detail-actions').append(button('向运维助手提问',()=>{if(!byId('chat-win').classList.contains('open'))toggleChat();byId('chat-text').value='请结合知识库解释：'+chosen.chapter;byId('chat-text').focus();},'btn primary'),button('复制摘要',async()=>{try{await navigator.clipboard.writeText(chosen.title+'\n'+chosen.chapter+'\n'+chosen.text);toast('摘要已复制');}catch{toast('浏览器未允许复制，请手动选取摘要',true);}}));
  }
  function filteredOrders() {
    const term=byId('wo-search').value.trim().toLowerCase(),status=byId('wo-status').value,risk=byId('wo-risk').value;
    return state.orders.filter(w=>(!term||[w.workorder_id,w.train_id,w.component].join(' ').toLowerCase().includes(term))&&(!status||w.status===status)&&(!risk||w.risk?.level===risk));
  }
  function renderOrders() {
    const list=filteredOrders(), pageCount=Math.max(1,Math.ceil(list.length/10));state.page=Math.min(state.page,pageCount);
    const visible=list.slice((state.page-1)*10,state.page*10);
    const chosen=visible.find(w=>w.workorder_id===state.orderId)||visible[0];state.orderId=chosen?.workorder_id||'';
    byId('wo-summary').innerHTML=Object.entries(statusNames).map(([status,label])=>metric(state.orders.filter(w=>w.status===status).length,label)).join('');
    byId('wo-body').innerHTML=visible.length?visible.map((w,i)=>`<tr><td><button class="btn mini" data-order="${i}">${esc(w.workorder_id)}</button></td><td><span class="mode mode-${w.status==='OPEN'?'PRIMARY':w.status==='REJECTED'?'DISABLED':'AUXILIARY'}">${esc(statusNames[w.status]||w.status)}</span></td><td>${esc(w.train_id||'—')}</td><td>${esc(w.component||'—')}</td><td><span class="lv lv-${esc(w.risk?.level||'UNKNOWN')}">${esc(riskCn(w.risk?.level))}</span></td><td>${esc(w.created_at?new Date(w.created_at*1000).toLocaleString('zh-CN'):'—')}</td><td><button class="btn mini" data-order="${i}">查看</button></td></tr>`).join(''):'<tr><td colspan="7">暂无符合条件的工单</td></tr>';
    byId('wo-body').querySelectorAll('[data-order]').forEach(el=>el.onclick=()=>{state.orderId=visible[Number(el.dataset.order)].workorder_id;renderOrders();});
    const pagination=byId('wo-pagination');pagination.innerHTML=`<span>共 ${list.length} 条 · ${state.page} / ${pageCount} 页</span>`;
    const prev=button('上一页',()=>{state.page--;renderOrders();}), next=button('下一页',()=>{state.page++;renderOrders();});prev.disabled=state.page<=1;next.disabled=state.page>=pageCount;pagination.append(prev,next);
    if(!chosen){byId('wo-detail').innerHTML=empty('暂无工单可供处置');return;}
    byId('wo-detail').innerHTML=`<h4 style="color:var(--cyan);overflow-wrap:anywhere">${esc(chosen.workorder_id)}</h4>${definition([['当前状态',statusNames[chosen.status]||chosen.status],['车辆',chosen.train_id],['部件',chosen.component],['风险等级',riskCn(chosen.risk?.level)],['风险评分',chosen.risk?.score],['创建时间',chosen.created_at?new Date(chosen.created_at*1000).toLocaleString('zh-CN'):'—']])}<h4>处置流程</h4><p>风险发现 → 工单草稿 → 人工确认 → 维修处置 → 闭环归档</p><div class="detail-actions"></div>`;
    const actions=byId('wo-detail').querySelector('.detail-actions');
    const act=(name,action)=>button(name,()=>reviewOrder(chosen,action),'btn '+(action==='reject'?'danger':'primary'));
    if(chosen.status==='DRAFT')actions.append(act('确认工单','confirm'),act('驳回','reject'));
    if(chosen.status==='OPEN')actions.append(act('关闭并归档','close'));
    actions.append(button('查看完整记录',()=>showDialog('工单记录',`<pre>${esc(JSON.stringify(chosen,null,2))}</pre>`)));
  }
  function reviewOrder(order, action) {
    const label={confirm:'确认工单',reject:'驳回工单',close:'关闭工单'}[action];
    showDialog(label,`<p>${esc(order.workorder_id)}</p><label>操作人<input id="review-operator" value="值班员"></label><label>处置备注<input id="review-note" placeholder="输入核查或处置说明"></label><div id="review-error" class="ui-error" role="alert"></div>`);
    const submit=button(label,async()=>{submit.disabled=true;try{const operator=byId('review-operator').value.trim();if(!operator)throw new Error('请填写操作人');await api('/api/v1/workorders/'+encodeURIComponent(order.workorder_id)+'/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operator,note:byId('review-note').value})});dialog.close();toast(label+'成功');refreshAll();}catch(e){byId('review-error').textContent=e.message;}finally{submit.disabled=false;}},'btn primary');dialog.append(submit);
  }
  function exportOrders() {
    const cell=v=>'"'+String(v??'').replace(/^[=+@-]/,"'").replace(/"/g,'""')+'"';
    const rows=[['工单号','状态','车辆','部件','风险'],...filteredOrders().map(w=>[w.workorder_id,statusNames[w.status],w.train_id,w.component,riskCn(w.risk?.level)])];
    const blob=new Blob(['\ufeff'+rows.map(row=>row.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='RailMind-工单.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  window.RailUI = {
    cabin(events) {
      const latest=events[0],ws=latest?.diagnosis?.windows||[];
      const conf=ws.map(w=>w.confidence).filter(v=>typeof v==='number');
      byId('cabin-summary').innerHTML='<div class="summary-strip">'+metric(events.length,'近期巡检记录')+metric(events.filter(e=>e.diagnosis?.severity==='NORMAL').length,'正常记录')+metric(events.filter(e=>['HIGH','WARNING'].includes(e.diagnosis?.severity)).length,'需关注记录')+metric(conf.length?Math.round(Math.min(...conf)*100)+'%':'—','最近窗口最低置信度')+'</div><h4>最近一次场景判定</h4><p class="conclusion">'+esc(ws.map(w=>w.scene_cn||w.scene).join(' → ')||'等待分析结果')+'</p><div class="ui-note">'+esc(latest?.diagnosis?.consistency?.note||'尚无窗口一致性结论')+'</div>';
    },
    capabilities(list) { state.caps=list;renderCapsView(); },
    knowledge(list) { state.kb=list;renderKnowledge(); },
    workorders(list) { state.orders=list;renderOrders(); },
    station(events) {
      const latest=events[0];
      if(!latest){byId('station-result').innerHTML=empty('暂无快检结论，等待检测记录');return;}
      const d=latest.diagnosis||{};
      byId('station-result').innerHTML=`<span class="lv lv-${esc(d.severity||'UNKNOWN')}">${esc(riskCn(d.severity))}</span><strong class="risk-value">${esc(anomalyCn(d.anomaly_type))}</strong><p>${esc(latest.asset?.train_id||'—')} · ${esc(latest.asset?.component||'—')}</p>${definition([['检测时间',fmtT(latest.ts)],['把手角度',(d.handles||[]).map(h=>h.angle_deg+'°').join(' / ')||'未提供'],['处置提示',d.severity==='NORMAL'?'检测未见异常':d.severity==='UNKNOWN'?'信息不足，需要人工复核':'请结合证据安排人工核查']])}`;
    }
  };
  setInterval(()=>document.querySelectorAll('[data-mirror]').forEach(el=>{el.textContent=byId(el.dataset.mirror)?.textContent||'—';}),1000);
  // Render hooks above will receive the first in-flight API refresh.
})();
