/* Reusable schematic replay. Observations describe the state BEFORE each action.
   The final success placement is inferred from the recorded terminal reward;
   no terminal observation was retained in the sample. */
(() => {
'use strict';
const DATA = window.ALFWORLD_TRAJECTORIES;
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const candle = '<rect x="-8" y="-17" width="16" height="29" rx="3" fill="#f9c967" stroke="#c38830" stroke-width="2"/><ellipse cx="0" cy="-17" rx="8" ry="3" fill="#fff0c9"/><path d="M0-17v-6" stroke="#876133" stroke-width="2"/>';
const paper = '<rect x="-10" y="-12" width="20" height="24" rx="3" fill="white" stroke="#9daeb6" stroke-width="2"/><ellipse cx="0" cy="-12" rx="10" ry="4" fill="#e7edef" stroke="#9daeb6"/><ellipse cx="0" cy="-12" rx="3" ry="2" fill="#a4b5bd"/><path d="M10-8h7v24H8" fill="#fff" stroke="#9daeb6" stroke-width="2"/>';
const OBS_SUCCESS = [
 '你在房间中央。周围有台面、浴缸、马桶、抽屉等；还不知道蜡烛在哪里。',
 '环顾房间，没有看到更多物品。',
 '台面 1 上有肥皂、喷壶和纸卷，没有蜡烛。',
 '你正面对台面 1，旁边没有看到其他物品。',
 '浴缸区域有浴缸和肥皂，没有蜡烛。',
 '你正面对浴缸区域，旁边没有看到其他物品。',
 '马桶 1 上有蜡烛 2 和一瓶皂液。找到蜡烛了！',
 '你从马桶 1 上拿起了蜡烛 2。',
 '回到台面 1，台面上仍有肥皂、喷壶和纸卷。'
];
const OBS_FAIL = [
 '你在房间中央。周围有柜子、台面、马桶、水槽等；还不知道蜡烛在哪里。',
 '环顾房间，没有看到更多物品。',
 '到达柜子 1，柜门关着。',
 '柜门打开了，里面有纸卷 1。',
 '没有发生变化。环境提示检查动作是否有效、格式是否正确。',
 '你从柜子 1 里拿起了纸卷 1。',
 '到达台面 1，台面上没有物品。',
 '没有发生变化。放置指令没有执行成功，纸卷仍在手里。',
 '没有发生变化。从台面拿纸卷的指令没有执行成功。'
];
const ACTIONS = {
 'look':'看一看当前位置', 'go to countertop 1':'走到台面 1',
 'go to bathtubbasin 1':'走到浴缸区域', 'go to toilet 1':'走到马桶 1',
 'take candle 2 from toilet 1':'从马桶 1 拿起蜡烛 2',
 'move candle 2 to countertop 1':'把蜡烛 2 放到台面 1',
 'go to cabinet 1':'走到柜子 1', 'open cabinet 1':'打开柜子 1',
 'look in cabinet 1':'尝试查看柜子内部',
 'take toiletpaper 1 from cabinet 1':'从柜子 1 拿起纸卷 1',
 'put toiletpaper 1 on countertop 1':'尝试把纸卷放到台面上',
 'take toiletpaper 1 from countertop 1':'尝试从台面拿起纸卷',
 'look around':'反复尝试环顾四周'
};
function observation(kind, i) {
 return (kind === 'success' ? OBS_SUCCESS : OBS_FAIL)[i] || '没有发生变化。环境再次提示检查动作是否有效、格式是否正确。';
}
function sceneState(kind, count) {
 const s = {place:'center', held:null, candle:null, cabinetOpen:false, paper:null, known:new Set()};
 for (let i=0;i<count;i++) {
  const a=DATA[kind].steps[i].action;
  if(a.startsWith('go to ')) {
   s.place=a.slice(6); s.known.add(s.place);
   if(s.place==='toilet 1' && kind==='success') s.candle='toilet';
  }
  if(a==='open cabinet 1') {s.cabinetOpen=true;s.paper='cabinet';}
  if(a==='take toiletpaper 1 from cabinet 1') {s.paper=null;s.held='paper';}
  if(a==='take candle 2 from toilet 1') {s.candle='held';s.held='candle';}
  if(a==='move candle 2 to countertop 1' && DATA[kind].success) {s.candle='countertop';s.held=null;}
  // Invalid actions have no effect, including failed put/take/look around.
 }
 return s;
}
const places={'center':[285,265],'countertop 1':[270,180],'bathtubbasin 1':[425,265],'toilet 1':[398,363],'cabinet 1':[147,220]};
function sceneMarkup(kind, uid) {
 const success=kind==='success';
 return `<svg class="alf-scene" viewBox="0 0 560 460" role="img" aria-labelledby="${uid}-scene-title"><title id="${uid}-scene-title">二维房间示意；物品随观察发现，小机器人按记录执行动作</title>
 <defs><pattern id="${uid}-tile" width="44" height="44" patternUnits="userSpaceOnUse"><path d="M44 0H0V44" fill="none" stroke="#e1e9e7" stroke-width="1"/></pattern><filter id="${uid}-shadow" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#547473" flood-opacity=".16"/></filter></defs>
 <rect x="16" y="22" width="528" height="418" rx="22" fill="#e3eeea"/><rect x="25" y="37" width="510" height="394" rx="14" fill="#f8fbf8"/><rect x="25" y="37" width="510" height="394" rx="14" fill="url(#${uid}-tile)"/>
 <g data-place="countertop 1"><rect class="alf-furniture" x="206" y="58" width="157" height="88" rx="9"/><rect x="212" y="64" width="145" height="48" rx="6" fill="#dbe8e4"/><path d="M216 122h137M285 116v25" stroke="#b0c5c3"/><text x="284" y="48" text-anchor="middle">台面 1 · 目标</text><g data-detail="countertop" visibility="hidden"><rect x="226" y="86" width="22" height="12" rx="5" fill="#d2b6cf"/><rect x="259" y="77" width="12" height="24" rx="3" fill="#a9cbd6"/><path d="M262 77v-6h12" fill="none" stroke="#7eabbc"/><g transform="translate(336 89) scale(.62)">${paper}</g></g><text class="alf-small" data-unknown="countertop" x="284" y="94" text-anchor="middle">物品待观察</text></g>
 ${success ? `<g data-place="bathtubbasin 1"><rect class="alf-furniture" x="438" y="179" width="77" height="122" rx="30"/><rect x="448" y="191" width="57" height="91" rx="23" fill="#dceef0"/><path d="M472 188v-14h12" stroke="#8eaeb4" stroke-width="5" fill="none"/><text x="476" y="324" text-anchor="middle">浴缸</text><g data-detail="bathtub" visibility="hidden"><rect x="449" y="260" width="25" height="11" rx="5" fill="#d2b6cf"/></g></g><rect class="alf-furniture" x="48" y="96" width="105" height="73" rx="7"/><path d="M53 120h95M53 144h95M93 109h17M93 134h17M93 158h17" stroke="#b0c5c3" stroke-width="3"/><text x="100" y="190" text-anchor="middle">抽屉</text>` : `<g data-place="cabinet 1"><rect class="alf-furniture" x="49" y="94" width="111" height="94" rx="7"/><rect x="57" y="102" width="95" height="78" rx="4" fill="#d6c4a9"/><path d="M57 143h95" stroke="#b29a79" stroke-width="4"/><g data-item="cabinet-paper" transform="translate(103 134)" visibility="hidden">${paper}</g><g class="alf-door"><rect x="53" y="98" width="103" height="85" rx="4" fill="#e8d9c1" stroke="#bda989" stroke-width="2"/><circle cx="140" cy="141" r="4" fill="#a38b66"/></g><text x="104" y="215" text-anchor="middle">柜子 1</text></g>`}
 <g data-place="toilet 1"><rect class="alf-furniture" x="429" y="347" width="78" height="27" rx="7"/><ellipse class="alf-furniture" cx="468" cy="391" rx="32" ry="29"/><ellipse cx="468" cy="391" rx="21" ry="19" fill="#dceef0"/><text x="468" y="451" text-anchor="middle">马桶 1</text><text class="alf-small" data-unknown="toilet" x="460" y="341" text-anchor="middle">物品待观察</text><g data-detail="toilet" visibility="hidden"><rect x="488" y="341" width="12" height="20" rx="4" fill="#b5cfb9"/></g></g>
 <rect class="alf-furniture" x="54" y="320" width="107" height="63" rx="12"/><ellipse cx="108" cy="352" rx="37" ry="19" fill="#dceef0"/><path d="M108 333v-15h13" stroke="#8eaeb4" stroke-width="5" fill="none"/><text x="108" y="408" text-anchor="middle">水槽</text>
 <path data-trail d="M285 265" fill="none" stroke="#94b8ac" stroke-width="3" stroke-dasharray="5 7" stroke-linecap="round"/>
 <g class="alf-robot-mover" style="transform:translate(285px,265px)"><ellipse cy="35" rx="31" ry="9" fill="#264e5020"/><g filter="url(#${uid}-shadow)"><rect x="-19" y="5" width="38" height="28" rx="12" fill="#f7fafb" stroke="#799aa4" stroke-width="2"/><rect x="-19" y="27" width="10" height="11" rx="4" fill="#426877"/><rect x="9" y="27" width="10" height="11" rx="4" fill="#426877"/><path d="M-20 12l-9 11M20 12l9 11" stroke="#7596a0" stroke-width="8" stroke-linecap="round"/><rect x="-27" y="-31" width="54" height="41" rx="15" fill="#fff" stroke="#799aa4" stroke-width="2"/><rect x="-20" y="-23" width="40" height="25" rx="9" fill="#203f4d"/><circle cx="-10" cy="-11" r="4" fill="#88e1cf"/><circle cx="10" cy="-11" r="4" fill="#88e1cf"/><path d="M0-31v-9" stroke="#7596a0" stroke-width="3"/><circle cy="-43" r="5" fill="#efb44e"/><rect x="-7" y="15" width="14" height="7" rx="3" fill="#70c3b4"/></g><g data-held transform="translate(35 18) scale(.85)"></g></g>
 <g class="alf-item" data-item="candle" visibility="hidden">${candle}</g>
 <g data-invalid visibility="hidden"><rect x="170" y="390" width="205" height="31" rx="15" fill="#fff0eb" stroke="#e6ab95"/><text x="272" y="411" text-anchor="middle" style="fill:#aa4c31;font-size:14px">动作无效 · 环境没有变化</text></g>
 </svg>`;
}
let serial=0;
function mount(root) {
 const uid=`alf-${++serial}`;
 const initial=root.dataset.mode || 'intro';
 let mode=initial, kind=mode==='fail'?'fail':'success', count=mode==='step'?6:0, timer=null;
 let running=false, autoEnabled=true, sceneVisible=false, playbackEpoch=0;
 root.innerHTML=`<div class="alf-top"><div><div class="alf-kicker">ALFWORLD · 跟着小机器人做任务</div><div class="alf-title">把一支蜡烛放到台面上</div></div><div class="alf-tabs" role="group" aria-label="选择演示"><button type="button" data-mode="intro">认识环境</button><button type="button" data-mode="success">成功轨迹</button><button type="button" data-mode="fail">失败轨迹</button></div></div><div class="alf-goal"><svg viewBox="-14 -28 28 45" aria-hidden="true">${candle}</svg><span><b>你的任务：</b>找到蜡烛 → 拿起来 → 放到台面 1</span></div><div class="alf-main"><div class="alf-map-panel"><div class="alf-map-title"><span data-room></span><span>二维示意 · 非原始房间布局</span></div><div class="alf-stage"><div data-scene></div><button type="button" class="alf-start" data-start>▶ 点击播放，看小机器人怎么做</button></div><div class="alf-pocket"><span>机器人手里：<strong data-pocket>空手</strong></span><span data-location>房间中央</span></div><div class="alf-legend"><span>↻ 自动循环 · 可暂停</span><span><i class="alf-dot"></i>当前位置</span><span><i class="alf-dot unknown"></i>物品随观察发现</span></div><p class="alf-map-caption">小机器人代表模型。它实际收到文字观察；房间仅画出与讲解有关的设施。</p></div><div class="alf-info"><div class="alf-card"><div class="alf-card-label"><span class="alf-tag">state</span> ① 机器人看到什么 · 执行前</div><p data-observation></p><details class="alf-raw"><summary>查看原始文字观察</summary><pre data-raw-obs></pre><p class="alf-state-note">这里展示当前观察；模型做决策时的输入还包含任务和此前的交互历史。</p></details></div><div class="alf-card alf-action"><div class="alf-card-label"><span class="alf-tag">action</span> ② 机器人做什么 · 动作</div><p data-action></p><details class="alf-raw"><summary>查看原始动作指令</summary><pre data-raw-action></pre></details></div><div class="alf-card alf-feedback"><div class="alf-card-label">③ 环境发生了什么 · 执行后</div><p data-feedback></p><details class="alf-raw"><summary>查看原始反馈 / 结果依据</summary><pre data-raw-feedback></pre></details></div><p class="alf-note" data-note></p><div class="alf-reward"><div class="alf-card-label"><span class="alf-tag">reward</span> 任务的终止奖励</div><div class="alf-status" role="status" aria-live="polite" data-status></div></div></div></div><div class="alf-controls"><button class="alf-play" type="button" data-play>▶ 播放动画</button><button type="button" data-prev aria-label="上一步">← 上一步</button><button type="button" data-next aria-label="下一步">下一步 →</button><button type="button" data-reset>重播</button><div class="alf-progress"><label for="${uid}-seek" class="alf-counter" data-counter></label><input id="${uid}-seek" type="range" min="0" value="0" aria-label="已执行动作数"></div><label>速度<select data-speed aria-label="播放速度"><option value="2200">正常</option><option value="1500">快速</option><option value="4000">慢速</option></select></label></div><div class="alf-foot">依据预置真实轨迹重放，中文为释义。成功、失败的任务指令相同，但来自不同房间；示意位置不代表实际距离。<span data-source></span></div>`;
 root.insertBefore(root.querySelector('.alf-controls'), root.querySelector('.alf-main'));
 const $=s=>root.querySelector(s);
 const put=(s,t)=>{$(s).textContent=t;};
 function stop(manual=false){
  if(timer)clearTimeout(timer);timer=null;running=false;playbackEpoch++;
  if(manual)autoEnabled=false;
  root.dataset.playing='false';put('[data-play]','▶ 播放动画');
  $('[data-start]').hidden=count!==0;
 }
 function newScene(){$('[data-scene]').innerHTML=sceneMarkup(kind,uid);}
 function render(){
  const data=DATA[kind], n=data.steps.length, s=sceneState(kind,count), step=count?data.steps[count-1]:null;
  root.dataset.completed=String(count);root.dataset.trajectory=kind;
  root.querySelectorAll('.alf-tabs button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode || (mode==='step'&&b.dataset.mode==='success'))));
  put('[data-room]',kind==='success'?'成功记录中的房间':'失败记录中的房间');
  const before=Math.max(0,count-1);
  put('[data-observation]',observation(kind,before));put('[data-raw-obs]',data.steps[before].observation);
  put('[data-action]',step?ACTIONS[step.action]||step.action:'尚未行动，先认识环境和任务目标。');
  put('[data-raw-action]',step?step.action:'尚未执行动作');
  let feedback=count===0?'蜡烛的位置还不知道。动画将逐步展示小机器人的寻找过程。':count<n?observation(kind,count):data.success?'蜡烛已经放到台面，任务完成！':'30 步耗尽，蜡烛没有放到台面，任务失败。';
  const nextObs=count<n?data.steps[count].observation:null;
  const invalid=!!(step && (nextObs?.startsWith('Nothing happens.') || (kind==='fail'&&count===n)));
  if(kind==='fail'&&count>=9&&count<n) feedback=`第 ${count-8} 次重复环顾指令，环境仍没有变化。纸卷还在手里。`;
  put('[data-feedback]',feedback);
  put('[data-raw-feedback]',count===0?'尚未执行动作':nextObs || `样本未保存最后一个动作后的观察。记录的终止 reward = ${data.reward}，success = ${data.success}。${data.success?'图中最终放置依据该动作及终止成功结果示意。':'未凭空补写环境原文。'}`);
  $('.alf-feedback').dataset.error=String(invalid);$('.alf-feedback').dataset.success=String(count===n&&data.success);
  put('[data-note]',mode==='step'?'从找到蜡烛的时刻开始，观察拿起动作与反馈；可暂停后逐步查看。':kind==='fail'&&count>=5?'纸卷并不是任务要求的蜡烛；后续无效放置不会让它离开机器人的手。':count===0?'先认识家具，再通过观察发现物品。机器人不会提前知道蜡烛的位置。':'画面显示执行后的状态；上方三格保留本步执行前的观察、动作和执行后的反馈。');
  put('[data-status]',count===n?data.success?'任务成功 · 9 步完成 · 终止奖励 +1.0':'任务失败 · 30 步耗尽 · 终止奖励 −0.1':`已执行 ${count} 步 / 最多 30 步 · 任务尚未结束`);
  $('[data-status]').className='alf-status '+(count===n?(data.success?'alf-result':'alf-fail'):'');
  put('[data-counter]',`${count} / ${n} 步`);$('input').max=n;$('input').value=count;
  $('[data-start]').hidden=count!==0||running;
  $('[data-prev]').disabled=count===0;$('[data-next]').disabled=count===n;
  put('[data-pocket]',s.held==='candle'?'蜡烛 2':s.held==='paper'?'纸卷 1（不是目标）':'空手');
  put('[data-location]',({'center':'房间中央','countertop 1':'台面前','bathtubbasin 1':'浴缸前','toilet 1':'马桶前','cabinet 1':'柜子前'})[s.place]);
  const xy=places[s.place];$('.alf-robot-mover').style.transform=`translate(${xy[0]}px,${xy[1]}px)`;
  // One persistent candle moves through world coordinates, including the hand.
  $('[data-held]').innerHTML=s.held==='paper'?paper:'';
  root.querySelectorAll('[data-place]').forEach(g=>g.querySelectorAll('.alf-furniture').forEach(r=>r.classList.toggle('alf-active',g.dataset.place===s.place)));
  const show=(selector,visible)=>{const el=$(selector);if(el)el.setAttribute('visibility',visible?'visible':'hidden');};
  show('[data-detail="countertop"]',kind==='success'&&s.known.has('countertop 1'));
  show('[data-unknown="countertop"]',!s.known.has('countertop 1'));
  show('[data-unknown="toilet"]',!s.known.has('toilet 1'));
  show('[data-detail="toilet"]',kind==='success'&&s.known.has('toilet 1'));
  show('[data-detail="bathtub"]',s.known.has('bathtubbasin 1'));
  show('[data-item="cabinet-paper"]',s.paper==='cabinet');
  if($('.alf-door'))$('.alf-door').classList.toggle('open',s.cabinetOpen);
  const ci=$('[data-item="candle"]');show('[data-item="candle"]',s.candle!==null);
  const candleXY=s.candle==='held'?[xy[0]+35,xy[1]+18]:s.candle==='countertop'?[300,89]:[455,351];
  ci.style.transform=`translate(${candleXY[0]}px,${candleXY[1]}px)`;
  show('[data-invalid]',invalid);
  let path='M285 265';for(let i=0;i<count;i++){const a=data.steps[i].action;if(a.startsWith('go to ')){const p=places[a.slice(6)];path+=` L${p[0]} ${p[1]}`;}}$('[data-trail]').setAttribute('d',path);
  put('[data-source]',`记录：batch ${data.eid.batch} / task ${data.eid.task} / run ${data.eid.run}。`);
 }
 function advance(){if(count<DATA[kind].steps.length)count++;render();}
 function schedule(){
  const epoch=playbackEpoch;
  // Flush the visual update, then await its actual transitions. Reading time
  // starts AFTER movement; stale callbacks cannot advance a paused/new replay.
  requestAnimationFrame(async()=>{
   if(!running||epoch!==playbackEpoch)return;
   const stage=$('.alf-stage');
   const motions=stage.getAnimations?stage.getAnimations({subtree:true}):[];
   await Promise.all(motions.map(a=>a.finished.catch(()=>{})));
   if(!running||epoch!==playbackEpoch)return;
   const dwell=count===DATA[kind].steps.length?5000:Math.max(1500,Number($('[data-speed]').value));
   timer=setTimeout(()=>{
    timer=null;if(!running||epoch!==playbackEpoch)return;
    if(count===DATA[kind].steps.length){count=mode==='step'?6:0;newScene();render();}
    else advance();
    schedule();
   },dwell);
  });
 }
 function start(){
  if(running||document.hidden)return;
  running=true;root.dataset.playing='true';put('[data-play]','Ⅱ 暂停动画');
  $('[data-start]').hidden=true;schedule();
 }
 function play(){if(running){stop(true);return;}autoEnabled=true;start();}
 function resumeIfVisible(){if(autoEnabled&&sceneVisible&&!document.hidden)start();}
 root.querySelectorAll('.alf-tabs button').forEach(b=>b.addEventListener('click',()=>{
  stop();mode=b.dataset.mode;kind=mode==='fail'?'fail':'success';count=0;
  newScene();render();resumeIfVisible();
 }));
 $('[data-play]').addEventListener('click',play);
 $('[data-start]').addEventListener('click',play);
 $('[data-next]').addEventListener('click',()=>{stop(true);advance();});
 $('[data-prev]').addEventListener('click',()=>{stop(true);count=Math.max(0,count-1);render();});
 $('[data-reset]').addEventListener('click',()=>{stop();count=mode==='step'?6:0;autoEnabled=true;newScene();render();start();});
 $('input').addEventListener('input',e=>{stop(true);count=Number(e.target.value);render();});
 $('[data-speed]').addEventListener('change',()=>{if(running){stop();start();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();else resumeIfVisible();});
 if('IntersectionObserver' in window)new IntersectionObserver(entries=>{
  sceneVisible=entries[0].isIntersecting&&entries[0].intersectionRatio>=.25;
  if(sceneVisible)resumeIfVisible();else stop();
 },{threshold:[0,.25]}).observe($('.alf-stage'));
 else {sceneVisible=true;resumeIfVisible();}

 newScene();render();
}
document.querySelectorAll('.alf-player').forEach(mount);
})();
