'use strict';
const $ = id => document.getElementById(id);
let playing=false, busy=false, latest=null, generation=0;
const finite = (value, digits=1) => Number.isFinite(value) ? value.toFixed(digits) : '—';
async function api(path, data) {
  const response = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Request failed');
  return value;
}
function buttons() {
  $('play').textContent=playing?'Ⅱ Pause':'▶ Run episode';
  $('play').disabled=(!playing && busy) || Boolean(latest?.done);
  $('step').disabled=busy || playing || Boolean(latest?.done);
  $('reset').disabled=busy;
  $('scenario').disabled=busy;
  $('control').disabled=busy;
}
function show(value) {
  latest=value;
  $('camera').src=value.image;
  $('clock').textContent=`${value.after.time.toFixed(1)} / 12.0 s`;
  $('speed').innerHTML=`${finite(value.after.ego_speed*3.6)} <small>km/h</small>`;
  const target=value.objects.find(obj=>obj.track===value.decision.target_id);
  $('depth').innerHTML=`${finite(target?.filtered_depth_m)} <small>m</small>`;
  $('ttc').innerHTML=`${finite(value.decision.ttc)} <small>s</small>`;
  const trueDepth=value.capture.obstacles.length?Math.min(...value.capture.obstacles.map(obj=>obj.depth)):null;
  $('truth').innerHTML=`${finite(trueDepth)} <small>m</small>`;
  $('accel').innerHTML=`${finite(value.decision.acceleration)} <small>m/s²</small>`;
  $('brake-bar').style.width=`${Math.max(0,-value.decision.acceleration/6)*100}%`;
  $('decision').textContent=value.after.collision?'Collision recorded':value.decision.mode;
  $('reason').textContent=value.after.collision?'The episode retains this failure. Reset or compare with another scenario.':value.decision.reason;
  $('timing').textContent=`${finite(value.model_ms,0)} ms / inference`;
  $('camera-status').textContent=value.after.collision?'COLLISION · reset to replay':value.done?'EPISODE COMPLETE':`CAPTURE t = ${value.capture.time.toFixed(1)} s · ${value.control?'CONTROL ON':'CONTROL OFF'}`;
  document.body.classList.toggle('collision',value.after.collision);
  document.body.classList.toggle('braking',value.decision.mode==='brake');
  $('count').textContent=`${value.objects.length} objects`;
  $('objects').replaceChildren();
  for (const obj of value.objects) {
    const row=document.createElement('tr');
    if (obj.track===value.decision.target_id) row.className='selected';
    for (const text of [`${obj.class} #${obj.track}`,finite(obj.score,2),`${finite(obj.filtered_depth_m)} m`,finite(obj.ttc_s)]) {
      const cell=document.createElement('td');cell.textContent=text;row.append(cell);
    }
    $('objects').append(row);
  }
  if (!value.objects.length) $('objects').innerHTML='<tr><td colspan="4">No kept detections · no substitute truth inputs</td></tr>';
  world(value.after); trace(value.history);
  if(value.done) playing=false;
}
function canvas(id) {
  const c=$(id), ctx=c.getContext('2d'), ratio=window.devicePixelRatio||1;
  const w=c.clientWidth,h=c.clientHeight;
  c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);
  return {ctx,w,h};
}
function world(state) {
  const {ctx,w,h}=canvas('world');
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#f5f8f1';ctx.fillRect(0,0,w,h);
  const px=22,scale=(h-42)/55,mid=w/2, y=z=>h-32-z*scale;
  ctx.fillStyle='#e6ece2';ctx.fillRect(mid-3.6*px,0,7.2*px,h);
  ctx.strokeStyle='#fdfefa';ctx.lineWidth=2;ctx.setLineDash([12,10]);
  for(const x of [-1.8,1.8]){ctx.beginPath();ctx.moveTo(mid+x*px,0);ctx.lineTo(mid+x*px,h);ctx.stroke();}ctx.setLineDash([]);
  ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillStyle='#819286';
  for(const z of [10,30,50]){ctx.fillText(`${z} m`,12,y(z));}
  ctx.fillStyle=state.collision?'#b4593a':'#246c56';ctx.fillRect(mid-1.8*px/2,y(0)-4.3*scale/2,1.8*px,4.3*scale);
  ctx.fillStyle='#fff';ctx.font='9px system-ui';ctx.textAlign='center';ctx.fillText('EGO',mid,y(0)+3);
  for(const obj of state.obstacles){const x=mid+obj.x*px,yy=y(obj.depth);if(yy<0||yy>h)continue;ctx.fillStyle='#db9a5e';ctx.fillRect(x-obj.width*px/2,yy-obj.length*scale/2,obj.width*px,obj.length*scale);ctx.fillStyle='#735133';ctx.font='10px system-ui';ctx.fillText(`${obj.depth.toFixed(1)} m`,x,yy-12);}
}
function trace(rows) {
  const {ctx,w,h}=canvas('trace'),left=44,right=16,top=18,bottom=30;
  ctx.clearRect(0,0,w,h);ctx.font='11px system-ui';ctx.textAlign='right';ctx.fillStyle='#87978c';
  const xx=t=>left+t/12*(w-left-right), yy=z=>h-bottom-z/65*(h-top-bottom);
  for(const depth of [0,20,40,60]){ctx.strokeStyle='#e8eee7';ctx.beginPath();ctx.moveTo(left,yy(depth));ctx.lineTo(w-right,yy(depth));ctx.stroke();ctx.fillText(`${depth} m`,left-8,yy(depth)+4);}
  ctx.textAlign='center';for(const t of [0,3,6,9,12])ctx.fillText(`${t} s`,xx(t),h-10);
  ctx.save();ctx.beginPath();ctx.rect(left,top,w-left-right,h-top-bottom);ctx.clip();
  for (const [key,color] of [['true_depth','#667d70'],['predicted_depth','#dc863c']]){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let connected=false;for(const row of rows){if(!Number.isFinite(row[key])){connected=false;continue;}if(connected)ctx.lineTo(xx(row.time),yy(row[key]));else ctx.moveTo(xx(row.time),yy(row[key]));connected=true;}ctx.stroke();}ctx.restore();
}
function fail(error){playing=false;$('error').textContent=error.message;$('error').hidden=false;}
async function reset(){if(busy)return;playing=false;generation++;busy=true;buttons();$('error').hidden=true;try{show(await api('/api/reset',{scenario:$('scenario').value,control:$('control').checked}));}catch(error){fail(error);}finally{busy=false;buttons();}}
async function step(){if(busy||latest?.done)return;busy=true;buttons();try{show(await api('/api/step',{}));}catch(error){fail(error);}finally{busy=false;buttons();}}
async function loop(token){if(!playing||generation!==token)return;const started=performance.now();await step();if(playing&&generation===token)setTimeout(()=>loop(token),Math.max(0,100-(performance.now()-started)));}
$('play').onclick=()=>{playing=!playing;buttons();if(playing)loop(++generation);};
$('step').onclick=step;$('reset').onclick=reset;$('scenario').onchange=reset;$('control').onchange=reset;
reset();

window.addEventListener('resize',()=>{if(latest){world(latest.after);trace(latest.history);}});
