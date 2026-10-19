'use strict';
const $=id=>document.getElementById(id);
const number=(value,digits=1)=>Number.isFinite(value)?value.toFixed(digits):'—';
let playing=false,busy=false,latest=null,epoch=0,activeMs=0,activeStart=0,shown=0;
async function request(path){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const data=await r.json();if(!r.ok)throw new Error(data.error||'Video request failed');return data;}
function buttons(){$('play').textContent=playing?'Ⅱ Pause':'▶ Run video';$('play').disabled=(!playing&&busy)||Boolean(latest?.done);$('step').disabled=busy||playing||Boolean(latest?.done);$('reset').disabled=busy;$('paced').disabled=playing;}
function elapsed(){return activeMs+(playing?performance.now()-activeStart:0);}
function pause(){if(playing){activeMs+=performance.now()-activeStart;playing=false;}epoch++;buttons();}
function rate(){if(!latest||shown<1)return;const fps=shown/(elapsed()/1000);$('display-fps').textContent=number(fps);const source=latest.info.native_fps;const meets=fps>=source;$('pace-status').textContent=meets?'Keeping pace with the video':'Playback is slower than the source';$('pace-detail').textContent=`${shown} frames displayed in ${(elapsed()/1000).toFixed(2)} active seconds · source ${source} FPS · no dropped frames.`;}
async function show(value,roundtripStart,count){latest=value;$('camera').src=value.image;await $('camera').decode();
 $('source').textContent=`${value.info.name} · ${value.info.width} × ${value.info.height}`;$('native').textContent=`${value.info.native_fps} FPS source`;$('device').textContent=`DEVICE: ${value.info.device}`;
 $('progress').max=value.info.frames;$('progress').value=value.index+1;$('frame').textContent=`${value.preview?'Preview · frame':'Frame'} ${value.index+1} / ${value.info.frames}`;$('media-time').textContent=`${((value.index+1)/value.info.native_fps).toFixed(1)} / ${value.info.duration_s.toFixed(1)} s`;
 $('model-ms').innerHTML=`${number(value.model_ms)} <small>ms</small>`;$('processing-fps').innerHTML=`${number(value.mean_processing_fps)} <small>FPS</small>`;$('roundtrip').innerHTML=`${number(performance.now()-roundtripStart)} <small>ms</small>`;
 $('camera-status').textContent=`LIVE INFERENCE · t = ${value.media_time_s.toFixed(1)} s${value.done?' · END OF CLIP':''}`;$('count').textContent=`${value.objects.length} objects`;
 $('objects').replaceChildren();for(const obj of value.objects){const row=document.createElement('tr');for(const text of [`${obj.class} #${obj.track}`,number(obj.score,2),`${number(obj.filtered_depth_m)} m`,`${number(obj.range_m)} m`,`${number(obj.closing_speed_mps)} m/s`,obj.ttc_s===null?'—':`${number(obj.ttc_s)} s`]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}$('objects').append(row);}
 if(!value.objects.length)$('objects').innerHTML='<tr><td colspan="6">No kept detections</td></tr>';
 await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
 if(count){shown++;rate();}if(value.done){pause();rate();}
}
function fail(error){pause();$('error').textContent=error.message;$('error').hidden=false;}
async function reset(){if(busy)return;pause();busy=true;buttons();$('error').hidden=true;activeMs=0;shown=0;$('display-fps').textContent='—';$('pace-status').textContent='Ready';$('pace-detail').textContent='Run continuously to measure playback rate.';try{const start=performance.now();await show(await request('/api/video/reset'),start,false);}catch(error){fail(error);}finally{busy=false;buttons();}}
async function step(count=false){if(busy||latest?.done)return;busy=true;buttons();try{const start=performance.now();await show(await request('/api/video/step'),start,count);}catch(error){fail(error);}finally{busy=false;buttons();}}
async function loop(token){if(!playing||token!==epoch)return;const start=performance.now();await step(true);if(playing&&token===epoch)setTimeout(()=>loop(token),$('paced').checked?Math.max(0,1000/latest.info.native_fps-(performance.now()-start)):0);}
$('play').onclick=()=>{if(playing){pause();rate();}else{playing=true;activeStart=performance.now();buttons();loop(++epoch);}};$('reset').onclick=reset;$('step').onclick=()=>step(false);reset();

$('paced').onchange=reset;
