import {ConstructionPreview} from '/construction.js';

const $=id=>document.getElementById(id);
const element=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
let catalog,selected,active,preview,result,frame=0,filter='all',jobs=[],workers={},polling=false,creating=false,loadingResult=false,previewUnavailable=false;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,6500);}
async function api(path,body){
  let options={cache:'no-store'};
  if(body){const auth=await fetch('/api/catalog',{cache:'no-store'});if(!auth.ok)throw Error('The local server is unavailable');const token=(await auth.json()).csrf;options={method:'POST',headers:{'Content-Type':'application/json','X-IsaacMin-Token':token},body:JSON.stringify(body)};}
  const response=await fetch(path,options),data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data;
}
function setRoute(job){history.replaceState(null,'',location.pathname+(job?'?job='+job:''));}
function clearPreview(){preview?.dispose();preview=null;previewUnavailable=false;$('construction-canvas').replaceChildren();for(const name of ['surface','terrainVertices','assetPoints'])delete $('construction-canvas').dataset[name];$('construction-wait').hidden=false;}
function assistantState(){
  $('agent-toggle').disabled=!selected;$('assistant-world').textContent=selected?selected.title+(active?' · current conversion':' · Minecraft preset'):'';
  $('planner-status').textContent=catalog.planner.status==='available'?'Ready · OpenClaw + NVIDIA Nemotron':'Assistant unavailable · '+catalog.planner.reason;
  $('agent-send').disabled=!selected||catalog.planner.status!=='available';renderConversation();
}
function sourceCards(){
  $('source-presets').replaceChildren();
  for(const preset of catalog.presets){
    const button=element('button','source-card'+(selected?.id===preset.id?' selected':''));button.dataset.preset=preset.id;button.setAttribute('aria-pressed',String(selected?.id===preset.id));
    const image=element('img');if(preset.minecraft[0])image.src=preset.minecraft[0].url;image.alt=preset.title+' Minecraft source';
    const copy=element('div','source-card-copy');copy.append(element('h3','',preset.title),element('p','',preset.subtitle));button.append(image,copy,element('span','card-arrow','↗'));button.onclick=()=>choose(preset.id);$('source-presets').append(button);
  }
}
function choose(id){
  selected=catalog.presets.find(p=>p.id===id);active=null;result=null;clearPreview();setRoute(null);
  $('welcome').hidden=false;$('choose-world').hidden=false;$('building').hidden=true;$('ready').hidden=true;$('setup').hidden=false;
  $('setup-image').src=selected.minecraft[0]?.url||'';$('setup-title').textContent=selected.title;$('setup-description').textContent=selected.subtitle;
  sourceCards();updateCreate();assistantState();$('setup').scrollIntoView({behavior:'smooth',block:'nearest'});
}
function home(){selected=null;active=null;result=null;clearPreview();setRoute(null);for(const id of ['setup','building','ready','assistant'])$(id).hidden=true;$('welcome').hidden=false;$('choose-world').hidden=false;sourceCards();assistantState();scrollTo({top:0,behavior:'smooth'});}
function existingBuild(){return jobs.find(j=>j.preset===selected?.id&&j.action==='convert'&&['queued','running'].includes(j.status));}
function updateCreate(){$('create-world').textContent=existingBuild()?'Continue current build →':'Create Isaac world →';$('create-world').disabled=creating;}
async function openBuild(job){
  if(active?.id!==job.id){clearPreview();result=null;frame=0;filter='all';$('build-counts').replaceChildren();$('build-warning').hidden=true;}
  active=job;selected=catalog.presets.find(p=>p.id===job.preset);setRoute(job.id);
  for(const id of ['welcome','choose-world','setup','ready'])$(id).hidden=true;
  $('building').hidden=false;$('build-title').textContent=selected.title+' is coming together';assistantState();await pollBuild();
}
$('create-world').onclick=async()=>{
  if(creating)return;creating=true;updateCreate();
  try{const current=existingBuild();const job=current||await api('/api/jobs',{preset:selected.id,action:'convert',operation_key:crypto.randomUUID()});await openBuild(job);$('building').scrollIntoView({behavior:'smooth'});}
  catch(error){toast(error.message);}finally{creating=false;updateCreate();}
};
function elapsed(created){const s=Math.max(0,Math.floor((Date.now()-Date.parse(created))/1000));return s<60?s+'s elapsed':Math.floor(s/60)+'m '+s%60+'s elapsed';}
function displayProgress(data){
  const job=data.job;active=job;
  $('build-state').textContent=job.status==='running'?'BUILD IN PROGRESS':job.status.toUpperCase();
  const stage=data.stages.find(s=>s.status==='running')||data.stages.find(s=>s.status==='pending');
  $('build-phase').textContent=job.status==='queued'?'Your build is in the queue':job.status==='complete'?'Your world is ready':stage?.label||'Preparing your world';
  const captures=data.captures.reduce((sum,c)=>sum+c.frames,0);
  const agent=job.pipeline_agent;
  $('build-agent').textContent=agent?`${agent.harness==='openclaw'?'OpenClaw + ':''}NVIDIA Nemotron · ${agent.status==='requesting_stage'?'choosing the next permitted step':agent.status==='blocked_at_stage'?'dispatch stopped; progress retained':agent.status==='complete'?'workflow complete':'executing '+agent.stage_label} · ${agent.dispatches} validated tool calls`:'Agent will begin when the worker is ready.';
  $('build-detail').textContent=job.error||(job.status==='queued'?'Waiting for the local worker. Your request is saved.':captures?`${captures} real Isaac captures saved so far. ${job.progress||''}`:job.progress||'Construction progress is saved automatically.');
  if(job.status==='queued'&&workers.work==='stopped')$('build-detail').textContent='The local worker is stopped. Restart the studio to continue; this job is saved.';
  $('build-milestones').textContent=`${data.completed_stages} of ${data.stage_count} stages`;$('build-elapsed').textContent=elapsed(job.created);
  $('build-meter-fill').style.width=(data.completed_stages/data.stage_count*100)+'%';
  $('build-stages').replaceChildren(...data.stages.map((s,i)=>{const li=element('li',s.status);li.append(element('span','step-mark',s.status==='complete'?'✓':String(i+1).padStart(2,'0')),element('span','',s.label));return li;}));
  $('build-resume').hidden=!['failed','interrupted','cancelled'].includes(job.status)||job.attempts>=3;$('build-cancel').hidden=job.status!=='queued';
  if(data.preview_errors.length){$('build-warning').hidden=false;$('build-warning').textContent='Part of the construction preview is unavailable. The world build and its quality settings are unaffected.';}
}
async function pollBuild(){
  if(!active||polling)return;polling=true;const id=active.id;
  try{
    const data=await api('/api/jobs/'+id+'/construction');if(active?.id!==id)return;displayProgress(data);
    if(data.job.status==='complete'){if(!result&&!loadingResult)await showResult(id);return;}
    if(Object.keys(data.previews).length&&!previewUnavailable){
      try{
        if(!preview){preview=new ConstructionPreview($('construction-canvas'));$('construction-canvas').addEventListener('construction-assets',e=>{
          $('build-counts').replaceChildren(...e.detail.groups.filter(g=>g.actual_instances).map(g=>element('span','',`${Number(g.actual_instances).toLocaleString()} ${g.name.toLowerCase()}`)));
        },{once:true});}
        await preview.update(data.previews);if(active?.id!==id)return;
        $('construction-wait').hidden=!!$('construction-canvas').dataset.surface;$('motion-toggle').textContent=preview.auto?'Pause rotation':'Resume rotation';
      }catch(error){if(!preview)previewUnavailable=true;$('build-warning').hidden=false;$('build-warning').textContent='3D preview unavailable in this browser. Live build stages and the final Isaac images remain available.';}
    }
  }catch(error){if(active?.id===id){$('build-warning').hidden=false;$('build-warning').textContent='Reconnecting to the local build. Your job is saved.';}}
  finally{polling=false;}
}
$('motion-toggle').onclick=()=>{if(preview){preview.auto=!preview.auto;$('motion-toggle').textContent=preview.auto?'Pause rotation':'Resume rotation';}};
for(const action of ['resume','cancel'])$('build-'+action).onclick=async()=>{const button=$('build-'+action);button.disabled=true;try{await api('/api/jobs/'+action,{id:active.id,operation_key:crypto.randomUUID()});await pollBuild();}catch(error){toast(error.message);}finally{button.disabled=false;}};
async function showResult(id){
  loadingResult=true;try{
    const data=await api('/api/results/'+id);if(active?.id!==id)return;result=data;clearPreview();
    $('building').hidden=true;$('ready').hidden=false;$('ready-title').textContent=selected.title;$('ready-source').src=selected.minecraft[0]?.url||'';
    $('ready-download').hidden=!data.download;if(data.download){$('ready-download').href=data.download.url;$('ready-download').textContent=`Download world · ${(data.download.bytes/1e9).toFixed(1)} GB ↓`;}
    $('ready-qualification').textContent='Development world. Full realism, motion and contact qualification remain pending.';
    $('open-result').href='/results/'+id;$('new-view-controls').dataset.preset=selected.id;$('new-view-controls').dataset.target=id;renderGallery();assistantState();
  }finally{loadingResult=false;}
}
function renderGallery(){
  if(!result)return;const entries=result.images;if(frame>=entries.length)frame=0;const entry=entries[frame];
  if(entry){$('ready-image').src=entry.url;$('ready-image').alt=entry.caption;$('ready-caption').textContent=entry.caption.replaceAll('_',' ');}
  $('ready-thumbnails').replaceChildren();entries.forEach((entry,index)=>{if(filter!=='all'&&entry.kind!==filter)return;const button=element('button','thumbnail'+(index===frame?' selected':''));button.setAttribute('aria-label',entry.caption);button.setAttribute('aria-pressed',String(index===frame));const image=element('img');image.src=entry.url;image.alt='';image.loading='lazy';button.append(image,element('span','',entry.caption.replaceAll('_',' ')));button.onclick=()=>{frame=index;renderGallery();};$('ready-thumbnails').append(button);});
  document.querySelectorAll('[data-ready-filter]').forEach(b=>b.classList.toggle('selected',b.dataset.readyFilter===filter));
}
function renderConversation(){
  const host=$('conversation'),old=host.scrollTop,follow=host.scrollHeight-old-host.clientHeight<60;host.replaceChildren();
  for(const job of [...jobs].reverse().filter(j=>j.action==='agent'&&j.preset===selected?.id&&(j.context_job||null)===(active?.id||null))){
    host.append(element('div','chat user',job.prompt));const reply=job.result?.answer||job.error||(job.status==='running'?'Reading the recorded world evidence…':job.status==='queued'?'Request saved; waiting for the assistant.':job.status);
    const message=element('div','chat',reply);message.dataset.jobId=job.id;host.append(message);
    if(['failed','interrupted','cancelled'].includes(job.status)&&job.attempts<3){const retry=element('button','quiet','Resume request');retry.onclick=async()=>{try{await api('/api/jobs/resume',{id:job.id,operation_key:crypto.randomUUID()});await pollJobs();}catch(error){toast(error.message);}};message.append(retry);}
  }
  const related=jobs.filter(j=>j.preset===selected?.id&&j.action!=='agent'&&(['queued','running','failed','interrupted'].includes(j.status)||(j.action==='views'&&j.target===active?.id))).slice(0,5);
  for(const job of related){const card=element('div','chat task-status');card.dataset.jobId=job.id;card.append(element('strong','',({convert:'World build',views:'New views',verify:'Verification',prepare:'Download preparation'}[job.action]||job.action)+' · '+job.status),element('p','',job.error||job.progress||job.result?.message||''));if(job.action==='convert'){const watch=element('button','quiet','Open this build');watch.onclick=()=>openBuild(job).catch(e=>toast(e.message));card.append(watch);}if(job.view_url){const link=element('a','quiet','View result ↗');link.href=job.view_url;card.append(link);}host.append(card);}
  host.scrollTop=follow?host.scrollHeight:old;
}
async function pollJobs(){
  try{
    const data=await api('/api/jobs');jobs=data.jobs;workers=data.workers||{};updateCreate();renderConversation();
    if(active)await pollBuild();
  }catch(error){/* The build panel reports reconnecting; retained jobs remain safe. */}
}
$('agent-toggle').onclick=()=>{$('assistant').hidden=false;$('prompt').focus();};$('agent-close').onclick=()=>$('assistant').hidden=true;
$('agent-form').onsubmit=async e=>{e.preventDefault();const prompt=$('prompt').value.trim();if(!prompt||!selected)return;$('agent-send').disabled=true;
  try{await api('/api/jobs',{preset:selected.id,action:'agent',prompt,operation_key:crypto.randomUUID(),...(active?{context_job:active.id}:{})});$('prompt').value='';await pollJobs();}catch(error){toast(error.message);}finally{assistantState();}};
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;$('prompt').focus();});
document.querySelectorAll('[data-ready-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.readyFilter;const first=result.images.findIndex(i=>filter==='all'||i.kind===filter);if(first>=0)frame=first;renderGallery();});
$('ready-expand').onclick=()=>{$('lightbox-image').src=$('ready-image').src;$('lightbox-caption').textContent=$('ready-caption').textContent;$('lightbox').showModal();};$('lightbox-close').onclick=()=>$('lightbox').close();$('lightbox').onclick=e=>{if(e.target===$('lightbox'))$('lightbox').close();};
$('back-to-worlds').onclick=home;$('choose-again').onclick=home;
document.addEventListener('isaacmin:views-queued',pollJobs);document.addEventListener('isaacmin:views-ready',()=>{if(active)showResult(active.id);});
async function init(){catalog=await api('/api/catalog');sourceCards();assistantState();await pollJobs();const id=new URLSearchParams(location.search).get('job');if(id){const job=await api('/api/jobs/'+id);if(job.action!=='convert')throw Error('Choose a world conversion');await openBuild(job);}}
init().catch(error=>toast('Could not open the workspace: '+error.message));setInterval(pollJobs,3000);
