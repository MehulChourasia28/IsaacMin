'use strict';
// Shared controls for a saved preset and a newly converted result.
for(const host of document.querySelectorAll('.view-controls')){
 const el=(tag,text)=>{const n=document.createElement(tag);if(text)n.textContent=text;return n;};
 const form=el('form');form.className='view-request';const title=el('h3','Create more views');
 const help=el('p','Render new images of this existing world. Each request keeps its geometry and materials.');
 const row=el('div');row.className='view-fields';
 function select(label,options,name){const wrap=el('label',label),input=el('select');input.name=name;for(const [value,text] of options){const o=el('option',text);o.value=value;input.append(o);}wrap.append(input);row.append(wrap);return input;}
 const kind=select('View type',[['ground','Ground · 0.6 m'],['aerial','Aerial']],'kind');
 const count=select('Number of views',[['1','1 view'],['3','3 views'],['6','6 views']],'count');
 const details=el('details'),summary=el('summary','Location and direction (optional)'),advanced=el('div');advanced.className='view-fields';details.append(summary,advanced);
 function number(label,name,placeholder,min,max){const wrap=el('label',label),input=el('input');input.type='number';input.name=name;input.step='any';input.placeholder=placeholder;if(min!==undefined)input.min=min;if(max!==undefined)input.max=max;wrap.append(input);advanced.append(wrap);return input;}
 const x=number('Minecraft X','x','Automatic'),z=number('Minecraft Z','z','Automatic');
 const heading=number('Starting heading °','heading','Auto · 0 N / 90 E',0,360),height=number('Aerial height (m)','height','Automatic',20,600);
 kind.onchange=()=>{height.parentElement.hidden=kind.value!=='aerial';};kind.onchange();
 const hint=el('p','Leave X and Z blank for supported automatic locations. Multiple views spread around the starting direction. Rendering takes a few minutes; you can keep browsing.');hint.className='subtle';
 const button=el('button','Render views');button.type='submit';button.className='secondary';
 const status=el('p');status.className='view-request-status';status.setAttribute('role','status');
 form.append(title,help,row,details,hint,button,status);host.append(form);
 form.onsubmit=async event=>{event.preventDefault();button.disabled=true;
  const requestedPreset=host.dataset.preset,requestedTarget=host.dataset.target||'preset';
  try{
   if((x.value==='')!==(z.value===''))throw Error('Enter both Minecraft X and Z, or leave both blank.');
   const view={kind:kind.value,count:Number(count.value)};
   if(x.value!=='')view.source_xz=[Number(x.value),Number(z.value)];
   if(heading.value!=='')view.heading_degrees=Number(heading.value);
   if(kind.value==='aerial'&&height.value!=='')view.height_m=Number(height.value);
   const catalog=await(await fetch('/api/catalog',{cache:'no-store'})).json();
   const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json','X-IsaacMin-Token':catalog.csrf},body:JSON.stringify({preset:requestedPreset,target:requestedTarget,action:'views',operation_key:crypto.randomUUID(),view})});
   const job=await response.json();if(!response.ok)throw Error(job.error||'Could not queue views');
   status.textContent='Views queued. Follow the request in workspace activity.';document.dispatchEvent(new CustomEvent('isaacmin:views-queued'));
   const timer=setInterval(async()=>{try{const r=await fetch('/api/jobs/'+job.id);if(!r.ok)return;const current=await r.json();
    if(current.status==='complete'){clearInterval(timer);status.textContent='Your new views are ready.';if(requestedTarget!=='preset')location.reload();else document.dispatchEvent(new CustomEvent('isaacmin:views-ready'));}
    else if(['failed','interrupted','cancelled'].includes(current.status)){clearInterval(timer);status.textContent=current.error||'The request stopped. Resume it from workspace activity.';}
    else status.textContent=current.progress||'Views queued. You can keep browsing.';
   }catch(_){}},3000);
  }catch(error){status.textContent=error.message;}finally{button.disabled=false;}
 };
}
