// Real UI -> real save conversion -> observed native outputs. No fake jobs or images.
const {chromium}=require('../.tools/demo-browser/node_modules/playwright');
const fs=require('fs'),path=require('path');
const output=path.resolve('artifacts/demo/construction_browser');fs.mkdirSync(output,{recursive:true});
const resuming=process.argv.includes('--resume');
const receipt=resuming?JSON.parse(fs.readFileSync('evidence/demo/construction_browser.json')):{status:'running',started_at_utc:new Date().toISOString(),events:[],errors:[],scope:'Actual browser and native conversion; the WebGL construction diagram is not an Isaac render or realism qualification'};
if(resuming){receipt.interruptions=[...(receipt.interruptions||[]),{at_utc:new Date().toISOString(),reason:receipt.failure||'Recorder resumed',scope:'Recorder interruption; same actual conversion retained'}];delete receipt.failure;receipt.status='running';}
const save=()=>fs.writeFileSync('evidence/demo/construction_browser.json',JSON.stringify(receipt,null,2)+'\n');
const check=(ok,message)=>{if(!ok)throw Error(message);};
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),base='http://127.0.0.1:8765';
  page.on('pageerror',e=>receipt.errors.push(e.message));
  const requested=[];page.on('request',r=>requested.push(r.url()));
  await page.goto(base);await page.locator('.source-card').nth(3).waitFor();
  await page.locator('.source-card img').evaluateAll(images=>Promise.all(images.map(i=>i.decode())));
  check(await page.locator('#ready').isHidden(),'Generated world visible on initial load');
  check(await page.locator('#setup').isHidden(),'World was preselected');
  check(await page.locator('#agent-toggle').isDisabled(),'Assistant has no selected-world boundary');
  const catalog=await(await page.request.get(base+'/api/catalog')).json();
  const isaacURLs=catalog.presets.flatMap(p=>p.isaac.map(i=>base+i.url));
  check(!requested.some(u=>isaacURLs.includes(u)),'Landing page requested old Isaac captures');
  receipt.landing={presets:4,source_only:true,preselected:false};
  await page.screenshot({path:path.join(output,'01_clean_start.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile landing overflow');
  await page.screenshot({path:path.join(output,'02_mobile_start.png'),fullPage:true});
  await page.setViewportSize({width:1440,height:1100});
  let job;
  if(resuming){job=await(await page.request.get(base+'/api/jobs/'+receipt.job_id)).json();await page.goto(base+'/?job='+job.id);}
  else{await page.locator('.source-card[data-preset=isaacmin1]').click();
    const queued=page.waitForResponse(r=>r.url()===base+'/api/jobs'&&r.request().method()==='POST');
    await page.locator('#create-world').click();const response=await queued;check(response.status()===202,'Conversion not queued');job=await response.json();}
  receipt.job_id=job.id;save();console.log(resuming?'Actual UI conversion resumed in recorder:':'Actual UI conversion started:',job.id);
  await page.locator('#building:visible, #ready:visible').waitFor();await page.reload();await page.locator('#building:visible, #ready:visible').waitFor();
  check(new URL(page.url()).searchParams.get('job')===job.id,'Reload lost job');receipt.refresh_keeps_job=true;
  const captured=new Set(receipt.events.map(e=>e.preview||'stage:'+e.phase));let final;
  for(let n=0;n<800;n++){
   let r;try{r=await page.request.get(base+'/api/jobs/'+job.id+'/construction');}catch(error){console.log('Local server reconnecting; same saved job');await page.waitForTimeout(3000);continue;}const state=await r.json();
   if(!state.job)throw Error(JSON.stringify(state));
   const phase=state.stages.find(s=>s.status==='running')?.id||state.job.status;
   if(!captured.has('stage:'+phase)){captured.add('stage:'+phase);receipt.events.push({at_utc:new Date().toISOString(),phase,completed:state.completed_stages});save();console.log('Actual stage:',phase);}
   for(const name of ['source','terrain','assets']){
    const attr=name==='assets'?'assetPoints':'surface';
    const ready=await page.locator('#construction-canvas').evaluate((el,{name,attr})=>name==='assets'?Number(el.dataset[attr])>0:el.dataset[attr]===(name==='source'?'actual_source_surface_sample':'actual_final_surface_sample'),{name,attr});
    if(ready&&!captured.has(name)){captured.add(name);await page.waitForTimeout(name==='assets'?8500:2200);
      await page.screenshot({path:path.join(output,'03_'+name+'.png'),fullPage:true});receipt.events.push({at_utc:new Date().toISOString(),preview:name,data:await page.locator('#construction-canvas').evaluate(el=>({...el.dataset}))});save();console.log('Actual preview:',name);}
   }
   if(state.job.status==='complete'){final=state;break;}
   if(['failed','interrupted','cancelled'].includes(state.job.status))throw Error(state.job.error||state.job.status);
   await page.waitForTimeout(3000);
  }
  check(final,'Conversion timed out');await page.locator('#ready').waitFor({timeout:20000});
  await page.locator('#ready-image').evaluate(i=>i.decode());
  check(captured.has('terrain')&&captured.has('assets'),'Missing real terrain or native asset preview');
  receipt.previews={source:captured.has('source'),terrain:captured.has('terrain'),assets:captured.has('assets')};
  receipt.final=await(await page.request.get(base+'/api/results/'+job.id)).json();
  const build=path.join('artifacts/demo/jobs',job.id,'build/assembled');
  const planned=['preview','overview'].flatMap(kind=>JSON.parse(fs.readFileSync(path.join(build,kind+'_request.json'))).poses);
  check(receipt.final.images.length===planned.length,'Native gallery does not match its complete planned capture set');
  receipt.planned_captures=planned.length;
  await page.locator('[data-ready-filter=aerial]').click();await page.locator('#ready-image').evaluate(i=>i.decode());
  await page.screenshot({path:path.join(output,'04_world_ready.png'),fullPage:true});
  await page.locator('#ready-expand').click();await page.locator('#lightbox[open]').waitFor();await page.keyboard.press('Escape');
  const download=await page.request.get(base+receipt.final.download.url,{headers:{Range:'bytes=0-15'}});
  check(download.status()===206&&(await download.body()).subarray(0,2).toString()==='PK','Download is not the real ZIP');
  receipt.download_range='pass';
  check((await page.locator('#new-view-controls').getAttribute('data-target'))===job.id,'Extra views target wrong world');
  await page.setViewportSize({width:390,height:844});check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile result overflow');
  await page.locator('#agent-toggle').click();const box=await page.locator('#agent-send').boundingBox();check(box&&box.y>=0&&box.y+box.height<=844,'Mobile assistant input obscured');
  await page.screenshot({path:path.join(output,'05_mobile_result.png'),fullPage:true});
  check(!receipt.errors.length,receipt.errors.join('; '));
  receipt.status='pass';receipt.finished_at_utc=new Date().toISOString();save();console.log('PASS actual construction UI conversion',job.id);
 }catch(error){receipt.status='failed';receipt.failure=error.message;save();throw error;}finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
