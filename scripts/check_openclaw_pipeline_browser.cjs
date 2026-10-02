// Actual browser button, hosted stage dispatch, native conversion and download.
const {chromium}=require('../.tools/demo-browser/node_modules/playwright');
const fs=require('fs'),path=require('path');
const output=path.resolve('artifacts/demo/openclaw_pipeline_browser');fs.mkdirSync(output,{recursive:true});
const receipt={status:'running',started_at_utc:new Date().toISOString(),events:[],errors:[]};
const save=()=>fs.writeFileSync('evidence/demo/openclaw_pipeline_browser.json',JSON.stringify(receipt,null,2)+'\n');
const check=(ok,message)=>{if(!ok)throw Error(message);};
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),base='http://127.0.0.1:8765';
  page.on('pageerror',e=>receipt.errors.push(e.message));
  await page.goto(base+'/library');await page.locator('#empty-library').waitFor();
  check(await page.locator('.workspace').isHidden(),'Old saved-world gallery remains visible');
  await page.screenshot({path:path.join(output,'01_empty_library.png'),fullPage:true});
  await page.goto(base);await page.locator('.source-card').nth(3).waitFor();
  const catalog=await(await page.request.get(base+'/api/catalog')).json();
  check(catalog.presets.length===4&&catalog.presets.every(p=>p.isaac.length===0&&!p.download),'Library is not clear');
  check(await page.locator('#setup').isHidden(),'Opening page is not a clean slate');
  await page.locator('.source-card img').evaluateAll(images=>Promise.all(images.map(i=>i.decode())));
  await page.screenshot({path:path.join(output,'02_clean_start.png'),fullPage:true});
  await page.locator('[data-preset="isaacmin"]').click();
  await page.locator('#create-world').click();await page.waitForURL(/job=job_/);
  const id=new URL(page.url()).searchParams.get('job');receipt.job_id=id;save();
  let last='',reloaded=false;
  for(let i=0;i<900;i++){
   const job=await(await page.request.get(base+'/api/jobs/'+id)).json();
   const state=job.status+':'+job.pipeline_agent?.status+':'+job.pipeline_agent?.current_stage;
   if(state!==last){last=state;receipt.events.push({at_utc:new Date().toISOString(),state,agent:job.pipeline_agent});save();}
   if(job.pipeline_agent?.dispatches>=2&&!reloaded){
    await page.reload();await page.locator('#build-agent').waitFor();
    check((await page.locator('#build-agent').innerText()).includes('NVIDIA Nemotron'),'Agent progress is missing');
    await page.screenshot({path:path.join(output,'03_agent_build.png'),fullPage:true});reloaded=true;
   }
   if(['failed','interrupted'].includes(job.status))throw Error('Actual conversion stopped: '+job.error);
   if(job.status==='complete'){
    await page.locator('#ready').waitFor({timeout:15000});
    const result=await(await page.request.get(base+'/api/results/'+id)).json();
    check(result.images.length===8,'Actual planned capture set is incomplete');
    check(job.pipeline_agent?.completed===11&&job.pipeline_agent?.dispatches===11&&job.pipeline_agent?.harness==='openclaw','OpenClaw stage dispatch is incomplete');
    const zip=await page.request.get(base+job.download_url,{headers:{Range:'bytes=0-3'}});
    check(zip.status()===206&&(await zip.body()).subarray(0,2).toString()==='PK','Portable archive is unavailable');
    await page.screenshot({path:path.join(output,'04_world_ready.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Mobile result overflows');
    await page.screenshot({path:path.join(output,'05_mobile_result.png'),fullPage:true});
    receipt.result=job;receipt.images=result.images;receipt.refresh_recovered=true;receipt.download_range='pass';
    receipt.status='pass';receipt.completed_at_utc=new Date().toISOString();save();return;
   }
   await page.waitForTimeout(3000);
  }
  throw Error('Actual conversion recorder deadline exceeded');
 }catch(error){receipt.status='failed';receipt.failure=String(error);save();throw error;}
 finally{await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});
