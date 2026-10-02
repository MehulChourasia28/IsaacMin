// Actual Chromium -> OpenClaw -> NVIDIA -> validated view job -> native Isaac.
const {chromium}=require('../.tools/demo-browser/node_modules/playwright');
const fs=require('fs'),path=require('path');
const output=path.resolve('artifacts/demo/openclaw_chat_browser');fs.mkdirSync(output,{recursive:true});
const receipt={status:'running',started_at_utc:new Date().toISOString(),errors:[]};
const save=()=>fs.writeFileSync('evidence/demo/openclaw_chat_browser.json',JSON.stringify(receipt,null,2)+'\n');
const check=(ok,msg)=>{if(!ok)throw Error(msg);};
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1080}}),base='http://127.0.0.1:8765';
  page.on('pageerror',e=>receipt.errors.push(e.message));
  const target='job_dbc17a079cb748b0a2990553a81b72b4';receipt.target=target;
  await page.goto(base+'/?job='+target);await page.locator('#ready').waitFor();
  const before=await(await page.request.get(base+'/api/results/'+target)).json();receipt.before_count=before.images.length;
  await page.locator('#agent-toggle').click();
  check((await page.locator('#planner-status').innerText()).includes('OpenClaw'),'Harness label absent');
  const prompt='Create exactly one more aerial view of this world, height 180 metres and heading 120 degrees. Use the default centre. Do not rebuild the world.';
  const existing=(await(await page.request.get(base+'/api/jobs')).json()).jobs.find(j=>j.action==='agent'&&j.context_job===target&&j.prompt===prompt);
  if(!existing){await page.locator('#prompt').fill(prompt);await page.locator('#agent-send').click();}
  let agent;
  for(let i=0;i<100;i++){
   const data=await(await page.request.get(base+'/api/jobs')).json();
   agent=data.jobs.find(j=>j.action==='agent'&&j.context_job===target&&j.prompt===prompt);
   if(agent&&['failed','interrupted'].includes(agent.status))throw Error(agent.error);
   if(agent?.status==='complete')break;
   await page.waitForTimeout(2000);
  }
  check(agent?.result?.harness==='openclaw','Actual assistant did not complete through OpenClaw');
  // Detailed tools are intentionally absent from the public HTTP job projection.
  const trace=JSON.parse(fs.readFileSync('state/demo_agent_runs/'+agent.id+'.json','utf8'));
  const calls=trace.events.filter(t=>t.tool==='request_views');
  check(calls.length===1&&calls[0].result.action==='views','Expected one actual view action');
  const chosen=calls[0].arguments;
  check(chosen.target===target&&chosen.view.count===1&&chosen.view.kind==='aerial'&&chosen.view.height_m===180&&chosen.view.heading_degrees===120,'Agent changed requested camera controls');
  const id=calls[0].result.id;receipt.agent=agent;receipt.tools=trace.events;receipt.view_job=id;save();
  await page.screenshot({path:path.join(output,'01_requested_view.png'),fullPage:true});
  for(let i=0;i<1000;i++){
   const job=await(await page.request.get(base+'/api/jobs/'+id)).json();
   if(['failed','interrupted'].includes(job.status))throw Error('Native view failed: '+job.error);
   if(job.status==='complete'){
    await page.reload();await page.locator('#ready').waitFor();
    const after=await(await page.request.get(base+'/api/results/'+target)).json();
    check(after.images.length===before.images.length+1,'New native view missing from correct world');
    receipt.result=job;receipt.after_count=after.images.length;
    await page.screenshot({path:path.join(output,'02_native_view_ready.png'),fullPage:true});
    receipt.status='pass';receipt.completed_at_utc=new Date().toISOString();save();return;
   }
   await page.waitForTimeout(2500);
  }
  throw Error('Native view deadline exceeded');
 }catch(error){receipt.status='failed';receipt.failure=String(error);save();throw error;}
 finally{await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});
