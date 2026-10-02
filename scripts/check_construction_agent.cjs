const {chromium}=require('../.tools/demo-browser/node_modules/playwright');
const fs=require('fs');
(async()=>{
 const observed=JSON.parse(fs.readFileSync('evidence/demo/construction_browser.json')),base='http://127.0.0.1:8765';
 const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:1000}});await page.goto(base+'/?job='+observed.job_id);
  await page.locator('#agent-toggle:enabled').waitFor();await page.locator('#agent-toggle').click();
  await page.locator('#prompt').fill('Which exact conversion am I viewing, what is its current status, and what stage is running? Read its saved job. Do not start or change any work.');
  const queued=page.waitForResponse(r=>r.url()===base+'/api/jobs'&&r.request().method()==='POST');
  await page.locator('#agent-send').click();const response=await queued,job=await response.json();
  if(response.status()!==202||job.context_job!==observed.job_id)throw Error('Wrong assistant conversion context');
  console.log('Actual NVIDIA context request:',job.id);
  let result;for(let i=0;i<100;i++){
   result=await(await page.request.get(base+'/api/jobs/'+job.id)).json();
   if(result.status==='complete')break;if(['failed','interrupted','cancelled'].includes(result.status))throw Error(result.error||result.status);await page.waitForTimeout(3000);
  }
  if(result.status!=='complete')throw Error('Hosted request timed out');
  const state=JSON.parse(fs.readFileSync('state/demo_agent_runs/'+job.id+'.json'));
  if(state.binding.context_job!==observed.job_id)throw Error('Context binding not retained');
  const inspection=state.events.find(e=>e.tool==='inspect_job'&&e.result.id===observed.job_id);
  if(!inspection)throw Error('No actual current-job inspection');
  if(state.events.some(e=>['start_job','request_views','control_job'].includes(e.tool)))throw Error('Read-only question changed work');
  await page.locator('#conversation [data-job-id="'+job.id+'"]').filter({hasText:result.result.answer.slice(0,20)}).waitFor({timeout:12000});
  await page.screenshot({path:'artifacts/demo/construction_browser/06_context_agent.png',fullPage:true});
  const receipt={status:'pass',at_utc:new Date().toISOString(),conversion:observed.job_id,agent_job:job.id,answer:result.result.answer,binding:state.binding,tools:state.events,scope:'Actual Chromium request to NVIDIA with current conversion context; read-only status, no image understanding or extra renderer call'};
  fs.writeFileSync('evidence/demo/construction_agent.json',JSON.stringify(receipt,null,2)+'\n');console.log('PASS hosted current-conversion context');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
