// Actual browser recording of the labelled construction diagram; NOT an Isaac video.
const {chromium}=require('../.tools/demo-browser/node_modules/playwright');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
 const root=process.cwd(),job=JSON.parse(fs.readFileSync('evidence/demo/construction_browser.json')).job_id;
 const directory=path.join(root,'artifacts/demo/jobs',job),output=path.join(root,'artifacts/demo/construction_browser');
 const metadata=JSON.parse(fs.readFileSync(path.join(directory,'construction/assets.json')));
 const scene=path.join(directory,'build/assembled/scene/world.usda'),before=hash(scene);
 if(before!==metadata.scene_sha256)throw Error('Native preview is not the current world');
 if(hash(path.join(directory,'construction/assets.bin'))!==metadata.sha256)throw Error('Native sample bytes changed');
 const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000},recordVideo:{dir:path.join(output,'recordings'),size:{width:1440,height:1000}}});
  const page=await context.newPage(),video=page.video();await page.goto('http://127.0.0.1:8765/?job='+job);
  await page.waitForFunction(()=>Number(document.querySelector('#construction-canvas').dataset.assetPoints)>0);
  const first=await page.locator('#construction-canvas').screenshot();
  await page.waitForTimeout(9000);const second=await page.locator('#construction-canvas').screenshot();
  if(first.equals(second))throw Error('Construction preview did not animate');
  await page.locator('#motion-toggle').click();
  if(await page.locator('#motion-toggle').textContent()!=='Resume rotation')throw Error('Rotation did not pause');
  await page.screenshot({path:path.join(output,'07_live_construction.png'),fullPage:true});
  await page.waitForTimeout(1000);await context.close();
  await video.saveAs(path.join(output,'construction_ui_recording.webm'));
  if(hash(scene)!==before)throw Error('Native scene bytes changed during read-only UI recording');
  const receipt={status:'pass',at_utc:new Date().toISOString(),job,scene_sha256:before,asset_samples:metadata.points,scene_preserved:true,
   moving_pixels:true,rotation_control:'pass',file:'artifacts/demo/construction_browser/construction_ui_recording.webm',
   sha256:hash(path.join(output,'construction_ui_recording.webm')),
   scope:'Actual Chromium recording of the labelled simplified construction preview, replaying actual completed stage data while the native build runs. NOT a final Isaac render, synthetic substitute for Isaac evidence, or per-instance timing trace.'};
  fs.writeFileSync('evidence/demo/construction_animation.json',JSON.stringify(receipt,null,2)+'\n');console.log(receipt);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
