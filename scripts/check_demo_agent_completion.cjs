// Real Chromium -> hosted NVIDIA tools -> persistent jobs -> native Isaac views.
const { chromium } = require('../.tools/demo-browser/node_modules/playwright');
const fs = require('fs');
const crypto = require('crypto');
const base = 'http://127.0.0.1:8765';
const output = 'evidence/demo/ui_agent_completion.json';
const check = (value, message) => { if (!value) throw Error(message); };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const result = { status: 'running', started_at_utc: new Date().toISOString(), checks: {}, agent_requests: [], errors: [],
    scope: 'Actual hosted requests and native capture; no automatic visual repair or realism qualification' };
  const save = () => fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  page.on('pageerror', error => result.errors.push(error.message));
  const getJob = async id => (await page.request.get(base + '/api/jobs/' + id)).json();
  async function waitJob(id, wanted = ['complete'], timeout = 240000) {
    const until = Date.now() + timeout;
    while (Date.now() < until) {
      const job = await getJob(id);
      if (wanted.includes(job.status)) return job;
      check(!['failed', 'interrupted', 'cancelled'].includes(job.status), JSON.stringify(job));
      await pause(1500);
    }
    throw Error('Timed out waiting for ' + id);
  }
  async function submitFromButton(button) {
    const response = page.waitForResponse(r => r.url() === base + '/api/jobs' && r.request().method() === 'POST');
    await button.click();
    const observed = await response, job = await observed.json();
    check(observed.status() === 202, JSON.stringify(job));
    return job;
  }
  async function ask(prompt) {
    if (!(await page.locator('#assistant').isVisible())) await page.locator('#agent-toggle').click();
    await page.locator('#prompt').fill(prompt);
    const queued = await submitFromButton(page.locator('#agent-send'));
    const job = await waitJob(queued.id);
    const checkpoint = JSON.parse(fs.readFileSync('state/demo_agent_runs/' + job.id + '.json'));
    result.agent_requests.push({ id: job.id, prompt, answer: job.result.answer, calls: checkpoint.calls, events: checkpoint.events });
    save(); console.log(JSON.stringify({ agent: job.id, tools: checkpoint.events.map(e => e.tool) }));
    return checkpoint;
  }
  try {
    await page.goto(base + '/?preset=isaacmin3');
    await page.locator('.preset.selected').waitFor();
    const conversion = 'job_efa3220eaba64f2f9d7effb82bd89d2d';
    const manifest = 'artifacts/demo/jobs/' + conversion + '/build/outdoor_build.json';
    const scene = JSON.parse(fs.readFileSync(manifest)).scene;
    const before = { manifest: hash(manifest), scene: hash(scene) };
    const captureRequest = await ask('Create one new aerial view of my latest completed conversion. Aim at Minecraft X -4297.03, Z -3046.76, use an aerial height of 160 metres and starting heading 75 degrees. Keep the world geometry and materials unchanged.');
    const capture = captureRequest.events.find(e => e.tool === 'request_views');
    check(capture && capture.result.action === 'views', 'Agent did not queue a real view');
    check(capture.result.target === conversion, 'Wrong render target');
    check(!captureRequest.events.some(e => e.tool === 'start_job'), 'Extra views must not rebuild');
    result.view_job = capture.result.id; save();
    await waitJob(capture.result.id, ['running', 'complete']);

    // While the native render occupies its lane, exercise queued-job controls.
    await page.locator('#agent-close').click();
    const cancelByUI = await submitFromButton(page.locator('#verify'));
    const row = page.locator(`[data-job-id="${cancelByUI.id}"]`);
    await row.getByRole('button', { name: 'Cancel', exact: true }).click();
    await waitJob(cancelByUI.id, ['cancelled']);
    result.checks.ui_queued_cancel = 'pass'; save();
    const second = await submitFromButton(page.locator('#verify'));
    const cancelled = await ask('Cancel only this queued verification job: ' + second.id + '. Do not start any other work.');
    check(cancelled.events.some(e => e.tool === 'control_job' && e.result.status === 'cancelled'), 'Hosted cancellation failed');
    check((await getJob(second.id)).status === 'cancelled', 'Job was not cancelled');
    result.checks.agent_queued_cancel = 'pass';
    const resumed = await ask('Resume the cancelled verification job ' + cancelByUI.id + '. Reuse that job; do not create another verification.');
    check(resumed.events.some(e => e.tool === 'control_job' && e.result.id === cancelByUI.id && e.result.status === 'queued'), 'Hosted resume failed');
    result.resumed_job = cancelByUI.id; result.checks.agent_resume = 'pass'; save();

    await page.locator('#agent-close').click();
    await page.locator('.preset').nth(0).click();
    const facts = await ask('Read the source biomes, centre and built extent of this selected world. Is its portable download ready, and is full motion/contact/realism qualification complete? Inspect only; do not queue work.');
    check(facts.events.some(e => e.tool === 'inspect_world' && e.result.topic === 'source' && e.result.context_biome_samples), 'Source evidence not read');
    check(!facts.events.some(e => ['start_job', 'request_views', 'control_job'].includes(e.tool)), 'Status question caused a mutation');
    result.checks.real_source_facts = 'pass';
    const help = await ask('Which questions and actions do you support, what are the new-view controls, and can you automatically inspect or repair the appearance?');
    check(help.events.some(e => e.tool === 'get_capabilities'), 'Capabilities not read');
    check(!help.events.some(e => ['start_job', 'request_views', 'control_job'].includes(e.tool)), 'Help question caused a mutation');
    result.checks.capability_help = 'pass'; save();

    const finished = await waitJob(capture.result.id, ['complete'], 15 * 60 * 1000);
    await waitJob(cancelByUI.id);
    check(before.manifest === hash(manifest) && before.scene === hash(scene), 'Existing world changed');
    result.checks.world_hashes_unchanged = 'pass'; result.native_result = finished;
    const report = JSON.parse(fs.readFileSync(finished.result.capture));
    check(report.frames.length === 1 && report.frames[0].pose.requested_height_m === 160, 'Capture did not retain requested camera');
    result.checks.native_requested_capture = 'pass';
    await page.goto(base + finished.view_url);
    const images = page.locator('.result-gallery img');
    check(await images.count() >= 13, 'Additional view not in conversion gallery');
    await images.last().scrollIntoViewIfNeeded(); await images.last().evaluate(image => image.decode());
    await page.screenshot({ path: 'artifacts/demo/browser/agent_requested_view_complete.png', fullPage: true });
    result.checks.result_gallery = 'pass';
    check(!result.errors.length, result.errors.join('; '));
    result.status = 'pass'; result.completed_at_utc = new Date().toISOString(); save();
    console.log(JSON.stringify({ status: result.status, checks: result.checks, view_job: result.view_job }));
  } catch (error) { result.status = 'failed'; result.error = error.message; save(); throw error; }
  finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
