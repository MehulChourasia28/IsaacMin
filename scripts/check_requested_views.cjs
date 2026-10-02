// Exercise real browser controls and actual native captures of existing worlds.
const { chromium } = require('../.tools/demo-browser/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

(async () => {
  const base = 'http://127.0.0.1:8765';
  const output = path.resolve('artifacts/demo/browser');
  const receipt = 'evidence/demo/requested_views_browser.json';
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const check = (v, message) => { if (!v) throw Error(message); };
  const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const result = { status: 'running', started_at_utc: new Date().toISOString(), jobs: [], errors,
    scope: 'Actual UI submission and native render over existing USD; no world repair or realism qualification' };
  const save = () => fs.writeFileSync(receipt, JSON.stringify(result, null, 2) + '\n');
  try {
    const presets = JSON.parse(fs.readFileSync('state/demo_presets.json')).presets;
    const conversion = 'job_efa3220eaba64f2f9d7effb82bd89d2d';
    for (const request of [
      { preset: 'isaacmin', url: '/?preset=isaacmin', kind: 'ground', count: '1',
        build: presets.find(p => p.id === 'isaacmin').build },
      { preset: 'isaacmin3', url: '/results/' + conversion, kind: 'aerial', count: '3',
        build: 'artifacts/demo/jobs/' + conversion + '/build', x: '-4297.03', z: '-3046.76', height: '180', heading: '45' }
    ]) {
      await page.goto(base + request.url);
      await page.locator('.view-request').waitFor();
      const manifest = path.join(request.build, 'outdoor_build.json');
      const scene = JSON.parse(fs.readFileSync(manifest)).scene;
      const before = { manifest: sha(manifest), scene: sha(scene) };
      await page.locator('select[name=kind]').selectOption(request.kind);
      await page.locator('select[name=count]').selectOption(request.count);
      if (request.x) {
        await page.locator('.view-request summary').click();
        for (const name of ['x', 'z', 'height', 'heading']) await page.locator(`.view-request input[name=${name}]`).fill(request[name]);
      }
      const pending = page.waitForResponse(r => r.url() === base + '/api/jobs' && r.request().method() === 'POST');
      await page.locator('.view-request button[type=submit]').click();
      const response = await pending, job = await response.json();
      check(response.status() === 202, JSON.stringify(job));
      result.jobs.push({ request, id: job.id, before, scene, manifest }); save();
      await page.screenshot({ path: path.join(output, 'requested_' + request.kind + '_queued.png'), fullPage: true });
      console.log(JSON.stringify({ queued: job.id, kind: request.kind, count: request.count }));
    }
    const deadline = Date.now() + 25 * 60 * 1000;
    while (Date.now() < deadline) {
      const jobs = (await (await page.request.get(base + '/api/jobs')).json()).jobs;
      for (const item of result.jobs) {
        const job = jobs.find(j => j.id === item.id);
        check(job && !['failed', 'interrupted'].includes(job.status), JSON.stringify(job));
        item.observed = job;
      }
      save();
      if (result.jobs.every(j => j.observed.status === 'complete')) break;
      await new Promise(resolve => setTimeout(resolve, 4000));
    }
    for (const item of result.jobs) {
      check(item.observed.status === 'complete', 'Native view deadline reached');
      check(item.before.manifest === sha(item.manifest) && item.before.scene === sha(item.scene), 'Existing world was changed');
      item.existing_world_hashes_unchanged = true;
      await page.goto(base + item.request.url);
      if (item.request.kind === 'ground') {
        await page.locator('[data-filter=ground]').click();
        await page.locator('.thumbnail').last().click();
        await page.locator('#isaac-image').evaluate(i => i.decode());
        check((await page.locator('#isaac-caption').textContent()).toLowerCase().includes('requested'), 'New ground view is absent');
      } else {
        const images = page.locator('.result-gallery img');
        check(await images.count() === 12, 'Expected 9 original plus 3 new captures');
        await images.last().scrollIntoViewIfNeeded();
        await images.last().evaluate(i => i.decode());
        const response = await page.request.get(base + '/download/' + conversion, { headers: { Range: 'bytes=0-15' } });
        check(response.status() === 206 && (await response.body()).subarray(0, 2).toString() === 'PK', 'Fresh conversion download failed');
        result.fresh_conversion_download = 'pass';
      }
      await page.screenshot({ path: path.join(output, 'requested_' + item.request.kind + '_complete.png'), fullPage: true });
    }
    check(!errors.length, errors.join('; '));
    result.status = 'pass'; result.completed_at_utc = new Date().toISOString(); save();
    console.log(JSON.stringify({ status: result.status, jobs: result.jobs.map(j => j.id) }));
  } catch (error) {
    result.status = 'failed'; result.error = error.message; save(); throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
