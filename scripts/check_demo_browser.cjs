// Actual local browser smoke check. No model calls, conversion jobs or synthetic images.
const { chromium } = require('../.tools/demo-browser/node_modules/playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = 'http://127.0.0.1:8765';
  const output = path.resolve('artifacts/demo/browser');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const check = (condition, message) => { if (!condition) throw Error(message); };
  await page.goto(base+'/library');
  await page.locator('.preset').nth(3).waitFor();
  const worlds = [];
  for (let index = 0; index < 4; index++) {
    await page.locator('.preset').nth(index).click();
    await page.locator('#source-image').evaluate(image => image.decode());
    const name = await page.locator('#world-title').textContent();
    const counts = {};
    for (const filter of ['ground', 'aerial', 'all']) {
      await page.locator(`[data-filter="${filter}"]`).click();
      counts[filter] = await page.locator('.thumbnail').count();
      check(counts[filter] > 0, name + ': missing ' + filter);
      await page.locator('.thumbnail').last().click();
      await page.locator('#isaac-image').evaluate(image => image.decode());
    }
    await page.locator('#isaac-expand').click();
    await page.locator('#lightbox[open]').waitFor();
    await page.keyboard.press('Escape');
    check(!(await page.locator('#lightbox').isVisible()), 'Lightbox did not close');
    check(await page.locator('#download').isVisible(), name + ': missing download');
    const href = await page.locator('#download').getAttribute('href');
    const response = await page.request.get(base + href, { headers: { Range: 'bytes=0-15' } });
    const bytes = await response.body();
    check(response.status() === 206 && bytes.subarray(0, 2).toString() === 'PK', 'ZIP range failed');
    await page.locator('[data-filter="aerial"]').click();
    await page.locator('#isaac-image').evaluate(image => image.decode());
    await page.screenshot({ path: path.join(output, `world_${index}_aerial.png`) });
    worlds.push({ name, counts, sourceLoaded: true, zipRange: 'pass' });
  }
  // A rejected location must show its real API error without queueing a render.
  const beforeJobs = (await (await page.request.get(base + '/api/jobs')).json()).jobs.map(j => j.id);
  await page.locator('.view-request summary').click();
  await page.locator('.view-request input[name=x]').fill('1000000000');
  await page.locator('.view-request input[name=z]').fill('1000000000');
  const rejected = page.waitForResponse(r => r.url() === base + '/api/jobs' && r.request().method() === 'POST');
  await page.locator('.view-request button[type=submit]').click();
  check((await rejected).status() === 400, 'Invalid camera point was accepted');
  await page.locator('.view-request-status').filter({ hasText: 'outside the built region' }).waitFor();
  const afterJobs = (await (await page.request.get(base + '/api/jobs')).json()).jobs.map(j => j.id);
  check(JSON.stringify(beforeJobs) === JSON.stringify(afterJobs), 'Rejected view queued work');
  await page.locator('#agent-toggle').click();
  check((await page.locator('#assistant-world').textContent()).includes('Frozen coast'), 'Wrong assistant world');
  await page.locator('#agent-close').click();
  await page.reload();
  await page.locator('.preset.selected').waitFor();
  check((await page.locator('#world-title').textContent()) === 'Frozen coast', 'Selection did not survive reload');
  await page.setViewportSize({ width: 390, height: 844 });
  check(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Mobile horizontal overflow');
  await page.locator('#agent-toggle').click();
  const sendBox = await page.locator('#agent-send').boundingBox();
  check(sendBox && sendBox.y >= 0 && sendBox.y + sendBox.height <= 844, 'Mobile assistant input is obscured');
  check((await page.locator('#conversation').boundingBox()).height >= 80, 'Conversation has no scroll area');
  await page.screenshot({ path: path.join(output, 'mobile_assistant.png') });
  await page.locator('#agent-close').click();
  await page.screenshot({ path: path.join(output, 'mobile_world.png'), fullPage: true });
  check(errors.length === 0, errors.join('; '));
  const result = { status: 'pass', at_utc: new Date().toISOString(), browser: await browser.version(),
    worlds, errors, lightbox: 'pass', mobile: 'pass', selectionReload: 'pass',
    invalidCameraFeedback: 'pass', mobileChatInput: 'pass',
    scope: 'Actual local app and delivered images/downloads; no realism or navigation qualification' };
  fs.writeFileSync('evidence/demo/browser_checks.json', JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
})().catch(error => { console.error(error.message); process.exitCode = 1; });
