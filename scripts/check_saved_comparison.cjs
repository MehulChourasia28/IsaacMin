// Real local browser/read-only checks. Uses existing saved-world captures.
const {chromium} = require('../.tools/demo-browser/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const check = (ok, message) => { if (!ok) throw Error(message); };
(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const base = 'http://127.0.0.1:8765', errors = [], results = [];
    const out = 'artifacts/demo/saved_comparison'; fs.mkdirSync(out, {recursive: true});
    page.on('pageerror', error => errors.push(error.message));
    const catalog = await (await page.request.get(base+'/api/catalog')).json();
    check(catalog.presets.length === 4, 'Four presets must remain available');
    for (const preset of catalog.presets) {
      check(preset.minecraft.length > 0, 'Missing source preview: '+preset.id);
      check((await page.request.get(base+preset.minecraft[0].url)).ok(), 'Source preview is unavailable');
    }
    const jobs = (await (await page.request.get(base+'/api/jobs')).json()).jobs;
    const conversions = jobs.filter(job => job.action === 'convert' && job.status === 'complete');
    check(conversions.length > 0, 'No completed conversions to check');
    for (const job of conversions) {
      await page.setViewportSize({width: 1440, height: 1100});
      await page.goto(base+'/library');
      await page.locator(`a[href="${job.view_url}"]`).first().click();
      await page.waitForURL(base+job.view_url);
      await page.locator('#result-source').evaluate(image => image.decode());
      await page.locator('#result-image').evaluate(image => image.decode());
      await page.locator('.thumbnail img').evaluateAll(images => Promise.all(images.map(image => image.decode())));
      const sourceUrl = await page.locator('#result-source').getAttribute('src');
      const left = await page.locator('#result-source').boundingBox();
      const right = await page.locator('#result-image').boundingBox();
      check(right.x >= left.x + left.width && Math.abs(right.y-left.y) < 2, 'Comparison is not side by side');
      check((await page.locator('.thumbnail.selected').getAttribute('data-kind')) === 'aerial', 'Initial comparison should use an aerial');
      await page.screenshot({path: path.join(out, job.preset+'_aerial.png')});
      const counts = {};
      for (const filter of ['ground','aerial','all']) {
        await page.locator(`[data-filter="${filter}"]`).click();
        counts[filter] = await page.locator('.thumbnail:visible').count();
        check(counts[filter] > 0, 'Missing '+filter+' views');
        await page.locator('.thumbnail:visible').last().click();
        await page.locator('#result-image').evaluate(image => image.decode());
        check(await page.locator('#result-source').getAttribute('src') === sourceUrl, 'Minecraft source changed with Isaac selection');
        check(await page.locator('#result-image-link').getAttribute('href') === await page.locator('#result-image').getAttribute('src'), 'Full-size image link differs');
      }
      check(await page.locator('.view-request button[type=submit]').isVisible(), 'Extra-view controls disappeared');
      const download = await page.request.get(base+job.download_url, {headers: {Range: 'bytes=0-15'}});
      check(download.status() === 206 && (await download.body()).subarray(0,2).toString() === 'PK', 'Saved download no longer works');
      await page.setViewportSize({width: 390, height: 844});
      check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile horizontal overflow');
      const mobileLeft = await page.locator('#result-source').boundingBox();
      const mobileRight = await page.locator('#result-image').boundingBox();
      check(mobileRight.y >= mobileLeft.y + mobileLeft.height, 'Mobile comparison must stack');
      await page.screenshot({path: path.join(out,job.preset+'_mobile.png'),fullPage: true});
      results.push({job: job.id,preset:job.preset,counts,sourceDecoded:true,desktopSideBySide:true,mobileStacked:true,downloadRange:'pass'});
    }
    check(errors.length === 0, errors.join('; '));
    const receipt = {status:'pass',at_utc:new Date().toISOString(),browser:await browser.version(),results,errors,
      presets:catalog.presets.length,planner:catalog.planner.status,scope:'Actual saved worlds in the local UI; no new conversion, model call or realism qualification'};
    fs.writeFileSync('evidence/demo/saved_comparison_browser.json',JSON.stringify(receipt,null,2)+'\n');
    console.log(JSON.stringify(receipt));
  } finally { await browser.close(); }
})().catch(error => {console.error(error.message);process.exitCode=1;});
