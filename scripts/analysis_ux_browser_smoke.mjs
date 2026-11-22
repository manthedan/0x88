#!/usr/bin/env node
import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = process.argv[2] ?? 'http://localhost:5173';
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL, args: ['--enable-unsafe-webgpu'] });

async function assertNamedButtons(surface) {
  const unnamed = surface.getByRole('button', { name: '', exact: true });
  assert.deepEqual(await unnamed.evaluateAll((nodes) => nodes.map((node) => node.outerHTML)), []);
}

try {
  const page = await browser.newPage({ serviceWorkers: 'block', viewport: { width: 1250, height: 900 } });
  await page.addInitScript(() => {
    window.lc0Searches = [];
    window.Worker = class extends Worker {
      constructor(...args) {
        super(...args);
        this.addEventListener('message', ({ data }) => {
          if (data?.type === 'searchResult' && data.result.stats) window.lc0Searches.push(data.result.stats);
        });
      }
    };
  });
  // Local builds serve friendly paths; the R2 release serves the same bytes
  // from content-addressed /artifacts/sha256/ URLs, so hold both forms.
  const model = /\/models\/lc0\/t1-256x10-distilled-swa-2432500\.batch1\.f16\.qdq8[^/]*\.onnx$/;
  const engine = /\/(?:stockfish\/[^/]+\.wasm|artifacts\/sha256\/[a-f0-9]{64}\/)/;
  const request = Promise.withResolvers();
  const download = Promise.withResolvers();
  await page.route(
    (url) => model.test(url.pathname) || engine.test(url.pathname),
    async (route) => {
      request.resolve();
      await download.promise;
      await route.continue();
    },
  );
  try {
    await page.goto(base);
    const link = page.getByRole('link', { name: 'Analyse this position' });
    await page.goto(new URL(await link.getAttribute('href'), base).href);
    await Promise.race([request.promise, page.waitForSelector('#analysisStatus[data-state=done]').then(() => assert.fail('Model request was not held'))]);
    assert.equal(await page.locator('#analysisStatus').textContent(), 'Loading…');
    assert.equal(await page.locator('#analyze').isDisabled(), true);
    assert.deepEqual(await page.locator('#engineSummaries .pending-text').allTextContents(), ['Preparing engine…', 'Preparing engine…']);
    await assertNamedButtons(page);
    assert.ok(await page.getByRole('button', { name: 'Copy position or game' }).isVisible());
    await page.getByRole('button', { name: 'Manage engines' }).click();
    const dialog = page.getByRole('dialog', { name: 'Manage engines' });
    assert.ok(await dialog.isVisible());
    await assertNamedButtons(dialog);
    await dialog.getByRole('button', { name: 'Close engine settings' }).click();
  } finally {
    download.resolve();
  }
  const done = page.getByRole('button', { name: 'Re-analyze' });
  await done.waitFor({ timeout: 60_000 });
  const initial = await page.locator('#engineSummaries .compare-move').allTextContents();
  console.log('Live preview position:', initial.join(' / '));
  const count = await page.evaluate(() => window.lc0Searches.length);
  assert.ok(count > 0, 'Lc0 must finish on the worker');
  await done.click();
  await page.waitForFunction((count) => window.lc0Searches.length > count, count, { timeout: 60_000 });
  await done.waitFor({ timeout: 60_000 });
  const { requestedVisits, completedVisits } = await page.evaluate(() => window.lc0Searches.at(-1));
  assert.equal(requestedVisits, 400);
  assert.equal(completedVisits, requestedVisits, 'Re-analyze must search the full visit budget, not just read a reused tree');
  assert.match(await page.locator('#engineSummaries').innerText(), /400 total visits/);
  await assertNamedButtons(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('tab', { name: /^Engines/ }).click();
  await page.getByRole('button', { name: 'Board tools', exact: true }).click();
  await assertNamedButtons(page);
  console.log('Analysis UX smoke passed');
} finally {
  await browser.close();
}
