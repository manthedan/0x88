#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium } from 'playwright';
import { parseScriptArgs } from './lib/cli.mjs';
import { waitForOutput } from './lib/server.mjs';
import { actionableErrorEntries, failedRequestEntries } from './production_browser_smoke.mjs';

const DEFAULT_PORT = 5182;
const DEFAULT_TIMEOUT_MS = 45_000;
// Settle time after a page reports ready, so late init errors still surface.
const SETTLE_MS = 2_000;
const CAPABILITIES = '[data-testid="browser-capabilities"]';
const PAGES = [
  { path: '/', ready: `!!document.querySelector('.site-header nav.primary a[href="/app/play/"]')` },
  { path: '/app/play/', ready: capabilitiesReady() },
  { path: '/app/analysis/', ready: capabilitiesReady() },
  { path: '/app/arena/', ready: capabilitiesReady() },
];

const USAGE = `Usage: node scripts/dist_browser_smoke.mjs [options]

Serves a built static client with scripts/serve_isolated_static.mjs (COOP/COEP)
and loads Home, Play, Analysis and Arena in headless Chromium. Fails on page
errors, actionable console errors, failed same-origin requests, or a page that
never initializes. Needs no GPU: it does not start an engine.

Requires a Playwright Chromium build: npx playwright install chromium

Options:
  --dist DIR       Built client directory (default dist-client)
  --port N         Static server port (default ${DEFAULT_PORT})
  --timeout MS     Per-page initialization timeout (default ${DEFAULT_TIMEOUT_MS})
  --out PATH       JSON artifact path
  -h, --help       Show this help
`;

function capabilitiesReady() {
  return `/WebGPU (?:ready|unavailable)/.test(document.querySelector('${CAPABILITIES}')?.textContent ?? '')`;
}

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      dist: { type: 'string', default: 'dist-client' },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      out: { type: 'string' },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

function startStaticServer(args) {
  const server = spawn(process.execPath, ['scripts/serve_isolated_static.mjs', resolve(args.dist), String(args.port)], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.ready = waitForOutput(server, { match: /listening|http:\/\//i, timeoutMs: 15_000, label: 'static server' });
  return server;
}

async function checkPage(page, origin, spec, timeoutMs) {
  const pageErrors = [];
  const consoleErrors = [];
  const failedRequests = [];
  const onPageError = (error) => pageErrors.push({ level: 'error', text: `Uncaught ${error?.stack ?? error}` });
  const onConsole = (message) => {
    if (message.type() === 'error') consoleErrors.push({ level: 'error', text: message.text() });
  };
  // Only this server's responses: engine and model bytes come from the artifact
  // host, which is production's concern, not the build's.
  const onRequestFailed = (request) => {
    const errorText = request.failure()?.errorText ?? 'request failed';
    if (request.url().startsWith(origin) && errorText !== 'net::ERR_ABORTED') failedRequests.push({ url: request.url(), errorText });
  };
  const onResponse = (response) => {
    if (response.url().startsWith(origin) && response.status() >= 400) failedRequests.push({ url: response.url(), status: response.status() });
  };
  page.on('pageerror', onPageError);
  page.on('console', onConsole);
  page.on('requestfailed', onRequestFailed);
  page.on('response', onResponse);
  const started = Date.now();
  try {
    const response = await page.goto(`${origin}${spec.path}`, { waitUntil: 'load', timeout: timeoutMs });
    if (!response?.ok()) throw new Error(`${spec.path} returned HTTP ${response?.status()}`);
    await page.waitForFunction(spec.ready, undefined, { timeout: timeoutMs, polling: 250 }).catch(() => {
      throw new Error(`${spec.path} did not initialize within ${timeoutMs}ms; errors: ${JSON.stringify([...pageErrors, ...consoleErrors])}`);
    });
    const crossOriginIsolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);
    if (!crossOriginIsolated) throw new Error(`${spec.path} is not cross-origin isolated`);
    await delay(SETTLE_MS);
    const errors = [...actionableErrorEntries(pageErrors), ...actionableErrorEntries(consoleErrors)];
    if (errors.length) throw new Error(`${spec.path} emitted actionable browser errors: ${JSON.stringify(errors)}`);
    const failed = failedRequestEntries(failedRequests);
    if (failed.length) throw new Error(`${spec.path} had failed requests: ${JSON.stringify(failed)}`);
    return { path: spec.path, status: 'ok', ms: Date.now() - started };
  } finally {
    page.off('pageerror', onPageError);
    page.off('console', onConsole);
    page.off('requestfailed', onRequestFailed);
    page.off('response', onResponse);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const origin = `http://127.0.0.1:${args.port}`;
  const server = startStaticServer(args);
  const report = { status: 'DIST_BROWSER_SMOKE_RUNNING', dist: args.dist, origin, startedAt: new Date().toISOString(), pages: [] };
  let browser;
  try {
    await server.ready;
    browser = await chromium.launch();
    // A fresh context per page: no service-worker or storage carry-over hides a cold-load failure.
    for (const spec of PAGES) {
      const context = await browser.newContext();
      try {
        report.pages.push(await checkPage(await context.newPage(), origin, spec, args.timeoutMs));
      } finally {
        await context.close();
      }
    }
    report.status = 'DIST_BROWSER_SMOKE_DONE';
  } catch (error) {
    report.status = 'DIST_BROWSER_SMOKE_FAILED';
    report.error = error?.message ?? String(error);
  } finally {
    await browser?.close();
    server.kill('SIGTERM');
  }
  report.finishedAt = new Date().toISOString();
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, `${JSON.stringify(report, null, 2)}\n`);
  }
  console.log(JSON.stringify(report, null, 2));
  if (report.status !== 'DIST_BROWSER_SMOKE_DONE') process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exitCode = 1;
});
