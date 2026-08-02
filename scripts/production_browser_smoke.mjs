#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const DEFAULT_BASE_URL = 'https://0x88.app';
const DEFAULT_TIMEOUT_MS = 180_000;
const EVAL_TIMEOUT_MS = 15_000;
const ENGINE_FAMILIES = ['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess', 'stormphrax', 'centipawn'];
const RUNTIME_TOKENS = ['Centipawn:', 'Reckless:', 'Viridithas:', 'Berserk:', 'PlentyChess:', 'Stormphrax:'];

function usage() {
  console.log(
    `Usage: node scripts/production_browser_smoke.mjs [options]\n\nRuns a production browser journey across Play, Arena, and Analysis. Centipawn and Stormphrax must each complete a first move. The smoke also verifies engine order, runtime diagnostics, page errors, failed network requests, and production cache headers.\n\nRequires a Playwright Chromium build: npx playwright install chromium\n\nOptions:\n  --base-url URL        Production origin (default ${DEFAULT_BASE_URL})\n  --timeout MS          Per-engine/browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --headed              Run Chromium with a visible window (debugging)\n  --out PATH            JSON artifact path\n  --dry-run             Print the planned journey without opening a browser\n  -h, --help            Show this help\n`,
  );
}

function parseArgs(argv) {
  const args = { baseUrl: DEFAULT_BASE_URL, timeoutMs: DEFAULT_TIMEOUT_MS, headed: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base-url') args.baseUrl = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--headed') args.headed = true;
    else if (arg === '--out') args.out = next();
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  const base = new URL(args.baseUrl);
  if (!['http:', 'https:'].includes(base.protocol)) throw new Error(`Unsupported --base-url protocol: ${base.protocol}`);
  args.baseUrl = base.origin;
  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) throw new Error(`Invalid --timeout: ${args.timeoutMs}`);
  return args;
}

async function openBrowserSession(args) {
  const browser = await chromium.launch({ headless: !args.headed });
  const context = await browser.newContext();
  const page = await context.newPage();
  const session = { args, browser, page, pageErrors: [], consoleErrors: [], failedRequests: [] };
  page.on('pageerror', (error) => session.pageErrors.push({ level: 'error', text: `Uncaught ${error?.stack ?? error}` }));
  page.on('console', (message) => {
    if (message.type() === 'error') session.consoleErrors.push({ level: 'error', text: message.text() });
  });
  page.on('requestfailed', (request) => {
    // Two Chromium failure classes are not transport failures and never surfaced in
    // the previous HAR-based pipeline:
    // - net::ERR_ABORTED: the page canceled the request itself (SPA navigation
    //   abandoning background engine downloads, deduped fetches).
    // - net::ERR_CACHE_WRITE_FAILURE: disk-cache-layer flake on large brotli
    //   artifacts; the app retries/dedups and consumes the artifact anyway.
    // Genuine transport failures (DNS, connection refused, timeout) still count.
    const errorText = request.failure()?.errorText ?? 'request failed';
    if (errorText === 'net::ERR_ABORTED' || errorText === 'net::ERR_CACHE_WRITE_FAILURE') return;
    session.failedRequests.push({ url: request.url(), errorText });
  });
  page.on('response', (response) => {
    const status = response.status();
    if (status >= 400) session.failedRequests.push({ url: response.url(), status });
  });
  return session;
}

async function closeSession(session) {
  try {
    await session.browser.close();
  } catch (error) {
    process.stderr.write(`[production-smoke] warning: failed to close browser: ${error.message ?? error}\n`);
  }
}

async function evaluate(session, expression, timeoutMs = EVAL_TIMEOUT_MS) {
  let timer;
  try {
    return await Promise.race([
      session.page.evaluate(expression),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`page.evaluate timed out after ${timeoutMs}ms`)), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function waitForEval(session, expression, validate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  let lastError;
  while (Date.now() < deadline) {
    try {
      lastValue = await evaluate(session, expression, Math.min(EVAL_TIMEOUT_MS, timeoutMs));
      if (validate(lastValue)) return lastValue;
    } catch (error) {
      lastError = error;
    }
    await delay(500);
  }
  throw new Error(`Timed out waiting for browser condition. Last value=${JSON.stringify(lastValue)} Last error=${lastError?.message ?? 'none'}`);
}

function collection(payload) {
  if (Array.isArray(payload)) return payload;
  for (const key of ['entries', 'errors', 'messages', 'logs', 'requests', 'result']) {
    if (Array.isArray(payload?.[key])) return payload[key];
  }
  return [];
}

function entryText(entry) {
  return typeof entry === 'string' ? entry : JSON.stringify(entry);
}

export function actionableErrorEntries(payload) {
  return collection(payload).filter((entry) => {
    const text = entryText(entry);
    if (!text || /favicon\.ico/i.test(text)) return false;
    // WebGPU session creation may legitimately fail on headless/unsupported GPUs;
    // ORT's explicit WASM fallback is a supported production runtime.
    if (/ORT WebGPU session failed; falling back to WASM/i.test(text)) return false;
    return /error|pageerror|uncaught|failed|exception/i.test(text);
  });
}

export function failedRequestEntries(payload) {
  return collection(payload).filter((entry) => {
    const text = entryText(entry);
    if (/favicon\.ico/i.test(text)) return false;
    const status = Number(entry?.status ?? entry?.statusCode ?? entry?.response?.status);
    if (Number.isFinite(status) && status >= 400) return true;
    const transportError =
      entry?.error ??
      entry?.errorText ??
      entry?.failure ??
      entry?.failureText ??
      entry?.response?.error ??
      (status === 0 ? entry?.response?.statusText : undefined);
    return entry?.failed === true || (typeof transportError === 'string' && transportError.trim().length > 0);
  });
}

async function inspectBrowser(session, surface) {
  const errors = [...actionableErrorEntries(session.pageErrors), ...actionableErrorEntries(session.consoleErrors)];
  if (errors.length) throw new Error(`${surface} emitted actionable browser errors: ${JSON.stringify(errors)}`);
  return { surface, errors: [] };
}

async function navigateSurface(session, path, fullNavigation = false) {
  const url = new URL(path, session.args.baseUrl);
  if (fullNavigation) {
    await session.page.goto(String(url), { waitUntil: 'domcontentloaded', timeout: 45_000 });
    return String(url);
  }
  const clicked = await evaluate(
    session,
    `(() => {
    const link = document.querySelector('.site-header nav.primary a[href="${url.pathname}"]');
    if (!link) return false;
    setTimeout(() => link.click(), 0);
    return true;
  })()`,
  );
  if (!clicked) throw new Error(`Could not find an internal link for ${url.pathname}`);
  await waitForEval(
    session,
    `location.pathname.endsWith('/') ? location.pathname.slice(0, -1) : location.pathname`,
    (value) => value === url.pathname.replace(/\/$/, ''),
    30_000,
  );
  return String(url);
}

async function inspectCapabilities(session, surface) {
  const text = await waitForEval(
    session,
    `document.querySelector('[data-testid="browser-capabilities"]')?.textContent ?? ''`,
    (value) => /WebGPU (?:ready|unavailable)/.test(value) && /WASM threads (?:ready|single-thread fallback)/.test(value),
    30_000,
  );
  if (!/Model cache/.test(text)) {
    const expanded = await evaluate(
      session,
      `(() => { const panel = document.querySelector('[data-testid="browser-capabilities"]'); if (panel) panel.open = true; return panel?.textContent ?? ''; })()`,
    );
    if (!/Model cache/.test(expanded)) throw new Error(`${surface} browser capability details did not initialize`);
  }
}

async function playFirstMove(session, engine, fullNavigation = false) {
  const url = await navigateSurface(session, '/app/play/', fullNavigation);
  await inspectCapabilities(session, `Play/${engine}`);
  await waitForEval(
    session,
    `(() => ({ engine: !!document.querySelector('#engineSelect option[value="${engine}"]'), color: !!document.querySelector('#colorSelect') }))()`,
    (value) => value?.engine && value?.color,
    60_000,
  );
  // The app reverts engine changes while an engine turn is in flight (playBrowser
  // guards with ctx.engineThinking and disables the select). A restored game may
  // start thinking immediately after navigation, so wait for the select to be
  // enabled before attempting the switch.
  await waitForEval(
    session,
    `(() => { const select = document.querySelector('#engineSelect'); return !!select && !select.disabled; })()`,
    (value) => value === true,
    session.args.timeoutMs,
  );
  const started = await evaluate(
    session,
    `(() => {
    const engine = document.querySelector('#engineSelect');
    const level = document.querySelector('#levelSelect');
    const color = document.querySelector('#colorSelect');
    engine.value = '${engine}';
    engine.dispatchEvent(new Event('change', { bubbles: true }));
    level.value = '0';
    level.dispatchEvent(new Event('change', { bubbles: true }));
    color.value = 'black';
    color.dispatchEvent(new Event('change', { bubbles: true }));
    return { engine: engine.value, level: level.value, color: color.value };
  })()`,
  );
  if (started?.engine !== engine || started?.level !== '0' || started?.color !== 'black')
    throw new Error(`Could not start ${engine} at fastest strength as White: ${JSON.stringify(started)}`);
  const result = await waitForEval(
    session,
    `(() => ({
    engine: document.querySelector('#engineSelect')?.value,
    color: document.querySelector('#colorSelect')?.value,
    status: document.querySelector('#status')?.textContent?.trim() ?? '',
    moves: document.querySelector('#moveList')?.textContent?.trim() ?? ''
  }))()`,
    (value) => value?.engine === engine && value?.color === 'black' && value?.status.includes('Your move') && value?.moves && value.moves !== 'No moves yet',
    session.args.timeoutMs,
  );
  await inspectBrowser(session, `Play/${engine}`);
  return { url, ...result };
}

async function inspectArena(session) {
  const url = await navigateSurface(session, '/app/arena/');
  await inspectCapabilities(session, 'Arena');
  const result = await waitForEval(
    session,
    `(() => ({
    title: document.title,
    families: [...(document.querySelector('.seat-fam')?.options ?? [])].map((option) => option.value),
    seats: [...document.querySelectorAll('.seat-fam')].map((select) => select.value)
  }))()`,
    (value) => ENGINE_FAMILIES.every((family) => value?.families?.includes(family)),
    60_000,
  );
  if (JSON.stringify(result.families) !== JSON.stringify(ENGINE_FAMILIES)) throw new Error(`Unexpected Arena family order: ${result.families.join(', ')}`);
  await inspectBrowser(session, 'Arena');
  return { url, ...result };
}

async function inspectAnalysis(session) {
  const url = await navigateSurface(session, '/app/analysis/');
  await inspectCapabilities(session, 'Analysis');
  const result = await waitForEval(
    session,
    `(() => ({
    title: document.title,
    families: [...(document.querySelector('.row-fam')?.options ?? [])].map((option) => option.value),
    runtimeText: document.querySelector('#recklessRuntimeInfo')?.textContent ?? ''
  }))()`,
    (value) => {
      if (!ENGINE_FAMILIES.every((family) => value?.families?.includes(family))) return false;
      if (!RUNTIME_TOKENS.every((token) => value?.runtimeText?.includes(token))) return false;
      return !/checking asset|detecting runtime/i.test(value.runtimeText);
    },
    60_000,
  );
  if (JSON.stringify(result.families) !== JSON.stringify(ENGINE_FAMILIES)) throw new Error(`Unexpected Analysis family order: ${result.families.join(', ')}`);
  if (/asset missing|runtime[^·]*failed|runtime[^·]*error/i.test(result.runtimeText))
    throw new Error(`Analysis runtime diagnostics reported a failure: ${result.runtimeText}`);
  await inspectBrowser(session, 'Analysis');
  return { url, ...result };
}

export function parseCacheControl(value) {
  const directives = new Map();
  const duplicates = new Set();
  for (const part of value.split(',')) {
    const [rawName, ...rawValue] = part.trim().split('=');
    if (!rawName) continue;
    const name = rawName.toLowerCase();
    const directiveValue = rawValue.join('=').trim().replace(/^"|"$/g, '');
    if (directives.has(name)) duplicates.add(name);
    directives.set(name, directiveValue || true);
  }
  Object.defineProperty(directives, 'duplicates', { value: duplicates, enumerable: false });
  return directives;
}

export function browserMaxAge(value) {
  const raw = parseCacheControl(value).get('max-age');
  if (raw === undefined || raw === true || !/^\d+$/.test(raw)) return null;
  return Number(raw);
}

function hasExactDirectives(directives, names) {
  return directives.duplicates.size === 0 && directives.size === names.length && names.every((name) => directives.has(name));
}

export function isExpectedAppShellCacheControl(value) {
  const directives = parseCacheControl(value);
  return hasExactDirectives(directives, ['public', 'max-age', 'must-revalidate']) && browserMaxAge(value) === 0;
}

export function isExpectedOrtCacheControl(value) {
  const directives = parseCacheControl(value);
  return (
    hasExactDirectives(directives, ['public', 'max-age', 'stale-while-revalidate']) &&
    browserMaxAge(value) === 3600 &&
    directives.get('stale-while-revalidate') === '86400'
  );
}

async function fetchWithRetries(url, attempts = 3) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fetch(url, {
        cache: 'no-store',
        headers: { 'user-agent': '0x88-production-smoke/1.0' },
        signal: AbortSignal.timeout(30_000),
      });
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await delay(attempt * 1_000);
    }
  }
  throw lastError;
}

async function probeHttp(args) {
  const checks = [
    { path: '/', cache: isExpectedAppShellCacheControl },
    // /ort/* is a stable, replaceable path, so it intentionally uses the exact
    // short revalidation policy rather than content-addressed `immutable` caching.
    { path: '/ort/ort-wasm-simd-threaded.asyncify.wasm', cache: isExpectedOrtCacheControl },
  ];
  const results = [];
  for (const check of checks) {
    const url = new URL(check.path, args.baseUrl);
    const response = await fetchWithRetries(url);
    const cacheControl = response.headers.get('cache-control') ?? '';
    await response.body?.cancel();
    if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
    if (!check.cache(cacheControl)) throw new Error(`${url} returned unexpected Cache-Control: ${cacheControl || '(missing)'}`);
    results.push({ url: String(url), status: response.status, cacheControl, contentLength: response.headers.get('content-length') });
  }
  return results;
}

async function runSmoke(args) {
  const report = {
    status: 'PRODUCTION_BROWSER_SMOKE_RUNNING',
    baseUrl: args.baseUrl,
    startedAt: new Date().toISOString(),
    play: {},
  };
  let session;
  try {
    report.http = await probeHttp(args);
    session = await openBrowserSession(args);
    report.play.centipawn = await playFirstMove(session, 'centipawn', true);
    report.arena = await inspectArena(session);
    report.analysis = await inspectAnalysis(session);
    report.play.stormphrax = await playFirstMove(session, 'stormphrax');
    report.failedRequests = failedRequestEntries(session.failedRequests);
    if (report.failedRequests.length) throw new Error(`Production journey had failed network requests: ${JSON.stringify(report.failedRequests)}`);
    report.status = 'PRODUCTION_BROWSER_SMOKE_DONE';
    report.finishedAt = new Date().toISOString();
    return report;
  } catch (error) {
    report.failedRequests = failedRequestEntries(session?.failedRequests ?? []);
    report.status = 'PRODUCTION_BROWSER_SMOKE_FAILED';
    report.finishedAt = new Date().toISOString();
    report.error = { name: error?.name ?? 'Error', message: error?.message ?? String(error), stack: error?.stack };
    const failure = error instanceof Error ? error : new Error(String(error));
    failure.report = report;
    throw failure;
  } finally {
    if (session) await closeSession(session);
  }
}

async function writeArtifact(path, artifact) {
  if (!path) return;
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(artifact, null, 2)}\n`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  if (args.dryRun) {
    console.log(
      JSON.stringify(
        { status: 'PRODUCTION_BROWSER_SMOKE_DRY_RUN', baseUrl: args.baseUrl, journey: ['Play/Centipawn', 'Arena', 'Analysis', 'Play/Stormphrax'] },
        null,
        2,
      ),
    );
    return;
  }
  try {
    const artifact = await runSmoke(args);
    await writeArtifact(args.out, artifact);
    console.log(JSON.stringify(artifact, null, 2));
  } catch (error) {
    const artifact = error.report ?? {
      status: 'PRODUCTION_BROWSER_SMOKE_FAILED',
      baseUrl: args.baseUrl,
      finishedAt: new Date().toISOString(),
      error: { name: error?.name ?? 'Error', message: error?.message ?? String(error), stack: error?.stack },
    };
    await writeArtifact(args.out, artifact);
    console.error(error?.stack ?? error);
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? error);
    process.exitCode = 1;
  });
}
