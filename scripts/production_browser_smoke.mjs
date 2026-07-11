#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, readFile, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

const DEFAULT_BASE_URL = 'https://0x88.app';
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';
const ENGINE_FAMILIES = ['lc0', 'sf', 'reckless', 'berserk', 'viridithas', 'plentychess', 'stormphrax', 'centipawn'];
const RUNTIME_TOKENS = ['Centipawn:', 'Reckless:', 'Viridithas:', 'Berserk:', 'PlentyChess:', 'Stormphrax:'];

function usage() {
  console.log(`Usage: node scripts/production_browser_smoke.mjs [options]\n\nRuns a production browser journey across Play, Arena, and Analysis. Centipawn and Stormphrax must each complete a first move. The smoke also verifies engine order, runtime diagnostics, page errors, failed network requests, and production cache headers.\n\nOptions:\n  --base-url URL        Production origin (default ${DEFAULT_BASE_URL})\n  --agent-browser BIN   Browser CLI (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Per-engine/browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --out PATH            JSON artifact path\n  --dry-run             Print the planned journey without opening a browser\n  -h, --help            Show this help\n`);
}

function parseArgs(argv) {
  const args = { baseUrl: DEFAULT_BASE_URL, agentBrowser: DEFAULT_AGENT_BROWSER, timeoutMs: DEFAULT_TIMEOUT_MS };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base-url') args.baseUrl = next();
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--timeout') args.timeoutMs = Number(next());
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

function spawnCapture(command, commandArgs, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    let settled = false;
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`${command} ${commandArgs.join(' ')} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const output = Buffer.concat(stdout).toString('utf8').trim();
      const errors = Buffer.concat(stderr).toString('utf8').trim();
      if (status !== 0) return finish(reject, new Error(`${command} ${commandArgs.join(' ')} failed with ${status}: ${errors || output}`));
      finish(resolve, output);
    });
  });
}

async function runAgent(args, session, commandArgs, timeoutMs = 30_000) {
  const output = await spawnCapture(args.agentBrowser, ['--json', '--session', session, ...commandArgs], timeoutMs);
  const parsed = output ? JSON.parse(output) : null;
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (!parsed.success) throw new Error(`${args.agentBrowser} ${commandArgs.join(' ')} failed: ${parsed.error ?? output}`);
    return parsed.data ?? parsed;
  }
  return parsed;
}

async function closeSession(args, session) {
  try {
    await runAgent(args, session, ['close'], 10_000);
  } catch (error) {
    process.stderr.write(`[production-smoke] warning: failed to close browser session: ${error.message ?? error}\n`);
  }
}

async function evaluate(args, session, expression, timeoutMs = 30_000) {
  const payload = await runAgent(args, session, ['eval', expression], timeoutMs);
  return payload?.result ?? payload;
}

async function waitForEval(args, session, expression, validate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  let lastError;
  while (Date.now() < deadline) {
    try {
      lastValue = await evaluate(args, session, expression, Math.min(15_000, timeoutMs));
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
    const transportError = entry?.error ?? entry?.errorText ?? entry?.failure ?? entry?.failureText ?? entry?.response?.error ?? (status === 0 ? entry?.response?.statusText : undefined);
    return entry?.failed === true || (typeof transportError === 'string' && transportError.trim().length > 0);
  });
}

async function stopHarAndReadFailures(args, session, harPath) {
  await runAgent(args, session, ['network', 'har', 'stop', harPath], 30_000);
  const har = JSON.parse(await readFile(harPath, 'utf8'));
  return failedRequestEntries(har?.log?.entries ?? []);
}

async function inspectBrowser(args, session, surface) {
  const errorsPayload = await runAgent(args, session, ['errors'], 15_000);
  const consolePayload = await runAgent(args, session, ['console'], 15_000);
  const errors = [...actionableErrorEntries(errorsPayload), ...actionableErrorEntries(consolePayload)];
  if (errors.length) throw new Error(`${surface} emitted actionable browser errors: ${JSON.stringify(errors)}`);
  return { surface, errors: [] };
}

async function navigateSurface(args, session, path, fullNavigation = false) {
  const url = new URL(path, args.baseUrl);
  if (fullNavigation) {
    await runAgent(args, session, ['open', String(url)], 45_000);
    return String(url);
  }
  const clicked = await evaluate(args, session, `(() => {
    const link = document.querySelector('.site-header nav.primary a[href="${url.pathname}"]');
    if (!link) return false;
    setTimeout(() => link.click(), 0);
    return true;
  })()`);
  if (!clicked) throw new Error(`Could not find an internal link for ${url.pathname}`);
  await waitForEval(args, session, `location.pathname.endsWith('/') ? location.pathname.slice(0, -1) : location.pathname`, (value) => value === url.pathname.replace(/\/$/, ''), 30_000);
  return String(url);
}

async function playFirstMove(args, session, engine, fullNavigation = false) {
  const url = await navigateSurface(args, session, '/app/play/', fullNavigation);
  await waitForEval(args, session, `(() => ({ engine: !!document.querySelector('#engineSelect option[value="${engine}"]'), color: !!document.querySelector('#colorSelect') }))()`, (value) => value?.engine && value?.color, 60_000);
  const started = await evaluate(args, session, `(() => {
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
  })()`);
  if (started?.engine !== engine || started?.level !== '0' || started?.color !== 'black') throw new Error(`Could not start ${engine} at fastest strength as White: ${JSON.stringify(started)}`);
  const result = await waitForEval(args, session, `(() => ({
    engine: document.querySelector('#engineSelect')?.value,
    color: document.querySelector('#colorSelect')?.value,
    status: document.querySelector('#status')?.textContent?.trim() ?? '',
    moves: document.querySelector('#moveList')?.textContent?.trim() ?? ''
  }))()`, (value) => value?.engine === engine && value?.color === 'black' && value?.status.includes('Your move') && value?.moves && value.moves !== 'No moves yet', args.timeoutMs);
  await inspectBrowser(args, session, `Play/${engine}`);
  return { url, ...result };
}

async function inspectArena(args, session) {
  const url = await navigateSurface(args, session, '/app/arena/');
  const result = await waitForEval(args, session, `(() => ({
    title: document.title,
    families: [...(document.querySelector('.seat-fam')?.options ?? [])].map((option) => option.value),
    seats: [...document.querySelectorAll('.seat-fam')].map((select) => select.value)
  }))()`, (value) => ENGINE_FAMILIES.every((family) => value?.families?.includes(family)), 60_000);
  if (JSON.stringify(result.families) !== JSON.stringify(ENGINE_FAMILIES)) throw new Error(`Unexpected Arena family order: ${result.families.join(', ')}`);
  await inspectBrowser(args, session, 'Arena');
  return { url, ...result };
}

async function inspectAnalysis(args, session) {
  const url = await navigateSurface(args, session, '/app/analysis/');
  const result = await waitForEval(args, session, `(() => ({
    title: document.title,
    families: [...(document.querySelector('.row-fam')?.options ?? [])].map((option) => option.value),
    runtimeText: document.querySelector('#recklessRuntimeInfo')?.textContent ?? ''
  }))()`, (value) => {
    if (!ENGINE_FAMILIES.every((family) => value?.families?.includes(family))) return false;
    if (!RUNTIME_TOKENS.every((token) => value?.runtimeText?.includes(token))) return false;
    return !/checking asset|detecting runtime/i.test(value.runtimeText);
  }, 60_000);
  if (JSON.stringify(result.families) !== JSON.stringify(ENGINE_FAMILIES)) throw new Error(`Unexpected Analysis family order: ${result.families.join(', ')}`);
  if (/asset missing|runtime[^·]*failed|runtime[^·]*error/i.test(result.runtimeText)) throw new Error(`Analysis runtime diagnostics reported a failure: ${result.runtimeText}`);
  await inspectBrowser(args, session, 'Analysis');
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
  return hasExactDirectives(directives, ['public', 'max-age', 'stale-while-revalidate'])
    && browserMaxAge(value) === 3600
    && directives.get('stale-while-revalidate') === '86400';
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
  const session = `production-smoke-${process.pid}`;
  const harPath = join(tmpdir(), `${session}.har`);
  let harStarted = false;
  const report = {
    status: 'PRODUCTION_BROWSER_SMOKE_RUNNING',
    baseUrl: args.baseUrl,
    startedAt: new Date().toISOString(),
    play: {},
  };
  try {
    report.http = await probeHttp(args);
    await runAgent(args, session, ['open', 'about:blank'], 30_000);
    await runAgent(args, session, ['network', 'har', 'start'], 15_000);
    harStarted = true;
    report.play.centipawn = await playFirstMove(args, session, 'centipawn', true);
    report.arena = await inspectArena(args, session);
    report.analysis = await inspectAnalysis(args, session);
    report.play.stormphrax = await playFirstMove(args, session, 'stormphrax');
    report.failedRequests = await stopHarAndReadFailures(args, session, harPath);
    harStarted = false;
    if (report.failedRequests.length) throw new Error(`Production journey had failed network requests: ${JSON.stringify(report.failedRequests)}`);
    report.status = 'PRODUCTION_BROWSER_SMOKE_DONE';
    report.finishedAt = new Date().toISOString();
    return report;
  } catch (error) {
    if (harStarted) {
      try {
        report.failedRequests = await stopHarAndReadFailures(args, session, harPath);
        harStarted = false;
      } catch (harError) {
        report.harError = harError?.message ?? String(harError);
      }
    }
    report.status = 'PRODUCTION_BROWSER_SMOKE_FAILED';
    report.finishedAt = new Date().toISOString();
    report.error = { name: error?.name ?? 'Error', message: error?.message ?? String(error), stack: error?.stack };
    const failure = error instanceof Error ? error : new Error(String(error));
    failure.report = report;
    throw failure;
  } finally {
    await closeSession(args, session);
    await unlink(harPath).catch(() => {});
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
    console.log(JSON.stringify({ status: 'PRODUCTION_BROWSER_SMOKE_DRY_RUN', baseUrl: args.baseUrl, journey: ['Play/Centipawn', 'Arena', 'Analysis', 'Play/Stormphrax'] }, null, 2));
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
