#!/usr/bin/env node
import assert from 'node:assert/strict';
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { extname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  actionableIntegrityErrors,
  integrityEngineSpecs,
  integritySmokeExitCode,
  integrityUiAssets,
  missingIntegrityAssets,
  parseIntegritySmokeArgs,
} from './lib/engine_integrity_smoke.mjs';

const args = parseIntegritySmokeArgs(process.argv.slice(2));
const repo = process.cwd();
// A persistent profile is needed to exercise disk caching of 99 MB artifacts;
// incognito contexts only have a small memory cache. Keep all output local.
const scratch = resolve(process.env.ENGINE_INTEGRITY_BROWSER_SCRATCH ?? '.local-dev-artifacts/engine-integrity-browser');
mkdirSync(scratch, { recursive: true });
const dist = args.dist;
const require = createRequire(join(repo, 'package.json'));
const { build } = await import(pathToFileURL(require.resolve('vite')));
const { chromium } = require('playwright');
const timeoutMs = 180000;
const temp = mkdtempSync(join(scratch, 'harness-'));
const specs = integrityEngineSpecs;
function assetFile(path) {
  const stagedFile = join(scratch, 'staged', path.split('/').at(-1));
  return path.startsWith('/reckless/') && existsSync(stagedFile) ? stagedFile : join(dist, path);
}
const source = (name) => JSON.stringify(join(repo, 'src', name));
// Compile the unmodified real Analysis prefetch function with inert UI progress
// callbacks. This checks the exact production fetch options and body drain.
const analysis = readFileSync(join(repo, 'src/lc0/analysisBrowser.ts'), 'utf8');
const prefetchSource = analysis.slice(
  analysis.indexOf('function createAnalysisAssetPreload('),
  analysis.indexOf('async function waitForAnalysisAssetPreload('),
);
assert.match(prefetchSource, /fetch\(rawUrl/);
writeFileSync(
  join(temp, 'entry.ts'),
  `
import { START_FEN, parseFen } from ${source('chess/board.ts')};
import { legalMoves } from ${source('chess/movegen.ts')};
import { moveToUci } from ${source('chess/moveCodec.ts')};
import { StockfishEngine } from ${source('lc0/stockfishEngine.ts')};
import { RecklessEngine } from ${source('lc0/recklessEngine.ts')};
import { ViridithasEngine } from ${source('lc0/viridithasEngine.ts')};
import { BerserkEngine } from ${source('lc0/berserkEngine.ts')};
import { PlentyChessEngine } from ${source('lc0/plentychessEngine.ts')};
import { StormphraxEngine } from ${source('lc0/stormphraxEngine.ts')};
let analysisAssetPreloadSequence = 0;
const preloadedAnalysisAssetUrls = new Set();
const preloadingAnalysisAssetPromises = new Map();
const assetFileLabel = (url) => url;
const showLoadingProgressItem = () => {};
const hideLoadingProgressItem = () => {};
${prefetchSource}
export async function prefetch(spec) {
  for (const path of spec.assets) await createAnalysisAssetPreload(spec.name, (globalThis.LC0_BROWSER_ASSET_BASE_URL ?? '') + path).promise;
}
export async function run(spec, timeoutMs) {
  const start = performance.now();
  const options = { depth: 1, hashMb: 16, threads: spec.threads ?? 1 };
  const base = (globalThis.LC0_BROWSER_ASSET_BASE_URL ?? '') + spec.base;
  let engine;
  if (spec.family === 'sf') engine = new StockfishEngine(options, base + '.js');
  else if (spec.family === 'reckless') engine = new RecklessEngine(options, base + '.wasm', { forceOneShot: true });
  else if (spec.family === 'viridithas') engine = new ViridithasEngine(options, base + '.wasm', { forceOneShot: true });
  else {
    const Engine = { berserk: BerserkEngine, plentychess: PlentyChessEngine, stormphrax: StormphraxEngine }[spec.family];
    engine = new Engine(options); // exercise default sibling .data loading
  }
  try {
    const move = await engine.bestMove(START_FEN, AbortSignal.timeout(timeoutMs));
    if (!legalMoves(parseFen(START_FEN)).map(moveToUci).includes(move)) throw Error('Illegal best move: ' + move);
    return { move, ms: Math.round(performance.now() - start) };
  } catch (error) { return { error: error.message }; }
  finally { engine.dispose(); }
}
`,
);
const instrumentation = `
(() => {
const audit = (kind) => self.postMessage({ __integrityAuditExecution: kind });
for (const key of ['compile', 'compileStreaming', 'instantiate', 'instantiateStreaming']) {
  const original = WebAssembly[key];
  if (original) WebAssembly[key] = function(...args) { audit(key); return original.apply(WebAssembly, args); };
}
if (self.importScripts) {
  const original = self.importScripts;
  self.importScripts = function(...args) { audit('importScripts'); return original.apply(self, args); };
}
})();\n`;
let outputs;
let tamperPath;
let tampered = 0;
let failures = 0;
let passed = 0;
let skipped = 0;
const requests = [];
const servers = [];
async function check(label, assets, task) {
  const missing = missingIntegrityAssets(assets, assetFile);
  if (missing.length) {
    skipped++;
    console.log(`SKIPPED ${label}: missing artifacts: ${missing.join(', ')}`);
    return;
  }
  try {
    const detail = await task();
    passed++;
    console.log(`PASS ${label}${detail ? ': ' + detail : ''}`);
  } catch (error) {
    failures++;
    console.log(`FAIL ${label}: ${error.stack ?? error}`);
  }
}
async function server() {
  const server = createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname;
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Timing-Allow-Origin', '*');
    res.setHeader('Vary', 'Origin');
    const type = {
      '.js': 'text/javascript',
      '.wasm': 'application/wasm',
      '.html': 'text/html',
      '.json': 'application/json',
      '.css': 'text/css',
      '.svg': 'image/svg+xml',
    };
    res.setHeader('Content-Type', type[extname(path)] ?? 'application/octet-stream');
    if (path.startsWith('/__integrity__/')) {
      const key = path.slice('/__integrity__/'.length);
      res.setHeader('Cache-Control', key ? 'public, max-age=3600' : 'no-cache');
      if (!key) {
        res.setHeader('Content-Type', 'text/html');
        return res.end('<!doctype html><title>Integrity check</title>');
      }
      assert.ok(outputs.has(key), `Unknown harness output: ${key}`);
      let body = outputs.get(key);
      if (/recklessWasiWorker|recklessBrowserApiWorker/.test(key)) body = instrumentation + body;
      return res.end(body);
    }
    let file = assetFile(decodeURIComponent(path));
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html');
    if (!file.startsWith(dist + sep) && !file.startsWith(join(scratch, 'staged') + sep)) {
      res.statusCode = 403;
      return res.end('Forbidden');
    }
    if (!existsSync(file)) {
      res.statusCode = 404;
      return res.end('Not found');
    }
    res.setHeader('Content-Type', type[extname(file)] ?? 'application/octet-stream');
    res.setHeader('Cache-Control', path.startsWith('/_app/') ? 'public, max-age=31536000, immutable' : 'public, max-age=3600');
    const size = statSync(file).size;
    res.setHeader('Content-Length', size);
    if (req.method === 'HEAD') return res.end();
    requests.push({ path, size, origin: req.headers.origin, mode: req.headers['sec-fetch-mode'], cookie: req.headers.cookie ?? null });
    if (path === tamperPath) {
      const bytes = readFileSync(file);
      bytes[bytes.length - 1] ^= 1;
      tampered++;
      return res.end(bytes);
    }
    createReadStream(file).pipe(res);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  servers.push(server);
  return `http://127.0.0.1:${server.address().port}`;
}
async function context(assetBase) {
  const context = await chromium.launchPersistentContext(mkdtempSync(join(temp, 'profile-')), {
    headless: true,
    serviceWorkers: 'block',
    args: ['--disk-cache-size=2147483647'],
  });
  await context.addInitScript(
    ({ instrumentation, assetBase }) => {
      if (assetBase) globalThis.LC0_BROWSER_ASSET_BASE_URL = assetBase;
      globalThis.__integrityExecution = [];
      const NativeBlob = Blob;
      globalThis.Blob = class extends NativeBlob {
        constructor(parts, options) {
          super(options?.type?.includes('javascript') ? [instrumentation, ...parts] : parts, options);
        }
      };
      const NativeWorker = Worker;
      globalThis.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          this.addEventListener('message', (event) => {
            if (event.data?.__integrityAuditExecution) globalThis.__integrityExecution.push(event.data.__integrityAuditExecution);
          });
        }
      };
    },
    { instrumentation, assetBase },
  );
  return context;
}
function errorsFor(page, requiredAssets) {
  const errors = [];
  const optionalMissingProbeUrls = new Set();
  const fetchedUrls = new Set();
  page.on('request', (request) => {
    if (request.method() === 'GET') fetchedUrls.add(request.url());
  });
  page.on('response', (response) => {
    const path = new URL(response.url()).pathname;
    if (requiredAssets && !requiredAssets.includes(path) && response.request().method() === 'HEAD' && [404, 410].includes(response.status())) {
      optionalMissingProbeUrls.add(response.url());
    }
  });
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push({ kind: 'console', text: message.text(), url: message.location().url });
  });
  page.on('pageerror', (error) => errors.push({ kind: 'pageerror', text: error.message }));
  return {
    actionable: () => actionableIntegrityErrors(errors, optionalMissingProbeUrls, fetchedUrls),
    optionalMissingProbeUrls,
  };
}
const run = (page, spec) => page.evaluate(async ({ spec, timeoutMs }) => (await import('/__integrity__/entry.js')).run(spec, timeoutMs), { spec, timeoutMs });
const network = (start, spec) => requests.slice(start).filter((r) => spec.assets.includes(r.path));
try {
  const bundle = await build({
    configFile: false,
    root: repo,
    envDir: false,
    envPrefix: '__INTEGRITY_UNUSED_',
    publicDir: false,
    base: '/__integrity__/',
    logLevel: 'error',
    build: {
      write: false,
      minify: true,
      rollupOptions: {
        input: join(temp, 'entry.ts'),
        preserveEntrySignatures: 'strict',
        output: { entryFileNames: 'entry.js', chunkFileNames: 'chunks/[name]-[hash].js', assetFileNames: 'assets/[name]-[hash][extname]' },
      },
    },
  });
  outputs = new Map(
    (Array.isArray(bundle) ? bundle.flatMap((b) => b.output) : bundle.output).map((item) => [item.fileName, item.type === 'chunk' ? item.code : item.source]),
  );
  assert.match(outputs.get('entry.js'), /export\{/);
  console.log(`PASS harness: preserved run/prefetch exports; ${outputs.size} real Vite/worker outputs; ephemeral servers; no Playwright routing`);
  const origin = await server();
  const assetBase = await server();
  for (const spec of args.uiOnly ? [] : specs) {
    await check(`${spec.name} legal moves, clean console, HTTP cache`, spec.assets, async () => {
      const ctx = await context();
      try {
        const page = await ctx.newPage();
        const errors = errorsFor(page);
        await page.goto(origin + '/__integrity__/');
        assert.equal(await page.evaluate(() => crossOriginIsolated), true);
        const before = requests.length;
        const cold = await run(page, spec);
        assert.ok(cold.move, JSON.stringify(cold));
        const coldNetwork = network(before, spec);
        const warmBefore = requests.length;
        const warm = await run(page, spec);
        assert.ok(warm.move, JSON.stringify(warm));
        const warmNetwork = network(warmBefore, spec);
        assert.equal(warmNetwork.length, 0, JSON.stringify(warmNetwork));
        await ctx.setOffline(true);
        const offline = await run(page, spec);
        assert.ok(offline.move, JSON.stringify(offline));
        assert.deepEqual(network(warmBefore, spec), []);
        assert.deepEqual(errors.actionable(), []);
        return `cold ${cold.ms}ms/${coldNetwork.length} GETs/${coldNetwork.reduce((n, r) => n + r.size, 0)}B; warm ${warm.ms}ms/0 GETs/0B; offline ${offline.ms}ms; ${cold.move},${warm.move}`;
      } finally {
        await ctx.close();
      }
    });
    await check(`${spec.name} tamper rejected before execution; clean retry`, spec.assets, async () => {
      const ctx = await context();
      try {
        const page = await ctx.newPage();
        const errors = errorsFor(page);
        await page.goto(origin + '/__integrity__/');
        tamperPath = spec.wasm;
        tampered = 0;
        const result = await run(page, spec);
        assert.equal(tampered, 2, 'initial response and single reload both tampered');
        assert.match(result.error ?? '', /Engine artifact integrity check failed.*SHA-256 mismatch.*Retry/);
        assert.deepEqual(await page.evaluate(() => globalThis.__integrityExecution), []);
        tamperPath = undefined;
        const retry = await run(page, spec);
        assert.ok(retry.move, JSON.stringify(retry));
        assert.ok((await page.evaluate(() => globalThis.__integrityExecution)).length);
        assert.deepEqual(errors.actionable(), []);
        return `2 corrupt GETs, zero engine execution; retry ${retry.move}`;
      } finally {
        tamperPath = undefined;
        await ctx.close();
      }
    });
  }
  for (const crossOrigin of args.uiOnly ? [] : [false, true]) {
    const spec = specs.find((s) => s.family === 'stormphrax');
    await check(`real Analysis prefetch shares ${crossOrigin ? 'cross-origin' : 'same-origin'} cache with adapter`, spec.assets, async () => {
      const ctx = await context(crossOrigin ? assetBase : undefined);
      try {
        const page = await ctx.newPage();
        const errors = errorsFor(page);
        await page.goto(origin + '/__integrity__/');
        const before = requests.length;
        await page.evaluate(async (spec) => (await import('/__integrity__/entry.js')).prefetch(spec), spec);
        const preload = network(before, spec);
        assert.equal(preload.length, spec.assets.length);
        const loaded = requests.length;
        const result = await run(page, spec);
        assert.ok(result.move, JSON.stringify(result));
        assert.deepEqual(network(loaded, spec), []);
        assert.deepEqual(errors.actionable(), []);
        return `${preload.length} prefetch GETs/${preload.reduce((n, r) => n + r.size, 0)}B; adapter 0 GETs/0B (${result.ms}ms); mode=${preload[0].mode}, credentials=same-origin`;
      } finally {
        await ctx.close();
      }
    });
  }
  for (const path of ['/app/play/', '/app/arena/']) {
    await check(`real UI ${path} ${path.includes('arena') ? 'one-game CPU match' : 'Stockfish opponent move'}`, integrityUiAssets(path), async () => {
      const ctx = await context();
      try {
        await ctx.addInitScript(() => localStorage.setItem('0x88-play-preferences-v1', JSON.stringify({ engineId: 'sf-lite', level: 0, color: 'white' })));
        const page = await ctx.newPage();
        const errors = errorsFor(page, integrityUiAssets(path));
        await page.goto(origin + path + '?seatA=sf:lite:1&seatB=sf:lite:1&sfThreads=1&delayMs=0&berserkVariant=emscripten');
        await page.waitForFunction(
          (arena) => (arena ? !!document.querySelector('.seat-fam option') : !!document.querySelector('#engineSelect option')),
          path.includes('arena'),
          { timeout: timeoutMs },
        );
        let detail;
        if (path.includes('play')) {
          await page.locator('#engineSelect').selectOption('sf-lite');
          await page.locator('#newGame').click();
          await page.locator('#moveInput').waitFor({ state: 'visible' });
          await page.waitForFunction(() => document.querySelector('#status').textContent === 'Your move', null, { timeout: timeoutMs });
          await page.locator('#moveInput').fill('e2e4');
          await page.locator('#moveInput').press('Enter');
          await page.waitForFunction(() => document.querySelectorAll('[data-ply]').length >= 2, null, { timeout: timeoutMs });
          detail = await page.locator('#status').innerText();
        } else {
          await page.locator('.seat-fam').nth(0).selectOption('sf');
          await page.locator('.seat-fam').nth(1).selectOption('berserk');
          await page.locator('.seat-var').nth(1).selectOption('emscripten');
          await page.locator('#budgetModeSelect').selectOption('fixed');
          await page.locator('.seat-strength').nth(0).fill('1');
          await page.locator('.seat-strength').nth(1).fill('1');
          await page.locator('#gamesInput').fill('1');
          await page.locator('#budgetModeSelect').selectOption('fixed');
          await page.locator('#start').click();
          await page.waitForFunction(() => document.querySelectorAll('#gameMoves [data-idx]').length >= 2, null, { timeout: timeoutMs });
          await page.waitForFunction(() => document.querySelector('#main').dataset.match === 'final', null, { timeout: timeoutMs });
          detail = await page.locator('#pairing').innerText();
          assert.equal(detail, '1 of 1 game played');
        }
        writeFileSync(join(scratch, path.includes('arena') ? 'arena-ui.txt' : 'play-ui.txt'), await page.locator('body').innerText());
        assert.deepEqual(errors.actionable(), []);
        return `${detail}${errors.optionalMissingProbeUrls.size ? `; ${errors.optionalMissingProbeUrls.size} optional HEAD 404/410s` : ''}`;
      } finally {
        await ctx.close();
      }
    });
  }
} catch (error) {
  failures++;
  console.log(`FAIL runner: ${error.stack ?? error}`);
} finally {
  for (const server of servers) await new Promise((resolve) => server.close(resolve));
  rmSync(temp, { recursive: true, force: true });
}
console.log(`RESULT ${passed} passed, ${failures} failed, ${skipped} skipped${args.requireAll ? ' (--require-all)' : ''}`);
process.exitCode = integritySmokeExitCode({ failures, skipped }, args.requireAll);
