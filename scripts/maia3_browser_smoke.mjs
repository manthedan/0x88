#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5203;
const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';

const USAGE = `Usage: node scripts/maia3_browser_smoke.mjs [options]\n\nRuns a browser smoke for the standalone Maia3 evaluator. The page creates and disposes evaluator workers, checks edge-position legal masking, records top-5 human-policy moves and WDL probabilities, and fails on browser errors.\n\nOptions:\n  --base-url URL        Use an existing server instead of starting Vite\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --cycles N            Evaluator create/evaluate/dispose cycles (default 2)\n  --self-elo N          Maia3 self Elo (default 1500)\n  --oppo-elo N          Maia3 opponent Elo (default self Elo)\n  --style MODE          argmax or sample (default argmax)\n  --temperature X       Sampling temperature passed to the page (default 1)\n  --top-p X             Nucleus sampling top-p passed to the page (default 1)\n  --out PATH            Optional JSON artifact path\n  --no-server           Do not auto-start Vite\n  --dry-run             Print smoke URL and exit\n  -h, --help            Show this help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'base-url': { type: 'string' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      'agent-browser': { type: 'string', default: DEFAULT_AGENT_BROWSER },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      cycles: { type: 'string', default: '2' },
      'self-elo': { type: 'string', default: '1500' },
      'oppo-elo': { type: 'string' },
      style: { type: 'string', default: 'argmax' },
      temperature: { type: 'string', default: '1' },
      'top-p': { type: 'string', default: '1' },
      'ort-ep': { type: 'string' },
      'grid-size': { type: 'string' },
      model: { type: 'string' },
      out: { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.explicitBaseUrl = args.baseUrl !== undefined;
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.cycles = Number(args.cycles);
  args.selfElo = Number(args.selfElo);
  if (args.oppoElo !== undefined) args.oppoElo = Number(args.oppoElo);
  args.temperature = Number(args.temperature);
  args.topP = Number(args.topP);
  if (args.gridSize !== undefined) args.gridSize = Number(args.gridSize);
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  if (!args.oppoElo) args.oppoElo = args.selfElo;
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
    ['cycles', args.cycles],
    ['self-elo', args.selfElo],
    ['oppo-elo', args.oppoElo],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  if (!['argmax', 'sample'].includes(args.style)) throw new Error(`Invalid --style: ${args.style}`);
  return args;
}

async function closeSession(args, session) {
  try {
    await runAgent(args, ['close'], 5_000, session);
  } catch (error) {
    process.stderr.write(`[maia3-smoke] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

async function waitForText(args, session, selector, pattern, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastText = '';
  while (Date.now() < deadline) {
    const payload = await runAgent(args, ['get', 'text', selector], Math.min(10_000, timeoutMs), session).catch((error) => ({
      error: error.message ?? String(error),
    }));
    lastText = payload?.text ?? payload?.result ?? '';
    if (typeof lastText === 'string' && pattern.test(lastText)) return lastText;
    await delay(500);
  }
  throw new Error(`Timed out waiting for ${selector} to match ${pattern}; last text=${JSON.stringify(lastText).slice(0, 500)}`);
}

function normalizeEntries(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.entries)) return payload.entries;
  if (Array.isArray(payload?.messages)) return payload.messages;
  if (Array.isArray(payload?.logs)) return payload.logs;
  if (Array.isArray(payload?.result)) return payload.result;
  if (typeof payload?.text === 'string')
    return payload.text
      .split('\n')
      .filter(Boolean)
      .map((text) => ({ text }));
  return [];
}

function isActionable(entry) {
  const text = typeof entry === 'string' ? entry : JSON.stringify(entry);
  if (!text) return false;
  if (/\[vite\] connecting|\[vite\] connected|favicon\.ico/i.test(text)) return false;
  if (/ORT WebGPU session failed; falling back to WASM/i.test(text)) return false;
  return /error|pageerror|uncaught|failed|exception/i.test(text);
}

async function browserErrors(args, session) {
  const errors = [];
  for (const command of [['errors'], ['console']]) {
    try {
      const payload = await runAgent(args, command, 10_000, session);
      errors.push(...normalizeEntries(payload).filter(isActionable));
    } catch (error) {
      errors.push({ error: `${command[0]} inspection failed: ${error.message ?? error}` });
    }
  }
  return errors;
}

function smokeUrl(args) {
  const url = new URL('/lab/lc0-maia3-smoke.html', args.baseUrl);
  url.searchParams.set('cycles', String(args.cycles));
  url.searchParams.set('selfElo', String(args.selfElo));
  url.searchParams.set('oppoElo', String(args.oppoElo));
  url.searchParams.set('style', args.style);
  url.searchParams.set('temperature', String(args.temperature));
  url.searchParams.set('topP', String(args.topP));
  if (args.ortEp) url.searchParams.set('ortEp', args.ortEp);
  if (args.gridSize) url.searchParams.set('gridSize', String(args.gridSize));
  if (args.model) url.searchParams.set('model', args.model);
  return url;
}

async function runSmoke(args) {
  const session = `maia3-smoke-${process.pid}`;
  const url = smokeUrl(args);
  process.stderr.write(`[maia3-smoke] ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    await waitForText(args, session, '#status', /MAIA3_BROWSER_SMOKE_(DONE|FAILED)/, args.timeoutMs);
    const payload = await runAgent(args, ['get', 'text', '#benchResult'], 10_000, session);
    const result = JSON.parse(payload.text ?? payload.result ?? '{}');
    const errors = await browserErrors(args, session);
    result.browserErrors = errors;
    if (args.out) {
      await mkdir(dirname(args.out), { recursive: true });
      await writeFile(args.out, `${JSON.stringify(result, null, 2)}\n`);
    }
    if (!result.ok) throw new Error(`Maia3 browser smoke failed: ${(result.errors ?? []).join('; ')}`);
    if (errors.length) throw new Error(`Maia3 browser smoke had actionable browser errors: ${JSON.stringify(errors).slice(0, 1000)}`);
    return result;
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    console.log(String(smokeUrl(args)));
    return;
  }
  const server = startViteServer(args);
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
    const result = await runSmoke(args);
    console.log(JSON.stringify({ ok: true, rows: result.rows?.length ?? 0, elapsedMs: result.elapsedMs, out: args.out }, null, 2));
  } finally {
    if (server) server.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : error);
  process.exitCode = 1;
});
