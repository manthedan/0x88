#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';

function usage() {
  console.log(`Usage: npm run productization:targeted-smoke -- [options]\n       node scripts/productization_fast_gate.mjs [options]\n\nTargeted 0x88 productization smoke for runtime-audit/Centipawn browser work. This is a fast, focused check of runtime/catalog/analysis invariants plus strict Centipawn custom WebGPU smoke wiring; it is not the full shipped-path LC0 WebGPU parity gate. Keep using lc0:browser-ci-smoke for browser WebGPU parity.\n\nOptions:\n  --strict-browser-smoke   Also run real analysis/arena Centipawn strict custom WebGPU smokes\n  --agent-browser BIN      Forwarded to strict browser smoke\n  --base-url URL           Forwarded to strict browser smoke\n  --timeout MS             Forwarded to strict browser smoke\n  --out PATH               Optional JSON artifact path\n  --dry-run                Print commands without running\n  -h, --help               Show this help\n`);
}

function parseArgs(argv) {
  const args = { strictBrowserSmoke: false, dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--strict-browser-smoke') args.strictBrowserSmoke = true;
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--base-url') args.baseUrl = next();
    else if (arg === '--timeout') args.timeoutMs = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

function commandPlan(args) {
  const smokeArgs = args.strictBrowserSmoke ? [] : ['--dry-run'];
  if (args.agentBrowser) smokeArgs.push('--agent-browser', args.agentBrowser);
  if (args.baseUrl) smokeArgs.push('--base-url', args.baseUrl);
  if (args.timeoutMs) smokeArgs.push('--timeout', args.timeoutMs);
  return [
    { name: 'typecheck', command: 'npm', args: ['run', 'typecheck'] },
    { name: 'strict-smoke-syntax', command: 'node', args: ['--check', 'scripts/lc0_tiny_strict_custom_webgpu_smoke.mjs'] },
    { name: 'runtime-productization-tests', command: 'node', args: ['--experimental-strip-types', '--test', 'tests/engine_catalog.test.mjs', 'tests/lc0_analysis_format.test.mjs', 'tests/lc0_stable_backend_defaults.test.mjs'] },
    { name: args.strictBrowserSmoke ? 'centipawn-strict-custom-webgpu-smoke' : 'centipawn-strict-custom-webgpu-smoke-dry-run', command: 'npm', args: ['run', 'lc0:centipawn-strict-custom-webgpu-smoke', '--', ...smokeArgs] },
  ];
}

function run(command, commandArgs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    const started = Date.now();
    child.stdout.on('data', (chunk) => { stdout.push(chunk); process.stdout.write(chunk); });
    child.stderr.on('data', (chunk) => { stderr.push(chunk); process.stderr.write(chunk); });
    child.on('error', reject);
    child.on('close', (status) => {
      const row = { command: [command, ...commandArgs], status, elapsedMs: Date.now() - started, stdout: Buffer.concat(stdout).toString('utf8'), stderr: Buffer.concat(stderr).toString('utf8') };
      if (status !== 0) reject(Object.assign(new Error(`${command} ${commandArgs.join(' ')} failed with ${status}`), { row }));
      else resolve(row);
    });
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const plan = commandPlan(args);
  if (args.dryRun) { console.log(JSON.stringify({ schema: '0x88.targeted-productization-smoke.plan.v1', commands: plan }, null, 2)); return; }
  const rows = [];
  const startedAt = new Date().toISOString();
  try {
    for (const step of plan) rows.push({ name: step.name, ...(await run(step.command, step.args)) });
    const artifact = { schema: '0x88.targeted-productization-smoke.v1', status: 'TARGETED_PRODUCTIZATION_SMOKE_DONE', startedAt, finishedAt: new Date().toISOString(), strictBrowserSmoke: args.strictBrowserSmoke, rows };
    if (args.out) { await mkdir(dirname(args.out), { recursive: true }); await writeFile(args.out, `${JSON.stringify(artifact, null, 2)}\n`); }
    console.log(JSON.stringify(artifact, null, 2));
  } catch (error) {
    const artifact = { schema: '0x88.targeted-productization-smoke.v1', status: 'TARGETED_PRODUCTIZATION_SMOKE_FAILED', startedAt, finishedAt: new Date().toISOString(), strictBrowserSmoke: args.strictBrowserSmoke, rows, failed: error.row, error: error.message };
    if (args.out) { await mkdir(dirname(args.out), { recursive: true }); await writeFile(args.out, `${JSON.stringify(artifact, null, 2)}\n`).catch(() => undefined); }
    console.error(error.stack ?? error.message);
    process.exit(1);
  }
}

main().catch((error) => { console.error(error.stack ?? error.message); process.exit(1); });
