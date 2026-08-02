#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { parseScriptArgs } from './lib/cli.mjs';

const USAGE = `Usage: node scripts/validate_change.mjs [options] [tests...]

Codifies the fast local validation flow:
  quick: typecheck + targeted tests in parallel
  final: full npm test, optionally followed by a client build

Options:
  --mode quick|final   Validation mode (default: quick)
  --tests LIST         Comma-separated focused test files for quick mode
  --skip-typecheck     Quick mode only, run targeted tests without typecheck
  --serial             Run quick checks sequentially instead of in parallel
  --with-build         Final mode only, run build:client after npm test
  --dry-run            Print commands without running
  -h, --help           Show this help
`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      mode: { type: 'string', default: 'quick' },
      tests: { type: 'string', multiple: true },
      'skip-typecheck': { type: 'boolean', default: false },
      serial: { type: 'boolean', default: false },
      'with-build': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    allowPositionals: true,
    usage: USAGE,
  });
  args.tests = [...(args.tests ?? []).flatMap((list) => list.split(',').filter(Boolean)), ...args.positionals];
  delete args.positionals;
  if (args.mode !== 'quick' && args.mode !== 'final') throw new Error('--mode must be quick or final');
  return args;
}

function plan(args) {
  if (args.mode === 'final') {
    const steps = [{ name: 'full-test', command: 'npm', args: ['test'] }];
    if (args.withBuild) steps.push({ name: 'build-client', command: 'npm', args: ['run', 'build:client'] });
    return steps;
  }
  const steps = [];
  if (!args.skipTypecheck) steps.push({ name: 'typecheck', command: 'npm', args: ['run', 'typecheck'] });
  steps.push({ name: 'targeted-tests', command: 'node', args: ['scripts/run_targeted_tests.mjs', ...args.tests] });
  return steps;
}

function runStep(step) {
  return new Promise((resolve) => {
    const started = Date.now();
    const child = spawn(step.command, step.args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const prefix = `[${step.name}]`;
    child.stdout.on('data', (chunk) => process.stdout.write(`${prefix} ${chunk}`));
    child.stderr.on('data', (chunk) => process.stderr.write(`${prefix} ${chunk}`));
    child.on('error', (error) => resolve({ ...step, status: 1, elapsedMs: Date.now() - started, error: error.message }));
    child.on('close', (status) => resolve({ ...step, status: status ?? 1, elapsedMs: Date.now() - started }));
  });
}

async function runSteps(steps, serial) {
  const rows = [];
  if (serial) {
    for (const step of steps) {
      const row = await runStep(step);
      rows.push(row);
      if (row.status !== 0) break;
    }
    return rows;
  }
  return Promise.all(steps.map(runStep));
}

function printSummary(rows) {
  console.log('\nValidation summary:');
  for (const row of rows) {
    const seconds = (row.elapsedMs / 1000).toFixed(1);
    console.log(`- ${row.status === 0 ? 'PASS' : 'FAIL'} ${row.name} (${seconds}s)`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const steps = plan(args);
  // `npm test` runs `svelte-kit sync`, which must not mutate .svelte-kit while Vite builds it.
  const serial = args.serial || (args.mode === 'final' && args.withBuild);
  if (args.dryRun) {
    console.log(JSON.stringify({ mode: args.mode, parallel: !serial, steps }, null, 2));
    return;
  }
  const rows = await runSteps(steps, serial);
  printSummary(rows);
  if (rows.some((row) => row.status !== 0)) process.exit(1);
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
