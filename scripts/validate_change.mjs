#!/usr/bin/env node
import { spawn } from 'node:child_process';

function usage() {
  console.log(`Usage: node scripts/validate_change.mjs [options] [tests...]

Codifies the fast local validation flow:
  quick: typecheck + targeted tests in parallel
  final: full npm test, optionally with a parallel build

Options:
  --mode quick|final   Validation mode (default: quick)
  --tests LIST         Comma-separated focused test files for quick mode
  --skip-typecheck     Quick mode only, run targeted tests without typecheck
  --serial             Run quick checks sequentially instead of in parallel
  --with-build         Final mode only, run build:client in parallel with npm test
  --dry-run            Print commands without running
  -h, --help           Show this help
`);
}

function parseArgs(argv) {
  const args = { mode: 'quick', tests: [], skipTypecheck: false, serial: false, withBuild: false, dryRun: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--mode') args.mode = next();
    else if (arg === '--tests') args.tests.push(...next().split(',').filter(Boolean));
    else if (arg === '--skip-typecheck') args.skipTypecheck = true;
    else if (arg === '--serial') args.serial = true;
    else if (arg === '--with-build') args.withBuild = true;
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
    else args.tests.push(arg);
  }
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
  if (args.help) { usage(); return; }
  const steps = plan(args);
  if (args.dryRun) {
    console.log(JSON.stringify({ mode: args.mode, parallel: !args.serial, steps }, null, 2));
    return;
  }
  const rows = await runSteps(steps, args.serial);
  printSummary(rows);
  if (rows.some((row) => row.status !== 0)) process.exit(1);
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
