#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { readdirSync } from 'node:fs';

const DEFAULT_TIMEOUT_MS = '120000';

function usage() {
  console.log(`Usage: node scripts/test_timings.mjs [options]

Runs test files one at a time and reports the slowest files. Use this when the
full suite feels slow and we need data before splitting or optimizing tests.

Options:
  --pattern REGEX   Only run matching test file paths
  --limit N         Stop after N matching files
  --top N           Number of slowest files to print (default: 15)
  --timeout MS      Per-file node:test timeout (default: 120000)
  --dry-run         Print selected files without running
  -h, --help        Show this help
`);
}

function parseArgs(argv) {
  const args = { top: 15, timeout: DEFAULT_TIMEOUT_MS, dryRun: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--pattern') args.pattern = next();
    else if (arg === '--limit') args.limit = Number(next());
    else if (arg === '--top') args.top = Number(next());
    else if (arg === '--timeout') args.timeout = next();
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

function selectedFiles(args) {
  let files = readdirSync('tests')
    .filter((file) => file.endsWith('.test.mjs'))
    .map((file) => `tests/${file}`)
    .sort();
  if (args.pattern) {
    const re = new RegExp(args.pattern);
    files = files.filter((file) => re.test(file));
  }
  if (Number.isFinite(args.limit)) files = files.slice(0, args.limit);
  return files;
}

function runFile(file, timeout) {
  return new Promise((resolve) => {
    const started = Date.now();
    const child = spawn('node', ['--experimental-strip-types', '--test', `--test-timeout=${timeout}`, file], { stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', (chunk) => process.stdout.write(chunk));
    child.stderr.on('data', (chunk) => process.stderr.write(chunk));
    child.on('error', (error) => resolve({ file, status: 1, elapsedMs: Date.now() - started, error: error.message }));
    child.on('close', (status) => resolve({ file, status: status ?? 1, elapsedMs: Date.now() - started }));
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const files = selectedFiles(args);
  if (args.dryRun) {
    console.log(files.join('\n'));
    return;
  }
  const rows = [];
  for (const file of files) {
    console.log(`\n[test-timings] ${file}`);
    rows.push(await runFile(file, args.timeout));
  }
  const sorted = [...rows].sort((a, b) => b.elapsedMs - a.elapsedMs).slice(0, args.top);
  console.log('\nSlowest test files:');
  for (const row of sorted) {
    console.log(`${(row.elapsedMs / 1000).toFixed(1)}s ${row.status === 0 ? 'PASS' : 'FAIL'} ${row.file}`);
  }
  if (rows.some((row) => row.status !== 0)) process.exit(1);
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
