#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { basename, relative } from 'node:path';

const TEST_TIMEOUT_MS = '120000';

function usage() {
  console.log(`Usage: node scripts/run_targeted_tests.mjs [options] [tests...]

Runs a focused node:test subset. With no explicit tests, it infers likely test
files from changed paths, then falls back to a no-op so typecheck can remain
the quick gate for unrelated changes.

Options:
  --base REF       Include git diff --name-only REF...HEAD when available (default: origin/main)
  --staged        Only infer from staged files
  --tests LIST    Comma-separated test files
  --dry-run       Print the inferred command without running
  -h, --help      Show this help
`);
}

function parseArgs(argv) {
  const args = { base: 'origin/main', staged: false, dryRun: false, tests: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--base') args.base = next();
    else if (arg === '--staged') args.staged = true;
    else if (arg === '--tests') args.tests.push(...next().split(',').filter(Boolean));
    else if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
    else args.tests.push(arg);
  }
  return args;
}

function gitLines(args) {
  const result = spawnSync('git', args, { encoding: 'utf8' });
  if (result.status !== 0) return [];
  return result.stdout.split('\n').map((line) => line.trim()).filter(Boolean);
}

function snakeCase(value) {
  return value
    .replace(/\.[^.]+$/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[-\s]+/g, '_')
    .toLowerCase();
}

function changedFiles(args) {
  if (args.staged) return gitLines(['diff', '--cached', '--name-only']);
  const files = new Set([
    ...gitLines(['diff', '--name-only']),
    ...gitLines(['diff', '--cached', '--name-only']),
    ...gitLines(['ls-files', '--others', '--exclude-standard']),
  ]);
  if (args.base) for (const file of gitLines(['diff', '--name-only', `${args.base}...HEAD`])) files.add(file);
  return [...files];
}

function existingTests() {
  return new Set(readdirSync('tests').filter((file) => file.endsWith('.test.mjs')).map((file) => `tests/${file}`));
}

function inferTests(files) {
  const tests = existingTests();
  const inferred = new Set();
  const add = (...names) => {
    for (const name of names) {
      const path = name.startsWith('tests/') ? name : `tests/${name}.test.mjs`;
      if (tests.has(path)) inferred.add(path);
    }
  };
  const exact = new Map([
    ['src/lc0/analysisBrowser.ts', ['engine_catalog', 'engine_resource_broker', 'lc0_analysis_format']],
    ['src/lc0/engineCatalog.ts', ['engine_catalog']],
    ['src/lc0/resourceBroker.ts', ['engine_resource_broker']],
    ['src/lc0/recklessVariants.ts', ['lc0_reckless_variants']],
    ['src/lc0/viridithasVariants.ts', ['viridithas_variants']],
    ['src/lc0/berserkVariants.ts', ['berserk_variants']],
    ['src/lc0/plentychessVariants.ts', ['plentychess_variants']],
    ['src/lc0/stockfishEngine.ts', ['lc0_stockfish_engine']],
    ['src/lc0/gameReview.ts', ['game_review']],
  ]);
  for (const file of files) {
    if (file.startsWith('tests/') && tests.has(file)) add(file);
    for (const name of exact.get(file) ?? []) add(name);
    const stem = snakeCase(basename(file));
    add(stem, `lc0_${stem}`);
  }
  return [...inferred].sort();
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { stdio: 'inherit' });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const tests = args.tests.length ? args.tests.map((test) => relative(process.cwd(), test)) : inferTests(changedFiles(args));
  if (!tests.length) {
    console.log('No targeted tests inferred. Run npm test before commit/deploy, or pass tests explicitly.');
    return;
  }
  const command = 'node';
  const commandArgs = ['--experimental-strip-types', '--test', `--test-timeout=${TEST_TIMEOUT_MS}`, ...tests];
  if (args.dryRun) {
    console.log([command, ...commandArgs].join(' '));
    return;
  }
  run(command, commandArgs);
}

try {
  main();
} catch (error) {
  console.error(error.stack ?? error.message);
  process.exit(1);
}
