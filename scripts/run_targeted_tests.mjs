#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { relative } from 'node:path';
import { parseScriptArgs } from './lib/cli.mjs';
import { buildImportGraph, formatUnmatched, selectTargetedTests } from './lib/targeted_tests.mjs';

const TEST_TIMEOUT_MS = '120000';

const USAGE = `Usage: node scripts/run_targeted_tests.mjs [options] [tests...]

Runs a focused node:test subset. With no explicit tests, it selects tests from
the import graph of tests/*.test.mjs and the eval/ scripts they exec, plus
filename and family mappings (search/PUCT, engine adapters, wasm, cloudflare).

An executable source change that matches nothing fails closed. It does not
exit 0. Pass tests explicitly, or use --on-unmatched full to run npm test.
Docs and other non-executable paths still select nothing and exit 0.

Options:
  --base REF            Include git diff --name-only REF...HEAD (default: origin/main)
  --staged              Only infer from staged files
  --files LIST          Comma-separated changed paths to infer from, instead of git
  --tests LIST          Comma-separated test files (skips inference)
  --on-unmatched MODE   fail (default) or full
  --dry-run             Print the inferred command without running
  -h, --help            Show this help
`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      base: { type: 'string', default: 'origin/main' },
      staged: { type: 'boolean', default: false },
      files: { type: 'string', multiple: true },
      tests: { type: 'string', multiple: true },
      'on-unmatched': { type: 'string', default: 'fail' },
      'dry-run': { type: 'boolean', default: false },
    },
    allowPositionals: true,
    usage: USAGE,
  });
  args.tests = [...(args.tests ?? []).flatMap((list) => list.split(',').filter(Boolean)), ...args.positionals];
  args.files = (args.files ?? []).flatMap((list) => list.split(',').filter(Boolean));
  delete args.positionals;
  if (args.onUnmatched !== 'fail' && args.onUnmatched !== 'full') throw new Error('--on-unmatched must be fail or full');
  return args;
}

function gitLines(args) {
  const result = spawnSync('git', args, { encoding: 'utf8' });
  if (result.status !== 0) return [];
  return result.stdout
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

function changedFiles(args) {
  if (args.files.length) return args.files;
  if (args.staged) return gitLines(['diff', '--cached', '--name-only']);
  const files = new Set([
    ...gitLines(['diff', '--name-only']),
    ...gitLines(['diff', '--cached', '--name-only']),
    ...gitLines(['ls-files', '--others', '--exclude-standard']),
  ]);
  if (args.base) for (const file of gitLines(['diff', '--name-only', `${args.base}...HEAD`])) files.add(file);
  return [...files];
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { stdio: 'inherit' });
  if (result.error) throw result.error;
  process.exitCode = result.status ?? 1;
}

function testCommand(tests) {
  return { command: 'node', args: ['--experimental-strip-types', '--test', `--test-timeout=${TEST_TIMEOUT_MS}`, ...tests] };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.tests.length) {
    const planned = testCommand(args.tests.map((test) => relative(process.cwd(), test)));
    if (args.dryRun) {
      console.log([planned.command, ...planned.args].join(' '));
      return;
    }
    run(planned.command, planned.args);
    return;
  }
  const graph = buildImportGraph(process.cwd());
  const decision = selectTargetedTests(changedFiles(args), graph);
  const fallback = decision.status === 'fail' && args.onUnmatched === 'full';
  if (decision.status === 'fail' && !fallback) {
    console.error(formatUnmatched(decision));
    process.exit(1);
  }
  if (decision.status === 'skip') {
    if (!decision.ignored.length && !decision.covered.length) console.log('No changed files. Pass tests explicitly, or run npm test.');
    else console.log(`No executable source changes matched tests. Nothing to run.${decision.ignored.length ? ` Ignored: ${decision.ignored.join(', ')}` : ''}`);
    return;
  }
  const planned = fallback ? { command: 'npm', args: ['test'] } : testCommand(decision.tests);
  if (fallback) console.error(`${formatUnmatched(decision)}\nFalling back to a full npm test because --on-unmatched full.`);
  else console.error(`Selected ${decision.tests.length} tests from ${decision.covered.length} changed path(s) in ${Math.round(graph.buildMs)}ms.`);
  if (args.dryRun) {
    console.log([planned.command, ...planned.args].join(' '));
    return;
  }
  run(planned.command, planned.args);
}

try {
  main();
} catch (error) {
  console.error(error.stack ?? error.message);
  process.exit(1);
}
