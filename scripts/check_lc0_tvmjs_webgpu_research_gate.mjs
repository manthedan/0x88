#!/usr/bin/env node
import { spawn } from 'node:child_process';

const CHECKS = [
  {
    id: 'evidence',
    command: process.execPath,
    args: [
      'scripts/summarize_lc0_tvmjs_webgpu_evidence.mjs',
      '--no-write',
      '--require-all-matches',
      '--min-search-rows', '94',
      '--min-stockfish-scored-runs', '3',
      '--min-fixed-suite-reports', '4',
    ],
  },
  {
    id: 'local-artifacts',
    command: process.execPath,
    args: ['scripts/check_lc0_tvmjs_webgpu_local_artifacts.mjs'],
  },
  {
    id: 'research-only',
    command: process.execPath,
    args: ['scripts/check_lc0_tvmjs_research_only.mjs'],
  },
];

function runCheck(check) {
  return new Promise((resolve) => {
    const child = spawn(check.command, check.args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.on('error', (error) => resolve({ ...check, ok: false, error: error.message }));
    child.on('close', (status) => {
      const out = Buffer.concat(stdout).toString('utf8');
      const err = Buffer.concat(stderr).toString('utf8');
      let parsed;
      try {
        const start = out.indexOf('{');
        parsed = start >= 0 ? JSON.parse(out.slice(start)) : undefined;
      } catch (error) {
        parsed = { parseError: error.message, stdout: out.slice(0, 2000) };
      }
      resolve({
        id: check.id,
        ok: status === 0 && parsed?.ok !== false,
        status,
        command: [check.command, ...check.args].join(' '),
        result: parsed,
        ...(err ? { stderr: err.slice(0, 4000) } : {}),
      });
    });
  });
}

async function main() {
  const checks = [];
  for (const check of CHECKS) checks.push(await runCheck(check));
  const result = {
    schema: 'lc0_browser.tvmjs_webgpu_research_gate.v1',
    generatedAt: new Date().toISOString(),
    ok: checks.every((check) => check.ok),
    checks,
    caveat: 'This gate verifies current research evidence/local artifacts/isolation. It is not a runtime promotion gate.',
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (!result.ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
