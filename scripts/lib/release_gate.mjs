import { spawnSync } from 'node:child_process';
import { setTimeout } from 'node:timers/promises';

const CI_TIMEOUT_MS = 10 * 60 * 1000;
const POLL_INTERVAL_MS = 10 * 1000;

export function validateReleaseOverrides(args) {
  for (const [key, flag] of [
    ['skipCiCheck', '--skip-ci-check'],
    ['acceptRedSmoke', '--accept-red-smoke'],
  ]) {
    if (args[key] !== undefined) {
      if (!args[key].trim()) throw new Error(`${flag} requires a non-empty reason; pass ${flag} "<reason>"`);
      args[key] = args[key].trim();
    }
  }
}

function command(exec, bin, args, fix, timeout = 30_000) {
  const result = exec(bin, args, { encoding: 'utf8', timeout, maxBuffer: 4 * 1024 * 1024 });
  if (result.error || result.status !== 0) {
    throw new Error(`${bin} ${args.join(' ')} failed: ${result.error?.message || result.stderr?.trim() || `exit ${result.status}`}. ${fix}`);
  }
  return result.stdout.trim();
}

export function checkReleaseTree(expectedCommit, exec = spawnSync) {
  const fix = 'Run from the repository root and fix git before deploying.';
  const commit = command(exec, 'git', ['rev-parse', 'HEAD'], fix);
  const dirty = command(exec, 'git', ['status', '--porcelain', '--untracked-files=no'], fix);
  if (dirty)
    throw new Error(
      `Tracked working tree is dirty:\n${dirty}\nCommit or discard these tracked changes before deploying; untracked files do not block release.`,
    );
  if (expectedCommit && commit !== expectedCommit)
    throw new Error('HEAD changed during release preparation; rerun the release command to check CI for the new HEAD.');
  return commit;
}

function latestRun(exec, args, workflow, timeout) {
  const fix = `Install/authenticate gh (gh auth login), check repository access, then inspect gh run list --workflow ${workflow} and retry.`;
  const fields = workflow === 'production-smoke' ? 'conclusion,status,event,displayTitle' : 'conclusion,status';
  const output = command(exec, 'gh', ['run', 'list', ...args, '--workflow', workflow, '--limit', '1', '--json', fields], fix, timeout);
  let runs;
  try {
    runs = JSON.parse(output);
  } catch {
    throw new Error(`gh returned invalid JSON for ${workflow}. ${fix}`);
  }
  if (!Array.isArray(runs) || (runs.length && (!runs[0] || typeof runs[0].status !== 'string'))) {
    throw new Error(`gh returned an invalid run list for ${workflow}. ${fix}`);
  }
  return runs[0];
}

export async function checkReleaseGate(
  args,
  { exec = spawnSync, now = Date.now, sleep = setTimeout, log = console.error, ciTimeoutMs = CI_TIMEOUT_MS, pollIntervalMs = POLL_INTERVAL_MS } = {},
) {
  validateReleaseOverrides(args);
  const overrides = [];
  for (const [key, flag] of [
    ['skipCiCheck', '--skip-ci-check'],
    ['acceptRedSmoke', '--accept-red-smoke'],
  ]) {
    if (args[key] !== undefined) {
      const entry = `${flag}: ${args[key]}`;
      overrides.push(entry);
      log(`[release-gate] WARNING override ${entry}`);
    }
  }

  const commit = checkReleaseTree(undefined, exec);
  command(
    exec,
    args.npmBin || 'npm',
    ['run', 'check:biome:tracked'],
    'Fix the reported Biome errors with npx biome check --write <failing-files>, then rerun npm run check:biome:tracked (untracked files are not linted).',
    120_000,
  );
  log('[release-gate] Tracked tree clean; tracked files pass Biome.');

  let ci = 'skipped';
  if (!args.prod) log('[release-gate] WARNING: draft deploy (no --prod); skipping GitHub ci check.');
  if (args.prod && args.skipCiCheck === undefined) {
    const deadline = now() + ciTimeoutMs;
    const fix = `Inspect gh run list --commit ${commit} --workflow ci, fix/rerun CI for this exact HEAD, then retry (or use --skip-ci-check "<reason>").`;
    while (true) {
      const remaining = deadline - now();
      if (remaining <= 0) throw new Error(`Timed out after ${ciTimeoutMs / 1000}s waiting for ci at ${commit}. ${fix}`);
      const run = latestRun(exec, ['--commit', commit, '--event', 'push'], 'ci', Math.min(30_000, remaining));
      if (!run) throw new Error(`No ci workflow run found for HEAD ${commit}. Push this commit and wait for CI to start. ${fix}`);
      if (run.status === 'completed') {
        if (run.conclusion !== 'success') throw new Error(`ci for HEAD ${commit} concluded ${run.conclusion || 'unknown'}, not success. ${fix}`);
        ci = 'success';
        log(`[release-gate] ci succeeded for HEAD ${commit}.`);
        break;
      }
      log(`[release-gate] ci for HEAD ${commit}: ${run.status}; waiting (${Math.max(0, Math.ceil((deadline - now()) / 1000))}s remaining).`);
      await sleep(Math.min(pollIntervalMs, Math.max(0, deadline - now())));
    }
  }

  let smoke = 'overridden';
  if (args.acceptRedSmoke === undefined) {
    const run = latestRun(exec, [], 'production-smoke');
    if (!run || run.status !== 'completed' || run.conclusion !== 'success') {
      const state = run ? `${run.status}/${run.conclusion || 'pending'}` : 'missing';
      throw new Error(
        `Latest production-smoke run is ${state}; production health is not green. Inspect gh run list --workflow production-smoke, fix/rerun the smoke and wait for success, or pass --accept-red-smoke "<reason>".`,
      );
    }
    // Schedules always probe production. Manual runs expose their actual target
    // in run-name; legacy/alternate-target manual runs cannot clear this gate.
    if (run.event !== 'schedule' && (run.event !== 'workflow_dispatch' || run.displayTitle !== 'production-smoke (https://0x88.app)')) {
      throw new Error(
        'Latest production-smoke run did not verify https://0x88.app. Rerun the updated production-smoke workflow with base_url=https://0x88.app, wait for the nightly, or pass --accept-red-smoke "<reason>".',
      );
    }
    smoke = 'success';
    log('[release-gate] Latest production-smoke succeeded against https://0x88.app.');
  }
  return { commit, ci, smoke, overrides };
}
