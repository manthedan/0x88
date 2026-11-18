#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process';
import { appendFileSync, cpSync, existsSync, lstatSync, mkdirSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compareRuns, HELP, parseArgs, plan, previewEnvironment } from './gate.mjs';
import { acquireLock, claimOutput, cleanupMarker, releaseAfterSession, runOwnedSession, shutdownDelay, watchShutdown } from './lifecycle.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
// Build/archive/git/file operations are confined to the calling worktree's ignored output tree.
if (realpathSync(process.cwd()) !== realpathSync(root)) throw Error(`Run from this worktree: ${root}`);
console.log(`worktree: ${process.cwd()}`);
const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(HELP);
  process.exit(0);
}
if (args.report) {
  report(resolve(args.report));
  process.exitCode = exitFor(resolve(args.report));
} else if (args['dry-run']) console.log(JSON.stringify(plan(args), null, 2));
else {
  const { signal, dispose } = watchShutdown();
  try {
    await run(signal);
  } catch (error) {
    if (error !== signal.reason) throw error;
    console.error(error.message);
    process.exitCode = error.exitCode;
  } finally {
    dispose();
  }
}

function command(bin, argv, cwd = root) {
  console.log(`pwd: ${process.cwd()}; ${bin} ${argv.join(' ')}`);
  const r = spawnSync(bin, argv, { cwd, stdio: 'inherit' });
  if (r.status !== 0) throw Error(`${bin} exited ${r.status}: ${r.error ?? ''}`);
}
function git(...argv) {
  console.log(`pwd: ${process.cwd()}; git ${argv.join(' ')}`);
  return spawnSync('git', argv, { cwd: root, encoding: 'utf8', maxBuffer: 1024 * 1024 });
}
function sourceAsset(source) {
  for (let dir = root; ; dir = dirname(dir)) {
    const candidate = resolve(dir, source);
    if (existsSync(candidate)) return candidate;
    if (dirname(dir) === dir) return null;
  }
}
function build(input, out, arm) {
  if (existsSync(input)) {
    const dir = resolve(input);
    const dist = existsSync(join(dir, 'app/analysis/index.html')) ? dir : join(dir, 'dist-client');
    if (!existsSync(join(dist, 'app/analysis/index.html'))) throw Error(`Not a built product preview: ${input}`);
    return { dist, input, kind: 'prebuilt' };
  }
  const rev = git('rev-parse', '--verify', `${input}^{commit}`);
  if (rev.status !== 0) throw Error(`Not a directory or commit: ${input}`);
  const commit = rev.stdout.trim(),
    dir = join(out, 'builds', arm),
    archive = join(out, 'builds', `${arm}.tar`);
  mkdirSync(dir, { recursive: true });
  command('git', ['archive', '--format=tar', '-o', archive, commit]);
  command('tar', ['-xf', archive, '-C', dir]);
  rmSync(archive);
  // Same lockfile: use the worktree's installed dependencies read-only. A dependency refactor gets
  // its own install; never run npm install/ci in the shared checkout/node_modules.
  const sameLock = readFileSync(join(dir, 'package-lock.json')).equals(readFileSync(join(root, 'package-lock.json')));
  if (sameLock) symlinkSync(realpathSync(join(root, 'node_modules')), join(dir, 'node_modules'), 'dir');
  else command('npm', ['ci', '--ignore-scripts'], dir);
  // Supply ignored local runtime/model assets without replacing historical manifests or tracked bytes.
  cpSync(join(root, 'public'), join(dir, 'public'), {
    recursive: true,
    dereference: false,
    filter: (src, dst) => lstatSync(src).isDirectory() || !existsSync(dst),
  });
  for (const family of ['lc0', 'maia3']) {
    const manifest = JSON.parse(readFileSync(join(dir, 'public/models', family, 'manifest.json'), 'utf8'));
    for (const entry of manifest.models ?? []) {
      if (!entry.file || !entry.source) continue;
      const target = join(dir, 'public/models', family, entry.file);
      if (existsSync(target)) continue;
      const asset = sourceAsset(entry.source);
      if (asset) {
        rmSync(target, { force: true });
        symlinkSync(asset, target);
      }
    }
  }
  command('env', ['BUILD_SCOPE=product', 'npm', 'run', 'build:client'], dir);
  return {
    dist: join(dir, 'dist-client'),
    commit,
    input,
    kind: 'archived-source',
    dependencies: sameLock ? 'shared-installed/same-lockfile' : 'isolated-npm-ci',
  };
}
async function serve(dist, out, arm, signal) {
  const socket = createServer();
  await new Promise((r) => socket.listen(0, '127.0.0.1', r));
  const port = socket.address().port;
  await new Promise((r) => socket.close(r));
  const child = spawn(process.execPath, [join(root, 'scripts/serve_isolated_static.mjs'), dist, String(port)], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
    env: previewEnvironment(port),
  });
  let spawnError;
  child.on('error', (error) => {
    spawnError = error;
  });
  const closed = new Promise((r) => child.once('close', r));
  const stop = async () => {
    child.kill('SIGTERM');
    await closed;
  };
  child.stdout.on('data', (d) => appendFileSync(join(out, `${arm}-server.log`), d));
  child.stderr.on('data', (d) => appendFileSync(join(out, `${arm}-server.log`), d));
  const base = `http://127.0.0.1:${port}`;
  try {
    for (let i = 0; i < 100; i++) {
      signal.throwIfAborted();
      if (spawnError) throw spawnError;
      if (child.exitCode !== null || child.signalCode !== null) throw Error(`Static server exited ${child.exitCode ?? child.signalCode}`);
      if (
        await fetch(`${base}/app/analysis/`, { signal: AbortSignal.any([signal, AbortSignal.timeout(1000)]) })
          .then(async (r) => {
            await r.body?.cancel();
            return r.ok;
          })
          .catch(() => false)
      )
        return { stop, base };
      await shutdownDelay(100, signal);
    }
    throw Error('Static server did not become ready');
  } catch (error) {
    await stop();
    throw error;
  }
}
async function run(signal) {
  const out = resolve(args.out ?? `.local-dev-artifacts/perf-routes/${new Date().toISOString().replace(/[:.]/g, '-')}`);
  if (args.benchlock && !existsSync(args.benchlock)) throw Error('benchlock does not exist');
  claimOutput(out);
  mkdirSync(join(out, 'sessions'));
  writeFileSync(join(out, 'plan.json'), JSON.stringify(args, null, 2));
  const servers = [];
  try {
    const builds = {};
    for (const arm of ['baseline', 'candidate']) {
      signal.throwIfAborted();
      if (args[`${arm}-url`]) builds[arm] = { input: args[arm], base: args[`${arm}-url`], kind: 'externally-served' };
      else {
        // Self comparisons share identical bytes, but use separate origins/sessions.
        builds[arm] = arm === 'candidate' && args.baseline === args.candidate ? { ...builds.baseline } : build(args[arm], out, arm);
        const server = await serve(builds[arm].dist, out, arm, signal);
        servers.push(server.stop);
        builds[arm].base = server.base;
      }
    }
    signal.throwIfAborted();
    writeFileSync(join(out, 'builds.json'), JSON.stringify(builds, null, 2));
    if (args['build-only']) return;
    const cells = plan(args);
    for (const [i, cell] of cells.entries()) {
      signal.throwIfAborted();
      const id = `${String(i).padStart(3, '0')}-${cell.browser}-${cell.journey}-${cell.arm}-r${cell.repeat}`;
      const specFile = join(out, 'sessions', `${id}.spec.json`),
        resultFile = join(out, 'sessions', `${id}.json`);
      writeFileSync(
        specFile,
        JSON.stringify({ ...cell, base: builds[cell.arm].base, source: builds[cell.arm], driver: args.driver, externalLock: !!args.benchlock }, null, 2),
      );
      console.log(`[${i + 1}/${cells.length}] ${id}`);
      let status, cleanupConfirmed;
      do {
        const release = args['lock-dir'] ? await acquireLock(resolve(args['lock-dir']), signal) : () => {};
        try {
          const commandArgs = [join(root, 'scripts/perf_routes/session.mjs'), specFile, resultFile];
          const bin = args.benchlock ?? process.execPath;
          status = await runOwnedSession(bin, args.benchlock ? [process.execPath, ...commandArgs] : commandArgs, signal, args.benchlock ? resultFile : null);
        } finally {
          cleanupConfirmed = releaseAfterSession(resultFile, release);
        }
        if (cleanupConfirmed && status === 75 && !signal.aborted) {
          console.log('Shared Safari busy; mutex released; retry in 30s');
          await shutdownDelay(30000, signal);
        }
      } while (cleanupConfirmed && status === 75 && !signal.aborted);
      const row = existsSync(resultFile)
        ? JSON.parse(readFileSync(resultFile, 'utf8'))
        : { ...cell, status: 'invalid', error: `session process exited ${status}` };
      if (!cleanupConfirmed) {
        row.status = 'invalid';
        row.error = `Browser cleanup unconfirmed; see ${cleanupMarker(resultFile)}${args['lock-dir'] ? `; mutex retained at ${args['lock-dir']}` : ''}`;
        console.error(row.error);
      }
      appendFileSync(join(out, 'runs.jsonl'), `${JSON.stringify(row)}\n`);
      signal.throwIfAborted();
      if (!cleanupConfirmed) {
        process.exitCode = 2;
        return;
      }
      report(out);
    }
  } finally {
    await Promise.all(servers.map((stop) => stop()));
    if (existsSync(join(out, 'runs.jsonl'))) report(out);
  }
  process.exitCode = exitFor(out);
}
function exitFor(dir) {
  const rows = JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8'));
  return rows.some((r) => r.status === 'regressed') ? 1 : rows.some((r) => r.status !== 'clean') ? 2 : 0;
}
function report(dir) {
  const options = JSON.parse(readFileSync(join(dir, 'plan.json'), 'utf8'));
  const runs = readFileSync(join(dir, 'runs.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
  const summary = compareRuns(runs, options);
  writeFileSync(join(dir, 'summary.json'), JSON.stringify(summary, null, 2));
  const fmt = (n) => (Number.isFinite(n) ? Number(n.toFixed(2)).toString() : '—');
  const spread = (s) => (s.n ? `${fmt(s.median)} [${fmt(s.min)}–${fmt(s.max)}]; MAD ${fmt(s.mad)}` : '—');
  const metrics = summary.flatMap((s) => s.metrics.filter((m) => m.baseline.n && m.candidate.n));
  const flagged = metrics.filter((m) => m.status === 'regressed').length;
  const lines = [
    '# Route performance gate',
    '',
    `Baseline: \`${options.baseline}\`; candidate: \`${options.candidate}\`.`,
    `Completed attempts: ${runs.length}/${plan(options).length}. Regressed metrics: ${flagged}/${metrics.length} (${fmt(metrics.length ? (flagged / metrics.length) * 100 : 0)}%).`,
    `Gate: **${summary.some((s) => s.status === 'regressed') ? 'REGRESSED' : summary.some((s) => s.status !== 'clean') ? 'INCOMPLETE / NOISY' : 'CLEAN'}**. A subset is not a full merge gate.`,
    'Memory MB = MiB. Peak total = simultaneous sampled renderer+owned Chrome GPU; independent kernel renderer/GPU peaks are separate. Settled/residual = median final 2s of fixed 20s windows; no forced GC or plateau claim.',
    'UA postResidualGc memory is sampled only after every natural footprint/worker/task/speed observation is frozen; the API-triggered GC cannot affect the natural retention numbers.',
    'Input = queued real user-move hook to two animation frames, not hardware-to-screen. Speed = median observed per-search visits/s or UCI NPS; Maia3 = worker round-trip evaluations/s. Safari GPU excluded (shared process).',
    '',
    '| Browser/journey | Metric | n B/C | Baseline median [min–max]; MAD | Candidate | Delta | Threshold | Verdict |',
    '|---|---|---:|---:|---:|---:|---:|---|',
  ];
  for (const s of summary) {
    for (const m of s.metrics)
      lines.push(
        `| ${s.browser}/${s.journey} | ${m.name} | ${m.baseline.n}/${m.candidate.n} | ${spread(m.baseline)} | ${spread(m.candidate)} | ${m.deltaRatio === null ? '— (zero baseline)' : `${fmt(m.deltaRatio * 100)}%`} | ${fmt(m.threshold * 100)}% | ${m.status} |`,
      );
    for (const r of s.invalid) lines.push(`\nInvalid ${s.browser}/${s.journey}/${r.arm}/r${r.repeat}: ${String(r.error).replaceAll('\n', ' ').slice(0, 500)}`);
  }
  lines.push('', '## Host/load and limitations');
  for (const host of new Set(runs.map((r) => JSON.stringify(r.host)))) lines.push(`- ${host}`);
  for (const limitation of new Set(runs.flatMap((r) => r.limitations ?? []))) lines.push(`- ${limitation}`);
  lines.push(
    '- Missing API samples are unavailable, never zero. Failed journeys stay in the denominator and cannot yield a clean gate. Noisy runs require a new balanced cohort, not selective outlier removal.',
    '',
  );
  writeFileSync(join(dir, 'summary.md'), lines.join('\n'));
  console.log(`summary: ${join(dir, 'summary.md')}`);
}
