#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { hostname } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { linkAssets, localAssets } from './assets.mjs';
import { browserIdentity, compareRuns, HELP, parseArgs, plan, platformProblem, previewEnvironment } from './gate.mjs';
import {
  acquireLock,
  cancellationMarker,
  claimOutput,
  cleanupMarker,
  liveOwner,
  lockRun,
  releaseAfterSession,
  runOwnedSession,
  shutdownDelay,
  watchShutdown,
} from './lifecycle.mjs';
import {
  buildFingerprint,
  manifestMismatches,
  measuringFiles,
  readJson,
  recordAttempt,
  reusableBuild,
  runManifest,
  selectSessions,
  sharesBaselineBuild,
  writeJsonAtomic,
} from './resume.mjs';

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
// Identify an arm before building so a resumed run can refuse a different cohort up front.
function armSource(input) {
  if (existsSync(input)) {
    const dir = resolve(input);
    const dist = existsSync(join(dir, 'app/analysis/index.html')) ? dir : join(dir, 'dist-client');
    if (!existsSync(join(dist, 'app/analysis/index.html'))) throw Error(`Not a built product preview: ${input}`);
    const version = join(dist, '_app/version.json');
    // The path alone does not identify bytes: a rebuild in place must not resume the old cohort.
    return { dist, input, kind: 'prebuilt', version: existsSync(version) ? readFileSync(version, 'utf8') : null, fingerprint: buildFingerprint(dist) };
  }
  const rev = git('rev-parse', '--verify', `${input}^{commit}`);
  if (rev.status !== 0) throw Error(`Not a directory or commit: ${input}`);
  return { input, kind: 'archived-source', commit: rev.stdout.trim() };
}
function build(source, out, arm) {
  if (source.kind === 'prebuilt') return source;
  const dir = join(out, 'builds', arm),
    record = join(out, 'builds', `${arm}.json`);
  const reused = reusableBuild(record, source.commit);
  if (reused) {
    console.log(`reusing ${arm} build of ${source.commit}`);
    return reused;
  }
  rmSync(record, { force: true });
  // Each attempt builds in its own directory: tar/npm orphaned by a SIGKILLed coordinator may still be
  // writing into an earlier one. Only a finished build is renamed into place (links are absolute).
  const stage = `${dir}.staging-${process.pid}-${Date.now()}`;
  const archive = `${stage}.tar`;
  mkdirSync(stage, { recursive: true });
  command('git', ['archive', '--format=tar', '-o', archive, source.commit]);
  command('tar', ['-xf', archive, '-C', stage]);
  rmSync(archive);
  // Same lockfile: use the worktree's installed dependencies read-only. A dependency refactor gets
  // its own install; never run npm install/ci in the shared checkout/node_modules.
  const sameLock = readFileSync(join(stage, 'package-lock.json')).equals(readFileSync(join(root, 'package-lock.json')));
  if (sameLock) symlinkSync(realpathSync(join(root, 'node_modules')), join(stage, 'node_modules'), 'dir');
  else command('npm', ['ci', '--ignore-scripts'], stage);
  const links = localAssets(root, stage);
  command('env', ['BUILD_SCOPE=product', 'npm', 'run', 'build:client'], stage);
  linkAssets(links, join(stage, 'dist-client'));
  // Intermediate copy of the client output; only dist-client is served.
  rmSync(join(stage, '.svelte-kit/output'), { recursive: true, force: true });
  // A tree left here by an unrecorded earlier attempt was only ever renamed in, never written to.
  rmSync(dir, { recursive: true, force: true });
  renameSync(stage, dir);
  const dist = join(dir, 'dist-client');
  const built = {
    dist,
    commit: source.commit,
    input: source.input,
    kind: source.kind,
    dependencies: sameLock ? 'shared-installed/same-lockfile' : 'isolated-npm-ci',
    linkedLocalAssets: links.size,
  };
  writeJsonAtomic(record, built);
  return built;
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
function harnessHash() {
  const hash = createHash('sha256');
  for (const file of measuringFiles(root, 'scripts/perf_routes/session.mjs', ['scripts/perf_matrix/pidfp.c', 'scripts/serve_isolated_static.mjs']))
    hash.update(`${file}\0`).update(readFileSync(join(root, file)));
  return hash.digest('hex');
}
async function run(signal) {
  const unsupported = platformProblem(args);
  if (unsupported && !args['build-only']) throw Error(unsupported);
  const out = resolve(args.out ?? `.local-dev-artifacts/perf-routes/${new Date().toISOString().replace(/[:.]/g, '-')}`);
  if (args.benchlock && !existsSync(args.benchlock)) throw Error('benchlock does not exist');
  const sources = {};
  for (const arm of ['baseline', 'candidate'])
    sources[arm] = args[`${arm}-url`] ? { input: args[arm], url: args[`${arm}-url`], kind: 'externally-served' } : armSource(args[arm]);
  const manifest = runManifest(args, sources, { host: hostname(), harness: harnessHash() });
  const resuming = args.resume && existsSync(out);
  if (!resuming) claimOutput(out);
  const unlock = lockRun(out);
  try {
    if (resuming) {
      const saved = readJson(join(out, 'manifest.json'));
      if (!saved) throw Error(`Cannot resume ${out}: no readable manifest.json (not started by a resumable run)`);
      const mismatches = manifestMismatches(saved, manifest);
      if (mismatches.length) throw Error(`Cannot resume ${out}; it measured a different cohort: ${mismatches.join(', ')}`);
    } else {
      mkdirSync(join(out, 'sessions'));
      writeJsonAtomic(join(out, 'plan.json'), args);
      writeJsonAtomic(join(out, 'manifest.json'), manifest);
    }
    await measure(signal, out, sources, resuming);
  } finally {
    unlock();
  }
}
async function measure(signal, out, sources, resuming) {
  const shared = sharesBaselineBuild(sources, resuming ? readJson(join(out, 'builds.json')) : null);
  const cells = selectSessions(plan(args), (id) => {
    const file = join(out, 'sessions', `${id}.json`);
    return { result: readJson(file), cleanupPending: existsSync(cleanupMarker(file)), owner: liveOwner(file) };
  });
  if (resuming) console.log(`resume ${out}: reusing ${cells.filter((c) => c.reuse).length}/${cells.length} sessions`);
  const servers = [];
  try {
    const builds = {};
    for (const arm of ['baseline', 'candidate']) {
      signal.throwIfAborted();
      if (sources[arm].kind === 'externally-served') builds[arm] = { input: args[arm], base: sources[arm].url, kind: 'externally-served' };
      else {
        // Self comparisons share identical bytes, but use separate origins/sessions.
        builds[arm] = arm === 'candidate' && shared ? { ...builds.baseline } : build(sources[arm], out, arm);
        const server = await serve(builds[arm].dist, out, arm, signal);
        servers.push(server.stop);
        builds[arm].base = server.base;
      }
    }
    signal.throwIfAborted();
    writeFileSync(join(out, 'builds.json'), JSON.stringify(builds, null, 2));
    if (args['build-only']) return;
    // runs.jsonl is rebuilt in plan order from the session files; keep the old index for audit.
    if (existsSync(join(out, 'runs.jsonl'))) renameSync(join(out, 'runs.jsonl'), join(out, `runs.${Date.now()}.prev.jsonl`));
    // Each session refuses, before measuring, a browser build other than the one already in the cohort.
    const cohortBrowser = {};
    const claim = (row) => (cohortBrowser[row.browser] ??= browserIdentity(row));
    for (const [i, { cell, id, reuse }] of cells.entries()) {
      signal.throwIfAborted();
      if (reuse) {
        claim(reuse);
        appendFileSync(join(out, 'runs.jsonl'), `${JSON.stringify(reuse)}\n`);
        continue;
      }
      const specFile = join(out, 'sessions', `${id}.spec.json`),
        resultFile = join(out, 'sessions', `${id}.json`);
      writeFileSync(
        specFile,
        JSON.stringify(
          {
            ...cell,
            base: builds[cell.arm].base,
            source: builds[cell.arm],
            driver: args.driver,
            externalLock: !!args.benchlock,
            expectedBrowser: cohortBrowser[cell.browser] ?? null,
          },
          null,
          2,
        ),
      );
      console.log(`[${i + 1}/${cells.length}] ${id}`);
      let status, cleanupConfirmed;
      do {
        // Per attempt (Safari-busy retries too): a stale cancellation marker would abort the new attempt,
        // and a stale result must not stand in for an attempt that wrote none.
        rmSync(cancellationMarker(resultFile), { force: true });
        rmSync(resultFile, { force: true });
        const release = args['lock-dir'] ? await acquireLock(resolve(args['lock-dir']), signal) : () => {};
        try {
          const commandArgs = [join(root, 'scripts/perf_routes/session.mjs'), specFile, resultFile];
          const bin = args.benchlock ?? process.execPath;
          status = await runOwnedSession(
            bin,
            args.benchlock ? [process.execPath, ...commandArgs] : commandArgs,
            signal,
            args.benchlock ? resultFile : null,
            resultFile,
          );
        } finally {
          cleanupConfirmed = releaseAfterSession(resultFile, release);
        }
        if (cleanupConfirmed && status === 75 && !signal.aborted) {
          console.log('Shared Safari busy; mutex released; retry in 30s');
          await shutdownDelay(30000, signal);
        }
      } while (cleanupConfirmed && status === 75 && !signal.aborted);
      const row = recordAttempt(cell, resultFile, {
        status,
        cleanupConfirmed,
        interruption: signal.aborted ? signal.reason.message : undefined,
        lockDir: args['lock-dir'],
      });
      if (!cleanupConfirmed) console.error(row.error);
      // Not an outcome (resume re-runs it), and never appended: the cohort stays combinable.
      if (row.browserMismatch && cleanupConfirmed)
        throw Error(`${id}: browser build changed mid-cohort (${row.browserMismatch}); restore ${row.expectedBrowser} and --resume, or start a new cohort`);
      claim(row);
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
    `Exempted aborts: ${runs.reduce((n, r) => n + (r.teardownAborts?.length ?? 0), 0)} at teardown/transitions; ${runs.reduce((n, r) => n + (r.optionalPrefetchFailures?.filter((entry) => entry.clientAborted).length ?? 0), 0)} optional.`,
    ...(options.repeats === 1 ? ['Same-build smoke only; at least five repeats per arm are required for a gate verdict.'] : []),
    ...(summary[0]?.platform === 'linux'
      ? [
          'Platform: **linux** (Chromium, NVIDIA). Memory = smaps_rollup Pss_Dirty+SwapPss per process plus nvidia-smi VRAM in the GPU process (`memory.gpuVramPeakMb` alone); peaks are sampled. Never compare with macOS results.',
        ]
      : []),
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
