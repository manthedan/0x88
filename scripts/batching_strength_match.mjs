#!/usr/bin/env node
// One benchlocked chunk of a batching-strength match.
// Exit 0: chunk saved, games remain. Exit 3: match complete.
// Exit 2: fatal probe/WebGPU. Exit 1: retryable failure.
// The caller must hold SP/benchlock. This process pauses the page before
// exiting so the GPU is idle when the lock is released.
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { BATCHING_STRENGTH_BUILD, strengthMatch } from '../src/lc0/batchingStrengthMatch.ts';
import { BATCHING_STRENGTH_GENERATOR_VERSION, distinctOpeningElo, equalVisitGate, pentanomialElo, queenSidePoints } from '../src/lc0/batchingStrengthScore.ts';
import { pauseStrengthPage } from './lib/batching_strength_pause.mjs';
import { runAgent } from './lib/process.mjs';

const SESSION = process.env.AGENT_BROWSER_SESSION || 'batching-strength';
process.env.AGENT_BROWSER_EXECUTABLE_PATH ||= '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
process.env.AGENT_BROWSER_ARGS ||= '--enable-unsafe-webgpu';
process.env.AGENT_BROWSER_SESSION = SESSION;
delete process.env.AGENT_BROWSER_WEBGPU;

const agent = { agentBrowser: 'node_modules/.bin/agent-browser', session: SESSION };

function arg(name, fallback = undefined) {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  return process.argv[index + 1];
}

function flag(name) {
  return process.argv.includes(name);
}

function loadavg() {
  try {
    return execFileSync('sysctl', ['-n', 'vm.loadavg'], { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function unwrap(data) {
  if (data == null) return null;
  if (typeof data === 'string') {
    const trimmed = data.trim();
    if (!trimmed) return null;
    try {
      return unwrap(JSON.parse(trimmed));
    } catch {
      return trimmed;
    }
  }
  if (typeof data === 'object') {
    if (typeof data.text === 'string') return unwrap(data.text);
    if (typeof data.result === 'string') {
      try {
        return JSON.parse(data.result);
      } catch {
        return data.result;
      }
    }
    if (data.result !== undefined && typeof data.result !== 'undefined') return data.result;
  }
  return data;
}

async function agentCall(command, timeoutMs, input) {
  const data = await runAgent(agent, command, timeoutMs, undefined, input);
  return unwrap(data);
}

async function readSnapshot() {
  const text = await agentCall(['get', 'text', '#bs-json'], 30_000);
  if (!text || typeof text !== 'object') return null;
  if (!text.matchId && !text.phase) return null;
  return text;
}

function loadGames(file) {
  const byIndex = new Map();
  if (!existsSync(file)) return byIndex;
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const game = JSON.parse(line);
      if (Number.isInteger(game.index)) byIndex.set(game.index, game);
    } catch {
      // Keep the games above a torn final line.
      break;
    }
  }
  return byIndex;
}

function nextIndex(games) {
  const indexes = [...games.keys()].sort((a, b) => a - b);
  for (let i = 0; i < indexes.length; i++) if (indexes[i] !== i) return i;
  return indexes.length;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

function armTimes(games) {
  const times = new Map();
  const visits = new Map();
  const add = (name, msList, visitList) => {
    if (!name) return;
    const ms = times.get(name) ?? [];
    const vs = visits.get(name) ?? [];
    for (const value of msList ?? []) if (Number.isFinite(value)) ms.push(value);
    for (const value of visitList ?? []) if (Number.isFinite(value)) vs.push(value);
    times.set(name, ms);
    visits.set(name, vs);
  };
  for (const game of games) {
    add(game.whiteName, game.whiteMoveMs, game.whiteVisits);
    add(game.blackName, game.blackMoveMs, game.blackVisits);
  }
  const out = {};
  for (const [name, ms] of times) {
    out[name] = {
      moves: ms.length,
      medianMs: median(ms),
      p10Ms: percentile(ms, 10),
      p90Ms: percentile(ms, 90),
      medianVisits: median(visits.get(name) ?? []),
    };
  }
  return out;
}

function summarize(spec, games, extra) {
  const ordered = [...games.values()].sort((a, b) => a.index - b.index);
  const estimate = pentanomialElo(ordered);
  const status = ordered.length === 0 ? 'not-run' : extra.earlyStop || ordered.length < extra.gamesTarget ? 'partial' : 'complete';
  const openingUnitEstimate = distinctOpeningElo(ordered);
  const repeatedOpeningPairs = estimate.pairs - openingUnitEstimate.pairs;
  const gate = status === 'complete' && repeatedOpeningPairs === 0 ? equalVisitGate(estimate) : null;
  let queenPoints = 0;
  let queenGames = 0;
  for (const game of ordered) {
    const points = queenSidePoints(game.result, game.queenLessColor ?? null);
    if (points === null) continue;
    queenPoints += points;
    queenGames += 1;
  }
  return {
    match: spec.id,
    status,
    label: spec.label,
    kind: spec.movetimeMs ? 'equal-time' : 'equal-visit',
    headline: spec.headline,
    build: BATCHING_STRENGTH_BUILD,
    generatorVersion: BATCHING_STRENGTH_GENERATOR_VERSION,
    games: ordered.length,
    gamesTarget: extra.gamesTarget,
    completePairs: estimate.pairs,
    observedOpeningUnits: openingUnitEstimate.pairs,
    repeatedOpeningPairs,
    openingUnitEstimate: repeatedOpeningPairs ? openingUnitEstimate : null,
    headlineMet: spec.headline ? ordered.length >= spec.games : true,
    wdl: { wins: estimate.wins, draws: estimate.draws, losses: estimate.losses },
    score: estimate.score,
    drawRate: estimate.drawRate,
    elo: estimate.elo,
    ci95: estimate.ci95,
    pairCounts: estimate.pairCounts,
    method: estimate.method,
    gate,
    queenSideWinRate: queenGames ? queenPoints / queenGames : null,
    queenSideGames: queenGames,
    arms: armTimes(ordered),
    ...extra,
  };
}

function saveAll(dir, games, summary) {
  const ordered = [...games.values()].sort((a, b) => a.index - b.index);
  writeFileSync(join(dir, 'games.jsonl'), ordered.map((game) => JSON.stringify(game)).join('\n') + (ordered.length ? '\n' : ''));
  writeFileSync(join(dir, 'games.pgn'), ordered.map((game) => game.pgn ?? '').join('\n'));
  writeFileSync(join(dir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
}

function matchUrl(base, spec, gamesTarget, start, openingCount, maxPlies, movetimeMs, runToken) {
  const url = new URL('/lab/batching-strength/', base);
  url.searchParams.set('match', spec.id);
  url.searchParams.set('runToken', runToken);
  url.searchParams.set('autostart', '1');
  url.searchParams.set('games', String(gamesTarget));
  url.searchParams.set('start', String(start));
  url.searchParams.set('openingCount', String(openingCount));
  url.searchParams.set('maxPlies', String(maxPlies));
  if (movetimeMs) url.searchParams.set('movetimeMs', String(movetimeMs));
  return url.toString();
}

async function main() {
  const matchId = arg('--match');
  if (!matchId) throw new Error('missing --match');
  const spec = { ...strengthMatch(matchId) };
  const minutes = Number(arg('--minutes', '16'));
  const gamesOverride = arg('--games');
  const dir = arg('--dir', join(process.env.BATCHING_STRENGTH_ARTIFACT_ROOT || 'artifacts/batching-strength', spec.id));
  const base = arg('--base-url', 'http://127.0.0.1:5187');
  const screenshot = arg('--screenshot');
  if (screenshot) mkdirSync(dirname(screenshot), { recursive: true });
  const restartBrowser = flag('--restart-browser');
  // Nominal-CI early stop is opt-out; docs explain the optional-stopping caveat.
  const gateStop = !flag('--no-gate-stop');
  const openingCount = Number(arg('--opening-count', String(spec.openingCount)));
  const maxPlies = Number(arg('--max-plies', String(spec.maxPlies)));
  const movetimeMs = arg('--movetime-ms') ? Number(arg('--movetime-ms')) : spec.movetimeMs;
  if (!(Number.isFinite(minutes) && minutes > 0 && minutes <= 20)) {
    throw Object.assign(new Error('--minutes must be > 0 and <= 20'), { exitCode: 2 });
  }
  for (const [name, value] of [
    ['opening-count', openingCount],
    ['max-plies', maxPlies],
  ]) {
    if (!Number.isInteger(value) || value < 2) throw Object.assign(new Error(`--${name} must be an integer >= 2`), { exitCode: 2 });
  }
  if (movetimeMs !== undefined && !(Number.isFinite(movetimeMs) && movetimeMs > 0)) {
    throw Object.assign(new Error('--movetime-ms must be positive'), { exitCode: 2 });
  }
  spec.movetimeMs = movetimeMs;
  const gamesFile = join(dir, 'games.jsonl');
  const stateFile = join(dir, 'state.json');
  const games = loadGames(gamesFile);
  let state = existsSync(stateFile) ? JSON.parse(readFileSync(stateFile, 'utf8')) : null;
  if (state && state.generatorVersion !== BATCHING_STRENGTH_GENERATOR_VERSION) {
    throw Object.assign(new Error(`generator version ${state.generatorVersion} != ${BATCHING_STRENGTH_GENERATOR_VERSION}`), { exitCode: 2 });
  }
  if (
    state &&
    (state.match !== spec.id || state.build !== BATCHING_STRENGTH_BUILD || state.maxPlies !== maxPlies || state.movetimeMs !== (movetimeMs ?? null))
  ) {
    throw Object.assign(new Error('checkpoint match/build/maxPlies/movetime differs; use a new --dir'), { exitCode: 2 });
  }
  if (state && state.openingCount !== openingCount) {
    throw Object.assign(new Error(`openingCount locked at ${state.openingCount}, got ${openingCount}`), { exitCode: 2 });
  }
  const gamesTarget = gamesOverride ? Number(gamesOverride) : (state?.gamesTarget ?? spec.games);
  if (!Number.isInteger(gamesTarget) || gamesTarget < 1) throw Object.assign(new Error('--games must be a positive integer'), { exitCode: 2 });
  mkdirSync(dir, { recursive: true });
  const runToken = state?.runToken ?? randomUUID();
  const startedLoad = loadavg();
  // Leave the last three minutes for the in-flight ply to finish so the GPU
  // is idle before benchlock is released. Total hold stays near --minutes.
  const graceMs = 3 * 60 * 1000;
  const deadline = Date.now() + Math.max(60_000, minutes * 60 * 1000 - graceMs);
  let exitCode = 0;
  let phase = 'setup';
  let ply = null;
  let fatalError = null;
  let shot = false;

  const writeState = (patch) => {
    state = {
      match: spec.id,
      runToken,
      build: BATCHING_STRENGTH_BUILD,
      generatorVersion: BATCHING_STRENGTH_GENERATOR_VERSION,
      openingCount,
      maxPlies,
      movetimeMs: movetimeMs ?? null,
      gamesTarget,
      gamesDone: nextIndex(games),
      probes: state?.probes ?? [],
      determinism: state?.determinism ?? [],
      backends: state?.backends ?? {},
      ...patch,
    };
    writeFileSync(stateFile, `${JSON.stringify(state, null, 2)}\n`);
    const summary = summarize(spec, games, {
      gamesTarget,
      probes: state.probes,
      determinism: state.determinism,
      backends: state.backends,
      earlyStop: existsSync(join(dir, 'early-stop.json')) ? JSON.parse(readFileSync(join(dir, 'early-stop.json'), 'utf8')) : null,
      loadavg: loadavg(),
    });
    saveAll(dir, games, summary);
  };

  try {
    if (nextIndex(games) >= gamesTarget || existsSync(join(dir, 'early-stop.json'))) {
      exitCode = 3;
      phase = 'done';
      writeState({});
      return;
    }
    if (restartBrowser) await agentCall(['close'], 20_000).catch(() => {});
    let snap = await readSnapshot().catch(() => null);
    const same =
      snap &&
      snap.matchId === spec.id &&
      snap.runToken === runToken &&
      snap.build === BATCHING_STRENGTH_BUILD &&
      snap.generatorVersion === BATCHING_STRENGTH_GENERATOR_VERSION &&
      snap.gamesTarget === gamesTarget &&
      snap.phase !== 'done' &&
      !snap.fatal &&
      snap.phase !== 'error' &&
      snap.phase !== 'idle';
    if (!same) {
      const url = matchUrl(base, spec, gamesTarget, nextIndex(games), openingCount, maxPlies, movetimeMs, runToken);
      process.stderr.write(`[batching-strength] open ${url}\n`);
      await agentCall(['open', url], 90_000);
      await agentCall(['set', 'viewport', '1280', '800'], 20_000).catch(() => {});
    } else if (snap.phase === 'paused') {
      await agentCall(['eval', '--stdin'], 30_000, 'window.__batchingStrength.resume()');
    }
    while (Date.now() < deadline) {
      snap = await readSnapshot().catch(() => null);
      if (
        snap?.matchId === spec.id &&
        (snap.fatal || snap.phase === 'error' || snap.phase === 'thinking' || snap.phase === 'between' || snap.phase === 'paused' || snap.phase === 'done')
      )
        break;
      await sleep(2000);
    }
    if (!snap || snap.matchId !== spec.id) throw new Error(`page did not reach ${spec.id} (phase ${snap?.phase ?? 'none'})`);
    if (snap.fatal) throw Object.assign(new Error(snap.error || 'fatal probe or WebGPU failure'), { exitCode: 2 });
    if (snap.build !== BATCHING_STRENGTH_BUILD || snap.runToken !== runToken) {
      throw Object.assign(new Error('built page protocol/token differs from driver; rebuild/reload before measuring'), { exitCode: 2 });
    }

    while (Date.now() < deadline) {
      snap = await readSnapshot().catch(() => null);
      if (!snap) throw new Error('snapshot disappeared');
      phase = snap.phase;
      ply = snap.ply;
      if (snap.probes?.length && !state?.probes?.length) {
        writeState({ probes: snap.probes, determinism: snap.determinism ?? [], backends: snap.backends ?? {} });
      }
      const pulled = unwrap(await agentCall(['eval', '--stdin'], 60_000, `JSON.stringify(window.__batchingStrength.pull(${nextIndex(games)}, 8))`));
      const incoming = pulled?.games ?? [];
      for (const game of incoming) games.set(game.index, game);
      if (incoming.length) writeState({});
      process.stderr.write(`[batching-strength] ${spec.id} ${snap.phase} ${nextIndex(games)}/${gamesTarget} ply=${snap.ply ?? '-'} load=${loadavg()}\n`);
      if (screenshot && !shot && (snap.phase === 'thinking' || nextIndex(games) > 0)) {
        await agentCall(['screenshot', screenshot], 30_000).catch((error) => process.stderr.write(`[batching-strength] screenshot failed: ${error.message}\n`));
        shot = true;
      }
      if (snap.fatal) throw Object.assign(new Error(snap.error || 'fatal'), { exitCode: 2 });
      if (snap.phase === 'error') throw new Error(snap.error || 'match error');
      if (nextIndex(games) >= gamesTarget) {
        exitCode = 3;
        phase = 'done';
        break;
      }
      const estimate = gateStop ? pentanomialElo([...games.values()]) : null;
      if (estimate && estimate.pairs >= 20 && (estimate.ci95[0] > -10 || estimate.ci95[1] < -10)) {
        const stop = {
          games: estimate.games,
          elo: estimate.elo,
          ci95: estimate.ci95,
          reason: 'nominal pair-profile CI entirely on one side of -10 Elo; optional stopping, not a sequentially valid test',
        };
        writeFileSync(join(dir, 'early-stop.json'), `${JSON.stringify(stop, null, 2)}\n`);
        phase = 'gate-stop';
        break;
      }
      // A done page may still have more records than one pull's limit. Drain
      // those before treating the match as complete.
      await sleep(4000);
    }

    if (exitCode !== 3) {
      snap = await pauseStrengthPage({
        requestPause: () => agentCall(['eval', '--stdin'], 30_000, 'window.__batchingStrength.pause()'),
        readSnapshot,
        sleep,
      });
      phase = snap.phase;
      ply = snap.ply;
      const pulled = await agentCall(['eval', '--stdin'], 60_000, `JSON.stringify(window.__batchingStrength.pull(${nextIndex(games)}, 20))`).catch(() => null);
      for (const game of unwrap(pulled)?.games ?? []) games.set(game.index, game);
    }
    writeState({
      probes: snap?.probes ?? state?.probes ?? [],
      determinism: snap?.determinism ?? state?.determinism ?? [],
      backends: snap?.backends ?? state?.backends ?? {},
    });
    if (exitCode !== 3 && (nextIndex(games) >= gamesTarget || existsSync(join(dir, 'early-stop.json')))) exitCode = 3;
  } catch (error) {
    fatalError = error instanceof Error ? error.message : String(error);
    exitCode = error?.exitCode || 1;
    process.stderr.write(`[batching-strength] ${fatalError}\n`);
    // A failed snapshot/driver call is not proof the page stopped computing.
    // Close this owned session before returning control to benchlock.
    await agentCall(['close'], 30_000).catch(() => {});
    try {
      writeState({ error: fatalError });
    } catch {
      // The status line below still reports the failure.
    }
  } finally {
    const status = {
      match: spec.id,
      gamesDone: nextIndex(games),
      gamesTarget,
      ply,
      phase,
      loadavg: loadavg(),
      loadavgStart: startedLoad,
      exitCode,
      error: fatalError,
    };
    writeFileSync(join(dir, 'chunk-status.json'), `${JSON.stringify(status, null, 2)}\n`);
    const chunks = join(dir, 'chunks.jsonl');
    writeFileSync(chunks, `${existsSync(chunks) ? readFileSync(chunks, 'utf8') : ''}${JSON.stringify({ ...status, at: new Date().toISOString() })}\n`);
    process.stdout.write(
      `CHUNK match=${spec.id} games=${status.gamesDone}/${gamesTarget} ply=${ply ?? '-'} phase=${phase} load=${status.loadavg} code=${exitCode}\n`,
    );
    process.exitCode = exitCode;
  }
}

try {
  await main();
} catch (error) {
  process.stderr.write(`[batching-strength] ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = error?.exitCode ?? 1;
}
