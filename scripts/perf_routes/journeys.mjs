import { sleep } from '../perf_matrix/common.mjs';
import { chooseHumanMove, playHumanTurn } from './human.mjs';

const PLAY_TURN = `(${playHumanTurn.toString()})(document.querySelector('#status')?.textContent)`;

export const DOCS_READY = `location.pathname === '/docs/' && document.querySelector('#doc-h1')?.textContent?.trim() === 'Using 0x88' && !!document.querySelector('#pages-analysis')`;
export const MIDGAME = 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 6';
export const routeFor = (name) => (name.startsWith('play') ? 'play' : name.startsWith('arena') ? 'arena' : 'analysis');

export async function waitFor(evaluate, expression, timeout = 120000) {
  const end = Date.now() + timeout;
  let last;
  do {
    last = await evaluate(expression);
    if (last) return last;
    await sleep(150);
  } while (Date.now() < end);
  throw Error(`Timed out: ${expression.slice(0, 200)}; last=${JSON.stringify(last)}`);
}

export async function setControl(evaluate, selector, value) {
  await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el || el.disabled) throw Error('Control missing/disabled: ' + ${JSON.stringify(selector)});
    if (el.type === 'checkbox') el.checked = ${JSON.stringify(value)};
    else el.value = ${JSON.stringify(String(value))};
    el.dispatchEvent(new Event('change', {bubbles:true}));
    return true;
  })()`);
}
const click = (evaluate, selector) =>
  evaluate(
    `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el || el.disabled) throw Error('Missing/disabled: '+${JSON.stringify(selector)}); el.click(); return true; })()`,
  );

async function analysisSetup(evaluate, four = false) {
  await waitFor(evaluate, `!!window.__analysisSchedulingState && document.querySelectorAll('#engineList .engine-row').length === 2`);
  const defaults = await evaluate(
    `[...document.querySelectorAll('#engineList .engine-row')].map(r => [r.querySelector('.row-fam')?.value,r.querySelector('.row-var')?.value])`,
  );
  if (
    JSON.stringify(defaults) !==
    JSON.stringify([
      ['lc0', 'small'],
      ['sf', 'lite'],
    ])
  )
    throw Error(`Unexpected default rows: ${JSON.stringify(defaults)}`);
  if (four) {
    // Add through the real UI, then explicitly select CPU rows, not whichever family is suggested next.
    for (const family of ['berserk', 'plentychess']) {
      await click(evaluate, '#addEngine');
      const index = await evaluate(`document.querySelectorAll('#engineList .engine-row').length - 1`);
      await setControl(evaluate, `.row-fam[data-i="${index}"]`, family);
    }
  }
  await setControl(evaluate, '#autoAnalyze', true);
  await setControl(evaluate, '#fenInput', MIDGAME);
  await click(evaluate, '#loadFen');
  await waitFor(evaluate, `window.__analysisSchedulingState?.().fen === ${JSON.stringify(MIDGAME)}`);
  await waitFor(
    evaluate,
    `window.__analysisSchedulingState?.().workerProgressMessages > 0 || /Analysis complete/.test(window.__analysisSchedulingState?.().message)`,
  );
}

export function observeMovePvs(move) {
  const summaries = document.querySelector('#engineSummaries');
  const count = document.querySelectorAll('#engineList .engine-row').length;
  if (!summaries || !count) return;
  move.fen = window.__analysisSchedulingState().fen;
  const record = () => {
    if (window.__analysisSchedulingState().fen !== move.fen) return;
    const rows = [...summaries.querySelectorAll('[data-pv-fen]')].filter((el) => el.dataset.pvFen === move.fen && el.querySelector('.pv-move'));
    if (new Set(rows.map((el) => el.dataset.engine)).size < count) return;
    move.pvAt = performance.now();
    observer.disconnect();
    clearTimeout(timer);
  };
  const observer = new MutationObserver(record);
  observer.observe(summaries, { childList: true, attributes: true, attributeFilter: ['data-pv-fen'], subtree: true });
  const timer = setTimeout(() => observer.disconnect(), 120000);
  record();
}

// Select a legal move through the accessible board, then invoke the existing real user-move hook.
// A queued task plus two animation frames measures main-thread queue/handler/render latency, not hardware latency.
export async function userMove(evaluate, analysis, index = 0, preferredMove = '') {
  const before = await evaluate(analysis ? `window.__analysisSchedulingState().fen` : `document.querySelectorAll('#moveList .san').length`);
  const result = await evaluate(`(() => {
    const host = document.querySelector('#keyboardBoard');
    const cells = [...host.querySelectorAll('[data-square]')];
    const color = ${analysis} ? (window.__analysisSchedulingState().fen.split(' ')[1] === 'w' ? 'White' : 'Black') : 'White';
    const own = cells.filter(c => c.getAttribute('aria-label').includes(color));
    const preferred = ${JSON.stringify(preferredMove)};
    // The host-side legal chooser adds no in-browser engine/worker to the measurements.
    own.sort((a,b) => Number(b.dataset.square === preferred.slice(0,2)) - Number(a.dataset.square === preferred.slice(0,2)) || Number(b.getAttribute('aria-label').includes('knight')) - Number(a.getAttribute('aria-label').includes('knight')) || a.dataset.square.localeCompare(b.dataset.square));
    for (const source of own) {
      if (preferred && source.dataset.square !== preferred.slice(0,2)) continue;
      host.querySelector('[data-square="'+source.dataset.square+'"]').click();
      const dests = [...host.querySelectorAll('[aria-label*="legal destination"]')].filter(c => !source.getAttribute('aria-label').includes('pawn') || !/[18]$/.test(c.dataset.square));
      if (!dests.length) continue;
      dests.sort((a,b) => Math.abs(a.dataset.square.charCodeAt(0)-100.5) - Math.abs(b.dataset.square.charCodeAt(0)-100.5) || a.dataset.square.localeCompare(b.dataset.square));
      const from = source.dataset.square;
      if (preferred && !dests.some(c => c.dataset.square === preferred.slice(2,4))) throw Error('Legal chooser move is not interactive on the board: '+preferred);
      const to = preferred ? preferred.slice(2,4) : dests[${index} % dests.length].dataset.square;
      const queuedAt = performance.now();
      window.__routeMove = {from,to,done:false};
      setTimeout(() => {
        window.__routeMove.start = performance.now();
        window[${JSON.stringify(analysis ? '__analysisUserMove' : '__playUserMove')}](from,to);
        requestAnimationFrame(() => requestAnimationFrame(() => {
          window.__routeMove.ms = performance.now()-queuedAt; window.__routeMove.done = true;
        }));
        ${analysis ? `(${observeMovePvs.toString()})(window.__routeMove);` : ''}
      }, 0);
      return {from,to};
    }
    throw Error('No legal non-promotion user move');
  })()`);
  await waitFor(evaluate, `window.__routeMove?.done`);
  await waitFor(
    evaluate,
    analysis ? `window.__analysisSchedulingState().fen !== ${JSON.stringify(before)}` : `document.querySelectorAll('#moveList .san').length > ${before}`,
  );
  return {
    ...result,
    ...(await evaluate(`({latencyMs:window.__routeMove.ms,started:window.__routeMove.start${analysis ? ',fen:window.__analysisSchedulingState().fen' : ''}})`)),
  };
}

export function arenaGameFailed(log) {
  return /forfeit|failed|error|timeout|\billegal\b|\bresigned\b|\bcancelled\b/i.test(log);
}

function analysisMoveObservation(move, deadline) {
  const worker = window.__routePerf.analysisProgress.find((p) => p.fen === move.fen && p.at >= move.started && p.at <= deadline);
  const { start, fen, pvAt } = window.__routeMove;
  if (worker && start === move.started && fen === move.fen && pvAt >= start && pvAt <= deadline) return { workerAt: worker.at, pvAt };
  return performance.now() > deadline ? { expired: true } : null;
}

export async function requireAnalysisForMove(evaluate, move, timeout = 120000) {
  // Check all evidence on the same page clock as the move, even if the first poll is delayed.
  const deadline = move.started + timeout;
  const observation = await waitFor(evaluate, `(${analysisMoveObservation.toString()})(${JSON.stringify(move)},${deadline})`, timeout);
  if (observation.expired) throw Error(`No current-FEN worker/PV progress observed within ${timeout}ms of user move`);
  return observation;
}

export async function headerNavigate(evaluate, route, mark = () => {}) {
  const id = await evaluate(`window.__routePerf.documentId`);
  mark('route:' + route);
  await click(evaluate, `.site-header nav.primary a[href="/app/${route}/"]`);
  await waitFor(
    evaluate,
    `location.pathname === '/app/${route}/' && !!document.querySelector('#ground cg-board') && ${route === 'analysis' ? '!!window.__analysisSchedulingState' : route === 'play' ? '!!window.__playUserMove' : "!!document.querySelector('#arenaSeatList .seat-fam')"}`,
  );
  if ((await evaluate(`window.__routePerf.documentId`)) !== id) throw Error('Header link caused a full navigation; client-side journey invalid');
  mark('route-ready');
}

export async function runJourney(evaluate, name, mark = () => {}, pause = sleep) {
  const evidence = { moves: [], navigation: [] };
  if (name.startsWith('analysis')) {
    await analysisSetup(evaluate, name === 'analysis-four');
    mark('analysis');
    if (!(await evaluate(`window.__analysisSchedulingState().analyzing`))) await click(evaluate, '#analyze');
    await waitFor(evaluate, `window.__analysisSchedulingState().analyzing`);
    const firstMove = await userMove(evaluate, true);
    evidence.moves.push(firstMove);
    const windowMs = 60000;
    evidence.analysisObservation = await requireAnalysisForMove(evaluate, firstMove, windowMs);
    await waitFor(evaluate, `performance.now() >= ${firstMove.started + windowMs}`, windowMs + 15000);
    if (name === 'analysis-four') {
      for (let i = 0; i < 10; i++) {
        const move = await userMove(evaluate, true, i);
        evidence.moves.push(move);
        await requireAnalysisForMove(evaluate, move);
      }
    }
  } else if (name.startsWith('play')) {
    await waitFor(evaluate, `!!window.__playUserMove && !document.querySelector('#engineSelect')?.disabled`);
    await setControl(evaluate, '#colorSelect', 'white');
    await setControl(evaluate, '#engineSelect', name === 'play-maia3' ? 'maia3' : 'lc0-small');
    await setControl(evaluate, '#levelSelect', '0');
    await waitFor(evaluate, `/is ready/.test(document.querySelector('#engineNote')?.textContent)`);
    await click(evaluate, '#newGame');
    for (let i = 0; i < 10; i++) {
      await waitFor(evaluate, PLAY_TURN);
      const sans = await evaluate(`[...document.querySelectorAll('#moveList .san')].map(el => el.textContent)`);
      evidence.moves.push(await userMove(evaluate, false, i, chooseHumanMove(sans)));
      await waitFor(evaluate, `document.querySelectorAll('#moveList .san').length >= ${(i + 1) * 2} && ${PLAY_TURN}`);
    }
    evidence.plies = await evaluate(`document.querySelectorAll('#moveList .san').length`);
  } else if (name.startsWith('arena')) {
    await waitFor(evaluate, `document.querySelectorAll('.seat-fam').length === 2 && !document.querySelector('#start')?.disabled`);
    for (const [i, family, variant] of name === 'arena-cpu'
      ? [
          [0, 'sf', 'lite'],
          [1, 'berserk', null],
        ]
      : [
          [0, 'lc0', 'small'],
          [1, 'sf', 'lite'],
        ]) {
      await setControl(evaluate, `.seat-fam[data-seat="${i}"]`, family);
      if (variant) await setControl(evaluate, `.seat-var[data-seat="${i}"]`, variant);
    }
    await setControl(evaluate, '#budgetModeSelect', 'movetime');
    await setControl(evaluate, '#movetimeInput', '100');
    await setControl(evaluate, '#gamesInput', '1');
    await setControl(evaluate, '#arenaConcurrencySelect', '1');
    await click(evaluate, '#start');
    await waitFor(evaluate, `document.querySelector('#main')?.dataset.phase === 'running'`);
    await waitFor(
      evaluate,
      `document.querySelector('#main')?.dataset.phase !== 'running' && document.querySelectorAll('#gameChips button').length === 1`,
      240000,
    );
    evidence.result = await evaluate(`document.querySelector('#gameChips').textContent`);
    evidence.log = await evaluate(`document.querySelector('#log').textContent`);
    if (arenaGameFailed(evidence.log)) throw Error(`Arena did not complete a valid game: ${evidence.log}`);
    if (/failed|error/i.test(await evaluate(`document.querySelector('#message').textContent`))) throw Error('Arena game failed');
  } else {
    await analysisSetup(evaluate);
    for (let cycle = 0; cycle < 3; cycle++) {
      for (const route of ['play', 'arena', 'analysis']) {
        await headerNavigate(evaluate, route, mark);
        evidence.navigation.push({ route, workers: await evaluate(`window.__routePerf.workers`) });
        await pause(1000);
      }
    }
  }
  mark('settle');
  await pause(20000);
  return evidence;
}
