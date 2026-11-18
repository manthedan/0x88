// UAT journeys. Selectors, readiness checks and move helpers come from the perf-route journeys,
// so a UI change that breaks one breaks both. Each journey is a list of labelled steps; the
// recorder captures a frame at every step boundary on top of its fixed cadence.
import { chooseHumanMove, playHumanTurn } from '../perf_routes/human.mjs';
import { arenaGameFailed, MIDGAME, setControl, userMove, waitFor } from '../perf_routes/journeys.mjs';

const PLAY_TURN = `(${playHumanTurn.toString()})(document.querySelector('#status')?.textContent)`;
// White to move, a7-a8 promotes; the engine has only a king, so the game outlives the promotion.
export const PROMOTION_FEN = '8/P5k1/8/8/8/8/6K1/8 w - - 0 1';
const ANALYSIS_READY = `!!document.querySelector('#ground cg-board') && !!window.__analysisSchedulingState`;
const PLAY_READY = `!!document.querySelector('#ground cg-board') && !!window.__playUserMove`;
const ARENA_READY = `!!document.querySelector('#ground cg-board') && !!document.querySelector('#arenaSeatList .seat-fam')`;
const HOME_READY = `location.pathname === '/' && !!document.querySelector('#hero-h1')`;
const GAME_OVER = `!!document.querySelector('#gameOverCard') && !document.querySelector('#gameOverCard').hidden`;
const ROWS = `[...document.querySelectorAll('#engineList .engine-row')].map(r => [r.querySelector('.row-fam')?.value, r.querySelector('.row-var')?.value])`;
const SAN_COUNT = `document.querySelectorAll('#moveList .san').length`;
const MANAGER_OPEN = `!!document.querySelector('#engineManager')?.open`;
const STOP_ENABLED = `!!document.querySelector('#stop') && !document.querySelector('#stop').disabled`;
const ANALYZE_ENABLED = `!!document.querySelector('#analyze') && !document.querySelector('#analyze').disabled`;
// One entry per engine card: a principal variation from either backend (worker or main-thread fallback), or its error line.
const ENGINE_RESULTS = `[...document.querySelectorAll('#engineSummaries .engine-card')].map(c => ({ name: c.getAttribute('aria-label') ?? '', error: c.querySelector('.line-error')?.textContent ?? null, pv: [...c.querySelectorAll('.result-line[data-pv]')].some(r => !!r.dataset.pv) }))`;
const VISIBLE = (selector) => `(() => { const e = document.querySelector('${selector}'); return !!e && e.getBoundingClientRect().width > 0; })()`;
// BT4's card: a principal variation for the position on the board, or the engine's error line.
const BT4_CARD = `[...document.querySelectorAll('#engineSummaries .engine-card')].find(c => /BT4/.test(c.getAttribute('aria-label') ?? ''))`;
const BT4_RESULT = `(() => { const card = ${BT4_CARD}; if (!card) return null; const error = card.querySelector('.line-error'); if (error) return { error: error.textContent }; const fen = window.__analysisSchedulingState().fen; return [...card.querySelectorAll('.result-line[data-pv]')].some(r => r.dataset.pv && r.dataset.pvFen === fen) ? { pv: true } : null; })()`;

/**
 * Click the visible header link, or the phone menu's link when the header nav is collapsed.
 * `query` (the entry's ?memoryClass override) is put on the link first: the bare link would drop the
 * memory policy, and Analysis reloads when it remounts at a different URL.
 */
async function navigateVia(ctx, href, ready, query = '') {
  const follow = async (selector) => {
    if (query) await ctx.evaluate(`(document.querySelector('${selector}[href="${href}"]').setAttribute('href', ${JSON.stringify(href + query)}), true)`);
    await ctx.click(`${selector}[href="${href + query}"]`);
  };
  const documentId = await ctx.evaluate('window.__uat?.documentId');
  const menu = await ctx.evaluate(
    `(() => { const b = document.querySelector('.site-header .menu-toggle'); return !!b && getComputedStyle(b).display !== 'none' && b.getBoundingClientRect().width > 0; })()`,
  );
  if (menu) {
    await ctx.click('.site-header .menu-toggle');
    await waitFor(ctx.evaluate, `!!document.querySelector('.mobile-nav-menu a[href="${href}"]')`, 10000);
    await ctx.pause(400); // let the reviewer see the open menu
    await follow('.mobile-nav-menu a');
  } else await follow('.site-header nav.primary a');
  await waitFor(ctx.evaluate, ready, 60000);
  if ((await ctx.evaluate('window.__uat?.documentId')) !== documentId)
    ctx.note('warning', `Navigation to ${href} reloaded the document (expected client-side routing)`);
}

async function ensureWhite(ctx) {
  // The visible colour chips drive the visually hidden #colorSelect.
  if (await ctx.evaluate(`!!document.querySelector('.color-choice button[data-color="white"]')`)) await ctx.click('.color-choice button[data-color="white"]');
  else await setControl(ctx.evaluate, '#colorSelect', 'white');
}

// A handed-off position defers loading the engine until Play is pressed ("Loads on Play"),
// so only a normal game can wait for readiness before starting.
async function chooseOpponent(ctx, { waitReady = true } = {}) {
  await waitFor(ctx.evaluate, `${PLAY_READY} && !document.querySelector('#engineSelect')?.disabled`, 120000);
  await ensureWhite(ctx);
  await setControl(ctx.evaluate, '#engineSelect', 'lc0-small');
  await setControl(ctx.evaluate, '#levelSelect', '0');
  if (waitReady) await waitFor(ctx.evaluate, `/is ready/.test(document.querySelector('#engineNote')?.textContent)`, 180000);
}

// Play also shows the game-over card when the engine forfeits with an illegal move or no move at all.
async function waitForEngine(ctx, expression, timeoutMs) {
  await waitFor(ctx.evaluate, `${expression} || ${GAME_OVER}`, timeoutMs);
  if (!(await ctx.evaluate(GAME_OVER))) return;
  const reason = await ctx.evaluate(`document.querySelector('#resultReason')?.textContent ?? ''`);
  if (/illegal move|returned no move/i.test(reason)) throw Error(`Engine forfeited: ${reason}`);
}

async function startGame(ctx) {
  // Same branch as the production smoke: a started game asks to confirm the restart.
  if (await ctx.evaluate(`!!document.querySelector('#restartBanner') && !document.querySelector('#restartBanner').hidden`)) await ctx.click('#confirmRestart');
  else await ctx.click('#newGame');
  await waitForEngine(ctx, PLAY_TURN, 180000);
}

async function playGame(ctx) {
  await ctx.step('Play ready', () => waitFor(ctx.evaluate, PLAY_READY, 120000));
  await ctx.step('choose Lc0 Small, level 0, White', () => chooseOpponent(ctx));
  await ctx.step('start game', () => startGame(ctx));
  for (let i = 0; i < 5; i++) {
    if (await ctx.evaluate(GAME_OVER)) break;
    await ctx.step(`user move ${i + 1}`, async () => {
      const sans = await ctx.evaluate(`[...document.querySelectorAll('#moveList .san')].map(el => el.textContent)`);
      await userMove(ctx.evaluate, false, i, chooseHumanMove(sans));
    });
    await ctx.step(`engine reply ${i + 1}`, () => waitForEngine(ctx, `(${SAN_COUNT} >= ${(i + 1) * 2} && ${PLAY_TURN})`, 120000));
  }
  await ctx.step('settle', () => ctx.pause(3000));
}

async function playPromotion(ctx) {
  await ctx.step('Analysis ready with promotion position', async () => {
    await waitFor(ctx.evaluate, `${ANALYSIS_READY} && window.__analysisSchedulingState().fen.split(' ')[0] === '${PROMOTION_FEN.split(' ')[0]}'`, 120000);
    await ctx.pause(2000);
  });
  await ctx.step('Play from here', async () => {
    // Up to 600 px the action lives in the closed Board tools disclosure.
    if (await ctx.evaluate(`${VISIBLE('#boardToolsButton')} && document.querySelector('#boardToolsButton').getAttribute('aria-expanded') !== 'true'`)) {
      await ctx.click('#boardToolsButton');
      await waitFor(ctx.evaluate, VISIBLE('#playFromHere'), 10000);
      await ctx.pause(600); // let the reviewer see the open menu
    }
    await ctx.click('#playFromHere');
    await waitFor(ctx.evaluate, `location.pathname === '/app/play/' && ${PLAY_READY}`, 60000);
  });
  await ctx.step('choose Lc0 Small, level 0, White', () => chooseOpponent(ctx, { waitReady: false }));
  await ctx.step('press Play (engine loads)', () => startGame(ctx));
  await ctx.step('drag a7-a8: promotion picker', async () => {
    await ctx.evaluate(`(window.__playUserMove('a7', 'a8'), true)`);
    await waitFor(ctx.evaluate, `!!document.querySelector('.promotion-overlay .promo-tile[data-promo="q"]')`, 10000);
    await ctx.pause(1200); // hold the overlay on screen
  });
  // Count from before each human move: a fast reply may already be on the list when the move lands.
  let before = 0;
  const engineReply = () => waitForEngine(ctx, `(${SAN_COUNT} >= ${before + 2} && ${PLAY_TURN})`, 120000);
  await ctx.step('pick queen', async () => {
    before = await ctx.evaluate(SAN_COUNT);
    await ctx.click('.promotion-overlay .promo-tile[data-promo="q"]');
    await waitFor(ctx.evaluate, `${SAN_COUNT} > ${before}`, 10000);
  });
  await ctx.step('engine reply after promotion', engineReply);
  for (let i = 0; i < 2; i++) {
    if (await ctx.evaluate(GAME_OVER)) break;
    await ctx.step(`user move ${i + 1} after promotion`, async () => {
      before = await ctx.evaluate(SAN_COUNT);
      await userMove(ctx.evaluate, false, i);
    });
    await ctx.step(`engine reply ${i + 1} after promotion`, engineReply);
  }
  await ctx.step('settle', () => ctx.pause(3000));
}

async function openManager(ctx) {
  if (!(await ctx.evaluate(MANAGER_OPEN))) await ctx.click('#manageEngines');
  await waitFor(ctx.evaluate, MANAGER_OPEN, 10000);
}
async function closeManager(ctx) {
  if (await ctx.evaluate(MANAGER_OPEN)) await ctx.click('#closeEngineManager');
  await waitFor(ctx.evaluate, `!(${MANAGER_OPEN})`, 10000);
}

async function ensureActiveSearch(ctx) {
  await waitFor(ctx.evaluate, `${STOP_ENABLED} || ${ANALYZE_ENABLED}`, 60000);
  if (await ctx.evaluate(STOP_ENABLED)) return;
  ctx.note('info', 'No search running; pressed Analyze to start one');
  await ctx.click('#analyze');
  await waitFor(ctx.evaluate, STOP_ENABLED, 30000);
}

/** Every engine card has a line or an error; any error fails the step with that engine's message. */
async function waitForEveryEngine(ctx, count, timeoutMs) {
  await waitFor(
    ctx.evaluate,
    `(() => { const cards = ${ENGINE_RESULTS}; return cards.length >= ${count} && cards.every(c => c.pv || c.error); })()`,
    timeoutMs,
  );
  const failed = (await ctx.evaluate(ENGINE_RESULTS)).filter((c) => c.error);
  if (failed.length) throw Error(failed.map((c) => `${c.name}: ${c.error}`).join('; '));
}

async function analysisEngines(ctx) {
  await ctx.step('two engines streaming', async () => {
    await waitFor(ctx.evaluate, `${ANALYSIS_READY} && document.querySelectorAll('#engineList .engine-row').length >= 2`, 120000);
    // Below 768 px only one results view shows; the steps below are about the engine cards.
    if (await ctx.evaluate(`innerWidth < 768 && ${VISIBLE('#enginesTab')} && document.querySelector('#enginesTab').getAttribute('aria-selected') !== 'true'`))
      await ctx.click('#enginesTab');
    ctx.note('info', `default rows ${JSON.stringify(await ctx.evaluate(ROWS))}`);
    // Analyze stays disabled while the first backend loads; auto-analysis may start on its own.
    await waitFor(ctx.evaluate, `window.__analysisSchedulingState().analyzing || ${ANALYZE_ENABLED}`, 180000);
    if (!(await ctx.evaluate(`window.__analysisSchedulingState().analyzing`))) await ctx.click('#analyze');
    await waitForEveryEngine(ctx, 2, 180000);
    await ctx.pause(5000);
  });
  await ctx.step('open engine manager', () => openManager(ctx));
  await ctx.step('add engine (Berserk)', async () => {
    const before = await ctx.evaluate(`document.querySelectorAll('#engineList .engine-row').length`);
    await ctx.click('#addEngineInManager');
    await waitFor(ctx.evaluate, `document.querySelectorAll('#engineList .engine-row').length === ${before + 1}`, 10000);
    await setControl(ctx.evaluate, `.row-fam[data-i="${before}"]`, 'berserk');
    await ctx.pause(1000);
  });
  await ctx.step('close manager, three engines streaming', async () => {
    await closeManager(ctx);
    await waitForEveryEngine(ctx, 3, 180000);
    await ctx.pause(3000);
  });
  await ctx.step('switch Lc0 Small to BT4 (consent dialog)', async () => {
    await openManager(ctx);
    const option = await ctx.evaluate(
      `(() => { const o = document.querySelector('.row-var[data-i="0"] option[value="bt4"]'); return o ? (o.disabled ? 'disabled' : 'ok') : 'missing'; })()`,
    );
    if (option !== 'ok') throw Error(`BT4 option is ${option} in row 1`);
    await setControl(ctx.evaluate, '.row-var[data-i="0"]', 'bt4');
    await waitFor(
      ctx.evaluate,
      `!!document.querySelector('dialog.large-net-consent')?.open || document.querySelector('.row-var[data-i="0"]')?.value === 'bt4'`,
      30000,
    );
    if (await ctx.evaluate(`!!document.querySelector('dialog.large-net-consent')?.open`)) {
      await ctx.pause(1200); // hold the dialog on screen
      await ctx.click('dialog.large-net-consent .large-net-consent-actions button.primary');
    } else ctx.note('warning', 'No large-net consent dialog appeared for BT4');
    await waitFor(ctx.evaluate, `document.querySelector('.row-var[data-i="0"]')?.value === 'bt4'`, 30000);
  });
  await ctx.step('BT4 download and first lines', async () => {
    await closeManager(ctx);
    // The download may start a moment after consent; a cached model shows no progress at all.
    await waitFor(ctx.evaluate, `document.querySelector('#downloadProgress')?.hidden === false`, 20000).catch(() =>
      ctx.note('info', 'No BT4 download progress shown (cached model?)'),
    );
    await waitFor(ctx.evaluate, `(() => { const p = document.querySelector('#downloadProgress'); return !p || p.hidden; })()`, 300000);
    // A hidden indicator also follows a failed init, and the CPU fallback can still be searching:
    // removing Berserk next must not race BT4's first result.
    await waitFor(ctx.evaluate, `!!${BT4_RESULT}`, 300000);
    const result = await ctx.evaluate(BT4_RESULT);
    if (result.error) throw Error(`BT4 failed: ${result.error}`);
    await ctx.pause(3000);
  });
  await ctx.step('remove Berserk', async () => {
    await openManager(ctx);
    const index = (await ctx.evaluate(ROWS)).findIndex(([family]) => family === 'berserk');
    if (index < 0) throw Error('Berserk row missing');
    await ctx.click(`.row-rm[data-i="${index}"]`);
    await waitFor(ctx.evaluate, `!${ROWS}.some(([family]) => family === 'berserk')`, 10000);
    await closeManager(ctx);
    await ctx.pause(3000);
  });
  // A refresh after removing an engine can reuse cached results and finish at once, so Stop
  // follows a board move (a fresh search) and presses Analyze first if nothing is running.
  await ctx.step('board move, then Stop mid-search', async () => {
    await userMove(ctx.evaluate, true);
    await ensureActiveSearch(ctx);
    await ctx.pause(1000); // progress on the Stop button
    await ensureActiveSearch(ctx);
    await ctx.click('#stop');
    await waitFor(ctx.evaluate, `!(${STOP_ENABLED})`, 15000);
    await ctx.pause(2500);
  });
  await ctx.step('Start (Analyze)', async () => {
    await waitFor(ctx.evaluate, ANALYZE_ENABLED, 30000);
    await ctx.click('#analyze');
    await waitFor(ctx.evaluate, `window.__analysisSchedulingState().analyzing`, 30000);
    await ctx.pause(5000);
  });
  await ctx.step('switch result tabs', async () => {
    for (const tab of ['#enginesTab', '#movesTab', '#reviewTab', '#enginesTab']) {
      const visible = await ctx.evaluate(`(() => { const t = document.querySelector('${tab}'); return !!t && t.getBoundingClientRect().width > 0; })()`);
      if (!visible) continue;
      await ctx.click(tab);
      await ctx.pause(1200);
    }
  });
  await ctx.step('settle', () => ctx.pause(3000));
}

async function arenaMatch(ctx) {
  await ctx.step('Arena ready', () =>
    waitFor(ctx.evaluate, `${ARENA_READY} && document.querySelectorAll('.seat-fam').length === 2 && !document.querySelector('#start')?.disabled`, 120000),
  );
  await ctx.step('configure Lc0 Small vs Stockfish lite, 1 game, 100 ms', async () => {
    for (const [seat, family, variant] of [
      [0, 'lc0', 'small'],
      [1, 'sf', 'lite'],
    ]) {
      await setControl(ctx.evaluate, `.seat-fam[data-seat="${seat}"]`, family);
      await setControl(ctx.evaluate, `.seat-var[data-seat="${seat}"]`, variant);
    }
    await setControl(ctx.evaluate, '#budgetModeSelect', 'movetime');
    await setControl(ctx.evaluate, '#movetimeInput', '100');
    await setControl(ctx.evaluate, '#gamesInput', '1');
    await setControl(ctx.evaluate, '#arenaConcurrencySelect', '1');
    await ctx.pause(800);
  });
  await ctx.step('start match', async () => {
    await ctx.click('#start');
    await waitFor(ctx.evaluate, `document.querySelector('#main')?.dataset.phase === 'running'`, 60000);
  });
  await ctx.step('game running', () =>
    waitFor(
      ctx.evaluate,
      `document.querySelector('#main')?.dataset.phase !== 'running' && document.querySelectorAll('#gameChips button').length === 1`,
      300000,
    ),
  );
  await ctx.step('finished match', async () => {
    await ctx.pause(4000);
    // Illegal moves and timeouts also produce a result chip; the perf gate's check tells them apart.
    const { chips, log, message } = await ctx.evaluate(
      `({ chips: document.querySelector('#gameChips')?.textContent ?? '', log: document.querySelector('#log')?.textContent ?? '', message: document.querySelector('#message')?.textContent ?? '' })`,
    );
    ctx.note('info', `result ${chips}`);
    if (arenaGameFailed(log) || /failed|error/i.test(message)) throw Error(`Arena game did not complete validly: ${message} ${log.slice(-300)}`);
  });
}

async function routeTour(ctx) {
  let query = '';
  await ctx.step('Analysis ready', async () => {
    await waitFor(ctx.evaluate, ANALYSIS_READY, 120000);
    query = await ctx.evaluate('location.search');
    await ctx.pause(3000);
  });
  for (const [label, href, ready] of [
    ['Play', '/app/play/', PLAY_READY],
    ['Arena', '/app/arena/', ARENA_READY],
    ['Home', '/', HOME_READY],
    ['Analysis', '/app/analysis/', ANALYSIS_READY],
  ]) {
    await ctx.step(`navigate to ${label}`, async () => {
      await navigateVia(ctx, href, ready, query);
      await ctx.pause(2500);
    });
  }
}

export const JOURNEYS = {
  'play-game': { entry: '/app/play/', run: playGame, about: 'Five moves vs Lc0 Small level 0 from the start position' },
  'play-promotion': {
    entry: `/app/analysis/?fen=${encodeURIComponent(PROMOTION_FEN)}`,
    run: playPromotion,
    about: 'Analysis → Play from here → promote a7-a8 through the picker → keep playing',
  },
  'analysis-engines': {
    entry: `/app/analysis/?fen=${encodeURIComponent(MIDGAME)}`,
    run: analysisEngines,
    about: 'Two engines streaming, add Berserk, Lc0 Small → BT4 via consent, remove, Stop/Start, move, tabs',
  },
  'arena-match': { entry: '/app/arena/', run: arenaMatch, about: 'One 100 ms/move game, Lc0 Small vs Stockfish lite' },
  'route-tour': { entry: '/app/analysis/', run: routeTour, about: 'Client-side header navigation Analysis → Play → Arena → Home → Analysis' },
};
