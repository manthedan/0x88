import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { coverage, parseArgs } from '../scripts/feature_walk.mjs';
import {
  boardMoveExpression,
  chooseAction,
  chooseTargetIndex,
  chooseValue,
  createCheckLog,
  createRng,
  eligibilityExpression,
  evaluateInvariants,
  landedOnDeclaredRoute,
  locateRoute,
  parseSeed,
  pollUntil,
  replayCommand,
  routeActions,
  routeInvariants,
  routeOwnedIds,
  shellQuote,
  snapshotExpression,
  usableNodes,
  walkStoragePresets,
} from '../scripts/lib/feature_walk_core.mjs';
import { FEATURES, ROUTES, routeForPath } from '../src/features/registry.ts';
import { ARENA_SEAT_STORAGE_KEY, engineResourceProfile } from '../src/lc0/engineCatalog.ts';
import { loadPlayPreferences, PLAY_PREFERENCES_STORAGE_KEY } from '../src/lc0/playPreferences.ts';

const draws = (rng, n) => Array.from({ length: n }, () => rng());

test('seeds are reproducible: same seed, same sequence; strings hash stably', () => {
  assert.deepEqual(draws(createRng(42), 5), draws(createRng('42'), 5));
  assert.notDeepEqual(draws(createRng(42), 5), draws(createRng(43), 5));
  assert.equal(parseSeed('ci-nightly'), parseSeed('ci-nightly'));
  assert.notEqual(parseSeed('ci-nightly'), parseSeed('ci-nightlz'));
  for (const value of draws(createRng('range'), 1000)) assert.ok(value >= 0 && value < 1);
});

test('chooseAction respects weights and returns null when nothing is eligible', () => {
  const heavy = { feature: 'f', action: { name: 'heavy', weight: 3 } };
  const light = { feature: 'f', action: { name: 'light' } };
  assert.equal(
    chooseAction(() => 0, [heavy, light]),
    heavy,
  );
  assert.equal(
    chooseAction(() => 0.74, [heavy, light]),
    heavy,
  );
  assert.equal(
    chooseAction(() => 0.76, [heavy, light]),
    light,
  );
  assert.equal(
    chooseAction(() => 0.5, []),
    null,
  );
  assert.equal(
    chooseAction(() => 0.5, [{ feature: 'f', action: { name: 'off', weight: 0 } }]),
    null,
  );
  const rng = createRng(7);
  const picks = Array.from({ length: 4000 }, () => chooseAction(rng, [heavy, light]));
  const heavyShare = picks.filter((pick) => pick === heavy).length / picks.length;
  assert.ok(Math.abs(heavyShare - 0.75) < 0.03, `heavy share ${heavyShare}`);
});

test('chooseValue only offers values the page has and prefers a change', () => {
  assert.equal(
    chooseValue(() => 0, ['maia3', 'lc0-small', 'sf-lite'], ['lc0-small', 'berserk'], 'berserk'),
    'lc0-small',
  );
  assert.equal(
    chooseValue(() => 0, ['1', '3'], null, '1'),
    '3',
  );
  assert.equal(
    chooseValue(() => 0.99, ['1'], null, '1'),
    '1',
  );
  assert.equal(
    chooseValue(() => 0, ['maia3'], ['berserk'], 'berserk'),
    null,
  );
});

test('route actions and invariants come from the features on that route', () => {
  const play = routeActions('play').map(({ feature, action }) => `${feature}/${action.name}`);
  assert.ok(play.includes('play.board/move'));
  assert.ok(play.includes('nav/go:analysis'));
  assert.ok(play.includes('consent.largeNet/decline'));
  assert.ok(!play.some((name) => name.startsWith('analysis.')));
  assert.deepEqual(
    new Set(routeInvariants('analysis')),
    new Set(['noPageErrors', 'noConsoleErrors', 'noHorizontalOverflow', 'uniqueIds', 'noFullReload', 'routeReady', 'ownedIdsPresent']),
  );
  assert.ok(routeOwnedIds('arena').includes('start'));
  assert.ok(!routeOwnedIds('arena').includes('newGame'));
});

function fakeNode({ id, visible = true, disabled = false, inModal = false, type, labels = [] }) {
  return {
    id,
    type,
    disabled,
    labels,
    inModal,
    checkVisibility: () => visible,
    getAttribute: () => null,
    closest: () => null,
  };
}

test('eligibility requires visible, enabled matches inside an open modal and a truthy whenExpr', () => {
  const nodes = {
    '#shown': [fakeNode({ id: 'shown' })],
    '#hidden': [fakeNode({ id: 'hidden', visible: false })],
    '#disabled': [fakeNode({ id: 'disabled', disabled: true })],
    '#toggle': [fakeNode({ id: 'toggle', visible: false, type: 'checkbox', labels: [fakeNode({ id: 'label' })] })],
    '#inDialog': [fakeNode({ id: 'inDialog', inModal: true })],
  };
  const candidates = [
    { action: { when: '#shown' } },
    { action: { when: '#hidden' } },
    { action: { when: '#disabled' } },
    { action: { when: '#toggle' } },
    { action: { when: '#shown', whenExpr: 'flag === 2' } },
    { action: { when: '#shown', whenExpr: 'missing.property' } },
    { action: { when: '#absent' } },
  ];
  const run = (modal) =>
    vm.runInNewContext(eligibilityExpression(candidates), {
      flag: 2,
      document: {
        querySelector: (selector) => (selector === 'dialog:modal' ? modal : null),
        querySelectorAll: (selector) => nodes[selector] ?? [],
      },
    });
  assert.deepEqual([...run(null)], [true, false, false, true, true, false, false]);
  const modal = { contains: (node) => node.inModal };
  assert.deepEqual([...run(modal)], [false, false, false, false, false, false, false]);
  assert.deepEqual(
    [
      ...vm.runInNewContext(eligibilityExpression([{ action: { when: '#inDialog' } }]), {
        document: { querySelector: () => modal, querySelectorAll: (s) => nodes[s] },
      }),
    ],
    [true],
  );
});

test('snapshot reports owned ids present, duplicates, widths, and readiness', () => {
  const ids = ['main', 'main', 'status', 'ground'].map((id) => ({ id }));
  const snap = vm.runInNewContext(snapshotExpression(['main', 'status', 'moveList'], 'window.ready'), {
    window: { ready: true, __featureWalkDocumentId: 'doc-1' },
    location: { pathname: '/app/play/' },
    document: { querySelectorAll: () => ids, documentElement: { scrollWidth: 420, clientWidth: 390 } },
  });
  assert.deepEqual(JSON.parse(JSON.stringify(snap)), {
    pathname: '/app/play/',
    documentId: 'doc-1',
    ready: true,
    scrollWidth: 420,
    viewportWidth: 390,
    presentIds: ['main', 'status'],
    duplicateIds: ['main'],
  });
});

// Fake accessible board: every selection change re-renders fresh cells, and a click on a
// stale (detached) cell does nothing, as with the real innerHTML re-render.
function fakeBoard(pieces, moves, selected = null) {
  const state = { selected, played: null, clicks: [], staleClicks: 0 };
  let generation = 0;
  let cells = [];
  const render = () => {
    const current = ++generation;
    cells = ['a7', 'a8', 'e1', 'e2', 'e6', 'e7'].map((square) => {
      const legal = state.selected && moves[state.selected]?.includes(square);
      const label = `${square}, ${pieces[square] ?? 'empty'}${legal ? ', legal destination' : ''}`;
      return {
        dataset: { square },
        getAttribute: (name) => (name === 'aria-selected' ? String(state.selected === square) : label),
        click() {
          if (current !== generation) return void state.staleClicks++;
          state.clicks.push(square);
          if (state.selected && moves[state.selected]?.includes(square)) {
            state.played = state.selected + square;
            state.selected = null;
          } else if (moves[square]) state.selected = state.selected === square ? null : square;
          render();
        },
      };
    });
  };
  render();
  const host = {
    querySelector(selector) {
      if (selector === '[aria-selected="true"]') return cells.find((cell) => cell.getAttribute('aria-selected') === 'true') ?? null;
      return cells.find((cell) => selector === `[data-square="${cell.dataset.square}"]`) ?? null;
    },
    querySelectorAll: (selector) =>
      selector === '[data-square]' ? cells : cells.filter((cell) => cell.getAttribute('aria-label').includes('legal destination')),
  };
  return { state, document: { querySelector: () => host } };
}

const PIECES = { a7: 'White pawn', e1: 'White king', e7: 'Black pawn' };

test('boardMove re-queries squares after each re-render and skips promotion-only pawns', () => {
  const board = fakeBoard(PIECES, { a7: ['a8'], e1: ['e2'] });
  const result = vm.runInNewContext(boardMoveExpression("'White'", 0), { document: board.document });
  assert.equal(result.move, 'e1e2');
  assert.equal(board.state.played, 'e1e2');
  assert.deepEqual(board.state.clicks, ['a7', 'a7', 'e1', 'e2']);
  assert.equal(board.state.staleClicks, 0);
  assert.equal(board.state.selected, null);
});

test('boardMove clears an existing selection first and skips positions without a supported move', () => {
  const board = fakeBoard(PIECES, { a7: ['a8'], e1: ['e2'] }, 'a7');
  assert.equal(vm.runInNewContext(boardMoveExpression("'White'", 0.5), { document: board.document }).move, 'e1e2');
  assert.equal(board.state.clicks[0], 'a7');
  const blocked = fakeBoard(PIECES, { a7: ['a8'] });
  assert.match(vm.runInNewContext(boardMoveExpression("'Black'", 0), { document: blocked.document }).skipped, /no legal non-promotion move for Black/);
  assert.equal(vm.runInNewContext(boardMoveExpression("'White'", 0), { document: blocked.document }).skipped, 'no legal non-promotion move for White');
  assert.equal(blocked.state.selected, null);
});

test('target choice draws only from usable matches', () => {
  // #navFirst, #navPrev usable at the live edge; #navNext, #navLast visible but disabled.
  const flags = [true, true, false, false];
  for (const seed of [1, 2, 3, 4, 5, 6]) assert.ok([0, 1].includes(chooseTargetIndex(createRng(seed), flags, true)));
  assert.equal(
    chooseTargetIndex(() => 0.99, [false, true, false, true], false),
    1,
  );
  assert.equal(
    chooseTargetIndex(() => 0.5, [false, false], true),
    -1,
  );
});

test('usableNodes rejects hidden, disabled, aria-disabled, inert, and outside-modal nodes', () => {
  const node = (overrides = {}) => ({ ...fakeNode({}), ...overrides });
  const nodes = [
    node(),
    node({ checkVisibility: () => false }),
    node({ disabled: true }),
    node({ getAttribute: (name) => (name === 'aria-disabled' ? 'true' : null) }),
    node({ closest: () => ({}) }),
  ];
  const result = vm.runInNewContext(`(${usableNodes.toString()})(nodes)`, { nodes, document: { querySelector: () => null } });
  assert.deepEqual([...result], [true, false, false, false, false]);
});

test('locateRoute requires the walk origin and a registered path', () => {
  const base = 'http://127.0.0.1:5181';
  assert.equal(locateRoute('http://127.0.0.1:5181/app/play/?x=1', base), 'play');
  assert.equal(locateRoute('http://127.0.0.1:5181/docs', base), 'docs');
  assert.equal(locateRoute('https://evil.example/app/play/', base), null);
  assert.equal(locateRoute('http://127.0.0.1:5182/app/play/', base), null);
  assert.equal(locateRoute('http://127.0.0.1:5181/lab/mobile-soak/', base), null);
});

test('replay command keeps every option that changes timing or stopping', () => {
  const args = parseArgs([
    '--base-url',
    'http://127.0.0.1:5181',
    '--steps',
    '12',
    '--start',
    'arena',
    '--viewport',
    'phone',
    '--settle-ms',
    '900',
    '--ready-timeout',
    '5000',
    '--keep-going',
    '--headed',
  ]);
  assert.equal(
    replayCommand(args, 77),
    'node scripts/feature_walk.mjs --base-url http://127.0.0.1:5181 --seed 77 --steps 12 --start arena --viewport phone --settle-ms 900 --ready-timeout 5000 --keep-going --headed',
  );
  const roundTrip = parseArgs(replayCommand(args, 77).split(' ').slice(2));
  for (const key of ['baseUrl', 'steps', 'start', 'viewport', 'settleMs', 'readyTimeout', 'keepGoing', 'headed']) assert.equal(roundTrip[key], args[key], key);
  assert.equal(roundTrip.seed, '77');
});

test('the Play move side follows the human strip, not board orientation', () => {
  const side = routeActions('play').find(({ action }) => action.name === 'move').action.steps[0].side;
  // After Flip the bottom strip shows the engine; the human strip is found by data-player.
  const strips = { '[data-player="you"] [data-meta]': { textContent: 'Black' }, '#bottomStrip [data-meta]': { textContent: '1500 rating · White' } };
  assert.equal(vm.runInNewContext(side, { document: { querySelector: (selector) => strips[selector] ?? null } }), 'Black');
});

const cleanSnap = { pathname: '/app/play/', documentId: 'a', ready: true, scrollWidth: 390, viewportWidth: 390, presentIds: ['status'], duplicateIds: [] };
const allInvariants = ['noPageErrors', 'noConsoleErrors', 'noHorizontalOverflow', 'uniqueIds', 'ownedIdsPresent', 'noFullReload', 'routeReady'];

test('invariants pass on a clean snapshot', () => {
  assert.deepEqual(
    evaluateInvariants(allInvariants, cleanSnap, { route: 'play', expectedIds: ['status'], pageErrors: [], consoleErrors: [], documentIdBefore: 'a' }),
    [],
  );
});

test('each invariant fails on its own fake evidence', () => {
  const failures = evaluateInvariants(
    allInvariants,
    { ...cleanSnap, documentId: 'b', ready: false, scrollWidth: 450, duplicateIds: ['flip'] },
    {
      route: 'play',
      expectedIds: ['status', 'moveList'],
      pageErrors: ['Uncaught TypeError: x is undefined'],
      consoleErrors: [{ level: 'error', text: 'Error: Missing #message' }],
      documentIdBefore: 'a',
    },
  );
  assert.deepEqual(
    failures.map((failure) => failure.invariant),
    allInvariants,
  );
  assert.deepEqual(failures.find((failure) => failure.invariant === 'ownedIdsPresent').detail, ['#moveList']);
  assert.match(failures.find((failure) => failure.invariant === 'noHorizontalOverflow').detail, /450px wide in a 390px/);
});

test('allowed reloads, a 1px rounding overflow, and the ORT WASM fallback warning do not fail', () => {
  const failures = evaluateInvariants(
    allInvariants,
    { ...cleanSnap, documentId: 'b', scrollWidth: 391 },
    {
      route: 'play',
      expectedIds: [],
      consoleErrors: [{ level: 'error', text: 'ORT WebGPU session failed; falling back to WASM' }],
      documentIdBefore: 'a',
      allowsReload: true,
    },
  );
  assert.deepEqual(failures, []);
});

test('CLI: dry run prints registry coverage; a live run needs --base-url', () => {
  const result = spawnSync(process.execPath, ['scripts/feature_walk.mjs', '--dry-run', '--seed', 'abc'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const output = JSON.parse(result.stdout);
  assert.equal(output.seed, parseSeed('abc'));
  assert.deepEqual(output.coverage, coverage());
  assert.ok(output.coverage.arena.actions.includes('arena.tournament/quickMatch'));
  assert.throws(() => parseArgs([]), /--base-url is required/);
  assert.throws(() => parseArgs(['--base-url', 'http://x', '--start', 'lab']), /Unknown --start route/);
  assert.throws(() => parseArgs(['--base-url', 'http://x', '--steps', '1.5']), /Invalid --steps/);
  assert.equal(parseArgs(['--base-url', 'http://127.0.0.1:5181/app/']).baseUrl, 'http://127.0.0.1:5181');
});

// Eligible action names on a route for a fake page where exactly `present` selectors match one usable node.
function eligibleNames(route, present, globals = {}) {
  const candidates = routeActions(route);
  const flags = vm.runInNewContext(eligibilityExpression(candidates), {
    ...globals,
    document: { querySelector: () => null, querySelectorAll: (selector) => (present.includes(selector) ? [fakeNode({})] : []), ...globals.document },
  });
  return candidates.filter((_, i) => flags[i]).map(({ feature, action }) => `${feature}/${action.name}`);
}

test('an open mobile menu offers its links and a close, never a re-toggle that hides them', () => {
  const open = eligibleNames(
    'arena',
    ['.site-header .menu-toggle', '.site-header .menu-toggle[aria-expanded="true"]', '.mobile-nav-menu a[href="/app/play/"]'],
    {
      location: { pathname: '/app/arena/' },
    },
  );
  assert.ok(open.includes('nav/go:play:menuOpen'));
  assert.ok(open.includes('nav/closeMenu'));
  assert.ok(!open.some((name) => name.endsWith(':mobile')));
  const closed = eligibleNames('arena', ['.site-header .menu-toggle', '.site-header .menu-toggle[aria-expanded="false"]'], {
    location: { pathname: '/app/arena/' },
  });
  assert.ok(closed.includes('nav/go:play:mobile'));
  assert.ok(!closed.includes('nav/go:arena:mobile'), 'no navigation to the current route');
});

test('Docs search is reachable on phones by opening the panel first', () => {
  assert.deepEqual(eligibleNames('docs', ['.toc .mobile-search']), ['docs.search/openSearch']);
  assert.ok(eligibleNames('docs', ['#docSearch']).includes('docs.search/search'));
});

test('Play moves are offered only at the live position', () => {
  const move = routeActions('play').find(({ action }) => action.name === 'move').action;
  const eligible = (status) => vm.runInNewContext(`!!(${move.whenExpr})`, { document: { querySelector: () => ({ textContent: status }) } });
  assert.equal(eligible('Your move'), true);
  assert.equal(eligible('You are in check'), true);
  assert.equal(eligible('Viewing 1. e4 — press → or End to return. Live game: Your move.'), false);
  assert.equal(eligible('Maia3 to move'), false);
});

test('walk choices stay reproducible and download-light', () => {
  const steps = FEATURES.flatMap((feature) =>
    feature.actions.flatMap((action) => action.steps.map((step) => ({ name: `${feature.id}/${action.name}`, step }))),
  );
  for (const { name, step } of steps) {
    // Big nets (BT4/t3) load without consent on desktop Play; Random color uses the app's own Math.random.
    for (const value of step.values ?? []) assert.doesNotMatch(String(value), /bt4|(^|-)t3$/, `${name} offers big net ${value}`);
    assert.doesNotMatch(step.target ?? '', /data-color(?!="(white|black)")/, `${name} can pick an unseeded color`);
    // .pv-move inserts moves; the preview has its own control.
    if (name === 'analysis.pvPreview/openPv') assert.match(step.target, /\[data-pv-preview\]/);
  }
});

test('Home and Docs wait for hydration, which the root layout signals', () => {
  for (const route of ['home', 'docs']) {
    const ready = (hydrated) =>
      vm.runInNewContext(`!!(${ROUTES[route].ready})`, { document: { querySelector: () => ({}), documentElement: { hasAttribute: () => hydrated } } });
    assert.equal(ready(false), false, `${route} ready before hydration`);
    assert.equal(ready(true), true);
  }
  assert.match(readFileSync('src/routes/+layout.svelte', 'utf8'), /onMount\([\s\S]*document\.documentElement\.dataset\.hydrated = ''/);
});

test('controls inside a disabled fieldset are not usable', () => {
  const insideDisabledFieldset = { ...fakeNode({}), matches: (selector) => selector === ':disabled' };
  const result = vm.runInNewContext(`(${usableNodes.toString()})(nodes)`, {
    nodes: [insideDisabledFieldset, fakeNode({})],
    document: { querySelector: () => null },
  });
  assert.deepEqual([...result], [false, true]);
});

test('pick actions are eligible whenever any of their targets is usable (e.g. only forward history at ply 0)', () => {
  for (const feature of FEATURES)
    for (const action of feature.actions) {
      const [step] = action.steps;
      if (action.steps.length === 1 && step.do === 'click' && step.pick) assert.equal(action.when, step.target, `${feature.id}/${action.name}`);
    }
  for (const name of ['play.game/browseMoves', 'analysis.toolbar/browseMoves', 'analysis.pvPreview/stepPv', 'arena.moves/review']) {
    const route = name.split('.')[0];
    const { action } = routeActions(route).find((candidate) => `${candidate.feature}/${candidate.action.name}` === name);
    assert.match(action.when, /#(navNext|navForward|pvForward|revNext)\b/, name);
  }
});

test('Arena quick match pins the single start position before Start', () => {
  const { action } = routeActions('arena').find((candidate) => candidate.action.name === 'quickMatch');
  const order = action.steps.map((step) => step.target);
  const opening = action.steps.find((step) => step.target === '#startingPositionSelect');
  assert.deepEqual(opening.values, ['start']);
  assert.ok(order.indexOf('#startingPositionSelect') < order.indexOf('#start'));
});

test('walk storage presets: Maia3 plays its deterministic top move; Arena seats only CPU engines', () => {
  const presets = new Map(walkStoragePresets().map(({ key, value }) => [key, value]));
  assert.ok(presets.has(PLAY_PREFERENCES_STORAGE_KEY));
  const loaded = loadPlayPreferences({ getItem: (key) => presets.get(key) ?? null }, new Set(['maia3']));
  assert.equal(loaded.engineId, 'maia3');
  assert.equal(loaded.maiaStyle, 'argmax');
  const seats = JSON.parse(presets.get(ARENA_SEAT_STORAGE_KEY));
  // Arena restores a roster of two or more seats; quickMatch never touches the families.
  assert.ok(seats.length >= 2);
  for (const seat of seats) assert.equal(engineResourceProfile(seat.family).resourceClass, 'cpu', seat.family);
  const pickSeat = routeActions('arena').find(({ action }) => action.name === 'pickSeat').action;
  for (const family of pickSeat.steps[0].values) assert.equal(engineResourceProfile(family).resourceClass, 'cpu', family);
});

test('navigation is judged against the registry-declared route, not the rendered link', () => {
  const base = 'http://127.0.0.1:5181';
  const toAnalysis = { navigatesTo: 'analysis' };
  assert.equal(landedOnDeclaredRoute(toAnalysis, `${base}/app/analysis/?fen=x`, base), true);
  // A regressed href="#" leaves the page where it was; that must not count as arriving.
  assert.equal(landedOnDeclaredRoute(toAnalysis, `${base}/app/play/#`, base), false);
  assert.equal(landedOnDeclaredRoute(toAnalysis, 'https://evil.example/app/analysis/', base), false);
  assert.equal(landedOnDeclaredRoute({}, `${base}/app/play/#`, base), true);
});

test('every in-app link and navigating button action declares where it lands', () => {
  for (const feature of FEATURES)
    for (const action of feature.actions)
      for (const step of action.steps) {
        const href = step.target?.match(/\ba\b[^\s]*\[href="([^"]+)"\]/)?.[1];
        if (href) assert.equal(action.navigatesTo, routeForPath(href), `${feature.id}/${action.name}`);
      }
  const declared = (route, name) => routeActions(route).find((candidate) => candidate.action.name === name).action.navigatesTo;
  assert.equal(declared('play', 'analyzeGame'), 'analysis');
  assert.equal(declared('analysis', 'playFromHere'), 'play');
  for (const route of ['analysis', 'play', 'arena']) assert.equal(declared('home', `heroCta:${route}`), route);
});

test('Play analyzeGame is offered only as the same-tab link (mid-game it opens a new tab)', () => {
  const { action } = routeActions('play').find((candidate) => candidate.action.name === 'analyzeGame');
  assert.equal(action.when, '#analyzeGame:not([target])');
  assert.equal(action.steps[0].target, action.when);
});

test('pollUntil resolves once the check passes, retries throws, and times out', async () => {
  let calls = 0;
  assert.equal(
    await pollUntil(
      () => {
        calls++;
        if (calls === 1) throw new Error('context destroyed');
        return calls >= 3;
      },
      1000,
      1,
    ),
    true,
  );
  assert.equal(calls, 3);
  assert.equal(await pollUntil(() => false, 20, 5), false);
  // The last sleep stops at the deadline and nothing is evaluated after it.
  let late = 0;
  const started = Date.now();
  assert.equal(
    await pollUntil(
      () => {
        late++;
        return false;
      },
      30,
      5_000,
    ),
    false,
  );
  assert.ok(Date.now() - started < 1_000);
  assert.equal(late, 1);
});

test('pollUntil does not check again when the last timer fires before Date.now() reaches the deadline', async (t) => {
  let now = 1_000;
  t.mock.method(Date, 'now', () => now);
  t.mock.method(globalThis, 'setTimeout', (resolve, ms) => {
    now += ms - 1;
    resolve();
  });
  let checks = 0;
  const result = await pollUntil(
    () => {
      checks++;
      return false;
    },
    30,
    5_000,
  );
  assert.equal(result, false);
  assert.equal(checks, 1);
});

test('check logs hand every entry to exactly one check, including ones logged between steps', () => {
  const log = createCheckLog();
  log.push('during start');
  assert.deepEqual(log.take(), ['during start']);
  log.push('after the check, before the next action');
  log.push('during the action');
  assert.deepEqual(log.take(), ['after the check, before the next action', 'during the action']);
  assert.deepEqual(log.take(), []);
});

test('replay command shell-quotes values (zsh globs IPv6 brackets)', () => {
  assert.equal(shellQuote('http://127.0.0.1:5181'), 'http://127.0.0.1:5181');
  assert.equal(shellQuote('http://[::1]:5181'), "'http://[::1]:5181'");
  assert.equal(shellQuote("it's"), "'it'\\''s'");
  const args = parseArgs(['--base-url', 'http://[::1]:5181/app/']);
  assert.match(replayCommand(args, 5), /^node scripts\/feature_walk\.mjs --base-url 'http:\/\/\[::1\]:5181' --seed 5 /);
  const shell = spawnSync('sh', ['-c', `printf '%s' ${shellQuote("a'b $HOME [x]")}`], { encoding: 'utf8' });
  assert.equal(shell.stdout, "a'b $HOME [x]");
});
