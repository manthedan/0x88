import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import vm from 'node:vm';
import { coverage, parseArgs } from '../scripts/feature_walk.mjs';
import {
  boardMoveExpression,
  chooseAction,
  chooseValue,
  createRng,
  eligibilityExpression,
  evaluateInvariants,
  parseSeed,
  routeActions,
  routeInvariants,
  routeOwnedIds,
  snapshotExpression,
} from '../scripts/lib/feature_walk_core.mjs';

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

test('boardMove clicks an own piece and a legal non-promotion destination', () => {
  const clicks = [];
  let selected = null;
  const cell = (square, label) => ({
    dataset: { square },
    getAttribute: () => (selected === 'e2' && ['e3', 'e4'].includes(square) ? `${label}, legal destination` : label),
    click() {
      clicks.push(square);
      selected = selected === square ? null : square;
    },
  });
  const cells = [cell('e2', 'e2, White pawn'), cell('e7', 'e7, Black pawn'), cell('e3', 'e3, empty'), cell('e4', 'e4, empty')];
  const host = {
    querySelectorAll: (selector) => (selector === '[data-square]' ? cells : cells.filter((c) => c.getAttribute().includes('legal destination'))),
  };
  const result = vm.runInNewContext(boardMoveExpression("'White'", 0), { document: { querySelector: () => host } });
  assert.equal(result.move, 'e2e3');
  assert.deepEqual(clicks, ['e2', 'e3']);
  selected = null;
  const none = vm.runInNewContext(boardMoveExpression("'Black'", 0.5), { document: { querySelector: () => host } });
  assert.match(none.error, /no legal non-promotion move for Black/);
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
