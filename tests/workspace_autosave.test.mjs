import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { gameTreeToPgn, parsePgnGame } from '../src/chess/pgn.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { replayPlayPosition } from '../src/lc0/playPosition.ts';
import { DEFAULT_PLAY_PREFERENCES } from '../src/lc0/playPreferences.ts';
import {
  createWorkspaceAutosave,
  decodeAnalysisSnapshot,
  decodePlaySnapshot,
  hasExplicitWorkspaceInput,
  hasPreviousAnalysis,
  PREVIOUS_ANALYSIS_KEY,
  preservePreviousAnalysis,
  readWorkspace,
  restorePreviousAnalysis,
  restoreTree,
  serializeTree,
  WORKSPACE_MAX_CHARS,
  WORKSPACE_VERSION,
  workspaceStorageKey,
} from '../src/lc0/workspaceAutosave.ts';

const engineIds = new Set(['maia3', 'sf-lite']);
const envelope = (kind, data, version = WORKSPACE_VERSION) => JSON.stringify({ kind, version, savedAt: Date.now(), data });
const analysisData = () => ({ tree: serializeTree(new GameTree()), headers: {}, result: '*' });
const resultSignature = (tree) => [tree.root, ...tree.mainlineFrom()].map((node) => `${node.fen}\u0000${node.move ? moveToUci(node.move) : ''}`).join('\u0001');
const playData = (changes = {}) => ({
  startFen: START_FEN,
  moves: ['e2e4'],
  humanColor: 'w',
  orientation: 'black',
  preferences: { ...DEFAULT_PLAY_PREFERENCES, color: 'random', maiaElo: 1900, maiaTemperature: 0.8, maiaTopP: 0.9 },
  gameOver: null,
  started: true,
  ...changes,
});
const storage = () => {
  const values = new Map();
  let writes = 0;
  return {
    values,
    get writes() {
      return writes;
    },
    getItem: (key) => values.get(key) ?? null,
    setItem(key, value) {
      writes++;
      values.set(key, value);
    },
    removeItem: (key) => values.delete(key),
  };
};

test('reopening an equal linked workspace never overwrites the previous analysis', () => {
  const store = storage();
  const original = envelope('analysis', analysisData());
  store.setItem(workspaceStorageKey('analysis'), original);
  const linked = new GameTree();
  linked.addUci('e2e4');
  assert.equal(preservePreviousAnalysis(store, linked), true);
  store.setItem(workspaceStorageKey('analysis'), envelope('analysis', { ...analysisData(), tree: serializeTree(linked) }));
  linked.toStart();
  assert.equal(preservePreviousAnalysis(store, linked), true);
  assert.equal(store.getItem(PREVIOUS_ANALYSIS_KEY), original);
});

test('equal moves with different PGN metadata preserve the complete original analysis', () => {
  const base = parsePgnGame('[Event "Original"]\n[White "Alice"]\n[Result "1-0"]\n\n1. e4! {centre} e5 (1... c5) 1-0');
  const snapshot = (parsed) => ({
    tree: serializeTree(parsed.tree),
    headers: parsed.tags,
    result: parsed.result,
    resultSignature: resultSignature(parsed.tree),
  });
  for (const change of [
    (parsed) => {
      parsed.tags.White = 'Bob';
    },
    (parsed) => {
      parsed.result = '0-1';
    },
    (parsed) => {
      parsed.tree.root.children[0].comment = 'different';
    },
    (parsed) => {
      parsed.tree.root.children[0].nags = [2];
    },
    (parsed) => {
      parsed.tree.root.children[0].children.pop();
    },
  ]) {
    const store = storage();
    const original = envelope('analysis', snapshot(base));
    store.setItem(workspaceStorageKey('analysis'), original);
    store.setItem(PREVIOUS_ANALYSIS_KEY, envelope('analysis', analysisData()));
    const replacement = { tree: restoreTree(serializeTree(base.tree)), tags: { ...base.tags }, result: base.result };
    change(replacement);
    assert.equal(preservePreviousAnalysis(store, replacement.tree, replacement.tags, replacement.result), true);
    assert.equal(store.getItem(PREVIOUS_ANALYSIS_KEY), original, 'All persisted content must participate in equality');
  }
  const store = storage();
  store.setItem(workspaceStorageKey('analysis'), envelope('analysis', snapshot(base)));
  const older = envelope('analysis', analysisData());
  store.setItem(PREVIOUS_ANALYSIS_KEY, older);
  const equal = { tree: restoreTree(serializeTree(base.tree)), tags: { ...base.tags }, result: base.result };
  equal.tree.toEnd();
  assert.equal(preservePreviousAnalysis(store, equal.tree, Object.fromEntries(Object.entries(equal.tags).reverse()), equal.result), true);
  assert.equal(store.getItem(PREVIOUS_ANALYSIS_KEY), older, 'Header order and current selection do not change the analysis');
});

test('autosave resumes after a smaller save, recovered storage or clearing, and reports later failures', () => {
  const store = storage();
  const pauses = [];
  let resumed = 0;
  const saver = createWorkspaceAutosave(
    'analysis',
    store,
    400,
    (reason) => pauses.push(reason),
    () => resumed++,
  );
  const oversized = { ...analysisData(), headers: { Event: 'x'.repeat(WORKSPACE_MAX_CHARS) } };
  saver.schedule(oversized);
  saver.flush();
  saver.schedule(analysisData());
  saver.flush();
  assert.equal(resumed, 1, 'a smaller analysis clears the paused notice');
  const write = store.setItem;
  store.setItem = () => {
    throw Error('quota');
  };
  saver.schedule({ ...analysisData(), headers: { Event: 'retry' } });
  saver.flush();
  store.setItem = write;
  saver.schedule({ ...analysisData(), headers: { Event: 'retry' } });
  saver.flush();
  assert.equal(resumed, 2, 'successful retry of the same snapshot clears the notice');
  saver.pause('too-large');
  saver.clear();
  assert.equal(resumed, 3, 'reset clears the paused notice');
  saver.pause('storage');
  saver.pause('storage');
  saver.pause('too-large');
  assert.deepEqual(pauses, ['too-large', 'storage', 'too-large', 'storage', 'too-large']);
  saver.clear();
});

test('analysis round trip retains nested variations, comments, headers, result and selected non-mainline node', () => {
  const parsed = parsePgnGame('[Event "My game"]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 {Centre} e5 (1... c5 2. Nf3 (2. Nc3)) 2. Nf3 1-0');
  const { tree, tags: headers, result } = parsed;
  tree.root.comment = 'Before the game';
  tree.goTo(tree.root.children[0].children[1].children[1]);
  const raw = envelope('analysis', { tree: serializeTree(tree), headers, result, resultSignature: resultSignature(tree) });
  const restored = decodeAnalysisSnapshot(raw);
  assert.ok(restored);
  assert.deepEqual(serializeTree(restored.tree), serializeTree(tree));
  assert.equal(restored.tree.current.san, 'Nc3');
  assert.equal(restored.tree.root.comment, 'Before the game');
  assert.deepEqual(restored.headers, headers);
  assert.equal(restored.result, '1-0');
  assert.equal(gameTreeToPgn(restored.tree, restored.headers, restored.result), gameTreeToPgn(tree, headers, result));
  restored.tree.addUci('b8c6');
  assert.equal(new Set(restored.tree.lineTo().map((node) => node.id)).size, 5, 'new nodes get unique ids');
});

test('autosave persists NAGs and variation-leading comments through a page refresh', () => {
  const { tree } = parsePgnGame('{Intro} 1. e4! ({d4 only} 1. d4 $6) e5 $2 *');
  // Assign explicitly as well, so persistence is tested independently of parsing.
  tree.root.children[1].beforeComment = 'd4 only';
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  saver.schedule({ tree: serializeTree(tree), headers: {}, result: '*' });
  saver.flush();
  const restored = readWorkspace('analysis', decodeAnalysisSnapshot, store);
  assert.ok(restored);
  assert.deepEqual(
    restored.tree.mainlineFrom().map((node) => node.nags),
    [[1], [2]],
  );
  assert.deepEqual(restored.tree.root.children[1].nags, [6]);
  assert.equal(restored.tree.root.children[1].beforeComment, 'd4 only');
  assert.equal(gameTreeToPgn(restored.tree), gameTreeToPgn(tree));
});

test('33 shallow PGN cycles import, autosave, restore and export merged deep branches', () => {
  const { tree } = parsePgnGame(`${'Nf3 (Nf3 Nc6) Nf6 (Nf6 Nc3) Ng1 (Ng1 Nc6) Ng8 (Ng8 Nc3) '.repeat(33)}*`, { strict: true });
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  const snapshot = serializeTree(tree);
  saver.schedule({ tree: snapshot, headers: {}, result: '*' });
  saver.flush();
  const restored = readWorkspace('analysis', decodeAnalysisSnapshot, store);
  assert.ok(restored);
  assert.deepEqual(serializeTree(restored.tree), snapshot);
  const reparsed = parsePgnGame(gameTreeToPgn(restored.tree), { strict: true });
  assert.deepEqual(reparsed.warnings, []);
  assert.deepEqual(serializeTree(reparsed.tree), snapshot);
});

test('5000-node nested GameTree autosaves and restores without recursive walks', () => {
  const tree = new GameTree();
  const cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  const alternatives = ['b1c3', 'b8c6', 'b1c3', 'b8c6'];
  for (let i = 0; i < 2500; i++) {
    const parent = tree.current;
    assert.ok(tree.addUci(alternatives[i % 4]));
    tree.goTo(parent);
    assert.ok(tree.addUci(cycle[i % 4]));
  }
  const snapshot = serializeTree(tree);
  assert.equal(snapshot.nodes.length, 5000);
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  saver.schedule({ tree: snapshot, headers: {}, result: '*' });
  saver.flush();
  const restored = readWorkspace('analysis', decodeAnalysisSnapshot, store);
  assert.ok(restored);
  assert.deepEqual(serializeTree(restored.tree), snapshot);
  const reparsed = parsePgnGame(gameTreeToPgn(restored.tree), { strict: true });
  assert.deepEqual(serializeTree(reparsed.tree).nodes, snapshot.nodes);
});

test('old-format version-1 analysis fixture decodes without optional annotations', () => {
  const raw = JSON.stringify({
    kind: 'analysis',
    version: 1,
    savedAt: 1,
    data: {
      headers: { White: 'Legacy' },
      result: '*',
      tree: {
        startFen: START_FEN,
        current: 2,
        rootComment: 'Intro',
        nodes: [
          { parent: 0, uci: 'e2e4', comment: 'Centre' },
          { parent: 1, uci: 'e7e5' },
        ],
      },
    },
  });
  const restored = decodeAnalysisSnapshot(raw);
  assert.ok(restored);
  assert.equal(restored.tree.current.san, 'e5');
  assert.equal(restored.tree.root.comment, 'Intro');
  assert.equal(restored.tree.root.children[0].comment, 'Centre');
  assert.equal(restored.tree.root.children[0].nags, undefined);
  assert.equal(restored.tree.root.children[0].beforeComment, undefined);
});

test('snapshot annotation validation and size limits remain enforced', () => {
  const data = analysisData();
  for (const patch of [
    { nags: '1' },
    { nags: [-1] },
    { nags: [256] },
    { nags: [1.5] },
    { nags: [null] },
    { nags: Array(257).fill(1) },
    { beforeComment: {} },
  ]) {
    const tree = { ...data.tree, nodes: [{ parent: 0, uci: 'e2e4', ...patch }] };
    assert.equal(decodeAnalysisSnapshot(envelope('analysis', { ...data, tree })), null);
  }
  const { tree } = parsePgnGame('1. e4! e5 $2 *');
  tree.root.children[0].beforeComment = 'x'.repeat(WORKSPACE_MAX_CHARS);
  const oversized = { ...data, tree: serializeTree(tree) };
  assert.equal(decodeAnalysisSnapshot(envelope('analysis', oversized)), null);
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  saver.schedule(data);
  saver.flush();
  assert.ok(store.getItem(workspaceStorageKey('analysis')));
  saver.schedule(oversized);
  saver.flush();
  assert.ok(store.getItem(workspaceStorageKey('analysis')));
  const tooManyNodes = { ...data.tree, nodes: Array(5001).fill({ parent: 0, uci: 'e2e4', nags: [1] }) };
  assert.equal(decodeAnalysisSnapshot(envelope('analysis', { ...data, tree: tooManyNodes })), null);
});

test('Analysis autosave never gives an edited mainline ownership of an imported result', () => {
  const { tree, tags: headers, result } = parsePgnGame('[Result "1-0"]\n\n1. e4 1-0');
  const signature = resultSignature(tree);
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  const roundTrip = (resultSignature) => {
    saver.schedule({ tree: serializeTree(tree), headers, result, resultSignature });
    saver.flush();
    return readWorkspace('analysis', decodeAnalysisSnapshot, store);
  };
  assert.equal(roundTrip(signature).result, '1-0', 'an unedited imported line retains its result');
  tree.toEnd();
  assert.ok(tree.addUci('e7e5'));
  const restored = roundTrip(signature);
  assert.equal(restored.result, '*');
  assert.equal(restored.headers.Result, '*');
  const exported = gameTreeToPgn(restored.tree, restored.headers, restored.result);
  assert.match(exported, /1\. e4 e5 \*$/);
  assert.doesNotMatch(exported, /1-0/);
  for (const legacySignature of [undefined, null, 12]) {
    const legacy = roundTrip(legacySignature);
    assert.ok(legacy, 'old or unknown result provenance does not discard the workspace');
    assert.equal(legacy.result, '*');
  }
  tree.back();
  tree.addUci('c7c5');
  assert.equal(roundTrip(signature).result, '*', 'adding variations does not make an edited mainline own the result');
});

test('custom root, castling, en passant and promotion round trip through legal replay', () => {
  for (const [fen, moves] of [
    ['r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', ['e1g1', 'e8c8']],
    ['4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', ['e5d6']],
    ['4k3/P7/8/8/8/8/8/6K1 w - - 0 1', ['a7a8n']],
  ]) {
    const tree = new GameTree(fen);
    for (const uci of moves) assert.ok(tree.addUci(uci));
    assert.deepEqual(serializeTree(restoreTree(serializeTree(tree))), serializeTree(tree));
  }
});

test('analysis rejects truncated JSON, mismatched/old schemas, bad FEN, indices and illegal or duplicate moves', () => {
  for (const raw of ['{', 'null', '[]', envelope('play', analysisData()), envelope('analysis', analysisData(), 0), envelope('analysis', analysisData(), 2)]) {
    assert.equal(decodeAnalysisSnapshot(raw), null);
  }
  const good = analysisData();
  for (const patch of [
    { startFen: 'garbage' },
    { current: -1 },
    { current: 1 },
    { current: 0.5 },
    { nodes: [{ parent: 1, uci: 'e2e4' }] },
    { nodes: [{ parent: 0, uci: 'e2e5' }] },
    {
      nodes: [
        { parent: 0, uci: 'e2e4' },
        { parent: 0, uci: 'e2e4' },
      ],
    },
    { nodes: [{ parent: 0, uci: 'e2e4', comment: {} }] },
  ])
    assert.equal(decodeAnalysisSnapshot(envelope('analysis', { ...good, tree: { ...good.tree, ...patch } })), null);
  assert.equal(decodeAnalysisSnapshot(envelope('analysis', { ...good, headers: { White: 12 } })), null);
  assert.equal(decodeAnalysisSnapshot(' '.repeat(WORKSPACE_MAX_CHARS + 1)), null);
});

test('play round trip retains opponent settings, actual random side and engine-to-move history', () => {
  for (const engineId of engineIds) {
    const data = playData({ preferences: { ...playData().preferences, engineId } });
    const restored = decodePlaySnapshot(envelope('play', data), engineIds);
    assert.ok(restored);
    assert.deepEqual(restored.preferences, data.preferences);
    assert.equal(restored.humanColor, 'w');
    assert.equal(restored.orientation, 'black');
    assert.equal(restored.board.turn, 'b', 'engine turn is retained, not inferred from human colour');
    assert.equal(restored.positions.length, 2);
    assert.deepEqual(restored.sans, ['e4']);
    assert.equal(boardToFen(restored.positions[0]), START_FEN);
  }
});

test('finished games stay finished, including resignation and a terminal board with missing outcome', () => {
  const resigned = playData({ gameOver: { result: '0-1', reason: 'resignation' } });
  assert.deepEqual(decodePlaySnapshot(envelope('play', resigned), engineIds).gameOver, resigned.gameOver);
  const mate = playData({ moves: ['f2f3', 'e7e5', 'g2g4', 'd8h4'] });
  assert.equal(decodePlaySnapshot(envelope('play', mate), engineIds).gameOver.result, '0-1');
  const repeated = playData({ moves: Array(2).fill(['g1f3', 'g8f6', 'f3g1', 'f6g8']).flat() });
  assert.equal(decodePlaySnapshot(envelope('play', repeated), engineIds).gameOver.result, '1/2-1/2');
});

test('play rejects corrupt settings instead of silently changing the opponent or difficulty', () => {
  for (const patch of [{ engineId: 'gone' }, { level: -1 }, { maiaElo: 999999 }, { maiaStyle: 'bad' }, { color: 'bad' }, { maiaTemperature: '1' }]) {
    assert.equal(decodePlaySnapshot(envelope('play', playData({ preferences: { ...playData().preferences, ...patch } })), engineIds), null);
  }
  for (const patch of [{ humanColor: 'black' }, { orientation: 'w' }, { started: 'yes' }, { moves: ['e2e5'] }, { gameOver: { result: '*', reason: 'bad' } }]) {
    assert.equal(decodePlaySnapshot(envelope('play', playData(patch)), engineIds), null);
  }
  assert.equal(decodePlaySnapshot(envelope('play', playData(), 0), engineIds), null);
});

test('a takeback round trip uses only the committed remaining history', () => {
  const data = playData({ moves: ['e2e4', 'e7e5', 'g1f3'] });
  data.moves.splice(-2);
  const restored = decodePlaySnapshot(envelope('play', data), engineIds);
  assert.equal(restored.positions.length, 2);
  assert.deepEqual(restored.sans, ['e4']);
  assert.equal(restored.board.turn, 'b');
});

test('autosave debounces, deduplicates render-only changes, and flushes the latest committed state', async () => {
  const store = storage();
  const saver = createWorkspaceAutosave('play', store, 15);
  saver.schedule(playData());
  saver.schedule(playData({ moves: ['d2d4'] }));
  assert.equal(store.writes, 0);
  await new Promise((resolve) => setTimeout(resolve, 30));
  assert.equal(store.writes, 1);
  saver.schedule(playData({ moves: ['d2d4'] }));
  saver.flush();
  assert.equal(store.writes, 1);
  saver.schedule(playData({ moves: ['d2d4', 'd7d5'] }));
  saver.flush();
  assert.equal(store.writes, 2);
  assert.deepEqual(decodePlaySnapshot(store.getItem(workspaceStorageKey('play')), engineIds).sans, ['d4', 'd5']);
});

test('clearing cancels pending writes and Analysis and Play never overwrite each other', async () => {
  const store = storage();
  const analysis = createWorkspaceAutosave('analysis', store, 10);
  const play = createWorkspaceAutosave('play', store, 10);
  analysis.schedule(analysisData());
  analysis.flush();
  play.schedule(playData());
  play.flush();
  analysis.schedule(analysisData());
  analysis.clear();
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(store.getItem(workspaceStorageKey('analysis')), null);
  assert.ok(store.getItem(workspaceStorageKey('play')));
  play.clear();
});

test('denied/full storage fails safely and invalid saved data is removed', () => {
  const denied = {
    getItem() {
      throw Error('denied');
    },
    setItem() {
      throw Error('denied');
    },
    removeItem() {
      throw Error('denied');
    },
  };
  assert.equal(readWorkspace('analysis', decodeAnalysisSnapshot, denied), null);
  const saver = createWorkspaceAutosave('analysis', denied);
  saver.schedule(analysisData());
  saver.flush();
  saver.clear();
  const store = storage();
  store.setItem(workspaceStorageKey('analysis'), '{broken');
  assert.equal(readWorkspace('analysis', decodeAnalysisSnapshot, store), null);
  assert.equal(store.getItem(workspaceStorageKey('analysis')), null);
  store.setItem(workspaceStorageKey('play'), envelope('play', playData()));
  const full = createWorkspaceAutosave('play', {
    ...store,
    setItem() {
      throw Error('quota');
    },
  });
  full.schedule(playData());
  full.flush();
  assert.ok(store.getItem(workspaceStorageKey('play')), 'last good save survives a failed write');
});

test('oversized work retains the last good saved game', () => {
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  saver.schedule(analysisData());
  saver.flush();
  saver.schedule({ ...analysisData(), headers: { Event: 'x'.repeat(WORKSPACE_MAX_CHARS) } });
  saver.flush();
  assert.ok(store.getItem(workspaceStorageKey('analysis')));
});

test('only consumed Analysis search keys classify as URL workspace input', () => {
  for (const key of ['fen', 'pgn']) assert.equal(hasExplicitWorkspaceInput(`?${key}=abc`), true);
  for (const key of ['game', 'position', 'handoff', 'playhandoff', 'playfrom', 'fromanalysis', 'startfen', 'from', 'play', 'FEN']) {
    assert.equal(hasExplicitWorkspaceInput(`?${key}=payload`), false, key);
  }
  assert.equal(hasExplicitWorkspaceInput('', '#fen=abc'), false);
  assert.equal(hasExplicitWorkspaceInput('?fen=abc&pgn=abc', '', 'play'), false);
  assert.equal(hasExplicitWorkspaceInput('?dev&model=local'), false);
});

test('save, explicit replacement and restore retain the full previous analysis', () => {
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  const { tree } = parsePgnGame('1. e4! {keep} e5 (1... c5 2. Nf3) *');
  tree.toEnd();
  saver.schedule({ tree: serializeTree(tree), headers: { Event: 'Original' }, result: '*' });
  saver.flush();
  const original = store.getItem(workspaceStorageKey('analysis'));
  assert.equal(preservePreviousAnalysis(store), true);
  saver.clear();
  saver.schedule(analysisData());
  saver.flush();
  assert.equal(hasPreviousAnalysis(store), true);
  const restored = restorePreviousAnalysis(store);
  assert.deepEqual(serializeTree(restored.tree), serializeTree(tree));
  assert.equal(restored.headers.Event, 'Original');
  assert.equal(store.getItem(workspaceStorageKey('analysis')), original);
  assert.equal(hasPreviousAnalysis(store), false);
});

test('failed backup or restore does not delete the current or previous save', () => {
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  saver.schedule(analysisData());
  saver.flush();
  const original = store.getItem(workspaceStorageKey('analysis'));
  const denied = {
    ...store,
    setItem() {
      throw Error('quota');
    },
  };
  assert.equal(preservePreviousAnalysis(denied), false);
  assert.equal(store.getItem(workspaceStorageKey('analysis')), original);
  assert.equal(preservePreviousAnalysis(store), true);
  assert.equal(restorePreviousAnalysis(denied), null);
  assert.equal(hasPreviousAnalysis(store), true);
});

test('play practice boundary (Play from here) round-trips and is validated against the move count', () => {
  const data = playData({ moves: ['e2e4', 'e7e5', 'g1f3'], practiceBoundary: 2 });
  const legacy = decodePlaySnapshot(envelope('play', data), engineIds);
  assert.equal(legacy.practiceBoundary, 2);
  assert.deepEqual(legacy.practicePosition, { startFen: START_FEN, moves: ['e2e4', 'e7e5'] });
  assert.equal(decodePlaySnapshot(envelope('play', playData()), engineIds).practiceBoundary, 0, 'absent means no boundary');
  for (const practiceBoundary of [-1, 4, 1.5, '2']) {
    assert.equal(decodePlaySnapshot(envelope('play', { ...data, practiceBoundary }), engineIds), null);
  }
});

test('Play from here retains its restart position with imported history or a zero-ply custom root', () => {
  for (const practicePosition of [
    { startFen: START_FEN, moves: ['e2e4', 'e7e5'] },
    { startFen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2', moves: [] },
  ]) {
    for (const extraMoves of [[], ['g1f3']]) {
      const store = storage();
      const saver = createWorkspaceAutosave('play', store);
      saver.schedule(
        playData({
          startFen: practicePosition.startFen,
          moves: [...practicePosition.moves, ...extraMoves],
          started: extraMoves.length > 0,
          practiceBoundary: practicePosition.moves.length,
          practicePosition,
        }),
      );
      saver.flush();
      const restored = readWorkspace('play', (raw) => decodePlaySnapshot(raw, engineIds), store);
      assert.deepEqual(restored.practicePosition, practicePosition);
      assert.equal(
        boardToFen(replayPlayPosition(restored.practicePosition).board),
        boardToFen(restored.positions[restored.practiceBoundary]),
        'side changes and restarts replay the imported setup, not the current or normal start',
      );
    }
  }
  assert.equal(decodePlaySnapshot(envelope('play', playData()), engineIds).practicePosition, null, 'ordinary legacy games remain ordinary');
  for (const practicePosition of [
    { startFen: START_FEN, moves: ['d2d4'] },
    { startFen: START_FEN, moves: ['e2e4', 'e7e5'] },
    { startFen: 'bad', moves: [] },
  ])
    assert.equal(decodePlaySnapshot(envelope('play', playData({ practiceBoundary: 1, practicePosition })), engineIds), null);
});

test('Copy PGN drops an imported result once the mainline no longer owns it', async () => {
  const { pgnResultForExport } = await import('../src/lc0/workspaceAutosave.ts');
  const headers = { White: 'A', Result: '1-0' };
  assert.deepEqual(pgnResultForExport(headers, '1-0', true), { headers, result: '1-0' });
  const edited = pgnResultForExport(headers, '1-0', false);
  assert.equal(edited.result, '*');
  assert.equal(edited.headers.Result, '*');
  assert.equal(headers.Result, '1-0', 'workspace headers are not mutated');
  const parsed = parsePgnGame('[Result "1-0"]\n\n1. e4 1-0');
  parsed.tree.toStart();
  const pgn = gameTreeToPgn(parsed.tree, edited.headers, edited.result);
  assert.match(pgn, /\[Result "\*"\]/);
  assert.match(pgn, /\*$/);
  assert.equal(pgnResultForExport({ White: 'A' }, '1-0', false).headers.Result, undefined);
});

test('quota, serialization and node-limit failures pause once per reason without removing the last save', () => {
  const store = storage();
  const seed = createWorkspaceAutosave('analysis', store);
  seed.schedule(analysisData());
  seed.flush();
  const original = store.getItem(workspaceStorageKey('analysis'));
  const notices = [];
  const saver = createWorkspaceAutosave(
    'analysis',
    {
      ...store,
      setItem() {
        throw Error('quota');
      },
    },
    400,
    (reason) => notices.push(reason),
  );
  saver.schedule(analysisData());
  saver.flush();
  saver.schedule({ ...analysisData(), headers: { Event: 'another' } });
  saver.flush();
  const circular = {};
  circular.tree = circular;
  saver.schedule(circular);
  saver.pause('too-large');
  assert.deepEqual(notices, ['storage', 'too-large']);
  assert.equal(store.getItem(workspaceStorageKey('analysis')), original);
  const tooLargeNotices = [];
  const oversized = createWorkspaceAutosave('analysis', store, 400, (reason) => tooLargeNotices.push(reason));
  oversized.schedule({ ...analysisData(), headers: { Event: 'x'.repeat(WORKSPACE_MAX_CHARS) } });
  oversized.flush();
  assert.deepEqual(tooLargeNotices, ['too-large']);
  const tree = new GameTree();
  for (let i = 0; i < 5001; i++) tree.addUci(['g1f3', 'g8f6', 'f3g1', 'f6g8'][i % 4]);
  assert.throws(() => serializeTree(tree), /Tree too large/);
  oversized.pause('too-large');
  assert.equal(store.getItem(workspaceStorageKey('analysis')), original);
});
