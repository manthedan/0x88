import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN, boardToFen } from '../src/chess/board.ts';
import { parsePgnGame, gameTreeToPgn } from '../src/chess/pgn.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { DEFAULT_PLAY_PREFERENCES } from '../src/lc0/playPreferences.ts';
import {
  WORKSPACE_MAX_CHARS, WORKSPACE_VERSION, createWorkspaceAutosave, decodeAnalysisSnapshot,
  decodePlaySnapshot, hasExplicitWorkspaceInput, readWorkspace, restoreTree, serializeTree, workspaceStorageKey,
} from '../src/lc0/workspaceAutosave.ts';

const engineIds = new Set(['maia3', 'sf-lite']);
const envelope = (kind, data, version = WORKSPACE_VERSION) => JSON.stringify({ kind, version, savedAt: Date.now(), data });
const analysisData = () => ({ tree: serializeTree(new GameTree()), headers: {}, result: '*' });
const playData = (changes = {}) => ({
  startFen: START_FEN, moves: ['e2e4'], humanColor: 'w', orientation: 'black',
  preferences: { ...DEFAULT_PLAY_PREFERENCES, color: 'random', maiaElo: 1900, maiaTemperature: 0.8, maiaTopP: 0.9 },
  gameOver: null, started: true, ...changes,
});
const storage = () => {
  const values = new Map();
  let writes = 0;
  return { values, get writes() { return writes; }, getItem: (key) => values.get(key) ?? null,
    setItem(key, value) { writes++; values.set(key, value); }, removeItem: (key) => values.delete(key) };
};

test('analysis round trip retains nested variations, comments, headers, result and selected non-mainline node', () => {
  const parsed = parsePgnGame('[Event "My game"]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 {Centre} e5 (1... c5 2. Nf3 (2. Nc3)) 2. Nf3 1-0');
  const { tree, tags: headers, result } = parsed;
  tree.root.comment = 'Before the game';
  tree.goTo(tree.root.children[0].children[1].children[1]);
  const raw = envelope('analysis', { tree: serializeTree(tree), headers, result });
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

test('custom root, castling, en passant and promotion round trip through legal replay', () => {
  for (const [fen, moves] of [
    ['r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', ['e1g1', 'e8c8']],
    ['4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', ['e5d6']],
    ['4k3/P7/8/8/8/8/8/6K1 w - - 0 1', ['a7a8n']],
  ]) {
    const tree = new GameTree(fen);
    moves.forEach((uci) => assert.ok(tree.addUci(uci)));
    assert.deepEqual(serializeTree(restoreTree(serializeTree(tree))), serializeTree(tree));
  }
});

test('analysis rejects truncated JSON, mismatched/old schemas, bad FEN, indices and illegal or duplicate moves', () => {
  for (const raw of ['{', 'null', '[]', envelope('play', analysisData()), envelope('analysis', analysisData(), 0), envelope('analysis', analysisData(), 2)]) {
    assert.equal(decodeAnalysisSnapshot(raw), null);
  }
  const good = analysisData();
  for (const patch of [
    { startFen: 'garbage' }, { current: -1 }, { current: 1 }, { current: 0.5 },
    { nodes: [{ parent: 1, uci: 'e2e4' }] }, { nodes: [{ parent: 0, uci: 'e2e5' }] },
    { nodes: [{ parent: 0, uci: 'e2e4' }, { parent: 0, uci: 'e2e4' }] },
    { nodes: [{ parent: 0, uci: 'e2e4', comment: {} }] },
  ]) assert.equal(decodeAnalysisSnapshot(envelope('analysis', { ...good, tree: { ...good.tree, ...patch } })), null);
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
  analysis.schedule(analysisData()); analysis.flush();
  play.schedule(playData()); play.flush();
  analysis.schedule(analysisData()); analysis.clear();
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(store.getItem(workspaceStorageKey('analysis')), null);
  assert.ok(store.getItem(workspaceStorageKey('play')));
  play.clear();
});

test('denied/full storage fails safely and invalid saved data is removed', () => {
  const denied = { getItem() { throw Error('denied'); }, setItem() { throw Error('denied'); }, removeItem() { throw Error('denied'); } };
  assert.equal(readWorkspace('analysis', decodeAnalysisSnapshot, denied), null);
  const saver = createWorkspaceAutosave('analysis', denied);
  saver.schedule(analysisData()); saver.flush(); saver.clear();
  const store = storage();
  store.setItem(workspaceStorageKey('analysis'), '{broken');
  assert.equal(readWorkspace('analysis', decodeAnalysisSnapshot, store), null);
  assert.equal(store.getItem(workspaceStorageKey('analysis')), null);
  store.setItem(workspaceStorageKey('play'), envelope('play', playData()));
  const full = createWorkspaceAutosave('play', { ...store, setItem() { throw Error('quota'); } });
  full.schedule(playData()); full.flush();
  assert.equal(store.getItem(workspaceStorageKey('play')), null, 'old game must not survive a failed replacement');
});

test('oversized work cannot leave a stale saved game behind', () => {
  const store = storage();
  const saver = createWorkspaceAutosave('analysis', store);
  saver.schedule(analysisData()); saver.flush();
  saver.schedule({ ...analysisData(), headers: { Event: 'x'.repeat(WORKSPACE_MAX_CHARS) } }); saver.flush();
  assert.equal(store.getItem(workspaceStorageKey('analysis')), null);
});

test('explicit FEN and handoff URLs take precedence without consuming the handoff', () => {
  for (const [search, hash] of [['?fen=abc', ''], ['?handoff=token', ''], ['?fromAnalysis=1', ''], ['', '#game=payload']]) {
    assert.equal(hasExplicitWorkspaceInput(search, hash), true);
  }
  assert.equal(hasExplicitWorkspaceInput('?dev&model=local&perfDial=eco'), false);
  assert.equal(hasExplicitWorkspaceInput(''), false);
});
