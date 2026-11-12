import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { boardToFen } from '../src/chess/board.ts';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { createLc0WasmInputEncoder } from '../src/lc0/wasmInputEncoder.ts';

const fenFixtures = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'));
const historyFixtures = JSON.parse(readFileSync('fixtures/lc0/history.json', 'utf8'));
const repetitionFixtures = JSON.parse(readFileSync('fixtures/lc0/repetition_history.json', 'utf8'));
const fenHistoryFixtures = JSON.parse(readFileSync('fixtures/lc0/fen_history.json', 'utf8'));
const wasmBytes = readFileSync('public/lc0/lc0_input_encoder.wasm');

function maxAbsDiff(a, b) {
  assert.equal(a.length, b.length);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

function assertEncodedEqual(actual, expected, label) {
  assert.equal(maxAbsDiff(actual.planes, expected.planes), 0, label);
  assert.deepEqual(actual.masks, expected.masks, `${label} masks`);
  assert.deepEqual(actual.values, expected.values, `${label} values`);
}

test('SIMD WASM LC0 input encoder matches JS FEN-only 112-plane encoding', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  for (const fixture of fenFixtures) {
    const expected = encodeLc0Classical112(fixture.fen);
    const actual = wasm.encodeFen(fixture.fen);
    assertEncodedEqual(actual, expected, fixture.id);
  }
});

test('SIMD WASM LC0 input encoder honors no-fill FEN-only encoding', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  const fen = fenFixtures.find((fixture) => fixture.id === 'black-to-move-after-e4').fen;
  const expected = encodeLc0Classical112(fen, { historyFill: 'no' }).planes;
  const actual = wasm.encodeFen(fen, { historyFill: false }).planes;
  assert.equal(maxAbsDiff(actual, expected), 0);
});

test('SIMD WASM LC0 input encoder preserves JS partial-FEN defaults', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  const fen = '  8/8/8/8/8/8/8/8  ';
  const expected = encodeLc0Classical112(fen).planes;
  const actual = wasm.encodeFen(fen).planes;
  assert.equal(maxAbsDiff(actual, expected), 0);
});

test('SIMD WASM LC0 input encoder preserves large safe FEN counters', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  for (const fen of ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 65537', '8/8/8/8/8/8/8/8 w - - 4294967297 1']) {
    const expected = encodeLc0Classical112(fen);
    const actual = wasm.encodeFen(fen);
    assertEncodedEqual(actual, expected, fen);
  }
});

test('SIMD WASM LC0 input encoder rejects malformed FEN fields like JS', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  for (const fen of [
    '8/8/8/8/8/8/8/8 w - - x 1',
    '8/8/8/8/8/8/8/8 w - - 0 0',
    '8/8/8/8/8/8/8/8 w KKKQ - 0 1',
    '8/8/8/8/8/8/8/8 w - a4 0 1',
    '8/8/8/8/8/8/8/8 w - - 0 1 extra',
    '8/8/8/8/8/8/8/8 w - - 9007199254740992 1',
    '8/8/8/8/8/8/8/8 w - - 0 999999999999999999999999999999',
  ]) {
    assert.throws(() => encodeLc0Classical112(fen), Error, `JS should reject ${fen}`);
    assert.throws(() => wasm.encodeFen(fen), Error, `WASM should reject ${fen}`);
  }
});

test('SIMD WASM LC0 input encoder matches JS explicit FEN history encoding', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  for (const fixture of historyFixtures) {
    const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
    const expected = encodeLc0Classical112({ positions });
    const actual = wasm.encodeFenHistory(positions.map(boardToFen));
    assertEncodedEqual(actual, expected, fixture.id);
  }
});

test('SIMD WASM LC0 input encoder fills explicit single-FEN history like a bare FEN', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  const fen = fenFixtures.find((fixture) => fixture.id === 'black-to-move-after-e4').fen;
  for (const historyFill of ['fen_only', 'no']) {
    const expected = encodeLc0Classical112({ positions: [fen] }, { historyFill });
    assertEncodedEqual(wasm.encodeFenHistory([fen], { historyFill: historyFill !== 'no' }), expected, `explicit ${historyFill}`);
    assertEncodedEqual(wasm.encodeFen(fen, { historyFill: historyFill !== 'no' }), expected, `bare ${historyFill}`);
  }
});

test('SIMD WASM LC0 input encoder matches JS history fill below FEN roots', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  for (const fixture of fenHistoryFixtures) {
    const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
    for (const historyFill of ['fen_only', 'no']) {
      const expected = encodeLc0Classical112({ positions }, { historyFill });
      const actual = wasm.encodeFenHistory(positions.map(boardToFen), { historyFill: historyFill !== 'no' });
      assertEncodedEqual(actual, expected, `${fixture.id} ${historyFill}`);
    }
  }
});

test('SIMD WASM LC0 input encoder matches JS repetition-plane history encoding', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  for (const fixture of repetitionFixtures) {
    const positions = buildBoardHistoryFromMoves(fixture.moves, fixture.startFen);
    const expected = encodeLc0Classical112({ positions });
    const actual = wasm.encodeFenHistory(positions.map(boardToFen));
    assertEncodedEqual(actual, expected, fixture.id);
  }
});

test('SIMD WASM LC0 input encoder checks repetition against explicit history older than 64 plies', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  const cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  const positions = buildBoardHistoryFromMoves(Array.from({ length: 17 }, () => cycle).flat());
  const expected = encodeLc0Classical112({ positions });
  const actual = wasm.encodeFenHistory(positions.map(boardToFen));
  assertEncodedEqual(actual, expected, 'long-repetition-history');
});

test('SIMD WASM LC0 input encoder recognizes the start board like JS whatever the castling letter order', async () => {
  // LC0 compares castling rights as a set (a bitmask), so `QKqk` is still the
  // start board and gets no history fill; startpos pieces with no castling
  // rights or black to move are other boards and are filled from the root.
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  const filledSlots = (encoded) => {
    let count = 0;
    for (let slot = 0; slot < 8; slot++) if (encoded.masks.slice(slot * 13, slot * 13 + 12).some((mask) => mask !== 0n)) count += 1;
    return count;
  };
  for (const [fen, moves, slots] of [
    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w QKqk - 0 1', [], 1],
    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w kqQK - 4 3', ['g1f3'], 2],
    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1', [], 8],
    ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1', [], 8],
  ]) {
    const positions = buildBoardHistoryFromMoves(moves, fen);
    const expected = encodeLc0Classical112({ positions });
    assert.equal(filledSlots(expected), slots, fen);
    assertEncodedEqual(wasm.encodeFenHistory(positions.map(boardToFen)), expected, fen);
    if (!moves.length) assertEncodedEqual(wasm.encodeFen(fen), encodeLc0Classical112(fen), `bare ${fen}`);
  }
});
