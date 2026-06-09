import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { boardToFen } from '../src/chess/board.ts';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { createLc0WasmInputEncoder } from '../src/lc0/wasmInputEncoder.ts';

const fenFixtures = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'));
const historyFixtures = JSON.parse(readFileSync('fixtures/lc0/history.json', 'utf8'));
const repetitionFixtures = JSON.parse(readFileSync('fixtures/lc0/repetition_history.json', 'utf8'));
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
  for (const fen of [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 65537',
    '8/8/8/8/8/8/8/8 w - - 4294967297 1',
  ]) {
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

test('SIMD WASM LC0 input encoder does not synthetic-fill explicit single-FEN history', async () => {
  const wasm = await createLc0WasmInputEncoder(wasmBytes);
  const fen = fenFixtures.find((fixture) => fixture.id === 'black-to-move-after-e4').fen;
  const expected = encodeLc0Classical112({ positions: [fen] }).planes;
  const actual = wasm.encodeFenHistory([fen]).planes;
  assert.equal(maxAbsDiff(actual, expected), 0);
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
