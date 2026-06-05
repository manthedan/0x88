import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseFen } from '../src/chess/board.ts';
import { legalPolicyPriors } from '../src/lc0/onnxEvaluator.ts';
import { LC0_POLICY_SIZE } from '../src/lc0/policyMap.ts';
import { createLc0WasmLegalPriors } from '../src/lc0/wasmLegalPriors.ts';

const fenFixtures = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'));
const historyFixtures = JSON.parse(readFileSync('fixtures/lc0/history.json', 'utf8'));
const wasmBytes = readFileSync('public/lc0/lc0_legal_priors.wasm');
const temperature = 1.359;

const logits = Float32Array.from({ length: LC0_POLICY_SIZE }, (_, i) => Math.sin(i * 0.017) * 2.5 + Math.cos(i * 0.031) * 0.75 + ((i % 13) - 6) * 0.01);

function assertLegalPriorsClose(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label} legal count`);
  for (let i = 0; i < expected.length; i++) {
    assert.equal(actual[i].uci, expected[i].uci, `${label} uci ${i}`);
    assert.equal(actual[i].index, expected[i].index, `${label} index ${i}`);
    assert.ok(Math.abs(actual[i].logit - expected[i].logit) < 1e-6, `${label} scaled logit ${i}`);
    assert.ok(Math.abs(actual[i].prior - expected[i].prior) < 1e-6, `${label} prior ${i}: ${actual[i].prior} vs ${expected[i].prior}`);
  }
}

test('SIMD WASM legal-prior path matches JS legal priors for representative FEN fixtures', async () => {
  const wasm = await createLc0WasmLegalPriors(wasmBytes);
  for (const fixture of fenFixtures) {
    const expected = legalPolicyPriors(parseFen(fixture.fen), logits, temperature);
    const actual = wasm.evaluateFen(fixture.fen, logits, { temperature }).legalPriors;
    assertLegalPriorsClose(actual, expected, fixture.id);
  }
});

test('SIMD WASM legal-prior path matches JS legal priors for representative history final FENs', async () => {
  const wasm = await createLc0WasmLegalPriors(wasmBytes);
  for (const fixture of historyFixtures) {
    const expected = legalPolicyPriors(parseFen(fixture.finalFen), logits, temperature);
    const actual = wasm.evaluateFen(fixture.finalFen, logits, { temperature }).legalPriors;
    assertLegalPriorsClose(actual, expected, fixture.id);
  }
});

test('SIMD WASM legal-prior path supports top-K candidate output', async () => {
  const wasm = await createLc0WasmLegalPriors(wasmBytes);
  const fen = fenFixtures.find((fixture) => fixture.id === 'startpos').fen;
  const expected = legalPolicyPriors(parseFen(fen), logits, temperature).slice(0, 5);
  const actual = wasm.evaluateFen(fen, logits, { temperature, topK: 5 }).legalPriors;
  assertLegalPriorsClose(actual, expected, 'startpos-top5');
});

test('SIMD WASM legal-prior path rejects stale castling rights like JS movegen', async () => {
  const wasm = await createLc0WasmLegalPriors(wasmBytes);
  for (const [fen, invalidCastles] of [
    ['4k3/8/8/8/8/8/8/4K3 w KQ - 0 1', ['e1g1', 'e1c1']],
    ['4k3/8/8/8/8/8/8/R3K3 w KQ - 0 1', ['e1g1']],
    ['4k3/8/8/8/8/8/8/4K2R w KQ - 0 1', ['e1c1']],
  ]) {
    const expected = legalPolicyPriors(parseFen(fen), logits, temperature);
    const actual = wasm.evaluateFen(fen, logits, { temperature }).legalPriors;
    assertLegalPriorsClose(actual, expected, fen);
    assert.ok(!actual.some((entry) => invalidCastles.includes(entry.uci)), `stale castling leaked for ${fen}`);
  }
});
