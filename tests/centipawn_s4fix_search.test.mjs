import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { createCentipawnEvaluatorCache } from '../src/lc0/centipawnEvaluatorCache.ts';
import { S4FIX_V_EMA_MODEL_ID, squareformerModelSettings } from '../src/nn/runtimeRegistry.ts';
import { centipawnSearchOptions } from '../src/search/centipawn.ts';
import { montyLitePuctPolicy } from '../src/search/puct.ts';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/centipawn_s4fix_search_fixture_n256.json', import.meta.url)));

test('all Centipawn product PUCT paths use the native model constants', () => {
  const expected = fixture.effective_options.search_options;
  for (const [site, native] of [
    ['cpuct', 'cpuct'], ['cpuctBase', 'cpuct_base'], ['cpuctFactor', 'cpuct_factor'],
    ['fpu', 'fpu'], ['fpuReduction', 'fpu_reduction'], ['avWeight', 'av_weight'],
  ]) assert.ok(Math.abs(centipawnSearchOptions[site] - expected[native]) < 1e-6, site);
  assert.equal(centipawnSearchOptions.cpuctSchedule, expected.cpuct_schedule);
  assert.equal(centipawnSearchOptions.fpuStrategy, expected.fpu_strategy);
  assert.equal(centipawnSearchOptions.searchPolicy, montyLitePuctPolicy);
  assert.equal(squareformerModelSettings(S4FIX_V_EMA_MODEL_ID).policyTemperature, fixture.effective_options.policy_temperature);
  assert.equal(fixture.effective_options.preset, 'monty_lc0both');
  for (const [surface, paths] of [['play', 1], ['analysis', 1], ['arena', 2]]) {
    const source = readFileSync(new URL(`../src/lc0/${surface}Browser.ts`, import.meta.url), 'utf8');
    assert.equal((source.match(/\.\.\.centipawnSearchOptions/g) ?? []).length, paths, surface);
    assert.match(source, /\/models\/s4fix_v_ema.onnx/);
  }
});

test('s4fix browser evaluation cache retains older repetition context', async () => {
  const position = fixture.positions.find((position) => position.label === 'game 30w ply 137');
  const history = JSON.parse(readFileSync(new URL('./fixtures/centipawn_s4fix_parity_fixture.json', import.meta.url))).positions.find((p) => p.label === position.label).history_fens;
  const seen = [];
  const cache = createCentipawnEvaluatorCache(async () => ({
    modelId: S4FIX_V_EMA_MODEL_ID,
    evaluator: { evaluate: async (_board, context) => { seen.push(context.historyFens.length); return { policy: new Map(), wdl: [0, 1, 0] }; } },
  }));
  const evaluator = await cache.get('s4fix', { spec: {}, options: {}, maxEntries: 4, label: 'test' }, () => true);
  await evaluator.evaluate(parseFen(position.fen), { historyFens: history });
  await evaluator.evaluate(parseFen(position.fen), { historyFens: history.slice(0, 16) });
  assert.deepEqual(seen, [137, 16]);
  assert.equal(evaluator.metrics().hits, 0, 'histories with different repetition features cannot alias');
  cache.clear();
});
