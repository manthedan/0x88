import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const worker = await readFile(new URL('../src/lc0/searchWorkerCore.ts', import.meta.url), 'utf8');

test('Analysis opts into persistent model and bounded evaluation caching on both paths', () => {
  assert.match(source, /cacheModel: true,[\s\S]*?requestPersistentModelStorage: true,[\s\S]*?evalCacheEntries: 2048/);
  assert.match(source, /loadLc0ModelForOrt\(MODEL_URL, \{\s*cache: true,\s*requestPersistentStorage: true/);
  assert.match(source, /new CachedLc0Evaluator\(nextEvaluator, \{ maxEntries: 2048 \}\)/);
});

test('Analysis retains only bounded LC0 trees and resets neural state on game replacement', () => {
  assert.match(source, /reuseTree: visits <= 16_000/);
  assert.match(source, /reuseTree: row.strength <= 16_000/);
  assert.match(source, /if \(analysisGameTree !== tree\)[\s\S]*?resetTree\(\)[\s\S]*?clearCache\?\.\(\)[\s\S]*?clearEvalCache: true/);
  assert.match(worker, /message\.clearEvalCache\)[\s\S]*?clearCache\?\.\(\)/);
});
