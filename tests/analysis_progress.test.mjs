import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const analysisSourceUrl = new URL('../src/lc0/analysisBrowser.ts', import.meta.url);

async function preparationSource() {
  const source = await readFile(analysisSourceUrl, 'utf8');
  return source.slice(source.indexOf('async function preloadAnalysisAsset'), source.indexOf('async function analyzeCurrent'));
}

test('analysis model assets are only preloaded once per mounted page', async () => {
  const source = await preparationSource();
  assert.match(source, /if \(!rawUrl \|\| preloadedAnalysisAssetUrls\.has\(rawUrl\)\) return;/);
  assert.match(source, /preloadedAnalysisAssetUrls\.add\(rawUrl\)/);
});

test('warm CPU engine preparation uses search progress without reopening startup progress', async () => {
  const source = await preparationSource();
  assert.match(source, /if \(prepared\) beginSearch\(\);/);
  assert.match(source, /if \(prepared\) \{\s*await start\(signal\);\s*return;\s*\}/);
  assert.match(source, /preparedCpuEngineKeys\.add\(preparationKey\)/);
});

test('warm Centipawn analysis skips its evaluator preparation indicator', async () => {
  const source = await readFile(analysisSourceUrl, 'utf8');
  assert.match(source, /const evaluatorPrepared = centipawnEvaluatorPromises\.has\(evaluatorKey\);/);
  assert.match(source, /if \(evaluatorPrepared\) beginSearch\(\);\s*else showLoadingProgressItem/);
});
