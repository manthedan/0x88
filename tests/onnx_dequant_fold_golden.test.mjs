import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { pinnedAssetMetadata, REQUIRED_ASSETS } from '../scripts/stage_lc0_test_assets.mjs';
import { countOnnxDequantizeLinear, foldOnnxDequantizeLinear } from '../src/nn/onnxDequantFold.ts';
import { skipUnlessAssets } from './lib/required_assets.mjs';

const golden = JSON.parse(readFileSync(new URL('./fixtures/onnx_dequant_fold_golden.json', import.meta.url)));
const directory = process.env.LC0_MEMORY_MODEL_DIR ?? '../models/lc0-bestnets/onnx';
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const logicalUrl = (model) => `/models/${model.file.startsWith('maia3_') ? 'maia3' : 'lc0'}/${model.file}`;

test('every shipped/fallback QDQ golden has pinned CI staging, with no moving-channel trust', async () => {
  const pinned = await pinnedAssetMetadata();
  const manifests = ['lc0', 'maia3'].flatMap((family) => JSON.parse(readFileSync(new URL(`../public/models/${family}/manifest.json`, import.meta.url))).models);
  assert.deepEqual(
    golden.models.map((m) => m.file).sort(),
    manifests
      .filter((m) => m.file.includes('.qdq'))
      .map((m) => m.file)
      .sort(),
  );
  for (const model of golden.models) {
    assert.ok(REQUIRED_ASSETS.includes(logicalUrl(model)), model.file);
    assert.deepEqual(pinned.get(logicalUrl(model)), { bytes: model.inputBytes, sha256: model.inputSha256 }, model.file);
  }
});

for (const model of golden.models) {
  const path = join(directory, model.file);
  test(`byte-identical legacy fold golden: ${model.file}`, { skip: skipUnlessAssets(path) }, () => {
    const input = readFileSync(path);
    assert.equal(input.byteLength, model.inputBytes);
    assert.equal(digest(input), model.inputSha256, 'golden only applies to the exact pinned artifact');
    const folded = foldOnnxDequantizeLinear(input);
    assert.equal(folded.bytes.byteLength, model.outputBytes);
    assert.equal(digest(folded.bytes), model.outputSha256, 'entire protobuf, not just tensor values, matches the pre-rewrite fold');
    assert.equal(folded.foldedNodes, model.foldedNodes);
    assert.equal(folded.removedInitializers, model.removedInitializers);
    assert.equal(folded.skippedNodes, model.skippedNodes);
    assert.equal(countOnnxDequantizeLinear(folded.bytes), 0);
  });
}
