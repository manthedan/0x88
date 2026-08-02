import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const MODEL_URL = '/models/bt4_soap_rem_c19000_final.onnx';
const META_URL = '/models/bt4_soap_rem_c19000_final.meta.json';
const TVMJS_MANIFEST_PATH = 'public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json';
const TVMJS_MANIFEST_URL = TVMJS_MANIFEST_PATH.replace(/^public/, '');

async function sha256(path) {
  return createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
}

test('Centipawn production manifest matches the tracked model and metadata', async () => {
  const manifest = JSON.parse(await readFile('public/models/centipawn.manifest.json', 'utf8'));
  for (const entry of manifest.models) {
    const path = `public/models/${entry.file}`;
    const bytes = await readFile(path);
    assert.equal(bytes.byteLength, entry.bytes, `${entry.file} byte length`);
    assert.equal(await sha256(path), entry.sha256, `${entry.file} sha256`);
  }
  assert.deepEqual(
    manifest.models.map((entry) => entry.url),
    [MODEL_URL, META_URL],
  );
});

test('stable artifact release maps the Centipawn logical URLs to immutable blobs', async () => {
  const channel = JSON.parse(await readFile('public/channels/stable.json', 'utf8'));
  const release = JSON.parse(await readFile(`public${channel.releaseManifestUrl}`, 'utf8'));
  const runtimeManifest = JSON.parse(await readFile(TVMJS_MANIFEST_PATH, 'utf8'));
  assert.equal(release.releaseId, channel.releaseId);
  const runtimeBase = TVMJS_MANIFEST_URL.replace(/\/manifest\.json$/, '');
  const logicalUrls = [MODEL_URL, META_URL, TVMJS_MANIFEST_URL, ...runtimeManifest.files.map((entry) => `${runtimeBase}/${entry.path}`)];
  for (const logicalUrl of logicalUrls) {
    const artifact = release.artifacts.find((entry) => entry.logicalUrl === logicalUrl);
    assert.ok(artifact, `missing ${logicalUrl}`);
    assert.match(artifact.artifactUrl, /^https:\/\/assets\.0x88\.app\/artifacts\/sha256\/[a-f0-9]{64}\//);
  }
});

test('promoted Centipawn TVMJS manifest matches every staged runtime artifact', async () => {
  const manifest = JSON.parse(await readFile(TVMJS_MANIFEST_PATH, 'utf8'));
  assert.equal(manifest.modelFamily, 'bt4-soap-rem-c19000-final');
  assert.equal(manifest.version, 'v2-shape-k16');
  assert.deepEqual(manifest.requiredFeatures, ['webgpu']);
  const base = TVMJS_MANIFEST_PATH.replace(/\/manifest\.json$/, '');
  for (const entry of manifest.files) {
    const path = `${base}/${entry.path}`;
    const bytes = await readFile(path);
    assert.equal(bytes.byteLength, entry.bytes, `${entry.path} byte length`);
    assert.equal(await sha256(path), entry.sha256, `${entry.path} sha256`);
    if (entry.path.endsWith('.probe.json')) {
      const text = bytes.toString('utf8');
      assert.doesNotMatch(text, /\/Users\/|\/opt\/homebrew|PYTHONPATH|DYLD_LIBRARY_PATH|TVM_LIBRARY_PATH/);
    }
  }
});
