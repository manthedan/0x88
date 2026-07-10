import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const MODEL_URL = '/models/bt4_soap_rem_c19000_final.onnx';
const META_URL = '/models/bt4_soap_rem_c19000_final.meta.json';

async function sha256(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

test('Centipawn production manifest matches the tracked model and metadata', async () => {
  const manifest = JSON.parse(await readFile('public/models/centipawn.manifest.json', 'utf8'));
  for (const entry of manifest.models) {
    const path = `public/models/${entry.file}`;
    const bytes = await readFile(path);
    assert.equal(bytes.byteLength, entry.bytes, `${entry.file} byte length`);
    assert.equal(await sha256(path), entry.sha256, `${entry.file} sha256`);
  }
  assert.deepEqual(manifest.models.map((entry) => entry.url), [MODEL_URL, META_URL]);
});

test('stable artifact release maps the Centipawn logical URLs to immutable blobs', async () => {
  const channel = JSON.parse(await readFile('public/channels/stable.json', 'utf8'));
  const release = JSON.parse(await readFile(`public${channel.releaseManifestUrl}`, 'utf8'));
  assert.equal(release.releaseId, channel.releaseId);
  for (const logicalUrl of [MODEL_URL, META_URL]) {
    const artifact = release.artifacts.find((entry) => entry.logicalUrl === logicalUrl);
    assert.ok(artifact, `missing ${logicalUrl}`);
    assert.match(artifact.artifactUrl, /^https:\/\/assets\.0x88\.app\/artifacts\/sha256\/[a-f0-9]{64}\//);
  }
});
