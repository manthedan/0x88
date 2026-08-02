import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { brotliDecompressSync } from 'node:zlib';

const SCRIPT = new URL('../scripts/publish_content_addressed_release.mjs', import.meta.url);
const ABC_SHA256 = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';

test('release publisher creates hash-keyed blobs, immutable release, and mutable channel', () => {
  const temp = mkdtempSync(join(tmpdir(), 'lc0-release-'));
  try {
    const asset = join(temp, 'model.onnx');
    const root = join(temp, 'public');
    writeFileSync(asset, 'abc');
    const duplicate = join(temp, 'same-bytes-different-name.data');
    writeFileSync(duplicate, 'abc');
    const publishArgs = [
      SCRIPT.pathname,
      '--root',
      root,
      '--release-id',
      'test.1',
      '--channel',
      'stable',
      '--asset',
      `small=${asset}`,
      '--asset',
      `duplicate=${duplicate}`,
    ];
    execFileSync(process.execPath, publishArgs);

    const release = JSON.parse(readFileSync(join(root, 'releases', 'test.1.json'), 'utf8'));
    const channel = JSON.parse(readFileSync(join(root, 'channels', 'stable.json'), 'utf8'));
    assert.equal(release.schema, 'lc0-webgpu.artifact-release.v2');
    assert.equal(release.artifacts[0].sha256, ABC_SHA256);
    assert.equal(release.artifacts[0].url, `/artifacts/sha256/${ABC_SHA256}/identity`);
    assert.equal(release.artifacts[1].url, release.artifacts[0].url, 'equal decoded bytes share one identity object regardless of filename');
    assert.equal(readFileSync(join(root, release.artifacts[0].url), 'utf8'), 'abc');
    const br = release.artifacts[0].representations.find((entry) => entry.encoding === 'br');
    assert.ok(br);
    assert.equal(release.artifacts[1].representations.find((entry) => entry.encoding === 'br').url, br.url, 'deterministic Brotli bytes are also shared');
    assert.equal(brotliDecompressSync(readFileSync(join(root, br.url))).toString(), 'abc');
    assert.equal(channel.releaseUrl, '/releases/test.1.json');

    // Idempotent publication of identical bytes/manifests is allowed.
    execFileSync(process.execPath, publishArgs);

    // A release ID is write-once: changing its manifest must fail.
    const other = join(temp, 'other.onnx');
    writeFileSync(other, 'abcd');
    const conflict = spawnSync(
      process.execPath,
      [SCRIPT.pathname, '--root', root, '--release-id', 'test.1', '--channel', 'stable', '--asset', `small=${other}`],
      { encoding: 'utf8' },
    );
    assert.notEqual(conflict.status, 0);
    assert.match(conflict.stderr, /Refusing to overwrite immutable release manifest/);
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});
