import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const manifest = JSON.parse(readFileSync(new URL('../public/models/lc0/manifest.json', import.meta.url), 'utf8'));

test('LC0 test asset staging rejects a stable release that differs from committed integrity metadata', () => {
  const pinned = manifest.models.find((model) => model.url.endsWith('.batch1.f32.onnx'));
  assert.ok(pinned);
  const changedSha256 = '0'.repeat(64);
  assert.notEqual(changedSha256, pinned.sha256);
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v1',
    releaseId: 'mutated-stable-channel',
    artifacts: [
      {
        logicalUrl: pinned.url,
        artifactUrl: `https://assets.example/artifacts/sha256/${changedSha256}/identity`,
        bytes: pinned.bytes,
        sha256: changedSha256,
      },
    ],
  };
  const releaseUrl = `data:application/json,${encodeURIComponent(JSON.stringify(release))}`;
  const channel = {
    schema: 'lc0_browser.artifact_channel_manifest.v1',
    releaseUrl,
  };
  const channelUrl = `data:application/json,${encodeURIComponent(JSON.stringify(channel))}`;
  const result = spawnSync(
    process.execPath,
    ['scripts/stage_lc0_test_assets.mjs', '--channel-url', channelUrl, '--root', join(tmpdir(), 'lc0-stage-pinned-test'), '--check'],
    { cwd: repoRoot, encoding: 'utf8' },
  );
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Stable artifact release integrity does not match the committed LC0 manifest/);
});
