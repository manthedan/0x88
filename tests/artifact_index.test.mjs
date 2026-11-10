import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { lstatSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { ARTIFACT_INDEX_PATH, buildArtifactIndex, entryUrls, serializeArtifactIndex } from '../scripts/write_artifact_index.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const committed = readFileSync(join(root, ARTIFACT_INDEX_PATH), 'utf8');
const index = JSON.parse(committed);

// Index URLs that a clean checkout intentionally does not contain. Each rule must
// match a .gitignore pattern (checked below) and at least one index URL.
const UNTRACKED_ALLOWLIST = [
  // Lc0 ONNX exports and lc0web packs: staged as local links by
  // `npm run lc0:prepare-model-assets` (needs ../models) or served by the artifact host.
  /^\/models\/lc0\/[^/]+\.onnx$/,
  /^\/models\/lc0\/[^/]+\.lc0web\/model\.lc0web\.json$/,
  // Maia 3 ONNX files: staged by `npm run maia3:stage-assets` or served by the artifact host.
  /^\/models\/maia3\/[^/]+\.onnx$/,
  // Reckless WASM binaries and the extracted NNUE: built locally by
  // `npm run reckless:build-release`; only its manifest and source archives are committed.
  /^\/reckless\/[^/]+\.(wasm|nnue)$/,
  // Research-only Lc0 TVMJS runtime directory, staged locally.
  /^\/runtimes\/lc0-tvmjs-webgpu\/$/,
];

const toPath = (url) => `public${url.replace(/\/$/, '')}`;

test('committed artifact index equals the generator output (run npm run artifact-index)', () => {
  assert.equal(committed, serializeArtifactIndex(buildArtifactIndex(root)));
});

for (const pattern of ['*.wasm', 'public/', '*']) {
  test(`artifact index output is unchanged by global excludes: ${pattern}`, (t) => {
    const directory = mkdtempSync(join(tmpdir(), 'artifact-index-excludes-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const excludes = join(directory, 'excludes');
    writeFileSync(excludes, `${pattern}\n`);

    const generated = spawnSync(
      process.execPath,
      [
        '--input-type=module',
        '--eval',
        "import { buildArtifactIndex, serializeArtifactIndex } from './scripts/write_artifact_index.mjs'; process.stdout.write(serializeArtifactIndex(buildArtifactIndex()));",
      ],
      {
        cwd: root,
        encoding: 'utf8',
        env: { ...process.env, GIT_CONFIG_COUNT: '1', GIT_CONFIG_KEY_0: 'core.excludesFile', GIT_CONFIG_VALUE_0: excludes },
      },
    );
    assert.equal(generated.status, 0, generated.stderr);
    assert.equal(generated.stdout, committed);
  });
}

test('every artifact index URL exists under public/ or is an allowlisted gitignored file', () => {
  const urls = [...new Set(index.entries.flatMap(entryUrls))];
  const allowlisted = urls.filter((url) => UNTRACKED_ALLOWLIST.some((rule) => rule.test(url)));
  const missing = urls.filter((url) => !allowlisted.includes(url)).filter((url) => !lstatExists(join(root, toPath(url))));
  assert.deepEqual(missing, [], 'index URLs with no file under public/');

  const check = spawnSync('git', ['check-ignore', '--no-index', '--stdin'], { cwd: root, input: allowlisted.map(toPath).join('\n'), encoding: 'utf8' });
  const ignored = new Set(check.stdout.split('\n').filter(Boolean));
  assert.deepEqual(
    allowlisted.filter((url) => !ignored.has(toPath(url))),
    [],
    'allowlisted URLs must be gitignored',
  );
  for (const rule of UNTRACKED_ALLOWLIST)
    assert.ok(
      urls.some((url) => rule.test(url)),
      `stale allowlist rule ${rule}`,
    );
});

test('artifact index records every engine manifest with its source archive and upstream pin', () => {
  const engines = index.entries.filter((entry) => entry.sourceArchiveUrl);
  assert.deepEqual(engines.map((entry) => entry.family).sort(), ['berserk', 'plentychess', 'reckless', 'stockfish', 'stormphrax', 'viridithas']);
  for (const entry of engines) {
    const manifest = JSON.parse(readFileSync(join(root, toPath(entry.manifestUrl)), 'utf8'));
    assert.equal(entry.sourceArchiveUrl, manifest.sourceArchive.url, entry.id);
    assert.ok(entry.versionOrSource.includes(manifest.upstream.commit), entry.id);
    assert.equal(entry.license, manifest.upstream.license, entry.id);
  }
});

function lstatExists(path) {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}
