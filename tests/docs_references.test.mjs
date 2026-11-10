import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// Cheap guards for the two kinds of doc rot that are mechanical to detect:
// personal machine paths, and `npm run` commands that no longer exist.

const root = fileURLToPath(new URL('..', import.meta.url));
const tracked = (...pathspecs) =>
  execFileSync('git', ['ls-files', '-z', '--', ...pathspecs], { cwd: root, encoding: 'utf8' })
    .split('\0')
    .filter(Boolean);
const scripts = new Set(Object.keys(JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).scripts));

// Scripts that docs mention although package.json lacks them, and why.
const ABSENT_SCRIPTS = {
  // Removed tooling; each mention is annotated as removed with its recovery commit.
  'analysis:model-manifest': 'removed',
  'lc0:puct-wasm-feasibility-audit': 'removed',
};

test('tracked docs contain no personal machine paths', () => {
  const files = tracked('*.md', 'docs/**/*.json', 'docs/*.json');
  const hits = [];
  for (const file of files) {
    readFileSync(join(root, file), 'utf8')
      .split('\n')
      .forEach((line, index) => {
        if (/\/Users\/[A-Za-z]|\/home\/[a-z][\w-]*\//.test(line)) hits.push(`${file}:${index + 1}`);
      });
  }
  assert.deepEqual(hits, [], 'use <workspace>/<checkout> placeholders or repo-relative paths');
});

test('every `npm run <script>` in tracked Markdown exists in package.json', () => {
  const missing = new Map();
  for (const file of tracked('*.md')) {
    for (const [, name] of readFileSync(join(root, file), 'utf8').matchAll(/npm run ([a-z0-9][\w:.-]*[\w])/gi)) {
      if (scripts.has(name)) continue;
      if (!missing.has(name)) missing.set(name, []);
      missing.get(name).push(file);
    }
  }
  const unexpected = [...missing].filter(([name]) => !ABSENT_SCRIPTS[name]);
  assert.deepEqual(unexpected, [], 'docs reference npm scripts that package.json does not define');
  for (const [name, reason] of Object.entries(ABSENT_SCRIPTS)) {
    if (reason !== 'removed') continue;
    assert.ok(!scripts.has(name), `${name} exists again; drop it from ABSENT_SCRIPTS`);
    assert.ok(missing.has(name), `stale ABSENT_SCRIPTS entry ${name}`);
  }
});
