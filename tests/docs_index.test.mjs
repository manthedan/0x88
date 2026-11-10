import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// docs/README.md is the index of record: every tracked Markdown file under docs/
// must be linked from it, and every relative link in it must resolve.

const root = fileURLToPath(new URL('..', import.meta.url));
const index = readFileSync(join(root, 'docs/README.md'), 'utf8');
const links = [...index.matchAll(/\]\(([^)\s#]+)(?:#[^)]*)?\)/g)].map((match) => match[1]).filter((href) => !/^[a-z]+:/i.test(href));

// Tracked docs deliberately left out of the index. Keep this empty unless a file
// genuinely should not be discoverable from docs/README.md.
const NOT_INDEXED = new Set([]);

test('every tracked docs/**/*.md is linked from docs/README.md', () => {
  const tracked = execFileSync('git', ['ls-files', '-z', '--', 'docs'], { cwd: root, encoding: 'utf8' })
    .split('\0')
    .filter((path) => path.endsWith('.md') && path !== 'docs/README.md')
    .map((path) => path.slice('docs/'.length));
  assert.ok(tracked.length > 50, `only ${tracked.length} tracked docs found`);
  const linked = new Set(links);
  assert.deepEqual(
    tracked.filter((path) => !linked.has(path) && !NOT_INDEXED.has(path)),
    [],
    'docs not linked from docs/README.md',
  );
  for (const path of NOT_INDEXED) assert.ok(tracked.includes(path), `stale NOT_INDEXED entry ${path}`);
});

test('every relative link in docs/README.md resolves', () => {
  assert.deepEqual(
    links.filter((href) => !existsSync(join(root, 'docs', href))),
    [],
  );
});
