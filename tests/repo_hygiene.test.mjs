import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { isAbsolute, posix } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));

function git(...args) {
  return execFileSync('git', args, { cwd: repoRoot, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

// A fresh clone must install: `npm ci` runs `svelte-kit sync`, which stats every
// file under public/, so one dangling tracked link breaks the install.
test('tracked symlinks resolve inside the repository', () => {
  const escaping = [];
  for (const line of git('ls-files', '-s', '-z').split('\0')) {
    if (!line.startsWith('120000 ')) continue;
    const [meta, path] = line.split('\t');
    const target = git('cat-file', 'blob', meta.split(' ')[1]);
    const resolved = posix.normalize(posix.join(posix.dirname(path), target));
    if (isAbsolute(target) || resolved === '..' || resolved.startsWith('../')) escaping.push(`${path} -> ${target}`);
  }
  assert.deepEqual(escaping, [], 'run `npm run lc0:prepare-model-assets` to create local model links; never commit them');
});

test('no tracked file matches an ignore rule', () => {
  assert.deepEqual(git('ls-files', '-ci', '--exclude-standard').split('\n').filter(Boolean), []);
});
