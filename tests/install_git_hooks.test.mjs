import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { installPrePushHook, PRE_PUSH_SHIM, prePushHookState } from '../scripts/install_git_hooks.mjs';

const STOCK_LFS_HOOK = `#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { printf >&2 "\\n%s\\n\\n" "This repository is configured for Git LFS but 'git-lfs' was not found on your path."; exit 2; }
git lfs pre-push "$@"
`;

function tempDir(t) {
  const dir = mkdtempSync(join(tmpdir(), 'hooks-install-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('replaces the stock Git LFS hook once, keeping a backup, and is idempotent', (t) => {
  const hooks = tempDir(t);
  writeFileSync(join(hooks, 'pre-push'), STOCK_LFS_HOOK);
  assert.equal(prePushHookState(hooks), 'stock-lfs');
  const first = installPrePushHook(hooks);
  assert.equal(first.action, 'installed');
  assert.equal(readFileSync(join(hooks, 'pre-push'), 'utf8'), PRE_PUSH_SHIM);
  assert.equal(readFileSync(join(hooks, 'pre-push.lfs-backup'), 'utf8'), STOCK_LFS_HOOK);
  assert.match(PRE_PUSH_SHIM, /^exec git lfs pre-push "\$@"$/m);
  assert.equal(installPrePushHook(hooks).action, 'unchanged');
  assert.equal(prePushHookState(hooks), 'installed');
});

test('installs into an empty hooks directory and refuses to replace a customized hook', (t) => {
  const hooks = tempDir(t);
  assert.equal(installPrePushHook(join(hooks, 'fresh')).action, 'installed');
  const custom = '#!/bin/sh\n./my-check.sh\ngit lfs pre-push "$@"\n';
  writeFileSync(join(hooks, 'pre-push'), custom);
  assert.equal(prePushHookState(hooks), 'custom');
  assert.throws(() => installPrePushHook(hooks), /customized hook/);
  assert.equal(readFileSync(join(hooks, 'pre-push'), 'utf8'), custom);
});

test('the shim runs the tracked hook with the pushed refs on stdin, and its failure blocks the push', (t) => {
  const dir = tempDir(t);
  const git = (cwd, ...args) => execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@example.invalid', ...args], { cwd, encoding: 'utf8' });
  const remote = join(dir, 'remote.git');
  const repo = join(dir, 'repo');
  git(dir, 'init', '-q', '--bare', remote);
  git(dir, 'init', '-q', repo);
  git(repo, 'config', 'core.hooksPath', join(repo, '.git', 'hooks'));
  git(repo, 'remote', 'add', 'origin', remote);
  mkdirSync(join(repo, '.githooks'));
  const seen = join(dir, 'stdin.txt');
  writeFileSync(join(repo, '.githooks', 'pre-push'), `#!/bin/sh\ncat > "${seen}"\nexit 1\n`);
  chmodSync(join(repo, '.githooks', 'pre-push'), 0o755);
  git(repo, 'add', '.');
  git(repo, 'commit', '-q', '-m', 'init');
  installPrePushHook(join(repo, '.git', 'hooks'));

  const blocked = spawnSync('git', ['push', '-q', 'origin', 'HEAD:refs/heads/main'], { cwd: repo, encoding: 'utf8' });
  assert.notEqual(blocked.status, 0, 'a failing tracked hook must block the push');
  assert.match(readFileSync(seen, 'utf8'), /^\S+ [0-9a-f]{40} refs\/heads\/main [0-9a-f]{40}$/m);

  writeFileSync(join(repo, '.githooks', 'pre-push'), `#!/bin/sh\ncat > "${seen}"\nexit 0\n`);
  rmSync(seen);
  git(repo, 'push', '-q', 'origin', 'HEAD:refs/heads/main');
  assert.ok(existsSync(seen));
});
