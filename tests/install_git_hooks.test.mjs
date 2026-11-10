import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { installPrePushHook, PRE_PUSH_SHIM, prePushHookState } from '../scripts/install_git_hooks.mjs';

// Captured from `git lfs install --local` with Git LFS 3.7.1 in a temporary repo.
const STOCK_LFS_HOOK = String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { printf >&2 "\n%s\n\n" "This repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting the 'pre-push' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks')."; exit 2; }
git lfs pre-push "$@"
`;

function runInstaller(hooks, ...args) {
  return spawnSync(process.execPath, [fileURLToPath(new URL('../scripts/install_git_hooks.mjs', import.meta.url)), '--hooks-dir', hooks, ...args], {
    encoding: 'utf8',
  });
}

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
  assert.equal(readFileSync(join(hooks, 'pre-push.lfs-backup'), 'utf8'), STOCK_LFS_HOOK);
});

test('replaces historical stock Git LFS hooks and normalized line endings, preserving their original bytes in backups', (t) => {
  const variants = [
    '#!/bin/sh\ngit lfs push --stdin $*',
    '#!/bin/sh\ngit lfs push --stdin "$@"',
    '#!/bin/sh\ngit lfs pre-push "$@"',
    ...[0, 2].map(
      (code) => String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository has been set up with Git LFS but Git LFS is not installed.\n"; exit ${code}; }
git lfs pre-push "$@"`,
    ),
    ...['.git/hooks/pre-push', "'.git/hooks/pre-push'", "the 'pre-push' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks')"].map(
      (location) => String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting ${location}.\n"; exit 2; }
git lfs pre-push "$@"`,
    ),
    STOCK_LFS_HOOK.trimEnd(),
    STOCK_LFS_HOOK.replaceAll('\n', '\r\n'),
  ];
  for (const hook of variants) {
    const hooks = tempDir(t);
    writeFileSync(join(hooks, 'pre-push'), hook);
    assert.equal(prePushHookState(hooks), 'stock-lfs', hook);
    assert.equal(installPrePushHook(hooks).action, 'installed');
    assert.equal(readFileSync(join(hooks, 'pre-push'), 'utf8'), PRE_PUSH_SHIM);
    assert.equal(readFileSync(join(hooks, 'pre-push.lfs-backup'), 'utf8'), hook);
    assert.equal(installPrePushHook(hooks).action, 'unchanged');
    assert.equal(readFileSync(join(hooks, 'pre-push.lfs-backup'), 'utf8'), hook);
  }
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

test('refuses LFS-like hooks containing additional commands without changing them or creating backups', (t) => {
  const variants = [
    '#!/bin/sh\ncommand -v git-lfs >/dev/null 2>&1 || exit 2; ./required-check.sh || exit 1\ngit lfs pre-push "$@"\n',
    STOCK_LFS_HOOK.replace('\ngit lfs', '; ./required-check.sh || exit 1\ngit lfs'),
    STOCK_LFS_HOOK.replace('; exit 2;', '; ./required-check.sh; exit 2;'),
    `${STOCK_LFS_HOOK}./required-check.sh || exit 1\n`,
    STOCK_LFS_HOOK.replace('#!/bin/sh', '#!/bin/sh -e'),
    STOCK_LFS_HOOK.replace('\ngit lfs pre-push "$@"', '\ncommand -v git-lfs >/dev/null 2>&1 || exit 2\ngit lfs pre-push "$@"'),
  ];
  for (const hook of variants) {
    const hooks = tempDir(t);
    writeFileSync(join(hooks, 'pre-push'), hook);
    assert.equal(prePushHookState(hooks), 'custom', hook);
    assert.throws(() => installPrePushHook(hooks), /customized hook/);
    const result = runInstaller(hooks);
    assert.equal(result.status, 1);
    assert.match(result.stderr, /customized hook/);
    assert.equal(readFileSync(join(hooks, 'pre-push'), 'utf8'), hook);
    assert.equal(existsSync(join(hooks, 'pre-push.lfs-backup')), false);
  }
});

test('--check reports missing, stock, custom, outdated, and installed hooks without modifying them', (t) => {
  const hooks = tempDir(t);
  for (const [state, hook] of [
    ['missing', null],
    ['stock-lfs', STOCK_LFS_HOOK],
    ['custom', '#!/bin/sh\ncommand -v git-lfs >/dev/null 2>&1 || exit 2; ./required-check.sh || exit 1\ngit lfs pre-push "$@"\n'],
    ['outdated', `${PRE_PUSH_SHIM}# old shim\n`],
    ['installed', PRE_PUSH_SHIM],
  ]) {
    if (hook !== null) writeFileSync(join(hooks, 'pre-push'), hook);
    const result = runInstaller(hooks, '--check');
    assert.equal(result.status, state === 'installed' ? 0 : 1);
    assert.equal(result.stderr, '');
    assert.deepEqual(JSON.parse(result.stdout), { hooksDir: hooks, prePush: state });
    assert.equal(existsSync(join(hooks, 'pre-push')), hook !== null);
    if (hook !== null) assert.equal(readFileSync(join(hooks, 'pre-push'), 'utf8'), hook);
    assert.equal(existsSync(join(hooks, 'pre-push.lfs-backup')), false);
  }
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
