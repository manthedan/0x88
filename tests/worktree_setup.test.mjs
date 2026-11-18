import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readlinkSync,
  realpathSync,
  rmSync,
  symlinkSync,
  truncateSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const SCRIPT = fileURLToPath(new URL('../scripts/worktree_setup.mjs', import.meta.url));
const PACK_URL = '/models/lc0/p.lc0web/model.lc0web.json';

const run = (cmd, args, cwd) => {
  const result = spawnSync(cmd, args, { cwd, encoding: 'utf8' });
  assert.equal(result.status, 0, `${cmd} ${args.join(' ')}: ${result.stderr}`);
};
const setup = (...args) => spawnSync(process.execPath, [SCRIPT, ...args], { encoding: 'utf8' });
const write = (path, text) => {
  mkdirSync(join(path, '..'), { recursive: true });
  writeFileSync(path, text);
};

// The real layout in miniature: <ws>/repo is the main checkout, <ws>/models holds model sources, and the
// main checkout's gitignored public/models links reach them through ../../../../models.
function fixture(t) {
  const ws = realpathSync(mkdtempSync(join(tmpdir(), 'worktree-setup-')));
  t.after(() => rmSync(ws, { recursive: true, force: true }));
  const repo = join(ws, 'repo');
  write(join(ws, 'models/lc0/a.onnx'), 'aaa');
  write(join(ws, 'models/maia3/m.onnx'), 'mm');
  write(join(ws, 'models/lc0web/p/model.lc0web.json'), '{}');
  write(join(ws, 'models/lc0web/p/weights.000.bin'), '0123');
  const lc0Manifest = {
    models: [{ file: 'a.onnx', bytes: 3 }],
    packs: [{ id: 'p', url: PACK_URL, metadataBytes: 2, shards: [{ file: 'weights.000.bin', bytes: 4 }] }],
  };
  write(join(repo, 'public/models/lc0/manifest.json'), JSON.stringify(lc0Manifest));
  write(join(repo, 'public/models/maia3/manifest.json'), JSON.stringify({ models: [{ file: 'm.onnx', bytes: 2 }] }));
  write(join(repo, '.gitignore'), 'node_modules\n*.onnx\n*.lc0web/\n');
  write(join(repo, 'package-lock.json'), '{"lockfileVersion":3}\n');
  run('git', ['init', '-q'], repo);
  run('git', ['add', '.'], repo);
  run('git', ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'init'], repo);
  write(join(repo, 'node_modules/.bin/vite'), '');
  symlinkSync('../../../../models/lc0/a.onnx', join(repo, 'public/models/lc0/a.onnx'));
  symlinkSync('../../../../models/maia3/m.onnx', join(repo, 'public/models/maia3/m.onnx'));
  mkdirSync(join(repo, 'public/models/lc0/p.lc0web'));
  for (const file of ['model.lc0web.json', 'weights.000.bin']) {
    symlinkSync(`../../../../../models/lc0web/p/${file}`, join(repo, 'public/models/lc0/p.lc0web', file));
  }
  const worktree = join(ws, 'trees/wt');
  run('git', ['worktree', 'add', '-q', '--detach', worktree], repo);
  return { ws, repo, worktree };
}

test('links node_modules, ../models and model links, then reruns as a no-op', (t) => {
  const { worktree } = fixture(t);
  const first = setup(worktree);
  assert.equal(first.status, 0, first.stderr);
  assert.match(first.stdout, /4 manifest files verified, 0 problems/);
  assert.ok(lstatSync(join(worktree, 'node_modules')).isSymbolicLink());
  assert.ok(existsSync(join(worktree, 'public', PACK_URL)));
  const second = setup(worktree);
  assert.equal(second.status, 0, second.stderr);
  assert.match(second.stdout, /0 links created/);
});

test('--check fails on a missing pack file or a truncated shard', (t) => {
  const { ws, worktree } = fixture(t);
  assert.equal(setup(worktree).status, 0);
  truncateSync(join(ws, 'models/lc0web/p/weights.000.bin'), 1);
  const truncated = setup(worktree, '--check');
  assert.equal(truncated.status, 1);
  assert.match(truncated.stderr, /shard weights\.000\.bin: 1 bytes, manifest says 4/);
  rmSync(join(ws, 'models/lc0web/p/model.lc0web.json'));
  assert.match(setup(worktree, '--check').stderr, /pack \/models\/lc0\/p\.lc0web\/model\.lc0web\.json missing/);
});

test('refuses a subdirectory of a worktree before touching anything', (t) => {
  const { worktree } = fixture(t);
  const result = setup(join(worktree, 'public'));
  assert.equal(result.status, 2);
  assert.match(result.stderr, /not a worktree root/);
  assert.equal(existsSync(join(worktree, 'public/node_modules')), false);
  assert.equal(existsSync(join(worktree, 'node_modules')), false);
});

test('does not write through a model directory that links out of the worktree', (t) => {
  const { ws, worktree } = fixture(t);
  const outside = join(ws, 'outside');
  mkdirSync(outside);
  symlinkSync(outside, join(worktree, 'public/models/lc0/p.lc0web'));
  const result = setup(worktree);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /links outside the worktree/);
  assert.deepEqual(readdirSync(outside), []);
});

test('a worktree whose lockfile differs gets no shared install and is told to npm ci', (t) => {
  const { worktree } = fixture(t);
  writeFileSync(join(worktree, 'package-lock.json'), '{"lockfileVersion":3,"changed":true}\n');
  const result = setup(worktree);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /package-lock\.json differs.*npm ci/);
  assert.equal(existsSync(join(worktree, 'node_modules')), false);
});

test('a worktree with its own install keeps it', (t) => {
  const { worktree } = fixture(t);
  writeFileSync(join(worktree, 'package-lock.json'), '{"lockfileVersion":3,"changed":true}\n');
  write(join(worktree, 'node_modules/.bin/vite'), '');
  const result = setup(worktree);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(lstatSync(join(worktree, 'node_modules')).isSymbolicLink(), false);
});

test('an existing sibling models link whose target is unavailable is reported, not replaced', (t) => {
  const { ws, worktree } = fixture(t);
  const unmounted = join(ws, 'volume/models');
  symlinkSync(unmounted, join(ws, 'trees/models'));
  const result = setup(worktree);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /trees\/models -> .*volume\/models does not resolve/);
  assert.equal(readlinkSync(join(ws, 'trees/models')), unmounted);
});
