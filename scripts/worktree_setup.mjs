#!/usr/bin/env node
// Make a fresh `git worktree` buildable and testable by borrowing the main checkout's local-only state.
//
//   node scripts/worktree_setup.mjs <worktree-path>            link, then verify
//   node scripts/worktree_setup.mjs <worktree-path> --check    verify only
//
// A new worktree lacks three untracked things:
//   1. node_modules: symlinked to the main checkout's install (relative link).
//   2. ../models: model sources live beside the repo (lc0_prepare_model_assets.mjs, the parity tests and
//      the public/models symlinks all resolve `<repo>/../models`). For a worktree in .claude/worktrees/<name>
//      that is .claude/worktrees/models, linked to the main checkout's ../models.
//   3. public/models/**: gitignored model symlinks (BT4, t3, t1, Maia3...). The main checkout's links are
//      mirrored; a link is copied verbatim when it resolves to the same file from the worktree, else it
//      points at the absolute target.
// Every step is idempotent and leaves anything that already resolves alone. Nothing is written through a
// symlinked directory that leads out of the worktree. Verification stats every model and every lc0web pack
// (metadata + shards) in public/models/{lc0,maia3}/manifest.json: exists, byte size matches; no hashing.
import { execFileSync } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, readlinkSync, realpathSync, statSync, symlinkSync, unlinkSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';

const realOrNull = (path) => {
  try {
    return realpathSync(path);
  } catch {
    return null;
  }
};

const git = (cwd, ...args) => execFileSync('git', ['-C', cwd, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();

/** Real path of `path`, or of its nearest existing ancestor, so a not-yet-created path can be placed. */
function realOfNearest(path) {
  for (let current = path; ; current = dirname(current)) {
    const real = realOrNull(current);
    if (real) return real;
    if (dirname(current) === current) return null;
  }
}

const within = (path, root) => path === root || path.startsWith(root + sep);

/** Create `link` -> `target` unless `link` already resolves to `expected`. Writes only where `link`'s parent is inside `root`. */
function ensureLink(link, target, expected, root, log) {
  const parent = realOfNearest(dirname(link));
  if (!parent || !within(parent, root)) {
    log.problems.push(`${link}: parent resolves outside ${root}; not writing there`);
    return;
  }
  let present = false;
  try {
    lstatSync(link);
    present = true;
  } catch {}
  if (present) {
    const actual = realOrNull(link);
    if (actual === expected) return;
    if (actual !== null) {
      log.problems.push(`${link} resolves to ${actual}, expected ${expected}; left alone`);
      return;
    }
    unlinkSync(link); // dangling link from an older layout
  }
  mkdirSync(dirname(link), { recursive: true });
  symlinkSync(target, link);
  log.created.push(`${link} -> ${target}`);
}

function mirrorModels(mainDir, worktreeDir, root, log) {
  for (const entry of readdirSync(mainDir, { withFileTypes: true })) {
    const source = join(mainDir, entry.name);
    const dest = join(worktreeDir, entry.name);
    if (entry.isDirectory()) {
      const destReal = realOrNull(dest);
      if (destReal && !within(destReal, root)) {
        // A directory link to the same shared folder is already right; anything else is not ours to fill.
        if (destReal !== realOrNull(source)) log.problems.push(`${dest} links outside the worktree to ${destReal}; left alone`);
        continue;
      }
      mirrorModels(source, dest, root, log);
      continue;
    }
    const expected = realOrNull(source);
    if (!expected) continue; // dangling in the main checkout too; nothing to mirror
    if (entry.isSymbolicLink()) {
      const text = readlinkSync(source);
      const verbatimTarget = resolve(dirname(dest), text);
      ensureLink(dest, realOrNull(verbatimTarget) === expected ? text : expected, expected, root, log);
    } else if (!existsSync(dest)) {
      ensureLink(dest, expected, expected, root, log); // untracked copy in the main checkout
    }
  }
}

function checkSize(path, bytes, label, log) {
  const size = realOrNull(path) ? statSync(path).size : null;
  if (size === null) log.problems.push(`${label} missing`);
  else if (typeof bytes === 'number' && size !== bytes) log.problems.push(`${label}: ${size} bytes, manifest says ${bytes}`);
  else log.verified += 1;
}

function verify(worktree, log) {
  if (!existsSync(join(worktree, 'node_modules/.bin/vite'))) log.problems.push('node_modules/.bin/vite missing');
  if (!realOrNull(join(worktree, '..', 'models'))) log.problems.push(`${resolve(worktree, '..', 'models')} does not resolve`);
  for (const dir of ['public/models/lc0', 'public/models/maia3']) {
    const manifest = JSON.parse(readFileSync(join(worktree, dir, 'manifest.json'), 'utf8'));
    for (const model of manifest.models ?? []) checkSize(join(worktree, dir, model.file), model.bytes, `${dir}/${model.file}`, log);
    for (const pack of manifest.packs ?? []) {
      const metadata = join(worktree, 'public', pack.url);
      checkSize(metadata, pack.metadataBytes, `pack ${pack.url}`, log);
      for (const shard of pack.shards ?? [])
        checkSize(join(dirname(metadata), shard.file), shard.bytes, `pack ${pack.id ?? pack.url} shard ${shard.file}`, log);
    }
  }
}

function main(argv) {
  const target = argv.find((arg) => !arg.startsWith('--'));
  if (!target) {
    console.error('usage: node scripts/worktree_setup.mjs <worktree-path> [--check]');
    return 2;
  }
  const worktree = realpathSync(resolve(target));
  // `git -C <worktree>/src` also succeeds; insist on the top level so links land where builds look.
  const top = realpathSync(git(worktree, 'rev-parse', '--show-toplevel'));
  if (top !== worktree) {
    console.error(`${worktree} is not a worktree root (git top level is ${top})`);
    return 2;
  }
  const main = realpathSync(dirname(git(worktree, 'rev-parse', '--path-format=absolute', '--git-common-dir')));
  if (main === worktree) {
    console.error(`${worktree} is the main checkout; nothing to set up`);
    return 2;
  }
  const log = { created: [], problems: [], verified: 0 };
  if (!argv.includes('--check')) {
    const nodeModules = join(main, 'node_modules');
    ensureLink(join(worktree, 'node_modules'), relative(worktree, nodeModules), realpathSync(nodeModules), worktree, log);
    const models = realOrNull(join(main, '..', 'models'));
    // The one deliberate write outside the worktree: the sibling `models` link every model path goes through.
    if (models) ensureLink(join(worktree, '..', 'models'), models, models, realpathSync(dirname(worktree)), log);
    else log.problems.push(`${resolve(main, '..', 'models')} missing in the main checkout; model links cannot resolve`);
    mirrorModels(join(main, 'public/models'), join(worktree, 'public/models'), worktree, log);
  }
  verify(worktree, log);
  for (const line of log.created) console.log(`linked ${line}`);
  for (const line of log.problems) console.error(`problem: ${line}`);
  console.log(`${worktree}: ${log.created.length} links created, ${log.verified} manifest files verified, ${log.problems.length} problems`);
  return log.problems.length > 0 ? 1 : 0;
}

process.exitCode = main(process.argv.slice(2));
