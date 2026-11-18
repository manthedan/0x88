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
// Every step is idempotent and leaves anything that already resolves alone. Verification stats every
// model in public/models/{lc0,maia3}/manifest.json (exists, byte size matches; no hashing).
import { execFileSync } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, readlinkSync, realpathSync, statSync, symlinkSync, unlinkSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';

const realOrNull = (path) => {
  try {
    return realpathSync(path);
  } catch {
    return null;
  }
};

function mainCheckoutOf(worktree) {
  const commonDir = execFileSync('git', ['-C', worktree, 'rev-parse', '--path-format=absolute', '--git-common-dir'], { encoding: 'utf8' }).trim();
  return dirname(commonDir);
}

/** Create `link` -> `target` unless `link` already resolves to `expected` (the real path the target should reach). */
function ensureLink(link, target, expected, log) {
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

function mirrorModels(mainDir, worktreeDir, log) {
  for (const entry of readdirSync(mainDir, { withFileTypes: true })) {
    const source = join(mainDir, entry.name);
    const dest = join(worktreeDir, entry.name);
    if (entry.isDirectory()) {
      mirrorModels(source, dest, log);
      continue;
    }
    const expected = realOrNull(source);
    if (!expected) continue; // dangling in the main checkout too; nothing to mirror
    if (entry.isSymbolicLink()) {
      const text = readlinkSync(source);
      const verbatimTarget = resolve(dirname(dest), text);
      ensureLink(dest, realOrNull(verbatimTarget) === expected ? text : expected, expected, log);
    } else if (!existsSync(dest)) {
      ensureLink(dest, expected, expected, log); // untracked copy in the main checkout
    }
  }
}

function verify(worktree, log) {
  if (!existsSync(join(worktree, 'node_modules/.bin/vite'))) log.problems.push('node_modules/.bin/vite missing');
  if (!realOrNull(join(worktree, '..', 'models'))) log.problems.push(`${resolve(worktree, '..', 'models')} does not resolve`);
  for (const dir of ['public/models/lc0', 'public/models/maia3']) {
    const manifest = JSON.parse(readFileSync(join(worktree, dir, 'manifest.json'), 'utf8'));
    for (const model of manifest.models ?? []) {
      const path = join(worktree, dir, model.file);
      const size = realOrNull(path) ? statSync(path).size : null;
      if (size === null) log.problems.push(`${dir}/${model.file} missing`);
      else if (typeof model.bytes === 'number' && size !== model.bytes) log.problems.push(`${dir}/${model.file}: ${size} bytes, manifest says ${model.bytes}`);
      else log.verified += 1;
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
  const main = realpathSync(mainCheckoutOf(worktree));
  if (main === worktree) {
    console.error(`${worktree} is the main checkout; nothing to set up`);
    return 2;
  }
  const log = { created: [], problems: [], verified: 0 };
  if (!argv.includes('--check')) {
    const nodeModules = join(main, 'node_modules');
    ensureLink(join(worktree, 'node_modules'), relative(worktree, nodeModules), realpathSync(nodeModules), log);
    const models = realOrNull(join(main, '..', 'models'));
    if (models) ensureLink(join(worktree, '..', 'models'), models, models, log);
    else log.problems.push(`${resolve(main, '..', 'models')} missing in the main checkout; model links cannot resolve`);
    mirrorModels(join(main, 'public/models'), join(worktree, 'public/models'), log);
  }
  verify(worktree, log);
  for (const line of log.created) console.log(`linked ${line}`);
  for (const line of log.problems) console.error(`problem: ${line}`);
  console.log(`${worktree}: ${log.created.length} links created, ${log.verified} manifest models verified, ${log.problems.length} problems`);
  return log.problems.length > 0 ? 1 : 0;
}

process.exitCode = main(process.argv.slice(2));
