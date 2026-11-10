#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseScriptArgs } from './lib/cli.mjs';

const USAGE = `Usage: node scripts/install_git_hooks.mjs [options]

Opt-in: makes pushes run the tracked .githooks/pre-push (Biome, then Git LFS).
Installs a small pre-push shim into the shared hooks directory
(git rev-parse --git-common-dir)/hooks, replacing the stock Git LFS hook (kept
as pre-push.lfs-backup). Worktrees whose checkout has no .githooks/pre-push keep
running only Git LFS. core.hooksPath is never set: every worktree shares this
config, and pointing it at .githooks would disable LFS where that is missing.
Idempotent; refuses to replace a customized pre-push hook.

Options:
  --hooks-dir DIR   Hooks directory (default: the common git dir's hooks/)
  --check           Report whether the shim is installed; exit 1 if not
  -h, --help        Show this help
`;

export const SHIM_MARKER = '# 0x88 pre-push shim, installed by `npm run hooks:install`.';
const LFS_CALL = 'git lfs pre-push "$@"';

export const PRE_PUSH_SHIM = String.raw`#!/bin/sh
${SHIM_MARKER}
# Runs the checked-out worktree's tracked .githooks/pre-push (Biome, then Git
# LFS); a worktree without one runs only the Git LFS hook, as before.
hook="$(git rev-parse --show-toplevel)/.githooks/pre-push"
if [ -f "$hook" ]; then exec sh "$hook" "$@"; fi
command -v git-lfs >/dev/null 2>&1 || { printf >&2 "\n%s\n\n" "This repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting the 'pre-push' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks')."; exit 2; }
exec ${LFS_CALL}
`;

// Complete stock pre-push bodies: hookBaseContent and the pre-push upgradeables in
// https://github.com/git-lfs/git-lfs/blob/v3.7.1/lfs/hook.go
// The first was also verified with `git lfs install --local` (3.7.1) in a temp repo.
// Keep variants here as whole bodies; a guard prefix can hide custom commands.
const STOCK_LFS_HOOKS = [
  String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { printf >&2 "\n%s\n\n" "This repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting the 'pre-push' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks')."; exit 2; }
git lfs pre-push "$@"`,
  String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting the 'pre-push' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks').\n"; exit 2; }
git lfs pre-push "$@"`,
  String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting '.git/hooks/pre-push'.\n"; exit 2; }
git lfs pre-push "$@"`,
  String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting .git/hooks/pre-push.\n"; exit 2; }
git lfs pre-push "$@"`,
  String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository has been set up with Git LFS but Git LFS is not installed.\n"; exit 0; }
git lfs pre-push "$@"`,
  String.raw`#!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository has been set up with Git LFS but Git LFS is not installed.\n"; exit 2; }
git lfs pre-push "$@"`,
  '#!/bin/sh\ngit lfs push --stdin $*',
  '#!/bin/sh\ngit lfs push --stdin "$@"',
  '#!/bin/sh\ngit lfs pre-push "$@"',
];

/** Compare the entire hook, normalizing only CRLF and surrounding whitespace. */
export function isStockLfsHook(text) {
  return STOCK_LFS_HOOKS.includes(text.replaceAll('\r\n', '\n').trim());
}

export function prePushHookState(hooksDir) {
  const path = join(hooksDir, 'pre-push');
  if (!existsSync(path)) return 'missing';
  const text = readFileSync(path, 'utf8');
  if (text === PRE_PUSH_SHIM) return 'installed';
  if (text.includes(SHIM_MARKER)) return 'outdated';
  return isStockLfsHook(text) ? 'stock-lfs' : 'custom';
}

export function installPrePushHook(hooksDir) {
  const path = join(hooksDir, 'pre-push');
  const state = prePushHookState(hooksDir);
  if (state === 'installed') return { path, action: 'unchanged' };
  if (state === 'custom') {
    throw new Error(
      `${path} is a customized hook; not replacing it. Add this line before its Git LFS call to run the tracked hook's checks:\n  sh "$(git rev-parse --show-toplevel)/.githooks/pre-push" "$@" || exit 1\n(the tracked hook then also runs git lfs pre-push; drop the duplicate call).`,
    );
  }
  mkdirSync(hooksDir, { recursive: true });
  const backup = `${path}.lfs-backup`;
  if (state === 'stock-lfs' && !existsSync(backup)) copyFileSync(path, backup);
  writeFileSync(path, PRE_PUSH_SHIM);
  chmodSync(path, 0o755);
  return { path, action: state === 'outdated' ? 'updated' : 'installed', ...(state === 'stock-lfs' ? { backup } : {}) };
}

function defaultHooksDir() {
  const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
  let hooksPath = '';
  try {
    hooksPath = git('config', '--get', 'core.hooksPath');
  } catch {
    // Unset: git uses <common git dir>/hooks.
  }
  if (hooksPath) throw new Error(`core.hooksPath is set to ${hooksPath}; this installer only manages the default hooks directory.`);
  return join(git('rev-parse', '--path-format=absolute', '--git-common-dir'), 'hooks');
}

function main() {
  const args = parseScriptArgs(process.argv.slice(2), {
    options: { 'hooks-dir': { type: 'string' }, check: { type: 'boolean', default: false } },
    usage: USAGE,
  });
  const hooksDir = args.hooksDir ? resolve(args.hooksDir) : defaultHooksDir();
  if (args.check) {
    const state = prePushHookState(hooksDir);
    console.log(JSON.stringify({ hooksDir, prePush: state }));
    if (state !== 'installed') process.exitCode = 1;
    return;
  }
  console.log(JSON.stringify(installPrePushHook(hooksDir)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
