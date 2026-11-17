#!/usr/bin/env node
// Biome over committed files only, so untracked local scratch files never block a release.
// Run via `npm run check:biome:tracked` (npm puts node_modules/.bin on PATH).
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

function fail(message) {
  console.error(message);
  process.exit(2);
}
// Releases run from macOS/Linux; the full path list exceeds the Windows command-line limit.
if (process.platform === 'win32') fail('check:biome:tracked is not supported on Windows; run npm run check:biome instead.');

function gitFiles(args) {
  const git = spawnSync('git', ['ls-files', '-z', ...args], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (git.error || git.status !== 0) fail(`git ls-files failed: ${git.error?.message || git.stderr.trim() || `exit ${git.status}`}`);
  return git.stdout.split('\0').filter(Boolean);
}

// Biome discovers nested configs next to the files it checks, so an untracked one could change the result.
const strayConfigs = gitFiles(['--others', '--exclude-standard']).filter((path) => /(^|\/)biome\.jsonc?$/.test(path));
if (strayConfigs.length) fail(`Untracked Biome config(s) would affect the tracked check: ${strayConfigs.join(', ')}. Move or commit them.`);

// Skip index entries absent on disk (sparse checkouts); Biome would report them as filesystem errors.
const files = gitFiles(['--cached']).filter((path) => existsSync(path));
if (!files.length) fail('git ls-files returned no tracked files on disk; refusing to report a clean lint.');
const biome = spawnSync('biome', ['check', '--files-ignore-unknown=true', '--', ...files], { stdio: 'inherit' });
if (biome.error) fail(`biome failed to start: ${biome.error.message}`);
process.exit(biome.status ?? 1);
