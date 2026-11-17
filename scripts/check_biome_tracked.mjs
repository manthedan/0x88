#!/usr/bin/env node
// Biome over committed files only, so untracked local scratch files never block a release.
// Run via `npm run check:biome:tracked` (npm puts node_modules/.bin on PATH).
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';

const git = spawnSync('git', ['ls-files', '-z', '--cached'], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
if (git.error || git.status !== 0) {
  console.error(`git ls-files failed: ${git.error?.message || git.stderr.trim() || `exit ${git.status}`}`);
  process.exit(2);
}
// Skip index entries absent on disk (sparse checkouts); Biome would report them as filesystem errors.
const files = git.stdout.split('\0').filter((path) => path && existsSync(path));
if (!files.length) {
  console.error('git ls-files returned no tracked files on disk; refusing to report a clean lint.');
  process.exit(2);
}
const biome = spawnSync('biome', ['check', '--no-errors-on-unmatched', '--files-ignore-unknown=true', '--', ...files], { stdio: 'inherit' });
if (biome.error) {
  console.error(`biome failed to start: ${biome.error.message}`);
  process.exit(2);
}
process.exit(biome.status ?? 1);
