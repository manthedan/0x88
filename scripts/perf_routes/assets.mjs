// Ignored local assets (models, runtimes, engine binaries; ~6 GB) are linked into a built arm instead
// of staged in its public/: Vite and SvelteKit each dereference-copy public/, which filled the disk.
// The static server follows links under the same origin, so served bytes and headers are unchanged.
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, rmSync, symlinkSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

function sourceAsset(root, source) {
  for (let dir = root; ; dir = dirname(dir)) {
    const candidate = resolve(dir, source);
    if (existsSync(candidate)) return candidate;
    if (dirname(dir) === dir) return null;
  }
}

// Map of public-relative path -> real file for every local asset the archived arm lacks. Removes
// dangling archived links at those paths (the old copy replaced them too) so the build can stat public/.
export function localAssets(root, dir) {
  const links = new Map();
  const walk = (rel) => {
    for (const entry of readdirSync(join(root, 'public', rel), { withFileTypes: true })) {
      const path = join(rel, entry.name);
      if (entry.isDirectory()) walk(path);
      // Never replace historical tracked bytes.
      else if (!existsSync(join(dir, 'public', path)) && existsSync(join(root, 'public', path))) links.set(path, realpathSync(join(root, 'public', path)));
    }
  };
  walk('');
  // Historical manifests may name models this worktree has not staged; find them by their source.
  for (const family of ['lc0', 'maia3']) {
    const manifest = JSON.parse(readFileSync(join(dir, 'public/models', family, 'manifest.json'), 'utf8'));
    for (const entry of manifest.models ?? []) {
      if (!entry.file || !entry.source) continue;
      const path = join('models', family, entry.file);
      if (links.has(path) || existsSync(join(dir, 'public', path))) continue;
      const asset = sourceAsset(root, entry.source);
      if (asset) links.set(path, realpathSync(asset));
    }
  }
  for (const path of links.keys()) rmSync(join(dir, 'public', path), { force: true });
  return links;
}

export function linkAssets(links, dist) {
  for (const [path, target] of links) {
    // Build output wins, as it did over Vite's public/ copy.
    if (existsSync(join(dist, path))) continue;
    mkdirSync(dirname(join(dist, path)), { recursive: true });
    symlinkSync(target, join(dist, path));
  }
}
