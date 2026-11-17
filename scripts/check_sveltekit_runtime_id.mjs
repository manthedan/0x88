#!/usr/bin/env node
import { readdir, readFile } from 'node:fs/promises';
import { dirname, extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const RUNTIME_ID = /__sveltekit_[a-z0-9_]+/g;

async function filesUnder(root, accept) {
  const files = [];
  async function visit(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (accept(path)) files.push(path);
    }
  }
  await visit(root);
  return files;
}

async function idsInFiles(files) {
  const ids = new Set();
  for (const file of files) {
    const source = await readFile(file, 'utf8');
    for (const id of source.match(RUNTIME_ID) || []) {
      if (id !== '__sveltekit_sw') ids.add(id);
    }
  }
  return [...ids].sort();
}

// Kit 3 passes its payload into kit.init instead of baking the global id into
// client chunks. Bind each page to the build version and its emitted entries.
async function checkPayloadBuild(output, pages, runtimeId) {
  const { version } = JSON.parse(await readFile(join(output, '_app', 'version.json'), 'utf8'));
  if (typeof version !== 'string' || !version) throw new Error('Missing SvelteKit build version');
  const encodedVersion = JSON.stringify(version).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  for (const page of pages) {
    const html = await readFile(page, 'utf8');
    const payload = new RegExp(`${runtimeId}\\s*=\\s*\\{[\\s\\S]*?version:\\s*${encodedVersion}`);
    if (!payload.test(html) || !html.includes(`kit.init(${runtimeId})`)) {
      throw new Error(`SvelteKit payload version mismatch: ${page}`);
    }
    for (const entry of ['start', 'app']) {
      const imports = [...html.matchAll(new RegExp(`import\\(["']([^"']*\\/_app/immutable/entry/${entry}\\.[^"'/]+\\.js)["']\\)`, 'g'))];
      if (imports.length !== 1) throw new Error(`Missing SvelteKit ${entry} entry: ${page}`);
      const path = resolve(dirname(page), imports[0][1]);
      if (!path.startsWith(join(output, '_app', 'immutable', 'entry') + sep)) throw new Error(`Unsafe SvelteKit entry: ${path}`);
      await readFile(path, 'utf8');
    }
  }
}

export async function checkSvelteKitRuntimeId(outputDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client') {
  const output = resolve(outputDir);
  const immutable = join(output, '_app', 'immutable');
  const [scripts, pages] = await Promise.all([
    filesUnder(immutable, (path) => extname(path) === '.js'),
    filesUnder(output, (path) => extname(path) === '.html'),
  ]);
  const [scriptIds, pageIds] = await Promise.all([idsInFiles(scripts), idsInFiles(pages)]);

  if (scriptIds.length === 0 && pageIds.length === 1) {
    await checkPayloadBuild(output, pages, pageIds[0]);
    return { status: 'SVELTEKIT_RUNTIME_ID_CHECK_DONE', runtimeId: pageIds[0] };
  }

  if (scriptIds.length !== 1 || pageIds.length !== 1 || scriptIds[0] !== pageIds[0]) {
    throw new Error(
      `SvelteKit runtime id mismatch: scripts=${scriptIds.join(',') || 'none'} pages=${pageIds.join(',') || 'none'}. ` +
        'Clean .svelte-kit/output and rebuild before publishing.',
    );
  }

  return { status: 'SVELTEKIT_RUNTIME_ID_CHECK_DONE', runtimeId: scriptIds[0] };
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  checkSvelteKitRuntimeId(process.argv[2])
    .then((report) => {
      console.log(JSON.stringify(report));
    })
    .catch((error) => {
      console.error(error.message || error);
      process.exit(1);
    });
}
