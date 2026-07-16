#!/usr/bin/env node
import { readdir, readFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
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

export async function checkSvelteKitRuntimeId(outputDir = 'dist-client') {
  const output = resolve(outputDir);
  const immutable = join(output, '_app', 'immutable');
  const [scripts, pages] = await Promise.all([
    filesUnder(immutable, (path) => extname(path) === '.js'),
    filesUnder(output, (path) => extname(path) === '.html'),
  ]);
  const [scriptIds, pageIds] = await Promise.all([idsInFiles(scripts), idsInFiles(pages)]);

  if (scriptIds.length !== 1 || pageIds.length !== 1 || scriptIds[0] !== pageIds[0]) {
    throw new Error(
      `SvelteKit runtime id mismatch: scripts=${scriptIds.join(',') || 'none'} pages=${pageIds.join(',') || 'none'}. ` +
      'Clean .svelte-kit/output and rebuild before publishing.',
    );
  }

  return { status: 'SVELTEKIT_RUNTIME_ID_CHECK_DONE', runtimeId: scriptIds[0] };
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  checkSvelteKitRuntimeId(process.argv[2]).then((report) => {
    console.log(JSON.stringify(report));
  }).catch((error) => {
    console.error(error.message || error);
    process.exit(1);
  });
}
