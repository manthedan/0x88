#!/usr/bin/env node
import { rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export async function cleanSvelteKitBuild(outputDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client') {
  const projectRoot = resolve('.');
  const output = resolve(outputDir);
  if (output === projectRoot) throw new Error('Refusing to clean the project root as a build output');

  await Promise.all([
    rm(resolve('.svelte-kit', 'output'), { recursive: true, force: true }),
    rm(output, { recursive: true, force: true }),
  ]);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  cleanSvelteKitBuild(process.argv[2]).catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
