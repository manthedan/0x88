#!/usr/bin/env node
import { rm } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const SAFE_OUTPUT_NAME = /^dist(?:[-_.][a-z0-9][a-z0-9._-]*)?$/i;

export async function cleanSvelteKitBuild(outputDir = process.env.NETLIFY_R2_RELEASE_DIST || 'dist-client', projectDir = '.') {
  const projectRoot = resolve(projectDir);
  const output = resolve(projectRoot, outputDir);
  const projectRelativeOutput = relative(projectRoot, output);
  const isDirectChild =
    projectRelativeOutput &&
    !isAbsolute(projectRelativeOutput) &&
    !projectRelativeOutput.startsWith(`..${sep}`) &&
    projectRelativeOutput !== '..' &&
    !projectRelativeOutput.includes(sep);

  if (!isDirectChild || !SAFE_OUTPUT_NAME.test(projectRelativeOutput)) {
    throw new Error(`Refusing to clean unsafe build output "${outputDir}". ` + 'Expected a direct project directory named dist or dist-<name>.');
  }

  await Promise.all([rm(resolve(projectRoot, '.svelte-kit', 'output'), { recursive: true, force: true }), rm(output, { recursive: true, force: true })]);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  cleanSvelteKitBuild(process.argv[2]).catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
