#!/usr/bin/env node
import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';

const CHECKS = [
  {
    path: 'src/nn/runtimeRegistry.ts',
    forbidden: [/tvmjs/i, /lc0-tvmjs/i, /lc0-tvmjs-webgpu/i],
    reason: 'TVMJS must not be part of the stable browser runtime registry.',
  },
  {
    path: 'src/nn/browserRuntimeEvaluator.ts',
    forbidden: [/tvmjs/i, /lc0-tvmjs/i, /lc0-tvmjs-webgpu/i],
    reason: 'TVMJS must not be instantiated by the stable browser runtime evaluator.',
  },
  {
    path: 'lc0-arena.html',
    forbidden: [/tvmjs/i, /lc0-tvmjs/i, /lc0-tvmjs-webgpu/i],
    reason: 'TVMJS must not be exposed as an arena LC0 backend option.',
  },
];

async function walkFiles(dir) {
  const out = [];
  for (const entry of await readdir(dir)) {
    const path = join(dir, entry);
    const st = await stat(path);
    if (st.isDirectory()) out.push(...await walkFiles(path));
    else out.push(path);
  }
  return out;
}

async function main() {
  const failures = [];
  const checked = [];
  for (const check of CHECKS) {
    const text = await readFile(check.path, 'utf8');
    checked.push(check.path);
    for (const re of check.forbidden) {
      if (re.test(text)) failures.push({ path: check.path, pattern: String(re), reason: check.reason });
    }
  }

  const srcFiles = (await walkFiles('src')).filter((path) => /\.(ts|tsx|js|mjs)$/.test(path));
  const srcMentions = [];
  for (const path of srcFiles) {
    const text = await readFile(path, 'utf8');
    if (/tvmjs|lc0-tvmjs|lc0-tvmjs-webgpu/i.test(text)) srcMentions.push(path);
  }
  if (srcMentions.length) failures.push({ path: 'src/**', pattern: 'tvmjs|lc0-tvmjs|lc0-tvmjs-webgpu', reason: `TVMJS mentions found in stable src tree: ${srcMentions.join(', ')}` });

  const result = {
    schema: 'lc0_browser.tvmjs_research_only_check.v1',
    generatedAt: new Date().toISOString(),
    ok: failures.length === 0,
    checked,
    stableSrcFilesChecked: srcFiles.length,
    failures,
    note: 'TVMJS whole-model WebGPU must remain isolated in smoke/scripts/docs until promotion evidence and release policy exist.',
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (failures.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
