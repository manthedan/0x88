#!/usr/bin/env node
// Policy check for the LC0 whole-model TVMJS/WebGPU runtime.
//
// HISTORY: until 2026-06-10 this enforced full research-only isolation (no
// TVMJS mentions anywhere in stable src). On 2026-06-10 the release owner
// promoted TVMJS to a VISIBLE, NON-DEFAULT runtime option in the arena and
// analysis pages (local evidence: 94/94 fixed-suite matches on t1, clean
// gates on t3/BT4; remaining caveats — non-Apple GPU coverage, hosting/cache
// policy — accepted for self-hosted use).
//
// The policy this script now enforces:
//   1. ORT stays the DEFAULT runtime: normalizeLc0Runtime must fall back to
//      'onnx' in both arena and analysis.
//   2. The stable NN runtime plumbing (runtimeRegistry, the generic
//      browserRuntimeEvaluator) stays TVMJS-free: page-level selection only.
//   3. TVMJS artifacts stay out of the committed tree (gitignored staging).
import { readFile } from 'node:fs/promises';

const FORBIDDEN = [
  {
    path: 'src/nn/runtimeRegistry.ts',
    forbidden: [/tvmjs/i, /lc0-tvmjs/i],
    reason: 'TVMJS selection is page-level; the stable runtime registry stays TVMJS-free.',
  },
  {
    path: 'src/nn/browserRuntimeEvaluator.ts',
    forbidden: [/tvmjs/i, /lc0-tvmjs/i],
    reason: 'TVMJS must not be instantiated by the stable browser runtime evaluator.',
  },
];

const DEFAULT_RUNTIME_ASSERTIONS = [
  { path: 'src/lc0/arenaBrowser.ts', pattern: /function normalizeLc0Runtime[\s\S]{0,600}?return 'onnx';/, reason: 'Arena LC0 runtime must default to onnx.' },
  { path: 'src/lc0/analysisBrowser.ts', pattern: /function normalizeLc0Runtime[\s\S]{0,600}?return 'onnx';/, reason: 'Analysis LC0 runtime must default to onnx.' },
];

async function main() {
  const failures = [];
  const checked = [];
  for (const check of FORBIDDEN) {
    const text = await readFile(check.path, 'utf8');
    checked.push(check.path);
    for (const re of check.forbidden) {
      if (re.test(text)) failures.push({ path: check.path, pattern: String(re), reason: check.reason });
    }
  }
  for (const assertion of DEFAULT_RUNTIME_ASSERTIONS) {
    const text = await readFile(assertion.path, 'utf8');
    checked.push(assertion.path);
    if (!assertion.pattern.test(text)) failures.push({ path: assertion.path, pattern: String(assertion.pattern), reason: assertion.reason });
  }

  const result = {
    schema: 'lc0_browser.tvmjs_research_only_check.v2',
    generatedAt: new Date().toISOString(),
    ok: failures.length === 0,
    checked,
    failures,
    note: 'Promoted 2026-06-10: TVMJS is a visible non-default LC0 runtime option; ORT remains the default and fallback, and the stable NN runtime plumbing stays TVMJS-free.',
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (failures.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
