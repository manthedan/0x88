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
//   1. ORT stays the DEFAULT runtime: the shared normalizeLc0Runtime falls
//      back to 'onnx', and both arena and analysis consume that normalizer.
//   2. LC0 whole-model TVMJS stays page-level and non-default. Centipawn has
//      a separately promoted TVMJS runtime in the shared registry.
import { readFile } from 'node:fs/promises';

const FORBIDDEN = [
  {
    path: 'src/nn/runtimeRegistry.ts',
    forbidden: [/lc0-tvmjs/i],
    reason: 'LC0 TVMJS selection remains page-level rather than registry-promoted.',
  },
  {
    path: 'src/nn/browserRuntimeEvaluator.ts',
    forbidden: [/lc0-tvmjs/i],
    reason: 'The shared evaluator may host Centipawn TVMJS, but not the LC0 whole-model runtime.',
  },
];

const DEFAULT_RUNTIME_ASSERTIONS = [
  {
    path: 'src/lc0/browserLc0Runtime.ts',
    pattern: /export function normalizeLc0Runtime[\s\S]{0,600}?return 'onnx';/,
    reason: 'The shared LC0 runtime normalizer must default to onnx.',
  },
  {
    path: 'src/lc0/arenaBrowser.ts',
    pattern: /normalizeLc0Runtime,\s*} from '\.\/browserLc0Runtime\.ts';/,
    reason: 'Arena must consume the shared LC0 runtime normalizer.',
  },
  {
    path: 'src/lc0/analysisBrowser.ts',
    pattern: /normalizeLc0Runtime,\s*} from '\.\/browserLc0Runtime\.ts';/,
    reason: 'Analysis must consume the shared LC0 runtime normalizer.',
  },
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
    note: 'LC0 TVMJS remains visible and non-default. This gate does not prohibit the separately promoted Centipawn TVMJS runtime.',
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (failures.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
