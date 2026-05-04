#!/usr/bin/env node
import { chooseBackend } from '../src/browser/backendPlan.ts';

const strategy = process.env.TINY_LEELA_RUNTIME_STRATEGY ?? 'progressive_fallback';

const profiles = [
  { id: 'desktop_chrome_webgpu', hasWebGpu: true, hasWasmSimd: true, hasIndexedDb: true, deviceClass: 'desktop' },
  { id: 'mobile_safari_wasm', hasWebGpu: false, hasWasmSimd: true, hasIndexedDb: true, deviceClass: 'mobile' },
  { id: 'locked_down_browser', hasWebGpu: false, hasWasmSimd: false, hasIndexedDb: false, deviceClass: 'desktop' },
  { id: 'desktop_no_cache', hasWebGpu: true, hasWasmSimd: true, hasIndexedDb: false, deviceClass: 'desktop' }
];

const latencyByBackend = { webgpu: 85, wasm: 210, none: 1000 };
let compatible = 0;
let latency = [];
let cacheHits = 0;
let progressive = 0;

for (const profile of profiles) {
  const plan = chooseBackend(profile, strategy);
  if (plan.backend !== 'none') compatible++;
  if (plan.cacheModel) cacheHits++;
  if (plan.progressiveLoad) progressive++;
  latency.push(latencyByBackend[plan.backend]);
}
latency.sort((a, b) => a - b);
const median = (latency[1] + latency[2]) / 2;

// Fixed runtime strategy score: compatibility dominates, then latency, then user-visible loading resilience.
const browserCompatibilityScore = compatible * 25 + cacheHits * 3 + progressive * 2 - median / 100;

console.log(`METRIC browser_compatibility_score=${browserCompatibilityScore.toFixed(6)}`);
console.log(`METRIC median_move_latency_ms=${median.toFixed(6)}`);
console.log(`METRIC runtime_profiles_evaluated=${profiles.length}`);
console.log(`METRIC compatible_profiles=${compatible}`);
