// Page-side half of the kernel-bench contract (see scripts/kernel_hillclimb/remote/README.md):
// a run either produces a complete, trustworthy result or an error string. Benchmark
// failures, uncaptured WebGPU errors, error-scope errors and device loss are errors,
// never a partial "ok" result.

/**
 * @param {{ result: object, benchRequested: boolean, benchError?: unknown,
 *           gpuErrors?: string[], deviceLost?: string|null }} s
 * @returns {{ ok: true, result: object } | { ok: false, error: string }}
 */
export function settleProbe({ result, benchRequested, benchError = null, gpuErrors = [], deviceLost = null }) {
  const problems = [];
  if (benchError) problems.push(`benchmark failed: ${benchError?.stack ?? benchError}`);
  if (gpuErrors.length) problems.push(`WebGPU errors: ${gpuErrors.join(' | ')}`);
  if (deviceLost) problems.push(`device lost: ${deviceLost}`);
  if (benchRequested && result?.webgpu?.available !== true) problems.push(`no WebGPU adapter: ${result?.webgpu?.reason ?? 'unknown'}`);
  if (benchRequested && result?.webgpu?.deviceError) problems.push(`requestDevice failed: ${result.webgpu.deviceError}`);
  if (benchRequested && result?.webgpu?.available && !result.bench && !result.webgpu.deviceError) problems.push('benchmark produced no result');
  return problems.length ? { ok: false, error: problems.join('; ') } : { ok: true, result };
}

/** Publish the outcome on window per the contract. */
export function publish(win, outcome) {
  if (outcome.ok) win.__kernelBenchResult = outcome.result;
  else win.__kernelBenchError = outcome.error;
}
