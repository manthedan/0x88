import type { Lc0Evaluation, Lc0EvaluationProvider, Lc0EvaluatorInput } from './onnxEvaluator.ts';

/** Cancellation between physical calls, not merely between logical batches. */
export async function simulProbe(
  provider: Lc0EvaluationProvider,
  inputs: Lc0EvaluatorInput[],
  physical: number,
  signal: AbortSignal,
): Promise<Lc0Evaluation[]> {
  signal.throwIfAborted();
  if (physical > 1 && provider.evaluateBatch) return await provider.evaluateBatch(inputs);
  const results: Lc0Evaluation[] = [];
  for (const input of inputs) {
    signal.throwIfAborted();
    results.push(await provider.evaluate(input));
  }
  return results;
}
