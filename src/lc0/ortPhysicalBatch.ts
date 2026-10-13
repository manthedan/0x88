// How many rows one session.run sees.
// 'declared' is the product default: a symbolic leading dim is batch 1, so a
// logical search batch becomes that many serial runs. 'logical' is the lab
// override for a dynamic leading dim: one run whose leading dim is the
// logical count. A fixed numeric dim is unchanged in either mode.
export type OrtPhysicalBatchMode = 'declared' | 'logical';

export function resolveOrtPhysicalBatchSize(declaredFirstDim: unknown, logicalCount: number, mode: OrtPhysicalBatchMode = 'declared'): number {
  const logical = Math.max(1, Math.floor(Number(logicalCount)) || 1);
  const fixed = typeof declaredFirstDim === 'number' && Number.isFinite(declaredFirstDim) && declaredFirstDim > 0 ? Math.floor(declaredFirstDim) : null;
  if (fixed !== null) return fixed;
  return mode === 'logical' ? logical : 1;
}
