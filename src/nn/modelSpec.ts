export interface ConvStudentSpec {
  name: string;
  filters: number;
  blocks: number;
  quantization: 'int8' | 'fp16' | 'fp32';
}

export function estimateParams(spec: ConvStudentSpec): number {
  const inputPlanes = 13;
  const actionSpace = 64 * 64 * 5;
  const stem = inputPlanes * spec.filters * 3 * 3 + spec.filters;
  const residual = spec.blocks * 2 * (spec.filters * spec.filters * 3 * 3 + spec.filters);
  const policyHead = spec.filters * actionSpace / 8; // compressed gathered-policy proxy
  const valueHead = spec.filters * 64 + 128 * 3;
  return Math.round(stem + residual + policyHead + valueHead);
}

export function estimateModelSizeMb(spec: ConvStudentSpec): number {
  const bytesPerParam = spec.quantization === 'int8' ? 1 : spec.quantization === 'fp16' ? 2 : 4;
  return estimateParams(spec) * bytesPerParam / (1024 * 1024);
}

export function estimatedPolicyTop1(spec: ConvStudentSpec): number {
  const capacity = Math.log2(1 + spec.filters * spec.blocks) / Math.log2(1 + 64 * 6);
  const quantPenalty = spec.quantization === 'int8' ? 0.015 : spec.quantization === 'fp16' ? 0.005 : 0;
  return Math.max(0, Math.min(1, 0.18 + 0.42 * capacity - quantPenalty));
}

export const CANDIDATE_SPECS: ConvStudentSpec[] = [
  { name: 'micro_16x2_int8', filters: 16, blocks: 2, quantization: 'int8' },
  { name: 'small_24x3_int8', filters: 24, blocks: 3, quantization: 'int8' },
  { name: 'balanced_48x5_int8', filters: 48, blocks: 5, quantization: 'int8' },
  { name: 'practical_64x6_int8', filters: 64, blocks: 6, quantization: 'int8' }
];
