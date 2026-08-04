import { isV0DeployProfile } from './engineCatalog.ts';

export const LC0_WHOLE_MODEL_WEBGPU_RUNTIME = 'whole-onnx-webgpu' as const;
export type BrowserLc0Runtime = 'onnx' | 'hybrid-ort-heads' | 'hybrid-wgsl-heads' | typeof LC0_WHOLE_MODEL_WEBGPU_RUNTIME;

export function normalizeLc0Runtime(value: string | null): BrowserLc0Runtime {
  if (isV0DeployProfile()) return 'onnx';
  const raw = (value ?? '').toLowerCase();
  if (raw === LC0_WHOLE_MODEL_WEBGPU_RUNTIME || raw === 'tvmjs-webgpu' || raw === 'lc0-tvmjs-webgpu') return LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
  if (raw === 'hybrid' || raw === 'lc0web' || raw === 'hybrid-ort-heads' || raw === 'wgsl-encoder') return 'hybrid-ort-heads';
  if (raw === 'hybrid-wgsl-heads' || raw === 'wgsl-heads' || raw === 'wgsl') return 'hybrid-wgsl-heads';
  return 'onnx';
}

export function initialLc0Runtime(params: URLSearchParams): BrowserLc0Runtime {
  if (isV0DeployProfile()) return 'onnx';
  if (params.get('headBackend') === 'wgsl' || params.get('hybridHeads') === 'wgsl') return 'hybrid-wgsl-heads';
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
}

export function lc0WholeModelRuntimeRequested(params: URLSearchParams): boolean {
  if (isV0DeployProfile()) return false;
  return (
    normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime')) === LC0_WHOLE_MODEL_WEBGPU_RUNTIME ||
    params.get('enableWholeModelWebgpu') === '1' ||
    params.get('enableTvmjs') === '1'
  );
}

export function installExperimentalLc0RuntimeOption(select: HTMLSelectElement): void {
  if ([...select.options].some((option) => option.value === LC0_WHOLE_MODEL_WEBGPU_RUNTIME)) return;
  const option = document.createElement('option');
  option.value = LC0_WHOLE_MODEL_WEBGPU_RUNTIME;
  option.textContent = 'TVM whole-model WebGPU (fast, small net)';
  select.appendChild(option);
}

export function lc0RuntimeLabel(runtime: BrowserLc0Runtime): string {
  if (runtime === LC0_WHOLE_MODEL_WEBGPU_RUNTIME) return 'TVM whole-model WebGPU (research)';
  if (runtime === 'hybrid-wgsl-heads') return 'WGSL encoder + WGSL heads';
  if (runtime === 'hybrid-ort-heads') return 'WGSL encoder + ORT heads';
  return 'ORT ONNX';
}

export function boundedIntValue(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Math.floor(Number(value ?? fallback));
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : fallback;
}

export function lc0EncoderLayers(params: URLSearchParams): number {
  return boundedIntValue(params.get('encoderLayers') ?? params.get('layers'), 10, 1, 32);
}

export function lc0WholeModelPhysicalBatch(params: URLSearchParams): number {
  return boundedIntValue(params.get('wholeModelBatch') ?? params.get('tvmBatch') ?? params.get('compiledBatch'), 8, 1, 64);
}

export function lc0WholeModelTensorCache(params: URLSearchParams): boolean {
  return params.get('wholeModelTensorCache') === '1' || params.get('tensorCache') === '1';
}
