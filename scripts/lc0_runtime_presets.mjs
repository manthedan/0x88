export const LC0_WEBGPU_RESEARCH_B4_PRESET = 'lc0-webgpu-research-b4';

const LC0_WEBGPU_RESEARCH_B4 = Object.freeze({
  runtimes: ['hybrid-wgsl-heads'],
  headBackend: 'wgsl',
  headBackends: ['wgsl'],
  inputBackend: 'wasm',
  legalPriorsBackend: 'js',
  legalPriorsBackends: ['js'],
  encoderKernel: 'mixed-tvm-ffn-smolgen-project',
  encoderKernels: ['mixed-tvm-ffn-smolgen-project'],
  lc0BatchSize: 4,
  batch: 4,
  batches: [4],
  batchPipelineDepth: 1,
  batchPipelineDepths: [1],
});

export const LC0_RUNTIME_PRESETS = Object.freeze({
  [LC0_WEBGPU_RESEARCH_B4_PRESET]: LC0_WEBGPU_RESEARCH_B4,
});

const DEFAULT_OPTION_FLAGS = Object.freeze({
  runtimes: ['--runtimes'],
  headBackend: ['--head-backend'],
  headBackends: ['--head-backends'],
  inputBackend: ['--input-backend'],
  legalPriorsBackend: ['--legal-priors-backend', '--hybrid-legal-priors'],
  legalPriorsBackends: ['--legal-priors-backends', '--legal-priors-backend'],
  encoderKernel: ['--encoder-kernel', '--encoder-kernel-variant'],
  encoderKernels: ['--encoder-kernels'],
  lc0BatchSize: ['--lc0-batch-size', '--batch-size', '--batch'],
  batch: ['--batch'],
  batches: ['--batches'],
  batchPipelineDepth: ['--batch-pipeline-depth', '--pipeline-depth'],
  batchPipelineDepths: ['--batch-pipeline-depths', '--pipeline-depths'],
});

function explicitFlags(argv) {
  const flags = new Set();
  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    flags.add(arg.includes('=') ? arg.slice(0, arg.indexOf('=')) : arg);
  }
  return flags;
}

function copyPresetValue(value) {
  return Array.isArray(value) ? [...value] : value;
}

export function applyLc0RuntimePreset(args, argv, optionFlags = DEFAULT_OPTION_FLAGS) {
  if (!args.preset) return args;
  const preset = LC0_RUNTIME_PRESETS[args.preset];
  if (!preset) throw new Error(`Unknown --preset: ${args.preset}; expected one of ${Object.keys(LC0_RUNTIME_PRESETS).join(', ')}`);
  const flags = explicitFlags(argv);
  for (const [key, value] of Object.entries(preset)) {
    if (!(key in args)) continue;
    const aliases = optionFlags[key] ?? DEFAULT_OPTION_FLAGS[key] ?? [];
    if (aliases.some((flag) => flags.has(flag))) continue;
    args[key] = copyPresetValue(value);
  }
  return args;
}

export function lc0RuntimeConfiguration(args) {
  return {
    preset: args.preset || null,
    runtimes: args.runtimes,
    headBackend: args.headBackend,
    inputBackend: args.inputBackend,
    encoderKernel: args.encoderKernel,
    encoderKernels: args.encoderKernels,
    legalPriorsBackend: args.legalPriorsBackend,
    legalPriorsBackends: args.legalPriorsBackends,
    leafBatchSize: args.lc0BatchSize ?? args.batch,
    leafBatchSizes: args.batches,
    speculativeSearchPipelineDepth: args.batchPipelineDepth,
    speculativeSearchPipelineDepths: args.batchPipelineDepths,
  };
}
