#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';

const PLAYABLE_100M_MODELS = [
  { id: 'cnn-32x4-100m-e3', architecture: 'cnn', onnx: 'public/models/cnn_32x4_100m_e3.onnx', meta: 'public/models/cnn_32x4_100m_e3.meta.json' },
  { id: 'cnn-48x5-100m-e3', architecture: 'cnn', onnx: 'public/models/cnn_48x5_100m_e3.onnx', meta: 'public/models/cnn_48x5_100m_e3.meta.json' },
  { id: 'cnn-64x6-100m-e3', architecture: 'cnn', onnx: 'public/models/cnn_64x6_100m_e3.onnx', meta: 'public/models/cnn_64x6_100m_e3.meta.json' },
  { id: 'cnn-80x5-100m-e3', architecture: 'cnn', onnx: 'public/models/cnn_80x5_100m_e3.onnx', meta: 'public/models/cnn_80x5_100m_e3.meta.json' },
  { id: 'cnn-96x8-100m-e8', architecture: 'cnn', onnx: 'public/models/cnn96x8_100m_e8.onnx', meta: 'public/models/cnn96x8_100m_e8.meta.json' },
  { id: 'moveformer-80x5-100m-e8-k128', architecture: 'moveformer', onnx: 'public/models/moveformer_80x5_100m_e8_k128.onnx', meta: 'public/models/moveformer_80x5_100m_e8_k128.meta.json' },
  { id: 'bt4-sampled1b-best', architecture: 'squareformer', onnx: 'public/models/bt4_sampled1b_best.onnx', meta: 'public/models/bt4_sampled1b_best.meta.json' },
  { id: 'bt4-1b-mix50-noav-best', architecture: 'squareformer', onnx: 'public/models/bt4_1b_mix50_noav_best.onnx', meta: 'public/models/bt4_1b_mix50_noav_best.meta.json' },
  { id: 'bt4-1b-mix50-noav-best-ema', architecture: 'squareformer', onnx: 'public/models/bt4_1b_mix50_noav_best_ema.onnx', meta: 'public/models/bt4_1b_mix50_noav_best_ema.meta.json' },
];

function arg(name, fallback = '') {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function flag(name) { return process.argv.includes(name); }
function listArg(name, fallback) { return arg(name, fallback).split(',').map((s) => s.trim()).filter(Boolean); }
function sha256(path) { return createHash('sha256').update(readFileSync(path)).digest('hex'); }
function readJson(path) { return JSON.parse(readFileSync(path, 'utf8')); }
function safeId(s) { return s.replace(/[^A-Za-z0-9_.-]+/g, '_'); }
function metaStr(meta, key) { return typeof meta[key] === 'string' ? meta[key] : undefined; }
function metaNum(meta, key) { return Number.isFinite(Number(meta[key])) ? Number(meta[key]) : undefined; }

function targetRuntime(target, threads) {
  if (target === 'browser_webgpu') return { name: 'onnxruntime-web', execution_providers: ['webgpu', 'wasm'], host_language: 'typescript', threads };
  if (target === 'browser_wasm') return { name: 'onnxruntime-web', execution_providers: ['wasm'], host_language: 'typescript', threads };
  if (target === 'local_cuda') return { name: 'onnxruntime-native', execution_providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], host_language: 'rust', threads };
  if (target === 'aws_batch_gpu') return { name: 'onnxruntime-native', execution_providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], host_language: 'mixed', threads };
  return { name: 'onnxruntime-native', execution_providers: ['CPUExecutionProvider'], host_language: 'rust', threads };
}

function inputCard(model, meta) {
  const kind = String(meta.kind ?? '').toLowerCase();
  const arch = String(meta.architecture ?? '').toLowerCase();
  let encoding = 'board_planes';
  if (model.architecture === 'moveformer' || arch.includes('move_token') || kind.includes('moveformer')) encoding = 'board_planes_plus_legal_action_features';
  else if (model.architecture === 'squareformer' || kind.includes('squareformer')) encoding = String(meta.input_format ?? '').includes('compact') ? 'compact_square_tokens' : 'square_feature_tokens';
  const fixedK = metaNum(meta, 'onnx_fixed_legal_moves') ?? metaNum(meta, 'max_legal_moves');
  return {
    encoding,
    history_plies: metaNum(meta, 'history_plies') ?? 0,
    input_planes: metaNum(meta, 'input_planes'),
    legal_bucket: fixedK ? `k${fixedK}` : (meta.onnx_dynamic_legal ? 'dynamic' : 'none'),
    batch_axis: meta.onnx_dynamic_batch !== false,
  };
}

function outputsCard(model, meta) {
  if (Array.isArray(meta.outputs) && meta.outputs.length) return meta.outputs.map(String);
  const outputs = model.architecture === 'moveformer' ? ['policy_logits_legal', 'wdl_logits'] : ['policy', 'wdl'];
  if (meta.av_head_exported === true) outputs.push('action_values');
  if (Array.isArray(meta.aux_heads_exported)) {
    for (const name of meta.aux_heads_exported.map(String)) if (!outputs.includes(name)) outputs.push(name);
  }
  return outputs;
}

function makeCard(model, meta, target, threads) {
  return {
    schema: 'export_target_card_v1',
    model: {
      id: model.id,
      architecture: model.architecture,
      checkpoint: metaStr(meta, 'checkpoint') ?? metaStr(meta, 'source_checkpoint'),
      model_sha256: sha256(model.onnx),
      meta_sha256: sha256(model.meta),
    },
    artifact: {
      onnx: model.onnx,
      meta: model.meta,
      onnx_bytes: statSync(model.onnx).size,
      external_data: false,
      simplified: /onnxsim|single/.test(model.onnx),
    },
    target,
    runtime: targetRuntime(target, threads),
    precision: /int8/i.test(model.onnx) ? 'dynamic_int8' : 'fp32',
    input: inputCard(model, meta),
    outputs: outputsCard(model, meta),
    parity: { status: 'pending', required_before_promotion: true },
    benchmarks: { status: 'pending', required_before_promotion: true },
    known_issues: target === 'browser_webgpu' ? ['Requires real browser WebGPU smoke; Node ORT Web is not sufficient.'] : [],
  };
}

function runBench(model, outDir) {
  const args = [
    '--experimental-strip-types', 'eval/onnx_inference_benchmark.mjs',
    '--model', model.onnx,
    '--meta', model.meta,
    '--label', model.id,
    '--positions', arg('--positions', '16'),
    '--repeats', arg('--repeats', '2'),
    '--warmup', arg('--warmup', '4'),
    '--batches', arg('--batches', '1,4,16'),
  ];
  const env = { ...process.env };
  const threads = arg('--threads', '');
  if (threads) env.ORT_NUM_THREADS = threads;
  const proc = spawnSync(process.execPath, args, { encoding: 'utf8', env, maxBuffer: 1024 * 1024 * 64 });
  const logPath = join(outDir, `${safeId(model.id)}.node_ort_bench.log`);
  writeFileSync(logPath, `${proc.stdout}${proc.stderr}`, 'utf8');
  if (proc.status !== 0) throw new Error(`benchmark failed for ${model.id}; see ${logPath}`);
  return logPath;
}

const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
const outDir = arg('--out', `artifacts/export_target_cards/playable_100m_${timestamp}`);
const targets = listArg('--targets', 'browser_wasm,browser_webgpu,native_cpu,mac_mini_native');
const threads = Number(arg('--threads', '0'));
const selected = new Set(listArg('--models', PLAYABLE_100M_MODELS.map((m) => m.id).join(',')));
const doBench = flag('--bench');
mkdirSync(outDir, { recursive: true });

const summary = { schema: 'playable_100m_model_matrix_v1', created_utc: new Date().toISOString(), out_dir: outDir, targets, models: [] };
for (const model of PLAYABLE_100M_MODELS.filter((m) => selected.has(m.id))) {
  const missing = [model.onnx, model.meta].filter((p) => !existsSync(p));
  if (missing.length) throw new Error(`${model.id} missing artifact(s): ${missing.join(', ')}`);
  const meta = readJson(model.meta);
  const cardPaths = [];
  for (const target of targets) {
    const card = makeCard(model, meta, target, threads);
    const path = join(outDir, `${safeId(model.id)}.${target}.export_target_card_v1.json`);
    writeFileSync(path, `${JSON.stringify(card, null, 2)}\n`, 'utf8');
    cardPaths.push(path);
  }
  const benchLog = doBench ? runBench(model, outDir) : null;
  summary.models.push({ id: model.id, onnx: model.onnx, meta: model.meta, card_paths: cardPaths, bench_log: benchLog });
}
const summaryPath = join(outDir, 'summary.json');
writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
console.log(`wrote ${summary.models.length} model entries to ${outDir}`);
console.log(summaryPath);
