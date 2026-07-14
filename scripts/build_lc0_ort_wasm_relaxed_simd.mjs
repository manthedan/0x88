#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const ORT_REPOSITORY = 'https://github.com/microsoft/onnxruntime.git';
const ORT_REMOTE_ALIASES = new Set([
  ORT_REPOSITORY,
  'git@github.com:microsoft/onnxruntime.git',
  'ssh://git@github.com/microsoft/onnxruntime.git',
]);
const ORT_TAG = 'v1.27.0';
const ORT_COMMIT = '8f0278c77bf44b0cc83c098c6c722b92a36ac4b5';
const EMSCRIPTEN_VERSION = '4.0.23';
const BUILD_JOBS = process.env.ORT_BUILD_JOBS ?? '8';
const sourceDir = resolve(process.env.ORT_SOURCE_DIR ?? '.local_ort/onnxruntime');
const outputRoot = resolve(process.env.ORT_WASM_OUTPUT_DIR ?? 'public/ort-experimental');
const skipBuild = process.env.ORT_SKIP_BUILD === '1';
function artifactNames(variant) {
  const simd = variant === 'relaxed' ? 'relaxedsimd' : 'simd';
  return [
    `ort-wasm-${simd}-threaded.asyncify.mjs`,
    `ort-wasm-${simd}-threaded.asyncify.wasm`,
  ];
}

function buildFlags(variant) {
  return [
    '--config', 'Release',
    '--build_dir', resolve(`.local_ort/build-${variant}`),
    '--parallel', BUILD_JOBS,
    '--build_wasm',
    '--skip_tests',
    '--emsdk_version', EMSCRIPTEN_VERSION,
    '--enable_wasm_simd',
    '--enable_wasm_threads',
    '--use_webgpu',
    '--disable_wasm_exception_catching',
    '--disable_rtti',
    ...(variant === 'relaxed' ? ['--enable_wasm_relaxed_simd'] : []),
  ];
}

if (process.argv.includes('--print-config')) {
  console.log(JSON.stringify({
    repository: ORT_REPOSITORY,
    tag: ORT_TAG,
    commit: ORT_COMMIT,
    emscriptenVersion: EMSCRIPTEN_VERSION,
    artifacts: Object.fromEntries(['fixed', 'relaxed'].map((variant) => [variant, artifactNames(variant)])),
    variants: Object.fromEntries(['fixed', 'relaxed'].map((variant) => [variant, buildFlags(variant)])),
  }, null, 2));
  process.exit(0);
}

function run(command, args, options = {}) {
  console.log(`$ ${command} ${args.join(' ')}`);
  execFileSync(command, args, { stdio: 'inherit', ...options });
}

function output(command, args, options = {}) {
  return execFileSync(command, args, { encoding: 'utf8', ...options }).trim();
}

function findFile(directory, basename) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      const found = findFile(path, basename);
      if (found) return found;
    } else if (entry.name === basename) {
      return path;
    }
  }
  return undefined;
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function ensureCommonJsBoundary() {
  const packagePath = join(dirname(sourceDir), 'package.json');
  if (existsSync(packagePath)) {
    const packageJson = JSON.parse(readFileSync(packagePath, 'utf8'));
    if (packageJson.type !== 'commonjs') throw new Error(`${packagePath} must set "type": "commonjs" for ORT build helpers`);
    return;
  }
  writeFileSync(packagePath, `${JSON.stringify({ private: true, type: 'commonjs' }, null, 2)}\n`);
}

function prepareSource() {
  mkdirSync(dirname(sourceDir), { recursive: true });
  ensureCommonJsBoundary();
  if (!existsSync(sourceDir)) {
    run('git', ['clone', '--branch', ORT_TAG, '--depth', '1', '--recurse-submodules', '--shallow-submodules', ORT_REPOSITORY, sourceDir]);
  }
  const remote = output('git', ['-C', sourceDir, 'remote', 'get-url', 'origin']);
  if (!ORT_REMOTE_ALIASES.has(remote)) throw new Error(`unexpected ORT source remote: ${remote}`);
  const status = output('git', ['-C', sourceDir, 'status', '--porcelain']);
  if (status) throw new Error(`ORT source tree is dirty: ${sourceDir}`);
  const commit = output('git', ['-C', sourceDir, 'rev-parse', 'HEAD']);
  if (commit !== ORT_COMMIT) {
    throw new Error(`ORT source must resolve ${ORT_TAG} to ${ORT_COMMIT}, found ${commit}`);
  }
  return commit;
}

function buildVariant(variant) {
  const buildDir = resolve(`.local_ort/build-${variant}`);
  const flags = buildFlags(variant);
  if (!skipBuild) run(join(sourceDir, 'build.sh'), flags, { cwd: sourceDir });
  const destination = join(outputRoot, variant);
  mkdirSync(destination, { recursive: true });
  const artifacts = artifactNames(variant).map((name) => {
    const source = findFile(buildDir, name);
    if (!source) throw new Error(`missing ${name} under ${buildDir}`);
    const target = join(destination, name);
    cpSync(source, target);
    return { name, path: target, bytes: readFileSync(target).byteLength, sha256: sha256(target) };
  });
  return { variant, flags, artifacts };
}

const commit = prepareSource();
const variants = [buildVariant('fixed'), buildVariant('relaxed')];
const fixedWasm = variants[0].artifacts.find((artifact) => artifact.name.endsWith('.wasm')).path;
const relaxedWasm = variants[1].artifacts.find((artifact) => artifact.name.endsWith('.wasm')).path;
run(process.execPath, ['scripts/inspect_wasm_simd.mjs', fixedWasm, relaxedWasm]);
run(process.execPath, [
  'scripts/inspect_wasm_simd.mjs',
  '--forbid-op', 'i32x4.relaxed_dot_i8x16_i7x16_add_s',
  fixedWasm,
]);
run(process.execPath, [
  'scripts/inspect_wasm_simd.mjs',
  '--require-op', 'i32x4.relaxed_dot_i8x16_i7x16_add_s',
  relaxedWasm,
]);

const manifest = {
  schema: 'lc0_browser.ort_wasm_relaxed_simd_build.v1',
  repository: ORT_REPOSITORY,
  tag: ORT_TAG,
  commit,
  expectedEmscriptenVersion: EMSCRIPTEN_VERSION,
  generatedAt: new Date().toISOString(),
  variants: variants.map((variant) => ({
    ...variant,
    flags: variant.flags.map((value) => value.startsWith(`${process.cwd()}/`) ? relative(process.cwd(), value) : value),
    artifacts: variant.artifacts.map((artifact) => ({
      ...artifact,
      path: relative(process.cwd(), artifact.path),
    })),
  })),
};
mkdirSync(outputRoot, { recursive: true });
writeFileSync(join(outputRoot, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`ORT WASM artifacts staged under ${outputRoot}`);
