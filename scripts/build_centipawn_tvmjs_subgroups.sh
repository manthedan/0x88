#!/usr/bin/env bash
# Rebuild the promoted f32/b16 recipe, changing only supports_subgroups.
# No compiler checkout/build or production/default mutation. See the evidence doc.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${TVM_SRC:?Set TVM_SRC to the read-only v2 compiler checkout}"
: "${TVM_ENV:?Set TVM_ENV to the TVM Python environment}"
: "${ONNX_PYTHON:?Set ONNX_PYTHON to Python with onnxsim v0.6.4}"
OUT="$REPO/artifacts/centipawn-subgroups"
V2="$REPO/public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16"
mkdir -p "$OUT"
[[ "$(git -C "$TVM_SRC" rev-parse HEAD)" == 94398e51fbb8bf9022da85d83e25f914ef033110 ]] || { echo 'Wrong compiler commit' >&2; exit 1; }
git -C "$TVM_SRC" diff -- src/tirx/transform/storage_rewrite.cc tests/python/tirx-transform/test_tir_transform_pointer_value_type_rewrite.py > "$OUT/compiler.patch"
cmp "$OUT/compiler.patch" "$V2/tvm-vector-size-storage-rewrite.patch"
export PYTHONDONTWRITEBYTECODE=1
"$ONNX_PYTHON" - <<'PY'
import hashlib, onnx, onnxsim
from pathlib import Path
src = Path('public/models/bt4_soap_rem_c19000_final.onnx')
assert hashlib.sha256(src.read_bytes()).hexdigest() == 'de2dd68518f6e2ed162da5ee27c67a5a4795567d6e39972e3705824572b7660e'
assert onnxsim.__version__ == 'v0.6.4'
model, valid = onnxsim.simplify(onnx.load(src), overwrite_input_shapes={'tokens': [16,64,24], 'attack_summary': [16,64,28]})
assert valid and len(model.graph.node) == 342 and len(model.graph.initializer) == 139
out = Path('artifacts/centipawn-subgroups/bt4-soap-rem-c19000-final.batch16.f32.onnx')
onnx.save(model, out)
assert out.stat().st_size == 16246184
assert hashlib.sha256(out.read_bytes()).hexdigest() == 'c4adcf876904baf3381727720391ffba2fa2981e0395334e81af9424815b0506'
PY
export PATH="/opt/homebrew/bin:/opt/homebrew/opt/llvm/bin:$TVM_ENV/bin:$PATH"
export TVM_LIBRARY_PATH="$TVM_SRC/build-tvmjs/lib"
export DYLD_LIBRARY_PATH="$TVM_LIBRARY_PATH:/opt/homebrew/opt/llvm/lib:${DYLD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_SRC/python"
CONFIG='{"selector":{"kind":"static_large","min_macs":8000000,"j_multiple":32,"k_multiple":16},"config":{"micro_size_k":16}}'
STEM='bt4-soap-rem-c19000-final.batch16.f32.webgpu.tvmjs-wasm.probe'
for variant in control v3-subgroups; do
  mkdir -p "$OUT/$variant"
  target='{"kind":"webgpu","supports_subgroups":false}'
  [[ "$variant" != v3-subgroups ]] || target='{"kind":"webgpu","supports_subgroups":true}'
  "$TVM_ENV/bin/python" scripts/lc0_tvm_whole_onnx_probe.py \
    --model "$OUT/bt4-soap-rem-c19000-final.batch16.f32.onnx" \
    --out "$OUT/$variant/$STEM.json" --target "$target" \
    --host-target '{"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}' \
    --dtype '{"tokens":"int32"}' --cast-int64-initializers-to-int32 \
    --trust-nonnegative-gather-indices --trust-runtime-gather-indices \
    --sanitize-onnx-names --max-fuse-depth 6 --dlight \
    --dlight-matmul-config "$CONFIG" --export-tvmjs-wasm --capture-module-sources
 done
# Staging uses v2's exact JS/runtime wasm; no TVMJS runtime/toolchain rebuild.
node scripts/stage_lc0_tvmjs_webgpu_artifacts.mjs \
  --model-family=bt4-soap-rem-c19000-final --dtype=f32 --version=v3-subgroups --batches=16 \
  --artifacts="$OUT/v3-subgroups" --tvm-src="$TVM_SRC" \
  --tvmjs-bundle="$V2/tvmjs.bundle.js" --tvmjs-runtime-wasm="$V2/tvmjs_runtime.wasm" \
  --out=public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v3-subgroups
node --input-type=module - <<'JS'
import { readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const root = 'public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/';
const baseline = JSON.parse(readFileSync(`${root}v2-shape-k16/manifest.json`));
const path = `${root}v3-subgroups/manifest.json`;
const manifest = JSON.parse(readFileSync(path));
manifest.status = 'experimental';
manifest.requiredFeatures = ['webgpu', 'subgroups'];
manifest.subgroupSize = 32;
manifest.fallbackManifest = '../v2-shape-k16/manifest.json';
manifest.compilerProvenance = { ...baseline.compilerProvenance, webgpu: { ...baseline.compilerProvenance.webgpu, requiredFeatures: manifest.requiredFeatures } };
manifest.buildRecipe = 'scripts/build_centipawn_tvmjs_subgroups.sh';
manifest.sourceModelSha256 = 'de2dd68518f6e2ed162da5ee27c67a5a4795567d6e39972e3705824572b7660e';
manifest.fixedModelSha256 = 'c4adcf876904baf3381727720391ffba2fa2981e0395334e81af9424815b0506';
const probeFile = manifest.files.find(f => f.path.endsWith('.probe.json'));
const probePath = `${root}v3-subgroups/${probeFile.path}`;
const probe = JSON.parse(readFileSync(probePath));
// Captured sources are local diagnostics, not staged runtime resources.
if (probe.module_sources) {
  delete probe.module_sources.dir;
  for (const module of probe.module_sources.modules) delete module.path;
}
writeFileSync(probePath, `${JSON.stringify(probe, null, 2)}\n`);
const probeBytes = readFileSync(probePath);
probeFile.bytes = probeBytes.length;
probeFile.sha256 = createHash('sha256').update(probeBytes).digest('hex');
copyFileSync(`${root}v2-shape-k16/tvm-vector-size-storage-rewrite.patch`, `${root}v3-subgroups/tvm-vector-size-storage-rewrite.patch`);
manifest.files.push(baseline.files.find(f => f.path === 'tvm-vector-size-storage-rewrite.patch'));
writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
JS
python3 scripts/inspect_centipawn_tvmjs_subgroups.py
