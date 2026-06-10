# LC0 Post-upload Pack Release Ownership Audit

Status: implemented for retained policy/value head tensors; stable defaults unchanged.

## Scope

Runtime Configuration focus: opt-in LC0 custom WebGPU/hybrid runtime paths, especially `lc0-webgpu-research-b4` and `hybrid-wgsl-heads`.

Stable defaults are unchanged: ORT ONNX/WebGPU remains the browser default.

Audited paths include:

- `src/lc0/modelPack.ts`
- `src/lc0/wgslMatmulAddProbe.ts`
- `src/lc0/searchWorker.ts`
- `src/lc0/policyOnlyBrowser.ts`
- `src/lc0/arenaBrowser.ts`

## Finding

`loadLc0WebModelPack()` returns `Uint8Array.subarray()` tensor views into downloaded shard buffers. Retaining a small tensor view can retain the full underlying shard `ArrayBuffer` until the view is unreachable.

Most encoder tensors are consumed during `Lc0WebHybridRuntime.create()` to create GPU buffers and are not retained afterward. Initial input tensors are copied into `Float32Array`/GPU resources.

The retained JS-side risk was the policy/value head tensor set:

- `Lc0WebHybridRuntime` retains `headTensors`;
- WGSL physical batch slots are created lazily by `createWgslBatchSlot()`;
- those lazy slots need policy/value head tensor bytes to create additional WGSL head GPU buffers.

Therefore, broad cleanup such as `pack = null` or `headTensors = undefined` is unsafe or insufficient.

## Implemented strategy

The retained policy/value head tensor views are now copied into exact-sized `Uint8Array` buffers via `detachPolicyValueHeadTensors()` immediately after loading them from the pack.

This keeps lazy WGSL batch-slot growth safe while allowing larger shard `ArrayBuffer`s to become collectible after runtime creation once the local `pack` and encoder tensor views leave scope.

## Alternatives considered

1. **Preallocate fixed physical slots**: create all expected WGSL batch slots at initialization, then release head tensors. This is more invasive and must be tied to `leafBatchSize`.
2. **Freeze capacity after warmup**: add an explicit runtime state transition after first b4 warmup that disallows future slot growth and releases head tensors. This is risky for generic/evolving scripts.
3. **Copy retained head tensors**: clone the retained head tensor payloads into exact-sized buffers. This was selected because it is local, does not change search semantics, does not require a new UI default, and preserves lazy batch-slot behavior.

## Footprint attribution

- This is **Pack Footprint / JS-retention** work.
- It does not change explicit GPU-buffer **Execution Footprint** accounting.
- It does not change evaluator `cacheFootprint`.
- It does not change stable defaults or Runtime Configuration selection.

## Validation plan

Required for code changes:

- `npm run typecheck`
- `./autoresearch.checks.sh`
- `npm run lc0:browser-hybrid-drift -- --preset lc0-webgpu-research-b4 --limit 9 --baseline-mode serial`
- matched fixed-suite smoke/rerun where practical
- autoreview with project checks before merge
