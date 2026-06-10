# Upstream PR: tvm — pad unaligned WebGPU GPU→CPU readback copies

Target repo: `apache/tvm`. Patch: `tvm-webgpu-unaligned-readback.patch`
(local commit `044cbd0d4` in `.deps/tvm-webgpu-src`).

## Title

`[WEB] Pad unaligned GPU-to-CPU readback copies to 4 bytes`

## Description (paste into PR body)

WebGPU requires `copyBufferToBuffer` sizes and `getMappedRange` sizes to be
multiples of 4 bytes. `WebGPUContext.deviceCopyFromGPU` in `web/src/webgpu.ts`
passes the tensor's `nbytes` directly, so any tensor whose byte size is not a
multiple of 4 cannot be read back — e.g. a 3-element f16 head (6 bytes) or a
1-element f16 scalar (2 bytes). For an LC0 chess network this made the WDL and
moves-left outputs unreadable at batch 1.

Fix: round the copy size up to a multiple of 4, clamped to the source buffer's
actual `size` (GPUBuffer exposes it), map the padded range, and store only the
requested `nbytes` back into wasm memory. When padding cannot fit inside the
source buffer, throw the previously implicit failure explicitly with a useful
message.

Validated by reading back `[1,3]` f16 (6B) and `[1,1]` f16 (2B) outputs of a
relax VM WebGPU module in Chrome (previously failed validation); large aligned
tensors unaffected.

## Submission steps (needs your GitHub auth; no gh CLI on this machine)

```bash
cd /Users/macthedan/projects/lc0_browser/.deps/tvm-webgpu-src
git checkout -b fix-webgpu-unaligned-readback 044cbd0d4
# fork apache/tvm on GitHub, then:
git remote add fork git@github.com:<you>/tvm.git
git push fork fix-webgpu-unaligned-readback
# open PR against apache/tvm main with the body above
```

Caveat for the PR: rebase onto current apache/tvm main first — this checkout
is at 15b1d9839 plus local commits; web/src/webgpu.ts may have moved.
