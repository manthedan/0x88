# Upstream PR: tvm-ffi — make StructuralEqual functor consistent with StructuralHash on tensor content

Target repo: `apache/tvm-ffi`. Patch: `tvm-ffi-structural-equal-content.patch`
(local commit `afea100` in `.deps/tvm-webgpu-src/3rdparty/tvm-ffi`).

## Title

`[FIX] StructuralEqual functor must compare tensor content (matches StructuralHash)`

## Description (paste into PR body)

`ffi::StructuralEqual::operator()` calls `Equal(lhs, rhs, false, /*skip_tensor_content=*/true)`
while `ffi::StructuralHash::operator()` hashes tensor **content**
(`skip_tensor_content = false`). Used together as the (hash, equal) pair of an
unordered container this violates the container invariant: two same-shape/dtype
tensors with different data compare "equal" but hash differently.

The functor pair is used as the (Hash, KeyEqual) of multiple containers, e.g.
`relax::ExecBuilderNode::const_dedup_map_` (`include/tvm/relax/exec_builder.h`)
and several arith/relax maps, and `StructuralEqual` is also called directly in
passes. Standard-library maps that pre-compare stored hash codes before
calling `key_eq` (libc++ does) can mask the broken equality, which makes
minimal synthetic repros unreliable — but any equality path reached without a
matching-hash precondition merges distinct constants. Observed end-to-end
effects in a production-sized relax module (ONNX-imported transformer, ~200
bindings), all of which disappear with this one-line fix and reappear when it
is reverted (verified by rebuild-bisection):

- `clip(x, 0, 7)` compiled to `min(max(x, **7**), 7)` ≡ constant 7 — the
  const-0 operand deduplicated into the const-7 pool entry;
- downstream tensors corrupted the same way (any call_tir constant operand is
  at risk);
- because collisions depend on process-specific bucket layout, **byte-identical
  IRModules produce different wrong outputs across runs** (we also observed
  alternating clean/all-NaN builds when float constants collided) — this
  presented for weeks as two separate "compiler constant-folding" and
  "nondeterministic builds" bugs;
- small modules are unaffected (too few constants to collide), which makes the
  bug look context-dependent and defeats naive test-case minimization.

Fix: pass `skip_tensor_content=false` in the functor so the (hash, equal) pair
agrees. Dedup becomes strictly sound; the only behavior change is that
same-shape tensors with different data no longer merge (which was the bug).

## Standalone repro

`repro_constant_pool_dedup.py` (in this directory): builds
`y = x + c_0 + ... + c_511` with 512 distinct 0-d int64 relax constants (kept
unfoldable by the runtime input), compiles for llvm, and checks the sum.
Fails on unpatched builds; passes with the fix.

## Suggested unit test (tests/cpp, gtest)

```cpp
TEST(StructuralEqualHash, FunctorPairConsistentOnTensorContent) {
  auto a = tvm::ffi::Tensor::FromNDAlloc(..., /*0-d int64, value 0*/);
  auto b = tvm::ffi::Tensor::FromNDAlloc(..., /*0-d int64, value 7*/);
  tvm::ffi::StructuralEqual eq;
  tvm::ffi::StructuralHash hash;
  EXPECT_FALSE(eq(tvm::ffi::Any(a), tvm::ffi::Any(b)));
  // invariant: eq(x, y) implies hash(x) == hash(y)
}
```

## Submission steps (needs your GitHub auth; no gh CLI on this machine)

```bash
cd /Users/macthedan/projects/lc0_browser/.deps/tvm-webgpu-src/3rdparty/tvm-ffi
git checkout -b fix-structural-equal-tensor-content afea100
# fork apache/tvm-ffi on GitHub, then:
git remote add fork git@github.com:<you>/tvm-ffi.git
git push fork fix-structural-equal-tensor-content
# open PR against apache/tvm-ffi main with the body above
```

Note: contributions follow the repo's pre-commit lint (`clang-format`, Google
style, 100-col) — the patch is comment + one literal, should pass as-is.
