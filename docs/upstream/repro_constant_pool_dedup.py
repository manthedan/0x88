"""Repro: relax VM constant-pool dedup merges distinct tensor constants.

ffi::StructuralEqual::operator() compares with skip_tensor_content=true while
ffi::StructuralHash::operator() hashes tensor content. Used together as the
(hash, equal) pair of ExecBuilderNode::const_dedup_map_, this violates the
unordered_map invariant: same-shape/dtype tensors with different data are
"equal" but hash differently. Whenever two such keys share a bucket, the map
"finds" the wrong constant and the emitted bytecode references the wrong pool
slot. The miscompile therefore only appears once the constant pool is large
enough for bucket collisions, and which constants collide depends on
process-specific layout — i.e. byte-identical modules can produce different
wrong outputs across runs.

This script builds y = x + c_0 + c_1 + ... + c_{N-1} with N distinct 0-d
int64 constants (kept out of FoldConstant's reach by the runtime input) and
checks the result.

CAVEAT: standard-library maps that pre-compare stored hash codes before
calling key_eq (libc++ does) mask the broken equality here, so this synthetic
check can pass on affected builds. It documents the invariant; the reliable
end-to-end repro is an ONNX-imported module with multiple same-shape scalar
constants feeding call_tir (see the PR description), which fails on affected
builds through an equality path without a hash precondition.
"""
import numpy as np
import tvm
from tvm import relax
from tvm.relax import op as R

N = 512
bb = relax.BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((), "int64"))
with bb.function("main", [x]):
    with bb.dataflow():
        acc = x
        for i in range(N):
            const = relax.const(i, "int64")
            acc = bb.emit(R.add(acc, const))
        out = bb.emit_output(acc)
    bb.emit_func_output(out)
mod = bb.get()

ex = tvm.relax.build(mod, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
result = int(vm["main"](tvm.runtime.tensor(np.array(1000, dtype="int64"))).numpy())
expected = 1000 + sum(range(N))
print(f"result={result} expected={expected}")
assert result == expected, (
    f"constant pool corruption: got {result}, expected {expected} "
    f"(delta {result - expected})"
)
print("OK: constant pool is sound")
