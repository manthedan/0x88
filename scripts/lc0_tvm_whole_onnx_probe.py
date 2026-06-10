#!/usr/bin/env python3
"""Probe whole-model ONNX -> TVM Relax -> target build feasibility.

This script is intentionally diagnostic and artifact-first.  It records every
stage in JSON so LC0 / Tiny Leela / future model TVM experiments are
reproducible instead of living in /tmp notebooks.

Expected durable environment from project root (/Users/macthedan/projects/lc0_browser):

  export TVM_SRC="$PWD/.deps/tvm-webgpu-src"
  export TVM_ENV="$PWD/.envs/tvm-mlc-py313"
  export TVM_LIBRARY_PATH="$TVM_SRC/build/lib"
  export DYLD_LIBRARY_PATH="$TVM_SRC/build/lib:${DYLD_LIBRARY_PATH:-}"
  export PYTHONPATH="$TVM_SRC/python:${PYTHONPATH:-}"
  "$TVM_ENV/bin/python" leelaweb/scripts/lc0_tvm_whole_onnx_probe.py \
    --model leelaweb/public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.onnx \
    --target webgpu \
    --out artifacts/tvm/lc0_batch1_f16_webgpu_probe.json

Do not add 3rdparty/tvm-ffi/python to PYTHONPATH for the current durable build;
use the installed apache-tvm-ffi wheel plus TVM build libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import onnx
from onnx import TensorProto, numpy_helper


def _elapsed() -> float:
    return time.perf_counter()


def _value_info_shape(value_info: Any) -> list[Any]:
    return [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def sanitize_onnx_value_names(model: Any) -> dict[str, Any]:
    """Replace ONNX value names with C/TVM-friendly identifiers.

    TVM's current Relax frontend only partially sanitizes names like
    /input/planes, which can later leak into generated C wrappers as invalid
    identifiers.  This pass rewrites graph edge names consistently before TVM
    import.  It does not change op semantics.
    """
    names: list[str] = []
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        names.append(value.name)
    for initializer in model.graph.initializer:
        names.append(initializer.name)
    for node in model.graph.node:
        names.extend(name for name in node.input if name)
        names.extend(name for name in node.output if name)

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        if not name or name in mapping:
            continue
        sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_")
        if not sanitized:
            sanitized = "value"
        if sanitized[0].isdigit():
            sanitized = f"v_{sanitized}"
        base = sanitized
        suffix = 1
        while sanitized in used and mapping.get(name) != sanitized:
            suffix += 1
            sanitized = f"{base}_{suffix}"
        mapping[name] = sanitized
        used.add(sanitized)

    def rename(name: str) -> str:
        return mapping.get(name, name)

    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        value.name = rename(value.name)
    for initializer in model.graph.initializer:
        initializer.name = rename(initializer.name)
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name:
                node.input[idx] = rename(name)
        for idx, name in enumerate(node.output):
            if name:
                node.output[idx] = rename(name)
        if node.name:
            node.name = rename(node.name)

    changed = {old: new for old, new in mapping.items() if old != new}
    return {"changed_count": len(changed), "changed_head": dict(list(changed.items())[:200])}


def install_nonnegative_gather_indices_patch(trust_runtime_indices: bool = False) -> dict[str, Any]:
    """Patch TVM's ONNX Gather converter for nonnegative indices.

    Upstream TVM currently emits signed-index wrap support for every signed
    Gather indices tensor.  For LC0 policy mapping, the indices initializer is
    constant and strictly nonnegative, so the wrap branch is dead but still
    introduces shape_to_tensor/take(int64), which WebGPU codegen rejects.

    trust_runtime_indices extends the same trust to non-constant indices.
    This is model-specific: squareformer compact-token exports clamp token
    indices into the embedding range in-graph (max/min) before every Gather,
    so the wrap branch is dead there too.
    """
    from tvm import relax  # type: ignore
    import tvm.relax.frontend.onnx.onnx_frontend as onnx_frontend  # type: ignore

    if getattr(onnx_frontend.Gather, "_lc0_nonnegative_gather_patch", False):
        return {"installed": False, "reason": "already_installed"}

    original = onnx_frontend.Gather._impl_v13.__func__
    stats = {"constant_nonnegative_gathers": 0, "trusted_runtime_gathers": 0}

    def patched(cls: Any, bb: Any, inputs: list[Any], attr: dict[str, Any], params: dict[str, Any]) -> Any:
        data = inputs[0]
        indices = inputs[1]
        axis = attr.get("axis", 0)
        # relax.take requires a Tensor data argument. Shape->Gather dim
        # extraction (common in PyTorch exports) passes a Shape expr here, so
        # leave anything non-tensor to the original converter.
        data_is_tensor = isinstance(getattr(data, "struct_info", None), relax.TensorStructInfo)
        if data_is_tensor and isinstance(indices, relax.Constant):
            array = indices.data.numpy()
            if array.size == 0 or array.min() >= 0:
                stats["constant_nonnegative_gathers"] += 1
                return relax.op.take(data, indices, axis)
        elif data_is_tensor and trust_runtime_indices:
            stats["trusted_runtime_gathers"] += 1
            return relax.op.take(data, indices, axis)
        return original(cls, bb, inputs, attr, params)

    onnx_frontend.Gather._impl_v13 = classmethod(patched)
    onnx_frontend.Gather._lc0_nonnegative_gather_patch = True
    onnx_frontend.Gather._lc0_nonnegative_gather_patch_stats = stats
    return {"installed": True, "stats": stats}


def get_nonnegative_gather_patch_stats() -> dict[str, Any] | None:
    try:
        import tvm.relax.frontend.onnx.onnx_frontend as onnx_frontend  # type: ignore

        return getattr(onnx_frontend.Gather, "_lc0_nonnegative_gather_patch_stats", None)
    except Exception:  # noqa: BLE001
        return None


def cast_int64_initializers_to_int32(model: Any) -> dict[str, Any]:
    """Downcast int64 ONNX initializers when values fit int32.

    LC0 exports Reshape/Split shape tensors as int64.  TVM's WebGPU codegen
    rejects i64, so this opt-in diagnostic mutation tests whether shape/index
    constants are the only i64 blocker.  It deliberately does not change graph
    value_info or floating weights.
    """
    converted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.data_type != TensorProto.INT64:
            continue
        array = numpy_helper.to_array(initializer)
        if array.size and (array.min() < -(2**31) or array.max() > 2**31 - 1):
            skipped.append({"name": initializer.name, "shape": list(array.shape), "min": int(array.min()), "max": int(array.max())})
            continue
        replacement = numpy_helper.from_array(array.astype("int32"), name=initializer.name)
        model.graph.initializer[index].CopyFrom(replacement)
        converted.append({"name": initializer.name, "shape": list(array.shape), "values_head": array.flatten()[:16].astype(int).tolist()})
    return {"converted_count": len(converted), "skipped_count": len(skipped), "converted": converted, "skipped": skipped}


def summarize_onnx(model: Any) -> dict[str, Any]:
    return {
        "ir_version": model.ir_version,
        "opsets": [{"domain": opset.domain, "version": opset.version} for opset in model.opset_import],
        "inputs": [
            {
                "name": value.name,
                "elem_type": value.type.tensor_type.elem_type,
                "shape": _value_info_shape(value),
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "elem_type": value.type.tensor_type.elem_type,
                "shape": _value_info_shape(value),
            }
            for value in model.graph.output
        ],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": dict(Counter(node.op_type for node in model.graph.node).most_common()),
    }


def run_step(result: dict[str, Any], name: str, fn: Callable[[], Any]) -> Any | None:
    start = _elapsed()
    try:
        value = fn()
        result["steps"][name] = {"ok": True, "seconds": _elapsed() - start}
        return value
    except Exception as exc:  # noqa: BLE001 - diagnostic script should capture all blockers
        result["steps"][name] = {
            "ok": False,
            "seconds": _elapsed() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-60:],
        }
        return None


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def iter_runtime_modules(module: Any) -> list[tuple[int, Any]]:
    seen: set[int] = set()
    out: list[tuple[int, Any]] = []

    def visit(current: Any, depth: int) -> None:
        ident = id(current)
        if ident in seen:
            return
        seen.add(ident)
        out.append((depth, current))
        for attr in ["imports", "imports_"]:
            try:
                children = getattr(current, attr)
                children = children() if callable(children) else children
            except Exception:
                continue
            for child in list(children or []):
                visit(child, depth + 1)

    visit(module, 0)
    return out


def capture_module_sources(executable: Any, out_path: Path) -> dict[str, Any]:
    source_dir = out_path.with_suffix(".sources")
    source_dir.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, (depth, module) in enumerate(iter_runtime_modules(executable.mod)):
        kind = str(getattr(module, "kind", "unknown"))
        for fmt in (["wgsl"] if kind == "webgpu" else ["c"] if kind == "c" else [""]):
            try:
                source = module.inspect_source(fmt) if fmt else module.inspect_source()
            except Exception as exc:  # noqa: BLE001
                captured.append({"module_index": index, "depth": depth, "kind": kind, "format": fmt, "ok": False, "error": repr(exc)})
                continue
            if not source:
                continue
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            duplicate = digest in seen_hashes
            seen_hashes.add(digest)
            ext = "wgsl" if fmt == "wgsl" or kind == "webgpu" else "c" if kind == "c" else "txt"
            source_path = source_dir / f"module{index:02d}.depth{depth}.{kind}.{ext}"
            if not duplicate:
                source_path.write_text(source, encoding="utf-8")
            captured.append(
                {
                    "module_index": index,
                    "depth": depth,
                    "kind": kind,
                    "format": fmt,
                    "ok": True,
                    "duplicate": duplicate,
                    "sha256": digest,
                    "bytes": len(source.encode("utf-8")),
                    "path": str(source_path) if not duplicate else None,
                }
            )
    return {"dir": str(source_dir), "modules": captured}


def parse_target_spec(tvm: Any, spec: str) -> Any:
    text = spec.strip()
    if text.startswith("{"):
        return tvm.target.Target(json.loads(text))
    return tvm.target.Target(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--out", required=True, help="JSON artifact path")
    parser.add_argument("--target", default="webgpu", help="TVM target, default webgpu")
    parser.add_argument(
        "--host-target",
        default=None,
        help='Optional TVM host target. Use JSON dict form for this TVM build, e.g. {"kind":"llvm","mtriple":"wasm32-unknown-unknown-wasm"}',
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="dtype_dict passed to Relax from_onnx; omit to preserve model dtypes",
    )
    parser.add_argument(
        "--cast-int64-initializers-to-int32",
        action="store_true",
        help="Opt-in diagnostic: downcast int64 ONNX initializer constants that fit int32 before TVM import",
    )
    parser.add_argument(
        "--trust-nonnegative-gather-indices",
        action="store_true",
        help="Opt-in diagnostic: skip TVM ONNX Gather negative-index wrap when indices are constant and nonnegative",
    )
    parser.add_argument(
        "--sanitize-onnx-names",
        action="store_true",
        help="Opt-in diagnostic: rewrite ONNX value/node names to C/TVM-friendly identifiers before import",
    )
    parser.add_argument(
        "--capture-module-sources",
        action="store_true",
        help="Write generated runtime module sources such as WGSL/C next to the JSON artifact",
    )
    parser.add_argument(
        "--export-tvmjs-wasm",
        action="store_true",
        help="Also try export_library(..., fcompile=tvmjs.create_tvmjs_wasm) for browser TVMJS runtime loading",
    )
    parser.add_argument(
        "--dlight",
        action="store_true",
        help="Opt-in: apply dlight GPU schedule rules (Matmul/GEMV/Reduction/GeneralReduction/Fallback) instead of TVM's naive DefaultGPUSchedule thread binding",
    )
    parser.add_argument(
        "--no-fuse-ops",
        action="store_true",
        help="Skip relax FuseOps/FuseTIR (use per-op kernels). Works around WebGPU per-stage storage-buffer limits for fused kernels with many inputs; relax FuseOps currently ignores target max_function_args",
    )
    parser.add_argument(
        "--max-fuse-depth",
        type=int,
        default=None,
        help="Set PassContext config relax.FuseOps.max_depth to cap fused-group size; bounds storage buffers per WebGPU kernel (FuseOps ignores target max_function_args)",
    )
    parser.add_argument(
        "--trust-runtime-gather-indices",
        action="store_true",
        help="Opt-in diagnostic: also skip Gather negative-index wrap for runtime indices; only valid for models that clamp indices in-graph (e.g. squareformer compact tokens)",
    )
    parser.add_argument(
        "--fix-batch-dim",
        type=int,
        default=None,
        help="Rewrite symbolic batch dims (dim_param) on graph inputs/outputs to this fixed value before import; WebGPU codegen needs static shapes",
    )
    parser.add_argument(
        "--dlight-matmul-config",
        default=None,
        help='JSON dict of dlight gpu.Matmul.Config overrides for the webgpu default, e.g. {"vector_size":4,"micro_size_k":16}',
    )
    parser.add_argument(
        "--detach-params",
        action="store_true",
        help="Opt-in: import with keep_params_in_input, detach weights from the module, and dump them as a tensor-cache (param_0..param_N, raw encoding) so the exported wasm contains no embedded weights",
    )
    parser.add_argument(
        "--tensor-cache-dir",
        default=None,
        help="Tensor-cache output directory for --detach-params; default <out>.tensor-cache",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    out_path = Path(args.out).resolve()
    result: dict[str, Any] = {
        "schema": "lc0_tvm_whole_onnx_probe_v1",
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size if model_path.exists() else None,
        "target": args.target,
        "host_target": args.host_target,
        "dtype_arg": args.dtype,
        "cast_int64_initializers_to_int32": args.cast_int64_initializers_to_int32,
        "trust_nonnegative_gather_indices": args.trust_nonnegative_gather_indices,
        "sanitize_onnx_names": args.sanitize_onnx_names,
        "capture_module_sources": args.capture_module_sources,
        "export_tvmjs_wasm": args.export_tvmjs_wasm,
        "dlight": args.dlight,
        "detach_params": args.detach_params,
        "env": {
            "python": sys.version,
            "executable": sys.executable,
            "TVM_LIBRARY_PATH": os.environ.get("TVM_LIBRARY_PATH"),
            "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "emcc": shutil.which("emcc"),
        },
        "steps": {},
    }

    def import_tvm() -> Any:
        import tvm  # type: ignore

        return tvm

    tvm = run_step(result, "import_tvm", import_tvm)
    if tvm is not None:
        result["tvm"] = {
            "version": getattr(tvm, "__version__", None),
            "file": getattr(tvm, "__file__", None),
            "lib": str(getattr(tvm.base, "_LIB", None)),
        }
        try:
            info = tvm.support.libinfo()
            result["tvm"]["libinfo"] = {key: str(value) for key, value in sorted(info.items())}
        except Exception as exc:  # noqa: BLE001
            result["tvm"]["libinfo_error"] = repr(exc)
        for runtime_name in ["llvm", "webgpu", "rpc"]:
            try:
                result["tvm"][f"runtime_enabled_{runtime_name}"] = bool(tvm.runtime.enabled(runtime_name))
            except Exception as exc:  # noqa: BLE001
                result["tvm"][f"runtime_enabled_{runtime_name}_error"] = repr(exc)
        try:
            result["tvm"]["target_kinds"] = [str(kind) for kind in tvm.target.Target.list_kinds()]
            result["tvm"]["has_webgpu_target"] = "webgpu" in result["tvm"]["target_kinds"]
        except Exception as exc:  # noqa: BLE001
            result["tvm"]["target_list_error"] = repr(exc)

    model = run_step(result, "onnx_load", lambda: onnx.load(str(model_path), load_external_data=False))
    if model is not None:
        result["onnx_before_mutation"] = summarize_onnx(model)
        if args.fix_batch_dim is not None:
            fixed = 0
            for value_info in list(model.graph.input) + list(model.graph.output):
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        dim.ClearField("dim_param")
                        dim.dim_value = args.fix_batch_dim
                        fixed += 1
            result["onnx_fixed_batch_dims"] = {"value": args.fix_batch_dim, "dims_rewritten": fixed}
        if args.cast_int64_initializers_to_int32:
            result["onnx_mutation"] = cast_int64_initializers_to_int32(model)
        if args.sanitize_onnx_names:
            result["onnx_name_sanitization"] = sanitize_onnx_value_names(model)
        result["onnx"] = summarize_onnx(model)

    detached_main_params: list[Any] = []

    def install_small_initializer_constant_patch(max_elements: int = 4096) -> dict[str, Any]:
        """Keep small ONNX initializers as inline relax constants under
        keep_params_in_input.

        Shape-feeding tensors (Reshape dims, Slice starts, the constant LC0
        policy-mapping Gather indices) must keep constant *values* for shape
        inference and the nonnegative-Gather patch; only large weight tensors
        become detachable function params.
        """
        from tvm import relax  # type: ignore
        from tvm.relax.frontend.onnx import onnx_frontend  # type: ignore

        stats: dict[str, Any] = {"max_elements": max_elements, "params": 0, "inline_constants": 0}

        import math

        def patched(self: Any, graph: Any) -> None:
            for init_tensor in graph.initializer:
                if not init_tensor.name.strip():
                    raise ValueError("Tensor's name is required.")
                array = self._parse_array(init_tensor)
                if self._keep_params_in_input and math.prod(array.shape) > max_elements:
                    var_name = init_tensor.name.strip("onnx::")
                    init_var = self._new_var(var_name, shape=array.shape, dtype=array.dtype)
                    self._nodes[init_tensor.name] = init_var
                    self._params[var_name] = (init_var, array)
                    stats["params"] += 1
                else:
                    self._nodes[init_tensor.name] = relax.const(array)
                    stats["inline_constants"] += 1

        onnx_frontend.ONNXGraphImporter._parse_graph_initializers = patched
        return stats

    def import_relax_onnx() -> Any:
        if args.trust_nonnegative_gather_indices or args.trust_runtime_gather_indices:
            result["onnx_frontend_patch"] = install_nonnegative_gather_indices_patch(trust_runtime_indices=args.trust_runtime_gather_indices)
        if args.detach_params:
            result["small_initializer_patch"] = install_small_initializer_constant_patch()
        from tvm.relax.frontend.onnx import from_onnx  # type: ignore

        dtype_arg = args.dtype if args.dtype else None
        # JSON dict form maps individual input names to dtypes, e.g.
        # '{"tokens":"int32"}' for integer-token models where only one input
        # must be downcast away from int64 for WebGPU codegen.
        if isinstance(dtype_arg, str) and dtype_arg.strip().startswith("{"):
            dtype_arg = json.loads(dtype_arg)
        # This source build's signature advertises GraphProto, but the implementation
        # reads ModelProto fields such as ir_version/opset_import.
        imported = from_onnx(model, dtype_dict=dtype_arg, keep_params_in_input=args.detach_params)
        if args.detach_params:
            from tvm.relax.frontend import detach_params  # type: ignore

            imported, params_dict = detach_params(imported)
            detached_main_params.extend(params_dict.get("main", []))
            import math

            dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8, "int8": 1, "uint8": 1}
            result["detached_params"] = {
                "count": len(detached_main_params),
                "total_bytes": sum(math.prod(p.shape) * dtype_bytes.get(str(p.dtype), 4) for p in detached_main_params),
            }
        return imported

    mod = run_step(result, "relax_from_onnx", import_relax_onnx) if tvm is not None and model is not None else None
    if mod is not None:
        if args.trust_nonnegative_gather_indices:
            result["onnx_frontend_patch_stats"] = get_nonnegative_gather_patch_stats()
        result["relax_module"] = {}
        try:
            functions = list(mod.functions.keys())
            result["relax_module"].update(
                {"function_count": len(functions), "functions": [str(function) for function in functions[:40]]}
            )
        except Exception as exc:  # noqa: BLE001
            result["relax_module"]["functions_error"] = repr(exc)

        def capture_script() -> bool:
            script = mod.script()
            result["relax_module"].update(
                {
                    "script_chars": len(script),
                    "script_head": script[:12000],
                    "int64_token_count": script.count("int64"),
                    "shape_to_tensor_token_count": script.count("shape_to_tensor"),
                }
            )
            return True

        run_step(result, "relax_module_script", capture_script)

    def build_relax() -> Any:
        from tvm import relax  # type: ignore
        import contextlib

        target = tvm.target.Target(parse_target_spec(tvm, args.target), host=parse_target_spec(tvm, args.host_target)) if args.host_target else parse_target_spec(tvm, args.target)
        result["build_target"] = str(target)
        fuse_depth_context = (
            tvm.transform.PassContext(config={"relax.FuseOps.max_depth": args.max_fuse_depth})
            if args.max_fuse_depth is not None
            else contextlib.nullcontext()
        )
        if args.max_fuse_depth is not None:
            result["max_fuse_depth"] = args.max_fuse_depth
        with fuse_depth_context:
            return build_relax_inner(relax, target)

    def build_relax_inner(relax: Any, target: Any) -> Any:
        build_mod = mod
        if args.no_fuse_ops:
            with tvm.target.Target(target):
                build_mod = tvm.ir.transform.Sequential([
                    relax.transform.LegalizeOps(),
                    relax.transform.AnnotateTIROpPattern(),
                    relax.transform.FoldConstant(),
                ])(mod)
            result["fuse_ops_skipped"] = True
        if args.dlight:
            try:
                from tvm import dlight as dl  # type: ignore
            except ImportError:
                # This TVM checkout ships dlight under s_tir.
                from tvm.s_tir import dlight as dl  # type: ignore

            # Record which rule actually scheduled each TIR function so a weak
            # dlight result can be attributed: the Matmul rule not firing is a
            # very different problem from Matmul tile sizes being wrong.
            rule_attribution: dict[str, str] = {}

            def function_name(func: Any, fallback: str = "?") -> str:
                # These PrimFuncs carry no global_symbol attr after the zero
                # pipeline; identify them structurally via the module instead.
                attr = func.attrs.get("global_symbol") if func.attrs is not None else None
                if attr is not None:
                    return str(attr)
                for gv, candidate in build_mod.functions_items():
                    if candidate.same_as(func):
                        return gv.name_hint
                return fallback

            rule_failures: dict[str, str] = {}
            crashed_functions: set[str] = set()

            class _RecordingRule:
                def __init__(self, rule: Any) -> None:
                    self._rule = rule
                    self._name = type(rule).__name__

                def apply(self, func: Any, rule_target: Any, tunable: bool) -> Any:
                    # A rule that crashes mid-schedule (e.g. rfactor bind on an
                    # awkward reduction) should fall through — but straight to
                    # Fallback (serial reduction, simple binding) rather than to
                    # sibling reduction rules that share the same assumptions
                    # and may emit subtly wrong cross-thread schedules.
                    name = function_name(func)
                    if name in crashed_functions and self._name != "Fallback":
                        return None
                    try:
                        space = self._rule.apply(func, rule_target, tunable)
                    except Exception as exc:  # noqa: BLE001
                        rule_failures[f"{name}:{self._name}"] = repr(exc)[:200]
                        crashed_functions.add(name)
                        return None
                    if space is not None:
                        rule_attribution[name] = self._name
                    return space

            # Lower relax ops to TIR first, then let dlight schedule the TIR
            # functions; relax.build's default pipeline skips already-scheduled
            # functions in DefaultGPUSchedule.
            with tvm.target.Target(target):
                if not args.no_fuse_ops:
                    build_mod = relax.get_pipeline("zero")(mod)
                prim_func_names = [
                    gv.name_hint
                    for gv, func in build_mod.functions_items()
                    if type(func).__name__ == "PrimFunc"
                ]
                matmul_rule = dl.gpu.Matmul()
                if args.dlight_matmul_config:
                    overrides = json.loads(args.dlight_matmul_config)
                    base_config = matmul_rule.get_configs(tvm.target.Target.current())
                    for key, value in overrides.items():
                        if not hasattr(base_config, key):
                            raise ValueError(f"Unknown dlight Matmul.Config field: {key}")
                        setattr(base_config, key, value)
                    matmul_rule.get_configs = lambda _target, _config=base_config: _config
                    result["dlight_matmul_config"] = {key: getattr(base_config, key) for key in vars(base_config)}
                build_mod = dl.ApplyDefaultSchedule(
                    _RecordingRule(matmul_rule),
                    _RecordingRule(dl.gpu.GEMV()),
                    _RecordingRule(dl.gpu.Reduction()),
                    _RecordingRule(dl.gpu.GeneralReduction()),
                    _RecordingRule(dl.gpu.Fallback()),
                )(build_mod)
            result["dlight_applied"] = True
            result["dlight_rule_attribution"] = {
                "counts": dict(Counter(rule_attribution.values())),
                "unscheduled": sorted(set(prim_func_names) - set(rule_attribution)),
                "functions": dict(sorted(rule_attribution.items())),
                **({"ruleFailures": rule_failures} if rule_failures else {}),
            }
        # Always build with the "default" pipeline: it contains the mandatory
        # VM-lowering passes (ToNonDataflow/CallTIRRewrite/...) and NO FuseOps,
        # so it preserves both the no-fuse intent and pre-applied dlight
        # schedules. relax_pipeline=None skips lowering entirely and fails
        # VMCodeGen with raw relax.call_tir.
        return relax.build(build_mod, target=target, relax_pipeline="default")

    executable = run_step(result, "relax_build_target", build_relax) if mod is not None else None
    if executable is not None:
        result["build"] = {"type": str(type(executable))}
        for attr in ["mod", "module", "executable"]:
            try:
                result["build"][f"{attr}_type"] = str(type(getattr(executable, attr)))
            except Exception:
                pass

        if args.capture_module_sources:
            run_step(result, "capture_module_sources", lambda: result.setdefault("module_sources", capture_module_sources(executable, out_path)))

        def host_target_is_wasm() -> bool:
            return bool(args.host_target and "wasm32" in args.host_target)

        def export_library() -> Path:
            lib_path = out_path.with_suffix(".tvm_export")
            executable.export_library(str(lib_path))
            result["build"].update({"export_library": str(lib_path), "export_library_bytes": lib_path.stat().st_size})
            return lib_path

        if args.export_tvmjs_wasm and host_target_is_wasm():
            result["steps"]["export_library"] = {
                "ok": True,
                "skipped": True,
                "reason": "TVMJS wasm export uses wasm32 host objects; native shared-library export would try to link them with the platform C++ linker.",
            }
        else:
            run_step(result, "export_library", export_library)

        if args.export_tvmjs_wasm:
            def export_tvmjs_wasm() -> Path:
                from tvm.contrib import tvmjs  # type: ignore

                wasm_path = out_path.with_suffix(".tvmjs.wasm")
                executable.export_library(str(wasm_path), fcompile=tvmjs.create_tvmjs_wasm)
                result["build"].update({"tvmjs_wasm": str(wasm_path), "tvmjs_wasm_bytes": wasm_path.stat().st_size})
                return wasm_path

            run_step(result, "export_tvmjs_wasm", export_tvmjs_wasm)

        if args.detach_params:
            def dump_tensor_cache_step() -> str:
                from tvm.contrib import tvmjs as tvmjs_contrib  # type: ignore

                if not detached_main_params:
                    raise RuntimeError("detach-params produced no main params")
                cache_dir = args.tensor_cache_dir or f"{out_path.with_suffix('')}.tensor-cache"
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                tvmjs_contrib.dump_tensor_cache(
                    {f"param_{i}": p for i, p in enumerate(detached_main_params)},
                    str(cache_dir),
                    encode_format="raw",
                    show_progress=False,
                )
                result["tensor_cache"] = {
                    "dir": str(cache_dir),
                    "param_count": len(detached_main_params),
                    "bytes": sum(f.stat().st_size for f in Path(cache_dir).iterdir() if f.is_file()),
                }
                return str(cache_dir)

            run_step(result, "dump_tensor_cache", dump_tensor_cache_step)

    def host_target_is_wasm_requested() -> bool:
        return bool(args.host_target and "wasm32" in args.host_target)

    required_steps = ["import_tvm", "onnx_load", "relax_from_onnx", "relax_build_target"]
    if args.capture_module_sources:
        required_steps.append("capture_module_sources")
    if not (args.export_tvmjs_wasm and host_target_is_wasm_requested()):
        required_steps.append("export_library")
    if args.export_tvmjs_wasm:
        required_steps.append("export_tvmjs_wasm")
    if args.detach_params:
        required_steps.append("dump_tensor_cache")

    required_failures = [
        step for step in required_steps
        if result["steps"].get(step, {}).get("ok") is not True
    ]
    result["required_steps"] = required_steps
    result["required_step_failures"] = required_failures
    result["ok"] = not required_failures

    write_result(out_path, result)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "model": str(model_path),
                "target": args.target,
                "steps": result["steps"],
                "required_steps": required_steps,
                "required_step_failures": required_failures,
                "ok": result["ok"],
                "has_webgpu_target": result.get("tvm", {}).get("has_webgpu_target"),
            },
            indent=2,
        )
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
