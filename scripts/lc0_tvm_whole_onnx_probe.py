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


def install_nonnegative_gather_indices_patch() -> dict[str, Any]:
    """Patch TVM's ONNX Gather converter for constant nonnegative indices.

    Upstream TVM currently emits signed-index wrap support for every signed
    Gather indices tensor.  For LC0 policy mapping, the indices initializer is
    constant and strictly nonnegative, so the wrap branch is dead but still
    introduces shape_to_tensor/take(int64), which WebGPU codegen rejects.
    """
    from tvm import relax  # type: ignore
    import tvm.relax.frontend.onnx.onnx_frontend as onnx_frontend  # type: ignore

    if getattr(onnx_frontend.Gather, "_lc0_nonnegative_gather_patch", False):
        return {"installed": False, "reason": "already_installed"}

    original = onnx_frontend.Gather._impl_v13.__func__
    stats = {"constant_nonnegative_gathers": 0}

    def patched(cls: Any, bb: Any, inputs: list[Any], attr: dict[str, Any], params: dict[str, Any]) -> Any:
        data = inputs[0]
        indices = inputs[1]
        axis = attr.get("axis", 0)
        if isinstance(indices, relax.Constant):
            array = indices.data.numpy()
            if array.size == 0 or array.min() >= 0:
                stats["constant_nonnegative_gathers"] += 1
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
        if args.cast_int64_initializers_to_int32:
            result["onnx_mutation"] = cast_int64_initializers_to_int32(model)
        if args.sanitize_onnx_names:
            result["onnx_name_sanitization"] = sanitize_onnx_value_names(model)
        result["onnx"] = summarize_onnx(model)

    def import_relax_onnx() -> Any:
        if args.trust_nonnegative_gather_indices:
            result["onnx_frontend_patch"] = install_nonnegative_gather_indices_patch()
        from tvm.relax.frontend.onnx import from_onnx  # type: ignore

        dtype_arg = args.dtype if args.dtype else None
        # This source build's signature advertises GraphProto, but the implementation
        # reads ModelProto fields such as ir_version/opset_import.
        return from_onnx(model, dtype_dict=dtype_arg)

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

        target = tvm.target.Target(parse_target_spec(tvm, args.target), host=parse_target_spec(tvm, args.host_target)) if args.host_target else parse_target_spec(tvm, args.target)
        result["build_target"] = str(target)
        build_mod = mod
        if args.dlight:
            try:
                from tvm import dlight as dl  # type: ignore
            except ImportError:
                # This TVM checkout ships dlight under s_tir.
                from tvm.s_tir import dlight as dl  # type: ignore

            # Lower relax ops to TIR first, then let dlight schedule the TIR
            # functions; relax.build's default pipeline skips already-scheduled
            # functions in DefaultGPUSchedule.
            with tvm.target.Target(target):
                build_mod = relax.get_pipeline("zero")(mod)
                build_mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                )(build_mod)
            result["dlight_applied"] = True
        return relax.build(build_mod, target=target)

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

    def host_target_is_wasm_requested() -> bool:
        return bool(args.host_target and "wasm32" in args.host_target)

    required_steps = ["import_tvm", "onnx_load", "relax_from_onnx", "relax_build_target"]
    if args.capture_module_sources:
        required_steps.append("capture_module_sources")
    if not (args.export_tvmjs_wasm and host_target_is_wasm_requested()):
        required_steps.append("export_library")
    if args.export_tvmjs_wasm:
        required_steps.append("export_tvmjs_wasm")

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
