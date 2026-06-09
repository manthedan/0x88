#!/usr/bin/env python3
"""Check whether the local TVM build can export TVMJS/WebGPU browser wasm.

This is a readiness check for the *TVMJS whole-model runtime* path. It is not
needed for the existing custom TVM-derived WGSL runtime path.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def bool_or_error(fn) -> dict[str, Any]:
    try:
        return {"ok": bool(fn())}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def main() -> int:
    default_root = Path(os.environ.get("LC0_BROWSER_ROOT", "/Users/macthedan/projects/lc0_browser"))
    tvm_src = Path(os.environ.get("TVM_SRC", str(default_root / ".deps/tvm-webgpu-src")))
    tvm_build_dir = os.environ.get("TVM_BUILD_DIR", "build")
    tvm_lib_dir = tvm_src / tvm_build_dir / "lib"
    os.environ.setdefault("TVM_LIBRARY_PATH", str(tvm_lib_dir))

    result: dict[str, Any] = {
        "schema": "lc0_browser.tvmjs_webgpu_toolchain_check.v1",
        "env": {
            "TVM_SRC": str(tvm_src),
            "TVM_BUILD_DIR": tvm_build_dir,
            "TVM_LIBRARY_PATH": os.environ.get("TVM_LIBRARY_PATH"),
            "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "emcc": shutil.which("emcc"),
        },
        "checks": {},
        "missing": [],
        "nextSteps": [],
    }

    try:
        import tvm  # type: ignore
    except Exception as exc:  # noqa: BLE001
        result["checks"]["import_tvm"] = {"ok": False, "error": repr(exc)}
        result["missing"].append("python TVM import")
        result["nextSteps"].append("Activate the durable TVM env and set PYTHONPATH to the TVM source python directory.")
        print(json.dumps(result, indent=2))
        return 2

    result["checks"]["import_tvm"] = {"ok": True, "version": getattr(tvm, "__version__", None)}
    result["checks"]["runtime_enabled_llvm"] = bool_or_error(lambda: tvm.runtime.enabled("llvm"))
    result["checks"]["runtime_enabled_webgpu"] = bool_or_error(lambda: tvm.runtime.enabled("webgpu"))
    result["checks"]["target_webgpu"] = bool_or_error(lambda: tvm.target.Target("webgpu").kind.name == "webgpu")

    try:
        from tvm.contrib import tvmjs  # type: ignore  # noqa: F401
        result["checks"]["import_tvmjs"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        result["checks"]["import_tvmjs"] = {"ok": False, "error": repr(exc)}

    bitcode_names = ["wasm_runtime.bc", "tvmjs_support.bc", "webgpu_runtime.bc"]
    bitcode: dict[str, Any] = {}
    for name in bitcode_names:
        try:
            paths = tvm.libinfo.find_lib_path(name, optional=True)
        except Exception as exc:  # noqa: BLE001
            paths = None
            bitcode[name] = {"ok": False, "error": repr(exc), "paths": []}
            continue
        bitcode[name] = {"ok": bool(paths), "paths": paths or []}
    result["checks"]["web_runtime_bitcode"] = bitcode

    web_dir = tvm_src / "web"
    result["checks"]["tvm_web_dir"] = {"ok": web_dir.is_dir(), "path": str(web_dir)}

    if not result["checks"]["runtime_enabled_llvm"].get("ok"):
        result["missing"].append("TVM LLVM codegen runtime (target.build.llvm)")
    if not result["env"]["emcc"]:
        result["missing"].append("Emscripten emcc on PATH")
    for name, check in bitcode.items():
        if not check.get("ok"):
            result["missing"].append(name)
    if not result["checks"]["import_tvmjs"].get("ok"):
        result["missing"].append("tvm.contrib.tvmjs import")

    if result["missing"]:
        result["ok"] = False
        result["nextSteps"] = [
            "Build or install a TVM variant with LLVM enabled so host target wasm32 can be generated.",
            "Install and activate Emscripten so emcc is on PATH.",
            "Build the TVM web runtime under $TVM_SRC/web so wasm_runtime.bc, tvmjs_support.bc, and webgpu_runtime.bc are discoverable.",
            "Then rerun the LC0 probe with --host-target '{\"kind\":\"llvm\",\"mtriple\":\"wasm32-unknown-unknown-wasm\"}' --export-tvmjs-wasm.",
        ]
        exit_code = 2
    else:
        result["ok"] = True
        result["nextSteps"] = ["Toolchain appears ready for a TVMJS/WebGPU export attempt."]
        exit_code = 0

    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
