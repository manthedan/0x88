from __future__ import annotations

import builtins
import json
import re
import time
from pathlib import Path
from typing import Any

_METRIC_RE = re.compile(r"^METRIC\s+([^=\s]+)=(.+?)\s*$")
_ORIGINAL_PRINT = builtins.print


def _coerce(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    raw = str(value)
    try:
        f = float(raw)
        return int(f) if f.is_integer() else f
    except ValueError:
        return raw


class MetricEmitter:
    """Emit legacy `METRIC key=value` lines plus optional JSONL rows."""

    def __init__(self, jsonl_path: str | Path | None = None, *, run_id: str | None = None):
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.run_id = run_id
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, key: str, value: Any, **extra: Any) -> None:
        _ORIGINAL_PRINT(f"METRIC {key}={value}", flush=True)
        self.write_jsonl(key, _coerce(value), **extra)

    def emit_many(self, prefix: str, metrics: dict[str, Any], **extra: Any) -> None:
        for key, value in metrics.items():
            label = f"{prefix}_{key}" if prefix else key
            if isinstance(value, float):
                self.emit(label, f"{value:.6f}", **extra)
            else:
                self.emit(label, value, **extra)

    def write_jsonl(self, key: str, value: Any, **extra: Any) -> None:
        if not self.jsonl_path:
            return
        row = {
            "schema": "tiny_leela.metric.v1",
            "time": time.time(),
            "key": key,
            "value": value,
        }
        if self.run_id:
            row["run_id"] = self.run_id
        row.update({k: v for k, v in extra.items() if v is not None})
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def install_metric_print_tee(jsonl_path: str | Path | None, *, run_id: str | None = None) -> MetricEmitter:
    """Mirror existing `print('METRIC key=value')` calls into JSONL.

    This lets older training scripts gain structured metrics without invasive
    rewrites. Non-METRIC prints are passed through unchanged.
    """

    emitter = MetricEmitter(jsonl_path, run_id=run_id)
    if not jsonl_path:
        return emitter

    def tee_print(*args: Any, **kwargs: Any) -> None:
        _ORIGINAL_PRINT(*args, **kwargs)
        sep = kwargs.get("sep", " ")
        text = sep.join(str(a) for a in args)
        m = _METRIC_RE.match(text.strip())
        if m:
            emitter.write_jsonl(m.group(1), _coerce(m.group(2)))

    builtins.print = tee_print
    return emitter
