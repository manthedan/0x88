#!/usr/bin/env python3
"""Inspect LC0 public chunk containers before adapter conversion.

This first-pass inspector is conservative: it identifies container/compression
shape, computes checksums, lists tar members, and samples bytes. Record-level V6
parsing belongs in training/lc0_adapter.py once the format contract is pinned.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import tarfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

MAGICS = {
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ\x00": "xz",
    b"PK\x03\x04": "zip",
    b"\x28\xb5\x2f\xfd": "zstd",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_magic(path: Path) -> str:
    head = path.read_bytes()[:16]
    for magic, name in MAGICS.items():
        if head.startswith(magic):
            return name
    if tarfile.is_tarfile(path):
        return "tar"
    return "raw"


def hex_preview(data: bytes, limit: int = 64) -> str:
    return data[:limit].hex()


def ascii_preview(data: bytes, limit: int = 128) -> str:
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data[:limit])


def read_sample(path: Path, kind: str, limit: int) -> bytes:
    if kind == "gzip":
        with gzip.open(path, "rb") as f:
            return f.read(limit)
    if kind == "xz":
        with lzma.open(path, "rb") as f:
            return f.read(limit)
    with path.open("rb") as f:
        return f.read(limit)


def inspect_tar(path: Path, limit_members: int, sample_bytes: int) -> dict[str, Any]:
    members = []
    with tarfile.open(path, "r:*") as tf:
        for i, m in enumerate(tf):
            if i >= limit_members:
                break
            entry: dict[str, Any] = {
                "name": m.name,
                "size": m.size,
                "type": "file" if m.isfile() else "dir" if m.isdir() else str(m.type),
            }
            if m.isfile() and sample_bytes > 0:
                f = tf.extractfile(m)
                if f is not None:
                    sample = f.read(sample_bytes)
                    entry["sample"] = {
                        "hex": hex_preview(sample),
                        "ascii": ascii_preview(sample),
                        "detected_inner_magic": detect_magic_from_bytes(sample),
                    }
            members.append(entry)
    return {"tar_members_sampled": members, "tar_member_limit": limit_members}


def inspect_zip(path: Path, limit_members: int, sample_bytes: int) -> dict[str, Any]:
    members = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist()[:limit_members]:
            entry: dict[str, Any] = {"name": info.filename, "size": info.file_size, "compressed_size": info.compress_size}
            if sample_bytes > 0 and not info.is_dir():
                with zf.open(info) as f:
                    sample = f.read(sample_bytes)
                entry["sample"] = {
                    "hex": hex_preview(sample),
                    "ascii": ascii_preview(sample),
                    "detected_inner_magic": detect_magic_from_bytes(sample),
                }
            members.append(entry)
    return {"zip_members_sampled": members, "zip_member_limit": limit_members}


def detect_magic_from_bytes(head: bytes) -> str:
    for magic, name in MAGICS.items():
        if head.startswith(magic):
            return name
    return "raw"


def inspect_path(path: Path, *, sample_bytes: int, limit_members: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    stat = path.stat()
    kind = detect_magic(path)
    report: dict[str, Any] = {
        "schema": "tiny_leela.lc0_chunk_inspect.v1",
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
        "container": kind,
    }

    try:
        if kind == "tar":
            report.update(inspect_tar(path, limit_members, sample_bytes))
        elif kind == "zip":
            report.update(inspect_zip(path, limit_members, sample_bytes))
        else:
            sample = read_sample(path, kind, sample_bytes)
            report["sample"] = {
                "decompressed_if_supported": kind in {"gzip", "xz"},
                "hex": hex_preview(sample),
                "ascii": ascii_preview(sample),
                "detected_inner_magic": detect_magic_from_bytes(sample),
            }
            if kind == "zstd":
                report["warning"] = "zstd decompression is not implemented in this inspector yet; sampled compressed bytes only"
    except Exception as exc:
        report["inspect_error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="LC0 raw chunk/sample paths")
    parser.add_argument("--sample-bytes", type=int, default=256)
    parser.add_argument("--limit-members", type=int, default=20)
    parser.add_argument("--out", default=None, help="Write JSON report to this path; stdout always receives report too")
    args = parser.parse_args()

    reports = [inspect_path(Path(p), sample_bytes=args.sample_bytes, limit_members=args.limit_members) for p in args.paths]
    payload: Any = reports[0] if len(reports) == 1 else {"schema": "tiny_leela.lc0_chunk_inspect_bundle.v1", "reports": reports}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
