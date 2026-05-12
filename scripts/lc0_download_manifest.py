#!/usr/bin/env python3
"""Download/manifest tiny LC0 public training-data samples.

This is intentionally format-agnostic. It records provenance before any adapter
code tries to interpret LC0 records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_LICENSE_URL = "https://storage.lczero.org/files/training_data/LICENSE.txt"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, *, max_bytes: int | None = None) -> tuple[int, str | None]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-leela-lc0-manifest/1"})
    content_type = None
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - explicit user-provided URL tool
        content_type = resp.headers.get("content-type")
        total = 0
        with tempfile.NamedTemporaryFile(dir=str(dst.parent), delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if max_bytes is not None and total > max_bytes:
                        raise RuntimeError(f"download exceeded --max-bytes ({max_bytes})")
                    tmp.write(block)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path.replace(dst)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
    return total, content_type


def read_text_url(url: str, *, max_bytes: int = 256 * 1024) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-leela-lc0-manifest/1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - explicit public license URL
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"license text exceeds {max_bytes} bytes: {url}")
    return data.decode("utf-8", errors="replace")


def build_manifest(args: argparse.Namespace, raw_path: Path, license_path: Path | None, content_type: str | None) -> dict[str, Any]:
    stat = raw_path.stat()
    return {
        "schema": "tiny_leela.lc0_public_manifest.v1",
        "created_at": utc_now(),
        "teacher": "lc0_public",
        "source": {
            "url": args.url,
            "run_id": args.run_id,
            "test_id": args.test_id,
            "chunk_id": args.chunk_id or raw_path.name,
            "format_hint": args.format_hint,
            "content_type": content_type,
            "license_url": args.license_url,
        },
        "local": {
            "raw_path": str(raw_path),
            "license_path": str(license_path) if license_path else None,
        },
        "integrity": {
            "bytes": stat.st_size,
            "sha256": sha256_file(raw_path),
        },
        "normalization_default": "stm_white_rankflip_v1",
        "notes": args.notes or None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="LC0 public chunk/sample URL")
    parser.add_argument("--run-id", default=None, help="LC0 run identifier, if known")
    parser.add_argument("--test-id", default=None, help="LC0 test identifier, if known")
    parser.add_argument("--chunk-id", default=None, help="Human chunk identifier; defaults to output filename")
    parser.add_argument("--format-hint", default=None, help="Expected format, e.g. v6/tar/gzip; informational")
    parser.add_argument("--raw-dir", default="data/external/lc0_public/raw")
    parser.add_argument("--manifest-dir", default="data/external/lc0_public/manifests")
    parser.add_argument("--output-name", default=None, help="Raw output filename; defaults to URL basename")
    parser.add_argument("--manifest-name", default=None, help="Manifest filename; defaults to raw filename + .manifest.json")
    parser.add_argument("--license-url", default=DEFAULT_LICENSE_URL)
    parser.add_argument("--skip-license", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=None, help="Safety cap for download size")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args(argv)

    url_name = Path(urllib.request.urlparse(args.url).path).name
    if not url_name and not args.output_name:
        parser.error("URL has no basename; pass --output-name")
    raw_path = Path(args.raw_dir) / (args.output_name or url_name)
    manifest_path = Path(args.manifest_dir) / (args.manifest_name or f"{raw_path.name}.manifest.json")

    print(f"download {args.url} -> {raw_path}", file=sys.stderr)
    nbytes, content_type = download(args.url, raw_path, max_bytes=args.max_bytes)
    print(f"downloaded bytes={nbytes} sha256={sha256_file(raw_path)}", file=sys.stderr)

    license_path = None
    if not args.skip_license and args.license_url:
        license_dir = Path(args.manifest_dir) / "licenses"
        license_dir.mkdir(parents=True, exist_ok=True)
        license_path = license_dir / "lc0_training_data_LICENSE.txt"
        text = read_text_url(args.license_url)
        license_path.write_text(text)

    manifest = build_manifest(args, raw_path, license_path, content_type)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
