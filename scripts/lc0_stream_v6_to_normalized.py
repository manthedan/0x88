#!/usr/bin/env python3
"""Stream one or more LC0 V6 tar/gzip inputs as Tiny Leela normalized JSONL.

This is the multi-input/stdout companion to training/lc0_adapter.py convert-v6.
It is intended for large LC0 cache builds where materializing a combined
normalized JSONL is optional and weighted JSONL should stay on FIFO.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the validated adapter implementation rather than duplicating decoder logic.
from training.lc0_adapter import (
    DropAudit,
    is_standard_initial_record,
    iter_record_bytes,
    parse_v6_record,
    record_board,
    record_to_example,
)


def stream_convert_v6(
    inputs: list[Path],
    out: TextIO,
    audit_path: Path,
    *,
    limit_records: int,
    top_k: int,
    min_prob: float,
    max_members: int | None,
    skip_nonstandard_initial_position: bool,
) -> None:
    audit = DropAudit()
    skipped_chunks: set[str] = set()
    seen_chunks: set[str] = set()
    emitted_total = 0
    for input_path in inputs:
        if limit_records > 0 and emitted_total >= limit_records:
            break
        for chunk_name, record_idx, rec_bytes in iter_record_bytes(input_path, max_members=max_members):
            if limit_records > 0 and emitted_total >= limit_records:
                break
            audit.total_records += 1
            audit.chunk_counts[chunk_name] = audit.chunk_counts.get(chunk_name, 0) + 1
            source_ref = {"input_path": str(input_path), "chunk": chunk_name, "record_idx": record_idx}
            try:
                record = parse_v6_record(rec_bytes)
                if skip_nonstandard_initial_position:
                    chunk_key = f"{input_path}:{chunk_name}"
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        if not is_standard_initial_record(record):
                            skipped_chunks.add(chunk_key)
                            audit.drop(
                                "unsupported_nonstandard_initial_position",
                                {"source_ref": source_ref, "fen": record_board(record).fen()},
                            )
                            continue
                    elif chunk_key in skipped_chunks:
                        audit.drop("unsupported_nonstandard_initial_position", {"source_ref": source_ref})
                        continue
                ex = record_to_example(
                    record,
                    source_ref=source_ref,
                    top_k=top_k,
                    min_prob=min_prob,
                    audit=audit,
                )
            except Exception as exc:  # keep large overnight runs moving; audit every failure class.
                audit.drop(
                    "parse_or_convert_error",
                    {"input_path": str(input_path), "chunk": chunk_name, "record_idx": record_idx, "error": f"{type(exc).__name__}: {exc}"},
                )
                continue
            if ex is None:
                continue
            out.write(json.dumps(ex, sort_keys=True) + "\n")
            emitted_total += 1
            audit.emitted_records += 1
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", default=[], help="LC0 .tar/.tar.part/.gz input; may be repeated")
    ap.add_argument("--input-list", help="Optional newline-delimited input path list")
    ap.add_argument("--output", default="-", help="Output JSONL path, or '-' for stdout")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--limit-records", type=int, default=0, help="Stop after this many emitted records; <=0 means no emitted limit")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-prob", type=float, default=1e-12)
    ap.add_argument("--max-members", type=int, default=None)
    ap.add_argument(
        "--include-nonstandard-initial-position",
        dest="skip_nonstandard_initial_position",
        action="store_false",
        help="Do not skip gzip members whose first record is not ordinary chess startpos.",
    )
    ap.set_defaults(skip_nonstandard_initial_position=True)
    args = ap.parse_args(argv)

    inputs = [Path(p) for p in args.input]
    if args.input_list:
        inputs.extend(Path(line.strip()) for line in Path(args.input_list).read_text().splitlines() if line.strip())
    if not inputs:
        ap.error("at least one --input or --input-list entry is required")
    if args.output == "-":
        stream_convert_v6(
            inputs,
            sys.stdout,
            Path(args.audit),
            limit_records=args.limit_records,
            top_k=args.top_k,
            min_prob=args.min_prob,
            max_members=args.max_members,
            skip_nonstandard_initial_position=args.skip_nonstandard_initial_position,
        )
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as out:
            stream_convert_v6(
                inputs,
                out,
                Path(args.audit),
                limit_records=args.limit_records,
                top_k=args.top_k,
                min_prob=args.min_prob,
                max_members=args.max_members,
                skip_nonstandard_initial_position=args.skip_nonstandard_initial_position,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
