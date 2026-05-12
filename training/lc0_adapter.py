#!/usr/bin/env python3
"""LC0 public chunk -> Tiny Leela normalized-example adapter.

This module implements the first correctness-first V6 decoder path for LC0 public
training chunks. It is intentionally conservative: unsupported formats and any
positive policy mass that cannot be mapped to a legal move drop the whole record
and are counted in an audit report.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import json
import math
import re
import struct
import sys
import tarfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

import chess

BOARD_NORMALIZATION = "stm_white_rankflip_v1"
TEACHER = "lc0_public"
LC0_POLICY_SIZE = 1858
LC0_V6_RECORD_BYTES = 8356
K_PLANES_PER_BOARD = 13
K_AUX_PLANE_BASE = K_PLANES_PER_BOARD * 8

LC0_V6_HEADER = struct.Struct("<II")
LC0_V6_PROBS = struct.Struct("<1858f")
LC0_V6_PLANES = struct.Struct("<104Q")
LC0_V6_TAIL = struct.Struct("<BBBBBBBB15fIHHfI")
assert LC0_V6_TAIL.size == LC0_V6_RECORD_BYTES - 8272

FILES = "abcdefgh"
RANKS = "12345678"


def _sq_name(idx: int) -> str:
    return FILES[idx % 8] + RANKS[idx // 8]


def _build_lc0_policy_moves() -> list[str]:
    moves: list[str] = []
    for fr in range(64):
        ff, rr = fr % 8, fr // 8
        for to in range(64):
            if to == fr:
                continue
            tf, tr = to % 8, to // 8
            df, dr = abs(tf - ff), abs(tr - rr)
            if df == 0 or dr == 0 or df == dr or (df, dr) in {(1, 2), (2, 1)}:
                moves.append(_sq_name(fr) + _sq_name(to))
    for file_idx in range(8):
        from_sq = _sq_name(6 * 8 + file_idx)
        for to_file in (file_idx - 1, file_idx, file_idx + 1):
            if not (0 <= to_file < 8):
                continue
            to_sq = _sq_name(7 * 8 + to_file)
            for promo in "qrb":
                moves.append(from_sq + to_sq + promo)
    if len(moves) != LC0_POLICY_SIZE:
        raise AssertionError(f"bad LC0 policy map size {len(moves)}")
    return moves


LC0_POLICY_MOVES = _build_lc0_policy_moves()


@dataclasses.dataclass(frozen=True)
class Wdl:
    win: float
    draw: float
    loss: float

    def as_dict(self) -> dict[str, float]:
        return {"win": self.win, "draw": self.draw, "loss": self.loss}


def qd_to_wdl(q: float, d: float, *, clamp: bool = False) -> Wdl:
    """Convert LC0 Q/D targets to WDL from side-to-move perspective."""
    if not math.isfinite(q) or not math.isfinite(d):
        raise ValueError(f"non-finite q/d: q={q!r} d={d!r}")
    w = (1.0 - d + q) / 2.0
    l = (1.0 - d - q) / 2.0
    out = Wdl(w, d, l)
    if clamp:
        vals = [max(0.0, min(1.0, x)) for x in (out.win, out.draw, out.loss)]
        s = sum(vals)
        if s <= 0:
            raise ValueError(f"cannot clamp/renormalize q/d: q={q!r} d={d!r}")
        out = Wdl(vals[0] / s, vals[1] / s, vals[2] / s)
    return out


def validate_wdl(wdl: Wdl, *, eps: float = 1e-5) -> list[str]:
    issues: list[str] = []
    vals = [wdl.win, wdl.draw, wdl.loss]
    if any(not math.isfinite(v) for v in vals):
        issues.append("non_finite_wdl")
    if any(v < -eps or v > 1.0 + eps for v in vals):
        issues.append("wdl_out_of_range")
    if abs(sum(vals) - 1.0) > eps:
        issues.append("wdl_sum_not_one")
    return issues


@dataclasses.dataclass
class DropAudit:
    total_records: int = 0
    emitted_records: int = 0
    drop_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    samples: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    max_samples_per_reason: int = 5
    input_format_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    chunk_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    positive_policy_mass_total: float = 0.0
    illegal_positive_policy_mass_total: float = 0.0
    unknown_move_count: int = 0
    illegal_positive_move_count: int = 0

    def drop(self, reason: str, sample: dict[str, Any] | None = None) -> None:
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1
        if sample is not None:
            bucket = self.samples.setdefault(reason, [])
            if len(bucket) < self.max_samples_per_reason:
                bucket.append(sample)

    def as_dict(self) -> dict[str, Any]:
        dropped = sum(self.drop_counts.values())
        return {
            "schema": "tiny_leela.lc0_adapter_audit.v1",
            "teacher": TEACHER,
            "board_normalization": BOARD_NORMALIZATION,
            "total_records": self.total_records,
            "emitted_records": self.emitted_records,
            "dropped_records": dropped,
            "drop_rate": (dropped / self.total_records) if self.total_records else 0.0,
            "unknown_move_count": self.unknown_move_count,
            "illegal_positive_move_count": self.illegal_positive_move_count,
            "positive_policy_mass_total": self.positive_policy_mass_total,
            "illegal_positive_policy_mass_total": self.illegal_positive_policy_mass_total,
            "illegal_positive_policy_mass_rate": (
                self.illegal_positive_policy_mass_total / self.positive_policy_mass_total
                if self.positive_policy_mass_total
                else 0.0
            ),
            "input_format_counts": dict(sorted(self.input_format_counts.items())),
            "chunk_counts": dict(sorted(self.chunk_counts.items())),
            "drop_counts": dict(sorted(self.drop_counts.items())),
            "samples": self.samples,
        }


def _rank_mirror_square(sq: str) -> str:
    return sq[0] + str(9 - int(sq[1]))


def _rank_mirror_uci(uci: str) -> str:
    return _rank_mirror_square(uci[:2]) + _rank_mirror_square(uci[2:4]) + uci[4:]


def _reverse_bits_in_bytes(mask: int) -> int:
    """Mirror files within every rank, matching LC0 ReverseBitsInBytes()."""
    out = 0
    for idx in range(64):
        if (mask >> idx) & 1:
            file_idx, rank_idx = idx % 8, idx // 8
            out |= 1 << (rank_idx * 8 + (7 - file_idx))
    return out


def _mirror_mask(mask: int) -> int:
    out = 0
    for idx in range(64):
        if (mask >> idx) & 1:
            file_idx, rank_idx = idx % 8, idx // 8
            out |= 1 << ((7 - rank_idx) * 8 + file_idx)
    return out


def _mask_has(mask: int, file_idx: int, rank_idx: int) -> bool:
    return bool((mask >> (rank_idx * 8 + file_idx)) & 1)


def _castling_rights(us_ooo: int, us_oo: int, them_ooo: int, them_oo: int, black_to_move: bool) -> str:
    rights = ""
    if not black_to_move:
        if us_oo:
            rights += "K"
        if us_ooo:
            rights += "Q"
        if them_oo:
            rights += "k"
        if them_ooo:
            rights += "q"
    else:
        if them_oo:
            rights += "K"
        if them_ooo:
            rights += "Q"
        if us_oo:
            rights += "k"
        if us_ooo:
            rights += "q"
    return rights or "-"


def _infer_ep_square(planes: list[int], black_to_move: bool) -> str:
    # Mirrors LC0's conservative decoder heuristic for classical input: compare
    # current and previous opponent-pawn planes. For black-to-move records, both
    # current and previous masks are mirrored before producing actual-board FEN.
    cur = planes[6]
    prev = planes[K_PLANES_PER_BOARD + 6]
    if black_to_move:
        cur = _mirror_mask(cur)
        prev = _mirror_mask(prev)
    diff = cur ^ prev
    if diff.bit_count() != 2 or prev == 0:
        return "-"
    squares = [idx for idx in range(64) if (diff >> idx) & 1]
    from_sq = next((idx for idx in squares if (prev >> idx) & 1), None)
    to_sq = next((idx for idx in squares if (cur >> idx) & 1), None)
    if from_sq is None or to_sq is None:
        return "-"
    ff, fr = from_sq % 8, from_sq // 8
    tf, tr = to_sq % 8, to_sq // 8
    if ff != tf or abs(fr - tr) != 2:
        return "-"
    ep_rank = 2 if black_to_move else 5  # zero-based ranks 3/6.
    return _sq_name(ep_rank * 8 + tf)


def planes_to_fen(
    planes: list[int],
    *,
    input_format: int,
    us_ooo: int,
    us_oo: int,
    them_ooo: int,
    them_oo: int,
    side_to_move_or_enpassant: int,
    rule50_count: int,
) -> str:
    if input_format != 1:
        raise ValueError(f"unsupported input_format={input_format}; first decoder supports classical format 1 only")
    black_to_move = bool(side_to_move_or_enpassant)
    # LC0 stores plane uint64s in a file-mirrored byte layout; its reference
    # reader applies ReverseBitsInBytes() before decoding/populating a board.
    planes = [_reverse_bits_in_bytes(p) for p in planes]
    piece_planes = planes[:12]
    if black_to_move:
        # Convert side-to-move-relative planes back to ordinary white-board FEN.
        piece_planes = [
            _mirror_mask(piece_planes[6]),
            _mirror_mask(piece_planes[7]),
            _mirror_mask(piece_planes[8]),
            _mirror_mask(piece_planes[9]),
            _mirror_mask(piece_planes[10]),
            _mirror_mask(piece_planes[11]),
            _mirror_mask(piece_planes[0]),
            _mirror_mask(piece_planes[1]),
            _mirror_mask(piece_planes[2]),
            _mirror_mask(piece_planes[3]),
            _mirror_mask(piece_planes[4]),
            _mirror_mask(piece_planes[5]),
        ]
    chars = "PNBRQKpnbrqk"
    board_rows: list[str] = []
    for rank_idx in range(7, -1, -1):
        row = ""
        empty = 0
        for file_idx in range(8):
            piece = None
            for plane_idx, ch in enumerate(chars):
                if _mask_has(piece_planes[plane_idx], file_idx, rank_idx):
                    piece = ch
                    break
            if piece:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
            else:
                empty += 1
        if empty:
            row += str(empty)
        board_rows.append(row)
    placement = "/".join(board_rows)
    stm = "b" if black_to_move else "w"
    rights = _castling_rights(us_ooo, us_oo, them_ooo, them_oo, black_to_move)
    ep = _infer_ep_square(planes, black_to_move)
    # Fullmove is unknown; keep legal FEN by using 1.
    return f"{placement} {stm} {rights} {ep} {int(rule50_count)} 1"


@dataclasses.dataclass(frozen=True)
class V6Record:
    version: int
    input_format: int
    probabilities: tuple[float, ...]
    planes: list[int]
    us_ooo: int
    us_oo: int
    them_ooo: int
    them_oo: int
    side_to_move_or_enpassant: int
    rule50_count: int
    invariance_info: int
    dummy: int
    root_q: float
    best_q: float
    root_d: float
    best_d: float
    root_m: float
    best_m: float
    plies_left: float
    result_q: float
    result_d: float
    played_q: float
    played_d: float
    played_m: float
    orig_q: float
    orig_d: float
    orig_m: float
    visits: int
    played_idx: int
    best_idx: int
    policy_kld: float
    reserved: int


def parse_v6_record(content: bytes) -> V6Record:
    if len(content) != LC0_V6_RECORD_BYTES:
        raise ValueError(f"expected {LC0_V6_RECORD_BYTES} bytes, got {len(content)}")
    version, input_format = LC0_V6_HEADER.unpack_from(content, 0)
    probs = LC0_V6_PROBS.unpack_from(content, 8)
    planes = list(LC0_V6_PLANES.unpack_from(content, 7440))
    tail = LC0_V6_TAIL.unpack_from(content, 8272)
    if version != 6:
        raise ValueError(f"expected V6 record, got version={version}")
    return V6Record(version, input_format, probs, planes, *tail)


def normalize_policy(policy: dict[str, float], legal_moves: set[str], audit: DropAudit | None = None) -> dict[str, float] | None:
    illegal_positive = {m: p for m, p in policy.items() if p > 0.0 and m not in legal_moves}
    if illegal_positive:
        if audit:
            audit.drop("illegal_positive_policy_mass", {"illegal": dict(list(illegal_positive.items())[:8])})
        return None
    kept = {m: float(p) for m, p in policy.items() if m in legal_moves and p > 0.0 and math.isfinite(float(p))}
    mass = sum(kept.values())
    if mass <= 0.0:
        if audit:
            audit.drop("zero_legal_policy_mass", {"legal_moves": sorted(legal_moves)[:16]})
        return None
    return {m: p / mass for m, p in sorted(kept.items())}


def build_normalized_example(
    *,
    source_ref: dict[str, Any],
    board: dict[str, Any],
    legal_moves: Iterable[str],
    policy_uci: dict[str, float],
    root_q: float,
    root_d: float,
    extra: dict[str, Any] | None = None,
    audit: DropAudit | None = None,
) -> dict[str, Any] | None:
    legal = set(legal_moves)
    sparse_policy = normalize_policy(policy_uci, legal, audit=audit)
    if sparse_policy is None:
        return None
    wdl = qd_to_wdl(root_q, root_d)
    issues = validate_wdl(wdl)
    if issues:
        if audit:
            audit.drop("invalid_root_wdl", {"root_q": root_q, "root_d": root_d, "issues": issues})
        return None
    return {
        "schema": "tiny_leela.lc0_normalized_example.v1",
        "teacher": TEACHER,
        "source_ref": source_ref,
        "board_normalization": BOARD_NORMALIZATION,
        "board": board,
        "legal_moves_uci": sorted(legal),
        "policy_target_uci": sparse_policy,
        "value_targets": {
            "root_q": root_q,
            "root_d": root_d,
            "wdl_root": wdl.as_dict(),
        },
        "metadata": extra or {},
    }


def _legal_policy_uci(record: V6Record, board: chess.Board, *, top_k: int, min_prob: float, audit: DropAudit) -> dict[str, float] | None:
    legal = {m.uci(): m for m in board.legal_moves}
    legal_by_from_to: dict[str, list[str]] = {}
    for uci in legal:
        legal_by_from_to.setdefault(uci[:4], []).append(uci)
    black_to_move = board.turn == chess.BLACK
    candidates: list[tuple[int, float]] = [
        (idx, float(p)) for idx, p in enumerate(record.probabilities) if math.isfinite(float(p)) and float(p) > min_prob
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    if top_k > 0:
        candidates = candidates[:top_k]
    mapped: dict[str, float] = {}
    illegal: dict[str, float] = {}
    unknown = 0
    for idx, prob in candidates:
        audit.positive_policy_mass_total += prob
        if not (0 <= idx < len(LC0_POLICY_MOVES)):
            unknown += 1
            continue
        raw_uci = LC0_POLICY_MOVES[idx]
        actual = _rank_mirror_uci(raw_uci) if black_to_move else raw_uci
        if actual in legal:
            mapped[actual] = mapped.get(actual, 0.0) + prob
            continue
        # LC0 encodes castling to the rook square; python-chess uses the king
        # destination square for ordinary UCI castling.
        castle_alias = None
        if actual in {"e1h1", "e8h8"}:
            castle_alias = actual[:2] + ("g1" if actual.startswith("e1") else "g8")
        elif actual in {"e1a1", "e8a8"}:
            castle_alias = actual[:2] + ("c1" if actual.startswith("e1") else "c8")
        if castle_alias and castle_alias in legal:
            mapped[castle_alias] = mapped.get(castle_alias, 0.0) + prob
            continue
        # LC0 encodes knight promotions as the corresponding from-to move with
        # no suffix; python-chess requires the trailing 'n'.
        if len(actual) == 4:
            knight = actual + "n"
            if knight in legal:
                mapped[knight] = mapped.get(knight, 0.0) + prob
                continue
            same_from_to = legal_by_from_to.get(actual, [])
            if len(same_from_to) == 1:
                mapped[same_from_to[0]] = mapped.get(same_from_to[0], 0.0) + prob
                continue
        illegal[actual] = illegal.get(actual, 0.0) + prob
    if unknown:
        audit.unknown_move_count += unknown
    if illegal:
        audit.illegal_positive_move_count += len(illegal)
        audit.illegal_positive_policy_mass_total += sum(illegal.values())
        audit.drop("illegal_positive_policy_mass", {"fen": board.fen(), "illegal": dict(list(illegal.items())[:12])})
        return None
    if not mapped:
        audit.drop("zero_mapped_policy_mass", {"fen": board.fen(), "top_candidates": candidates[:8]})
        return None
    mass = sum(mapped.values())
    return {k: v / mass for k, v in sorted(mapped.items())}


def record_to_example(record: V6Record, *, source_ref: dict[str, Any], top_k: int, min_prob: float, audit: DropAudit) -> dict[str, Any] | None:
    audit.input_format_counts[str(record.input_format)] = audit.input_format_counts.get(str(record.input_format), 0) + 1
    if record.input_format != 1:
        audit.drop("unsupported_input_format", {"input_format": record.input_format, "source_ref": source_ref})
        return None
    try:
        fen = planes_to_fen(
            record.planes,
            input_format=record.input_format,
            us_ooo=record.us_ooo,
            us_oo=record.us_oo,
            them_ooo=record.them_ooo,
            them_oo=record.them_oo,
            side_to_move_or_enpassant=record.side_to_move_or_enpassant,
            rule50_count=record.rule50_count,
        )
        board = chess.Board(fen)
    except Exception as exc:
        audit.drop("invalid_board_decode", {"source_ref": source_ref, "error": f"{type(exc).__name__}: {exc}"})
        return None
    policy = _legal_policy_uci(record, board, top_k=top_k, min_prob=min_prob, audit=audit)
    if policy is None:
        return None
    root_wdl = qd_to_wdl(record.root_q, record.root_d)
    issues = validate_wdl(root_wdl)
    if issues:
        audit.drop("invalid_root_wdl", {"source_ref": source_ref, "root_q": record.root_q, "root_d": record.root_d, "issues": issues})
        return None
    result_wdl = qd_to_wdl(record.result_q, record.result_d, clamp=True)
    best_wdl = qd_to_wdl(record.best_q, record.best_d, clamp=True)
    played_wdl = qd_to_wdl(record.played_q, record.played_d, clamp=True)
    return {
        "schema": "tiny_leela.lc0_normalized_example.v1",
        "teacher": TEACHER,
        "source_ref": source_ref,
        "board_normalization": BOARD_NORMALIZATION,
        "board": {"fen": board.fen(), "input_format": record.input_format},
        "legal_moves_uci": sorted(m.uci() for m in board.legal_moves),
        "policy_target_uci": policy,
        "value_targets": {
            "root_q": record.root_q,
            "root_d": record.root_d,
            "wdl_root": root_wdl.as_dict(),
            "result_q": record.result_q,
            "result_d": record.result_d,
            "wdl_result": result_wdl.as_dict(),
            "best_q": record.best_q,
            "best_d": record.best_d,
            "wdl_best": best_wdl.as_dict(),
            "played_q": record.played_q,
            "played_d": record.played_d,
            "wdl_played": played_wdl.as_dict(),
        },
        "metadata": {
            "root_m": record.root_m,
            "best_m": record.best_m,
            "plies_left": record.plies_left,
            "played_m": record.played_m,
            "orig_q": record.orig_q,
            "orig_d": record.orig_d,
            "orig_m": record.orig_m,
            "visits": record.visits,
            "played_idx": record.played_idx,
            "best_idx": record.best_idx,
            "played_move_lc0": LC0_POLICY_MOVES[record.played_idx] if record.played_idx < len(LC0_POLICY_MOVES) else None,
            "best_move_lc0": LC0_POLICY_MOVES[record.best_idx] if record.best_idx < len(LC0_POLICY_MOVES) else None,
            "policy_kld": record.policy_kld,
            "invariance_info": record.invariance_info,
            "top_k": top_k,
            "policy_min_prob": min_prob,
        },
    }


def _records_from_gzip_bytes(blob: bytes) -> Iterator[bytes]:
    data = gzip.decompress(blob)
    usable = len(data) - (len(data) % LC0_V6_RECORD_BYTES)
    for off in range(0, usable, LC0_V6_RECORD_BYTES):
        yield data[off : off + LC0_V6_RECORD_BYTES]


def iter_record_bytes(path: Path, *, max_members: int | None = None) -> Iterator[tuple[str, int, bytes]]:
    name = path.name
    if name.endswith(".gz"):
        blob = path.read_bytes()
        for i, rec in enumerate(_records_from_gzip_bytes(blob)):
            yield (str(path), i, rec)
        return
    if ".tar" in name:
        count_members = 0
        with path.open("rb") as f:
            try:
                tf = tarfile.open(fileobj=f, mode="r|*")
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".gz"):
                        continue
                    if max_members is not None and count_members >= max_members:
                        break
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    blob = extracted.read()
                    count_members += 1
                    for i, rec in enumerate(_records_from_gzip_bytes(blob)):
                        yield (member.name, i, rec)
            except (tarfile.ReadError, EOFError):
                # Partial HTTP-range tar samples are expected to end abruptly.
                return
        return
    raise ValueError(f"unsupported input path (expected .gz or .tar/.tar.part): {path}")


def convert_v6(input_path: Path, output_path: Path, audit_path: Path, *, limit_records: int, top_k: int, min_prob: float, max_members: int | None) -> None:
    audit = DropAudit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out:
        for chunk_name, record_idx, rec_bytes in iter_record_bytes(input_path, max_members=max_members):
            if limit_records > 0 and audit.emitted_records >= limit_records:
                break
            audit.total_records += 1
            audit.chunk_counts[chunk_name] = audit.chunk_counts.get(chunk_name, 0) + 1
            try:
                record = parse_v6_record(rec_bytes)
                ex = record_to_example(
                    record,
                    source_ref={"input_path": str(input_path), "chunk": chunk_name, "record_idx": record_idx},
                    top_k=top_k,
                    min_prob=min_prob,
                    audit=audit,
                )
            except Exception as exc:
                audit.drop("parse_or_convert_error", {"chunk": chunk_name, "record_idx": record_idx, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if ex is None:
                continue
            out.write(json.dumps(ex, sort_keys=True) + "\n")
            audit.emitted_records += 1
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n")


def convert_jsonl_smoke(input_path: Path, output_path: Path, audit_path: Path) -> None:
    audit = DropAudit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as src, output_path.open("w") as out:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            audit.total_records += 1
            try:
                rec = json.loads(line)
                ex = build_normalized_example(
                    source_ref=rec.get("source_ref", {"line": line_no}),
                    board=rec.get("board", {}),
                    legal_moves=rec["legal_moves_uci"],
                    policy_uci=rec["policy_target_uci"],
                    root_q=float(rec["root_q"]),
                    root_d=float(rec["root_d"]),
                    extra=rec.get("metadata", {}),
                    audit=audit,
                )
            except Exception as exc:
                audit.drop("parse_or_convert_error", {"line": line_no, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if ex is None:
                continue
            out.write(json.dumps(ex, sort_keys=True) + "\n")
            audit.emitted_records += 1
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    qd = sub.add_parser("qd-to-wdl", help="Convert one Q/D pair to WDL")
    qd.add_argument("--q", type=float, required=True)
    qd.add_argument("--d", type=float, required=True)
    qd.add_argument("--clamp", action="store_true")

    smoke = sub.add_parser("jsonl-smoke", help="Canonicalize smoke JSONL records; not a binary LC0 parser")
    smoke.add_argument("--input", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--audit", required=True)

    conv = sub.add_parser("convert-v6", help="Convert LC0 V6 .gz/.tar sample to auditable normalized JSONL")
    conv.add_argument("--input", required=True)
    conv.add_argument("--output", required=True)
    conv.add_argument("--audit", required=True)
    conv.add_argument("--limit-records", type=int, default=1000, help="Stop after this many emitted records; <=0 means no emitted limit")
    conv.add_argument("--top-k", type=int, default=8, help="Keep top-k positive policy entries per raw position")
    conv.add_argument("--min-prob", type=float, default=1e-12)
    conv.add_argument("--max-members", type=int, default=None, help="Limit .gz members read from tar sample")

    args = parser.parse_args(argv)
    if args.cmd == "qd-to-wdl":
        out = qd_to_wdl(args.q, args.d, clamp=args.clamp)
        print(json.dumps(out.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.cmd == "jsonl-smoke":
        convert_jsonl_smoke(Path(args.input), Path(args.output), Path(args.audit))
        return 0
    if args.cmd == "convert-v6":
        convert_v6(
            Path(args.input),
            Path(args.output),
            Path(args.audit),
            limit_records=args.limit_records,
            top_k=args.top_k,
            min_prob=args.min_prob,
            max_members=args.max_members,
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
