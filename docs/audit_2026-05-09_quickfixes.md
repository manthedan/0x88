# Audit 2026-05-09 — concrete quick fixes

Companion to [`audit_2026-05-09.md`](audit_2026-05-09.md). This file lists actionable, file:line-specific items, sorted by effort. Effort estimates are wall-clock for a familiar maintainer.

## ~30 minutes each

### TS — fix zero-value backup in batched PUCT
File: `src/search/puct.ts:329`

```ts
if (sel.node.expanded) { prepared.push({ sel, value: 0 }); continue; }
```

Change to one of:

```ts
// (a) assert unreachable
if (sel.node.expanded) throw new Error('selectLeaf returned an expanded non-terminal node');

// (b) skip backup but unwind virtual visits
if (sel.node.expanded) {
  for (const edge of sel.path) edge.virtualVisits = Math.max(0, edge.virtualVisits - 1);
  // do not increment completedVisits or visit counts
  continue;
}
```

Add a regression test to `eval/puct_core_tests.mjs` that constructs the path-convergence shape and verifies Q is not biased toward zero.

### TS — explicit stride in `squareformerEvaluator.ts`
File: `src/nn/squareformerEvaluator.ts:41-46`

Replace the module-level `let dataStride = 0` with an explicit parameter:

```ts
function addBoardFeatures(data: Float32Array, board: BoardState, boardIndex: number, planesPerBoard: number, stride: number) {
  const base = boardIndex * planesPerBoard;
  for (let sq = 0; sq < 64; sq++) data[sq * stride + base + pieceId(board.squares[sq])] = 1;
}
```

Update the two call sites in `squareformerFloatInput` to pass `inputDim`.

### TS — wire `tsc --noEmit` into `npm test`
File: `package.json`

```json
"scripts": {
  "typecheck": "tsc --noEmit",
  "test": "npm run typecheck && node --test"
}
```

### Repo — remove or hide root-level dead weight
- `git rm Nibbler.standalone.html` if not used by any tooling. (`grep -r Nibbler .` first.)
- Add to `.gitignore`:
  ```
  awscliv2.zip
  tiny-leela-netlify-sfaux-512.zip
  /tiny_leela_research.md
  ```
  (Keep the `docs/` copy.)

### Docs — move clearly historical docs to archive
```bash
mkdir -p docs/archive
git mv docs/scaling-and-architecture-todo.md docs/archive/
git mv docs/architecture-ladder-1m-64x6-smoke.md docs/archive/
git mv docs/lc0-inspired-roadmap.md docs/archive/
git mv docs/maia-inspired-roadmap.md docs/archive/
git mv docs/expert-handoff-current-state.md docs/archive/
```

Update `docs/README.md` to reflect archived state.

---

## ~2-4 hours each

### TS — typed-array board representation (perf)
File: `src/chess/board.ts`

`squares: (Piece | null)[]` allocates a 64-element JS Array with mostly-strings. Replace with `Uint8Array(64)` where each byte is `colorBit << 4 | roleBits`. Update `cloneBoard` to use `new Uint8Array(squares)` (single typed-array copy, ~5× faster than spread). Adjust `parseFen`, `boardToFen`, and downstream consumers (`movegen.ts`, `studentEvaluator.ts`).

This is invasive, so gate it behind a test pass: rerun `tests/chess_rules.test.mjs`, `tests/move_codec_roundtrip.test.mjs`, `tests/policy_map.test.mjs`, `eval/puct_core_tests.mjs`.

### TS — cache `kingSquare` per `BoardState`
File: `src/chess/movegen.ts:127`

Add an optional `kingCache?: Partial<Record<Color, number>>` to `BoardState`. Populate lazily in `kingSquare()`; invalidate (set to undefined) inside `makeMove` only when the moved piece is a king. Wins on the inner `legalMoves` loop, especially for endgames.

### TS — discriminated union for batched PUCT items
File: `src/search/puct.ts:316-362`

Replace the `prepared` array's loose shape with:

```ts
type Prepared =
  | { kind: 'terminal'; sel: SelectedLeaf; value: number }
  | { kind: 'duplicate'; sel: SelectedLeaf }
  | { kind: 'eval'; sel: SelectedLeaf; slot: number }
  | { kind: 'noLegal'; sel: SelectedLeaf; value: number };
```

Switch on `kind` in the backup loop. This makes `evalSlot ?? 0` impossible and the duplicate-leaf case explicit.

### Python — promote a shared encoding module
Create `training/_lib/__init__.py`, `training/_lib/encoding.py` with:

```python
PIECES = ".PNBRQKpnbrqk"  # index 0 = empty
PIECE_INDEX = {ch: i for i, ch in enumerate(PIECES)}

def fixed_policy_moves() -> list[str]: ...
def move_to_action_id(move: str) -> int: ...
def policy_index(move: str) -> int: ...
def compact_tokens_to_residual_planes(tok, history=2, state_planes=False): ...
def decode_move_classes_np(m): ...
def expand_collection(paths): ...
def open_av_cache_dir(p): ...
```

Migrate at least:
- `train_residual_torch.py`
- `train_board_cnn.py`
- `train_residual_av_multicache_torch.py`
- `train_channelformer_av_multicache_torch.py`
- `train_squareformer_v2_torch.py`

Leave the others alone for now if it's risky to touch them; the import surface is small.

### Python — `ruff format` pass over `training/`
```bash
.venv-onnx/bin/python -m pip install ruff
.venv-onnx/bin/ruff format training/ --line-length 110
```

Open as a separate PR. Reviewers can ignore the diff and just run the tests.

### Cross-language parity test
File: `tests/encoding_parity.test.mjs` (new)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { POLICY_MOVES, POLICY_SIZE } from '../src/chess/policyMap.ts';

test('fixed_policy_moves parity (Python ↔ TypeScript)', () => {
  const stdout = execFileSync('.venv-onnx/bin/python', ['-c',
    'import sys; sys.path.insert(0, "training"); from _lib.encoding import fixed_policy_moves; import json; print(json.dumps(fixed_policy_moves()))'
  ], { encoding: 'utf8' });
  const py = JSON.parse(stdout);
  assert.equal(py.length, POLICY_SIZE);
  assert.deepEqual(py, POLICY_MOVES);
});
```

Also add a parity test that asserts every legal move in `eval/opening_suite_v1.fen` produces the same `move_to_action_id` in both languages.

### Python ↔ TS movegen perft parity
File: `tests/perft_parity.test.mjs` (new)

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';

function perft(board, depth) {
  if (depth === 0) return 1;
  let nodes = 0;
  for (const m of legalMoves(board)) nodes += perft(makeMove(board, m), depth - 1);
  return nodes;
}

const fens = [
  ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 3, 8902],
  ['r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', 2, 2039],
  // Kiwipete-style and EP edge cases here
];

for (const [fen, depth, expected] of fens) {
  test(`perft TS ${fen} depth ${depth}`, () => {
    assert.equal(perft(parseFen(fen), depth), expected);
  });
  test(`perft Rust ${fen} depth ${depth}`, () => {
    const out = execFileSync('cargo', ['run', '--release', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml',
      '--bin', 'tiny-leela-rust-eval', '--', '--perft', String(depth), '--fen', fen
    ], { encoding: 'utf8' });
    assert.match(out, new RegExp(`nodes\\s*=\\s*${expected}`));
  });
}
```

(You may need to wire `--perft` into `rust/tiny_leela_core/src/bin/eval.rs` first; it's a 30-line addition.)

---

## ~1 day

### Anchor-Elo rollup in the model manifest
File: `eval/build_model_manifest.py`

Add a column reading from `artifacts/anchor_arena/<model>__vs_anchor.json`:

```
| Model | … | Anchor pool Elo (Stockfish 1320, 32v, 20 pairs) |
| cnn_80x5_100m_e3 | … | -85 ± 35 |
```

If the anchor JSON is absent for a model, write `—` and add it to a `missing_anchor` list in the manifest's "TODO" section. This single change shifts the manifest from "we have models" to "we have models with measured strength."

### Kill-criteria column in the model manifest
File: `eval/model_manifest_overrides.json` and `eval/build_model_manifest.py`

Add a per-tag kill-criterion field:

```json
{
  "cnn-av": { "promote_if": "≥ +30 Elo over equal-budget classic-PUCT CNN @ 64v anchor", "deadline": "2026-06-15" },
  "moveformer": { "status": "parked", "park_marker": "artifacts/moveformer_10m_supervised_mf64x6_e3/PARK_MOVEFORMER" },
  "squareformer_v2": { "promote_if": "≥ baseline CNN policy CE *and* AV-MSE @ equal latency", "deadline": "2026-07-01" }
}
```

Surface the deadline + status in the manifest so monthly review is mechanical.

### `training/_lib/metrics.py` JSONL emitter
File: `training/_lib/metrics.py` (new)

```python
import json, os, sys, time
from pathlib import Path

def emit(label: str, jsonl_out: str | None = None, **kv):
    print('METRIC ' + label + ' ' + ' '.join(f'{k}={v:.6f}' if isinstance(v, float) else f'{k}={v}' for k, v in kv.items()), flush=True)
    if jsonl_out:
        Path(jsonl_out).parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_out, 'a') as f:
            f.write(json.dumps({'ts': time.time(), 'label': label, **kv}) + '\n')
```

Replace ad-hoc `print(f'METRIC ...', flush=True)` calls in trainers with `emit(...)`.

### Finish the UCI wrapper for OpenBench
File: `scripts/uci_tiny_leela.mjs`

The expert handoff says `OpenBench adoption is blocked on a stable Tiny Leela UCI wrapper.` This is the highest-leverage external-eval unlock. Specifically:

- handle `uci`, `isready`, `position fen ... moves ...`, `go nodes N`, `go movetime ms`, `stop`, `quit`;
- print `info nodes N nps Y pv ...` so OpenBench reads progress;
- bestmove from `searchRoot` with deterministic temperature=0;
- exit cleanly on `quit`.

Cross-test against `cutechess-cli` locally before submitting to OpenBench.

---

## ~1 week (project-direction items)

### Decision log
File: `docs/decisions.md` (new)

```markdown
# Tiny Leela decisions log

## 2026-05-09 Muon parked as default optimizer
Tried late-stage Muon switch from AdamW e2 baseline (1M smoke).
AdamW continuation kept slightly better policy CE/top1.
**Decision:** AdamW remains default; `--optimizer muon` flag retained but undocumented.
**Reopen if:** A real plateau on a longer run motivates a separate Muon LR sweep.

## 2026-05-09 MoveFormer parked
Park marker: `artifacts/moveformer_10m_supervised_mf64x6_e3/PARK_MOVEFORMER`.
**Decision:** No new MoveFormer queue work until CNN-AV/SquareFormer-AV gates resolve.
**Reopen if:** Tactical-MoveFormer K128 sidecar shows ≥ +30 Elo at fixed budget.
```

### Triage doc template
Top of `docs/deepresearch_architecture_triage_2026-05.md` and `docs/unsloth_rl_economics_triage_2026-05.md`:

```markdown
# Triage summary

| Field | Value |
| --- | --- |
| Verdict | NO CHANGE |
| Adopted | (one bullet each) |
| Rejected | (one bullet each) |
| Open question carried into roadmap | (single sentence) |
```

Everything below the table is reference material.

### Cap the four "active" architecture roadmaps at one canonical doc
- Make `unified_squareformer_architecture_roadmap.md` the canonical doc.
- Replace `transformer_model_roadmap.md`, `squareformer_v2_v3_implementation_plan.md`, `small_bt4_progression.md` with one-line stubs that say *"Merged into `unified_squareformer_architecture_roadmap.md` on 2026-05-DD."*
- Or invert: keep the small ones, demote the large unified to reference.

Either way: one canonical, the rest stub. Today there are four parallel canonicals, each ~1000 lines.

---

## Optional / lower priority

### Move Rust artifacts/build outputs out of repo
The `rust/tiny_leela_core/target/` is gitignored, but the audit found wasm-build files at `rust/tiny_leela_core/target/wasm32-unknown-unknown/release/build/serde-.../out/private.rs`. Make sure `cargo clean` is part of release prep.

### `eval/lint_eval_protocols.py`
This script exists. Run it from CI. If it fails today, add an allowlist; the goal is to make new arena writers without protocol cards loud.

### `tests/chess_rules.test.mjs` is fragile
The test uses `execFileSync('node', ['--experimental-strip-types', '--input-type=module', '-e', source])`. This works but means the test runs in a subprocess per `it()`. Slow and easy to break (escaping, etc.). Consider rewriting as direct imports — the `import` syntax is now first-class in node tests.

### `dovetail.*` files at repo root
If still in active use, document where. If not, move under `archive/dovetail/` or add to `.gitignore`.
