# Maia3 browser model provenance

This project stages Maia3 as a separate human-move-prediction engine, not as an
Lc0 network and not as a PUCT/search default.

## Asset

- Staged URL: `/models/maia3/maia3_simplified.onnx`
- Local staging symlink: `public/models/maia3/maia3_simplified.onnx`
  (clean checkouts: `npm run maia3:stage-assets` downloads + verifies it)
- Source repository: <https://github.com/CSSLab/maia-platform-frontend>
- Model project / UCI engine: <https://github.com/CSSLab/maia3>
- Source path: `public/maia3/maia3_simplified.onnx` at pinned commit
  `0013cc8e6ec52c88f5b3d694781d4cc8427cb91a`
  ("Switch to fp16 ONNX model (87MB → 44MB)", 2026-03-27)
- Size: `45,683,686` bytes
- SHA-256: `405bf76c15727dad8728b352c06a8f3c1b80fb2760e8d666b32485c63d75b856`
- The staged fp16 file is byte-identical to the upstream file at the pinned
  commit (the staging script refuses any other hash).

## Derived artifact: maia3_simplified.qdq8.onnx (the default browser model)

Promoted 2026-06-11 as `MAIA3_DEFAULT_MODEL_URL` (the browser loader falls
back to the fp16 file when the QDQ artifact is not staged).

- **This is a local modification of the upstream model**: weight-only int8
  quantization of the fp16 file. Derivation (reproduced by
  `npm run maia3:stage-assets` when missing):
  1. opset 17 → 21 via `onnx.version_converter` (Maia3 uses old-style
     `ReduceMean` axes; DequantizeLinear with f16 scales needs opset ≥ 19);
  2. `scripts/lc0_quantize_onnx_weights_qdq.py` — MatMul B initializers
     > 4096 elements stored per-output-channel int8 + f16 scales with
     in-graph DequantizeLinear (43 tensors; compute stays f16).
- Size: `28,138,813` bytes (0.61× of fp16)
- SHA-256: `4141faeca4b0aa9f99073d70abb43211264ed2eb04cb96214ba85a86dd455a10`
- Quantization gates (2026-06-11): weight relRMS ≤ 1.3%; output parity vs
  fp16 over 40 positions × 4 elo conditions: 159/160 top-1 agreement, prior
  and value-probability drift ≤ 0.021. Speed neutral-or-better on both EPs
  (WebGPU 23.8 vs 30.4 ms/eval steady; wasm 136.7 vs 146.9).

## Licensing / compliance notes

- `CSSLab/maia3` is AGPL-3.0 licensed (SPDX confirmed via the GitHub license
  API, 2026-06-11).
- `CSSLab/maia-platform-frontend` is GPL-3.0 licensed (likewise confirmed).
- Keep Maia3 integration modular and preserve provenance. If this repository or
  a hosted build distributes Maia3-derived model/runtime assets, preserve the
  upstream notices and provide corresponding source for the covered components
  and any local modifications.

## Source offer (for hosted/distributed builds)

If a deployed build serves `/models/maia3/maia3_simplified.onnx`:

- The model file is a byte-identical copy of
  `CSSLab/maia-platform-frontend@0013cc8e/public/maia3/maia3_simplified.onnx`
  (GPL-3.0); model training/engine source lives at
  <https://github.com/CSSLab/maia3> (AGPL-3.0).
- Local modifications statement: this repository adds a browser ORT runtime
  (evaluator, worker, Play UI), and serves a locally derived int8-QDQ
  quantization of the model as the default (see "Derived artifact" above for
  the exact derivation recipe); the upstream fp16 file is also staged,
  byte-identical, and used as the runtime fallback.
- A deployed site must surface this provenance document (or an equivalent
  notice) and the upstream repository links from its licensing/about page,
  through the same mechanism as the existing GPL engine corresponding-source
  offers (Stockfish/Berserk/PlentyChess).

## Verification hooks

- `npm run maia3:check-assets` verifies the staged symlink target, byte length,
  SHA-256, manifest, and provenance-notice tokens.
- `npm run maia3:upstream-move-map-parity -- --upstream-dir /path/to/maia-platform-frontend`
  compares the local algorithmic 4352-move indexer against upstream
  `all_moves_maia3*.json` without vendoring those JSON files.
- `npm run maia3:upstream-tensor-parity` compares our (64, 12) board-token
  encoding (including the mirror-to-white transform) against upstream
  `tensor.ts` extracted from the pinned commit, across handcrafted edge cases
  and random playouts.
- `npm run maia3:browser-smoke` runs a real browser/ORT worker smoke over
  normal, mirrored-black, castling, en-passant, promotion, checkmate, and
  stalemate positions; it records top-5 legal human-policy moves and WDL
  probabilities to an optional JSON artifact.

## Upstream output-parity audit (2026-06-11)

Because the staged model is byte-identical to upstream, output parity reduces
to input/post-processing parity. Status: **covered**.

- Board tokens: `maia3:upstream-tensor-parity` — 370 positions, 0 mismatches.
- Move-index map: `maia3:upstream-move-map-parity` (existing).
- Elo inputs: both sides pass **raw floats** as `[1]`-shaped float32 tensors
  named `elo_self`/`elo_oppo` (no normalization, no categories) — verified by
  inspection of upstream `maia.ts` `evaluateMaia3` vs our `maia3Worker.ts`.
- Outputs: both read `logits_move`/`logits_value`; our legal-move softmax is
  identical to upstream at temperature 1. `logits_value` channel order is
  **[Loss, Draw, Win] for the side to move** (the reverse of the LC0 [W,D,L]
  convention used elsewhere in this repo) — documented on `Maia3Evaluation`,
  with `maia3WinProbability` reproducing upstream's white-perspective score.
- Lifecycle: 25 create/evaluate/dispose worker cycles green (1 cache miss +
  24 Cache API hits, sha256 valid on all loads, no browser errors).

## Regression gating decision (2026-06-11)

- `npm run maia3:gate` (= `maia3:check-assets` + `maia3:upstream-tensor-parity`)
  is the pre-merge/productization gate for Maia3 changes.
- It is intentionally NOT part of `npm test`: the asset check fails on clean
  checkouts without the staged model, and the repo already has one
  missing-artifact `npm test` failure mode we do not want to add to.
- `maia3:upstream-move-map-parity` needs a local upstream checkout
  (`--upstream-dir`) and `maia3:browser-smoke` needs a browser + the 45MB
  model; both stay manual/release checks.

## Integration policy

- Maia3 is a policy/value human model. Do not route it through Lc0 PUCT by
  default.
- Play-page Maia3 modes should be explicit model-choice modes: deterministic
  argmax or human-style sampling (`temperature`, `topP`).
- Any future Maia3+search mode must be labeled experimental and less authentic
  than model-only Maia play.
