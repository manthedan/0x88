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
- **Local modifications: none** — the staged file is byte-identical to the
  upstream file at the pinned commit (the staging script refuses any other
  hash).

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
  (evaluator, worker, Play UI) around the unmodified model; the model itself
  is not altered.
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
- `npm run maia3:browser-smoke` runs a real browser/ORT worker smoke over
  normal, mirrored-black, castling, en-passant, promotion, checkmate, and
  stalemate positions; it records top-5 legal human-policy moves and WDL
  probabilities to an optional JSON artifact.

## Integration policy

- Maia3 is a policy/value human model. Do not route it through Lc0 PUCT by
  default.
- Play-page Maia3 modes should be explicit model-choice modes: deterministic
  argmax or human-style sampling (`temperature`, `topP`).
- Any future Maia3+search mode must be labeled experimental and less authentic
  than model-only Maia play.
