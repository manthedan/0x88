# Maia3 browser model provenance

This project stages Maia3 as a separate human-move-prediction engine, not as an
Lc0 network and not as a PUCT/search default.

## Asset

- Staged URL: `/models/maia3/maia3_simplified.onnx`
- Local staging symlink: `public/models/maia3/maia3_simplified.onnx`
- Source repository: <https://github.com/CSSLab/maia-platform-frontend>
- Model project / UCI engine: <https://github.com/CSSLab/maia3>
- Source path observed: `public/maia3/maia3_simplified.onnx`
- Size: `45,683,686` bytes
- SHA-256: `405bf76c15727dad8728b352c06a8f3c1b80fb2760e8d666b32485c63d75b856`

## Licensing / compliance notes

- `CSSLab/maia3` is AGPL-3.0 licensed.
- `CSSLab/maia-platform-frontend` is GPL-3.0 licensed.
- Keep Maia3 integration modular and preserve provenance. If this repository or
  a hosted build distributes Maia3-derived model/runtime assets, preserve the
  upstream notices and provide corresponding source for the covered components
  and any local modifications.

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
