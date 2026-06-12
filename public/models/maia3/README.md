# Maia3 model assets

Staged from the CSSLab Maia3 browser frontend work (clean checkouts:
`npm run maia3:stage-assets`).

- `maia3_simplified.qdq8.onnx` — **default browser model**: locally derived
  weight-only int8 QDQ quantization (28.1MB; parity-gated against fp16).
- `maia3_simplified.onnx` — upstream fp16 file, byte-identical to
  https://github.com/CSSLab/maia-platform-frontend at pinned commit
  `0013cc8e6ec52c88f5b3d694781d4cc8427cb91a`; runtime fallback when the QDQ
  artifact is absent.
- Maia3 project / UCI engine: https://github.com/CSSLab/maia3
- License note: Maia3 is AGPL-3.0; the platform frontend is GPL-3.0. Preserve upstream notices and provide corresponding source for distributed covered components and local modifications (the QDQ file is such a modification; its derivation recipe is in the provenance doc).
- Local provenance + source offer: `docs/model_provenance/maia3.md`
