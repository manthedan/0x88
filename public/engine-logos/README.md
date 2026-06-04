# Engine logos (drop-in)

Small (favicon-sized) engine marks shown next to engine names and in the eval-bar
chips. The UI loads these by fixed filename and **falls back to the text chip**
(e.g. "Lc0", "SF") when a file is missing — so the app looks unchanged until you
add them.

Expected files (PNG or SVG; ~16–64px square, transparent background preferred):

| File             | Engine            | Used for                          |
|------------------|-------------------|-----------------------------------|
| `lc0.png`        | Lc0 (and Lc0 BT4) | Lc0 / Lc0 BT4 chips + names        |
| `stockfish.png`  | Stockfish         | Stockfish Lite + Stockfish         |
| `reckless.png`   | Reckless          | Reckless                           |

To use SVG instead, save as `lc0.svg` etc. and update the extension in
`logoUrlForEngine()` (src/lc0/arenaBrowser.ts) / the analysis equivalent.

## Sourcing (official marks — not committed here)

These are project **trademarks**, separate from the engines' code licenses. Using
them to identify the actual engines in the UI is normally nominative use, but don't
imply endorsement or alter them; check each project's branding guidance.

- **Lc0 / Leela Chess Zero** — lczero.org / github.com/LeelaChessZero (GPLv3 code).
- **Stockfish** — stockfishchess.org / github.com/official-stockfish/Stockfish (GPLv3).
- **Reckless** — github.com/codedeliveryservice/Reckless (AGPL-3.0).

Like the engine `*.wasm` binaries, logo files are intentionally not committed.
