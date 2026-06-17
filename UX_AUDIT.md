# UX Audit — leelaweb frontends

Findings from a UX audit of the four user-facing pages (Home, Play, Analysis,
Arena) plus supporting TS modules.

## P0 — Correctness / blocking UX

### 1. Play: no "New game" prompt when changing engine/color mid-game
`newGame()` only fires on button click. Switching engine, level, or color via
dropdowns only resets state silently if no moves have been played. Once a
single move is made, changing engine/color has no visible effect until the
user clicks "New game". Fix: show an inline confirm bar or auto-start a new
game with undo via history.

### 2. Play: board does not flip on color change in dropdown
`orientation` is only recomputed inside `newGame()`. Picking "You play black"
in the dropdown before move one does not flip the board. Fix: set orientation
from colorSelect in the engineSelect change handler when `!moves.length`.

### 3. Play/Analysis: promotion picker has no keyboard support
`boardUx.ts` builds tile buttons but they are not focus-trapped, there is no
Escape-to-cancel binding, and no autofocus. Fix: autofocus queen tile, bind
Escape and arrow keys.

### 4. All pages: no `:focus-visible` styles defined
Button/select/input resets inherit browser defaults which are invisible on
macOS Safari/FF. Add a shared focus ring rule.

### 5. Home: `navigator.storage.estimate?.().catch` crashes when estimate is missing
If `estimate` is undefined, `?.catch` throws. Wrap with `Promise.resolve(...)`.

## P1 — Important UX

### 6. Play: no WakeLock during engine thinking
BT4 net can think for several seconds; laptops that sleep abandon the move.
Request a screen wake lock in engineTurn and release in finally.

### 7. No `prefers-reduced-motion` respect
Chessground animation, eval-bar transitions, hover shadows all run
unconditionally. Gate them behind the media query.

### 8. Play/Analysis: "Copy PGN" gives no feedback
Add a transient "Copied" label that reverts after ~1.5s.

### 9. Analysis: right panel is overwhelmingly dense
Cram comparison, lines, Maia3, move list, review, and opening explorer into
one scroll. Consider accordions or a two-tab layout.

### 10. Analysis: FEN input has no Enter handler
PGN username and profile name bind Enter; fenInput does not. Add same handler.

### 11. Arena: hidden inputs are still in tab order
`<div hidden>` contains real inputs for visits/depth. Use `inert` or
`disabled` to remove from a11y tree.

### 12. Play: resign has no confirm and no undo
Add double-click-to-confirm or an undo toast.

## P2 — Polish

### 13. Home cards: "→" is read by screen readers
Wrap decorative arrows in `aria-hidden="true"`.

### 14. Home: cache "Clear" button has no confirm
Big net deletion is one click away from a multi-minute re-download. Confirm
with size shown.

### 15. Play/Arena: board coordinates cannot be toggled
Add a toggle for streamers/screenshots.

### 16. Arena/Play: status regions have no aria-live
`#matchScore`, `#message`, Play's `#status` need `aria-live="polite"`.

### 17. No navigator.vibrate on mobile move
Cheap opt-in haptic feedback for human move and game-over.

### 18. Home footer link "LC0 single-engine console" is unclear
Relabel to "Developer / single-engine console".

### 19. Shared header duplicated across four HTML files
Extract to a small header.ts; also lets you set .active from current URL.

### 20. Play: engineNote uses monospace for human-readable errors
Split into a status line (mono) and an error box (sans).
