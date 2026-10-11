// View-only move navigation for a linear game record (Play page). The game
// itself is never mutated: a "view ply" selects which recorded position the
// board shows, while the live game keeps advancing underneath it.
//
// View ply convention: ply N is the position after N half-moves (0 = start
// position). `null` means "following the live game": the board tracks the
// latest position and new moves auto-follow. A concrete number means the user
// is browsing; it stays put when new moves arrive (lichess behaviour).

export type MoveNavAction = 'first' | 'prev' | 'next' | 'last';

/** Clamp a ply into [0, totalPlies]; non-finite input falls back to totalPlies. */
export function clampPly(ply: number, totalPlies: number): number {
  const total = Math.max(0, Math.floor(totalPlies));
  if (!Number.isFinite(ply)) return total;
  return Math.max(0, Math.min(total, Math.floor(ply)));
}

/**
 * Normalize a stored view ply against the current game length: anything at or
 * past the live ply collapses to `null` (live), so reaching the end of the
 * record resumes auto-follow.
 */
export function normalizeViewPly(viewPly: number | null, totalPlies: number): number | null {
  if (viewPly === null) return null;
  const ply = clampPly(viewPly, totalPlies);
  return ply >= totalPlies ? null : ply;
}

/** The ply the board should show for a (possibly live) view ply. */
export function displayedPly(viewPly: number | null, totalPlies: number): number {
  const normalized = normalizeViewPly(viewPly, totalPlies);
  return normalized === null ? Math.max(0, Math.floor(totalPlies)) : normalized;
}

/** Jump straight to a ply (move-list click); the live ply returns `null`. */
export function viewPlyFor(targetPly: number, totalPlies: number): number | null {
  return normalizeViewPly(clampPly(targetPly, totalPlies), totalPlies);
}

/** Apply a navigation action and return the new view ply (`null` = live). */
export function navigateViewPly(viewPly: number | null, totalPlies: number, action: MoveNavAction): number | null {
  const current = displayedPly(viewPly, totalPlies);
  switch (action) {
    case 'first':
      return viewPlyFor(0, totalPlies);
    case 'prev':
      return viewPlyFor(current - 1, totalPlies);
    case 'next':
      return viewPlyFor(current + 1, totalPlies);
    case 'last':
      return null;
  }
}

/** Which navigation buttons can act from the current view. */
export function moveNavAvailability(viewPly: number | null, totalPlies: number): Record<MoveNavAction, boolean> {
  const current = displayedPly(viewPly, totalPlies);
  const total = Math.max(0, Math.floor(totalPlies));
  return { first: current > 0, prev: current > 0, next: current < total, last: current < total };
}

/**
 * Human label for a ply in a game that starts with White to move, e.g.
 * ply 3 "Nf3" -> "2. Nf3", ply 4 "Nc6" -> "2… Nc6", ply 0 -> start position.
 */
export function plyLabel(ply: number, san: string | undefined): string {
  if (ply <= 0 || !san) return 'the start position';
  const moveNumber = Math.ceil(ply / 2);
  return ply % 2 === 1 ? `${moveNumber}. ${san}` : `${moveNumber}… ${san}`;
}

export interface MoveNavKeyEvent {
  key: string;
  altKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
  target?: unknown;
}

/** Elements whose own keyboard handling must win over board navigation. */
const KEYBOARD_OWNER_SELECTOR = 'input,textarea,select,[contenteditable]:not([contenteditable="false"]),.promotion-overlay';

/** True when the key event's target is a form control or other keyboard owner. */
export function isKeyboardOwnerTarget(target: unknown): boolean {
  if (!target || typeof target !== 'object') return false;
  const node = target as { isContentEditable?: boolean; closest?: (selector: string) => unknown };
  if (node.isContentEditable) return true;
  return typeof node.closest === 'function' && !!node.closest(KEYBOARD_OWNER_SELECTOR);
}

/**
 * Map a keydown to a navigation action, following the analysis page
 * (ArrowLeft/Right step, ArrowUp/Down jump to start/end) plus Home/End.
 * Modified keys (Alt+Left is browser Back) and keys aimed at form controls
 * are ignored.
 */
export function moveNavActionForKey(event: MoveNavKeyEvent): MoveNavAction | null {
  if (event.altKey || event.ctrlKey || event.metaKey) return null;
  if (isKeyboardOwnerTarget(event.target)) return null;
  switch (event.key) {
    case 'ArrowLeft':
      return 'prev';
    case 'ArrowRight':
      return 'next';
    case 'ArrowUp':
    case 'Home':
      return 'first';
    case 'ArrowDown':
    case 'End':
      return 'last';
    default:
      return null;
  }
}
