// Shared board UX for the play / arena / analysis pages so the three boards
// behave identically: live check highlighting and a lichess-style in-board
// promotion overlay (instead of auto-queening or detached buttons).
import type { Key } from 'chessground/types';
import { squareName, type BoardState, type Color } from '../chess/board.ts';
import { inCheck, legalMoves } from '../chess/movegen.ts';
import type { Move } from '../chess/moveCodec.ts';

/**
 * Chessground `check` config value for a position: chessground only renders
 * the check highlight when this is set (highlight.check alone enables the
 * style, not the state). `true` highlights the king of `turnColor`, which is
 * always the side to move in our usage.
 */
export function boardCheck(board: BoardState): boolean {
  return inCheck(board, board.turn);
}

/** Chessground movable.dests map for the side to move. */
export function legalDests(board: BoardState): Map<Key, Key[]> {
  const dests = new Map<Key, Key[]>();
  for (const move of legalMoves(board)) {
    const from = squareName(move.from) as Key;
    dests.set(from, [...(dests.get(from) ?? []), squareName(move.to) as Key]);
  }
  return dests;
}

/**
 * All legal moves matching a chessground drag/click (from, to). Length 0 =
 * illegal; 1 = play it; 4 = a promotion, show the overlay and let the user
 * pick (never auto-queen).
 */
export function matchUserMoves(board: BoardState, from: Key, to: Key): Move[] {
  return legalMoves(board).filter((move) => squareName(move.from) === from && squareName(move.to) === to);
}

const PROMO_PIECE_CLASS: Record<string, string> = { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' };

let overlayStylesInjected = false;
function injectOverlayStyles(): void {
  if (overlayStylesInjected) return;
  overlayStylesInjected = true;
  const style = document.createElement('style');
  style.textContent = `
.promotion-overlay{position:absolute;inset:0;z-index:20;background:rgba(0,0,0,0.35)}
.promotion-overlay .promo-tile{position:absolute;width:12.5%;height:12.5%;background:#f0f0f0;border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,0.5);cursor:pointer;padding:0;border:0}
.promotion-overlay .promo-tile:hover{background:#ffd966;border-radius:12%}
.promotion-overlay .promo-tile:focus-visible{outline:3px solid #5a6e2a;outline-offset:2px}
.promotion-overlay .promo-tile piece{display:block;width:100%;height:100%;background-size:cover}
`;
  document.head.appendChild(style);
}

export interface PromotionOverlayOptions {
  /** The element the Chessground board was mounted into. */
  boardContainer: HTMLElement;
  orientation: 'white' | 'black';
  /** Side that is promoting. */
  color: Color;
  /** All matching promotion moves for the drag (same from/to, 4 pieces). */
  choices: Move[];
  onPick: (move: Move) => void;
  onCancel: () => void;
}

/**
 * Lichess-style promotion picker: piece tiles rendered over the promotion
 * file, starting at the target square and extending toward the middle of the
 * board. Clicking anywhere else cancels. Uses chessground's own piece sprite
 * classes, so it must be appended inside the .cg-wrap element.
 */
export function showPromotionOverlay(options: PromotionOverlayOptions): void {
  injectOverlayStyles();
  const host = options.boardContainer.querySelector<HTMLElement>('.cg-wrap') ?? options.boardContainer;
  host.querySelector('.promotion-overlay')?.remove();
  if (!options.choices.length) return;

  const overlay = document.createElement('div');
  overlay.className = 'promotion-overlay';
  const target = squareName(options.choices[0].to);
  const fileIdx = target.charCodeAt(0) - 97;
  const rankIdx = Number(target[1]) - 1;
  const col = options.orientation === 'white' ? fileIdx : 7 - fileIdx;
  const startRow = options.orientation === 'white' ? 7 - rankIdx : rankIdx;
  const dir = startRow === 0 ? 1 : -1;

  // Stable piece order: queen first, then rook/bishop/knight.
  const order = ['q', 'r', 'b', 'n'];
  const sorted = [...options.choices].sort((a, b) => order.indexOf(a.promotion ?? 'q') - order.indexOf(b.promotion ?? 'q'));
  const tiles: HTMLButtonElement[] = [];
  sorted.forEach((move, i) => {
    const tile = document.createElement('button');
    tile.className = 'promo-tile';
    tile.style.left = `${col * 12.5}%`;
    tile.style.top = `${(startRow + i * dir) * 12.5}%`;
    tile.dataset.promo = move.promotion ?? 'q';
    tile.type = 'button';
    tile.setAttribute('aria-label', `Promote to ${PROMO_PIECE_CLASS[move.promotion ?? 'q']}`);
    const piece = document.createElement('piece');
    piece.className = `${options.color === 'w' ? 'white' : 'black'} ${PROMO_PIECE_CLASS[move.promotion ?? 'q']}`;
    piece.setAttribute('aria-hidden', 'true');
    tile.appendChild(piece);
    tile.addEventListener('click', (event) => {
      event.stopPropagation();
      overlay.remove();
      options.onPick(move);
    });
    overlay.appendChild(tile);
    tiles.push(tile);
  });

  // Keyboard: Escape cancels, Up/Down cycle through tiles in display order.
  // Tiles are placed along the promotion file; display order matches the
  // sorted array (index 0 = topmost tile shown).
  const onKeydown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      close();
      options.onCancel();
    } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
      const current = tiles.indexOf(document.activeElement as HTMLButtonElement);
      const dirKey = event.key === 'ArrowDown' ? 1 : -1;
      const next = (current + dirKey + tiles.length) % tiles.length;
      tiles[next].focus();
    }
  };
  const close = (): void => {
    overlay.removeEventListener('keydown', onKeydown);
    overlay.remove();
  };
  overlay.addEventListener('keydown', onKeydown);
  overlay.tabIndex = -1;

  overlay.addEventListener('click', () => {
    close();
    options.onCancel();
  });
  host.appendChild(overlay);
  // Autofocus the queen tile so keyboard users can pick immediately.
  requestAnimationFrame(() => tiles[0]?.focus());
}

export function hidePromotionOverlay(boardContainer: HTMLElement): void {
  boardContainer.querySelector('.promotion-overlay')?.remove();
}
