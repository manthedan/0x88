// Shared board UX for the play / arena / analysis pages so the three boards
// behave identically: live check highlighting and a lichess-style in-board
// promotion overlay (instead of auto-queening or detached buttons).
import { squareName, type BoardState, type Color } from '../chess/board.ts';
import { inCheck } from '../chess/movegen.ts';
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
  sorted.forEach((move, i) => {
    const tile = document.createElement('button');
    tile.className = 'promo-tile';
    tile.style.left = `${col * 12.5}%`;
    tile.style.top = `${(startRow + i * dir) * 12.5}%`;
    tile.dataset.promo = move.promotion ?? 'q';
    const piece = document.createElement('piece');
    piece.className = `${options.color === 'w' ? 'white' : 'black'} ${PROMO_PIECE_CLASS[move.promotion ?? 'q']}`;
    tile.appendChild(piece);
    tile.addEventListener('click', (event) => {
      event.stopPropagation();
      overlay.remove();
      options.onPick(move);
    });
    overlay.appendChild(tile);
  });
  overlay.addEventListener('click', () => {
    overlay.remove();
    options.onCancel();
  });
  host.appendChild(overlay);
}

export function hidePromotionOverlay(boardContainer: HTMLElement): void {
  boardContainer.querySelector('.promotion-overlay')?.remove();
}
