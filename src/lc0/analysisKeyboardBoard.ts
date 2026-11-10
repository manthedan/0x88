import type { Key } from 'chessground/types';
import { type BoardState, boardToFen, squareIndex, squareName } from '../chess/board.ts';
import { legalDests } from './boardUx.ts';

const ROLES: Record<string, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };

export function analysisBoardSquares(board: BoardState, orientation: 'white' | 'black'): { key: Key; label: string }[] {
  return Array.from({ length: 64 }, (_, index) => {
    const file = orientation === 'white' ? index % 8 : 7 - (index % 8);
    const rank = orientation === 'white' ? 7 - Math.floor(index / 8) : Math.floor(index / 8);
    const square = rank * 8 + file;
    const piece = board.squares[square];
    return {
      key: squareName(square) as Key,
      label: `${squareName(square)}, ${piece ? `${piece[0] === 'w' ? 'White' : 'Black'} ${ROLES[piece[1]]}` : 'empty'}`,
    };
  });
}

/** A pointer-transparent, roving-focus grid over Chessground, using the same legal move path. */
export function createAnalysisKeyboardBoard(
  host: HTMLElement,
  options: {
    onSelect: (key: Key | null) => void;
    onMove: (from: Key, to: Key) => void;
    announce: (text: string) => void;
  },
) {
  let board: BoardState;
  let orientation: 'white' | 'black' = 'white';
  let fen = '';
  let selected: Key | null = null;
  let focused: Key = 'e2';
  host.setAttribute('role', 'grid');
  host.setAttribute('aria-label', 'Chessboard');
  host.setAttribute('aria-describedby', 'boardInstructions');
  host.setAttribute('aria-rowcount', '8');
  host.setAttribute('aria-colcount', '8');

  const focusSquare = (key: Key) => {
    focused = key;
    for (const cell of host.querySelectorAll<HTMLButtonElement>('[data-square]')) {
      cell.tabIndex = cell.dataset.square === key ? 0 : -1;
      if (cell.tabIndex === 0) cell.focus({ preventScroll: true });
    }
  };
  const render = (position: BoardState, facing: 'white' | 'black') => {
    const hadFocus = host.contains(document.activeElement);
    board = position;
    orientation = facing;
    const nextFen = boardToFen(position);
    if (nextFen !== fen) {
      selected = null;
      if (hadFocus) options.announce(`${board.turn === 'w' ? 'White' : 'Black'} to move.`);
    }
    fen = nextFen;
    const destinations = selected ? (legalDests(board).get(selected) ?? []) : [];
    const squares = analysisBoardSquares(board, orientation);
    host.innerHTML = Array.from(
      { length: 8 },
      (_, row) =>
        `<div role="row" class="keyboard-row">${squares
          .slice(row * 8, row * 8 + 8)
          .map(
            ({ key, label }, col) =>
              `<button type="button" role="gridcell" class="keyboard-square" data-square="${key}" tabindex="${key === focused ? 0 : -1}" aria-label="${label}${destinations.includes(key) ? ', legal destination' : ''}" aria-selected="${key === selected}" style="left:${col * 12.5}%;top:${row * 12.5}%"></button>`,
          )
          .join('')}</div>`,
    ).join('');
    if (hadFocus) focusSquare(focused);
  };
  const activate = (key: Key) => {
    const dests = legalDests(board);
    if (selected && dests.get(selected)?.includes(key)) {
      const from = selected;
      selected = null;
      focused = key;
      options.onSelect(null);
      options.announce(`${from} to ${key}.`);
      options.onMove(from, key);
      return;
    }
    if (dests.has(key)) {
      selected = selected === key ? null : key;
      options.onSelect(selected);
      options.announce(
        selected
          ? `${analysisBoardSquares(board, orientation).find((square) => square.key === key)!.label} selected. Legal destinations: ${dests.get(key)!.join(', ')}.`
          : 'Selection cancelled.',
      );
      render(board, orientation);
    } else {
      const piece = board.squares[squareIndex(key)];
      options.announce(
        `${key}: ${piece ? 'This piece cannot move here.' : 'Empty square.'}${selected ? ' Choose a legal destination or press Escape.' : ' Select a piece of the side to move.'}`,
      );
    }
  };
  const onKeydown = (event: KeyboardEvent) => {
    const cell = (event.target as HTMLElement).closest<HTMLElement>('[data-square]');
    if (!cell) return;
    const key = cell.dataset.square as Key;
    // Assistive technology can focus a non-tabstop square directly.
    focused = key;
    const squares = analysisBoardSquares(board, orientation);
    const index = squares.findIndex((square) => square.key === key);
    const row = Math.floor(index / 8),
      col = index % 8;
    let next = index;
    if (event.key === 'ArrowLeft') next = row * 8 + Math.max(0, col - 1);
    else if (event.key === 'ArrowRight') next = row * 8 + Math.min(7, col + 1);
    else if (event.key === 'ArrowUp') next = Math.max(0, row - 1) * 8 + col;
    else if (event.key === 'ArrowDown') next = Math.min(7, row + 1) * 8 + col;
    else if (event.key === 'Home') next = event.ctrlKey ? 0 : row * 8;
    else if (event.key === 'End') next = event.ctrlKey ? 63 : row * 8 + 7;
    else if (event.key === 'Enter' || event.key === ' ') activate(key);
    else if (event.key === 'Escape') {
      selected = null;
      options.onSelect(null);
      options.announce('Selection cancelled.');
      render(board, orientation);
    } else return;
    event.preventDefault();
    event.stopPropagation();
    if (next !== index) focusSquare(squares[next].key);
  };
  // Click supports assistive-technology activation; pointer events still pass to Chessground.
  const onClick = (event: MouseEvent) => {
    const cell = (event.target as HTMLElement).closest<HTMLElement>('[data-square]');
    if (cell) {
      focused = cell.dataset.square as Key;
      activate(focused);
    }
  };
  host.addEventListener('keydown', onKeydown);
  host.addEventListener('click', onClick);
  return {
    render,
    focus: focusSquare,
    destroy: () => {
      host.removeEventListener('keydown', onKeydown);
      host.removeEventListener('click', onClick);
      host.replaceChildren();
    },
  };
}
