import { writable } from 'svelte/store';

export type ThemeMode = 'light' | 'dark';
export type BoardStyle = 'brown' | 'night' | 'blue' | 'green';

const THEME_KEY = '0x88-theme';
const BOARD_KEY = '0x88-board';
const BOARDS: BoardStyle[] = ['brown', 'night', 'blue', 'green'];

function prefersDark(): boolean {
  try {
    return matchMedia('(prefers-color-scheme: dark)').matches;
  } catch {
    return false;
  }
}

function initialTheme(): ThemeMode {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
  } catch {}
  return prefersDark() ? 'dark' : 'light';
}

function initialBoard(): BoardStyle {
  try {
    const saved = localStorage.getItem(BOARD_KEY);
    if (saved && BOARDS.includes(saved as BoardStyle)) return saved as BoardStyle;
  } catch {}
  return 'brown';
}

export const theme = writable<ThemeMode>(initialTheme());
export const boardStyle = writable<BoardStyle>(initialBoard());

export function toggleTheme(): void {
  theme.update((t) => (t === 'dark' ? 'light' : 'dark'));
}

function applyTheme(mode: ThemeMode): void {
  const root = document.documentElement;
  root.dataset.theme = mode;
  try {
    localStorage.setItem(THEME_KEY, mode);
  } catch {}
}

function applyBoard(style: BoardStyle): void {
  const root = document.documentElement;
  if (style === 'brown') delete root.dataset.board;
  else root.dataset.board = style;
  try {
    localStorage.setItem(BOARD_KEY, style);
  } catch {}
}

export function syncThemeToDom(): () => void {
  const unsubTheme = theme.subscribe((v) => applyTheme(v));
  const unsubBoard = boardStyle.subscribe((v) => applyBoard(v));

  try {
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      const saved = localStorage.getItem(THEME_KEY);
      if (!saved) theme.set(prefersDark() ? 'dark' : 'light');
    });
  } catch {}

  return () => { unsubTheme(); unsubBoard(); };
}
