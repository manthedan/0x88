import { writable } from 'svelte/store';

export type ThemeMode = 'light' | 'dark';
export type BoardStyle = 'brown' | 'night' | 'blue' | 'green';

const THEME_KEY = '0x88-theme';
const BOARD_KEY = '0x88-board';
const BOARDS: BoardStyle[] = ['brown', 'night', 'blue', 'green'];
let explicitTheme: ThemeMode | null = null;

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
    if (saved === 'dark' || saved === 'light') {
      explicitTheme = saved;
      return saved;
    }
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
  theme.update((t) => {
    const next = t === 'dark' ? 'light' : 'dark';
    explicitTheme = next;
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {}
    return next;
  });
}

function applyTheme(mode: ThemeMode): void {
  const root = document.documentElement;
  root.dataset.theme = mode;
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

  let media: MediaQueryList | undefined;
  const followSystem = () => {
    if (explicitTheme) return;
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(THEME_KEY);
    } catch {}
    if (saved !== 'light' && saved !== 'dark') theme.set(prefersDark() ? 'dark' : 'light');
  };
  try {
    media = matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', followSystem);
    followSystem();
  } catch {}

  return () => {
    media?.removeEventListener('change', followSystem);
    unsubTheme();
    unsubBoard();
  };
}
