export type NavItem = {
  href: string;
  label: string;
};

export type Workspace = NavItem & {
  /** Heading shown on the landing mode index. */
  title: string;
  /** Body copy shown on the landing mode index. */
  blurb: string;
};

/**
 * The three workspaces, in the order every nav surface presents them.
 * Reorder here and the header, both footers, the landing index and the
 * docs TOC all follow. Landing copy lives here too so the index and the
 * nav can never disagree about what a page is called.
 */
export const workspaces: Workspace[] = [
  {
    href: '/app/analysis/',
    label: 'Analysis',
    title: 'Position analysis',
    blurb:
      'Run Lc0, Stockfish, and six other top engines on one position, many of them in a browser for the first time. Compare their evaluations side by side.',
  },
  {
    href: '/app/arena/',
    label: 'Arena',
    title: 'Engine tournaments',
    blurb: 'Run matches or larger tournaments under shared time controls. The arena records each game and updates the standings as it plays.',
  },
  {
    href: '/app/play/',
    label: 'Play',
    title: 'Play with contempt!',
    blurb: 'Specialized engines such as Maia3 and Leela Queen Odds emulate or exploit human playstyles.',
  },
];

/** Header nav and the site footers: workspaces bracketed by Home and Docs. */
export const primaryNav: NavItem[] = [
  { href: '/', label: 'Home' },
  ...workspaces.map(({ href, label }) => ({ href, label })),
  { href: '/docs/', label: 'Docs' },
];

/** Footer link rows omit Home; the wordmark already links there. */
export const footerNav: NavItem[] = primaryNav.filter((item) => item.href !== '/');
