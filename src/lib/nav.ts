export type NavItem = {
  href: string;
  label: string;
};

/** The three workspaces, in the order shared header and footer links present them. */
const workspaces: NavItem[] = [
  {
    href: '/app/analysis/',
    label: 'Analysis',
  },
  {
    href: '/app/arena/',
    label: 'Arena',
  },
  {
    href: '/app/play/',
    label: 'Play',
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
