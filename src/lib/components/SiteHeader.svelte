<script lang="ts">
  import { page } from '$app/stores';
  import { theme, boardStyle, toggleTheme, type BoardStyle } from '$lib/stores/theme';

  export let pageTitle = '';

  $: path = $page.url.pathname;
  $: isDark = $theme === 'dark';

  function isActive(href: string): boolean {
    if (href === '/') return path === '/';
    const base = href.replace(/\/$/, '');
    return path === base || path.startsWith(`${base}/`);
  }

  function onBoardChange(event: Event) {
    boardStyle.set((event.target as HTMLSelectElement).value as BoardStyle);
  }
</script>

<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <div class="wrap">
    <a class="brand" href="/" aria-label="0x88.app home">
      <span class="brand-mark" aria-hidden="true">0x88</span>
      <span class="brand-name">0x88.app</span>
    </a>
    {#if pageTitle}
      <span class="page-title">{pageTitle}</span>
    {/if}
    <nav class="primary" aria-label="Primary">
      <a href="/" class:active={isActive('/')} aria-current={isActive('/') ? 'page' : undefined}>Home</a>
      <a href="/app/play/" class:active={isActive('/app/play/')} aria-current={isActive('/app/play/') ? 'page' : undefined}>Play</a>
      <a href="/app/analysis/" class:active={isActive('/app/analysis/')} aria-current={isActive('/app/analysis/') ? 'page' : undefined}>Analysis</a>
      <a href="/app/arena/" class:active={isActive('/app/arena/')} aria-current={isActive('/app/arena/') ? 'page' : undefined}>Arena</a>
      <a href="/docs/" class:active={isActive('/docs/')} aria-current={isActive('/docs/') ? 'page' : undefined}>Docs</a>
    </nav>
    <button class="theme-toggle" type="button" aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'} title={isDark ? 'Switch to light theme' : 'Switch to dark theme'} aria-pressed={isDark} on:click={toggleTheme}>{isDark ? '\u2600' : '\u263E'}</button>
    <select class="board-style-select" aria-label="Board style" title="Board style" value={$boardStyle} on:change={onBoardChange}>
      <option value="brown">Brown</option>
      <option value="night">Night</option>
      <option value="blue">Blue</option>
      <option value="green">Green</option>
    </select>
  </div>
</header>
