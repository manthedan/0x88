<script lang="ts">
  import { page } from '$app/stores';
  import { theme, boardStyle, toggleTheme, type BoardStyle } from '$lib/stores/theme';

  export let pageTitle = '';
  let mobileOpen = false;
  let mobileNavEl: HTMLElement | null = null;

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

  function onWindowClick(event: MouseEvent) {
    if (!mobileOpen) return;
    if (mobileNavEl && event.target instanceof Node && mobileNavEl.contains(event.target)) return;
    mobileOpen = false;
  }

  function onWindowKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') mobileOpen = false;
  }
</script>

<svelte:window on:click={onWindowClick} on:keydown={onWindowKeydown} />

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
    <div class="mobile-nav" class:open={mobileOpen} bind:this={mobileNavEl}>
      <button class="menu-toggle" type="button" aria-label={mobileOpen ? 'Close navigation' : 'Open navigation'} title="Menu" aria-expanded={mobileOpen} on:click={() => mobileOpen = !mobileOpen}><span aria-hidden="true"></span></button>
      {#if mobileOpen}
        <nav class="mobile-nav-menu" aria-label="Mobile navigation">
          <a href="/" class:active={isActive('/')} aria-current={isActive('/') ? 'page' : undefined}>Home</a>
          <a href="/app/play/" class:active={isActive('/app/play/')} aria-current={isActive('/app/play/') ? 'page' : undefined}>Play</a>
          <a href="/app/analysis/" class:active={isActive('/app/analysis/')} aria-current={isActive('/app/analysis/') ? 'page' : undefined}>Analysis</a>
          <a href="/app/arena/" class:active={isActive('/app/arena/')} aria-current={isActive('/app/arena/') ? 'page' : undefined}>Arena</a>
          <a href="/docs/" class:active={isActive('/docs/')} aria-current={isActive('/docs/') ? 'page' : undefined}>Docs</a>
          <div class="mobile-board-field">
            <label for="mobileBoardSelect">Board style</label>
            <select id="mobileBoardSelect" value={$boardStyle} on:change={onBoardChange}>
              <option value="brown">Brown</option>
              <option value="night">Night</option>
              <option value="blue">Blue</option>
              <option value="green">Green</option>
            </select>
          </div>
        </nav>
      {/if}
    </div>
    <button class="theme-toggle" type="button" aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'} title={isDark ? 'Switch to light theme' : 'Switch to dark theme'} aria-pressed={isDark} on:click={toggleTheme}>{#if isDark}<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6L13 13M13 3l-1.4 1.4M4.4 11.6L3 13"/></svg>{:else}<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13.2 9.7A5.5 5.5 0 1 1 6.3 2.8a4.4 4.4 0 0 0 6.9 6.9z"/></svg>{/if}</button>
    <select class="board-style-select" aria-label="Board style" title="Board style" value={$boardStyle} on:change={onBoardChange}>
      <option value="brown">Brown</option>
      <option value="night">Night</option>
      <option value="blue">Blue</option>
      <option value="green">Green</option>
    </select>
  </div>
</header>
