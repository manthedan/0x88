<script lang="ts">
import { page } from '$app/stores';
import { primaryNav } from '$lib/nav';
import { type BoardStyle, boardStyle, theme, toggleTheme } from '$lib/stores/theme';

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
      {#each primaryNav as item (item.href)}
        <a href={item.href} class:active={isActive(item.href)} aria-current={isActive(item.href) ? 'page' : undefined}>{item.label}</a>
      {/each}
    </nav>
    <div class="mobile-nav" class:open={mobileOpen} bind:this={mobileNavEl}>
      <button class="menu-toggle" type="button" aria-label={mobileOpen ? 'Close navigation' : 'Open navigation'} title="Menu" aria-expanded={mobileOpen} on:click={() => mobileOpen = !mobileOpen}><span aria-hidden="true"></span></button>
      {#if mobileOpen}
        <nav class="mobile-nav-menu" aria-label="Mobile navigation">
          {#each primaryNav as item (item.href)}
            <a href={item.href} class:active={isActive(item.href)} aria-current={isActive(item.href) ? 'page' : undefined}>{item.label}</a>
          {/each}
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
    <button class="theme-toggle" type="button" aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'} title={isDark ? 'Switch to light theme' : 'Switch to dark theme'} aria-pressed={isDark} on:click={toggleTheme}>{#if isDark}<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4"/></svg>{:else}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/></svg>{/if}</button>
    <select class="board-style-select" aria-label="Board style" title="Board style" value={$boardStyle} on:change={onBoardChange}>
      <option value="brown">Brown</option>
      <option value="night">Night</option>
      <option value="blue">Blue</option>
      <option value="green">Green</option>
    </select>
  </div>
</header>
