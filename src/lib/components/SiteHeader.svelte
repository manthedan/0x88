<script lang="ts">
  import { theme, boardStyle, toggleTheme, type BoardStyle } from '$lib/stores/theme';

  export let pageTitle = '';

  $: isDark = $theme === 'dark';

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
      <a href="/" class:active={pageTitle === ''}>Home</a>
      <a href="/app/play/" class:active={pageTitle === 'Play'}>Play</a>
      <a href="/app/analysis/" class:active={pageTitle === 'Analysis'}>Analysis</a>
      <a href="/app/arena/" class:active={pageTitle === 'Arena'}>Arena</a>
      <a href="/docs/" class:active={pageTitle === 'Docs'}>Docs</a>
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
