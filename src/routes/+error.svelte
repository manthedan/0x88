<script lang="ts">
import SiteHeader from '#lib/components/SiteHeader.svelte';
import { page } from '$app/state';

const notFound = $derived(page.status === 404);
const heading = $derived(notFound ? 'Page not found' : 'Something went wrong');
</script>

<svelte:head>
  <title>{heading} — 0x88 Chess</title>
  <meta name="robots" content="noindex" />
</svelte:head>

<!-- Leave by full page load: this document may be the host's 404 response, so a workspace
     reached from it must get its own cross-origin isolation headers, not inherit these. -->
<div class="error-page" data-sveltekit-reload>
  <SiteHeader />
  <main id="main" class="wrap">
    <p class="status">{page.status}</p>
    <h1>{heading}</h1>
    {#if notFound}
      <p>There is no page at <code>{page.url.pathname}</code>.</p>
    {:else if page.error?.message}
      <p>{page.error.message}</p>
    {/if}
    <p class="links"><a href="/">Home</a> <a href="/app/play/">Play</a> <a href="/docs/">Docs</a></p>
  </main>
</div>

<style>
  .error-page{display:contents}
  main{padding-top:64px; padding-bottom:64px}
  .status{margin:0 0 8px; font-family:var(--mono); font-size:12px; letter-spacing:.09em; color:var(--accent-soft-text); font-weight:650}
  h1{margin:0 0 12px; font-size:clamp(32px,4vw,44px); line-height:1.05}
  p{color:var(--ink-soft); line-height:1.6}
  code{font-family:var(--mono); font-size:.9em; overflow-wrap:anywhere}
  .links{display:flex; gap:20px; margin-top:24px}
  .links a{color:var(--ink); font-weight:650}
</style>
