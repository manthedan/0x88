<script lang="ts">
import { onMount } from 'svelte';
import { installBatchingStrengthPage } from '../../../lc0/batchingStrengthBrowser.ts';

onMount(() => {
  const api = installBatchingStrengthPage();
  const params = new URLSearchParams(location.search);
  if (params.get('autostart') === '1') void api.startFromParams(params);
});
</script>

<svelte:head>
  <title>Batching strength</title>
</svelte:head>

<main>
  <h1>Batching strength</h1>
  <p>Lab match. Product defaults are unchanged. Both arms share this page's search path; only the variant switch differs.</p>
  <pre id="bs-status">idle</pre>
  <pre id="bs-json">{'{}'}</pre>
</main>

<style>
  main {
    font: 16px/1.4 ui-sans-serif, system-ui, sans-serif;
    margin: 24px;
    max-width: 920px;
  }
  pre {
    white-space: pre-wrap;
    background: #111;
    color: #e8e8e8;
    padding: 12px;
    min-height: 2.5em;
  }
</style>
