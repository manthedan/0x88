<script lang="ts">
  import { afterUpdate, onMount } from 'svelte';
  import { registerServiceWorker } from '$lib/pwa/register';

  onMount(() => {
    registerServiceWorker();
    window.__0x88SyncThemeControls?.();
  });

  afterUpdate(() => {
    window.__0x88SyncThemeControls?.();
  });
</script>

<slot />
