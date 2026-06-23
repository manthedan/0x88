<script lang="ts">
  import { onMount } from 'svelte';
  import { registerServiceWorker } from '$lib/pwa/register';
  import { theme, boardStyle, syncThemeToDom } from '$lib/stores/theme';

  onMount(() => {
    registerServiceWorker();
    const stop = syncThemeToDom();
    // Sync stores with any DOM state set by the early FOUC-prevention script
    const root = document.documentElement;
    if (root.dataset.theme === 'dark' || root.dataset.theme === 'light') {
      theme.set(root.dataset.theme);
    }
    if (root.dataset.board) {
      boardStyle.set(root.dataset.board as 'night' | 'blue' | 'green');
    }
    return stop;
  });
</script>

<slot />
