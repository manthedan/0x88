<script lang="ts">
import { onMount } from 'svelte';
import { registerServiceWorker } from '#lib/pwa/register.ts';
import { boardStyle, syncThemeToDom, theme } from '#lib/stores/theme.ts';

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
  if (new URLSearchParams(location.search).get('memprobe') === '1') {
    void import('#lib/memprobe.ts').then((module) => module.mountMemprobe());
  }
  return stop;
});
</script>

<slot />
