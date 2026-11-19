<script lang="ts">
// Global styles go through Vite so they ship content-hashed under /_app/immutable/ and Kit links
// them render-blocking in every prerendered page. Cascade order matters: Chessground base,
// board, pieces, then the app shell (and component styles after that).
import '../styles/chessground.base.css';
import '../styles/chessground.brown.css';
import 'chessground/assets/chessground.cburnett.css';
import '../styles/app-shell.css';
import { onMount } from 'svelte';
import { registerServiceWorker } from '#lib/pwa/register.ts';
import { boardStyle, syncThemeToDom, theme } from '#lib/stores/theme.ts';

onMount(() => {
  registerServiceWorker();
  // Hydration-complete signal for automated walks (scripts/feature_walk.mjs); prerendered pages look ready earlier.
  document.documentElement.dataset.hydrated = '';
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
