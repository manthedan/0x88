<script lang="ts">
import { onMount } from 'svelte';
import { isMemoryConstrainedDevice } from '../../lc0/deviceMemoryClass.ts';
import { PHONE_LARGE_NET_REASON, phoneLargeNetPreference } from '../../lc0/phoneLargeNetPolicy.ts';

let { showHint = true }: { showHint?: boolean } = $props();
let constrained = $state(false);
let enabled = $state(false);
onMount(() => {
  constrained = isMemoryConstrainedDevice();
  enabled = phoneLargeNetPreference.enabled();
  return phoneLargeNetPreference.subscribe(() => {
    enabled = phoneLargeNetPreference.enabled();
  });
});
</script>

<p id="phoneLargeNetFallback" class="phone-large-net-preference" role="status" hidden>{PHONE_LARGE_NET_REASON}</p>
{#if constrained && showHint}
  <p class="phone-large-net-preference" role="status">
    {#if enabled}
      BT4 and t3 on for this phone · <button type="button" onclick={() => phoneLargeNetPreference.setEnabled(false)}>Turn off</button>
    {:else}
      BT4 and t3 are off on phones (crash risk).
    {/if}
  </p>
{/if}

<style>
  .phone-large-net-preference { font-size: 12px; color: var(--muted, #aaa); line-height: 1.5; }
  button { font: inherit; padding: 4px 6px; text-decoration: underline; min-height: 36px; }
  :global(.large-net-consent) { box-sizing: border-box; width: min(460px, calc(100vw - 32px)); max-height: calc(100dvh - 32px); overflow: auto; border: 1px solid var(--border-input); border-radius: 12px; padding: 20px; background: var(--panel, #202026); color: var(--ink); box-shadow: 0 16px 64px #0008; }
  :global(.large-net-consent::backdrop) { background: #0009; }
  :global(.large-net-consent h2) { margin: 0 0 12px; font-size: 20px; }
  :global(.large-net-consent p) { line-height: 1.6; }
  :global(.large-net-consent-actions) { display: flex; justify-content: flex-end; gap: 12px; margin-top: 20px; }
  :global(.large-net-consent button) { min-height: 44px; padding: 8px 16px; }
</style>
