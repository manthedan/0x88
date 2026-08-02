<script lang="ts">
import { primaryNav } from '$lib/nav';
</script>

<slot />
<footer class="site-footer">
  <div class="wrap">
    <span>Chess engines run locally. Source is public.</span>
    <span class="footer-links">
      {#each primaryNav as item (item.href)}
        <a href={item.href}>{item.label}</a>
      {/each}
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>
