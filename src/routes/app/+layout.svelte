<script lang="ts">
  import SiteHeader from '$lib/components/SiteHeader.svelte';
</script>

<SiteHeader pageTitle="App" />
<slot />
<footer class="site-footer">
  <div class="wrap">
    <span>Engines run entirely in your browser. Nothing is sent to a server.</span>
    <span class="footer-links">
      <a href="/">Home</a>
      <a href="/app/play/">Play</a>
      <a href="/app/analysis/">Analysis</a>
      <a href="/app/arena/">Arena</a>
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>
