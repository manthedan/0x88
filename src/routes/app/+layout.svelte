<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <div class="wrap">
    <a class="brand" href="/" aria-label="0x88.app home">
      <span class="brand-mark" aria-hidden="true">0x88</span>
      <span class="brand-name">0x88.app</span>
    </a>
    <span class="page-title">App</span>
    <nav class="primary" aria-label="Primary">
      <a href="/">Home</a>
      <a href="/app/play">Play</a>
      <a href="/app/analysis">Analysis</a>
      <a href="/app/arena">Arena</a>
      <a href="/docs" data-sveltekit-reload>Docs</a>
    </nav>
    <button id="themeToggle" class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle night mode">&#9790;</button>
    <select id="boardStyleSelect" class="board-style-select" aria-label="Board style" title="Board style">
      <option value="brown">Brown</option>
      <option value="night">Night</option>
      <option value="blue">Blue</option>
      <option value="green">Green</option>
    </select>
  </div>
</header>
<slot />
<footer class="site-footer">
  <div class="wrap">
    <span>Engines run entirely in your browser. Nothing is sent to a server.</span>
    <span class="footer-links">
      <a href="/">Home</a>
      <a href="/app/play">Play</a>
      <a href="/app/analysis">Analysis</a>
      <a href="/app/arena">Arena</a>
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>
