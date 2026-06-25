<script lang="ts">
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  const title = "0x88 — specialized chess engines in your browser";
  const description = "Run specialized chess engines entirely in your browser. Human-like neural networks, multi-engine analysis, and live engine tournaments — no installs, no servers, powered by WebGPU and WebAssembly.";
  const github = "https://github.com/manthedan/0x88";
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>
<SiteHeader />
<main id="main">

<!-- ===== Hero ===== -->
<section class="hero" aria-labelledby="hero-h1">
  <div class="wrap">
    <h1 id="hero-h1">Run specialized chess engines <span class="accent">entirely in your browser.</span></h1>
    <p class="lede">
      No installation. No configuration. State-of-the-art neural networks and classical
      engines run locally on WebGPU and WebAssembly — nothing is ever sent to a server.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="/app/play/">Launch app <span class="arrow" aria-hidden="true">→</span></a>
      <a class="btn btn-ghost" href={github} rel="noopener">View on GitHub</a>
    </div>
  </div>
</section>

<!-- ===== Three modes ===== -->
<section class="modes" aria-labelledby="modes-h">
  <div class="wrap">
    <div class="modes-head">
      <p class="kicker">What's inside</p>
      <h2 id="modes-h">Three modes, one browser tab.</h2>
      <p>Each mode loads its engines on first use and caches them locally. Open a page and go.</p>
    </div>
    <div class="modes-grid">
      <a class="mode" href="/app/play/">
        <span class="mode-icon" aria-hidden="true">♟</span>
        <h3>Human vs. Computer</h3>
        <p>Play against neural networks fine-tuned to mimic real human play at a chosen rating — not the sterile, perfect lines of a raw engine. Takebacks and PGN export included.</p>
      </a>
      <a class="mode" href="/app/analysis/">
        <span class="mode-icon" aria-hidden="true">🔍</span>
        <h3>Multi-Engine Analysis</h3>
        <p>Run several engines on the same position at once and see exactly where they agree, diverge, or miss the best move — side by side, on your own games.</p>
      </a>
      <a class="mode" href="/app/arena/">
        <span class="mode-icon" aria-hidden="true">⚔</span>
        <h3>Engine Arena</h3>
        <p>Set the parameters and watch different engine architectures play each other live: matches, gauntlets, and round-robins with standings and per-game eval charts.</p>
      </a>
    </div>
  </div>
</section>

</main>
<footer class="site-footer">
  <div class="wrap">
    <span>Engines run entirely in your browser. Nothing is sent to a server.</span>
    <span class="footer-links">
      <a href="/app/play/">Play</a>
      <a href="/app/analysis/">Analysis</a>
      <a href="/app/arena/">Arena</a>
      <a href="/docs/">Docs</a>
      <a href={github} rel="noopener">GitHub</a>
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>

<style>
  .hero{
    position:relative; overflow:hidden;
    background:
      radial-gradient(ellipse 80% 60% at 70% 0%, color-mix(in srgb, var(--accent-soft) 70%, transparent), transparent 60%),
      radial-gradient(ellipse 60% 50% at 20% 100%, color-mix(in srgb, var(--accent-soft) 40%, transparent), transparent 60%),
      linear-gradient(180deg, var(--bg-2), var(--bg));
    border-bottom:1px solid var(--rule);
  }
  .hero::before{
    content:""; position:absolute; inset:0; pointer-events:none; opacity:.06;
    background-image:
      repeating-linear-gradient(90deg, var(--ink) 0 6.25%, transparent 6.25% 12.5%),
      repeating-linear-gradient(0deg, var(--ink) 0 6.25%, transparent 6.25% 12.5%);
    background-size:64px 64px; background-position:top right;
    -webkit-mask-image:radial-gradient(ellipse 60% 80% at 80% 20%, black, transparent 70%);
            mask-image:radial-gradient(ellipse 60% 80% at 80% 20%, black, transparent 70%);
  }
  .hero .wrap{position:relative; padding:80px 24px 72px}
  .hero h1{
    font-size:clamp(34px, 6vw, 54px); line-height:1.06; margin:0 0 20px;
    max-width:20ch; letter-spacing:-.02em;
  }
  .hero h1 .accent{color:var(--accent)}
  .hero .lede{
    font-size:clamp(15px, 2vw, 18px); color:var(--ink-soft); max-width:60ch;
    line-height:1.6; margin:0;
  }
  .hero .cta-row{display:flex; flex-wrap:wrap; gap:12px; margin-top:32px}
  .btn{
    display:inline-flex; align-items:center; gap:8px;
    padding:13px 22px; border-radius:10px; font-weight:600; font-size:14.5px;
    text-decoration:none; border:1px solid transparent; cursor:pointer;
    transition:transform .15s ease, box-shadow .15s ease, background .15s;
  }
  .btn:hover{text-decoration:none}
  .btn-primary{
    background:var(--accent); color:#fbf8f0;
    box-shadow:0 6px 20px -10px color-mix(in srgb, var(--accent) 80%, black);
  }
  .btn-primary:hover{background:var(--accent-2); transform:translateY(-1px)}
  .btn-ghost{background:var(--panel); color:var(--ink); border-color:var(--rule)}
  .btn-ghost:hover{border-color:var(--accent); transform:translateY(-1px)}
  .btn .arrow{transition:transform .15s ease}
  .btn:hover .arrow{transform:translateX(3px)}

  .modes .wrap{padding:64px 24px 72px}
  .modes-head{margin-bottom:32px; max-width:60ch}
  .modes-head .kicker{
    font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.1em;
    color:var(--accent); margin:0 0 10px;
  }
  .modes-head h2{font-size:clamp(24px,4vw,32px); margin:0 0 10px; line-height:1.15}
  .modes-head p{color:var(--muted); font-size:15px; margin:0; line-height:1.6}
  .modes-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:18px}
  .mode{
    display:flex; flex-direction:column; gap:12px;
    background:var(--panel); border:1px solid var(--rule); border-radius:var(--radius-lg);
    padding:26px 24px; text-decoration:none; color:inherit;
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }
  .mode:hover{
    border-color:color-mix(in srgb, var(--accent) 35%, var(--rule));
    box-shadow:var(--shadow-lg); transform:translateY(-3px); text-decoration:none;
  }
  .mode-icon{
    width:40px; height:40px; border-radius:11px;
    background:var(--accent-soft); color:var(--accent-deep);
    display:grid; place-items:center; font-size:20px;
  }
  .mode h3{font-size:19px; margin:0; line-height:1.2}
  .mode p{color:var(--muted); margin:0; font-size:14px; line-height:1.6}

  footer.site-footer{margin-top:0}

  @media (max-width:760px){
    .modes-grid{grid-template-columns:1fr}
    .hero .wrap{padding:56px 20px 48px}
    .modes .wrap{padding:48px 20px 56px}
  }
</style>
