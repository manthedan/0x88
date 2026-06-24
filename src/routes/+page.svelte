<script lang="ts">
  import { onMount } from 'svelte';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  const title = "0x88 Chess \u2014 state-of-the-art chess, zero installs";
  const description = "State-of-the-art chess engines running entirely in your browser. Leela Chess Zero on WebGPU, plus Stockfish, Berserk, Viridithas, PlentyChess and more on WebAssembly. No installs, no servers \u2014 democratizing chess technology and pushing browser deep-learning deployment.";
  onMount(() => {
    let cleanup: () => void = () => undefined;
    let mounted = true;
    void import('../lc0/homeBrowser').then((module) => {
      if (!mounted) return;
      cleanup = module.mountHomeBrowser();
    });
    return () => {
      mounted = false;
      cleanup();
    };
  });
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
    <p class="eyebrow"><span class="dot" aria-hidden="true"></span> Open source · runs entirely client-side</p>
    <h1 id="hero-h1">State-of-the-art chess, <span class="accent">zero installs.</span></h1>
    <p class="lede">
      The strongest neural networks and classical engines — <strong>Leela Chess Zero on WebGPU</strong>,
      Stockfish, Berserk, Viridithas, and PlentyChess on WebAssembly — running in a single browser tab.
      No downloads. No servers. No technical setup. Just open a page and play.
    </p>
    <div class="cta-row">
      <a class="btn btn-primary" href="/app/play/">Play now <span class="arrow" aria-hidden="true">→</span></a>
      <a class="btn btn-ghost" href="/app/analysis/">Analyze a game</a>
      <a class="btn btn-ghost" href="/app/arena/">Watch engines duel</a>
    </div>

    <div class="stats" role="list">
      <div class="stat" role="listitem"><div class="v">7+</div><div class="l">engines, ready to play</div></div>
      <div class="stat" role="listitem"><div class="v">WebGPU</div><div class="l">deep-learning inference</div></div>
      <div class="stat" role="listitem"><div class="v">100%</div><div class="l">client-side, private</div></div>
      <div class="stat" role="listitem"><div class="v">0</div><div class="l">servers in the loop</div></div>
    </div>
  </div>
</section>

<!-- ===== Three actions ===== -->
<section class="block" aria-labelledby="actions-h">
  <div class="wrap">
    <div class="section-head">
      <p class="kicker">What you can do right now</p>
      <h2 id="actions-h">Three doors in. Pick one.</h2>
      <p>Everything below loads the engine inside your browser on first use and caches it for next time. Nothing leaves your machine.</p>
    </div>
    <div class="cards-3">
      <a class="action-card" href="/app/play/">
        <div class="ac-head">
          <span class="ac-tag">Play</span>
          <span class="ac-icon" aria-hidden="true">♟</span>
        </div>
        <h3>Play a game</h3>
        <p>Face the human-like Maia nets trained on millions of real games (1100–1900 Elo), or take on the strongest engines directly. Takebacks, undo, and PGN export included.</p>
        <span class="ac-arrow">Start playing <span class="a" aria-hidden="true">→</span></span>
      </a>
      <a class="action-card" href="/app/analysis/">
        <div class="ac-head">
          <span class="ac-tag">Analyze</span>
          <span class="ac-icon" aria-hidden="true">🔍</span>
        </div>
        <h3>Analyze any position</h3>
        <p>Run multiple engines side by side on your games. Get accuracy scores, critical moments, and an opening explorer built from your own PGNs. Import a game and dig in.</p>
        <span class="ac-arrow">Open analysis board <span class="a" aria-hidden="true">→</span></span>
      </a>
      <a class="action-card" href="/app/arena/">
        <div class="ac-head">
          <span class="ac-tag">Compete</span>
          <span class="ac-icon" aria-hidden="true">⚔</span>
        </div>
        <h3>Engine arena</h3>
        <p>Stage engine-vs-engine matches, gauntlets, and round-robin tournaments. Live standings, Elo estimates, and per-game eval charts — watch the strongest players in chess fight it out.</p>
        <span class="ac-arrow">Enter the arena <span class="a" aria-hidden="true">→</span></span>
      </a>
    </div>
  </div>
</section>

<!-- ===== Capabilities & storage ===== -->
<section class="block" aria-labelledby="caps-h">
  <div class="wrap">
    <div class="section-head">
      <p class="kicker">Live diagnostics</p>
      <h2 id="caps-h">What's running in your browser</h2>
      <p>This site uses the browser as a deep-learning runtime. Here's what your device currently supports, and what's stored locally from past visits.</p>
    </div>
    <div class="caps-panel">
      <div class="caps-left">
        <h3>Capabilities</h3>
        <p>WebGPU unlocks the large Leela Chess Zero neural nets (t3, BT4) at GPU speed. WebAssembly runs every classical NNUE engine. Threads accelerate the WASM builds when the page is cross-origin isolated.</p>
        <div id="caps" aria-live="polite"><span class="cap">Checking capabilities…</span></div>
        <p class="capnote" id="capNote"></p>
      </div>
      <div class="caps-right">
        <h4>Downloads &amp; storage</h4>
        <div id="storage" aria-live="polite"></div>
      </div>
    </div>
  </div>
</section>

<!-- ===== Engines ===== -->
<section class="block" aria-labelledby="engines-h">
  <div class="wrap">
    <div class="section-head">
      <p class="kicker">The roster</p>
      <h2 id="engines-h">The engines</h2>
      <p>Two distinct families of chess AI live here. <strong>Neural networks</strong> evaluate positions with deep learning on WebGPU. <strong>NNUE engines</strong> evaluate efficiently on CPU via WebAssembly. All are state-of-the-art in their class.</p>
    </div>
    <div class="engine-filters" role="tablist" aria-label="Filter engines">
      <button class="engine-filter active" data-filter="all" role="tab" aria-selected="true">All</button>
      <button class="engine-filter" data-filter="neural" role="tab" aria-selected="false">Neural · WebGPU</button>
      <button class="engine-filter" data-filter="nnue" role="tab" aria-selected="false">NNUE · WebAssembly</button>
    </div>
    <div class="engines" id="engineGrid">
      <div class="engine" data-family="neural">
        <div class="engine-head"><b>Maia3</b><span class="rt">neural · WASM</span></div>
        <span class="desc">A single Elo-conditioned human-like model with a 600–2600 rating slider. Trained on real human games to play like a person at any level.</span>
        <div class="meta"><span>human-like</span><span>rating-conditioned</span></div>
      </div>
      <div class="engine" data-family="neural">
        <div class="engine-head"><b>Leela Chess Zero</b><span class="rt">neural · WebGPU / WASM</span></div>
        <span class="desc">The open-source neural-network project. Small t1 net for fast play, plus Leela Queen Odds, with large nets (t3, BT4) available on analysis/arena builds when hosted.</span>
        <div class="meta"><span>policy + WDL + MLH</span><span>smolgen attention</span></div>
      </div>
      <div class="engine" data-family="nnue">
        <div class="engine-head"><b>Stockfish 18</b><span class="rt">NNUE · WASM</span></div>
        <span class="desc">The strongest classical chess engine. Shipped in Lite and full-network flavors, with threaded builds available on cross-origin-isolated pages.</span>
        <div class="meta"><span>open source</span><span>SIMD + threads</span></div>
      </div>
      <div class="engine" data-family="nnue">
        <div class="engine-head"><b>Viridithas</b><span class="rt">NNUE · WASM (Rust)</span></div>
        <span class="desc">A modern Rust engine with relaxed-SIMD acceleration. Consistent top-tier strength in computer chess tournaments.</span>
        <div class="meta"><span>Rust</span><span>relaxed-SIMD</span></div>
      </div>
      <div class="engine" data-family="nnue">
        <div class="engine-head"><b>Berserk</b><span class="rt">NNUE · WASM (C)</span></div>
        <span class="desc">A very strong C engine compiled with Emscripten. Proven contender in the engine chess scene.</span>
        <div class="meta"><span>C / Emscripten</span><span>single-thread</span></div>
      </div>
      <div class="engine" data-family="nnue">
        <div class="engine-head"><b>PlentyChess</b><span class="rt">NNUE · WASM (C++)</span></div>
        <span class="desc">A top-tier C++ engine with feature-detected SIMD builds. Ships with its processed NNUE network embedded.</span>
        <div class="meta"><span>C++ / Emscripten</span><span>SIMD</span></div>
      </div>
      <div class="engine" data-family="nnue">
        <div class="engine-head"><b>Reckless</b><span class="rt">NNUE · WASM (Rust)</span></div>
        <span class="desc">Rust NNUE engine shipped with scalar, SIMD, and relaxed SIMD WASI builds for Play, Analysis, and Arena.</span>
        <div class="meta"><span>Rust</span><span>SIMD / relaxed SIMD</span></div>
      </div>
    </div>
  </div>
</section>

<!-- ===== Mission ===== -->
<section class="mission" aria-labelledby="mission-h">
  <div class="wrap">
    <div class="mission-grid">
      <div>
        <p class="kicker" style="font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--accent); margin:0 0 10px;">Why this exists</p>
        <h2 id="mission-h">Democratizing chess technology.</h2>
        <p>
          The best chess software has historically lived behind compile steps, GUI installs, UCI configurations, and a quiet wall of technical knowledge.
          <strong>This project tears that wall down.</strong> Every engine here — from the neural networks that learn from self-play to the hand-tuned NNUE evaluators — runs in a browser tab, the same way a webpage does.
        </p>
        <p>
          It's also a research platform. Running Leela Chess Zero on <strong>WebGPU</strong> pushes the browser as a first-class deep-learning runtime: quantized ONNX inference, custom WGSL kernels, and a progressive ladder of network sizes (t1 → t3 → BT4) that explores how far client-side GPU compute can go.
        </p>
      </div>
      <div class="principles">
        <div class="principle">
          <div class="pn" aria-hidden="true">1</div>
          <div>
            <h4>Privacy by architecture</h4>
            <p>Your games, your analysis, your moves never leave your device. There is no server to ship them to — the engines run locally, full stop.</p>
          </div>
        </div>
        <div class="principle">
          <div class="pn" aria-hidden="true">2</div>
          <div>
            <h4>Zero install</h4>
            <p>No downloads, no setup, no accounts. Open a URL, play the strongest engines in the world. That's the whole experience.</p>
          </div>
        </div>
        <div class="principle">
          <div class="pn" aria-hidden="true">3</div>
          <div>
            <h4>Open and remixable</h4>
            <p>The engines are open source. The browser runtime, the quantization pipeline, and the model packs are all documented here for anyone to learn from or build on.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

</main>
<footer class="site-footer">
  <div class="wrap">
    <div class="footer-grid">
      <div class="footer-brand">
        <a class="brand" href="/" aria-label="0x88.app — home">
          <span class="brand-mark" aria-hidden="true">0x88</span>
          <span class="brand-name">0x88.app</span>
        </a>
        <p>State-of-the-art chess engines, running entirely in your browser. An open-source project to democratize chess technology and push browser deep-learning deployment.</p>
      </div>
      <div class="footer-col">
        <h4>Play &amp; analyze</h4>
        <ul>
          <li><a href="/app/play/">Play</a></li>
          <li><a href="/app/analysis/">Analysis</a></li>
          <li><a href="/app/arena/">Arena</a></li>
        </ul>
      </div>
      <div class="footer-col">
        <h4>Learn more</h4>
        <ul>
          <li><a href="/docs/">Docs: pages, engines &amp; licenses</a></li>
          <li><a href="/single-engine/">Developer / single-engine console</a></li>
          <li><span>Browser benchmark (lab)</span></li>
        </ul>
      </div>
    </div>
    <div class="footer-bottom">
      <span>Everything runs client-side. Engine binaries and networks are downloaded on first use and cached locally.</span>
      <span><code>0x88.app · v0</code></span>
    </div>
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
  .hero .wrap{position:relative; padding:64px 24px 72px}
  .hero .eyebrow{
    display:inline-flex; align-items:center; gap:8px;
    font-family:var(--mono); font-size:11px; letter-spacing:.08em; text-transform:uppercase;
    color:var(--accent-deep); background:var(--accent-soft);
    padding:6px 12px; border-radius:99px; border:1px solid color-mix(in srgb, var(--accent) 25%, transparent);
    margin:0 0 20px;
  }
  .hero .eyebrow .dot{width:6px; height:6px; border-radius:50%; background:var(--accent); animation:pulse 2.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
  @media (prefers-reduced-motion: reduce){.hero .eyebrow .dot{animation:none}}
    .hero h1{
    font-size:clamp(34px, 6vw, 56px); line-height:1.05; margin:0 0 18px;
    max-width:18ch; letter-spacing:-.02em;
  }
  .hero h1 .accent{color:var(--accent); font-style:italic}
  .hero .lede{
    font-size:clamp(15px, 2vw, 18px); color:var(--ink-soft); max-width:62ch;
    line-height:1.6; margin:0 0 12px;
  }
  .hero .lede strong{color:var(--ink); font-weight:600}
  .hero .cta-row{display:flex; flex-wrap:wrap; gap:12px; margin-top:28px}
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
  .stats{
    display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
    background:var(--rule); border:1px solid var(--rule);
    border-radius:var(--radius); overflow:hidden; margin-top:40px;
  }
  .stat{background:var(--panel); padding:18px 20px}
  .stat .v{font-family:var(--serif); font-size:26px; font-weight:600; color:var(--accent-deep); line-height:1}
  .stat .l{font-size:12px; color:var(--muted); margin-top:4px; letter-spacing:.01em}
  section.block{padding:64px 0; border-bottom:1px solid var(--rule)}
  section.block:last-of-type{border-bottom:none}
  .section-head{margin-bottom:32px; max-width:72ch}
  .section-head .kicker{
    font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.1em;
    color:var(--accent); margin:0 0 10px;
  }
  .section-head h2{font-size:clamp(26px,4vw,36px); margin:0 0 10px; line-height:1.15}
  .section-head p{color:var(--muted); font-size:15.5px; margin:0; max-width:60ch; line-height:1.6}
  .cards-3{display:grid; grid-template-columns:repeat(3,1fr); gap:18px}
  .action-card{
    position:relative; display:flex; flex-direction:column; gap:10px;
    background:var(--panel); border:1px solid var(--rule); border-radius:var(--radius-lg);
    padding:24px; text-decoration:none; color:inherit; overflow:hidden;
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }
  .action-card::after{
    content:""; position:absolute; left:0; right:0; top:0; height:3px;
    background:linear-gradient(90deg, var(--accent), var(--accent-2));
    transform:scaleX(0); transform-origin:left; transition:transform .25s ease;
  }
  .action-card:hover{
    border-color:color-mix(in srgb, var(--accent) 35%, var(--rule));
    box-shadow:var(--shadow-lg); transform:translateY(-3px); text-decoration:none;
  }
  .action-card:hover::after{transform:scaleX(1)}
  .action-card .ac-head{display:flex; align-items:center; justify-content:space-between; gap:12px}
  .action-card .ac-tag{
    font-family:var(--mono); font-size:10.5px; text-transform:uppercase; letter-spacing:.08em;
    color:var(--accent-deep); background:var(--accent-soft);
    padding:4px 10px; border-radius:99px;
  }
  .action-card .ac-icon{
    width:36px; height:36px; border-radius:10px;
    background:var(--accent-soft); color:var(--accent-deep);
    display:grid; place-items:center; font-size:18px;
  }
  .action-card h3{font-size:20px; margin:4px 0 0; line-height:1.2}
  .action-card p{color:var(--muted); margin:0; font-size:14px; line-height:1.6}
  .action-card .ac-arrow{margin-top:auto; padding-top:8px; font-weight:600; font-size:13px; color:var(--accent); display:flex; align-items:center; gap:6px}
  .action-card .ac-arrow .a{transition:transform .15s ease}
  .action-card:hover .ac-arrow .a{transform:translateX(3px)}
  .caps-panel{
    background:var(--panel); border:1px solid var(--rule); border-radius:var(--radius-lg);
    padding:28px; display:grid; grid-template-columns:1.1fr 1fr; gap:32px; align-items:start;
  }
  .caps-left h3{margin:0 0 8px; font-size:19px}
  .caps-left p{margin:0 0 16px; color:var(--muted); font-size:14px; line-height:1.6}
  :global(#caps){display:flex; flex-wrap:wrap; gap:8px}
  :global(.cap){
    font-family:var(--mono); font-size:12px; border:1px solid var(--rule);
    border-radius:99px; padding:7px 14px; background:var(--bg); color:var(--muted);
    display:inline-flex; align-items:center; gap:6px;
  }
  :global(.cap.ok){border-color:color-mix(in srgb, var(--accent) 40%, transparent); color:var(--accent-deep); background:var(--accent-soft)}
  :global(.cap.ok::before){content:"\2713"; font-weight:700}
  :global(.cap.no){border-color:color-mix(in srgb, var(--warn) 40%, transparent); color:var(--warn); background:var(--warn-soft)}
  :global(.cap.no::before){content:"\2715"}
  :global(.cap:not(.ok):not(.no)::before){content:"\2022"; opacity:.5}
  :global(.capnote){font-size:13px; color:var(--muted); margin:14px 0 0; line-height:1.55; max-width:50ch}
  .caps-right h4{margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted)}
  :global(#storage){display:grid; gap:10px}
  :global(.store-row){
    display:grid; grid-template-columns:1fr auto auto; align-items:center; gap:14px;
    background:var(--bg); border:1px solid var(--rule); border-radius:var(--radius-sm);
    padding:12px 14px; font-size:13px;
  }
  :global(.store-row .store-info){display:grid; gap:2px; min-width:0}
  :global(.store-row .store-info b){font-weight:600; font-size:13.5px}
  :global(.store-row .store-info span){color:var(--muted); font-size:12px; line-height:1.4}
  :global(.store-row .store-size){font-family:var(--mono); font-size:11.5px; color:var(--muted-2); white-space:nowrap}
  :global(.store-row button){
    font:inherit; font-size:12px; padding:7px 12px; border:1px solid var(--rule);
    border-radius:6px; background:var(--panel); cursor:pointer; color:var(--ink-soft);
    transition:border-color .15s, color .15s, background .15s;
  }
  :global(.store-row button:disabled){opacity:.45; cursor:not-allowed}
  :global(.store-row button:hover:not(:disabled)){border-color:var(--warn); color:var(--warn); background:var(--warn-soft)}
  :global(.store-row button.clearing){opacity:.7}
  .engine-filters{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:22px}
  .engine-filter{
    font:inherit; font-size:13px; padding:7px 14px; border-radius:99px;
    background:var(--panel); border:1px solid var(--rule); color:var(--muted);
    cursor:pointer; transition:all .15s;
  }
  .engine-filter:hover{color:var(--ink); border-color:var(--accent)}
  .engine-filter.active{background:var(--accent); color:#fbf8f0; border-color:var(--accent)}
  .engines{display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px}
  .engine{
    background:var(--panel); border:1px solid var(--rule); border-radius:var(--radius);
    padding:18px 20px; display:flex; flex-direction:column; gap:8px;
    transition:border-color .15s, box-shadow .15s;
  }
  .engine:hover{border-color:color-mix(in srgb, var(--accent) 30%, var(--rule)); box-shadow:var(--shadow-sm)}
  .engine-head{display:flex; align-items:baseline; justify-content:space-between; gap:10px}
  .engine b{font-size:16px; font-family:var(--serif)}
  .engine .rt{
    font-family:var(--mono); font-size:10.5px; color:var(--accent-deep);
    background:var(--accent-soft); padding:3px 8px; border-radius:99px; white-space:nowrap;
  }
  .engine[data-family="nnue"] .rt{color:var(--gold); background:color-mix(in srgb, var(--gold) 12%, transparent)}
  .engine span.desc{color:var(--muted); font-size:13px; line-height:1.5}
  .engine .meta{display:flex; gap:10px; flex-wrap:wrap; margin-top:4px}
  .engine .meta span{font-family:var(--mono); font-size:10.5px; color:var(--muted-2)}
  .mission{
    background:
      radial-gradient(ellipse 70% 50% at 50% 0%, color-mix(in srgb, var(--accent-soft) 60%, transparent), transparent 70%),
      var(--panel);
    border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
  }
  .mission .wrap{padding:72px 24px}
  .mission-grid{display:grid; grid-template-columns:1.2fr 1fr; gap:56px; align-items:center}
  .mission h2{font-size:clamp(26px,4vw,38px); margin:0 0 18px; line-height:1.15}
  .mission p{color:var(--ink-soft); font-size:16px; line-height:1.7; margin:0 0 14px; max-width:54ch}
  .mission p strong{color:var(--ink)}
  .principles{display:grid; gap:18px}
  .principle{display:flex; gap:16px; align-items:start}
  .principle .pn{
    flex-shrink:0; width:36px; height:36px; border-radius:10px;
    background:var(--accent); color:#fbf8f0; display:grid; place-items:center;
    font-family:var(--serif); font-weight:600; font-size:16px;
  }
  .principle h4{margin:0 0 4px; font-size:15.5px; font-family:var(--sans)}
  .principle p{margin:0; color:var(--muted); font-size:13.5px; line-height:1.55}
  footer.site-footer{background:var(--bg-2); border-top:1px solid var(--rule); padding:48px 0 32px; margin-top:0}
  .footer-grid{display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:40px; margin-bottom:32px}
  .footer-brand .brand{margin-bottom:14px}
  .footer-brand p{color:var(--muted); font-size:13.5px; line-height:1.6; margin:0; max-width:42ch}
  .footer-col h4{font-family:var(--sans); font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted-2); margin:0 0 12px; font-weight:600}
  .footer-col ul{list-style:none; padding:0; margin:0; display:grid; gap:8px}
  .footer-col a{color:var(--ink-soft); font-size:14px}
  .footer-col a:hover{color:var(--accent)}
  .footer-bottom{
    border-top:1px solid var(--rule); padding-top:20px;
    display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;
    font-size:12px; color:var(--muted);
  }
  .footer-bottom code{font-family:var(--mono); font-size:11px; background:var(--panel); padding:2px 6px; border-radius:4px; border:1px solid var(--rule)}
  @media (max-width:900px){
    .stats{grid-template-columns:repeat(2,1fr)}
    .caps-panel{grid-template-columns:1fr}
    .mission-grid{grid-template-columns:1fr; gap:32px}
    .footer-grid{grid-template-columns:1fr 1fr}
    .footer-brand{grid-column:1 / -1}
  }
  @media (max-width:680px){
    .cards-3{grid-template-columns:1fr}
    .hero .wrap{padding:44px 20px 52px}
    section.block{padding:48px 0}
    .footer-grid{grid-template-columns:1fr}
    .footer-bottom{flex-direction:column; gap:6px}
    .stats{grid-template-columns:1fr 1fr; gap:1px}
  }
</style>
