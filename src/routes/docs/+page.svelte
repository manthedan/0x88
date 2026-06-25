<script lang="ts">
  import { onMount } from 'svelte';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  const title = "0x88 Chess — docs: pages, engines, licenses";
  const description = "What each 0x88.app page does, the chess engines running in your browser, who built them, how we package them, and our GPL/AGPL corresponding-source commitments.";
  onMount(() => {
    const headings = document.querySelectorAll('.doc-content section[id], .doc-content h3[id]');
    const tocLinks = document.querySelectorAll<HTMLAnchorElement>('.toc a[href^="#"]');
    if (!('IntersectionObserver' in window) || !headings.length) return;
    const linkMap = new Map<string, HTMLAnchorElement>();
    tocLinks.forEach((link) => {
      const href = link.getAttribute('href');
      if (href) linkMap.set(href.slice(1), link);
    });
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.getAttribute('id');
        const link = id ? linkMap.get(id) : undefined;
        if (!link) return;
        tocLinks.forEach((item) => item.classList.remove('active'));
        link.classList.add('active');
      });
    }, { rootMargin: '-88px 0px -70% 0px', threshold: 0 });
    headings.forEach((heading) => observer.observe(heading));
    return () => observer.disconnect();
  });
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>
<SiteHeader pageTitle="Docs" />
<main id="main">

<section class="doc-hero" aria-labelledby="doc-h1">
  <div class="wrap">
    <p class="kicker">Documentation</p>
    <h1 id="doc-h1">What's here, who built it, where the source lives.</h1>
    <p>
      <strong>0x88.app</strong> is a browser-first chess platform. The same engines that win computer-chess championships run here in a single browser tab, no installs and no servers. This page explains what each page does, catalogues every engine running on the site, credits the people who built them, and lays out exactly how each binary is packaged and where its corresponding source lives &mdash; so our GPL/AGPL commitments are concrete, not abstract.
    </p>
  </div>
</section>

<div class="doc-body">
<div class="wrap">

<!-- ===== TOC (left rail) ===== -->
<aside class="toc" aria-label="On this page">
  <h4>On this page</h4>
  <ul>
    <li><a href="#pages">The pages</a>
      <ul>
        <li><a href="#pages-play">Play</a></li>
        <li><a href="#pages-analysis">Analysis</a></li>
        <li><a href="#pages-arena">Arena</a></li>
      </ul>
    </li>
    <li><a href="#human-vs-computer">Human vs computer play</a></li>
    <li><a href="#neural-runtimes">Neural browser runtimes</a></li>
    <li><a href="#cpu-wasm">CPU WASM runtimes</a>
      <ul>
        <li><a href="#cpu-wasm-relaxed-simd">Relaxed SIMD</a></li>
      </ul>
    </li>
    <li><a href="#engines">The engines</a>
      <ul>
        <li><a href="#engines-lc0">Leela Chess Zero</a></li>
        <li><a href="#engines-lqo">Leela Queen Odds (LQO)</a></li>
        <li><a href="#engines-maia3">Maia3</a></li>
        <li><a href="#engines-stockfish">Stockfish 18</a></li>
        <li><a href="#engines-berserk">Berserk</a></li>
        <li><a href="#engines-viridithas">Viridithas</a></li>
        <li><a href="#engines-plentychess">PlentyChess</a></li>
        <li><a href="#engines-reckless">Reckless</a></li>
      </ul>
    </li>
    <li><a href="#licenses">Licenses &amp; source</a>
      <ul>
        <li><a href="#licenses-per-engine">Per-engine source links</a></li>
      </ul>
    </li>
    <li><a href="#cdn">Artifact CDN &amp; caching</a></li>
    <li><a href="#removal">I'm in this project and I don't like it</a></li>
  </ul>
</aside>

<div class="doc-content">

<!-- ===== PAGES ===== -->
<section id="pages">
  <h2>The pages <a class="anchor-link" href="#pages" aria-label="Link to this section">#</a></h2>
  <p class="lead">Three user-facing entry points, each running entirely client-side. Open a URL, pick an engine, and everything else loads into your browser on demand.</p>

  <div class="page-block">
    <h3 id="pages-play"><span class="pn">Play</span> Play a game <a class="anchor-link" href="#pages-play" aria-label="Link to this section">#</a></h3>
    <span class="pg-url">/app/play</span>
    <p>Play chess against the engine of your choice, or against the human-like Maia3 model that imitates how real players at any rating actually play. Pick your color, pick your opponent, pick a strength level, and the rest is a normal game of chess &mdash; takebacks and PGN export included.</p>
    <ul>
      <li><strong>Maia3 rating slider (600&ndash;2600)</strong> &mdash; trained on millions of human games; matches the move distribution of players at a chosen Elo rather than playing the objectively best move.</li>
      <li><strong>Engine opponents at five strength levels</strong> &mdash; from the small Leela Chess Zero net up to the strongest NNUE engines, with depth/node limits used as the strength dial.</li>
      <li><strong>Full game lifecycle</strong> &mdash; takebacks, resign, new game, flip board, move list, and one-click PGN export.</li>
    </ul>
    <p>Nothing about your game is sent anywhere. The engine binary and (if needed) the neural network download on first use, then cache locally for next time.</p>
  </div>

  <div class="page-block">
    <h3 id="pages-analysis"><span class="pn">Analysis</span> Analysis board <a class="anchor-link" href="#pages-analysis" aria-label="Link to this section">#</a></h3>
    <span class="pg-url">/app/analysis</span>
    <p>The analysis board is the power-user surface. Load a position by FEN or a full game by PGN, then run <em>multiple engines side by side</em> on it. Each engine's evaluation, principal variation, and best move are shown in a comparison table so you can see where engines agree and where they disagree &mdash; that disagreement is often the most interesting thing in a position.</p>
    <ul>
      <li><strong>Multi-engine comparison</strong> &mdash; add as many engines as your machine can handle; each contributes its eval, PV, and best move to a shared table.</li>
      <li><strong>Game review</strong> &mdash; annotate every move of a PGN with accuracy scores, critical moments (best/good/inaccuracy/mistake/blunder), and a win-probability chart.</li>
      <li><strong>Human-move explorer (Maia3)</strong> &mdash; see what rated humans actually play in the current position, with a per-rating move distribution. Useful for understanding practical chances versus engine truth.</li>
      <li><strong>Opening explorer</strong> &mdash; once you've loaded your own PGNs into a local IndexedDB database, the explorer surfaces transpositions and frequency stats over <em>your</em> games, not a generic opening book.</li>
      <li><strong>PGN database</strong> &mdash; import Lichess or Chess.com games by username, store them locally, and search by position across the whole collection.</li>
    </ul>
  </div>

  <div class="page-block">
    <h3 id="pages-arena"><span class="pn">Arena</span> Engine arena <a class="anchor-link" href="#pages-arena" aria-label="Link to this section">#</a></h3>
    <span class="pg-url">/app/arena</span>
    <p>The arena is where engines play each other. Schedule head-to-head matches, gauntlets (one engine vs a field), or round-robin tournaments, and watch games play out on the board in real time with live evaluation bars and per-move scoring. It's part spectacle, part testbed &mdash; the same interface the project uses internally to benchmark new Leela Chess Zero networks.</p>
    <ul>
      <li><strong>Tournament formats</strong> &mdash; head-to-head, gauntlet, round-robin, with configurable games-per-pairing and color alternation.</li>
      <li><strong>Live standings and Elo estimates</strong> &mdash; running score table, Elo updates after each game, and a final ranking.</li>
      <li><strong>Per-game eval charts and move replays</strong> &mdash; click any finished game to replay it with the eval bar; click any move to jump the board.</li>
      <li><strong>Configurable engine runtime</strong> &mdash; pick engine, strength, opening book, and time control; the arena handles scheduling and the engines do the rest.</li>
    </ul>
  </div>
</section>

<!-- ===== HUMAN VS COMPUTER PLAY ===== -->
<section id="human-vs-computer">
  <h2>Human vs computer play <a class="anchor-link" href="#human-vs-computer" aria-label="Link to this section">#</a></h2>
  <p class="lead">There are two different ways to make chess AI useful against people. Maia3 is the neural human-modeling lane: it asks what a rated human is likely to play. LQO and Monty-style contempt search are the practical-engine lane: they keep strong search, but stop assuming the opponent will always find the perfect defense.</p>

  <div class="callout info">
    <h4>Short version</h4>
    <p><strong>Maia3</strong> is for human authenticity &mdash; sparring against a 1500-ish style, exploring common human moves, and estimating rating-conditioned human outcomes. <strong>LQO/search contempt</strong> is for practical pressure &mdash; queen-odds play, anti-draw bias, traps, and moves that maximize chances against a fallible human.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Question</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Maia3 neural lane</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">LQO / Monty contempt lane</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">What is it modeling?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">The move distribution and expected result of real humans at a chosen rating.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">A strong engine's practical chances against an imperfect, budget-limited human opponent.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">How does it pick moves?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Samples from, or takes argmax of, a rating-conditioned neural human policy.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Runs engine search with odds-calibrated evaluation and contempt knobs such as drawScore, searchContemptLimit, or contemptElo.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">What should the UI promise?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Human-like play: "what would a 1500-rated player do?"</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Practical engine play: "what move creates the most problems for this human?"</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Best fit</td>
        <td style="padding:10px">Human sparring, human-move explorer, and rating-conditioned analysis.</td>
        <td style="padding:10px">Queen odds, anti-draw pressure, trap-aware play, and practical-vs-objective comparisons.</td>
      </tr>
    </tbody>
  </table>

  <p>In product copy, do not call contempt search "human-like": it models the opponent's limitations, not the engine's own style. Conversely, do not route Maia3 through LC0 PUCT by default; a future mixed-policy mode can be useful, but it should be labeled experimental. The full design note lives in <a href="https://github.com/manthedan/0x88/blob/main/docs/human_vs_computer_play.md">docs/human_vs_computer_play.md</a>.</p>
</section>

<!-- ===== NEURAL BROWSER RUNTIMES ===== -->
<section id="neural-runtimes">
  <h2>Neural browser runtimes <a class="anchor-link" href="#neural-runtimes" aria-label="Link to this section">#</a></h2>
  <p class="lead">The LC0 and Maia lanes are browser neural inference stacks, not CPU UCI ports. They start from ONNX models, WebGPU kernels, or compiler-generated runtimes, then feed policy/value outputs into the Play, Analysis, and Arena surfaces.</p>

  <div class="callout info">
    <h4>Stable baseline</h4>
    <p><strong>ONNX Runtime WebGPU</strong> is the stable neural baseline and rollback path. TVMJS, custom WGSL, WebNN, and QDQ artifacts all need to prove themselves against ORT WebGPU with fixture drift and search-move gates before they can become product defaults.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Lane</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">What it is</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">ONNX WebGPU</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">ONNX Runtime Web running LC0/Maia models on the browser GPU, with ORT WASM as fallback/control.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Stable default baseline for neural models where WebGPU is available.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">ONNX QDQ</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Weight-only int8 quantize/dequantize ONNX graphs: smaller files, float compute after in-graph dequantization.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Per-model promotion after real chess fixture/search drift gates; mainly a download/cache win.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Custom WGSL</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Hand-written WebGPU kernels for hot LC0 subgraphs: encoder blocks, heads, legal-prior/readback experiments.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Explicit hybrid/runtime lane; parity and readback/batch-fill evidence decide promotion.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">TVMJS WebGPU</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Compiler path from ONNX to TVM Relax to browser-loadable TVMJS wasm plus WebGPU pipelines.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Research/opt-in; generated artifacts need separate hosting, provenance, and cross-device gates.</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">ORT WebNN EP</td>
        <td style="padding:10px">ONNX Runtime through WebNN, potentially reaching CoreML/ANE/NPU hardware instead of WebGPU.</td>
        <td style="padding:10px">Promising but flag-gated; not shippable until WebNN is unflagged and model-specific correctness gates pass.</td>
      </tr>
    </tbody>
  </table>

  <p>WebNN is worth tracking because early probes show real acceleration on supported hardware, but it currently requires Chrome's <code>WebMachineLearningNeuralNetwork</code> feature flag and has model-shape/precision caveats. Likewise, QDQ should be judged on real chess positions: random tensor comparisons can overstate int8 drift compared with actual LC0/Maia activations.</p>
  <p>Full notes live in <a href="https://github.com/manthedan/0x88/blob/main/docs/neural_browser_runtimes.md">docs/neural_browser_runtimes.md</a>, with supporting runbooks for <a href="https://github.com/manthedan/0x88/blob/main/docs/lc0_tvmjs_research_runbook.md">TVMJS</a>, <a href="https://github.com/manthedan/0x88/blob/main/docs/lc0web_custom_inference_checkpoint.md">custom WGSL</a>, and <a href="https://github.com/manthedan/0x88/blob/main/docs/lc0_t3_qdq_webnn_2026-06-10.md">QDQ/WebNN probes</a>.</p>
</section>

<!-- ===== CPU WASM RUNTIMES ===== -->
<section id="cpu-wasm">
  <h2>CPU WASM runtimes <a class="anchor-link" href="#cpu-wasm" aria-label="Link to this section">#</a></h2>
  <p class="lead">Not every engine here is a WebGPU neural net. Stockfish, Reckless, Berserk, Viridithas, PlentyChess, and the Monty lab port are CPU engines compiled to WebAssembly, then wrapped as browser workers with a UCI-style control plane.</p>

  <div class="callout info">
    <h4>Two build lanes</h4>
    <p><strong>Emscripten</strong> is the preferred intake path for C/C++ engines: it gives us Stockfish.js-style JS glue, a worker-loadable <code>.wasm</code>, and optional <code>.data</code> sidecars for NNUE files. <strong>Rust WASI</strong> is the preferred path for Rust engines: compile to <code>wasm32-wasip1</code>, then run through the browser WASI shim in one-shot, batch, or persistent-worker mode.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Runtime path</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Engines</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Build targets</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stockfish.js package</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Stockfish 18</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Lite/full, single-thread/threaded flavors; threaded builds require cross-origin isolation.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Emscripten UCI worker</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Berserk, PlentyChess</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Single-thread baseline first, then explicit <code>simd128</code>, SSE4.1-shaped wasm SIMD, or relaxed-SIMD variants after smoke/benchmark evidence.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Rust <code>wasm32-wasip1</code></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Reckless, Viridithas, Monty lab</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Scalar fallback, <code>+simd128</code> preferred variants, and relaxed-SIMD as experimental until proven per engine.</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Direct browser API</td>
        <td style="padding:10px">Reckless experiments</td>
        <td style="padding:10px">Bypasses UCI text for structured calls, but only graduates if it beats the simpler UCI/WASI path on lifecycle or latency.</td>
      </tr>
    </tbody>
  </table>

  <h3 id="cpu-wasm-relaxed-simd">Relaxed SIMD <a class="anchor-link" href="#cpu-wasm-relaxed-simd" aria-label="Link to this section">#</a></h3>
  <p>Relaxed SIMD is a separate WebAssembly feature from ordinary <code>simd128</code>. A relaxed build can fail to validate on a browser that supports baseline SIMD, so every relaxed artifact needs feature detection, asset fallback, and a non-relaxed path.</p>
  <p>The main win for these chess engines is the relaxed integer dot product used in NNUE layers. When the activation operand is proven to stay in <code>[0, 127]</code>, <code>i32x4_relaxed_dot_i8x16_i7x16_add</code> is value-exact and can replace slower <code>maddubs</code>/<code>dpbusd</code> emulation. Reckless, Viridithas, Berserk, and PlentyChess all have this proof in their current SIMD audit lanes.</p>
  <p>The rule is still parity-first: inspect the artifact to confirm relaxed opcodes are present, then require fixed-depth equality for best move, score, node count, and PV before promoting a runtime ladder such as <code>relaxed-simd &gt; simd128 &gt; scalar</code>. Without that evidence, relaxed SIMD remains a lab or benchmark variant.</p>

  <div class="callout info">
    <h4>Benchmark context</h4>
    <p>The current relaxed-SIMD snapshot is measured in a Chromium browser on macOS with an Apple M4 chip. Treat the NPS deltas as device/runtime-specific engineering evidence, not universal engine-speed claims; the exact fixed-depth parity result is the stronger promotion signal.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Engine</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Apple M4 / Chromium evidence</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Parity gate</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Reckless</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>+24%</code> NPS vs old kernels</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>60/60</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Viridithas</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>+14%</code> NPS over standard SIMD</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>40/40</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Berserk</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>1.50M</code> NPS vs <code>1.38M</code> SIMD (<code>+8%</code>)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>40/40</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">PlentyChess</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>992k</code> NPS vs <code>603k</code> default (<code>+64%</code>)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>40/40</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Stockfish</td>
        <td style="padding:10px">Not measured; current package is upstream Stockfish.js rather than a local relaxed-SIMD build.</td>
        <td style="padding:10px">N/A</td>
      </tr>
    </tbody>
  </table>

  <p>The promotion rule is deliberately conservative: build scripts must pin upstream source and NNUE assets, Node and browser smokes must pass, benchmarks must justify the variant, and GPL/AGPL artifacts need manifests plus matching source archives before public distribution. Full notes live in <a href="https://github.com/manthedan/0x88/blob/main/docs/cpu_wasm_runtimes.md">docs/cpu_wasm_runtimes.md</a> and the C/C++ porting recipe in <a href="https://github.com/manthedan/0x88/blob/main/docs/browser_c_engine_porting.md">docs/browser_c_engine_porting.md</a>.</p>
</section>

<!-- ===== ENGINES ===== -->
<section id="engines">
  <h2>The engines <a class="anchor-link" href="#engines" aria-label="Link to this section">#</a></h2>
  <p class="lead">Two families of chess AI. Neural networks evaluate positions with deep learning on WebGPU; NNUE engines evaluate efficiently on CPU via WebAssembly. All are state-of-the-art in their class, all are open source, and all are the work of the authors credited below &mdash; not us. We package and deploy them; they did the hard part.</p>

  <div class="engine-entry">
    <h3 id="engines-lc0">Leela Chess Zero <a class="anchor-link" href="#engines-lc0" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>LeelaChessZero &middot; <a href="https://github.com/LeelaChessZero/lc0" rel="noopener">github.com/LeelaChessZero/lc0</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="/models/lc0/manifest.json">/models/lc0/manifest.json</a></dd>
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (browser runtime, ONNX export &amp; quantization scripts)</dd>
    </dl>
    <p>A neural-network engine trained from scratch by self-play, in the tradition of AlphaZero. The networks here (t1, t3, and BT4) are real LC0 weights running on the browser's GPU through ONNX Runtime WebGPU. <strong>This project's headline research effort</strong> &mdash; exploring how far browser WebGPU can go as a deep-learning inference runtime, with quantized ONNX, custom WGSL kernels, and a progressive ladder of net sizes. The related Leela Queen Odds net (used for the Play page's queen-odds bot) is listed separately below.</p>
    <details>
      <summary>Architecture &amp; packaging details</summary>
      <div class="details-body">
        <p>LC0 uses a residual CNN with a policy head and value head, trained via self-play reinforcement learning. The browser deployment exports these to ONNX and applies QDQ (quantize-dequantize) int8 quantization to the weight tensors while keeping the computation graph in a WebGPU-friendly format. A progressive ladder of network sizes is served so weaker devices get a smaller net automatically.</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-lqo">Leela Queen Odds (LQO) <a class="anchor-link" href="#engines-lqo" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>notune &middot; <a href="https://github.com/notune/LeelaQueenOdds" rel="noopener">github.com/notune/LeelaQueenOdds</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span> (derived from LC0)</dd>
      <dt>Artifact</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/models/lc0/lqo_v2.f16.qdq8.onnx">/models/lc0/lqo_v2.f16.qdq8.onnx</a> (~96 MB, QDQ int8)</dd>
      <dt>Derived from</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/models/lc0/lqo_v2.f16.onnx">lqo_v2.f16.onnx</a> (fp16 source export)</dd>
      <dt>Manifest</dt><dd><a href="/models/lc0/manifest.json">/models/lc0/manifest.json</a></dd>
    </dl>
    <p>The public net behind the Lichess queen-odds bot: a Leela Chess Zero network fine-tuned to win against humans from a <strong>queen-down</strong> starting position. Selecting it on the Play page removes your queen before move one, and the bot then presses hardest for tricks &mdash; the same dynamic the Lichess bot is famous for. Not a general analysis net (it evaluates the queen-odds start as equal), but a uniquely instructive opponent for practicing from behind.</p>
    <details>
      <summary>Search parameters &amp; packaging details</summary>
      <div class="details-body">
        <p>The Lichess bot runs the upstream net with search-contempt at 12&ndash;15k nodes. The browser deployment scales that down to visit budgets and applies three README-derived knobs: <code>cpuct 1.5</code>, <code>drawScore -0.5</code> (aggressive anti-draw lean), and <code>searchContemptLimit 24</code>. These were A/B-validated at queen odds vs Maia 1900 (92% vs 58% baseline).</p>
        <p>The served artifact is a QDQ int8 quantization of the fp16 export, derived in-tree and shipped from the LC0 manifest alongside the other big nets (BT4, t3).</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-maia3">Maia3 <a class="anchor-link" href="#engines-maia3" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>CSSLab (Jon Klein, Reid McIlroy-Young, et al.) &middot; <a href="https://github.com/CSSLab/maia3" rel="noopener">github.com/CSSLab/maia3</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Provenance</dt><dd><a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">/docs/model_provenance/maia3.md</a></dd>
      <dt>Frontend</dt><dd><a href="https://github.com/CSSLab/maia-platform-frontend" rel="noopener">github.com/CSSLab/maia-platform-frontend</a> (byte-identical upstream fp16 model)</dd>
    </dl>
    <p>A human-move-prediction model trained on millions of real Lichess games. Where Stockfish asks "what's best?", Maia asks "what would a 1500-rated human actually play here?" &mdash; and answers it surprisingly well, across the whole rating ladder. The Play page uses it to give you an opponent that feels like a person, not a sandboxed grandmaster.</p>
    <div class="callout info">
      <h4>AGPL source offer</h4>
      <p>The default browser artifact is a local int8 quantization of the upstream fp16 file. Because Maia3 is AGPL-3.0, the <a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">derivation recipe</a> is part of the source offer, not just the upstream link.</p>
    </div>
  </div>

  <div class="engine-entry">
    <h3 id="engines-stockfish">Stockfish 18 <a class="anchor-link" href="#engines-stockfish" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>official-stockfish &middot; via <a href="https://github.com/nmrugg/stockfish.js" rel="noopener">github.com/nmrugg/stockfish.js</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/stockfish/stockfish-18.0.7.manifest.json">/stockfish/stockfish-18.0.7.manifest.json</a></dd>
      <dt>Source</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/stockfish/stockfish-18.0.7-corresponding-source.tar.gz">stockfish-18.0.7-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>The strongest classical chess engine in the world, and has been for years. A community project with hundreds of contributors, continually refined through distributed testing (FishCooking). The NNUE evaluation lets it calculate god-like evaluations with a forward pass cheap enough to search hundreds of millions of nodes per second on a CPU. We ship Stockfish.js 18 in Lite and full-network flavors.</p>
    <details>
      <summary>Architecture &amp; packaging details</summary>
      <div class="details-body">
        <p>Stockfish uses an efficiently updatable neural network (NNUE) as its static evaluation function. The network takes a HalfKAv2 feature representation of the board, runs through a feature transformer and two small linear layers with ClippedReLU activations, and outputs a single eval value. The browser build compiles Stockfish to WebAssembly via Emscripten, with the NNUE network embedded. Both the Lite (smaller net) and full-network variants are served from the manifest.</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-berserk">Berserk <a class="anchor-link" href="#engines-berserk" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Jay Honnold &middot; <a href="https://github.com/jhonnold/berserk" rel="noopener">github.com/jhonnold/berserk</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/berserk/berserk-emscripten-single-thread.manifest.json">/berserk/berserk-emscripten-single-thread.manifest.json</a></dd>
      <dt>Source</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz">berserk-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>A remarkably strong, remarkably compact engine written in C. Jay Honnold's work is a textbook example of how far clean code and a well-tuned NNUE can go &mdash; Berserk consistently outranks engines with far larger codebases. Compiled here with Emscripten for the browser.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-viridithas">Viridithas <a class="anchor-link" href="#engines-viridithas" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Cosmo Bobak &middot; <a href="https://github.com/cosmobobak/viridithas" rel="noopener">github.com/cosmobobak/viridithas</a></dd>
      <dt>License</dt><dd><span class="lic mit">MIT</span></dd>
      <dt>Manifest</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/viridithas/viridithas-wasip1.manifest.json">/viridithas/viridithas-wasip1.manifest.json</a></dd>
      <dt>Source</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/viridithas/viridithas-wasip1-corresponding-source.tar.gz">viridithas-wasip1-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>A modern Rust engine and a consistent top-tier competitor in computer-chess tournaments. Rust's safety guarantees plus Cosmo Bobak's tuning make it both fast and approachable. The browser build here uses relaxed-SIMD for the heavy NNUE inner loops, which WebAssembly finally exposes to engines.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-plentychess">PlentyChess <a class="anchor-link" href="#engines-plentychess" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Yoshie2000 &middot; <a href="https://github.com/Yoshie2000/PlentyChess" rel="noopener">github.com/Yoshie2000/PlentyChess</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/plentychess/plentychess-emscripten-single-thread.manifest.json">/plentychess/plentychess-emscripten-single-thread.manifest.json</a></dd>
      <dt>Source</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz">plentychess-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>A top-tier C++ engine and frequent contender at the top of rating lists. PlentyChess ships its own NNUE architecture and a feature-detected SIMD build so the browser picks the fastest instruction set the device supports. Compiled here with Emscripten, with the processed NNUE network embedded.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-reckless">Reckless <a class="anchor-link" href="#engines-reckless" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>CodeDeliveryService &middot; <a href="https://github.com/codedeliveryservice/Reckless" rel="noopener">github.com/codedeliveryservice/Reckless</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Notice</dt><dd><a href="/reckless/NOTICE.md">/reckless/NOTICE.md</a></dd>
      <dt>Source</dt><dd><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/reckless/reckless-scalar-corresponding-source.tar.gz">scalar</a>, <a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/reckless/reckless-simd128-corresponding-source.tar.gz">SIMD</a>, and <a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz">relaxed SIMD</a></dd>
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (build scripts &amp; release policy)</dd>
    </dl>
    <p>An aggressive, sacrificial Rust engine that lives up to its name &mdash; it plays sharp, entertaining chess and is a favorite in the computer-chess streaming scene.</p>
    <div class="callout info">
      <h4>Public v0 engine</h4>
      <p>Reckless is published through the project CDN for Play, Analysis, and Arena. The browser selects relaxed SIMD or SIMD when supported, and falls back to the scalar WASI build otherwise. Matching source archives for the scalar, SIMD, and relaxed SIMD builds are hosted beside the binaries.</p>
    </div>
  </div>
</section>

<!-- ===== LICENSES ===== -->
<section id="licenses">
  <h2>Licenses &amp; corresponding source <a class="anchor-link" href="#licenses" aria-label="Link to this section">#</a></h2>
  <p class="lead">Most of the engines on this site are GPL- or AGPL-licensed. That's not a box we tick &mdash; it's the reason this project exists in the open. Here's exactly what we redistribute, where each component's source lives, and how we honor the corresponding-source obligation.</p>

  <div class="callout info">
    <h4>The short version</h4>
    <p>Every GPL- or AGPL-licensed engine we redistribute ships with a matching <strong>corresponding-source archive</strong> from the same deployment as the binary. The archive contains the exact upstream source at the pinned commit, our local patches, the build script that produced the browser artifact, and the toolchain used. This is what the GPL means by "the scripts used to control compilation and installation of the executable."</p>
    <p>Upstream GitHub links are <em>not</em> a sufficient corresponding-source offer on their own &mdash; the FSF position and our own policy agree on this &mdash; so each engine has a real <code>*-corresponding-source.tar.gz</code> sitting next to its <code>.wasm</code>. The Maia3 model is under AGPL-3.0, and the default browser artifact is a local int8 quantization of the upstream fp16 file, so the <a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">derivation recipe</a> is part of the source offer.</p>
    <p><strong>Full policy:</strong> <a href="https://github.com/manthedan/0x88/blob/main/docs/engine_artifact_distribution.md">docs/engine_artifact_distribution.md</a> &middot; <strong>Hosted artifact index:</strong> <a href="https://github.com/manthedan/0x88/blob/main/docs/hosted_artifacts.md">docs/hosted_artifacts.md</a> &middot; <strong>This project's source:</strong> <a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a></p>
  </div>

  <h3 id="licenses-per-engine" style="font-size:20px">Per-engine source links <a class="anchor-link" href="#licenses-per-engine" aria-label="Link to this section">#</a></h3>
  <p>Each row lists the binary we ship, its license, and the matching source offer. This is the same information as the metadata blocks above, collected in one place for quick reference.</p>
  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Engine</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">License</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Upstream</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Corresponding source</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Leela Chess Zero</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/LeelaChessZero/lc0" rel="noopener">LeelaChessZero/lc0</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/manthedan/0x88" rel="noopener">0x88 repo</a> (browser runtime + export scripts)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Leela Queen Odds (LQO)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/notune/LeelaQueenOdds" rel="noopener">notune/LeelaQueenOdds</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/models/lc0/lqo_v2.f16.onnx">fp16 source export</a> + <a href="https://github.com/manthedan/0x88" rel="noopener">0x88 repo</a> (QDQ int8 derivation scripts)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Maia3</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/CSSLab/maia3" rel="noopener">CSSLab/maia3</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">Provenance doc</a> + <a href="https://github.com/CSSLab/maia-platform-frontend" rel="noopener">upstream frontend</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stockfish 18</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/nmrugg/stockfish.js" rel="noopener">nmrugg/stockfish.js</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/stockfish/stockfish-18.0.7-corresponding-source.tar.gz">stockfish-18.0.7-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Berserk</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/jhonnold/berserk" rel="noopener">jhonnold/berserk</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz">berserk-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Viridithas</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">MIT</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/cosmobobak/viridithas" rel="noopener">cosmobobak/viridithas</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/viridithas/viridithas-wasip1-corresponding-source.tar.gz">viridithas-wasip1-corresponding-source.tar.gz</a> (license still honored)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">PlentyChess</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/Yoshie2000/PlentyChess" rel="noopener">Yoshie2000/PlentyChess</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz">plentychess-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Reckless</td>
        <td style="padding:10px; font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px"><a href="https://github.com/codedeliveryservice/Reckless" rel="noopener">codedeliveryservice/Reckless</a></td>
        <td style="padding:10px"><a href="/reckless/NOTICE.md">NOTICE.md</a> + <a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/reckless/reckless-scalar-corresponding-source.tar.gz">scalar</a>, <a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/reckless/reckless-simd128-corresponding-source.tar.gz">SIMD</a>, and <a href="https://pub-c3fb64db6e434c738bc86cb1a56d6384.r2.dev/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz">relaxed SIMD</a> source archives</td>
      </tr>
    </tbody>
  </table>

  <blockquote>
    <p>A note on engine names and branding: the names "Stockfish", "Leela Chess Zero", "Berserk", "Viridithas", "PlentyChess", "Reckless", and "Maia" are project trademarks of their respective owners, separate from the code licenses. We use them here solely to identify the engines (nominative use) and do not claim endorsement. Consult each project's own branding guidance if in doubt.</p>
  </blockquote>
</section>

<!-- ===== CDN ===== -->
<section id="cdn">
  <h2>Artifact CDN &amp; caching <a class="anchor-link" href="#cdn" aria-label="Link to this section">#</a></h2>
  <p class="lead">Engine binaries and neural-network models are served from a Cloudflare R2 bucket fronted by a Worker that resolves friendly URLs to content-addressed blobs. Artifacts range from 130 KB to 60+ MB, so caching and compression are critical for the site to function.</p>

  <div class="callout info">
    <h4>Two-plane model</h4>
    <p><strong>Control plane</strong> &mdash; channel and release manifests (mutable, short TTL). <strong>Data plane</strong> &mdash; content-addressed blobs under <code>/artifacts/sha256/</code> (immutable, 1-year edge cache). The Worker maps friendly URLs like <code>/viridithas/viridithas-relaxed-simd128.wasm</code> to the correct blob by reading the channel manifest.</p>
  </div>

  <p>The Worker also handles range requests (delegated to R2's native reader), sets <code>no-transform</code> on binary types to prevent CDN auto-compression of WASM, and attaches CORS/CORP headers for cross-origin isolation. Pre-compression (brotli + gzip sidecars) is done at publish time, not at the edge. Browser-side, the Cache Storage API validates responses by byte length and SHA-256, and compiled <code>WebAssembly.Module</code> objects are cached per worker session.</p>

  <div class="callout warn">
    <h4>Known failure mode: cache poisoning</h4>
    <p>If the origin ever returns a 0-byte or truncated response for a large WASM file, Cloudflare caches it. Symptoms: engine appears to load but never produces moves, eventually timing out. Diagnose by comparing <code>Content-Length</code> between <code>Accept-Encoding: br,gzip</code> and <code>identity</code> requests. Fix by purging the cached URL via the Cloudflare API.</p>
  </div>

  <p>Full architecture, compression pipeline details, artifact size table, and operational playbook (diagnostics, publishing, adding new engines) are in <a href="https://github.com/manthedan/0x88/blob/main/docs/cdn_artifact_caching.md">docs/cdn_artifact_caching.md</a>.</p>
</section>

<!-- ===== REMOVAL ===== -->
<section id="removal">
  <h2>I'm in this project and I don't like it <a class="anchor-link" href="#removal" aria-label="Link to this section">#</a></h2>
  <p class="lead">Thank you for your contribution to chess. The engines and models above are the work of their authors &mdash; this project only packages them for the browser.</p>
  <p>If you maintain one of them and would prefer it not be included here, just <a href="https://github.com/manthedan/0x88/issues/new?title=Engine%20removal%20request&amp;body=Which%20engine%20or%20model%3A%20%0A%0AAre%20you%20a%20maintainer%20or%20rights%20holder%3A%20%0A%0AAnything%20else%3A%20" rel="noopener">open an issue on GitHub</a> and I'll remove it &mdash; no questions asked.</p>
  <p>
    <a class="removal-cta" href="https://github.com/manthedan/0x88/issues/new?title=Engine%20removal%20request&amp;body=Which%20engine%20or%20model%3A%20%0A%0AAre%20you%20a%20maintainer%20or%20rights%20holder%3A%20%0A%0AAnything%20else%3A%20" rel="noopener">Request removal on GitHub <span aria-hidden="true">&rarr;</span></a>
  </p>
</section>

</div><!-- /doc-content -->

</div><!-- /wrap -->
</div><!-- /doc-body -->

</main>
<footer class="site-footer">
  <div class="wrap">
    <span>Engine binaries and source archives are linked above. Everything runs client-side.</span>
    <span class="footer-links">
      <a href="/">Home</a>
      <a href="https://github.com/manthedan/0x88" rel="noopener">Source</a>
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>

<style>
  .doc-hero{
    border-bottom:1px solid var(--rule);
    background:
      radial-gradient(ellipse 70% 60% at 50% 0%, color-mix(in srgb, var(--accent-soft) 55%, transparent), transparent 65%),
      var(--panel);
  }
  .doc-hero .wrap{padding:48px 24px 40px}
  .doc-hero .kicker{font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--accent); margin:0 0 10px}
  .doc-hero h1{font-size:clamp(28px,5vw,42px); margin:0 0 14px; line-height:1.1; letter-spacing:-.015em}
  .doc-hero p{margin:0; color:var(--ink-soft); font-size:16px; max-width:62ch; line-height:1.6}
  .doc-hero p strong{color:var(--ink)}
  .doc-body .wrap{
    padding:48px 24px 64px;
    display:grid; grid-template-columns:220px 1fr;
    gap:56px; align-items:start;
  }
  .toc{
    position:sticky; top:88px;
    font-size:13px; max-height:calc(100vh - 100px); overflow-y:auto;
    padding-right:16px; border-right:1px solid var(--rule);
  }
  .toc h4{
    font-family:var(--sans); font-size:11px;
    text-transform:uppercase; letter-spacing:.07em;
    color:var(--muted-2); margin:0 0 12px; font-weight:600;
  }
  .toc ul{list-style:none; padding:0; margin:0; display:grid; gap:2px}
  .toc a{
    color:var(--muted); text-decoration:none;
    display:block; padding:4px 10px; border-radius:5px;
    line-height:1.35; font-size:13px;
    border-left:2px solid transparent; margin-left:-2px;
  }
  .toc a:hover{color:var(--ink); background:var(--rule-soft)}
  .toc a.active{
    color:var(--accent); font-weight:500;
    background:var(--accent-soft); border-left-color:var(--accent);
  }
  .toc ul ul{margin:2px 0 4px 12px; gap:1px}
  .toc ul ul a{font-size:12px; padding:3px 8px}
  .doc-content{max-width:760px}
  .doc-content section{scroll-margin-top:88px; padding:0 0 48px}
  .doc-content section:last-child{padding-bottom:0}
  .doc-content h2{
    font-size:28px; margin:8px 0 14px;
    line-height:1.2; letter-spacing:-.01em;
    display:flex; align-items:center; gap:10px;
  }
  .doc-content h2::before{
    content:""; display:block; width:28px; height:3px;
    background:var(--accent); border-radius:2px;
  }
  .doc-content h2 .anchor-link,
  .doc-content h3 .anchor-link{
    margin-left:8px; opacity:0;
    color:var(--muted-2); font-size:.7em;
    text-decoration:none; font-family:var(--mono);
    transition:opacity .12s;
  }
  .doc-content h2:hover .anchor-link,
  .doc-content h3:hover .anchor-link{opacity:1}
  .doc-content h2 .anchor-link:hover,
  .doc-content h3 .anchor-link:hover{color:var(--accent); opacity:1}
  .doc-content h3{font-size:20px; margin:36px 0 10px; line-height:1.25}
  .doc-content h3 .anchor-link{font-size:.6em; vertical-align:middle}
  .doc-content > section > p.lead{
    color:var(--muted); margin:0 0 28px;
    font-size:16px; max-width:64ch; line-height:1.65;
  }
  .doc-content p{margin:0 0 14px; line-height:1.7; color:var(--ink-soft); max-width:72ch}
  .doc-content ul, .doc-content ol{
    margin:0 0 16px; padding-left:24px;
    line-height:1.7; color:var(--ink-soft); max-width:70ch;
  }
  .doc-content ul li::marker{color:var(--accent)}
  .doc-content ol li::marker{color:var(--accent); font-weight:600}
  .doc-content code{
    font-family:var(--mono); font-size:.88em;
    background:var(--rule-soft); padding:1px 6px;
    border-radius:4px; color:var(--ink);
  }
  .doc-content a{font-weight:500}
  .page-block{
    background:var(--panel); border:1px solid var(--rule);
    border-radius:var(--radius); padding:24px 28px; margin-bottom:20px;
  }
  .page-block h3{margin:0 0 6px; font-size:20px; display:flex; align-items:center; gap:12px}
  .page-block h3 .pn{
    font-family:var(--mono); font-size:11px;
    color:var(--accent-deep); background:var(--accent-soft);
    padding:3px 10px; border-radius:99px;
    font-weight:700; letter-spacing:.04em; text-transform:uppercase;
  }
  .page-block .pg-url{
    font-family:var(--mono); font-size:12px;
    color:var(--muted); margin:0 0 14px; display:block;
  }
  .page-block p{margin:0 0 12px}
  .page-block p:last-child{margin-bottom:0}
  .page-block ul{margin:0 0 12px}
  .page-block ul li::marker{color:var(--accent)}
  .engine-entry{padding:8px 0 32px; border-bottom:1px solid var(--rule-soft)}
  .engine-entry:last-of-type{border-bottom:none}
  .engine-meta{
    display:grid; grid-template-columns:auto 1fr;
    gap:4px 16px; margin:10px 0 16px;
    font-size:13px; max-width:100%; border-collapse:collapse;
  }
  .engine-meta dt{
    font-family:var(--mono); font-size:11px;
    text-transform:uppercase; letter-spacing:.05em;
    color:var(--muted-2); font-weight:600;
    white-space:nowrap; padding:4px 0; vertical-align:top;
  }
  .engine-meta dd{margin:0; padding:4px 0; color:var(--ink-soft); vertical-align:top}
  .engine-meta dd a{font-weight:500}
  .engine-meta .lic{
    font-family:var(--mono); font-size:11px; font-weight:700;
    padding:2px 9px; border-radius:99px;
    white-space:nowrap; display:inline-block;
  }
  .engine-meta .lic.gpl{background:color-mix(in srgb, var(--gold) 16%, transparent); color:var(--gold)}
  .engine-meta .lic.agpl{background:var(--warn-soft); color:var(--warn)}
  .engine-meta .lic.mit{background:var(--accent-soft); color:var(--accent-deep)}
  .engine-entry details{
    margin-top:8px; border:1px solid var(--rule);
    border-radius:var(--radius-sm); background:var(--panel); overflow:hidden;
  }
  .engine-entry details summary{
    cursor:pointer; padding:10px 16px;
    font-family:var(--mono); font-size:12px;
    color:var(--muted); font-weight:600;
    letter-spacing:.03em; text-transform:uppercase;
    list-style:none; user-select:none;
  }
  .engine-entry details summary::-webkit-details-marker{display:none}
  .engine-entry details summary::before{content:"\25B8  "; color:var(--accent); transition:transform .12s; display:inline-block}
  .engine-entry details[open] summary::before{content:"\25BE  "}
  .engine-entry details summary:hover{color:var(--ink); background:var(--rule-soft)}
  .engine-entry details .details-body{padding:0 16px 14px}
  .engine-entry details .details-body p:last-child{margin-bottom:0}
  .callout{
    border-radius:var(--radius); padding:22px 26px;
    margin:20px 0; border:1px solid var(--rule);
  }
  .callout.info{
    background:var(--accent-soft);
    border-color:color-mix(in srgb, var(--accent) 30%, var(--rule));
  }
  .callout.warn{
    background:var(--warn-soft);
    border-color:color-mix(in srgb, var(--warn) 30%, var(--rule));
  }
  .callout h4{
    margin:0 0 10px; font-size:14px;
    font-family:var(--sans); letter-spacing:.02em;
    display:flex; align-items:center; gap:8px;
  }
  .callout.info h4{color:var(--accent-deep)}
  .callout.warn h4{color:var(--warn)}
  .callout h4::before{
    font-family:var(--mono); font-size:14px;
    width:20px; height:20px; border-radius:4px;
    display:inline-flex; align-items:center; justify-content:center;
  }
  .callout.info h4::before{content:"i"; background:var(--accent); color:var(--panel); font-style:italic}
  .callout.warn h4::before{content:"!"; background:var(--warn); color:var(--panel); font-weight:700}
  .callout p{margin:0 0 10px; line-height:1.65; font-size:14px; max-width:64ch}
  .callout p:last-child{margin:0}
  .callout code{
    font-family:var(--mono); font-size:12px;
    background:var(--panel); padding:1px 6px;
    border-radius:4px; border:1px solid var(--rule);
  }
  .callout a{font-weight:500}
  .removal-cta{
    display:inline-flex; align-items:center; gap:8px;
    margin-top:4px; padding:11px 18px; border-radius:10px;
    background:var(--accent); color:#fbf8f0; font-weight:600; font-size:14px;
    border:1px solid var(--accent); text-decoration:none;
    transition:background .15s ease, transform .15s ease;
  }
  .removal-cta:hover{background:var(--accent-2); transform:translateY(-1px); text-decoration:none}
  .removal-cta span{transition:transform .15s ease}
  .removal-cta:hover span{transform:translateX(3px)}
  .doc-content blockquote{
    margin:24px 0; padding:16px 22px;
    border-left:3px solid var(--rule);
    background:var(--bg-2); border-radius:0 var(--radius-sm) var(--radius-sm) 0;
    color:var(--muted); font-size:14px; line-height:1.65; max-width:68ch;
  }
  .doc-content blockquote p{margin:0; color:var(--muted)}
  @media(max-width:860px){
    .doc-body .wrap{grid-template-columns:1fr; gap:0}
    .toc{
      position:static; max-height:none; padding:0 0 16px;
      margin-bottom:24px; border:none; border-bottom:1px solid var(--rule);
    }
  }
</style>
