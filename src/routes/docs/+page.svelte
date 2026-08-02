<script lang="ts">
import { onMount } from 'svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';

const title = '0x88 Chess — documentation';
const description = 'Documentation for the 0x88 browser chess workspaces, engine runtimes, upstream projects, licenses, and corresponding-source archives.';
let tocOpen = false;
onMount(() => {
  const headings = document.querySelectorAll('.doc-content section[id], .doc-content h3[id]');
  const tocLinks = document.querySelectorAll<HTMLAnchorElement>('.toc a[href^="#"]');
  if (!('IntersectionObserver' in window) || !headings.length) return;
  const linkMap = new Map<string, HTMLAnchorElement>();
  tocLinks.forEach((link) => {
    const href = link.getAttribute('href');
    if (href) linkMap.set(href.slice(1), link);
  });
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.getAttribute('id');
        const link = id ? linkMap.get(id) : undefined;
        if (!link) return;
        tocLinks.forEach((item) => {
          item.classList.remove('active');
        });
        link.classList.add('active');
      });
    },
    { rootMargin: '-88px 0px -70% 0px', threshold: 0 },
  );
  headings.forEach((heading) => {
    observer.observe(heading);
  });
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
    <h1 id="doc-h1">Browser chess engines and their source</h1>
    <p>
      <strong>0x88.app</strong> is a browser chess platform built from open-source engines and models. This guide explains the app and its browser runtimes. It also credits each upstream project and links every redistributed binary to its license and corresponding source.
    </p>
  </div>
</section>

<div class="doc-body">
<div class="wrap">

<!-- ===== TOC (left rail) ===== -->
<aside class="toc" class:open={tocOpen} aria-label="On this page">
  <h4>On this page</h4>
  <button class="toc-toggle" type="button" aria-expanded={tocOpen} on:click={() => tocOpen = !tocOpen}>
    <span>On this page</span><span aria-hidden="true">{tocOpen ? '−' : '+'}</span>
  </button>
  <ul>
    <li><a href="#pages">The pages</a>
      <ul>
        <li><a href="#pages-analysis">Analysis</a></li>
        <li><a href="#pages-arena">Arena</a></li>
        <li><a href="#pages-play">Play</a></li>
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
        <li><a href="#engines-stormphrax">Stormphrax</a></li>
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
  <p class="lead">0x88 has dedicated workspaces for analysis, engine tournaments, and play. Each workspace runs client-side and loads its engines when needed.</p>

  <div class="page-block">
    <h3 id="pages-analysis"><span class="pn">Analysis</span> Analysis board <a class="anchor-link" href="#pages-analysis" aria-label="Link to this section">#</a></h3>
    <span class="pg-url">/app/analysis</span>
    <p>The Analysis page runs several engines on the same FEN or PGN. Its comparison table places each best move, evaluation, and principal variation together so disagreements are easy to inspect.</p>
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
    <p>The Arena schedules engine matches and tournaments. Games appear live on the board, and finished results feed the standings and evaluation charts. The project also uses this workspace to compare browser engine builds and Leela Chess Zero networks.</p>
    <ul>
      <li><strong>Tournament formats</strong> &mdash; head-to-head, gauntlet, round-robin, with configurable games-per-pairing and color alternation.</li>
      <li><strong>Live standings and Elo estimates</strong> &mdash; running score table, Elo updates after each game, and a final ranking.</li>
      <li><strong>Per-game eval charts and move replays</strong> &mdash; click any finished game to replay it with the eval bar; click any move to jump the board.</li>
      <li><strong>Configurable engine runtime</strong> &mdash; pick engine, strength, opening book, and time control; the arena handles scheduling and the engines do the rest.</li>
    </ul>
  </div>

  <div class="page-block">
    <h3 id="pages-play"><span class="pn">Play</span> Play a game <a class="anchor-link" href="#pages-play" aria-label="Link to this section">#</a></h3>
    <span class="pg-url">/app/play</span>
    <p>The Play page runs a game against Maia3 or one of the available chess engines. Choose a color and strength, then play with takebacks and PGN export available throughout the game.</p>
    <ul>
      <li><strong>Maia3 rating slider (600&ndash;2600)</strong> &mdash; conditions Maia3's move probabilities and human-game outcome predictions on the selected rating.</li>
      <li><strong>Engine opponents at five strength levels</strong> &mdash; adjusts visit, depth, or node limits according to the selected engine.</li>
      <li><strong>Full game lifecycle</strong> &mdash; takebacks, resign, new game, flip board, move list, and one-click PGN export.</li>
    </ul>
    <p>Nothing about your game is sent anywhere. The engine binary and (if needed) the neural network download on first use, then cache locally for next time.</p>
  </div>
</section>

<!-- ===== HUMAN VS COMPUTER PLAY ===== -->
<section id="human-vs-computer">
  <h2>Human vs computer play <a class="anchor-link" href="#human-vs-computer" aria-label="Link to this section">#</a></h2>
  <p class="lead">Maia3 predicts a rating-conditioned human move distribution and human-game outcome. Leela Queen Odds is an Lc0-compatible network trained for games where the engine starts without its queen. The optional contempt controls belong to 0x88's search implementation and model an opponent with limited reply search.</p>

  <div class="callout info">
    <h4>Model boundaries</h4>
    <p>Maia3 is used directly as a move policy. Leela Queen Odds runs through 0x88's PUCT search from its specialized starting position. Contempt search does not make a conventional engine human-like or give it a personality.</p>
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
        <td style="padding:10px; border-bottom:1px solid var(--rule)">A queen-odds policy/value model, optionally combined with a limited-reply opponent model in 0x88's search.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">How does it pick moves?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Samples from, or takes argmax of, a rating-conditioned neural human policy.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Runs 0x88's PUCT search over the specialized network, with optional controls such as <code>drawScore</code> and <code>searchContemptLimit</code>.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">What should the UI promise?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Rating-conditioned human move probabilities and expected game result.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">A specialized queen-odds opponent with clearly labeled search settings.</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Best fit</td>
        <td style="padding:10px">Human sparring, human-move explorer, and rating-conditioned analysis.</td>
        <td style="padding:10px">Queen-odds games and experiments comparing ordinary search with a limited-reply opponent model.</td>
      </tr>
    </tbody>
  </table>

  <p>Maia3's WDL output predicts the result of a human game at the selected rating; it is not an engine centipawn evaluation. The Leela Queen Odds and contempt experiments use 0x88's PUCT search. The full design note lives in <a href="https://github.com/manthedan/0x88/blob/main/docs/human_vs_computer_play.md">docs/human_vs_computer_play.md</a>.</p>
</section>

<!-- ===== NEURAL BROWSER RUNTIMES ===== -->
<section id="neural-runtimes">
  <h2>Neural browser runtimes <a class="anchor-link" href="#neural-runtimes" aria-label="Link to this section">#</a></h2>
  <p class="lead">Lc0 and Maia3 are loaded as browser models rather than compiled UCI executables. 0x88 supplies input encoding, inference, and legal-move mapping. Its own PUCT implementation searches the Lc0 and Leela Queen Odds outputs, while Maia3 is used directly as a rating-conditioned move policy.</p>

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
  <p class="lead">Stockfish, Reckless, Berserk, Viridithas, PlentyChess, Stormphrax, and the Monty lab port are CPU engines compiled to WebAssembly. Browser workers provide their UCI-style control layer.</p>

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
  <p class="lead">These entries separate upstream engine and model facts from 0x88's browser packaging. The <a href="https://computerchess.org.uk/ccrl/4040/" rel="noopener">CCRL 40/15 list</a> and <a href="https://wiki.chessdom.org/Current_Engine_Status" rel="noopener">TCEC engine status</a> provide context for native builds under their own hardware and tournament conditions; neither measures the browser artifacts served here.</p>

  <div class="engine-entry">
    <h3 id="engines-lc0">Leela Chess Zero <a class="anchor-link" href="#engines-lc0" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Project</dt><dd>Leela Chess Zero &middot; <a href="https://github.com/LeelaChessZero/lc0" rel="noopener">github.com/LeelaChessZero/lc0</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0-or-later</span></dd>
      <dt>Manifest</dt><dd><a href="/models/lc0/manifest.json">/models/lc0/manifest.json</a></dd>
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (browser runtime, ONNX export &amp; quantization scripts)</dd>
    </dl>
    <p>Leela Chess Zero is an open-source UCI engine that combines neural-network policy and value evaluation with Monte Carlo tree search. 0x88 does not compile the upstream C++ engine into WebAssembly. It exports selected Lc0 network weights (T1, T3, and BT4) to browser formats, evaluates them with ONNX Runtime WebGPU or custom WebGPU kernels, and searches them with 0x88's own PUCT implementation.</p>
    <details>
      <summary>Architecture &amp; packaging details</summary>
      <div class="details-body">
        <p>The served networks expose policy and value/WDL outputs used by the browser search. 0x88 exports the weights to ONNX and, for selected artifacts, applies QDQ int8 weight quantization while retaining floating-point computation after dequantization. The T1, T3, and BT4 options form a size ladder for different download, memory, and inference budgets.</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-lqo">Leela Queen Odds (LQO) <a class="anchor-link" href="#engines-lqo" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>notune &middot; <a href="https://github.com/notune/LeelaQueenOdds" rel="noopener">github.com/notune/LeelaQueenOdds</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Artifact</dt><dd><a href="https://assets.0x88.app/models/lc0/lqo_v2.f16.qdq8.onnx">/models/lc0/lqo_v2.f16.qdq8.onnx</a> (~96 MB, QDQ int8)</dd>
      <dt>Derived from</dt><dd><a href="https://assets.0x88.app/models/lc0/lqo_v2.f16.onnx">lqo_v2.f16.onnx</a> (fp16 source export)</dd>
      <dt>Manifest</dt><dd><a href="/models/lc0/manifest.json">/models/lc0/manifest.json</a></dd>
    </dl>
    <p>Leela Queen Odds v2 is an Lc0-compatible network trained specifically to play human opponents while the engine starts without its queen. The Play page therefore removes the bot's queen, not the user's. Upstream warns that the network is unsuitable for ordinary-position analysis because it evaluates its specialized queen-odds starting position as equal.</p>
    <details>
      <summary>Search parameters &amp; packaging details</summary>
      <div class="details-body">
        <p>Upstream recommends 15,000 nodes when the network plays White and 12,000 when it plays Black, with <code>ScLimit</code> 40/32, <code>CPuct 1.5</code>, <code>FpuValue 0.4</code>, color-specific <code>DrawScore</code>, and <code>SwapColors</code> when the network plays Black. 0x88 uses lower visit budgets and maps those settings into its own PUCT and search-contempt implementation.</p>
        <p>The served artifact is a QDQ int8 quantization of the fp16 export, derived in-tree and shipped from the LC0 manifest alongside the other big nets (BT4, t3).</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-maia3">Maia3 <a class="anchor-link" href="#engines-maia3" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Authors</dt><dd>Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, and Ashton Anderson (CSSLab) &middot; <a href="https://github.com/CSSLab/maia3" rel="noopener">github.com/CSSLab/maia3</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Paper</dt><dd><a href="https://arxiv.org/abs/2605.19091" rel="noopener">Chessformer: A Unified Architecture for Chess Modeling</a></dd>
      <dt>Provenance</dt><dd><a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">/docs/model_provenance/maia3.md</a></dd>
      <dt>Frontend</dt><dd><a href="https://github.com/CSSLab/maia-platform-frontend" rel="noopener">github.com/CSSLab/maia-platform-frontend</a> (byte-identical upstream fp16 model)</dd>
    </dl>
    <p>Maia-3 is a family of Chessformer models for predicting human moves across skill levels. The browser model returns rating-conditioned move probabilities and human-game WDL predictions rather than a searched centipawn evaluation. The Play page can sample from that policy or choose its highest-probability move.</p>
    <div class="callout info">
      <h4>AGPL source offer</h4>
      <p>The default browser artifact is a local int8 quantization of the upstream fp16 file. The AGPL source offer includes the <a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">derivation recipe</a> for that artifact.</p>
    </div>
  </div>

  <div class="engine-entry">
    <h3 id="engines-stockfish">Stockfish 18 <a class="anchor-link" href="#engines-stockfish" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Project</dt><dd><a href="https://github.com/official-stockfish/Stockfish" rel="noopener">official-stockfish/Stockfish</a> &middot; browser package via <a href="https://github.com/nmrugg/stockfish.js" rel="noopener">nmrugg/stockfish.js</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/stockfish/stockfish-18.0.7-corresponding-source.tar.gz">stockfish-18.0.7-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Stockfish is a community-developed UCI engine derived from Glaurung 2.1. It combines alpha-beta search with NNUE evaluation, and proposed changes are tested through the distributed Fishtest system. Stockfish 18 led the <a href="https://computerchess.org.uk/ccrl/4040/" rel="noopener">July 2026 CCRL 40/15 list</a>, and Stockfish finished first in the <a href="https://wiki.chessdom.org/Current_Engine_Status" rel="noopener">TCEC Season 29 Superfinal</a>. 0x88 ships the Stockfish.js 18.0.7 package in Lite and full-network forms.</p>
    <div class="callout info">
      <h4>Relaxed SIMD candidate</h4>
      <p>The Lite single-thread build has a feature-detected relaxed-SIMD candidate. On our Chromium public-asset validation set &mdash; opening, tactical, quiet middlegame, castling-rights, en-passant, promotion, and endgame FENs; fixed depths 7/9 plus 120 ms movetime &mdash; it measured <strong>2.34M aggregate NPS vs 2.01M</strong> for the baseline Lite single build, a <strong>1.16&times;</strong> overall speedup. Fixed-depth parity was exact: 14/14 same best move, score, and PV. Full single-threaded, Lite pthread, and full pthread relaxed builds remain reproducible local candidates while we gather deeper full-net/threaded evidence. Browsers without the relaxed-SIMD dot-product opcode automatically keep using the baseline Lite single artifact.</p>
    </div>
    <details>
      <summary>Architecture &amp; packaging details</summary>
      <div class="details-body">
        <p>Stockfish uses an efficiently updatable neural network as its static evaluation function inside search. The Stockfish.js package compiles the engine to WebAssembly through Emscripten and embeds the required NNUE data. The manifest serves both the smaller Lite package and the full-network variants.</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-berserk">Berserk <a class="anchor-link" href="#engines-berserk" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Jay Honnold &middot; <a href="https://github.com/jhonnold/berserk" rel="noopener">github.com/jhonnold/berserk</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz">berserk-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Berserk 14 is Jay Honnold's UCI engine written in C. Its documented implementation uses principal-variation search with standard alpha-beta pruning techniques and a bucketed NNUE evaluation. 0x88 compiles the pinned release with Emscripten and serves scalar, WebAssembly SIMD, and relaxed-SIMD variants.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-viridithas">Viridithas <a class="anchor-link" href="#engines-viridithas" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Cosmo Bobak &middot; <a href="https://github.com/cosmobobak/viridithas" rel="noopener">github.com/cosmobobak/viridithas</a></dd>
      <dt>License</dt><dd><span class="lic mit">MIT</span> at pinned commit <a href="https://github.com/cosmobobak/viridithas/commit/20d7402065cae084715183e019fdd18089e2dfac" rel="noopener"><code>20d7402</code></a>; upstream changed to AGPL-3.0-only in version 20.0.0</dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/viridithas/viridithas-wasip1-corresponding-source.tar.gz">viridithas-wasip1-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Viridithas is Cosmo Bobak's Rust UCI engine with NNUE evaluation. The project documents that its network training does not use evaluations produced by other chess engines. 0x88 pins commit <code>20d7402</code>, whose license is MIT, and serves scalar, WebAssembly SIMD, and relaxed-SIMD WASI builds.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-plentychess">PlentyChess <a class="anchor-link" href="#engines-plentychess" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Yoshie2000 &middot; <a href="https://github.com/Yoshie2000/PlentyChess" rel="noopener">github.com/Yoshie2000/PlentyChess</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz">plentychess-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>PlentyChess 7.0.66 is Patrick Leonhardt's C++17 UCI engine. Its threat-input NNUE is trained on more than 15 billion self-generated standard-chess and Fischer-random positions, with self-distillation used for part of the training. 0x88 compiles the pinned revision with Emscripten and serves scalar, SSE4.1-shaped WebAssembly SIMD, and relaxed-SIMD variants with the network embedded.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-stormphrax">Stormphrax <a class="anchor-link" href="#engines-stormphrax" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Ciekce &middot; <a href="https://github.com/Ciekce/Stormphrax" rel="noopener">github.com/Ciekce/Stormphrax</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0-or-later</span></dd>
      <dt>Network</dt><dd><code>undertown.nnue</code>, trained from Stormphrax self-play data</dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/stormphrax/stormphrax-emscripten-single-thread-corresponding-source.tar.gz">stormphrax-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Stormphrax 8.0.0 is Ciekce's C++ UCI engine with an explicit Chess960 focus. Its NNUE starts from random weights and is trained on data generated by Stormphrax rather than a third-party engine. 0x88 serves a single-thread Emscripten build with WebAssembly SIMD and a feature-detected relaxed-SIMD candidate.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-reckless">Reckless <a class="anchor-link" href="#engines-reckless" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Project</dt><dd>CodeDeliveryService &middot; <a href="https://github.com/codedeliveryservice/Reckless" rel="noopener">github.com/codedeliveryservice/Reckless</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Notice</dt><dd><a href="/reckless/NOTICE.md">/reckless/NOTICE.md</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/reckless/reckless-scalar-corresponding-source.tar.gz">scalar</a>, <a href="https://assets.0x88.app/reckless/reckless-simd128-corresponding-source.tar.gz">SIMD</a>, and <a href="https://assets.0x88.app/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz">relaxed SIMD</a></dd>
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (build scripts &amp; release policy)</dd>
    </dl>
    <p>Reckless is a competitive Rust UCI engine using alpha-beta search and NNUE evaluation. Native Reckless 0.9.0 placed second on the <a href="https://computerchess.org.uk/ccrl/4040/" rel="noopener">July 2026 CCRL 40/15 list</a>, and a development build finished second in the <a href="https://wiki.chessdom.org/Current_Engine_Status" rel="noopener">TCEC Season 29 Superfinal</a>. 0x88 pins development commit <a href="https://github.com/codedeliveryservice/Reckless/commit/0010617448bdef4c8cd7d4f4825b7e42c8bc262a" rel="noopener"><code>0010617</code></a> and serves scalar, WebAssembly SIMD, and relaxed-SIMD WASI builds.</p>
    <div class="callout info">
      <h4>Public v0 engine</h4>
      <p>Reckless is published through the project CDN for Play, Analysis, and Arena. The browser selects relaxed SIMD or SIMD when supported, and falls back to the scalar WASI build otherwise. Matching source archives for the scalar, SIMD, and relaxed SIMD builds are hosted beside the binaries.</p>
    </div>
  </div>
</section>

<!-- ===== LICENSES ===== -->
<section id="licenses">
  <h2>Licenses &amp; corresponding source <a class="anchor-link" href="#licenses" aria-label="Link to this section">#</a></h2>
  <p class="lead">Most engines on this site use GPL or AGPL licenses. This section identifies each redistributed artifact, its upstream project, and its corresponding-source archive.</p>

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
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0-or-later</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/LeelaChessZero/lc0" rel="noopener">LeelaChessZero/lc0</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/manthedan/0x88" rel="noopener">0x88 repo</a> (browser runtime + export scripts)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Leela Queen Odds (LQO)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/notune/LeelaQueenOdds" rel="noopener">notune/LeelaQueenOdds</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/models/lc0/lqo_v2.f16.onnx">fp16 source export</a> + <a href="https://github.com/manthedan/0x88" rel="noopener">0x88 repo</a> (QDQ int8 derivation scripts)</td>
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
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/stockfish/stockfish-18.0.7-corresponding-source.tar.gz">stockfish-18.0.7-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Berserk</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/jhonnold/berserk" rel="noopener">jhonnold/berserk</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz">berserk-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Viridithas</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">MIT (pinned commit)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/cosmobobak/viridithas/commit/20d7402065cae084715183e019fdd18089e2dfac" rel="noopener">cosmobobak/viridithas@20d7402</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/viridithas/viridithas-wasip1-corresponding-source.tar.gz">viridithas-wasip1-corresponding-source.tar.gz</a> (license still honored)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">PlentyChess</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/Yoshie2000/PlentyChess" rel="noopener">Yoshie2000/PlentyChess</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz">plentychess-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stormphrax</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0-or-later</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/Ciekce/Stormphrax/tree/v8.0.0" rel="noopener">Ciekce/Stormphrax v8.0.0</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/stormphrax/stormphrax-emscripten-single-thread-corresponding-source.tar.gz">stormphrax-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Reckless</td>
        <td style="padding:10px; font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px"><a href="https://github.com/codedeliveryservice/Reckless" rel="noopener">codedeliveryservice/Reckless</a></td>
        <td style="padding:10px"><a href="/reckless/NOTICE.md">NOTICE.md</a> + <a href="https://assets.0x88.app/reckless/reckless-scalar-corresponding-source.tar.gz">scalar</a>, <a href="https://assets.0x88.app/reckless/reckless-simd128-corresponding-source.tar.gz">SIMD</a>, and <a href="https://assets.0x88.app/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz">relaxed SIMD</a> source archives</td>
      </tr>
    </tbody>
  </table>

  <blockquote>
    <p>A note on engine names and branding: the names "Stockfish", "Leela Chess Zero", "Berserk", "Viridithas", "PlentyChess", "Stormphrax", "Reckless", and "Maia" are project trademarks of their respective owners, separate from the code licenses. We use them here solely to identify the engines (nominative use) and do not claim endorsement. Consult each project's own branding guidance if in doubt.</p>
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

  <p>Full architecture, compression pipeline details, artifact size table, and operational playbook (diagnostics, publishing, adding new engines) are in <a href="https://github.com/manthedan/0x88/blob/main/docs/cdn_artifact_caching.md">docs/cdn_artifact_caching.md</a>.</p>
</section>

<!-- ===== REMOVAL ===== -->
<section id="removal">
  <h2>I'm in this project and I don't like it <a class="anchor-link" href="#removal" aria-label="Link to this section">#</a></h2>
  <p class="lead">The engines and models above belong to their respective projects. 0x88 packages them for browser use.</p>
  <p>Maintainers and rights holders can request removal by <a href="https://github.com/manthedan/0x88/issues/new?title=Engine%20removal%20request&amp;body=Which%20engine%20or%20model%3A%20%0A%0AAre%20you%20a%20maintainer%20or%20rights%20holder%3A%20%0A%0AAnything%20else%3A%20" rel="noopener">opening an issue on GitHub</a>. The requested engine or model will be removed.</p>
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
    display:grid; grid-template-columns:220px minmax(0,1fr);
    gap:56px; align-items:start;
  }
  /* Wide tables (per-engine source links) scroll inside their own box
     instead of forcing the whole column past narrow viewports. */
  .doc-content :global(table){display:block; overflow-x:auto}
  /* Long slash-joined tokens (best/good/inaccuracy/…) have no natural
     break points; let them wrap rather than overflow on phones. */
  .doc-content{overflow-wrap:break-word}
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
  .toc-toggle{display:none}
  .toc ul{list-style:none; padding:0; margin:0; display:grid; gap:2px}
  .toc a{
    color:var(--muted); text-decoration:none;
    display:block; padding:4px 10px; border-radius:5px;
    line-height:1.35; font-size:13px;
    border-left:2px solid transparent; margin-left:-2px;
  }
  .toc a:hover{color:var(--ink); background:var(--rule-soft)}
  :global(.toc a.active){
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
  .doc-content ul{
    margin:0 0 16px; padding-left:24px;
    line-height:1.7; color:var(--ink-soft); max-width:70ch;
  }
  .doc-content ul li::marker{color:var(--accent)}
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
  .callout h4{
    margin:0 0 10px; font-size:14px;
    font-family:var(--sans); letter-spacing:.02em;
    display:flex; align-items:center; gap:8px;
  }
  .callout.info h4{color:var(--accent-deep)}
  .callout h4::before{
    font-family:var(--mono); font-size:14px;
    width:20px; height:20px; border-radius:4px;
    display:inline-flex; align-items:center; justify-content:center;
  }
  .callout.info h4::before{content:"i"; background:var(--accent); color:var(--panel); font-style:italic}
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
    .doc-body .wrap{grid-template-columns:minmax(0,1fr); gap:0}
    .toc{
      position:static; max-height:none; padding:0;
      margin-bottom:28px; border:1px solid var(--rule); border-radius:var(--radius);
      background:var(--panel); overflow:hidden;
    }
    .toc h4{display:none}
    .toc-toggle{
      width:100%; min-height:46px; padding:10px 14px; border:0; border-radius:0;
      display:flex; align-items:center; justify-content:space-between;
      background:transparent; color:var(--ink); font-family:var(--mono); font-size:11px;
      text-transform:uppercase; letter-spacing:.08em; font-weight:650;
    }
    .toc.open .toc-toggle{border-bottom:1px solid var(--rule)}
    .toc:not(.open)>ul{display:none}
    .toc>ul{padding:10px 8px 12px}
  }
</style>
