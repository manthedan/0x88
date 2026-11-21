<script lang="ts">
import { onMount } from 'svelte';
import { type BrowserCapabilityReport, detectBrowserCapabilities } from '#lib/browserCapabilities.ts';
import AnalysisPreview from '#lib/components/AnalysisPreview.svelte';
import SiteHeader from '#lib/components/SiteHeader.svelte';
import { footerNav } from '#lib/nav.ts';
import { engineFamilyOptions } from '../lc0/engineCatalog.ts';

const title = '0x88 — Chess engines in the browser';
const description =
  'Run Lc0, Stockfish and other chess engines in a browser tab. Compare lines on one position, play Maia3 and run engine tournaments on your own GPU and CPU.';
const github = 'https://github.com/manthedan/0x88';
const engineCount = engineFamilyOptions().length;

let caps: BrowserCapabilityReport | null = null;
let capsFailed = false;

onMount(() => {
  detectBrowserCapabilities()
    .then((report) => (caps = report))
    .catch(() => (capsFailed = true));
});

type Tone = 'ok' | 'warn' | 'neutral';
type DeviceSummary = { tone: Tone; headline: string; detail: string };

function summarize(report: BrowserCapabilityReport | null, failed: boolean): DeviceSummary {
  if (failed) return { tone: 'warn', headline: 'Could not check this browser.', detail: 'Each engine picks its own runtime when it loads.' };
  if (!report) return { tone: 'neutral', headline: 'Checking this browser…', detail: 'Looking for WebGPU, WebAssembly SIMD and threads.' };
  if (!report.wasm) return { tone: 'warn', headline: 'This browser cannot run the engines.', detail: 'WebAssembly is unavailable.' };
  if (report.webgpu === true)
    return { tone: 'ok', headline: 'WebGPU is available for Lc0.', detail: 'Neural engines use the GPU; CPU engines run as WebAssembly.' };
  if (report.webgpu === null)
    return { tone: 'warn', headline: 'WebGPU timed out.', detail: 'Neural engines fall back to WebAssembly if WebGPU stays unavailable.' };
  return {
    tone: 'warn',
    headline: 'No WebGPU in this browser.',
    detail: 'Neural engines fall back to WebAssembly and run slower. CPU engines are unaffected.',
  };
}

$: summary = summarize(caps, capsFailed);
$: facts =
  caps && caps.wasm
    ? [
        { ok: caps.webgpu === true, text: `WebGPU ${caps.webgpu === true ? 'ready' : caps.webgpu === null ? 'timed out' : 'unavailable'}` },
        { ok: caps.wasmSimd, text: caps.wasmRelaxedSimd ? 'Relaxed SIMD' : caps.wasmSimd ? 'WASM SIMD' : 'No WASM SIMD' },
        { ok: caps.threadedWasm, text: caps.threadedWasm ? `${caps.threads} threads` : 'Single-thread' },
      ]
    : [];

const kinds = [
  {
    id: 'selective',
    name: 'Neural tree search',
    text: 'A large neural network ranks candidate moves and evaluates each position, keeping the search narrow and selective.',
    includes: 'Lc0',
    runtime: 'GPU',
    href: '/app/analysis/',
    go: 'Open analysis',
  },
  {
    id: 'wide',
    name: 'Alpha-beta engines',
    text: 'Search far more positions on the CPU, pairing alpha-beta pruning with fast NNUE evaluation.',
    includes: 'Stockfish',
    runtime: 'CPU',
    href: '/app/arena/',
    go: 'Run a tournament',
  },
  {
    id: 'human',
    name: 'Human-style opponents',
    text: 'Models trained to pick human moves at any rating from 600 to 2600, plus an opponent that starts without its queen.',
    includes: 'Maia3',
    runtime: 'GPU',
    href: '/app/play/',
    go: 'Play a game',
  },
];

// Search-shape sketches for the ledger, drawn in a 132×52 box. Each is a list of edges with the
// engine's chosen line marked; they illustrate the row's sentence and carry no measured data.
type Edge = { x1: number; y1: number; x2: number; y2: number; main: boolean; leaf: boolean };
const edge = (a: number[], b: number[], main = false, leaf = false): Edge => ({ x1: a[0], y1: a[1], x2: b[0], y2: b[1], main, leaf });

// Few moves considered at each step, one line followed deep.
const line = [
  [4, 26],
  [29, 26],
  [54, 21],
  [79, 28],
  [104, 23],
  [128, 26],
];
const selective: Edge[] = [
  ...line.slice(1).map((point, i) => edge(line[i], point, true)),
  edge(line[0], [29, 9]),
  edge(line[0], [29, 43]),
  edge([29, 9], [54, 5]),
  edge(line[1], [54, 38]),
  edge(line[2], [79, 10]),
  edge(line[3], [104, 41]),
];

// Every reply examined to a fixed depth: three levels of three.
const wide: Edge[] = (() => {
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const groups = (values: number[]) => [0, 1, 2].map((g) => values.slice(g * (values.length / 3), (g + 1) * (values.length / 3)));
  const leaves = Array.from({ length: 27 }, (_, i) => 2 + (i * 48) / 26);
  const upper = groups(leaves).flatMap((third) => groups(third).map(mean));
  const top = groups(upper).map(mean);
  const best = 16;
  const out: Edge[] = [];
  top.forEach((y, t) => {
    out.push(edge([4, mean(top)], [44, y], t === Math.floor(best / 9)));
  });
  upper.forEach((y, u) => {
    out.push(edge([44, top[Math.floor(u / 3)]], [86, y], u === Math.floor(best / 3)));
  });
  leaves.forEach((y, l) => {
    out.push(edge([86, upper[Math.floor(l / 3)]], [128, y], l === best, true));
  });
  return out;
})();
const trees: Record<string, Edge[]> = { selective, wide };

// The Maia3 rating range, with one rating picked out.
const ratingX = (rating: number) => 4 + ((rating - 600) / 2000) * 124;
const ratingTicks = Array.from({ length: 11 }, (_, i) => ratingX(600 + i * 200));
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>
<SiteHeader landing />
<main id="main" class="home">

<section class="hero" aria-labelledby="hero-h1">
  <div class="wrap hero-grid">
    <div class="hero-copy">
      <h1 id="hero-h1">Lc0 and Stockfish in a browser tab</h1>
      <p class="lede">Compare engines on one position, review games, play Maia3, or run engine tournaments on your own GPU and CPU. No account and no server-side analysis.</p>
      <div class="cta-row">
        <a class="btn btn-primary" href="/app/analysis/">Open analysis</a>
        <a class="btn btn-text" href="/app/play/">Play a game</a>
        <a class="btn btn-text" href="/app/arena/">Run a tournament</a>
      </div>
    </div>

    <div class="preview-wrap"><AnalysisPreview /></div>

    <div id="device-details" class="device" data-tone={summary.tone} aria-live="polite" aria-busy={!caps && !capsFailed}>
      <p class="device-headline"><i aria-hidden="true"></i>{summary.headline}</p>
      <p class="device-detail">{summary.detail}</p>
      <ul class="device-facts" aria-label="Detected features">
        {#each facts as fact (fact.text)}
          <li data-tone={fact.ok ? 'ok' : 'warn'}>{fact.text}</li>
        {/each}
      </ul>
      <a class="device-link" href="/docs/#under-the-hood">How engines run in the browser</a>
    </div>
  </div>
</section>

<section class="engines" aria-labelledby="engines-h">
  <div class="wrap">
    <div class="engines-head">
      <h2 id="engines-h">{engineCount} engines, plus Maia3</h2>
      <p class="thanks">Built with open-source engines and networks. Thanks to their authors. <a href="/docs/#licenses">Licences and source</a></p>
    </div>
    <ul class="kinds">
      {#each kinds as kind (kind.id)}
        <li>
          <a class="kind" href={kind.href}>
            <svg class="kind-sketch" viewBox="0 0 132 52" aria-hidden="true">
              {#if kind.id === 'human'}
                <path class="sketch-rule" d={`M4 30H128${ratingTicks.map((x) => `M${x} 26V34`).join('')}`} />
                <circle class="sketch-pick" cx={ratingX(1500)} cy="30" r="3.5" />
                <text class="sketch-pick-label" x={ratingX(1500)} y="17" text-anchor="middle">1500</text>
                <text x="4" y="49">600</text>
                <text x="128" y="49" text-anchor="end">2600</text>
              {:else}
                {#each trees[kind.id] as e}<line class:main={e.main} class:leaf={e.leaf} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} />{/each}
                {#each trees[kind.id].filter((e) => !e.leaf || e.main) as e}<circle class:main={e.main} cx={e.x2} cy={e.y2} r="1.7" />{/each}
                <circle class="main" cx="4" cy="26" r="2.4" />
              {/if}
            </svg>
            <h3>{kind.name}</h3>
            <p class="kind-text">{kind.text}</p>
            <p class="kind-facts">Includes {kind.includes} <span aria-hidden="true">·</span> {kind.runtime}</p>
            <span class="kind-go">{kind.go} <span aria-hidden="true">→</span></span>
          </a>
        </li>
      {/each}
    </ul>
  </div>
</section>

<section class="local" aria-labelledby="local-h">
  <div class="wrap local-grid">
    <h2 id="local-h">No server does the thinking</h2>
    <div>
      <p>Engine binaries and neural networks download on first use and stay cached in your browser. Your games, positions and PGNs are never uploaded.</p>
      <a href="/docs/">Read the docs</a>
    </div>
  </div>
</section>

</main>
<footer class="site-footer">
  <div class="wrap">
    <span>Open source under GPL-3.0-or-later.</span>
    <span class="footer-links">
      {#each footerNav as item (item.href)}
        <a href={item.href}>{item.label}</a>
      {/each}
      <a href={github} rel="noopener">GitHub</a>
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>

<style>
  /* Home-only contrast: frozen workspace tokens are deliberately untouched. */
  .home{--muted:color-mix(in srgb,var(--ink) 76%,var(--panel)); --rule-strong:color-mix(in srgb,var(--ink) 28%,var(--panel)); --home-boundary:color-mix(in srgb,var(--ink) 56%,var(--panel)); --border-input:var(--home-boundary)}
  /* Text links on Home are ink with an accent underline; the filled button is the only solid colour. */
  .home :is(.btn-text,.device-link,.thanks a,.local a){
    color:var(--ink); font-weight:600; text-decoration:underline; text-decoration-color:var(--accent);
    text-decoration-thickness:1.5px; text-underline-offset:4px;
  }
  .home :is(.btn-text,.device-link,.thanks a,.local a):hover{text-decoration-thickness:2.5px}

  .hero{border-bottom:1px solid var(--rule-strong); background:var(--bg-2)}
  .hero-grid{
    padding-top:52px; padding-bottom:56px;
    display:grid; grid-template-columns:minmax(340px,.78fr) minmax(0,1.22fr);
    grid-template-areas:"copy preview" "device preview";
    grid-template-rows:auto 1fr;
    column-gap:44px; row-gap:32px; align-items:start;
  }
  .hero-copy{grid-area:copy; padding-top:6px}
  .preview-wrap{grid-area:preview; min-width:0; align-self:start}
  .hero h1{
    margin:0 0 22px; font-size:clamp(38px,3.7vw,54px);
    line-height:1.02; letter-spacing:-.035em; font-weight:600; text-wrap:balance;
  }
  .lede{max-width:46ch; margin:0; font-size:17px; line-height:1.6; color:var(--ink-soft)}
  .cta-row{display:flex; flex-wrap:wrap; align-items:center; gap:6px 24px; margin-top:30px}
  .btn{min-height:46px; display:inline-flex; align-items:center; font-size:15px; white-space:nowrap}
  .btn-primary{
    padding:11px 22px; border-radius:7px; background:var(--accent); color:var(--on-accent);
    font-weight:650; text-decoration:none; transition:background .15s;
  }
  .btn-primary:hover{background:var(--accent-hover); color:var(--on-accent); text-decoration:none}

  /* One status line set against the bottom of the hero, level with the preview's lower edge. */
  .device{
    grid-area:device; align-self:end; padding-top:16px; scroll-margin-top:84px;
    border-top:1px solid var(--rule-strong); font-size:13px; line-height:1.55;
  }
  .device p{margin:0}
  .device-headline{display:flex; align-items:baseline; gap:9px; color:var(--ink); font-size:15px; font-weight:650}
  .device-headline i{width:8px; height:8px; flex:0 0 8px; border-radius:50%; background:var(--muted-2); transform:translateY(-1px)}
  .device[data-tone=ok] .device-headline i{background:var(--accent)}
  .device[data-tone=warn] .device-headline i{background:var(--warn)}
  .device-detail{margin-top:2px; padding-left:17px; color:var(--muted)}
  /* The line is reserved before detection finishes so nothing below it moves. */
  .device-facts{
    display:flex; flex-wrap:wrap; min-height:1.55em; margin:8px 0 0; padding:0 0 0 17px; list-style:none;
    color:var(--ink-soft); font-family:var(--mono); font-size:12px; line-height:1.55;
  }
  .device-facts li+li::before{content:"·"; margin:0 .7em; color:var(--muted)}
  .device-facts li[data-tone=warn]{color:var(--warn)}
  .device-link{display:inline-block; margin:8px 0 0 17px}

  .engines .wrap{padding-top:64px; padding-bottom:72px}
  .engines-head{margin-bottom:30px}
  .engines h2{margin:0; font-size:clamp(28px,3.2vw,40px); line-height:1.08; letter-spacing:-.025em; text-wrap:balance}
  .thanks{margin:12px 0 0; color:var(--muted); font-size:16px; line-height:1.6}
  .thanks a{white-space:nowrap}

  /* A ruled ledger, one engine kind per row; the whole row is the link. */
  .kinds{margin:0; padding:0; list-style:none}
  .kind{
    display:grid; grid-template-columns:132px minmax(0,.8fr) minmax(0,1.2fr) 204px;
    column-gap:40px; align-items:center; padding:26px 0;
    border-top:1px solid var(--rule-strong); color:inherit; text-decoration:none;
    transition:background .15s,box-shadow .15s;
  }
  .kinds li:last-child .kind{border-bottom:1px solid var(--rule-strong)}
  /* The rules stay on the text column; only the hover tint reaches past it. */
  .kind:hover{background:var(--panel); box-shadow:-14px 0 var(--panel),14px 0 var(--panel); text-decoration:none}
  .kind-sketch{grid-row:1 / 3; width:132px; height:52px; overflow:visible; color:color-mix(in srgb,var(--ink) 42%,var(--bg))}
  .kind-sketch :is(line,.sketch-rule){stroke:currentColor; stroke-width:1; fill:none}
  .kind-sketch circle{fill:currentColor}
  .kind-sketch line.leaf{stroke-width:.6}
  .kind-sketch line.main{stroke:var(--accent); stroke-width:1.75}
  .kind-sketch :is(circle.main,.sketch-pick){fill:var(--accent)}
  .kind-sketch text{fill:var(--muted); font:500 10px/1 var(--mono)}
  .kind-sketch .sketch-pick-label{fill:var(--ink); font-weight:650}
  .kind h3{grid-row:1 / 3; margin:0; font-size:25px; line-height:1.15; letter-spacing:-.015em}
  .kind-text{grid-row:1 / 3; margin:0; max-width:52ch; color:var(--ink-soft); font-size:15px; line-height:1.55}
  .kind-facts{align-self:end; margin:0; color:var(--muted); font-family:var(--mono); font-size:12px; line-height:1.5}
  .kind-go{align-self:start; margin-top:4px; color:var(--ink); font-size:14px; font-weight:650; white-space:nowrap}
  .kind-go span{display:inline-block; margin-left:3px; color:var(--accent); transition:transform .15s}
  .kind:hover .kind-go span{transform:translateX(4px)}

  .local{border-top:1px solid var(--rule-strong); background:var(--bg-workspace)}
  .local-grid{
    padding-top:60px; padding-bottom:64px; display:grid;
    grid-template-columns:minmax(0,1.1fr) minmax(0,.9fr); gap:20px 44px; align-items:end;
  }
  .local h2{margin:0; max-width:18ch; font-size:clamp(30px,3.6vw,46px); line-height:1.04; letter-spacing:-.03em; text-wrap:balance}
  .local p{margin:0 0 14px; max-width:50ch; color:var(--ink-soft); font-size:16px; line-height:1.6}
  footer.site-footer{margin-top:0}

  /* Keep the three hero links on one row while the copy column is at its narrowest. */
  @media(max-width:1180px){
    .hero-grid{grid-template-columns:minmax(372px,.78fr) minmax(0,1.22fr); column-gap:32px}
    .cta-row{column-gap:16px}
    .btn{font-size:14px}
    .btn-primary{padding-inline:16px}
  }
  @media(max-width:1020px){
    .kind{grid-template-columns:132px minmax(0,1fr) 204px; column-gap:28px}
    .kind h3{grid-row:1; grid-column:2; align-self:end; font-size:22px}
    .kind-text{grid-row:2; grid-column:2; align-self:start; margin-top:6px}
    .kind-facts,.kind-go{grid-column:3}
  }
  @media(max-width:899px){
    .hero-grid{grid-template-columns:1fr; grid-template-areas:"copy" "device" "preview"; grid-template-rows:none; padding-top:44px; padding-bottom:48px; row-gap:26px}
    .hero-copy{max-width:640px; padding-top:0}
    .device{max-width:640px}
    .device-detail{display:none}
    .preview-wrap{width:100%; max-width:820px}
    .local-grid{grid-template-columns:1fr; align-items:start}
    .local h2{max-width:none}
  }
  @media(max-width:600px){
    .hero-grid{padding:20px 16px 32px; row-gap:18px}
    .hero h1{font-size:clamp(34px,10vw,42px); margin-bottom:12px}
    .lede{font-size:15px; line-height:1.55}
    .cta-row{margin-top:18px; gap:4px 28px}
    .btn-primary{flex:1 1 100%; justify-content:center}
    .device{padding-top:12px}
    .device-facts{margin-top:4px}
    .device-link{margin-top:4px}
    .engines .wrap{padding:40px 16px 44px}
    .engines-head{margin-bottom:22px}
    .thanks{font-size:15px}
    .kind{grid-template-columns:minmax(0,1fr) auto; column-gap:16px; margin:0 -16px; padding:20px 16px}
    .kind:hover{box-shadow:none}
    .kind-sketch{grid-row:1; grid-column:2; align-self:center}
    .kind h3{grid-row:1; grid-column:1; align-self:center}
    .kind-text{grid-row:2; grid-column:1 / -1; margin-top:8px}
    .kind-facts{grid-row:3; grid-column:1; align-self:center; margin-top:10px}
    .kind-go{grid-row:3; grid-column:2; align-self:center; margin-top:10px}
    .local-grid{padding:40px 16px 44px}
  }
</style>
