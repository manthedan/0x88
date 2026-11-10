<script lang="ts">
import { onMount } from 'svelte';
import AnalysisPreview from '$lib/components/AnalysisPreview.svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';
import { type BrowserCapabilityReport, detectBrowserCapabilities } from '$lib/browserCapabilities';
import { footerNav } from '$lib/nav';
import { engineFamilyOptions } from '../lc0/engineCatalog.ts';

const title = '0x88 | Browser chess engine workbench';
const description = 'See where chess engines disagree on one position, play human-style opponents and run engine tournaments. Computation stays on your device.';
const github = 'https://github.com/manthedan/0x88';
const engineCount = engineFamilyOptions().length;

// Alphabetical, so the order implies no ranking. 0x88's own Centipawn isn't credited here.
const credits = [
  ['Berserk', 'berserk'],
  ['Leela Chess Zero', 'lc0'],
  ['Leela Queen Odds', 'lqo'],
  ['Maia3', 'maia3'],
  ['PlentyChess', 'plentychess'],
  ['Reckless', 'reckless'],
  ['Stockfish', 'stockfish'],
  ['Stormphrax', 'stormphrax'],
  ['Viridithas', 'viridithas'],
];

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
  if (failed) return { tone: 'warn', headline: 'Could not check this browser.', detail: 'Each engine picks a fallback runtime when it loads.' };
  if (!report) return { tone: 'neutral', headline: 'Checking this browser…', detail: 'Looking for WebGPU, WebAssembly SIMD and threads.' };
  if (!report.wasm) return { tone: 'warn', headline: 'This browser cannot run the engines.', detail: 'WebAssembly is unavailable.' };
  if (report.webgpu === true) return { tone: 'ok', headline: 'Neural engines run on WebGPU here.', detail: 'Other engines run as WebAssembly on the CPU.' };
  if (report.webgpu === null)
    return { tone: 'warn', headline: 'WebGPU did not answer.', detail: 'Neural engines fall back to WebAssembly if the GPU stays unavailable.' };
  return { tone: 'warn', headline: 'No WebGPU in this browser.', detail: 'Neural engines fall back to WebAssembly and run slower. CPU engines are unaffected.' };
}

$: summary = summarize(caps, capsFailed);
$: readiness = capsFailed ? 'Device check unavailable'
  : !caps ? 'Checking this device…'
  : !caps.wasm ? 'WebAssembly unavailable'
  : caps.webgpu === true ? `WebGPU ready · ${caps.threadedWasm ? `${caps.threads} threads` : 'single-thread'}`
  : `WebGPU ${caps.webgpu === null ? 'unknown' : 'unavailable'} · WASM fallback`;
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
      <h1 id="hero-h1">Your chess engine workbench.</h1>
      <p class="lede">See where engines disagree, play human-style opponents and run tournaments in your browser, with no account or install.</p>
      <div class="cta-row">
        <a class="btn btn-primary" href="/app/analysis/">Open analysis <span aria-hidden="true">→</span></a>
        <a class="btn btn-secondary" href="/app/play/">Play a game</a>
        <a class="btn btn-arena" href="/app/arena/"><span aria-hidden="true">♜</span> Arena · Engine tournaments <span aria-hidden="true">→</span></a>
      </div>
      <a class="device-readiness" data-tone={summary.tone} href="/docs/#neural-runtimes" aria-live="polite"><i aria-hidden="true"></i>{readiness}<span aria-hidden="true">↓</span></a>
    </div>

    <div class="preview-wrap"><AnalysisPreview /></div>

    <div id="device-details" class="device" data-tone={summary.tone} aria-live="polite" aria-busy={!caps && !capsFailed}>
      <p class="device-label">This device</p>
      <p class="device-headline"><i aria-hidden="true"></i>{summary.headline}</p>
      <p class="device-detail">{summary.detail}</p>
      {#if caps && caps.wasm}
        <ul class="device-chips" aria-label="Detected features">
          <li data-tone={caps.webgpu === true ? 'ok' : 'warn'}>WebGPU {caps.webgpu === true ? 'ready' : caps.webgpu === null ? 'unknown' : 'unavailable'}</li>
          <li data-tone={caps.wasmSimd ? 'ok' : 'warn'}>{caps.wasmRelaxedSimd ? 'Relaxed SIMD' : caps.wasmSimd ? 'WASM SIMD' : 'No WASM SIMD'}</li>
          <li data-tone={caps.threadedWasm ? 'ok' : 'warn'}>{caps.threadedWasm ? `${caps.threads} threads` : 'Single-thread'}</li>
        </ul>
      {/if}
      <a class="device-link" href="/docs/#neural-runtimes">How engines run in the browser <span aria-hidden="true">→</span></a>
    </div>
  </div>
</section>

<section class="engines" aria-labelledby="engines-h">
  <div class="wrap">
    <p class="eyebrow"><span></span> Engines</p>
    <h2 id="engines-h">{engineCount} engines, plus human-style opponents</h2>
    <div class="kinds">
      <a class="kind" href="/app/analysis/">
        <h3>Neural networks</h3>
        <span class="kind-runtime">GPU</span>
        <p>A trained network judges each position, and the search stays narrow and selective.</p>
        <span class="kind-go">Open analysis <span aria-hidden="true">→</span></span>
      </a>
      <a class="kind" href="/app/arena/">
        <h3>Alpha-beta engines</h3>
        <span class="kind-runtime">CPU</span>
        <p>These search far more positions and evaluate each one quickly.</p>
        <span class="kind-go">Run a tournament <span aria-hidden="true">→</span></span>
      </a>
      <a class="kind" href="/app/play/">
        <h3>Human-style opponents</h3>
        <span class="kind-runtime">GPU</span>
        <p>Play the move a person at your chosen rating would, from 600 to 2600, or face a bot that starts without its queen.</p>
        <span class="kind-go">Play a game <span aria-hidden="true">→</span></span>
      </a>
    </div>
    <p class="credits">
      Built with open-source work from
      {#each credits as [name, anchor], i (anchor)}<a href={`/docs/#engines-${anchor}`}>{name}</a>{i < credits.length - 2 ? ', ' : i === credits.length - 2 ? ' and ' : ''}{/each}.
      Thanks to their authors. <a href="/docs/#licenses">Licenses and source</a>
    </p>
  </div>
</section>

<section class="principle" aria-label="Local-first principle">
  <div class="wrap principle-grid">
    <div>
      <p class="eyebrow"><span></span> Privacy</p>
      <h2>Local chess computation</h2>
    </div>
    <p>Engine binaries and neural networks load when needed, then cache in your browser. Your chess data remains on your hardware.</p>
    <a href="/docs/">Read the technical notes <span aria-hidden="true">→</span></a>
  </div>
</section>

</main>
<footer class="site-footer">
  <div class="wrap landing-footer-wrap">
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
  .hero{border-bottom:1px solid var(--rule); background:var(--bg-2)}
  .hero-grid{
    padding-top:44px; padding-bottom:48px;
    display:grid; grid-template-columns:minmax(360px,.8fr) minmax(0,1.2fr);
    grid-template-areas:"copy preview" "device preview";
    grid-template-rows:auto 1fr;
    column-gap:24px; row-gap:24px; align-items:start;
  }
  .hero-copy{grid-area:copy; max-width:600px}
  .preview-wrap{grid-area:preview; min-width:0; align-self:start}
  .eyebrow{
    display:flex; align-items:center; gap:9px; margin:0 0 16px;
    font-family:var(--mono); font-size:12px; line-height:1.4;
    text-transform:uppercase; letter-spacing:.09em; color:var(--accent-soft-text); font-weight:650;
  }
  .eyebrow span{width:24px; height:2px; background:var(--accent); flex:0 0 auto}
  .hero h1{
    max-width:none; margin:0 0 20px; font-size:clamp(40px,4vw,58px);
    line-height:1.02; letter-spacing:-.035em; font-weight:600;
  }
  .lede{max-width:54ch; margin:0; font-size:17px; line-height:1.65; color:var(--ink-soft)}
  .cta-row{display:flex; flex-wrap:wrap; gap:10px; margin-top:28px}
  .btn{
    min-height:46px; padding:11px 19px; display:inline-flex; align-items:center; gap:11px;
    border:1px solid var(--border-input); border-radius:7px;
    font-size:14px; font-weight:650; text-decoration:none; white-space:nowrap;
    transition:background .15s,border-color .15s,color .15s,transform .15s;
  }
  .btn:hover{text-decoration:none; transform:translateY(-1px)}
  .btn-primary{background:var(--accent); border-color:var(--accent); color:var(--on-accent)}
  .btn-primary:hover{background:var(--accent-hover); color:var(--on-accent)}
  .btn-secondary{background:var(--panel); color:var(--ink)}
  .btn-secondary:hover{border-color:var(--accent)}
  .btn-arena{flex:0 0 auto; justify-content:flex-start; min-height:40px; padding-block:7px; background:transparent; color:var(--ink-soft); font-size:13px}
  .btn-arena:hover{background:var(--panel); border-color:var(--accent)}
  .btn-arena span{color:var(--accent-soft-text)}
  .device-readiness{display:flex; align-items:center; gap:7px; width:fit-content; min-height:28px; margin-top:9px; color:var(--ink-soft); font-size:13px}
  .device-readiness i{width:7px; height:7px; flex-shrink:0; border-radius:50%; background:var(--muted)}
  .device-readiness[data-tone=ok] i{background:var(--accent)}
  .device-readiness[data-tone=warn] i{background:var(--warn)}
  .device-readiness>span{color:var(--ink); margin-left:3px}
  /* Wide layout: the full device card sits directly below, so the one-line summary would repeat it. */
  @media(min-width:900px){ .device-readiness{display:none} }

  /* Device check: live WebGPU / SIMD / threads result, under the hero actions. */
  .device{
    grid-area:device; align-self:start; max-width:460px; padding:14px 16px; scroll-margin-top:84px;
    border:1px solid var(--rule-strong); border-radius:8px;
    background:var(--panel);
  }
  .device p{margin:0}
  .device-label{font-family:var(--mono); font-size:12px; letter-spacing:.09em; text-transform:uppercase; color:var(--muted)}
  .device .device-headline{margin-top:6px}
  .device .device-detail{margin-top:3px}
  .device-headline{display:flex; align-items:center; gap:8px;  color:var(--ink); font-size:15px; font-weight:650}
  .device-headline i{width:8px; height:8px; flex:0 0 8px; border-radius:50%; background:var(--muted-2)}
  .device[data-tone=ok] .device-headline i{background:var(--accent)}
  .device[data-tone=warn] .device-headline i{background:var(--warn)}
  .device-detail{ color:var(--muted); font-size:13px; line-height:1.5}
  .device-chips{display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 0; padding:0; list-style:none}
  .device-chips li{padding:2px 8px; border:1px solid var(--rule-strong); border-radius:999px; color:var(--ink-soft); font-family:var(--mono); font-size:12px}
  .device-chips li[data-tone=ok]{color:var(--accent-soft-text); border-color:var(--home-boundary)}
  .device-chips li[data-tone=warn]{color:var(--warn); border-color:color-mix(in srgb, var(--warn) 50%, var(--rule))}
  .device-link{display:inline-block; margin-top:10px; color:var(--ink); font-size:13px; font-weight:650}
  .device-link span{color:var(--accent); margin-left:3px}



  .engines .wrap{padding-top:56px; padding-bottom:52px}
  .engines .eyebrow{margin:0 0 10px}
  .engines h2{margin:0 0 26px; font-size:clamp(28px,3.4vw,38px); line-height:1.1; letter-spacing:-.02em}
  .kinds{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px}
  .kind{display:grid; grid-template-columns:1fr auto; align-content:start; gap:8px 12px; padding:18px; border:1px solid var(--rule-strong); border-radius:12px; color:inherit; text-decoration:none}
  .kind:hover{background:var(--panel); text-decoration:none}
  .kind h3{margin:0; font-size:20px; line-height:1.2}
  .kind-runtime{align-self:center; padding:1px 6px; border:1px solid var(--rule-strong); border-radius:999px; color:var(--ink-soft); font-family:var(--mono); font-size:12px}
  .kind p{grid-column:1 / -1; margin:0; color:var(--muted); font-size:14px; line-height:1.55}
  .kind-go{grid-column:1 / -1; margin-top:4px; color:var(--ink); font-size:13px; font-weight:650}
  .kind-go span{color:var(--accent); margin-left:3px}
  .credits{margin:18px 0 0; color:var(--muted); font-size:13px; line-height:1.6}
  .credits a{color:inherit; text-decoration:underline; text-decoration-color:var(--rule-strong); text-underline-offset:3px}
  .credits a:hover{color:var(--ink)}
  @media (max-width:760px){
    .engines .wrap{padding:44px 16px 40px}
    .kinds{grid-template-columns:1fr; gap:10px}
    .kind{padding:14px}
  }
  .principle{border-top:1px solid var(--rule-strong); background:var(--bg-workspace)}
  .principle-grid{
    padding-top:44px; padding-bottom:44px; display:grid;
    grid-template-columns:minmax(250px,.8fr) minmax(280px,1fr) 190px;
    gap:34px; align-items:center;
  }
  .principle .eyebrow{margin:0 0 7px}
  .principle h2{margin:0; font-size:28px; line-height:1.1}
  .principle-grid>p{margin:0; color:var(--muted); line-height:1.65; font-size:14px}
  .principle a{white-space:nowrap; color:var(--ink); font-weight:650; font-size:13px}
  .principle a span{color:var(--accent); margin-left:5px}
  footer.site-footer{margin-top:0}
  :global(.landing-footer-wrap){justify-content:flex-end}

  @media(max-width:1200px){
  }
  @media(max-width:899px){
    .hero-grid{grid-template-columns:1fr; grid-template-areas:"copy" "preview"; grid-template-rows:none; padding-top:48px; padding-bottom:52px}
    .hero-copy{max-width:650px; padding-top:0}
    .hero h1{max-width:14ch}
    .preview-wrap{width:100%; max-width:820px}
    .device{display:none}
  }
  @media(max-width:820px){
    .principle-grid{grid-template-columns:1fr 1.2fr}
    .principle-grid>a{grid-column:1}
  }
  @media(max-width:600px){
    .hero-grid{padding:16px 16px 32px; row-gap:14px}
    .hero h1{font-size:clamp(32px,8.7vw,40px); max-width:none; margin-bottom:10px}
    .lede{font-size:14.5px; line-height:1.5}
    .cta-row{margin-top:14px; gap:6px}
    .cta-row .btn{flex:1 1 0; justify-content:center; padding-inline:12px}
    .cta-row .btn-arena{flex:0 0 auto; min-height:34px; padding-block:4px; justify-content:flex-start}
    .device-readiness{margin-top:5px; min-height:26px; font-size:12px}
    .device{max-width:none}
    .principle-grid{padding:38px 16px; grid-template-columns:1fr; gap:12px}
    .principle-grid>a{grid-column:auto; margin-top:5px}
  }
</style>
