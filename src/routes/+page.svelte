<script lang="ts">
import { onMount } from 'svelte';
import AnalysisPreview from '$lib/components/AnalysisPreview.svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';
import { type BrowserCapabilityReport, detectBrowserCapabilities } from '$lib/browserCapabilities';
import { HUMAN_OPPONENTS, type RosterEngine, searchEngineRoster } from '$lib/engineRoster';
import { footerNav, workspaces } from '$lib/nav';

const title = '0x88 | Browser chess engine workbench';
const description = 'Play against Maia3, compare browser chess engines on one position, or run engine tournaments. Computation stays on your device.';
const github = 'https://github.com/manthedan/0x88';

const searchEngines = searchEngineRoster();

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
  if (report.webgpu === true) return { tone: 'ok', headline: 'This device runs Lc0 on WebGPU.', detail: 'Neural engines use the GPU; CPU engines run as WebAssembly.' };
  if (report.webgpu === null)
    return { tone: 'warn', headline: 'WebGPU did not answer.', detail: 'Neural engines fall back to WebAssembly if the GPU stays unavailable.' };
  return { tone: 'warn', headline: 'No WebGPU in this browser.', detail: 'Neural engines fall back to WebAssembly and run slower. CPU engines are unaffected.' };
}

function runtimeLabel(engine: RosterEngine, report: BrowserCapabilityReport | null): { text: string; tone: Tone; title: string } {
  if (engine.runtime === 'cpu') return { text: 'CPU', tone: 'neutral', title: 'Runs as WebAssembly on the CPU' };
  if (!report) return { text: 'GPU', tone: 'neutral', title: 'Uses WebGPU when available' };
  return report.webgpu === true
    ? { text: 'GPU', tone: 'ok', title: 'Runs on WebGPU on this device' }
    : { text: 'GPU · fallback', tone: 'warn', title: 'No WebGPU here: falls back to WebAssembly' };
}

$: summary = summarize(caps, capsFailed);
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>
<SiteHeader />
<main id="main">

<section class="hero" aria-labelledby="hero-h1">
  <div class="wrap hero-grid">
    <div class="hero-copy">
      <h1 id="hero-h1">A chess engine workbench in your browser.</h1>
      <p class="lede">
        0x88 runs games and analysis on your device, from a Maia3 opponent to
        multi-engine tournaments. It needs no account or installation.
      </p>
      <div class="cta-row">
        <a class="btn btn-primary" href="/app/analysis/">Open analysis <span aria-hidden="true">→</span></a>
        <a class="btn btn-secondary" href="/app/play/">Play a game</a>
      </div>
      <a class="cta-tertiary" href="/app/arena/">Or run an engine tournament in the <b>Arena</b> <span aria-hidden="true">→</span></a>
    </div>

    <div class="preview-wrap"><AnalysisPreview /></div>

    <div class="device" data-tone={summary.tone} aria-live="polite" aria-busy={!caps && !capsFailed}>
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
    <div class="section-intro">
      <p class="eyebrow"><span></span> Engines</p>
      <h2 id="engines-h">Analysis engines and human-like opponents</h2>
      <p>Each engine downloads the first time you use it, then stays cached in this browser.</p>
    </div>
    <div class="roster">
      <div class="roster-group">
        <div class="roster-head">
          <h3>Analysis &amp; Arena</h3>
          <a href="/app/analysis/">Open analysis <span aria-hidden="true">→</span></a>
        </div>
        <ul class="roster-list">
          {#each searchEngines as engine (engine.id)}
            {@const runtime = runtimeLabel(engine, caps)}
            <li style={`--engine-color:${engine.color}`}>
              <span class="roster-avatar" aria-hidden="true">{engine.monogram}</span>
              <a class="roster-name" href={engine.docHref}>{engine.name}</a>
              <span class="roster-runtime" data-tone={runtime.tone} title={runtime.title}>{runtime.text}</span>
              {#if engine.role}<p>{engine.role}</p>{/if}
            </li>
          {/each}
        </ul>
      </div>
      <div class="roster-group">
        <div class="roster-head">
          <h3>Human-style opponents</h3>
          <a href="/app/play/">Open play <span aria-hidden="true">→</span></a>
        </div>
        <ul class="roster-list roster-list-play">
          {#each HUMAN_OPPONENTS as engine (engine.id)}
            {@const runtime = runtimeLabel(engine, caps)}
            <li style={`--engine-color:${engine.color}`}>
              <span class="roster-avatar" aria-hidden="true">{engine.monogram}</span>
              <a class="roster-name" href={engine.docHref}>{engine.name}</a>
              <span class="roster-runtime" data-tone={runtime.tone} title={runtime.title}>{runtime.text}</span>
              <p>{engine.role}</p>
            </li>
          {/each}
        </ul>
        <p class="roster-note">Every analysis engine is also a Play opponent, at five strength levels.</p>
      </div>
    </div>
  </div>
</section>

<section class="modes" aria-labelledby="modes-h">
  <div class="wrap">
    <div class="section-intro">
      <p class="eyebrow"><span></span> Workspaces</p>
      <h2 id="modes-h">Analyse, compete, play</h2>
    </div>
    <div class="mode-index">
      {#each workspaces as ws, i (ws.href)}
        <a class="mode" href={ws.href}>
          <span class="mode-number">{String(i + 1).padStart(2, '0')}</span>
          <h3>{ws.title}</h3>
          <p>{ws.blurb}</p>
          <span class="mode-action">Open {ws.label.toLowerCase()} <b aria-hidden="true">→</b></span>
        </a>
      {/each}
    </div>
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
  .hero{border-bottom:1px solid var(--rule); background:var(--bg-2)}
  .hero-grid{
    padding-top:56px; padding-bottom:56px;
    display:grid; grid-template-columns:minmax(0,.74fr) minmax(600px,1.26fr);
    grid-template-areas:"copy preview" "device preview";
    grid-template-rows:auto 1fr;
    column-gap:clamp(40px,5vw,72px); row-gap:28px; align-items:start;
  }
  .hero-copy{grid-area:copy; max-width:600px; padding-top:18px}
  .preview-wrap{grid-area:preview; min-width:0; align-self:center}
  .eyebrow{
    display:flex; align-items:center; gap:9px; margin:0 0 16px;
    font-family:var(--mono); font-size:11px; line-height:1.4;
    text-transform:uppercase; letter-spacing:.09em; color:var(--accent-deep); font-weight:650;
  }
  .eyebrow span{width:24px; height:2px; background:var(--accent); flex:0 0 auto}
  .hero h1{
    max-width:13ch; margin:0 0 22px; font-size:clamp(42px,5.2vw,64px);
    line-height:.99; letter-spacing:-.035em; font-weight:600;
  }
  .lede{max-width:54ch; margin:0; font-size:17px; line-height:1.65; color:var(--ink-soft)}
  .cta-row{display:flex; flex-wrap:wrap; gap:10px; margin-top:28px}
  .btn{
    min-height:46px; padding:11px 19px; display:inline-flex; align-items:center; gap:11px;
    border:1px solid var(--border-input); border-radius:7px;
    font-size:14px; font-weight:650; text-decoration:none;
    transition:background .15s,border-color .15s,color .15s,transform .15s;
  }
  .btn:hover{text-decoration:none; transform:translateY(-1px)}
  .btn-primary{background:var(--accent); border-color:var(--accent); color:var(--on-accent)}
  .btn-primary:hover{background:var(--accent-hover); color:var(--on-accent)}
  .btn-secondary{background:var(--panel); color:var(--ink)}
  .btn-secondary:hover{border-color:var(--accent)}
  .cta-tertiary{display:inline-flex; align-items:center; gap:5px; min-height:32px; margin-top:12px; color:var(--ink-soft); font-size:14px}
  .cta-tertiary b{color:var(--ink); font-weight:650}
  .cta-tertiary span{color:var(--accent)}
  .cta-tertiary:hover b{text-decoration:underline; text-underline-offset:3px}

  /* Device check: live WebGPU / SIMD / threads result, under the hero actions. */
  .device{
    grid-area:device; align-self:start; max-width:460px; padding:14px 16px;
    border:1px solid var(--rule-strong); border-left:3px solid var(--muted-2); border-radius:8px;
    background:var(--panel);
  }
  .device[data-tone=ok]{border-left-color:var(--accent)}
  .device[data-tone=warn]{border-left-color:var(--warn)}
  .device p{margin:0}
  .device-label{font-family:var(--mono); font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:var(--muted)}
  .device .device-headline{margin-top:6px}
  .device .device-detail{margin-top:3px}
  .device-headline{display:flex; align-items:center; gap:8px;  color:var(--ink); font-size:15px; font-weight:650}
  .device-headline i{width:8px; height:8px; flex:0 0 8px; border-radius:50%; background:var(--muted-2)}
  .device[data-tone=ok] .device-headline i{background:var(--accent)}
  .device[data-tone=warn] .device-headline i{background:var(--warn)}
  .device-detail{ color:var(--muted); font-size:13px; line-height:1.5}
  .device-chips{display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 0; padding:0; list-style:none}
  .device-chips li{padding:2px 8px; border:1px solid var(--rule-strong); border-radius:999px; color:var(--ink-soft); font-family:var(--mono); font-size:12px}
  .device-chips li[data-tone=ok]{color:var(--accent-soft-text); border-color:color-mix(in srgb, var(--accent) 50%, var(--rule))}
  .device-chips li[data-tone=warn]{color:var(--warn); border-color:color-mix(in srgb, var(--warn) 50%, var(--rule))}
  .device-link{display:inline-block; margin-top:10px; color:var(--ink); font-size:13px; font-weight:650}
  .device-link span{color:var(--accent); margin-left:3px}

  .section-intro{max-width:720px; margin-bottom:32px}
  .section-intro .eyebrow{margin:0 0 10px}
  .section-intro h2{margin:0; font-size:clamp(30px,4vw,42px); line-height:1.08; letter-spacing:-.025em}
  .section-intro p:not(.eyebrow){margin:12px 0 0; color:var(--muted); font-size:15px; line-height:1.6}

  /* Engine roster */
  .engines{border-bottom:1px solid var(--rule)}
  .engines .wrap{padding-top:68px; padding-bottom:72px}
  .roster{display:grid; grid-template-columns:minmax(0,2fr) minmax(0,1fr); gap:28px 40px; align-items:start}
  .roster-head{display:flex; align-items:baseline; justify-content:space-between; gap:16px; padding-bottom:10px; border-bottom:1px solid var(--rule-strong)}
  .roster-head h3{margin:0; font-size:20px}
  .roster-head a{color:var(--ink); font-size:13px; font-weight:650; white-space:nowrap}
  .roster-head a span{color:var(--accent); margin-left:3px}
  .roster-list{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); column-gap:28px; margin:0; padding:0; list-style:none}
  .roster-list-play{grid-template-columns:minmax(0,1fr)}
  .roster-list li{
    display:grid; grid-template-columns:28px minmax(0,1fr) auto; align-items:center;
    column-gap:10px; row-gap:3px; align-content:start; padding:14px 0; border-bottom:1px solid var(--rule);
  }
  .roster-avatar{
    width:28px; height:28px; display:grid; place-items:center; border-radius:6px;
    background:var(--engine-color); color:#fff; font:700 11.5px/1 var(--mono); box-shadow:inset 0 0 0 1px #ffffff26;
  }
  .roster-name{color:var(--ink); font-size:15px; font-weight:650}
  .roster-name:hover{text-decoration:underline; text-underline-offset:3px}
  .roster-runtime{padding:1px 7px; border:1px solid var(--rule-strong); border-radius:999px; color:var(--muted); font-family:var(--mono); font-size:11px; white-space:nowrap}
  .roster-runtime[data-tone=ok]{color:var(--accent-soft-text); border-color:color-mix(in srgb, var(--accent) 50%, var(--rule))}
  .roster-runtime[data-tone=warn]{color:var(--warn); border-color:color-mix(in srgb, var(--warn) 50%, var(--rule))}
  .roster-list p{grid-column:2 / -1; margin:0; color:var(--muted); font-size:13px; line-height:1.5}
  .roster-note{margin:14px 0 0; color:var(--muted); font-size:13px; line-height:1.5}

  .modes .wrap{padding-top:68px; padding-bottom:82px}
  .mode-index{border-bottom:1px solid var(--rule-strong)}
  .mode{
    display:grid; grid-template-columns:120px minmax(250px,.8fr) minmax(300px,1fr) 112px;
    gap:24px; align-items:start; padding:25px 4px;
    border-top:1px solid var(--rule-strong); color:inherit; text-decoration:none;
    transition:background .15s,padding .15s;
  }
  .mode:hover{padding-left:12px; padding-right:12px; background:var(--panel); text-decoration:none}
  .mode-number{padding-top:4px; font-family:var(--mono); font-size:15px; letter-spacing:.02em; color:var(--accent-deep); font-weight:700}
  .mode h3{margin:0; font-size:23px; line-height:1.18}
  .mode p{margin:0; max-width:58ch; font-size:14px; line-height:1.65; color:var(--muted)}
  .mode-action{justify-self:end; padding-top:4px; color:var(--ink); font-size:13px; font-weight:650; white-space:nowrap}
  .mode-action b{margin-left:5px; color:var(--accent); font-weight:700}

  .principle{border-top:1px solid var(--rule-strong); background:var(--bg-workspace)}
  .principle-grid{
    padding-top:44px; padding-bottom:44px; display:grid;
    grid-template-columns:minmax(250px,.8fr) minmax(280px,1fr) auto;
    gap:34px; align-items:center;
  }
  .principle .eyebrow{margin:0 0 7px}
  .principle h2{margin:0; font-size:28px; line-height:1.1}
  .principle-grid>p{margin:0; color:var(--muted); line-height:1.65; font-size:14px}
  .principle a{white-space:nowrap; color:var(--ink); font-weight:650; font-size:13px}
  .principle a span{color:var(--accent); margin-left:5px}
  footer.site-footer{margin-top:0}
  :global(.landing-footer-wrap){justify-content:flex-end}

  @media(max-width:1180px){
    .hero-grid{grid-template-columns:minmax(0,.85fr) minmax(500px,1.15fr); column-gap:36px}
    .mode{grid-template-columns:95px minmax(220px,.8fr) 1fr 100px; gap:18px}
  }
  @media(max-width:980px){
    .hero-grid{grid-template-columns:1fr; grid-template-areas:"copy" "preview" "device"; grid-template-rows:none; padding-top:48px; padding-bottom:52px}
    .hero-copy{max-width:650px; padding-top:0}
    .hero h1{max-width:14ch}
    .preview-wrap{width:100%; max-width:680px}
    .roster{grid-template-columns:1fr}
  }
  @media(max-width:820px){
    .mode{grid-template-columns:90px 1fr 1.15fr; gap:18px}
    .mode-action{grid-column:2; justify-self:start; padding-top:0}
    .principle-grid{grid-template-columns:1fr 1.2fr}
    .principle-grid>a{grid-column:1}
  }
  @media(max-width:600px){
    .hero-grid{padding:26px 16px 36px; row-gap:22px}
    .hero h1{font-size:clamp(36px,11vw,48px); margin-bottom:14px}
    .lede{font-size:15.5px; line-height:1.55}
    .cta-row{margin-top:20px; gap:8px}
    .cta-row .btn{flex:1 1 0; justify-content:center; padding-inline:12px}
    .cta-tertiary{margin-top:6px}
    .device{max-width:none}
    .section-intro{margin-bottom:22px}
    .engines .wrap{padding:48px 16px 52px}
    .roster-list{grid-template-columns:1fr}
    .roster-list li{padding:11px 0}
    .modes .wrap{padding:48px 16px 56px}
    .mode{grid-template-columns:1fr; gap:8px; padding:24px 0}
    .mode:hover{padding-left:8px; padding-right:8px}
    .mode-number{padding:0}
    .mode h3{font-size:22px}
    .mode-action{grid-column:auto; margin-top:5px}
    .principle-grid{padding:38px 16px; grid-template-columns:1fr; gap:12px}
    .principle-grid>a{grid-column:auto; margin-top:5px}
  }
</style>
