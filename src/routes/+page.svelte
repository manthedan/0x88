<script lang="ts">
import AnalysisPreview from '$lib/components/AnalysisPreview.svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';
import { footerNav, workspaces } from '$lib/nav';

const title = '0x88 | Browser chess engine workbench';
const description = 'Play against Maia3, compare browser chess engines on one position, or run engine tournaments. Computation stays on your device.';
const github = 'https://github.com/manthedan/0x88';
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
    </div>

    <div class="preview-wrap"><AnalysisPreview /></div>
  </div>
</section>

<section class="modes" aria-labelledby="modes-h">
  <div class="wrap">
    <div class="section-intro">
      <p class="eyebrow"><span></span> Workspaces</p>
      <h2 id="modes-h">Chess workspaces</h2>
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
    <p class="principle-mark" aria-hidden="true">0x88</p>
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
    min-height:540px; padding-top:60px; padding-bottom:60px;
    display:grid; grid-template-columns:minmax(0,.88fr) minmax(520px,1.12fr);
    gap:clamp(44px,6vw,86px); align-items:center;
  }
  .hero-copy{max-width:600px}
  .preview-wrap{min-width:0}
  .eyebrow{
    display:flex; align-items:center; gap:9px; margin:0 0 16px;
    font-family:var(--mono); font-size:11px; line-height:1.4;
    text-transform:uppercase; letter-spacing:.09em; color:var(--accent-deep); font-weight:650;
  }
  .eyebrow span{width:24px; height:2px; background:var(--accent); flex:0 0 auto}
  .hero h1{
    max-width:13ch; margin:0 0 22px; font-size:clamp(42px,5.5vw,66px);
    line-height:.99; letter-spacing:-.035em; font-weight:600;
  }
  .lede{max-width:54ch; margin:0; font-size:17px; line-height:1.65; color:var(--ink-soft)}
  .cta-row{display:flex; flex-wrap:wrap; gap:10px; margin-top:30px}
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
  .modes .wrap{padding-top:72px; padding-bottom:82px}
  .section-intro{display:grid; grid-template-columns:minmax(220px,.65fr) 1.35fr; gap:32px; align-items:end; margin-bottom:36px}
  .section-intro .eyebrow{margin:0 0 5px}
  .section-intro h2{max-width:18ch; margin:0; font-size:clamp(30px,4vw,44px); line-height:1.06; letter-spacing:-.025em}
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

  .principle{border-top:1px solid var(--rule); background:var(--bg-workspace)}
  .principle-grid{
    padding-top:44px; padding-bottom:44px; display:grid;
    grid-template-columns:90px minmax(250px,.75fr) minmax(280px,1fr) auto;
    gap:34px; align-items:center;
  }
  .principle-mark{margin:0; font-family:var(--mono); font-weight:800; font-size:24px; color:var(--accent); letter-spacing:-.06em}
  .principle .eyebrow{margin:0 0 7px}
  .principle h2{margin:0; font-size:28px; line-height:1.1}
  .principle-grid>p:not(.principle-mark){margin:0; color:var(--muted); line-height:1.65; font-size:14px}
  .principle a{white-space:nowrap; color:var(--ink); font-weight:650; font-size:13px}
  .principle a span{color:var(--accent); margin-left:5px}
  footer.site-footer{margin-top:0}
  :global(.landing-footer-wrap){justify-content:flex-end}

  @media(max-width:1040px){
    .hero-grid{grid-template-columns:minmax(0,.85fr) minmax(460px,1.15fr); gap:38px}
    .mode{grid-template-columns:95px minmax(220px,.8fr) 1fr 100px; gap:18px}
  }
  @media(max-width:820px){
    .hero-grid{min-height:0; grid-template-columns:1fr; padding-top:54px; padding-bottom:58px}
    .hero-copy{max-width:650px}
    .hero h1{max-width:14ch}
    .preview-wrap{width:100%; max-width:650px}
    .section-intro{grid-template-columns:1fr; gap:8px}
    .mode{grid-template-columns:90px 1fr 1.15fr; gap:18px}
    .mode-action{grid-column:2; justify-self:start; padding-top:0}
    .principle-grid{grid-template-columns:70px 1fr 1.2fr}
    .principle-grid>a{grid-column:2}
  }
  @media(max-width:600px){
    .hero-grid{padding:42px 20px 48px; gap:34px}
    .hero h1{font-size:clamp(38px,12vw,50px)}
    .lede{font-size:16px}
    .modes .wrap{padding:52px 20px 60px}
    .mode{grid-template-columns:1fr; gap:8px; padding:24px 0}
    .mode:hover{padding-left:8px; padding-right:8px}
    .mode-number{padding:0}
    .mode h3{font-size:22px}
    .mode-action{grid-column:auto; margin-top:5px}
    .principle-grid{padding:38px 20px; grid-template-columns:1fr; gap:12px}
    .principle-mark{display:none}
    .principle-grid>a{grid-column:auto; margin-top:5px}
  }
</style>
