<script lang="ts">
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  const title = "0x88 — a chess engine workbench in your browser";
  const description = "Play Maia3 at your rating, compare Lc0 with Stockfish and other engines, or run live tournaments. Everything runs locally with WebGPU and WebAssembly.";
  const github = "https://github.com/manthedan/0x88";
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
      <p class="eyebrow"><span></span> Local chess compute · WebGPU + WASM</p>
      <h1 id="hero-h1">A chess engine workbench, inside your browser.</h1>
      <p class="lede">
        Play a human-like opponent, compare serious engines on the same position,
        or run a tournament. No account, no installation, and no game leaves your device.
      </p>
      <div class="cta-row">
        <a class="btn btn-primary" href="/app/play/">Play a game <span aria-hidden="true">→</span></a>
        <a class="btn btn-secondary" href="/app/analysis/">Open analysis</a>
      </div>
      <div class="trust-row" aria-label="Product characteristics">
        <span>Maia3</span><span>Lc0</span><span>Stockfish</span><span>+ 6 engines</span>
      </div>
    </div>

    <div class="workbench" aria-label="Analysis workbench preview">
      <div class="workbench-head">
        <div><span class="signal"></span><strong>POSITION LAB</strong><span>startpos · white</span></div>
        <span class="local-state">LOCAL</span>
      </div>
      <div class="workbench-body">
        <div class="preview-board cg-wrap" aria-label="Chess position">
          <cg-board>
            <square class="last-move" style="transform:translate(400%,400%)"></square>
            <square class="last-move" style="transform:translate(400%,600%)"></square>
            <piece class="black king" style="transform:translate(600%,0%)"></piece>
            <piece class="black queen" style="transform:translate(300%,0%)"></piece>
            <piece class="black rook" style="transform:translate(500%,0%)"></piece>
            <piece class="black bishop" style="transform:translate(200%,0%)"></piece>
            <piece class="black knight" style="transform:translate(500%,200%)"></piece>
            <piece class="black pawn" style="transform:translate(200%,100%)"></piece>
            <piece class="black pawn" style="transform:translate(300%,300%)"></piece>
            <piece class="black pawn" style="transform:translate(400%,200%)"></piece>
            <piece class="black pawn" style="transform:translate(600%,100%)"></piece>
            <piece class="black pawn" style="transform:translate(700%,100%)"></piece>
            <piece class="white king" style="transform:translate(600%,700%)"></piece>
            <piece class="white queen" style="transform:translate(300%,700%)"></piece>
            <piece class="white rook" style="transform:translate(500%,700%)"></piece>
            <piece class="white bishop" style="transform:translate(200%,400%)"></piece>
            <piece class="white knight" style="transform:translate(500%,500%)"></piece>
            <piece class="white pawn" style="transform:translate(100%,600%)"></piece>
            <piece class="white pawn" style="transform:translate(300%,500%)"></piece>
            <piece class="white pawn" style="transform:translate(400%,400%)"></piece>
            <piece class="white pawn" style="transform:translate(600%,600%)"></piece>
            <piece class="white pawn" style="transform:translate(700%,600%)"></piece>
          </cg-board>
        </div>
        <div class="engine-output">
          <div class="position-eval"><span>WHITE</span><strong>+0.31</strong><small>3 engines agree</small></div>
          <div class="engine-line active">
            <span class="engine-key">LC0</span><strong>e4</strong><span class="line-score">+0.31</span>
            <code>e4 e6 d4 d5 Nc3</code>
          </div>
          <div class="engine-line">
            <span class="engine-key">SF</span><strong>Nf3</strong><span class="line-score">+0.28</span>
            <code>Nf3 d5 d4 Nf6 c4</code>
          </div>
          <div class="engine-line human">
            <span class="engine-key">M3</span><strong>e4</strong><span class="line-score">31%</span>
            <code>most played at 1500</code>
          </div>
        </div>
      </div>
      <div class="workbench-foot">
        <span><i></i> WebGPU ready</span><span>400 visits</span><span>0.8s</span>
      </div>
    </div>
  </div>
</section>

<section class="modes" aria-labelledby="modes-h">
  <div class="wrap">
    <div class="section-intro">
      <p class="eyebrow"><span></span> Choose a workspace</p>
      <h2 id="modes-h">Three ways to put the engines to work.</h2>
    </div>
    <div class="mode-index">
      <a class="mode" href="/app/play/">
        <span class="mode-number">01 · PLAY</span>
        <h3>Play a human, not a depth setting.</h3>
        <p>Set Maia3 to a rating from 600–2600 and face the moves people at that level actually choose. Or select one of nine classical and neural engines.</p>
        <span class="mode-action">Open play <b aria-hidden="true">→</b></span>
      </a>
      <a class="mode" href="/app/analysis/">
        <span class="mode-number">02 · ANALYSIS</span>
        <h3>Find the disagreement.</h3>
        <p>Put Lc0, Stockfish, and specialist engines on one position. Compare evaluations and principal variations without sending your PGN away.</p>
        <span class="mode-action">Open analysis <b aria-hidden="true">→</b></span>
      </a>
      <a class="mode" href="/app/arena/">
        <span class="mode-number">03 · ARENA</span>
        <h3>Make the engines prove it.</h3>
        <p>Run head-to-head matches, gauntlets, or round robins with equal time controls, live boards, standings, and per-game output.</p>
        <span class="mode-action">Open arena <b aria-hidden="true">→</b></span>
      </a>
    </div>
  </div>
</section>

<section class="principle" aria-label="Local-first principle">
  <div class="wrap principle-grid">
    <p class="principle-mark" aria-hidden="true">0x88</p>
    <div>
      <p class="eyebrow"><span></span> The local-first difference</p>
      <h2>Your position stays yours.</h2>
    </div>
    <p>Engine binaries and neural networks load on demand and cache in your browser. Analysis, games, and tournaments run on your hardware—not a remote chess service.</p>
    <a href="/docs/">Read the technical notes <span aria-hidden="true">→</span></a>
  </div>
</section>

</main>
<footer class="site-footer">
  <div class="wrap">
    <span>0x88 is an open-source browser chess workbench.</span>
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
  .hero{border-bottom:1px solid var(--rule); background:var(--bg-2)}
  .hero-grid{
    min-height:600px; padding-top:72px; padding-bottom:72px;
    display:grid; grid-template-columns:minmax(0,.88fr) minmax(520px,1.12fr);
    gap:clamp(44px,6vw,86px); align-items:center;
  }
  .hero-copy{max-width:600px}
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
  .trust-row{
    display:flex; flex-wrap:wrap; gap:7px 18px; margin-top:28px; padding-top:18px;
    border-top:1px solid var(--rule); color:var(--muted); font-family:var(--mono); font-size:11px;
  }
  .trust-row span:not(:last-child)::after{content:"/"; margin-left:18px; color:var(--faint)}

  .workbench{
    min-width:0; overflow:hidden; border:1px solid var(--rule-strong); border-radius:10px;
    background:var(--panel); box-shadow:0 30px 70px -45px rgba(45,35,20,.6);
  }
  .workbench-head,.workbench-foot{
    min-height:42px; padding:9px 12px; display:flex; align-items:center; justify-content:space-between;
    gap:12px; font-family:var(--mono); font-size:9px; letter-spacing:.06em; color:var(--muted);
  }
  .workbench-head{border-bottom:1px solid var(--rule)}
  .workbench-head>div{display:flex; align-items:center; gap:8px; min-width:0}
  .workbench-head strong{font-size:10px; color:var(--ink); letter-spacing:.1em}
  .signal{width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 0 3px var(--accent-soft)}
  .local-state{padding:2px 6px; border:1px solid var(--accent-tint-border); color:var(--accent-deep); border-radius:3px; font-weight:700}
  .workbench-body{display:grid; grid-template-columns:minmax(230px,.9fr) minmax(230px,1fr); gap:0}
  .preview-board.cg-wrap{
    width:100%!important; height:auto!important; aspect-ratio:1; box-sizing:border-box;
    border-right:1px solid var(--rule); background:var(--board-light);
  }
  .preview-board cg-board{position:relative; display:block; overflow:hidden}
  .engine-output{min-width:0; padding:16px 15px; display:grid; align-content:start; gap:0}
  .position-eval{
    display:grid; grid-template-columns:1fr auto; align-items:end; gap:2px 10px;
    padding:0 4px 14px; border-bottom:1px solid var(--rule);
  }
  .position-eval>span{font-family:var(--mono); font-size:9px; letter-spacing:.08em; color:var(--muted)}
  .position-eval strong{grid-row:1/3; grid-column:2; font-family:var(--mono); font-size:25px; line-height:1; color:var(--accent-deep)}
  .position-eval small{font-size:11px; color:var(--muted)}
  .engine-line{
    display:grid; grid-template-columns:36px 38px 46px minmax(0,1fr); gap:5px; align-items:center;
    min-height:48px; padding:8px 4px; border-bottom:1px solid var(--rule-soft); font-size:11px;
  }
  .engine-line.active{border-left:2px solid var(--accent); padding-left:7px}
  .engine-key{font-family:var(--mono); font-size:9px; color:var(--muted); font-weight:700}
  .engine-line strong{font-family:var(--serif); color:var(--ink); font-size:14px}
  .line-score{font-family:var(--mono); color:var(--accent-deep); font-weight:700}
  .engine-line code{overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--muted); font-family:var(--mono); font-size:9px}
  .engine-line.human .line-score{color:var(--warn)}
  .workbench-foot{border-top:1px solid var(--rule); justify-content:flex-start; gap:18px}
  .workbench-foot span{display:flex; align-items:center; gap:5px}
  .workbench-foot span:first-child{margin-right:auto; color:var(--accent-deep)}
  .workbench-foot i{width:6px; height:6px; border-radius:50%; background:var(--accent)}

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
  .mode-number{padding-top:5px; font-family:var(--mono); font-size:10px; letter-spacing:.08em; color:var(--accent-deep); font-weight:700}
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

  @media(max-width:1040px){
    .hero-grid{grid-template-columns:minmax(0,.85fr) minmax(460px,1.15fr); gap:38px}
    .workbench-body{grid-template-columns:minmax(205px,.9fr) minmax(210px,1fr)}
    .mode{grid-template-columns:95px minmax(220px,.8fr) 1fr 100px; gap:18px}
  }
  @media(max-width:820px){
    .hero-grid{min-height:0; grid-template-columns:1fr; padding-top:54px; padding-bottom:58px}
    .hero-copy{max-width:650px}
    .hero h1{max-width:14ch}
    .workbench{max-width:650px}
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
    .trust-row{gap:6px 12px}
    .trust-row span:not(:last-child)::after{margin-left:12px}
    .workbench-body{grid-template-columns:138px minmax(0,1fr)}
    .engine-output{padding:10px 9px}
    .position-eval{padding-bottom:9px}
    .position-eval strong{font-size:20px}
    .engine-line{grid-template-columns:30px 31px 38px minmax(0,1fr); min-height:39px; padding:5px 3px}
    .engine-line code{font-size:8px}
    .workbench-head>div>span:last-child{display:none}
    .workbench-foot{gap:12px}
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
  @media(max-width:360px){
    .workbench-body{grid-template-columns:116px minmax(0,1fr)}
    .engine-line{grid-template-columns:27px 29px 34px minmax(0,1fr)}
    .engine-line code{display:none}
    .trust-row span:not(:last-child)::after{display:none}
  }
</style>
