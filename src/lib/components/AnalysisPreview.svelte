<script lang="ts">
  type PieceRole = 'rook' | 'knight' | 'bishop' | 'queen' | 'king' | 'pawn';
  type Piece = { color: 'white' | 'black'; role: PieceRole; file: number; rank: number };

  const backRank: PieceRole[] = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook'];
  const pieces: Piece[] = [
    ...backRank.map((role, file) => ({ color: 'black' as const, role, file, rank: 0 })),
    ...backRank.map((role, file) => ({ color: 'white' as const, role, file, rank: 7 })),
    ...Array.from({ length: 8 }, (_, file) => ({ color: 'black' as const, role: 'pawn' as const, file, rank: 1 })),
    ...Array.from({ length: 8 }, (_, file) => ({ color: 'white' as const, role: 'pawn' as const, file, rank: 6 })),
  ];
</script>

<div class="analysis-preview" role="img" aria-label="Preview of the 0x88 multi-engine Analysis workspace">
  <div class="preview-grid" aria-hidden="true">
    <section class="board-panel">
      <div class="board-row">
        <div class="evalbar"><span></span></div>
        <div class="preview-board cg-wrap">
          <cg-board>
            {#each pieces as piece}
              <piece class={`${piece.color} ${piece.role}`} style={`transform:translate(${piece.file * 100}%,${piece.rank * 100}%)`}></piece>
            {/each}
            <svg class="arrows" viewBox="0 0 800 800" preserveAspectRatio="none">
              <defs>
                <marker id="preview-arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
                  <path d="M0,0 L5,2.5 L0,5 Z"></path>
                </marker>
              </defs>
              <line x1="350" y1="650" x2="350" y2="450"></line>
              <line class="secondary" x1="450" y1="650" x2="450" y2="450"></line>
              <line class="secondary" x1="650" y1="750" x2="550" y2="550"></line>
            </svg>
          </cg-board>
        </div>
      </div>
      <div class="board-nav">
        <span>|◀</span><span>◀</span><span>▶</span><span>▶|</span><span>⇅</span>
      </div>
      <div class="move-list">No moves yet. Drag a piece or load a PGN.</div>
      <div class="position-tools-preview"><span>+</span> Position &amp; PGN</div>
    </section>

    <section class="analysis-panel">
      <div class="mini-overview">
        <div><span>Current position</span><strong>White to move</strong></div>
        <div><span>Evaluation</span><strong>+0.13</strong></div>
      </div>
      <div class="mini-status">Analyzed · Lc0 small · 400 visits</div>
      <div class="section-title"><span>▾</span> ENGINES</div>
      <div class="analyze-row"><span class="primary">Analyze</span><span class="disabled">Stop</span></div>
      <div class="options-row">
        <span class="toggle"><i></i>Auto</span>
        <span class="stepper"><b>−</b><em>LINES<strong>3</strong></em><b>+</b></span>
      </div>
      <div class="engine-row-preview">
        <img src="/engine-logos/lc0.svg" alt="" />
        <span>Lc0</span><b>→</b><span>Small</span><b>→</b><code>400</code><small>visits</small>
      </div>
      <div class="add-engine">+ Add engine</div>

      <div class="section-title comparison"><span>▾</span> COMPARISON</div>
      <div class="consensus">1/1 engines prefer d4 · eval spread unavailable</div>
      <div class="compare-head"><span>ENGINE</span><span>BEST</span><span>EVAL</span><span>Δ</span><span>PV</span></div>
      <div class="compare-line"><span><img src="/engine-logos/lc0.svg" alt="" />Lc0</span><strong>d4</strong><code>+0.13</code><code>0</code><code>d4 d5 c4 c6</code></div>

      <div class="section-title lines-title"><span>▾</span> LINES</div>
      <div class="legend"><i></i>Lc0</div>
      <div class="analysis-line"><strong>+0.13</strong><code>d4 d5 c4 c6 cxd5</code></div>
      <div class="analysis-line"><strong>+0.12</strong><code>Nf3 d5 d4 Nf6</code></div>
      <div class="analysis-line"><strong>+0.11</strong><code>e4 e5 Nf3 Nc6</code></div>
    </section>
  </div>
</div>

<style>
  .analysis-preview{
    container-type:inline-size;
    min-width:0; overflow:hidden;
    border:1px solid var(--rule-strong); border-radius:10px;
    background:var(--bg-workspace);
    box-shadow:0 30px 70px -45px rgba(45,35,20,.6);
  }
  .preview-grid{
    display:grid; grid-template-columns:minmax(240px,1.06fr) minmax(265px,.94fr);
    gap:10px; padding:10px;
  }
  .board-panel,.analysis-panel{
    min-width:0; border:1px solid var(--rule); border-radius:8px;
    background:var(--panel); overflow:hidden;
  }
  .board-panel{padding:9px}
  .board-row{display:grid; grid-template-columns:8px minmax(0,1fr); gap:7px}
  .evalbar{position:relative; overflow:hidden; border-radius:3px; background:var(--eval-black)}
  .evalbar span{position:absolute; inset:auto 0 0; height:51.3%; background:var(--eval-white); border-top:1px solid var(--accent)}
  .preview-board.cg-wrap{
    width:100%!important; height:auto!important; aspect-ratio:1; box-sizing:border-box;
    overflow:hidden; border:1px solid var(--border-input); border-radius:4px;
    background:var(--board-light);
  }
  .preview-board cg-board{position:relative; display:block; overflow:hidden}
  .arrows{position:absolute; inset:0; width:100%; height:100%; z-index:4; pointer-events:none}
  .arrows line{stroke:#4a8b32; stroke-width:15; opacity:.82; marker-end:url(#preview-arrow)}
  .arrows line.secondary{stroke:#8aa14e; opacity:.48}
  .arrows path{fill:#4a8b32}
  .board-nav{display:grid; grid-template-columns:repeat(5,1fr); gap:4px; margin-top:7px}
  .board-nav span{
    height:25px; display:grid; place-items:center;
    border:1px solid var(--border-input); border-radius:4px;
    color:var(--text-soft); font-size:9px;
  }
  .move-list{
    margin-top:6px; padding:7px 8px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
    border:1px solid var(--rule); border-radius:4px; background:var(--panel-inset);
    color:var(--muted); font-family:var(--mono); font-size:8px;
  }
  .position-tools-preview{
    margin-top:6px; padding:6px 8px; border:1px solid var(--rule); border-radius:4px;
    color:var(--muted-2); font-family:var(--mono); font-size:7px; letter-spacing:.06em; text-transform:uppercase;
  }
  .position-tools-preview span{color:var(--accent); margin-right:4px}
  .analysis-panel{padding:9px 10px}
  .mini-overview{display:grid; grid-template-columns:1fr .8fr; gap:7px; padding-bottom:7px; border-bottom:1px solid var(--rule)}
  .mini-overview div{min-width:0; display:grid; gap:1px}
  .mini-overview div+div{padding-left:7px; border-left:1px solid var(--rule); text-align:right}
  .mini-overview span{font-family:var(--mono); font-size:5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted)}
  .mini-overview strong{overflow:hidden; white-space:nowrap; text-overflow:ellipsis; font-family:var(--serif); font-size:10px; color:var(--ink)}
  .mini-status{
    margin:7px 0 9px; padding:5px 6px; border-left:2px solid var(--accent); background:var(--panel-inset);
    color:var(--muted); font-family:var(--mono); font-size:6px;
  }
  .section-title{
    color:var(--muted-2); font-family:var(--mono); font-size:8px;
    line-height:1; letter-spacing:.11em; font-weight:650;
  }
  .section-title>span{color:var(--accent); margin-right:4px}
  .analyze-row{display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:9px}
  .analyze-row span,.add-engine{
    min-height:25px; display:grid; place-items:center;
    border:1px solid var(--border-input); border-radius:5px;
    font-size:9px; color:var(--text-soft);
  }
  .analyze-row .primary{background:var(--accent); border-color:var(--accent); color:var(--on-accent); font-weight:650}
  .analyze-row .disabled{opacity:.55}
  .options-row{height:29px; display:flex; align-items:center; gap:10px; margin-top:6px; font-size:9px; color:var(--text-soft)}
  .toggle{display:flex; align-items:center; gap:5px}
  .toggle i{width:22px; height:12px; padding:1px; display:block; border-radius:99px; background:var(--accent)}
  .toggle i::after{content:""; width:8px; height:8px; display:block; margin-left:10px; border-radius:50%; background:#fff}
  .stepper{height:25px; display:flex; align-items:stretch; overflow:hidden; border:1px solid var(--border-input); border-radius:5px}
  .stepper b{width:21px; display:grid; place-items:center; background:var(--panel-inset); font-size:11px}
  .stepper em{width:31px; display:grid; place-items:center; border-inline:1px solid var(--border-input); font-style:normal; font-size:5px; letter-spacing:.06em; color:var(--muted)}
  .stepper em strong{display:block; font-size:9px; line-height:.8; color:var(--ink)}
  .engine-row-preview{
    display:grid; grid-template-columns:13px minmax(42px,1fr) 8px minmax(45px,1fr) 8px 38px auto;
    gap:4px; align-items:center; margin-top:6px; font-size:8px; color:var(--ink);
  }
  .engine-row-preview img,.compare-line img{width:11px; height:11px; object-fit:contain}
  .engine-row-preview>span,.engine-row-preview>code{
    min-width:0; padding:5px 6px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
    border:1px solid var(--border-input); border-radius:4px; background:var(--card);
  }
  .engine-row-preview>b{color:var(--muted); text-align:center}
  .engine-row-preview code{font-size:8px; font-style:italic}
  .engine-row-preview small{color:var(--muted); font-size:6px}
  .add-engine{margin-top:7px; background:var(--card)}
  .comparison{margin-top:13px}
  .consensus{
    margin-top:8px; padding:6px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
    border:1px solid var(--rule); border-radius:4px; background:var(--panel-inset);
    color:var(--muted); font-size:7px;
  }
  .compare-head,.compare-line{display:grid; grid-template-columns:1.15fr .7fr .7fr .35fr 1.4fr; gap:4px; align-items:center}
  .compare-head{padding:8px 3px 4px; color:var(--muted); font-size:5px; letter-spacing:.05em}
  .compare-line{padding:5px 3px; border-block:1px solid var(--rule); font-size:7px}
  .compare-line>span{display:flex; align-items:center; gap:3px; color:var(--ink)}
  .compare-line code{overflow:hidden; white-space:nowrap; text-overflow:ellipsis; font-size:6px}
  .lines-title{margin-top:13px}
  .legend{display:flex; align-items:center; gap:5px; margin-top:7px; color:var(--muted); font-size:7px}
  .legend i{width:7px; height:7px; border-radius:2px; background:var(--chart-1)}
  .analysis-line{
    display:grid; grid-template-columns:38px minmax(0,1fr); gap:5px;
    padding:5px 3px 5px 7px; border-top:1px solid var(--rule); border-left:2px solid var(--chart-1);
    color:var(--ink); font-size:7px;
  }
  .analysis-line strong{color:var(--accent-deep); font-family:var(--mono)}
  .analysis-line code{overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:var(--text-soft); font-size:7px}

  @container(max-width:520px){
    .preview-grid{grid-template-columns:minmax(96px,.72fr) minmax(0,1.28fr); gap:6px; padding:6px}
    .board-panel{padding:6px}
    .board-row{grid-template-columns:5px minmax(0,1fr); gap:4px}
    .board-nav{gap:2px; margin-top:4px}
    .board-nav span{height:18px; font-size:7px}
    .move-list{margin-top:4px; padding:5px; font-size:6px}
    .position-tools-preview{margin-top:4px; padding:4px; font-size:5px}
    .analysis-panel{padding:7px}
    .mini-overview{padding-bottom:5px}
    .mini-overview strong{font-size:7px}
    .mini-status{margin:5px 0 6px; padding:4px; font-size:5px}
    .section-title{font-size:6px}
    .analyze-row{gap:4px; margin-top:6px}
    .analyze-row span,.add-engine{min-height:20px; font-size:7px}
    .options-row{height:24px; gap:7px; margin-top:3px; font-size:7px}
    .engine-row-preview{grid-template-columns:10px minmax(28px,1fr) 6px minmax(31px,1fr) 6px 30px; gap:2px; margin-top:3px; font-size:6px}
    .engine-row-preview small{display:none}
    .engine-row-preview>span,.engine-row-preview>code{padding:4px}
    .engine-row-preview img{width:9px; height:9px}
    .add-engine{margin-top:4px}
    .comparison,.lines-title{margin-top:8px}
    .consensus{margin-top:5px; padding:4px; font-size:5px}
    .compare-head{padding-top:5px}
    .compare-line{padding:3px 2px}
    .compare-line img{display:none}
    .analysis-line{grid-template-columns:30px minmax(0,1fr); padding:3px 2px 3px 5px; font-size:6px}
    .analysis-line code{font-size:6px}
    .legend{margin-top:5px}
  }
</style>
