<script lang="ts">
  import { onMount } from 'svelte';
  const title = "0x88 Chess — analysis board";
  const description = "Multi-engine chess analysis in your browser: compare Leela Chess Zero and Stockfish lines side by side, review games with accuracy scores, and explore openings from your own games.";
  onMount(() => {
    let cleanup: () => void = () => undefined;
    let mounted = true;
    void import('../../../lc0/analysisBrowser').then((module) => {
      if (!mounted) return;
      cleanup = module.mountAnalysisBrowser();
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

<main id="main">
  <section class="panel" aria-label="Board">
    <div class="board-wrap">
      <div class="evalbar" title="Evaluation (white advantage)"><div class="mid"></div><div class="white" id="evalWhite" style="height:50%"></div></div>
      <div class="board-shell"><div id="ground"></div></div>
    </div>
    <div class="nav">
      <button id="navStart" type="button" title="Start (Up)">|◀</button>
      <button id="navBack" type="button" title="Back (Left)">◀</button>
      <button id="navForward" type="button" title="Forward (Right)">▶</button>
      <button id="navEnd" type="button" title="End (Down)">▶|</button>
      <button id="flip" type="button" title="Flip board">⇅</button>
    </div>
    <div class="row">
      <div class="field" style="flex:1 1 100%"><label for="fenInput">FEN</label>
        <input id="fenInput" type="text" spellcheck="false" autocomplete="off" placeholder="paste a FEN" /></div>
      <button id="loadFen" type="button">Load FEN</button>
      <button id="reset" type="button">Reset</button>
    </div>
    <h2>PGN</h2>
    <textarea id="pgnInput" spellcheck="false" placeholder="paste PGN (with variations) and Load"></textarea>
    <div class="row">
      <button id="loadPgn" type="button">Load PGN</button>
      <button id="copyPgn" type="button">Copy PGN</button>
    </div>
    <div id="message" aria-live="polite">Loading model…</div>
    <div id="downloadProgress" class="model-load-progress" hidden></div>
  </section>
  <section class="panel app-sidebar" aria-label="Analysis">
    <div class="row">
      <button id="analyze" class="primary" type="button" disabled>Analyze</button>
      <button id="stop" type="button" disabled>Stop</button>
      <label class="small"><input type="checkbox" id="autoAnalyze" checked /> auto</label>
      <div class="field"><label for="multiPvInput">Lines</label>
        <input id="multiPvInput" type="number" min="1" max="10" step="1" value="3" /></div>
    </div>
    <div id="engineList" class="engine-list"></div>
    <div class="row">
      <button id="addEngine" type="button">+ Add engine</button>
      <div class="field"><label for="engineProfileSelect">Profile</label>
        <select id="engineProfileSelect"><option value="">default</option></select></div>
    </div>
    <details class="advanced-settings">
      <summary>Profiles &amp; engine settings</summary>
      <div class="row profile-row">
        <div class="field"><label for="engineProfileName">Name</label>
          <input id="engineProfileName" type="text" spellcheck="false" autocomplete="off" placeholder="analysis profile" /></div>
        <button id="saveEngineProfile" type="button">Save profile</button>
        <button id="deleteEngineProfile" type="button">Delete</button>
        <button id="exportEngineProfiles" type="button">Export profiles</button>
        <button id="importEngineProfiles" type="button">Import profiles</button>
        <input id="importEngineProfilesFile" type="file" accept="application/json,.json" style="display:none" />
      </div>
      <div id="engineProfileSummary" class="profile-summary small">Manual engine setup.</div>
      <div class="row">
        <div class="field"><label for="lc0RuntimeSelect">LC0 backend</label>
          <select id="lc0RuntimeSelect"><option value="onnx">ORT ONNX</option><option value="hybrid-ort-heads">WGSL encoder + ORT heads</option><option value="hybrid-wgsl-heads">WGSL encoder + WGSL heads (experimental)</option></select></div>
      </div>
      <details class="runtime-details"><summary>Engine runtime details</summary>
        <dl class="status-inline">
          <div><dt>Backend</dt><dd id="backend">—</dd></div>
          <div><dt>LC0 audit</dt><dd id="runtimeAudit">—</dd></div>
        </dl>
        <div id="recklessRuntimeInfo" class="small">Reckless: detecting runtime…</div>
      </details>
    </details>
    <details class="section-block" open>
      <summary>Engine comparison</summary>
      <div id="engineConsensus" class="compare-summary">No analysis yet.</div>
      <div id="analysisSearchProgress" class="search-progress-grid" hidden></div>
      <table id="engineCompare" class="engine-compare"><thead><tr><th>Engine</th><th>Best</th><th>Eval</th><th>Δ</th><th>Search</th><th>PV</th></tr></thead><tbody></tbody></table>
    </details>
    <details class="section-block" open>
      <summary>Engine lines</summary>
      <div id="engineLegend" class="legend"></div>
      <ol id="lines" class="lines"></ol>
      <dl class="status-inline">
        <div><dt>Side</dt><dd id="sideToMove">—</dd></div>
        <div><dt>Eval</dt><dd id="posEval">—</dd></div>
      </dl>
    </details>
    <details class="section-block">
      <summary>Human moves · Maia3</summary>
      <div class="row">
        <button id="maia3Enable" type="button">Load Maia3 (~28MB, cached)</button>
        <span id="maia3PanelStatus" class="small"></span>
      </div>
      <div id="maia3Grid" class="maia3-grid" hidden></div>
      <div id="maia3Caption" class="small" hidden>What rated humans actually play here (Maia3 move predictions; score = expected points for White in a human game at that rating, not an engine eval). ✓ = matches the current best engine move.</div>
    </details>
    <details class="section-block" open>
      <summary>Moves</summary>
      <div id="movelist" class="movelist"></div>
    </details>
    <details class="section-block">
      <summary>Game review</summary>
      <div class="row">
        <button id="reviewGame" type="button">Review game</button>
        <button id="reviewStop" type="button" disabled>Stop</button>
        <button id="reviewCopyPgn" type="button" hidden>Copy annotated PGN</button>
      </div>
      <div id="reviewStatus" class="small">Load a PGN or play moves, then review the mainline with the first selected UCI engine.</div>
      <div id="reviewSummary" hidden></div>
      <div id="reviewChart" class="review-chart" hidden></div>
      <ol id="reviewCritical" class="review-critical" hidden></ol>
    </details>
    <details class="section-block">
      <summary>Opening explorer</summary>
      <div class="row">
        <div class="field"><label for="importSite">Site</label>
          <select id="importSite"><option value="lichess">Lichess</option><option value="chesscom">Chess.com</option></select></div>
        <div class="field" style="flex:1 1 120px"><label for="importUser">Username</label>
          <input id="importUser" type="text" spellcheck="false" autocomplete="off" placeholder="your username" /></div>
        <div class="field"><label for="importMax">Max</label>
          <input id="importMax" type="number" min="1" max="300" step="1" value="40" /></div>
        <div class="field"><label for="importColor">Color</label>
          <select id="importColor"><option value="">both</option><option value="white">white</option><option value="black">black</option></select></div>
        <button id="fetchGames" type="button">Fetch games</button>
      </div>
      <textarea id="importGamesInput" spellcheck="false" placeholder="fetch by username above, or paste many games (PGN) here, then Import"></textarea>
      <div class="row">
        <button id="importGames" type="button">Import games</button>
        <button id="downloadPgn" type="button">Download PGN</button>
        <span id="importInfo" class="small"></span>
      </div>
      <div class="row database-row">
        <div class="field"><label for="pgnDbSelect">Local database</label>
          <select id="pgnDbSelect"><option value="">no saved collections</option></select></div>
        <div class="field" style="flex:1 1 180px"><label for="pgnDbName">Collection name</label>
          <input id="pgnDbName" type="text" spellcheck="false" autocomplete="off" placeholder="saved PGN collection" /></div>
        <button id="savePgnDb" type="button">Save to DB</button>
        <button id="loadPgnDb" type="button">Load</button>
        <button id="renamePgnDb" type="button">Rename</button>
        <button id="duplicatePgnDb" type="button">Duplicate</button>
        <button id="deletePgnDb" type="button">Delete</button>
        <button id="exportPgnDbCollection" type="button">Export PGN</button>
        <button id="exportPgnDb" type="button">Export DB</button>
        <button id="importPgnDb" type="button">Import DB</button>
        <input id="importPgnDbFile" type="file" accept="application/json,.json" style="display:none" />
        <button id="searchPgnDbPosition" type="button">Search position</button>
      </div>
      <div id="pgnDbInfo" class="small">Local PGN database: checking IndexedDB…</div>
      <div id="pgnDbList" class="pgn-db-list small"></div>
      <div id="pgnDbSearchResults" class="pgn-db-results small"></div>
      <table id="opening" class="opening"><thead><tr><th>Move</th><th class="num">Games</th><th>White / Draw / Black</th></tr></thead><tbody></tbody></table>
    </details>
  </section>
</main>

<style>
  main{
    display:grid;
    grid-template-columns:minmax(320px,680px) minmax(360px,460px);
    gap:20px; align-items:start; justify-content:center; padding:18px;
  }
  .app-sidebar{
    position:sticky; top:72px;
    max-height:calc(100vh - 84px); overflow-y:auto;
  }
  .board-wrap{display:flex; gap:8px}
  .evalbar{
    width:18px; border:1px solid var(--rule); border-radius:4px;
    overflow:hidden; background:#3a3a3a; position:relative; flex:0 0 auto;
  }
  .evalbar .white{
    position:absolute; left:0; right:0; bottom:0;
    background:#f4f1e8; transition:height .25s;
  }
  .evalbar .mid{
    position:absolute; left:0; right:0; top:50%;
    height:1px; background:var(--accent); opacity:.6;
  }
  .board-shell{flex:1 1 auto}
  .nav{display:flex; gap:6px; margin-top:10px}
  .nav button{flex:1 1 auto}
  input[type=number]{width:80px}
  :global(.lines){list-style:none; margin:8px 0 0; padding:0}
  :global(.lines li){
    display:grid; grid-template-columns:62px 1fr; gap:8px;
    align-items:baseline; padding:6px 6px;
    border-top:1px solid var(--rule); cursor:pointer; font-size:13px;
    padding-left:9px;
  }
  :global(.lines li:hover){background:var(--soft)}
  :global(.lines .score){font-family:var(--mono); font-weight:700}
  :global(.lines .score.neg){color:#a5461b}
  :global(.lines .score.pos){color:var(--accent)}
  :global(.lines .pv){font-family:var(--mono); overflow-wrap:anywhere}
  :global(.lines .eng){display:inline-flex; align-items:center; gap:4px; font-size:10px; color:var(--muted)}
  :global(.engine-logo){width:16px; height:16px; object-fit:contain; border-radius:3px; vertical-align:middle; flex:0 0 auto}
  :global(.engine-row .engine-logo){width:18px; height:18px}
  :global(.legend .engine-logo), :global(.lines .engine-logo), :global(table.engine-compare .engine-logo){width:14px; height:14px}
  :global(.engine-name-with-logo){display:inline-flex; align-items:center; gap:5px}
  :global(.lines li.placeholder){display:block; color:var(--muted); border-left:none; cursor:default}
  :global(.lines li.placeholder:hover){background:none}
  :global(.review-chart svg){
    display:block; width:100%; height:auto;
    border:1px solid var(--rule); border-radius:6px; background:#fff; margin-top:6px;
  }
  :global(.review-summary){display:flex; gap:18px; margin:6px 0; font-size:12px}
  :global(.review-summary .acc){font-size:16px; font-weight:700; font-family:var(--mono)}
  :global(.review-badge){
    display:inline-block; border-radius:3px; padding:0 5px;
    font-size:10px; font-weight:700; color:#fff; vertical-align:1px;
  }
  :global(.review-badge.best){background:#4a7a2a}
  :global(.review-badge.good){background:#7a9a4a}
  :global(.review-badge.inaccuracy){background:#c08a00}
  :global(.review-badge.mistake){background:#d2691e}
  :global(.review-badge.blunder){background:#a5461b}
  :global(.review-badge.forced){background:#999}
  :global(.review-critical){list-style:none; margin:6px 0 0; padding:0}
  :global(.review-critical li){
    padding:4px 6px; border-top:1px solid var(--rule);
    cursor:pointer; font-size:12px;
  }
  :global(.review-critical li:hover){background:var(--soft)}
  :global(.review-critical .mono){font-family:var(--mono)}
  :global(.compare-summary){
    margin-top:8px; padding:8px; border:1px solid var(--rule);
    border-radius:6px; background:#fff; font-size:12px; color:var(--muted);
  }
  :global(table.engine-compare){width:100%; border-collapse:collapse; font-size:12px; margin-top:8px}
  :global(table.engine-compare th), :global(table.engine-compare td){
    text-align:left; padding:5px 7px;
    border-bottom:1px solid var(--rule); vertical-align:top;
  }
  :global(table.engine-compare th){
    color:var(--muted); font-weight:600; font-size:10px; text-transform:uppercase;
  }
  :global(table.engine-compare .mono){font-family:var(--mono)}
  :global(table.engine-compare .pv){font-family:var(--mono); overflow-wrap:anywhere}
  :global(table.engine-compare .agree){color:var(--accent); font-weight:700}
  :global(.model-load-progress), :global(.search-progress-grid){
    margin-top:8px; padding:8px; border:1px solid var(--rule);
    border-radius:6px; background:#fff;
  }
  :global(.model-load-progress progress), :global(.search-progress-grid progress), :global(.search-progress-cell progress){
    width:100%; height:9px; accent-color:var(--accent);
  }
  :global(.loading-progress-row), :global(.search-progress-row){display:grid; gap:3px; margin:4px 0}
  :global(.dl-label), :global(.search-progress-text){font-family:var(--mono); font-size:11px; color:var(--muted)}
  :global(.search-progress-cell){display:grid; gap:3px; min-width:120px}
  .profile-row{align-items:end}
  .profile-row select{min-width:150px}
  .profile-row input[type=text]{min-width:130px}
  :global(.profile-summary){
    margin-top:6px; padding:7px 8px;
    border:1px dashed var(--rule); border-radius:6px; background:#fff;
  }
  :global(.legend){display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; font-size:12px; color:var(--muted)}
  :global(.legend .key){display:inline-flex; align-items:center; gap:5px}
  :global(.legend .dot){width:11px; height:11px; border-radius:3px; display:inline-block}
  :global(.movelist){
    font-family:var(--mono); font-size:13px; line-height:1.7;
    margin-top:8px; max-height:240px; overflow:auto;
  }
  :global(.maia3-grid){margin-top:8px; font-size:13px}
  :global(.maia3-grid .maia3-row){
    display:flex; align-items:center; gap:8px;
    padding:3px 0; border-bottom:1px solid var(--rule);
  }
  :global(.maia3-grid .maia3-elo){font-family:var(--mono); width:48px; color:var(--muted)}
  :global(.maia3-grid .maia3-score){font-family:var(--mono); width:44px}
  :global(.maia3-grid .maia3-move){
    display:inline-flex; align-items:center; gap:4px;
    margin-right:10px; font-family:var(--mono);
  }
  :global(.maia3-grid .maia3-bar){
    display:inline-block; height:8px;
    background:#5a6e2a; border-radius:2px; opacity:0.75;
  }
  :global(.movelist .mv){cursor:pointer; padding:0 3px; border-radius:4px}
  :global(.movelist .mv:hover){background:var(--soft)}
  :global(.movelist .mv.current){background:var(--accent); color:white}
  :global(.movelist .var){color:var(--muted)}
  .section-block{margin-top:10px}
  .section-block>summary{
    cursor:pointer; font-size:13px; text-transform:uppercase;
    letter-spacing:.04em; color:var(--muted); font-weight:600;
    list-style:none; user-select:none; padding:4px 0; transition:color .12s;
  }
  .section-block>summary::-webkit-details-marker{display:none}
  .section-block>summary::before{
    content:"\25B8"; display:inline-block; margin-right:6px;
    font-size:10px; transition:transform .15s; color:var(--muted-2);
  }
  .section-block[open]>summary::before{transform:rotate(90deg)}
  .section-block>summary:hover{color:var(--ink)}
  .section-block[open]>summary{margin-bottom:2px}
  .section-block[open]>summary::before{color:var(--accent)}
  #message{
    margin-top:10px; padding:8px; border:1px solid var(--rule);
    border-radius:6px; background:var(--soft);
    font-family:var(--mono); font-size:12px;
  }
  :global(.pgn-db-list), :global(.pgn-db-results){display:grid; gap:6px; margin-top:8px}
  :global(.pgn-db-list .empty), :global(.pgn-db-results .empty){
    padding:7px 8px; border:1px dashed var(--rule);
    border-radius:6px; background:#fff;
  }
  :global(.pgn-db-card){display:grid; grid-template-columns:1fr auto; gap:3px 8px; text-align:left; background:#fff}
  :global(.pgn-db-card.selected){border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset}
  :global(.pgn-db-card .name){font-weight:700; color:var(--ink)}
  :global(.pgn-db-card .meta), :global(.pgn-db-hit .meta){color:var(--muted); font-size:11px}
  :global(.pgn-db-hit){padding:7px 8px; border:1px solid var(--rule); border-radius:6px; background:#fff}
  :global(.pgn-db-hit .name){font-weight:700; color:var(--ink)}
  :global(.pgn-db-hit .moves){font-family:var(--mono); color:var(--ink); margin-top:3px; overflow-wrap:anywhere}
  :global(table.opening){width:100%; border-collapse:collapse; font-size:13px; margin-top:8px}
  :global(table.opening th), :global(table.opening td){text-align:left; padding:5px 8px; border-bottom:1px solid var(--rule)}
  :global(table.opening th){color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase}
  :global(table.opening td.num), :global(table.opening th.num){text-align:right; font-family:var(--mono)}
  :global(table.opening tr.mv){cursor:pointer}
  :global(table.opening tr.mv:hover td){background:var(--soft)}
  :global(table.opening .san){font-family:var(--mono); font-weight:700}
  :global(.wdlbar){
    display:flex; height:14px; border-radius:3px; overflow:hidden;
    min-width:120px; border:1px solid var(--rule);
  }
  :global(.wdlbar .w){background:#f4f1e8}
  :global(.wdlbar .d){background:#b9b3a4}
  :global(.wdlbar .b){background:#3a3a3a}
  @media (max-width:900px){
    main{grid-template-columns:1fr}
    .app-sidebar{position:static; max-height:none; overflow-y:visible}
  }
  @media (prefers-reduced-motion: reduce){
    .evalbar .white{transition:none}
    :global(.lines li:hover){background:none}
    .section-block>summary::before{transition:none}
  }
</style>
