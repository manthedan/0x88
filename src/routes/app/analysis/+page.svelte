<script lang="ts">
  import { onMount } from 'svelte';
  const title = "0x88 Chess — analysis board";
  const description = "Multi-engine chess analysis in your browser: compare Leela Chess Zero and Stockfish lines side by side, review games with accuracy scores, and explore openings from your own games.";
  const styles = "/* Page-specific layout only; tokens, base typography, header, and footer\n   come from app-shell.css. */\n*{box-sizing:border-box}\nmain{display:grid;grid-template-columns:minmax(320px,540px) minmax(340px,1fr);gap:20px;align-items:start;padding:18px}\n.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px;min-width:0}\n.board-wrap{display:flex;gap:8px}\n.evalbar{width:18px;border:1px solid var(--rule);border-radius:4px;overflow:hidden;background:#3a3a3a;position:relative;flex:0 0 auto}\n.evalbar .white{position:absolute;left:0;right:0;bottom:0;background:#f4f1e8;transition:height .25s}\n.evalbar .mid{position:absolute;left:0;right:0;top:50%;height:1px;background:var(--accent);opacity:.6}\n.board-shell{flex:1 1 auto;aspect-ratio:1}#ground{width:100%;height:100%;border-radius:6px;overflow:hidden;box-shadow:0 14px 38px -26px #000;background:var(--board-dark)}\n.cg-wrap{width:100%!important;height:100%!important}\n.nav{display:flex;gap:6px;margin-top:10px}\nbutton{border:1px solid var(--rule);border-radius:6px;padding:8px 10px}\nbutton.primary{background:var(--accent);border-color:var(--accent);color:white;font-weight:700}\n.nav button{flex:1 1 auto}\n.row{display:flex;flex-wrap:wrap;gap:8px;align-items:end;margin-top:10px}\n.engine-list{display:grid;gap:6px;margin-top:10px}.engine-row{display:flex;align-items:center;gap:6px;min-width:0}.engine-row select{flex:1 1 0;min-width:0;font:inherit;font-size:13px;padding:6px 8px;border:1px solid var(--rule);border-radius:6px;background:white}.engine-row .arrow{color:var(--muted);font-weight:700}.engine-row .row-strength{flex:0 0 72px;width:72px;font:inherit;font-size:13px;font-family:var(--mono);padding:6px 8px;border:1px solid var(--rule);border-radius:6px}.engine-row .row-unit{flex:0 0 auto;color:var(--muted);font-size:11px}.engine-row .row-rm{flex:0 0 auto;padding:4px 9px;line-height:1;color:var(--muted)}\n.engine-logo{width:16px;height:16px;object-fit:contain;border-radius:3px;vertical-align:middle;flex:0 0 auto}.engine-row .engine-logo{width:18px;height:18px}.legend .engine-logo,.lines .engine-logo,table.engine-compare .engine-logo{width:14px;height:14px}.engine-name-with-logo{display:inline-flex;align-items:center;gap:5px}\n.field{display:grid;gap:3px}.field label{font-size:11px;color:var(--muted)}\ninput[type=number]{width:80px}\ntextarea{min-height:64px;font-family:var(--mono);font-size:12px;resize:vertical}\n.lines{list-style:none;margin:8px 0 0;padding:0}\n.lines li{display:grid;grid-template-columns:62px 1fr;gap:8px;align-items:baseline;padding:6px 6px;border-top:1px solid var(--rule);cursor:pointer;font-size:13px}\n.lines li:hover{background:var(--soft)}\n.lines .score{font-family:var(--mono);font-weight:700}\n.lines .score.neg{color:#a5461b}.lines .score.pos{color:var(--accent)}\n.lines .pv{font-family:var(--mono);overflow-wrap:anywhere}.lines .eng{display:inline-flex;align-items:center;gap:4px;font-size:10px;color:var(--muted)}\n.lines li{padding-left:9px}\n.review-chart svg{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;background:#fff;margin-top:6px}\n.review-summary{display:flex;gap:18px;margin:6px 0;font-size:12px}\n.review-summary .acc{font-size:16px;font-weight:700;font-family:var(--mono)}\n.review-badge{display:inline-block;border-radius:3px;padding:0 5px;font-size:10px;font-weight:700;color:#fff;vertical-align:1px}\n.review-badge.best{background:#4a7a2a}.review-badge.good{background:#7a9a4a}\n.review-badge.inaccuracy{background:#c08a00}.review-badge.mistake{background:#d2691e}\n.review-badge.blunder{background:#a5461b}.review-badge.forced{background:#999}\n.review-critical{list-style:none;margin:6px 0 0;padding:0}\n.review-critical li{padding:4px 6px;border-top:1px solid var(--rule);cursor:pointer;font-size:12px}\n.review-critical li:hover{background:var(--soft)}\n.review-critical .mono{font-family:var(--mono)}\n.lines li.placeholder{display:block;color:var(--muted);border-left:none;cursor:default}\n.lines li.placeholder:hover{background:none}\n.compare-summary{margin-top:8px;padding:8px;border:1px solid var(--rule);border-radius:6px;background:#fff;font-size:12px;color:var(--muted)}\ntable.engine-compare{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}table.engine-compare th,table.engine-compare td{text-align:left;padding:5px 7px;border-bottom:1px solid var(--rule);vertical-align:top}table.engine-compare th{color:var(--muted);font-weight:600;font-size:10px;text-transform:uppercase}table.engine-compare .mono{font-family:var(--mono)}table.engine-compare .pv{font-family:var(--mono);overflow-wrap:anywhere}table.engine-compare .agree{color:var(--accent);font-weight:700}.profile-row{align-items:end}.profile-row select{min-width:150px}.profile-row input[type=text]{min-width:130px}.profile-summary{margin-top:6px;padding:7px 8px;border:1px dashed var(--rule);border-radius:6px;background:#fff}\ndetails.runtime-details{margin:6px 0}\ndetails.runtime-details summary{cursor:pointer;font-size:12px;color:var(--muted)}\ndetails.runtime-details summary:hover{color:inherit}\n.advanced-settings{margin-top:10px;border:1px solid var(--rule);border-radius:7px;background:#fff;padding:8px 10px}.advanced-settings>summary{cursor:pointer;color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}.advanced-settings>summary:hover{color:inherit}.advanced-settings[open]>summary{margin-bottom:4px}\n.status-inline{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;margin:6px 0 2px}\n.status-inline dt{color:var(--muted);display:inline}\n.status-inline dd{margin:0 0 0 6px;font-family:var(--mono);display:inline;overflow-wrap:anywhere}\n.status-inline div{display:flex;align-items:baseline}\n.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-size:12px;color:var(--muted)}\n.legend .key{display:inline-flex;align-items:center;gap:5px}\n.legend .dot{width:11px;height:11px;border-radius:3px;display:inline-block}\n.movelist{font-family:var(--mono);font-size:13px;line-height:1.7;margin-top:8px;max-height:240px;overflow:auto}\n.maia3-grid{margin-top:8px;font-size:13px}\n.maia3-grid .maia3-row{display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid var(--rule)}\n.maia3-grid .maia3-elo{font-family:var(--mono);width:48px;color:var(--muted)}\n.maia3-grid .maia3-score{font-family:var(--mono);width:44px}\n.maia3-grid .maia3-move{display:inline-flex;align-items:center;gap:4px;margin-right:10px;font-family:var(--mono)}\n.maia3-grid .maia3-bar{display:inline-block;height:8px;background:#5a6e2a;border-radius:2px;opacity:0.75}\n.movelist .mv{cursor:pointer;padding:0 3px;border-radius:4px}.movelist .mv:hover{background:var(--soft)}\n.movelist .mv.current{background:var(--accent);color:white}\n.movelist .var{color:var(--muted)}\n.status-grid{display:grid;grid-template-columns:90px 1fr;gap:4px 10px;font-size:12px;margin-top:8px}\n.status-grid dt{color:var(--muted)}.status-grid dd{margin:0;font-family:var(--mono);overflow-wrap:anywhere}\nh2{font-size:13px;margin:14px 0 0;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}\n#message{margin-top:10px;padding:8px;border:1px solid var(--rule);border-radius:6px;background:var(--soft);font-family:var(--mono);font-size:12px}\n.small{font-size:12px;color:var(--muted)}\n.pgn-db-list,.pgn-db-results{display:grid;gap:6px;margin-top:8px}.pgn-db-list .empty,.pgn-db-results .empty{padding:7px 8px;border:1px dashed var(--rule);border-radius:6px;background:#fff}.pgn-db-card{display:grid;grid-template-columns:1fr auto;gap:3px 8px;text-align:left;background:#fff}.pgn-db-card.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}.pgn-db-card .name{font-weight:700;color:var(--ink)}.pgn-db-card .meta,.pgn-db-hit .meta{color:var(--muted);font-size:11px}.pgn-db-hit{padding:7px 8px;border:1px solid var(--rule);border-radius:6px;background:#fff}.pgn-db-hit .name{font-weight:700;color:var(--ink)}.pgn-db-hit .moves{font-family:var(--mono);color:var(--ink);margin-top:3px;overflow-wrap:anywhere}\ntable.opening{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}\ntable.opening th,table.opening td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--rule)}\ntable.opening th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase}\ntable.opening td.num,table.opening th.num{text-align:right;font-family:var(--mono)}\ntable.opening tr.mv{cursor:pointer}table.opening tr.mv:hover td{background:var(--soft)}\ntable.opening .san{font-family:var(--mono);font-weight:700}\n.wdlbar{display:flex;height:14px;border-radius:3px;overflow:hidden;min-width:120px;border:1px solid var(--rule)}\n.wdlbar .w{background:#f4f1e8}.wdlbar .d{background:#b9b3a4}.wdlbar .b{background:#3a3a3a}\n@media(max-width:900px){main{grid-template-columns:1fr}}\n[hidden]{display:none !important}\n:where(button,select,input,textarea,a):focus-visible{outline:2px solid var(--accent);outline-offset:2px}\n@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}.evalbar .white{transition:none}.lines li:hover{background:none}}";
  ;
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
<div>{@html `<style>${styles}</style>`}</div>
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
  </section>
  <section class="panel" aria-label="Analysis">
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
    <h2>Engine comparison</h2>
    <div id="engineConsensus" class="compare-summary">No analysis yet.</div>
    <table id="engineCompare" class="engine-compare"><thead><tr><th>Engine</th><th>Best</th><th>Eval</th><th>Δ</th><th>Search</th><th>PV</th></tr></thead><tbody></tbody></table>
    <h2>Engine lines</h2>
    <div id="engineLegend" class="legend"></div>
    <ol id="lines" class="lines"></ol>
    <dl class="status-inline">
      <div><dt>Side</dt><dd id="sideToMove">—</dd></div>
      <div><dt>Eval</dt><dd id="posEval">—</dd></div>
    </dl>
    <h2>Human moves · Maia3</h2>
    <div class="row">
      <button id="maia3Enable" type="button">Load Maia3 (~28MB, cached)</button>
      <span id="maia3PanelStatus" class="small"></span>
    </div>
    <div id="maia3Grid" class="maia3-grid" hidden></div>
    <div id="maia3Caption" class="small" hidden>What rated humans actually play here (Maia3 move predictions; score = expected points for White in a human game at that rating, not an engine eval). ✓ = matches the current best engine move.</div>
    <h2>Moves</h2>
    <div id="movelist" class="movelist"></div>
    <h2>Game review</h2>
    <div class="row">
      <button id="reviewGame" type="button">Review game</button>
      <button id="reviewStop" type="button" disabled>Stop</button>
      <button id="reviewCopyPgn" type="button" hidden>Copy annotated PGN</button>
    </div>
    <div id="reviewStatus" class="small">Load a PGN or play moves, then review the mainline with the first selected UCI engine.</div>
    <div id="reviewSummary" hidden></div>
    <div id="reviewChart" class="review-chart" hidden></div>
    <ol id="reviewCritical" class="review-critical" hidden></ol>
    <h2>Opening explorer</h2>
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
  </section>
</main>
