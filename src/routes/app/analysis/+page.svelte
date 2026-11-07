<script lang="ts">
import { onMount } from 'svelte';
import { afterNavigate } from '$app/navigation';
import BrowserCapabilities from '$lib/components/BrowserCapabilities.svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';

let mountedUrl: URL | null = null;
afterNavigate(({ to }) => {
  if (mountedUrl && to?.url.pathname === mountedUrl.pathname && to.url.search !== mountedUrl.search) location.reload();
});
const title = '0x88 Chess — analysis board';
const description =
  'Multi-engine chess analysis in your browser: compare Leela Chess Zero and Stockfish lines side by side, review games with accuracy scores, and explore openings from your own games.';
const devMode = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('dev');
onMount(() => {
  mountedUrl = new URL(location.href);
  let cleanup: () => void = () => undefined;
  let mounted = true;
  void import('../../../lc0/analysisBrowser')
    .then((module) => {
      if (!mounted) return;
      cleanup = module.mountAnalysisBrowser();
    })
    .catch((error) => {
      if (!mounted) return;
      console.error('[analysis] failed to load page controller', error);
      const node = document.getElementById('message');
      if (node) node.textContent = `Page failed to initialize: ${error instanceof Error ? error.message : String(error)}`;
    });
  const multiPv = document.getElementById('multiPvInput') as HTMLInputElement | null;
  const dec = document.getElementById('multiPvDec');
  const inc = document.getElementById('multiPvInc');
  const clampPv = (v: number) => Math.min(10, Math.max(1, v));
  dec?.addEventListener('click', () => {
    if (multiPv) {
      multiPv.value = String(clampPv(Number(multiPv.value) - 1));
      multiPv.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  inc?.addEventListener('click', () => {
    if (multiPv) {
      multiPv.value = String(clampPv(Number(multiPv.value) + 1));
      multiPv.dispatchEvent(new Event('change', { bubbles: true }));
    }
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

<SiteHeader pageTitle="Analysis" />
<main id="main" class="app-main analysis-workspace">
  <div class="board-column">
  <section class="panel board-panel" aria-label="Board">
    <div class="board-wrap">
      <div class="evalbar" title="Evaluation (white advantage)"><div class="mid"></div><div class="white" id="evalWhite" style="height:50%"></div></div>
      <div class="board-shell">
        <div id="ground"></div>
        <div id="keyboardBoard"></div>
      </div>
      <p id="boardInstructions" class="sr-only">Arrow keys explore squares. Enter or Space selects a piece, then a legal destination. Escape cancels selection. Tab leaves the board.</p>
      <p id="boardAnnouncement" class="sr-only" aria-live="polite"></p>
    </div>
    <div class="board-toolbar">
    <span id="boardEval" class="board-eval" title="Evaluation from White's perspective">—</span>
    <div class="nav" aria-label="Move navigation">
      <button id="navStart" type="button" title="Start (Up)" aria-label="Go to first move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14M18 5l-8 7 8 7" /></svg></button>
      <button id="navBack" type="button" title="Back (Left)" aria-label="Go back one move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-8 7 8 7" /></svg></button>
      <button id="navForward" type="button" title="Forward (Right)" aria-label="Go forward one move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 8 7-8 7" /></svg></button>
      <button id="navEnd" type="button" title="End (Down)" aria-label="Go to last move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M6 5l8 7-8 7" /></svg></button>
      <button id="flip" type="button" title="Flip board" aria-label="Flip board"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 20V4m-4 4 4-4 4 4m4-4v16m-4-4 4 4 4-4" /></svg></button>
    </div>
    <div class="board-actions">
      <button id="openPosition" type="button" aria-controls="positionTools" aria-expanded="false">Load FEN / PGN</button>
      <span id="sideToMove">White to move</span>
    </div>
    </div>
    <details id="positionTools" class="position-tools">
      <summary hidden>Position input &amp; export</summary>
      <div class="position-tools-body">
        <div class="row fen-row">
          <div class="field"><label for="fenInput">FEN</label>
            <input id="fenInput" type="text" spellcheck="false" autocomplete="off" placeholder="Paste a FEN" /></div>
          <button id="loadFen" type="button">Load FEN</button>
          <button id="reset" type="button">Reset</button>
        </div>
        <div class="pgn-heading"><label for="pgnInput">PGN</label><small>Variations supported</small></div>
        <textarea id="pgnInput" spellcheck="false" placeholder="Paste a PGN"></textarea>
        <div class="row pgn-actions">
          <button id="loadPgn" type="button">Load PGN</button>
          <button id="copyPgn" type="button">Copy PGN</button>
        </div>
      </div>
    </details>
  </section>
  </div>
  <section class="panel app-sidebar" aria-label="Analysis">
    <header class="analysis-overview">
      <div class="evaluation-readout">
        <span>White's evaluation</span>
        <strong id="posEval">—</strong>
        <small id="evalEngine">Awaiting analysis</small>
      </div>
      <div class="best-move-readout">
        <span>Best move</span>
        <strong id="bestMove">—</strong>
        <small id="bestMoveEngine">Click a line to play</small>
      </div>
      <div class="analyze-actions">
        <button id="analyze" class="primary" type="button" disabled>Analyze</button>
        <button id="stop" type="button" disabled>Stop</button>
        <button id="manageEngines" type="button" aria-haspopup="dialog" aria-controls="engineManager">Manage engines · <span id="engineCount">1</span></button>
      </div>
    </header>
    <div id="engineSummaries" class="engine-summaries" aria-label="Engine summaries and board arrow legend"></div>
    <div id="message" aria-live="polite">Loading model…</div>
    <div id="runtimeWarning" class="runtime-warning" role="status" aria-live="polite" hidden></div>
    <div id="downloadProgress" class="model-load-progress" hidden></div>
    <div id="analysisSearchProgress" class="search-progress-grid" hidden></div>
    <div id="analysisContent" class="analysis-content" data-view="lines">
    <div class="workbench-tabs" role="tablist" aria-label="Analysis views">
      <button id="linesTab" type="button" role="tab" aria-selected="true" aria-controls="linesPanel" data-view="lines">Lines</button>
      <button id="comparisonTab" type="button" role="tab" aria-selected="false" aria-controls="comparisonSection" tabindex="-1" data-view="comparison">Comparison</button>
      <button id="movesTab" class="mobile-moves-tab" type="button" role="tab" aria-selected="false" aria-controls="historyPanel" tabindex="-1" data-view="moves">Moves</button>
      <span>Click a line to play</span>
    </div>
    <section id="historyPanel" class="history-panel" aria-label="Game moves">
      <header><h2>Game moves</h2><span>← → to navigate</span></header>
      <div id="movelist" class="movelist move-panel" aria-label="Move history"></div>
    </section>
    <div id="comparisonSection" class="section-block comparison-panel" role="tabpanel" aria-labelledby="comparisonTab" tabindex="0">
      <h2 class="sr-only">Comparison</h2>
      <div id="engineConsensus" class="compare-summary">No analysis yet.</div>
      <table id="engineCompare" class="engine-compare"><thead><tr><th scope="col">Engine</th><th scope="col">Best</th><th scope="col">Eval</th><th scope="col" id="deltaBaseline">Δ cp</th></tr></thead><tbody></tbody></table>
    </div>
    <div id="linesPanel" class="section-block lines-panel" role="tabpanel" aria-labelledby="linesTab" tabindex="0">
      <div id="engineLegend" class="legend"></div>
      <div id="lines" class="lines"></div>
    </div>
    </div>
    <dialog id="engineManager" class="engine-manager" aria-labelledby="engineManagerTitle">
      <header class="manager-heading"><h2 id="engineManagerTitle">Manage engines</h2><button id="closeEngineManager" type="button" aria-label="Close engine settings">✕</button></header>
      <div id="engineSetup" class="engine-setup">
      <p id="engineSetupSummary">Lc0 · Small</p>
      <p class="small">Compare engines on the same position. Set each engine's model and search budget below.</p>
      <div class="analyze-controls">
        <label class="toggle" for="autoAnalyze" title="Analyze automatically after the position changes">
          <input type="checkbox" id="autoAnalyze" checked />
          <span class="toggle-track"><span class="toggle-thumb"></span></span>
          <span class="toggle-label">Auto analysis</span>
        </label>
        <div class="stepper">
          <button id="multiPvDec" type="button" class="stepper-btn" aria-label="Decrease lines">&minus;</button>
          <label class="stepper-field" for="multiPvInput">
            <span class="stepper-label">Lines</span>
            <input id="multiPvInput" type="number" min="1" max="10" step="1" value="3" />
          </label>
          <button id="multiPvInc" type="button" class="stepper-btn" aria-label="Increase lines">+</button>
        </div>
      </div>
      <div id="engineList" class="engine-list">
        <div class="engine-list-loading">Preparing engine controls…</div>
      </div>
      <div class="row add-engine-row">
        <button id="addEngine" type="button">+ Add engine</button>
        {#if devMode}
        <div class="field"><label for="engineProfileSelect">Profile</label>
          <select id="engineProfileSelect"><option value="">default</option></select></div>
        {/if}
      </div>

    {#if devMode}
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
    {:else}
    <!-- Hidden but in DOM so analysisBrowser.ts can find them when dev mode is off -->
    <div hidden>
      <select id="engineProfileSelect"><option value="">default</option></select>
      <input id="engineProfileName" type="text" />
      <button id="saveEngineProfile" type="button">Save</button>
      <button id="deleteEngineProfile" type="button">Delete</button>
      <button id="exportEngineProfiles" type="button">Export</button>
      <button id="importEngineProfiles" type="button">Import</button>
      <input id="importEngineProfilesFile" type="file" />
      <div id="engineProfileSummary"></div>
      <select id="lc0RuntimeSelect"><option value="onnx">ORT ONNX</option><option value="hybrid-ort-heads">WGSL encoder + ORT heads</option><option value="hybrid-wgsl-heads">WGSL encoder + WGSL heads (experimental)</option></select>
      <span id="backend"></span>
      <span id="runtimeAudit"></span>
      <div id="recklessRuntimeInfo"></div>
    </div>
    {/if}
      </div>
    </dialog>
    <details class="section-block human-moves">
      <summary>Human moves <span>Maia3</span></summary>
      <div class="row">
        <button id="maia3Enable" type="button">Load Maia3 (~28MB, cached)</button>
        <span id="maia3PanelStatus" class="small"></span>
      </div>
      <div id="maia3Grid" class="maia3-grid" hidden></div>
      <div id="maia3Caption" class="small" hidden>What rated humans actually play here (Maia3 move predictions; score = expected points for White in a human game at that rating, not an engine eval). ✓ = matches the current best engine move.</div>
    </details>
    <div class="capabilities-slot"><BrowserCapabilities /></div>
    <!-- Parked: features below are intentionally hidden. Remove hidden to re-enable. -->
    <details class="section-block" hidden>
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
    <details class="section-block" hidden>
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
  .analysis-workspace{
    --muted:color-mix(in srgb, var(--ink) 78%, var(--panel));
    --muted-2:var(--muted); --text-soft:var(--muted);
    --border-input:color-mix(in srgb, var(--ink) 38%, var(--panel));
    --board-size:min(720px, calc(100dvh - 166px), calc(100vw - 520px));
    grid-template-columns:calc(var(--board-size) + 34px) minmax(0, 470px); gap:16px;
    max-width:1288px; padding:16px 24px 32px; font-size:14px; font-variant-numeric:tabular-nums;
  }
  .board-column{min-width:0}
  .board-panel{min-width:0; padding:8px}
  .board-wrap{display:grid; grid-template-columns:14px 1fr; gap:4px; position:relative}
  .board-toolbar{display:flex; align-items:center; gap:8px; margin-top:8px}
  .board-eval{flex:0 0 48px; text-align:center; color:var(--ink); font:700 12px var(--mono)}
  .analysis-workspace .app-sidebar{position:static; display:flex; flex-direction:column; min-width:0}
  .sr-only{position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip-path:inset(50%); white-space:nowrap; border:0}
  .board-shell{position:relative}
  :global(.analysis-workspace #ground coords){font-size:13px; font-weight:750; text-shadow:0 1px 1px #0008}
  :global(.analysis-workspace #ground coords.files){text-shadow:none}
  :global(.analysis-workspace #ground .cg-shapes){filter:drop-shadow(0 0 1px #fff9)}
  :global(.analysis-workspace #keyboardBoard){position:absolute; inset:0; z-index:5; pointer-events:none}
  :global(.analysis-workspace .keyboard-row){display:contents}
  :global(.analysis-workspace .keyboard-square){position:absolute; width:12.5%; height:12.5%; padding:0; min-height:0; border:0; border-radius:0; background:transparent; pointer-events:none}
  :global(.analysis-workspace .keyboard-square:focus-visible){outline:3px solid #fff; outline-offset:-4px; box-shadow:inset 0 0 0 6px #225ca0; background:#fff2}
  .analysis-workspace .section-block>summary{font:650 13px/1.3 var(--sans); letter-spacing:0; text-transform:none; color:var(--ink)}
  .analysis-workspace .section-block>summary::before{font-size:12px}
  .board-actions{display:flex; flex-direction:column; align-items:center; gap:2px}
  .board-actions button{font-size:12px; padding:5px 8px; color:var(--ink); min-height:30px; white-space:nowrap}
  .board-actions span{font-size:11px; color:var(--muted)}
  .nav svg{width:20px; height:20px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round}
  .analysis-workspace :global(button:focus-visible), .analysis-workspace :global(summary:focus-visible), .analysis-workspace :global(select:focus-visible), .analysis-workspace :global(input:focus-visible){outline:2px solid var(--accent); outline-offset:2px}
  .analysis-content{display:grid; min-width:0}
  .history-panel{grid-row:1; min-width:0; overflow:hidden; padding:0; border-top:1px solid var(--rule)}
  .history-panel header{display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-bottom:1px solid var(--rule)}
  .history-panel h2{font:650 14px var(--sans); letter-spacing:0; text-transform:none; margin:0; padding:0}
  .history-panel header span{font-size:12px; color:var(--muted)}
  .engine-manager{width:min(600px, calc(100vw - 24px)); max-height:85dvh; padding:20px; color:var(--ink); background:var(--panel); border:1px solid var(--border-input); border-radius:12px; box-shadow:0 20px 80px #0008}
  .engine-manager::backdrop{background:#0009}
  .manager-heading{display:flex; align-items:center; justify-content:space-between; gap:12px}
  .manager-heading h2{font:650 20px var(--sans); color:var(--ink); text-transform:none; letter-spacing:0; margin:0}
  .engine-setup>p{margin:12px 0}
  .engine-setup>p:first-child{font-weight:650}
  .engine-summaries{padding:0 12px 4px}
  :global(.engine-summary){display:grid; grid-template-columns:minmax(0,1fr) 62px 58px; align-items:center; gap:8px; min-height:44px; border-top:1px solid var(--rule)}
  :global(.engine-summary-name){padding:5px 0; min-width:0; border:0; background:transparent; text-align:left; color:var(--ink)}
  :global(.engine-summary-name strong){display:flex; align-items:center; gap:7px; font:650 13px var(--sans)}
  :global(.engine-summary-name small){display:block; margin:3px 0 0 15px; font:11px var(--mono); color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  :global(.engine-dot){display:inline-block; width:8px; height:8px; flex:0 0 8px; border-radius:50%; background:var(--engine-color); box-shadow:0 0 0 1px color-mix(in srgb, var(--engine-color), white 30%)}
  :global(.engine-summary-score){font:700 15px var(--mono); text-align:right}
  :global(.engine-summary .compare-move){font-size:14px; min-height:30px; padding:4px; text-align:center}
  .workbench-tabs{grid-row:2; display:flex; align-items:center; gap:4px; padding:0 8px; border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); background:var(--panel-inset)}
  .workbench-tabs button{border:0; border-radius:0; border-bottom:2px solid transparent; min-height:36px; padding:8px; font-size:13px; background:transparent}
  .workbench-tabs button[aria-selected=true]{color:var(--ink); border-bottom-color:var(--accent); font-weight:650}
  .workbench-tabs span{margin-left:auto; font-size:11px; color:var(--muted)}
  .workbench-tabs .mobile-moves-tab{display:none}
  .analysis-workspace .lines-panel, .analysis-workspace .comparison-panel{grid-row:3; border:0; padding:8px 12px; max-height:clamp(150px, calc(100dvh - 690px), 256px); overflow:auto; overscroll-behavior:contain; scrollbar-gutter:stable}
  :global(.analysis-content[data-view=comparison]) .lines-panel, :global(.analysis-content:not([data-view=comparison])) .comparison-panel{display:none}
  :global(.analysis-workspace .engine-badge){display:inline-flex; justify-content:center; align-items:center; width:26px; height:26px; flex:0 0 26px; border:2px solid var(--border-input); border-radius:6px; background:var(--panel-inset); color:var(--ink); font:700 12px var(--mono)}
  :global(.analysis-workspace .engine-row select), :global(.analysis-workspace .engine-row .row-strength){font-size:14px}
  :global(.analysis-workspace .engine-row .row-rm){min-height:36px}
  :global(.analysis-workspace .engine-row .row-unit){font-size:12px}
  :global(.analysis-workspace .engine-row .row-strength){flex-basis:66px; width:66px}
  .evalbar{
    width:14px; border:1px solid var(--border-input); border-radius:5px;
    overflow:hidden; background:var(--eval-black); position:relative; flex:0 0 auto;
    box-sizing:border-box;
  }
  .evalbar .white{
    position:absolute; left:0; right:0; bottom:0;
    background:var(--eval-white); transition:height .25s;
  }
  .evalbar .mid{
    position:absolute; left:0; right:0; top:50%;
    height:1px; background:var(--accent); opacity:.6;
  }
  .board-shell{min-width:0}
  .nav{display:flex; flex:1; gap:5px; min-width:0}
  .nav button{
    flex:1 1 auto; height:36px; border-radius:7px;
    border:1px solid var(--border-input); background:var(--panel-inset);
    color:var(--text-soft); font-size:14px; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
  }
  .nav button:hover:not(:disabled){border-color:var(--border-input-hover); background:var(--card); color:var(--ink)}
  .nav button:disabled{opacity:.35; cursor:default}
  .row{margin-top:10px}
  .engine-list{margin-top:0}
  .engine-list-loading{
    padding:8px 0; color:var(--muted); font-family:var(--mono); font-size:11px;
  }
  :global(.runtime-warning){
    margin-top:7px; padding:7px 9px; border:1px solid color-mix(in srgb, var(--warn) 45%, var(--rule));
    border-radius:6px; background:var(--warn-soft); color:var(--warn); font-family:var(--mono); font-size:10px; line-height:1.4;
  }
  .field{margin-top:10px}
  .position-tools{border:0; overflow:hidden}
  .position-tools[open]{margin-top:10px; border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset)}
  .position-tools>summary{
    padding:11px 13px; cursor:pointer; list-style:none; user-select:none;
    font-size:13px; font-weight:650; color:var(--muted);
  }
  .position-tools>summary::-webkit-details-marker{display:none}
  .position-tools>summary::before{content:"+"; display:inline-block; width:14px; color:var(--accent); font-size:13px}
  .position-tools[open]>summary::before{content:"−"}
  .position-tools-body{padding:2px 13px 13px; border-top:1px solid var(--rule)}
  .fen-row .field{flex:1 1 100%; margin-top:10px}
  .pgn-heading{display:flex; justify-content:space-between; align-items:center; margin:14px 0 6px}
  .pgn-heading label{font-size:13px; color:var(--muted)}
  .pgn-heading small{color:var(--muted); font-size:12px}
  .pgn-actions{margin-top:8px}
  input[type=number]:not(.stepper-field input){width:80px}
  :global(.lines){margin:0; padding:0}
  :global(.engine-result + .engine-result){margin-top:6px}
  :global(.engine-result h3){display:flex; gap:8px; align-items:center; margin:0 0 4px; font:650 13px var(--sans); color:var(--ink)}
  :global(.engine-result h3 small){margin-left:auto; font:11px var(--mono); color:var(--muted); font-weight:400}
  :global(.engine-result ol){list-style:none; margin:0; padding:0}
  :global(.result-line){border-top:1px solid var(--rule); border-radius:4px; position:relative; min-height:30px}
  :global(.result-line:hover), :global(.result-line:focus-within), :global(.result-line.previewing){background:var(--accent-soft)}
  :global(.line-play){display:grid; grid-template-columns:12px 55px minmax(0,1fr); gap:6px; width:calc(100% - 32px); min-height:30px; padding:4px 2px; text-align:left; border:0; background:transparent; color:var(--ink); align-items:baseline}
  :global(.line-rank){font:12px var(--mono); color:var(--muted)}
  :global(.lines .score){font:700 14px var(--mono); color:var(--ink)}
  :global(.lines .pv){font:14px/1.5 var(--sans); color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  :global(.pv-expanded){margin:0}
  :global(.pv-expanded summary){position:absolute; right:0; top:0; display:grid; place-items:center; width:32px; height:30px; font-size:20px; color:var(--muted); cursor:pointer; list-style:none}
  :global(.pv-expanded summary::-webkit-details-marker){display:none}
  :global(.pv-expanded[open] summary){transform:rotate(180deg)}
  :global(.pv-expanded p){font:14px/1.8 var(--sans); color:var(--ink); margin:0 8px 6px 22px; overflow-wrap:anywhere}
  :global(.pv-expanded small){font-size:12px; color:var(--muted)}
  :global(.line-error){padding:10px; color:var(--warn); font-size:14px; overflow-wrap:anywhere}
  :global(.engine-logo){width:16px; height:16px; object-fit:contain; border-radius:3px; vertical-align:middle; flex:0 0 auto}
  :global(.engine-logo-placeholder){display:inline-block; width:16px; height:16px; border-radius:3px; background:var(--rule-soft); flex:0 0 auto; vertical-align:middle}
  :global(.engine-row .engine-logo), :global(.engine-row .engine-logo-placeholder){width:18px; height:18px}
  :global(.legend .engine-logo), :global(.lines .engine-logo), :global(table.engine-compare .engine-logo),
  :global(.legend .engine-logo-placeholder), :global(.lines .engine-logo-placeholder), :global(table.engine-compare .engine-logo-placeholder){width:14px; height:14px}
  :global(.engine-name-with-logo){display:inline-flex; align-items:center; gap:5px}
  :global(.lines li.placeholder){display:block; color:var(--muted); border-left:none; cursor:default}
  :global(.lines li.placeholder:hover){background:none}
  :global(.review-chart svg){
    display:block; width:100%; height:auto;
    border:1px solid var(--rule); border-radius:6px; background:var(--card); margin-top:6px;
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
  :global(.review-critical li:hover){background:var(--accent-soft)}
  :global(.review-critical .mono){font-family:var(--mono)}
  :global(.compare-summary){
    margin-top:0; padding:8px 10px; border:0; border-left:2px solid var(--accent);
    border-radius:0 var(--radius-sm) var(--radius-sm) 0;
    background:var(--accent-soft); font-size:14px; line-height:1.5; color:var(--ink);
  }
  :global(table.engine-compare){width:100%; border-collapse:collapse; font-size:14px; margin-top:8px}
  :global(table.engine-compare th), :global(table.engine-compare td){
    text-align:left; padding:6px 5px;
    border-bottom:1px solid var(--rule); vertical-align:middle;
  }
  :global(table.engine-compare thead th){color:var(--muted); font-weight:500; font-size:12px}
  :global(table.engine-compare tbody th){color:var(--ink); font-weight:550; font-size:13px}
  :global(.compare-move){font:700 14px var(--mono); padding:6px; min-height:36px; color:var(--ink); background:var(--panel-inset)}
  :global(table.engine-compare .mono){font-family:var(--mono); white-space:nowrap}
  :global(table.engine-compare .pv){font-family:var(--mono); font-size:10px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:150px}
  :global(table.engine-compare .agree){color:var(--accent); font-weight:700}
  :global(table.engine-compare tbody tr:hover){background:var(--accent-soft)}
  :global(.model-load-progress), :global(.search-progress-grid){
    margin-top:8px; padding:8px; border:1px solid var(--rule);
    border-radius:var(--radius-sm); background:var(--panel-inset);
  }
  :global(.model-load-progress progress), :global(.search-progress-grid progress), :global(.search-progress-cell progress){
    width:100%; height:9px; accent-color:var(--accent);
  }
  :global(.loading-progress-row), :global(.search-progress-row){display:grid; gap:3px; margin:4px 0}
  :global(.dl-label), :global(.search-progress-text){font-family:var(--mono); font-size:11px; color:var(--muted)}
  :global(.search-activity-row){
    min-width:0; min-height:20px; display:grid; grid-template-columns:7px minmax(0,1fr);
    align-items:center; gap:8px;
  }
  :global(.search-activity-mark){
    width:7px; height:7px; border-radius:1px; background:var(--accent);
    box-shadow:0 0 0 2px color-mix(in srgb, var(--accent) 16%, transparent);
  }
  :global(.search-activity-row .search-progress-text){
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .analysis-overview{display:grid; grid-template-columns:1fr 1fr auto; gap:8px; padding:10px 12px; background:var(--panel-inset)}
  .analysis-overview>div{min-width:0; display:grid; gap:3px; align-content:center}
  .analysis-overview span, .analysis-overview small{font-size:12px; line-height:1.4; color:var(--muted)}
  .analysis-overview strong{overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:var(--ink); font:700 22px/1.3 var(--mono)}
  .analysis-overview .analyze-actions{display:flex; flex-direction:column; gap:5px; justify-content:center}
  .analyze-actions button{min-height:30px; padding:5px 10px; font-size:13px}
  .analyze-actions #manageEngines, .analyze-actions #manageEngines span{font-size:11px; color:var(--muted)}
  .analyze-actions #stop:disabled, .analyze-actions:has(#stop:not(:disabled)) #analyze{display:none}
  :global(.analysis-overview[data-has-result]) #analyze{background:transparent; border-color:var(--border-input); color:var(--ink)}
  :global(.analysis-overview[data-has-result]) .evaluation-readout small, :global(.analysis-overview[data-has-result]) .best-move-readout small{display:flex; align-items:center; gap:5px; font-size:11px}
  :global(.analysis-overview[data-has-result]) .evaluation-readout small::before, :global(.analysis-overview[data-has-result]) .best-move-readout small::before{content:''; width:6px; height:6px; border-radius:50%; background:var(--engine-color)}
  .app-sidebar #message{
    margin:0 12px 6px; padding:3px 8px; max-height:42px; overflow:auto;
    border:0; border-left:2px solid var(--accent); border-radius:0 var(--radius-sm) var(--radius-sm) 0;
    background:var(--panel-inset); color:var(--text-soft);
    font-size:12px; line-height:1.5;
  }
  .app-sidebar :global(.model-load-progress){margin:6px 12px 0}
  .analyze-controls{display:flex; justify-content:space-between; align-items:center; gap:8px; padding:10px 12px}
  .toggle{
    display:inline-flex; align-items:center; gap:6px; cursor:pointer; user-select:none;
  }
  .toggle input{position:absolute; opacity:0; width:0; height:0}
  .toggle-track{
    width:32px; height:18px; border-radius:99px; flex:0 0 auto;
    background:var(--border-input); border:1px solid var(--border-input);
    transition:background .15s, border-color .15s;
    display:inline-flex; align-items:center; padding:1px;
  }
  .toggle-thumb{
    width:12px; height:12px; border-radius:50%; background:#fff;
    transition:transform .15s; box-shadow:0 1px 2px rgba(0,0,0,.15);
    transform:translateX(0);
  }
  .toggle input:checked + .toggle-track{background:var(--accent); border-color:var(--accent)}
  .toggle input:checked + .toggle-track .toggle-thumb{transform:translateX(14px)}
  .toggle input:focus-visible + .toggle-track{box-shadow:var(--focus-ring)}
  .toggle-label{font-size:12px; color:var(--text-soft); font-weight:500}
  .stepper{
    display:inline-flex; align-items:center; gap:0;
    border:1px solid var(--border-input); border-radius:var(--radius-sm);
    background:var(--panel); overflow:hidden; height:36px;
  }
  .stepper-btn{
    flex:0 0 30px; height:100%; padding:0; border:none; border-radius:0;
    background:var(--panel-inset); color:var(--text-soft); font-size:17px; font-weight:600;
    cursor:pointer; transition:background .12s, color .12s;
    display:flex; align-items:center; justify-content:center;
  }
  .stepper-btn:hover:not(:disabled){background:var(--accent-soft); color:var(--accent-deep)}
  .stepper-field{
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding:0 8px; min-width:44px; border-inline:1px solid var(--border-input);
    height:100%; background:var(--panel);
  }
  .stepper-label{font-size:12px; color:var(--muted); line-height:1; font-weight:500}
  .stepper-field input{
    width:28px; text-align:center; border:none; background:transparent;
    padding:0; font-family:var(--mono); font-size:14px; font-weight:700; color:var(--ink);
    appearance:textfield;
    -moz-appearance:textfield;
  }
  .stepper-field input::-webkit-inner-spin-button,
  .stepper-field input::-webkit-outer-spin-button{ -webkit-appearance:none; margin:0 }
  .stepper-field input:focus{box-shadow:none; outline:none}
  .profile-row{align-items:end}
  :global(.profile-row select){min-width:150px}
  .profile-row input[type=text]{min-width:130px}
  :global(.profile-summary){
    margin-top:6px; padding:7px 8px;
    border:1px dashed var(--rule); border-radius:6px; background:var(--card);
  }
  :global(.legend){display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; font-size:12px; color:var(--muted)}
  :global(.legend .key){display:inline-flex; align-items:center; gap:5px}
  :global(.legend .dot){width:11px; height:11px; border-radius:3px; display:inline-block}
  :global(.movelist){font:14px/1.4 var(--sans); min-height:44px; max-height:clamp(180px, calc(100dvh - 570px), 350px); overflow:auto; background:var(--panel-inset); scroll-padding:8px; overscroll-behavior:contain; scrollbar-gutter:stable; padding:4px 0}
  :global(.move-pair){display:grid; grid-template-columns:44px 1fr 1fr; align-items:center; border-bottom:1px solid var(--rule)}
  :global(.move-number), :global(.missing-ply){text-align:center; color:var(--muted); font-size:12px}
  :global(.history-empty){font:13px/1.5 var(--sans); margin:0; padding:12px; color:var(--muted)}
  :global(.variation){margin:4px 8px 4px 28px; padding-left:8px; border-left:2px solid var(--border-input)}
  :global(.variation-label){font:12px var(--sans); color:var(--muted)}
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
  :global(.movelist .mv){cursor:pointer; min-height:24px; padding:2px 10px; border:0; border-left:2px solid transparent; border-radius:3px; text-align:left; background:transparent; color:var(--ink); font:inherit}
  :global(.movelist .mv:hover){background:var(--accent-soft)}
  :global(.movelist .mv.current){background:color-mix(in srgb, var(--accent) 23%, transparent); border-left-color:var(--accent); color:var(--ink); font-weight:650}
  :global(.movelist .var){color:var(--muted)}
  .human-moves>summary span{
    margin-left:5px; padding:2px 5px; border:1px solid var(--accent-tint-border); border-radius:3px;
    color:var(--accent-deep); font-size:8px; letter-spacing:.06em;
  }
  .add-engine-row{margin-top:3px; align-items:center}
  .add-engine-row button{
    flex:0 0 auto; width:auto; min-height:0; padding:4px 1px; border:0; background:transparent;
    color:var(--ink); font-size:14px; font-weight:650;
  }
  .add-engine-row button:hover{background:transparent; color:var(--ink)}
  :global(.pgn-db-list), :global(.pgn-db-results){display:grid; gap:6px; margin-top:8px}
  :global(.pgn-db-list .empty), :global(.pgn-db-results .empty){
    padding:7px 8px; border:1px dashed var(--rule);
    border-radius:6px; background:var(--card);
  }
  :global(.pgn-db-card){display:grid; grid-template-columns:1fr auto; gap:3px 8px; text-align:left; background:var(--card)}
  :global(.pgn-db-card.selected){border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset}
  :global(.pgn-db-card .name){font-weight:700; color:var(--ink)}
  :global(.pgn-db-card .meta), :global(.pgn-db-hit .meta){color:var(--muted); font-size:11px}
  :global(.pgn-db-hit){padding:7px 8px; border:1px solid var(--rule); border-radius:6px; background:var(--card)}
  :global(.pgn-db-hit .name){font-weight:700; color:var(--ink)}
  :global(.pgn-db-hit .moves){font-family:var(--mono); color:var(--ink); margin-top:3px; overflow-wrap:anywhere}
  :global(table.opening){width:100%; border-collapse:collapse; font-size:13px; margin-top:8px}
  :global(table.opening th), :global(table.opening td){text-align:left; padding:5px 8px; border-bottom:1px solid var(--rule)}
  :global(table.opening th){color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase}
  :global(table.opening td.num), :global(table.opening th.num){text-align:right; font-family:var(--mono)}
  :global(table.opening tr.mv){cursor:pointer}
  :global(table.opening tr.mv:hover td){background:var(--accent-soft)}
  :global(table.opening .san){font-family:var(--mono); font-weight:700}
  :global(.wdlbar){
    display:flex; height:14px; border-radius:3px; overflow:hidden;
    min-width:120px; border:1px solid var(--rule);
  }
  :global(.wdlbar .w){background:#f4f1e8}
  :global(.wdlbar .d){background:#b9b3a4}
  :global(.wdlbar .b){background:#3a3a3a}
  @media (max-width:900px){
    .analysis-workspace{grid-template-columns:1fr; max-width:680px; padding:12px; gap:10px}
    .board-column{display:contents}
    .board-panel, .analysis-workspace .app-sidebar{grid-column:1; grid-row:auto}
    .workbench-tabs{grid-row:1}
    .workbench-tabs .mobile-moves-tab{display:block}
    .workbench-tabs span{font-size:10px}
    .history-panel{grid-row:2; display:none}
    .analysis-workspace .lines-panel, .analysis-workspace .comparison-panel{grid-row:2; max-height:360px}
    :global(.analysis-content[data-view=moves]) .history-panel{display:block}
    :global(.analysis-content[data-view=moves]) .lines-panel{display:none}
    :global(.movelist){max-height:320px}
    :global(.movelist .mv){min-height:36px}
  }
  @media (max-width:600px){
    .analysis-workspace{padding:8px 4px 32px; gap:8px}
    .board-panel{padding:0; border:0; background:transparent}
    .board-wrap{grid-template-columns:8px 1fr; gap:2px}
    .evalbar{width:8px; border-radius:2px}
    .board-panel :global(#ground){border-radius:3px}
    .board-toolbar{flex-wrap:wrap; gap:4px; margin:6px 2px 0}
    .board-eval{flex-basis:44px; font-size:11px}
    .nav{gap:3px}
    .nav button{height:36px; min-width:0; padding:4px}
    .board-actions{flex-direction:row; width:100%; justify-content:space-between; margin:0 2px}
    .board-actions button{min-height:30px}
    .analysis-overview{padding:8px 12px; gap:6px; grid-template-columns:1fr .8fr auto}
    .analysis-overview strong{font-size:22px}
    .analysis-overview small{font-size:11px}
    .analyze-actions button{min-height:32px}
    .app-sidebar #message{margin:0 12px 6px}
    :global(.analysis-workspace #ground coords){font-size:11px}
    :global(.engine-summary){min-height:40px}
    :global(.line-play){min-height:36px; padding-block:7px}
    :global(.pv-expanded summary){height:36px}
    .engine-summaries{padding-bottom:2px}
    .workbench-tabs span{display:none}
    .workbench-tabs button{flex:1; min-height:38px}
    .app-sidebar :global(.model-load-progress){margin-inline:12px}
    .analyze-controls{padding-block:8px}
    .analysis-workspace .section-block{padding:10px 12px}
    :global(.analysis-workspace .engine-row){flex-wrap:wrap}
    :global(.analysis-workspace .engine-row .row-unit){display:block}
    :global(.analysis-workspace .engine-row .row-fam){flex-basis:calc(100% - 40px)}
    .engine-manager{padding:14px}
    :global(table.engine-compare th), :global(table.engine-compare td){padding-inline:3px}
    :global(table.engine-compare tbody th){font-size:12px}
  }
  @media (prefers-reduced-motion: reduce){
    .evalbar .white{transition:none}
    :global(.lines li:hover){background:none}
  }
</style>
