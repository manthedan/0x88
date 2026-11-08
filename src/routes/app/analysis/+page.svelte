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
    <div class="board-readout">
      <button id="evalSourceButton" class="eval-chip" type="button" aria-haspopup="menu" aria-expanded="false" aria-controls="evalSourceMenu" data-switchable="false"><span class="eval-prefix">Eval:</span><span id="evalSource" class="eval-source" aria-hidden="true">Preparing…</span><span id="evalScore" class="eval-score">—</span><svg class="eval-caret" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg></button>
      <span id="evalTurn" class="eval-turn" data-turn="w">White to move</span>
      <div id="evalSourceMenu" class="eval-menu" role="menu" aria-label="Engine shown on the eval bar" hidden></div>
    </div>
    <div class="board-wrap">
      <div class="evalbar" aria-hidden="true"><div class="white" id="evalWhite"></div><div class="mid"></div></div>
      <div class="board-shell">
        <div id="ground"></div>
        <div id="keyboardBoard"></div>
      </div>
      <p id="boardInstructions" class="sr-only">Arrow keys explore squares. Enter or Space selects a piece, then a legal destination. Escape cancels selection. Tab leaves the board.</p>
      <p id="boardAnnouncement" class="sr-only" aria-live="polite"></p>
    </div>
    <div class="board-toolbar">
    <div class="nav" aria-label="Move navigation">
      <button id="navStart" type="button" title="Start (Up)" aria-label="Go to first move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14M18 5l-8 7 8 7" /></svg></button>
      <button id="navBack" type="button" title="Back (Left)" aria-label="Go back one move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-8 7 8 7" /></svg></button>
      <button id="navForward" type="button" title="Forward (Right)" aria-label="Go forward one move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 8 7-8 7" /></svg></button>
      <button id="navEnd" type="button" title="End (Down)" aria-label="Go to last move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M6 5l8 7-8 7" /></svg></button>
    </div>
    <div class="board-tools" aria-label="Board tools">
      <button id="flip" class="tool-button icon-only" type="button" title="Flip board" aria-label="Flip board"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 20V4m-4 4 4-4 4 4m4-4v16m-4-4 4 4 4-4" /></svg></button>
      <button id="quickCopyFen" class="tool-button copy-button" type="button" title="Copy this position's FEN" aria-label="Copy FEN"><svg class="copy-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg><svg class="copied-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg><span class="tool-label">FEN</span></button>
      <button id="quickCopyPgn" class="tool-button copy-button" type="button" title="Copy the game as PGN" aria-label="Copy PGN"><svg class="copy-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg><svg class="copied-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg><span class="tool-label">PGN</span></button>
      <button id="openPosition" class="tool-button position-toggle" type="button" aria-controls="positionTools" aria-expanded="false" title="Load a game or position (PGN or FEN)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m-4-4 4 4 4-4M5 19h14"/></svg><span class="tool-label">Load</span></button>
    </div>
    </div>
    <details id="positionTools" class="position-tools">
      <summary hidden>Position input &amp; export</summary>
      <div class="position-tools-body">
        <div class="row fen-row">
          <div class="field"><label for="fenInput">FEN</label>
            <input id="fenInput" type="text" spellcheck="false" autocomplete="off" placeholder="Paste a FEN" /></div>
          <button id="loadFen" type="button">Load FEN</button>
          <button id="copyFen" type="button">Copy FEN</button>
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
      <div class="position-status">
        <span id="analysisStatus" class="analysis-status" data-state="loading" role="status">Loading…</span>
      </div>
      <div class="analyze-actions">
        <button id="manageEngines" class="manage-engines" type="button" aria-label="Manage engines" title="Manage engines: models, search budget, arrows" aria-haspopup="dialog" aria-controls="engineManager"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/></svg><span class="manage-label">Engines</span><span id="engineCount" class="engine-count">2</span></button>
        <button id="analyze" class="primary run-action" type="button" disabled><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg><span id="analyzeLabel">Analyze</span></button>
        <button id="stop" class="primary run-action stop-action" type="button" disabled><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1"/></svg>Stop<span id="stopProgress" class="stop-progress"></span></button>
      </div>
    </header>
    <div id="message" aria-live="polite">Loading model…</div>
    <div id="runtimeWarning" class="runtime-warning" role="status" aria-live="polite" hidden></div>
    <div id="downloadProgress" class="model-load-progress" hidden></div>
    <div id="analysisSearchProgress" class="search-progress-grid" hidden></div>
    <div id="analysisContent" class="analysis-content" data-view="moves">
    <button id="addEngine" class="add-engine-inline" type="button"><span class="add-engine-plus" aria-hidden="true">+</span><span class="add-engine-text"><strong>Add engine</strong><small id="addEngineHint">more available</small></span></button>
    <div class="workbench-tabs" role="tablist" aria-label="Analysis views">
      <button id="movesTab" type="button" role="tab" aria-selected="true" aria-controls="historyPanel" data-view="moves">Moves</button>
      <button id="enginesTab" type="button" role="tab" aria-selected="false" aria-controls="enginesPanel" tabindex="-1" data-view="engines">Engines<span id="enginesTabCount" class="engine-count">2</span></button>
      <button id="reviewTab" type="button" role="tab" aria-selected="false" aria-controls="reviewPanel" tabindex="-1" data-view="review">Review</button>
      <button id="comparisonTab" type="button" role="tab" aria-selected="false" aria-controls="comparisonSection" tabindex="-1" data-view="comparison">Compare</button>
      <button id="humanTab" type="button" role="tab" aria-selected="false" aria-controls="humanPanel" tabindex="-1" data-view="human" title="What rated humans play here (Maia3)">Human<span class="tab-badge">Maia3</span></button>
    </div>
    <!-- DOM order keeps the mobile active panel after its tab; desktop CSS retains engine-first layout. -->
    <div id="enginesPanel" class="engine-strip" role="tabpanel" aria-labelledby="enginesTab" tabindex="-1">
      <div id="comparisonInsight" class="comparison-insight" aria-label="Engine comparison insight" hidden></div>
      <div id="engineSummaries" class="engine-summaries" aria-label="Engines and their best lines"></div>
      <div id="engineLegend" class="legend" hidden></div>
    </div>
    <div id="historyPanel" class="history-panel" role="tabpanel" aria-labelledby="movesTab" tabindex="0">
      <p id="openingName" class="opening-name" hidden></p>
      <div id="movelist" class="movelist move-panel" aria-label="Move history"></div>
    </div>
    <div id="reviewPanel" class="review-panel" role="tabpanel" aria-labelledby="reviewTab" tabindex="0">
      <div class="review-settings">
        <div class="field"><label for="reviewEngine">Reviewer</label><select id="reviewEngine"><option value="sf:lite">Stockfish Lite</option></select></div>
        <div class="field"><label for="reviewDepth">Depth per position</label><select id="reviewDepth"><option value="8">8 · Quick</option><option value="12" selected>12 · Standard</option><option value="16">16 · Deeper</option></select></div>
      </div>
      <div class="review-actions">
        <button id="reviewGame" class="primary" type="button">Review game</button>
        <button id="reviewStop" type="button" disabled>Cancel review</button>
      </div>
      <p id="reviewStatus" class="review-status" role="status">Load a PGN or play moves, then review the mainline. Choose a CPU reviewer here; add other engines in Engines.</p>
      <progress id="reviewProgress" aria-label="Game review progress" hidden></progress>
      <div id="reviewResults" hidden>
        <p class="review-metric-label">Approximate accuracy</p>
        <div id="reviewSummary" hidden></div>
        <details class="review-explanation"><summary>How this review is measured</summary><p>Accuracy is a simple mean of move scores from estimated winning-chance loss, excluding forced moves. It is not a calibrated result prediction or the same game score as Lichess or Chess.com. Inaccuracies lose at least 5 percentage points, mistakes 10, and blunders 20. Shallow searches can change these labels.</p></details>
        <p class="review-metric-label review-graph-heading">White advantage <span id="reviewGraphSource"></span></p>
        <div id="reviewChart" class="review-chart" role="slider" tabindex="0" aria-label="Game evaluation graph" aria-describedby="reviewGraphHint" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0" hidden></div>
        <p id="reviewSelection" class="review-selection"></p>
        <p id="reviewGraphHint" class="review-hint">Click the graph or use arrow keys to explore. Home and End jump to either end.</p>
        <div class="review-moments-heading"><h2>Key moments</h2><nav aria-label="Key moments"><button id="reviewPrevious" type="button" aria-label="Previous key moment" disabled>Previous</button><button id="reviewNext" type="button" aria-label="Next key moment" disabled>Next</button></nav></div>
        <p id="reviewMomentsEmpty" class="review-hint" hidden>No inaccuracies, mistakes or blunders found at this depth.</p>
        <ol id="reviewCritical" class="review-critical" aria-label="Largest losses, ranked by size" hidden></ol>
        <div class="review-exports"><button id="reviewCopyPgn" type="button" hidden>Copy reviewed PGN</button><button id="reviewDownloadPgn" type="button" hidden>Download PGN</button></div>
        <p class="review-hint">Exports the reviewed mainline with engine evals and error NAGs. Variations and existing notes are not included.</p>
      </div>
    </div>
    <div id="comparisonSection" class="section-block comparison-panel" role="tabpanel" aria-labelledby="comparisonTab" tabindex="0">
      <h2 class="sr-only">Comparison</h2>
      <div id="engineConsensus" class="compare-summary">No analysis yet.</div>
      <p class="compare-help">Agreement means the same recommended move. Eval spread is the gap between highest and lowest evaluations, in pawns from White’s perspective. A ✓ marks the most-supported move.</p>
      <table id="engineCompare" class="engine-compare"><thead><tr><th scope="col">Engine</th><th scope="col">Best</th><th scope="col">White’s eval</th><th scope="col" id="deltaBaseline">Δ</th></tr></thead><tbody></tbody></table>
    </div>
    <div id="humanPanel" class="section-block human-panel" role="tabpanel" aria-labelledby="humanTab" tabindex="0">
      <h2 class="sr-only">Human moves (Maia3)</h2>
      <p class="human-intro">What players at each rating actually choose here, from the Maia3 human-move model. Runs locally after a one-time download.</p>
      <div class="row">
        <button id="maia3Enable" type="button">Load Maia3 (~28MB, cached)</button>
        <span id="maia3PanelStatus" class="small"></span>
      </div>
      <div id="maia3Grid" class="maia3-grid" hidden></div>
      <div id="maia3Caption" class="small" hidden>What rated humans actually play here (Maia3 move predictions; score = expected points for White in a human game at that rating, not an engine eval). ✓ = matches the current best engine move.</div>
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
      <label class="arrow-preference" for="showAllLineArrows"><input id="showAllLineArrows" type="checkbox" /> <span>Show arrows for all lines<small>Solid = best move (#1); faint = alternatives (#2+). Colours match engine rows. Hover or focus a line to isolate its arrow.</small></span></label>
      <div id="engineList" class="engine-list">
        <div class="engine-list-loading">Preparing engine controls…</div>
      </div>
      <div class="row add-engine-row">
        <button id="addEngineInManager" type="button">+ Add engine</button>
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
      <div class="capabilities-slot"><BrowserCapabilities /></div>
      </div>
    </dialog>
    <!-- Parked: features below are intentionally hidden. Remove hidden to re-enable. -->
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
    --engine-color:var(--accent); /* overridden inline per engine row */
    --analysis-header-height:64px; /* measured locally by the page controller */
    --selection-bg:color-mix(in srgb, var(--ink) 12%, var(--panel-inset));
    --result-highlight:color-mix(in srgb, var(--ink) 13%, var(--panel));
    --board-size:min(704px, calc(100dvh - 200px), calc(100vw - 514px));
    grid-template-columns:calc(var(--board-size) + 34px) minmax(0, 470px); gap:16px; align-items:start;
    max-width:1288px; padding:16px 24px 32px; font-size:14px; font-variant-numeric:tabular-nums;
  }
  .board-column{min-width:0}
  .board-panel{min-width:0; padding:8px}
  .board-wrap{display:grid; grid-template-columns:18px 1fr; gap:0; position:relative}
  .board-toolbar{display:flex; align-items:stretch; gap:12px; margin:8px 0 0 18px; height:40px}
  /* Desktop: the panel shares the board column's baseline (board + toolbar), so there is no dead block under it. */
  .analysis-workspace .app-sidebar{position:static; display:flex; flex-direction:column; min-width:0; height:calc(var(--board-size) + 98px); max-height:none; overflow:hidden}
  .analysis-workspace .app-sidebar>*{flex:0 0 auto}
  /* Engine blocks shrink (and scroll) before the move list drops below ~5 rows. */
  .analysis-workspace .engine-strip{order:0; flex:0 1 auto; min-height:0}
  :global(.analysis-workspace .sr-only){position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip-path:inset(50%); white-space:nowrap; border:0}
  .board-shell{position:relative; min-width:0}
  :global(.analysis-workspace #ground coords){font-size:12px; font-weight:700; opacity:1}
  :global(.analysis-workspace #ground .cg-shapes){filter:drop-shadow(0 0 1px #fff9)}
  :global(.analysis-workspace #keyboardBoard){position:absolute; inset:0; z-index:5; pointer-events:none}
  :global(.analysis-workspace .keyboard-row){display:contents}
  :global(.analysis-workspace .keyboard-square){position:absolute; width:12.5%; height:12.5%; padding:0; min-height:0; border:0; border-radius:0; background:transparent; pointer-events:none}
  :global(.analysis-workspace .keyboard-square:focus-visible){outline:3px solid #fff; outline-offset:-4px; box-shadow:inset 0 0 0 6px #225ca0; background:#fff2}
  .analysis-workspace .section-block>summary{font:650 13px/1.3 var(--sans); letter-spacing:0; text-transform:none; color:var(--ink)}
  .analysis-workspace .section-block>summary::before{font-size:12px}
  /* Secondary, right-aligned tool group: flip, copy FEN/PGN, load. Nav keeps the wide primary buttons. */
  .board-tools{display:flex; align-items:stretch; gap:6px; flex:0 0 auto}
  .tool-button{display:inline-flex; align-items:center; justify-content:center; gap:6px; height:100%; min-height:0; padding:0 11px; border-radius:7px; border:1px solid var(--border-input); background:transparent; color:var(--ink); font:600 13px var(--sans); white-space:nowrap}
  .tool-button:hover:not(:disabled){background:var(--panel-2); color:var(--ink)}
  .tool-button.icon-only{width:40px; padding:0}
  .copy-button{min-width:72px}
  .copy-button .copied-icon{display:none}
  .copy-button:global([data-copied=true]){border-color:var(--accent); color:var(--accent-soft-text)}
  .copy-button:global([data-copied=true]) .copy-icon{display:none}
  .copy-button:global([data-copied=true]) .copied-icon{display:block}
  /* Copied: check icon + accent outline in place (no width change); the live message line says what was copied. */
  .position-toggle:global([aria-expanded=true]){border-color:var(--accent)}
  .tool-button svg, .analyze-actions svg{width:16px; height:16px; flex:0 0 16px; fill:none; stroke:currentColor; stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round}
  .nav svg{width:20px; height:20px; fill:none; stroke:currentColor; stroke-width:2.3; stroke-linecap:round; stroke-linejoin:round}
  .analysis-workspace :global(button:focus-visible), .analysis-workspace :global(summary:focus-visible), .analysis-workspace :global(select:focus-visible), .analysis-workspace :global(input:focus-visible){outline:2px solid var(--accent); outline-offset:2px}
  .analysis-workspace .app-sidebar>.analysis-content{display:flex; flex-direction:column; flex:1 1 auto; min-width:0; min-height:0}
  .engine-manager{width:min(600px, calc(100vw - 24px)); max-height:85dvh; padding:20px; color:var(--ink); background:var(--panel); border:1px solid var(--border-input); border-radius:12px; box-shadow:0 20px 80px #0008}
  .engine-manager::backdrop{background:#0009}
  .manager-heading{display:flex; align-items:center; justify-content:space-between; gap:12px}
  .manager-heading h2{font:650 20px var(--sans); color:var(--ink); text-transform:none; letter-spacing:0; margin:0}
  .engine-setup>p{margin:12px 0}
  .engine-setup>p:first-child{font-weight:650}
  .capabilities-slot{margin-top:16px}
  .capabilities-slot :global(.capabilities){margin:0}
  .arrow-preference{display:flex; align-items:flex-start; gap:8px; margin:0 0 16px; padding:12px; border:1px solid var(--rule); border-radius:8px; color:var(--ink); font-size:14px}
  .arrow-preference input{margin-top:4px; accent-color:var(--accent)}
  .arrow-preference small{display:block; margin-top:3px; font-size:13px; color:var(--muted)}
  :global(.engine-logo-frame){display:inline-grid; place-items:center; width:40px; height:28px; flex:0 0 40px; padding:2px; box-sizing:border-box; border-radius:5px; background:#ede8dd; box-shadow:inset 0 0 0 1px #0000001f; overflow:hidden}
  :global(.engine-logo-frame.wordmark){padding:3px; background:#fff}
  :global(.engine-logo-frame .engine-logo), :global(.engine-logo-frame .engine-logo-placeholder){display:block; width:100%; height:100%; border-radius:3px; object-fit:contain}
  :global(.engine-logo-initial){font:700 13px/1 var(--sans); color:#3b352b}
  :global(.analysis-workspace .engine-row){align-items:center}
  :global(.analysis-workspace .engine-row select), :global(.analysis-workspace .engine-row .row-strength){font-size:14px}
  :global(.analysis-workspace .engine-row .row-rm){min-height:36px}
  :global(.analysis-workspace .engine-row .row-unit){font-size:12px}
  :global(.analysis-workspace .engine-row .row-strength){flex-basis:76px; width:76px}
  .evalbar{
    --eval-pct:50%;
    width:18px; border:1px solid #8a8780; border-right:0; border-radius:4px 0 0 4px;
    overflow:hidden; background:#555550; position:relative; flex:0 0 auto;
    box-sizing:border-box;
  }
  .evalbar .white{
    position:absolute; left:0; right:0; bottom:0; height:var(--eval-pct);
    background:var(--eval-white); transition:height .25s, width .25s;
  }
  .board-readout{position:relative; display:flex; align-items:center; justify-content:space-between; gap:8px; min-height:34px; padding:0 0 6px}
  .eval-turn{display:inline-flex; align-items:center; gap:6px; color:var(--muted); font:500 13px var(--sans); white-space:nowrap}
  .eval-turn::before{content:''; width:9px; height:9px; border:1px solid #999; border-radius:50%; background:#f4efe4}
  .eval-turn:global([data-turn=b])::before{background:#252522}
  /* Readable source switcher outside the fill; the bar itself stays flush to the board. */
  .eval-chip{display:inline-flex; align-items:center; gap:7px; min-height:28px; padding:0 8px; border:1px solid var(--border-input); border-radius:6px; background:var(--panel-inset); color:var(--ink); font:600 13px var(--sans)}
  .eval-chip:hover:not(:disabled){background:var(--panel-2); color:var(--ink); border-color:var(--accent)}
  .eval-prefix{color:var(--muted); font-weight:400}
  .eval-caret{width:12px; height:12px; fill:none; stroke:currentColor; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round}
  .eval-menu{position:absolute; left:0; top:100%; z-index:20; width:min(280px, 100%); padding:4px; border:1px solid var(--border-input); border-radius:9px; background:var(--panel); box-shadow:0 14px 40px #0008}
  .eval-menu[hidden]{display:none}
  :global(.eval-menu-title){margin:4px 8px 4px; font:600 13px var(--sans); color:var(--muted)}
  :global(.eval-menu-item){display:grid; grid-template-columns:22px minmax(0,1fr) auto; align-items:center; gap:9px; width:100%; min-height:38px; padding:0 9px; border:0; border-radius:6px; background:transparent; color:var(--ink); text-align:left; font:600 14px var(--sans)}
  :global(.eval-menu-item:hover:not(:disabled)), :global(.eval-menu-item:focus-visible){background:var(--panel-2); color:var(--ink)}
  :global(.eval-menu-item[aria-checked=true]){background:var(--result-highlight); }
  :global(.eval-menu-item:disabled){opacity:.5}
  :global(.eval-menu-name){overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  :global(.eval-menu-score){font:700 14px var(--mono); font-variant-numeric:tabular-nums}
  .eval-source{display:block; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font:600 13px/22px var(--sans)}
  .eval-source:global([data-short]:not([data-short='']))::before{content:attr(data-short); display:inline-grid; vertical-align:middle; place-items:center; width:22px; height:22px; margin-right:6px; border-radius:4px; background:var(--engine-color); color:#fff; font:700 11px var(--mono)}
  .eval-score{color:var(--ink); font:750 14px/1 var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap}
  /* Two-column layout: the engine cards beside the board print this value; the button's label still carries it. */
  @media (min-width:901px){ .eval-score{display:none} }
  .evalbar .mid{
    position:absolute; left:0; right:0; top:50%;
    height:2px; background:#99978e; box-shadow:0 1px 0 #0006;
  }
  .nav{display:flex; flex:1 1 auto; gap:6px; min-width:0}
  .nav button{
    flex:1 1 0; height:100%; min-height:0; min-width:0; padding:0; border-radius:7px;
    border:1px solid var(--border-input-hover); background:var(--panel-2);
    color:var(--ink); font-size:14px; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
  }
  .nav button:hover:not(:disabled){border-color:var(--border-input-hover); background:var(--card); color:var(--ink)}
  /* Same frame in every state; disabled only dims the glyph. */
  .nav button:disabled{opacity:1; cursor:default; color:var(--ink)}
  .nav button:disabled svg{opacity:.3}
  .row{margin-top:10px}
  .engine-list{margin-top:0}
  .engine-list-loading{
    padding:8px 0; color:var(--muted); font-size:12px;
  }
  :global(.runtime-warning){
    margin-top:7px; padding:7px 9px; border:1px solid color-mix(in srgb, var(--warn) 45%, var(--rule));
    border-radius:6px; background:var(--warn-soft); color:var(--warn); font-size:12px; line-height:1.4;
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
  :global(.compare-summary){
    margin-top:0; padding:8px 10px; border:0;
    border-radius:var(--radius-sm);
    background:var(--accent-soft); font-size:14px; line-height:1.5; color:var(--ink);
  }
  :global(table.engine-compare){width:100%; border-collapse:collapse; font-size:14px; margin-top:8px}
  :global(table.engine-compare th), :global(table.engine-compare td){
    text-align:left; padding:6px 5px;
    border-bottom:1px solid var(--rule); vertical-align:middle;
  }
  :global(table.engine-compare thead th){color:var(--muted); font-weight:500; font-size:12px}
  :global(table.engine-compare tbody th){color:var(--ink); font-weight:550; font-size:13px}
  :global(.compare-move){font:650 14px var(--sans); padding:6px; min-height:36px; color:var(--ink); background:var(--panel-inset)}
  :global(table.engine-compare .mono){font-family:var(--mono); white-space:nowrap}
  :global(table.engine-compare tbody tr:hover){background:var(--accent-soft)}
  :global(.model-load-progress), :global(.search-progress-grid){
    margin-top:8px; padding:8px; border:1px solid var(--rule);
    border-radius:var(--radius-sm); background:var(--panel-inset);
  }
  :global(.model-load-progress progress), :global(.search-progress-grid progress){
    width:100%; height:9px; accent-color:var(--accent);
  }
  :global(.loading-progress-row), :global(.search-progress-row){display:grid; gap:3px; margin:4px 0}
  :global(.dl-label), :global(.search-progress-text){font-size:12px; font-variant-numeric:tabular-nums; color:var(--muted)}
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
  .app-sidebar #message{
    margin:6px 12px; padding:4px 8px; max-height:44px; overflow:auto;
    border:0; border-radius:var(--radius-sm);
    background:var(--panel-inset); color:var(--text-soft);
    font-size:12px; line-height:1.5;
  }
  .app-sidebar :global(#message[data-routine=true]){position:absolute; width:1px; height:1px; margin:-1px; padding:0; overflow:hidden; clip-path:inset(50%)}
  .app-sidebar :global(.model-load-progress){margin:6px 12px 0}
  /* Per-engine progress lives in the summary rows (meter + stats) and on the Stop button. */
  .app-sidebar #analysisSearchProgress{display:none}
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
  .profile-row input[type=text]{min-width:130px}
  :global(.profile-summary){
    margin-top:6px; padding:7px 8px;
    border:1px dashed var(--rule); border-radius:6px; background:var(--card);
  }
  /* #engineLegend: shown when opening-book stats exist (not covered by the capture harness). */
  :global(.legend){display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; font-size:12px; color:var(--muted)}
  :global(.legend .key){display:inline-flex; align-items:center; gap:5px}
  :global(.legend .dot){width:11px; height:11px; border-radius:3px; display:inline-block}
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
  :global(.pgn-db-card .meta), :global(.pgn-db-hit .meta){color:var(--muted); font-size:12px}
  :global(.pgn-db-hit){padding:7px 8px; border:1px solid var(--rule); border-radius:6px; background:var(--card)}
  :global(.pgn-db-hit .name){font-weight:700; color:var(--ink)}
  :global(.pgn-db-hit .moves){font-family:var(--mono); color:var(--ink); margin-top:3px; overflow-wrap:anywhere}
  :global(table.opening){width:100%; border-collapse:collapse; font-size:13px; margin-top:8px}
  :global(table.opening th), :global(table.opening td){text-align:left; padding:5px 8px; border-bottom:1px solid var(--rule)}
  :global(table.opening th){color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase}
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
  /* ── Right column, Lichess-shaped: header → engine strip → top lines → tabs → move list (fills the rest). ── */
  .analysis-overview{display:flex; align-items:center; justify-content:space-between; gap:12px; min-height:52px; padding:6px 10px 6px 14px; background:var(--panel-inset); border-bottom:1px solid var(--rule)}
  .position-status{display:flex; align-items:center; gap:10px; min-width:0}
  /* Quiet, neutral run state: text + glyph, no pill competing with the primary action. */
  .analysis-status{display:inline-flex; align-items:center; gap:5px; color:var(--muted); font:500 12.5px/1 var(--sans); white-space:nowrap}
  .analysis-status::before{content:''; width:6px; height:6px; flex:0 0 auto; border-radius:50%; background:currentColor; opacity:.7}
  :global(.analysis-status[data-state=running]){color:var(--ink)}
  :global(.analysis-status[data-state=running])::before{width:8px; height:8px; border:2px solid color-mix(in srgb, var(--accent) 30%, transparent); border-top-color:var(--accent); background:transparent; opacity:1; animation:analysis-spin .8s linear infinite}
  :global(.analysis-status[data-state=done])::before{content:'✓'; width:auto; height:auto; background:none; opacity:1; font-size:12px; font-weight:800}
  @keyframes -global-analysis-spin{to{transform:rotate(360deg)}}
  .analyze-actions{display:flex; align-items:stretch; gap:8px; flex:0 0 auto}
  .analyze-actions button{display:inline-flex; align-items:center; justify-content:center; gap:7px; min-height:38px; padding:0 13px; border-radius:8px; font-size:14px; white-space:nowrap}
  .manage-engines{background:transparent; color:var(--ink); border-color:var(--border-input)}
  .engine-count{display:inline-grid; place-items:center; min-width:20px; height:20px; padding:0 5px; border-radius:10px; background:var(--panel-2); font:700 12px/1 var(--sans); font-variant-numeric:tabular-nums}
  .manage-engines:hover:not(:disabled){background:var(--panel); color:var(--ink)}
  /* Analyze / Stop is the one filled primary in every state. */
  .run-action{min-width:108px; font-weight:650}
  .run-action svg{fill:currentColor; stroke:none}
  #analyze:disabled{opacity:.55}
  /* Done: the primary action steps down to a quiet outlined "Re-analyze"; the ✓ Done label carries the state. */
  .analysis-overview:global([data-status=done]) #analyze{background:transparent; color:var(--ink); border:1px solid var(--border-input); font-weight:600}
  .analysis-overview:global([data-status=done]) #analyze:hover:not(:disabled){background:var(--panel-2); border-color:var(--accent)}
  .analysis-overview:global([data-status=done]) #analyze svg{display:none}
  .stop-action{position:relative; overflow:hidden; isolation:isolate; --progress:0%}
  .stop-action::before{content:''; position:absolute; inset:0 auto 0 0; width:var(--progress); background:#0000002e; z-index:-1; transition:width .2s}
  .stop-progress{font:600 12px var(--sans); font-variant-numeric:tabular-nums; opacity:.85}
  .stop-progress:empty{display:none}
  .analyze-actions #stop:disabled, .analyze-actions:has(#stop:not(:disabled)) #analyze{display:none}

  /* Engine strip */
  .engine-strip{display:flex; flex-direction:column; min-width:0}
  .engine-summaries{flex:0 1 auto; min-height:0; overflow:auto; overscroll-behavior:contain; scrollbar-width:thin}
  .analysis-workspace .analysis-content>:is(.history-panel, .comparison-panel, .human-panel, .review-panel){min-height:208px}
  :global(.comparison-insight){border-bottom:1px solid var(--rule)}
  :global(.comparison-insight[hidden]){display:none}
  :global(.insight-action){display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:2px 12px; width:100%; min-height:48px; padding:6px 12px; border:0; border-radius:0; background:transparent; color:var(--ink); text-align:left; line-height:1.3}
  :global(.insight-action:hover:not(:disabled)){background:var(--result-highlight)}
  :global(.insight-headline){display:flex; align-items:center; gap:7px; font:650 14px/1.3 var(--sans)}
  :global(.insight-headline strong){font-weight:650}
  :global(.insight-groups){display:flex; flex-wrap:wrap; align-items:center; gap:6px 12px; width:100%; margin-top:3px}
  :global(.insight-group){display:inline-flex; align-items:center; gap:4px; font:600 13px var(--sans)}
  :global(.insight-group .engine-avatar){width:22px; height:20px; flex-basis:22px; font-size:11px}
  :global(.insight-spread){font:13px/1.3 var(--sans); font-variant-numeric:tabular-nums; color:var(--muted); white-space:nowrap}
  :global(.insight-spread[data-level=apart]){color:var(--ink)}
  :global(.insight-spread[data-level=wide]){color:var(--warn); font-weight:650}
  /* One block per engine: header row, then its principal variation indented under the name. */
  :global(.engine-card){position:relative; border-bottom:1px solid var(--rule)}
  :global(.engine-card.arrow-focus){background:var(--result-highlight)}
  :global(.engine-card.open){background:color-mix(in srgb, var(--engine-color) 6%, var(--panel))}
  :global(.engine-summary){display:grid; grid-template-columns:minmax(0,1fr) auto 64px 30px; align-items:center; gap:6px; min-height:34px; padding:3px 4px 0 9px}
  :global(.engine-summary-name){display:flex; align-items:center; gap:8px; padding:0; min-width:0; min-height:32px; border:0; border-radius:0; background:transparent; text-align:left; color:var(--ink)}
  :global(.engine-summary-name:hover:not(:disabled)){background:transparent; color:var(--ink)}
  :global(.engine-summary-name:hover strong){text-decoration:underline; text-underline-offset:3px; text-decoration-color:var(--muted)}
  :global(.engine-summary-text){display:flex; align-items:baseline; gap:8px; min-width:0}
  :global(.engine-summary-name strong){font:650 13.5px/1.2 var(--sans); white-space:nowrap}
  :global(.engine-summary-name small){font:13px/1.2 var(--sans); font-variant-numeric:tabular-nums; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0}
  :global(.engine-summary-score){display:inline-flex; align-items:center; justify-content:flex-end; gap:6px; min-width:62px; font:700 17px/1 var(--mono); font-variant-numeric:tabular-nums; letter-spacing:-.02em; text-align:right; color:var(--ink)}
  :global(.engine-summary .compare-move){height:28px; min-height:0; padding:0 4px; border:0; border-radius:4px; background:color-mix(in srgb, var(--engine-color) 15%, var(--panel)); font:750 15px var(--sans); text-align:center; color:var(--ink)}
  :global(.engine-summary .compare-move:hover){background:color-mix(in srgb, var(--engine-color) 28%, var(--panel)); text-decoration:underline; text-underline-offset:3px}
  :global(.agreement-mark){margin-right:4px; font-size:12px}
  :global(.compare-move-empty){text-align:center; color:var(--muted)}
  :global(.engine-gear){display:grid; place-items:center; width:30px; height:30px; min-height:0; padding:0; border:0; border-radius:6px; background:transparent}
  :global(.engine-gear:hover:not(:disabled)){background:var(--panel-2)}
  :global(.engine-settings-icon){width:16px; height:16px; fill:none; stroke:var(--muted); stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round}
  :global(.engine-gear:hover .engine-settings-icon), :global(.engine-gear:focus-visible .engine-settings-icon){stroke:var(--ink)}
  :global(.engine-summary-meter){position:absolute; left:0; bottom:-1px; height:2px; width:var(--progress); background:var(--engine-color); transition:width .2s}
  :global(.engine-summary-meter.indeterminate){width:100%; opacity:.55; animation:analysis-pulse 1.4s ease-in-out infinite}
  @keyframes -global-analysis-pulse{50%{opacity:.15}}
  :global(.engine-avatar){display:inline-grid; place-items:center; width:22px; height:22px; flex:0 0 22px; border-radius:5px; background:var(--engine-color); color:#fff; font:700 10.5px/1 var(--mono); letter-spacing:0; box-shadow:inset 0 0 0 1px #ffffff26}
  /* A real outlined button: multi-engine comparison is the product's differentiator. */
  .add-engine-inline{order:1; flex:0 0 auto; display:flex; align-items:center; justify-content:center; gap:8px; width:auto; min-height:36px; margin:8px 10px; padding:0 12px; border:1px solid var(--border-input); border-radius:8px; background:transparent; color:var(--ink); text-align:left}
  .add-engine-inline:hover:not(:disabled){border-color:var(--accent); background:var(--accent-soft); color:var(--ink)}
  .add-engine-plus{display:inline-grid; place-items:center; width:18px; height:18px; flex:0 0 18px; color:var(--accent); font:700 19px/1 var(--sans)}
  .add-engine-text{display:flex; align-items:baseline; gap:6px; min-width:0}
  .add-engine-text strong{font:650 14px/1.3 var(--sans); white-space:nowrap}
  .add-engine-text small{font:13px/1.35 var(--sans); color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .add-engine-text small::before{content:'· '}

  /* PV rows inside each engine block; the top line hangs under the engine name (39px = pad + avatar + gap). */
  :global(.engine-lines){list-style:none; margin:0; padding:0 0 5px}
  :global(.result-line){position:relative; display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:center; min-height:26px}
  :global(.result-line.alt-line){border-top:1px dashed var(--rule-soft)}
  :global(.result-line:has(.line-play:hover)), :global(.result-line:focus-within), :global(.result-line.previewing){background:var(--result-highlight)}
  :global(.line-play){display:flex; align-items:center; gap:8px; width:100%; min-width:0; min-height:26px; padding:0 6px 0 39px; border:0; border-radius:0; background:transparent; color:var(--ink); text-align:left}
  :global(.line-play:hover:not(:disabled)){background:transparent; color:var(--ink)}
  :global(.line-rank){flex:0 0 12px; font:600 13px var(--mono); color:var(--muted); text-align:center}
  :global(.engine-lines .score){flex:0 0 50px; font:700 13.5px var(--mono); font-variant-numeric:tabular-nums; color:var(--ink); text-align:right}
  :global(.engine-lines .pv){flex:1 1 auto; min-width:0; display:flex; flex-wrap:wrap; column-gap:.42em; height:1.5em; overflow:hidden; font:14px/1.5 var(--sans); color:var(--ink)}
  :global(.engine-lines .pv-move){white-space:nowrap}
  :global(.engine-lines .pv-move:first-child){font-weight:700}
  :global(.engine-lines .pv-num){color:var(--muted); font-weight:400; font-variant-numeric:tabular-nums}
  :global(.line-toggle){display:inline-flex; align-items:center; justify-content:flex-end; gap:2px; min-width:76px; min-height:26px; padding:0 8px 0 6px; border:0; border-radius:5px; background:transparent; color:var(--muted); font:600 13px var(--sans); font-variant-numeric:tabular-nums; white-space:nowrap}
  :global(.line-toggle:hover:not(:disabled)){background:var(--panel-2); color:var(--ink)}
  :global(.line-toggle svg){width:15px; height:15px; fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; transition:transform .15s}
  :global(.line-toggle[aria-expanded=true] svg){transform:rotate(180deg)}
  :global(.engine-lines .line-error), :global(.engine-lines .pending-line){display:block; min-height:24px; padding:2px 10px 0 39px; font-size:13px}
  :global(.line-error-text){display:block; color:var(--warn); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  :global(.pending-text){color:var(--muted)}
  :global(.legend){display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; padding:6px 12px 0; font-size:13px; color:var(--muted)}
  :global(.legend[hidden]){display:none}
  :global(.legend .key){display:inline-flex; align-items:center; gap:5px}
  :global(.legend .dot){width:11px; height:11px; border-radius:3px; display:inline-block}

  /* Tabs: Moves is the default; the lines block above is always on screen at desktop widths. */
  .workbench-tabs{order:2; flex:0 0 auto; display:flex; align-items:center; gap:2px; min-height:38px; padding:0 6px; border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); background:var(--panel-inset)}
  .workbench-tabs [role=tab]{display:inline-flex; align-items:center; gap:5px; min-height:37px; padding:0 9px; border:0; border-radius:0; border-bottom:2px solid transparent; background:transparent; color:var(--muted); font-size:13px; font-weight:550; white-space:nowrap}
  .workbench-tabs [role=tab]:hover:not(:disabled){background:transparent; color:var(--ink)}
  .workbench-tabs [role=tab][aria-selected=true]{color:var(--ink); border-bottom-color:var(--accent); font-weight:650}
  #enginesTab{display:none}
  .tab-badge{padding:1px 4px; border:1px solid var(--accent-tint-border); border-radius:3px; color:var(--accent-soft-text); font-size:11px; font-weight:600}
  .history-panel, .analysis-workspace .comparison-panel, .analysis-workspace .human-panel, .analysis-workspace .review-panel{order:3; flex:1 1 0; min-height:0}
  .history-panel{display:none; flex-direction:column; min-width:0; padding:0; overflow:hidden}
  :global(.analysis-content[data-view=moves]) .history-panel{display:flex}
  .analysis-workspace .comparison-panel{border:0; padding:8px 12px; overflow:auto; overscroll-behavior:contain}
  .analysis-workspace .human-panel{border:0; padding:4px 12px 10px; overflow:auto; overscroll-behavior:contain}
  :global(.analysis-content:not([data-view=comparison])) .comparison-panel, :global(.analysis-content:not([data-view=human])) .human-panel{display:none}
  .review-panel{display:none; min-width:0; padding:12px; overflow:auto; overscroll-behavior:contain; scrollbar-width:thin}
  :global(.analysis-content[data-view=review]) .review-panel{display:block}
  /* Review gets the full sidebar below the tabs, rather than competing with PV cards. */
  :global(.analysis-content[data-view=review]) .engine-strip, :global(.analysis-content[data-view=review]) .add-engine-inline{display:none}
  .review-settings{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:10px}
  .review-settings select{width:100%; min-width:0; font-size:13px; min-height:36px; padding:4px 6px}
  .review-settings label{font-size:13px; color:var(--muted); margin-bottom:4px; display:block}
  .review-actions, .review-exports{display:flex; flex-wrap:wrap; gap:8px; margin-top:10px}
  .review-actions button, .review-exports button{min-height:36px; padding:5px 12px; font-size:13px}
  #reviewStop:disabled{display:none}
  .review-actions:has(#reviewStop:not(:disabled)) #reviewGame{display:none}
  .review-status{font:13px/1.5 var(--sans); color:var(--muted); margin:10px 0}
  #reviewProgress{width:100%; height:6px; accent-color:var(--accent)}
  .review-metric-label{font:600 13px var(--sans); margin:14px 0 8px}
  :global(.review-summary){display:grid; grid-template-columns:1fr 1fr; gap:12px; padding-bottom:10px; border-bottom:1px solid var(--rule)}
  :global(.review-side){display:grid; align-content:start; gap:4px}
  :global(.review-side strong){font:600 13px var(--sans)}
  :global(.review-accuracy){font:650 24px/1.15 var(--sans); letter-spacing:-.02em; color:var(--ink)}
  :global(.review-counts){font:12px/1.6 var(--sans); color:var(--muted)}
  .review-graph-heading{display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 10px}
  #reviewGraphSource{font:400 12px var(--sans); color:var(--muted)}
  .review-explanation{margin-top:8px; font:12px/1.5 var(--sans); color:var(--muted)}
  .review-explanation summary{cursor:pointer}
  .review-explanation p{margin:6px 0}
  .review-chart{width:100%; min-width:0; cursor:crosshair; border-radius:4px; background:var(--panel-inset)}
  .review-chart:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  .review-chart :global(svg){display:block; width:100%; height:auto}
  .review-selection{min-height:2.8em; margin:6px 0 2px; font:600 12px/1.4 var(--sans)}
  .review-hint{margin:4px 0 8px; color:var(--muted); font:12px/1.5 var(--sans)}
  .review-moments-heading{display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:14px}
  .review-moments-heading h2{font:600 14px var(--sans); margin:0; text-transform:none; letter-spacing:0}
  .review-moments-heading nav{display:flex; gap:4px}
  .review-moments-heading button{font-size:12px; padding:4px 8px; min-height:32px}
  :global(.review-critical){padding:0; margin:8px 0; list-style:none}
  :global(.review-critical button){display:flex; align-items:center; gap:8px; width:100%; min-height:38px; padding:6px 8px; border:0; border-bottom:1px solid var(--rule); border-radius:0; background:transparent; text-align:left; font:13px var(--sans); color:var(--ink)}
  :global(.review-critical button:hover), :global(.review-critical button[aria-current=step]){background:var(--selection-bg)}
  :global(.review-critical strong){font-weight:650; margin-right:auto}
  :global(.review-loss){font-size:12px; color:var(--muted); white-space:nowrap}
  :global(.review-badge){font-size:12px; font-weight:600}
  :global(.mv-class){font:700 12px var(--sans); margin-left:3px}
  :global(.mv-class.blunder), :global(.review-badge.blunder){color:var(--warn)}
  :global(.mv-class.mistake), :global(.review-badge.mistake){color:var(--warn)}
  :global(.mv-class.inaccuracy), :global(.review-badge.inaccuracy){color:var(--accent-soft-text)}
  .compare-help{margin:0 0 12px; font:13px/1.5 var(--sans); color:var(--muted)}
  .human-intro{margin:6px 0 0; font-size:13px; line-height:1.45; color:var(--text-soft)}

  .opening-name{flex:0 0 auto; margin:0; padding:7px 12px; border-bottom:1px solid var(--rule); color:var(--muted); font:13px/1.4 var(--sans)}
  .opening-name[hidden]{display:none}
  .opening-name :global(strong){margin-right:6px; color:var(--ink); font:600 12px var(--mono)}
  /* Move list: compact SAN columns; remaining width is quiet, not a gap between moves. */
  .history-panel :global(.movelist){flex:1 1 auto; min-height:0; max-height:none}
  :global(.movelist){font:14px/1.4 var(--sans); overflow:auto; scroll-snap-type:y proximity; background:var(--panel); scroll-padding:8px; overscroll-behavior:contain; scrollbar-width:thin; padding:0}
  :global(.move-pair){display:grid; grid-template-columns:36px minmax(90px,110px) minmax(90px,110px); align-items:stretch; scroll-snap-align:start}
  :global(.move-pair:nth-child(even) .mv){background:color-mix(in srgb, var(--ink) 3%, var(--panel))}
  :global(.move-number), :global(.missing-ply){display:grid; place-items:center; color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; background:var(--panel-inset); border-right:1px solid var(--rule-soft)}
  :global(.missing-ply){background:transparent; border:0}
  :global(.history-empty){display:grid; gap:10px; margin:0; padding:18px 16px; font:13.5px/1.55 var(--sans); color:var(--muted)}
  :global(.history-empty p){margin:0}
  :global(.history-empty strong){color:var(--ink); font-weight:650}
  :global(.quick-pgn){width:100%; min-height:72px; resize:vertical; padding:8px 10px; border:1px solid var(--border-input); border-radius:7px; background:var(--panel-inset); color:var(--ink); font:13px/1.45 var(--mono)}
  :global(.quick-pgn::placeholder){font-family:var(--sans); color:var(--muted)}
  :global(.quick-pgn-actions){display:flex; align-items:center; gap:14px}
  :global(.quick-pgn-hint){font-size:13px; color:var(--muted)}
  :global(.quick-pgn-load){min-height:32px; padding:0 14px; border:1px solid var(--border-input-hover); border-radius:7px; background:var(--panel-2); color:var(--ink); font:600 13px var(--sans)}
  /* Rare long SAN + large eval pairs wrap the score rather than spilling into the next ply. */
  :global(.movelist .mv){display:flex; flex-wrap:wrap; align-items:center; align-content:center; justify-content:space-between; gap:0 5px; cursor:pointer; min-height:24px; padding:0 9px; border:0; border-radius:0; text-align:left; background:transparent; color:var(--ink); font:inherit}
  :global(.movelist .mv:hover:not(.current)){background:var(--accent-soft) !important; color:var(--ink)}
  /* The current move is unmistakable (Lichess-strength): tinted cell plus an edge marker. */
  :global(.movelist .mv.current){background:color-mix(in srgb, var(--accent) 28%, var(--panel)) !important; color:var(--ink); font-weight:700}
  :global(.movelist .mv-eval){margin-left:auto; white-space:nowrap; font:13px var(--mono); font-variant-numeric:tabular-nums; color:var(--muted); font-weight:400}
  :global(.movelist .mv.current .mv-eval){color:inherit; opacity:.85}
  @media (max-width:900px){
    .analysis-workspace{grid-template-columns:1fr; max-width:680px; padding:12px; gap:10px}
    .board-column{display:contents}
    .board-panel, .analysis-workspace .app-sidebar{grid-column:1; grid-row:auto}
    .analysis-workspace .app-sidebar{display:grid; grid-template-columns:minmax(50px,1fr) auto auto auto; column-gap:5px; height:auto; max-height:none; overflow:visible}
    .analysis-workspace .app-sidebar>*{grid-column:1 / -1}
    .board-toolbar{height:44px}
    /* Full PV cards have their own mobile view; Add engine stays beside the run control. */
    .analysis-overview, .analyze-actions, .analysis-workspace .app-sidebar>.analysis-content{display:contents}
    .analysis-workspace .engine-strip{display:none; grid-column:1 / -1; grid-row:3}
    .analysis-workspace :global(.analysis-content[data-view=engines] > .engine-strip){display:flex}
    .position-status{grid-column:1; grid-row:1; min-height:44px; padding-left:7px}
    .analyze-actions button{grid-row:1; align-self:center; min-height:36px; margin-block:4px}
    .manage-engines{grid-column:2}
    .manage-label, .manage-engines .engine-count{display:none}
    .run-action{grid-column:4; margin-right:5px}
    .add-engine-inline{grid-column:3 !important; grid-row:1; align-self:center; min-height:36px; margin:4px 0; padding:0 9px; gap:4px}
    #addEngineHint{display:none}
    .workbench-tabs{grid-column:1 / -1; grid-row:2; position:sticky; top:var(--analysis-header-height); z-index:10}
    #enginesTab{display:inline-flex}
    .analysis-workspace .analysis-content>:is(.history-panel, .comparison-panel, .human-panel, .review-panel){min-height:0; grid-column:1 / -1; grid-row:3; flex:0 0 auto}
    .analysis-workspace .review-panel{max-height:340px}
    /* Keep the existing mobile engine actions in their first row while Review is open. */
    :global(.analysis-content[data-view=review]) .add-engine-inline{display:flex}
    .app-sidebar #message{grid-column:1 / -1; grid-row:4}
    .app-sidebar #runtimeWarning{grid-column:1 / -1; grid-row:5}
    .app-sidebar #downloadProgress{grid-column:1 / -1; grid-row:6}
    .engine-summaries{max-height:none; overflow:visible}
    .history-panel :global(.movelist){max-height:260px}
    .analysis-workspace .comparison-panel, .analysis-workspace .human-panel{max-height:none}
    :global(.movelist .mv){min-height:32px}
  }
  @media (max-width:600px){
    .analysis-workspace{padding:8px 4px 32px; gap:8px}
    .board-panel{padding:0; border:0; background:transparent}
    /* Labels have their own baseline; the full-width 10px fill remains unobstructed. */
    .board-readout{min-height:32px; padding:0 0 4px; gap:4px}
    .eval-chip{min-width:0; max-width:calc(100% - 108px); gap:5px; padding:0 5px; min-height:28px}
    .eval-score, .eval-caret{flex-shrink:0}
    .eval-source{font-size:12px}
    .eval-turn{font-size:12px; gap:4px}
    .eval-prefix{display:none}
    .board-wrap{grid-template-columns:1fr; gap:0}
    .evalbar{width:auto; height:10px; border:1px solid #8a8780; border-bottom:0; border-radius:3px 3px 0 0}
    .evalbar .white{top:0; bottom:0; left:0; right:auto; height:auto; width:var(--eval-pct)}
    .evalbar .mid{top:0; bottom:0; left:50%; width:2px; height:auto}
    .board-panel :global(#ground){border-radius:3px}
    .board-toolbar{display:grid; grid-template-columns:1fr; height:auto; gap:4px; margin:6px 0 0}
    .nav{gap:4px; height:44px}
    .nav button{min-width:0; padding:4px}
    .board-tools{display:grid; grid-template-columns:repeat(4,1fr); gap:4px; height:36px}
    .tool-button, .tool-button.icon-only{width:auto; padding:0 6px; gap:5px; font-size:12.5px}
    .copy-button{min-width:0}
    .copy-button svg{width:14px; height:14px; flex-basis:14px}
    .position-status{min-width:0}
    .analysis-status{gap:4px; font-size:12px; color:var(--muted)}
    .analyze-actions button{min-height:36px; padding:0 8px; font-size:13px; gap:5px}
    .run-action{min-width:90px}
    .engine-count{min-width:18px; height:18px; padding:0 4px}
    .add-engine-plus{width:14px; flex-basis:14px}
    .add-engine-text strong{font-size:13px}
    .opening-name{white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:7px 9px}
    .history-panel :global(.movelist){max-height:256px}
    /* Four recent pairs fit beside the navigation without a second page scroll. */
    .history-panel :global(.movelist:has(.move-pair)){max-height:120px}
    :global(.movelist .mv){min-height:30px}
    :global(.move-pair){grid-template-columns:32px minmax(90px,110px) minmax(90px,110px)}
    .app-sidebar #message{margin:4px 10px}
    :global(.analysis-workspace #ground coords){font-size:11px}
    :global(.insight-action){min-height:44px; padding:6px 10px; gap:3px}
    :global(.insight-headline){font-size:14px}
    :global(.insight-spread){font-size:13px}
    :global(.engine-summary){grid-template-columns:minmax(0,1fr) auto 56px 36px; gap:5px; min-height:40px; padding:2px 0 0 7px}
    :global(.engine-summary-name){min-height:38px; gap:7px}
    :global(.engine-summary-text){display:grid; gap:2px}
    :global(.engine-summary-name strong){font-size:13px}
    :global(.engine-summary-name small){font-size:13px}
    :global(.engine-summary-score){min-width:56px; font-size:16px; gap:4px}
    :global(.engine-gear){width:36px; height:40px}
    .workbench-tabs{padding:0 2px; gap:0}
    .workbench-tabs [role=tab]{flex:1 1 auto; justify-content:center; padding:0 6px; min-height:42px}
    .review-panel{padding:10px}
    .review-settings select, .review-actions button, .review-exports button, .review-moments-heading button, :global(.review-critical button){min-height:40px}
    .tab-badge{display:none}
    :global(.engine-summary){padding-top:0}
    :global(.line-play){min-height:36px; padding-left:36px}
    :global(.result-line){min-height:36px}
    :global(.line-toggle){min-height:36px; min-width:72px}
    :global(.engine-lines .line-error), :global(.engine-lines .pending-line){padding-left:36px}
    .position-tools :is(button, input), .pgn-actions button{min-height:44px}
    .analysis-workspace .section-block{padding:8px 10px}
    :global(.analysis-workspace .engine-row){flex-wrap:wrap}
    :global(.analysis-workspace .engine-row .row-unit){display:block}
    :global(.analysis-workspace .engine-row .row-fam){flex-basis:calc(100% - 40px)}
    .engine-manager{padding:14px}
    :global(table.engine-compare th), :global(table.engine-compare td){padding-inline:3px}
    :global(table.engine-compare tbody th){font-size:12px}
  }
  @media (max-width:360px){
    .eval-turn{font-size:11px}
    .eval-source:global([data-short]:not([data-short='']))::before{display:none}
    .analyze-actions button{padding-inline:6px}
    .workbench-tabs [role=tab]{font-size:12px; padding-inline:4px}
  }
  @media (prefers-reduced-motion: reduce){
    .evalbar .white{transition:none}
  }
</style>
