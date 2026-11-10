<script lang="ts">
import { onMount } from 'svelte';
import { afterNavigate, beforeNavigate } from '$app/navigation';
import BrowserCapabilities from '$lib/components/BrowserCapabilities.svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';

let mountedUrl: URL | null = null;
let isRunActive = () => false;
beforeNavigate(({ cancel, willUnload }) => {
  if (isRunActive() && !willUnload && !window.confirm('Leave this tournament? The current game will be lost. Completed games are saved.')) cancel();
});
afterNavigate(({ to }) => {
  if (mountedUrl && to?.url.pathname === mountedUrl.pathname && to.url.search !== mountedUrl.search) location.reload();
});
const title = '0x88 Chess — engine arena';
const description =
  'Run chess engine tournaments in your browser: head-to-head matches, round-robins, and gauntlets with standings, Elo estimates, and per-game charts.';
const devMode = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('dev');
onMount(() => {
  mountedUrl = new URL(location.href);
  let cleanup: () => void = () => undefined;
  let mounted = true;
  void import('../../../lc0/arenaBrowser')
    .then((module) => {
      if (!mounted) return;
      isRunActive = module.isArenaRunActive;
      cleanup = module.mountArenaBrowser();
    })
    .catch((error) => {
      if (!mounted) return;
      console.error('[arena] failed to load page controller', error);
      const node = document.getElementById('message');
      if (node) node.textContent = `Page failed to initialize: ${error instanceof Error ? error.message : String(error)}`;
    });
  return () => {
    mounted = false;
    isRunActive = () => false;
    cleanup();
  };
});
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>

<SiteHeader pageTitle="Arena" />
<main id="main" class="app-main arena-workspace" data-phase="setup" data-match="none" data-config="closed">
  <div class="board-column">
  <section class="panel board-panel" aria-label="Board">
    <div id="blackSideLabel" class="player-strip" aria-label="Black player"></div>
    <div class="board-with-eval">
      <div id="arenaEvalBar" class="arena-eval-bar empty" role="img" aria-label="Waiting for first search">
        {#each ['w', 'b'] as side}
          <div class="eval-lane unavailable" data-side={side}><div class="eval-fill"></div><span class="eval-cap"></span><span class="eval-lane-label"></span></div>
        {/each}
        <div class="eval-midline"></div><span class="eval-end w" aria-hidden="true">W</span><span class="eval-end b" aria-hidden="true">B</span>
      </div>
      <div class="board-shell"><div id="ground"></div></div>
    </div>
    <div id="whiteSideLabel" class="player-strip" aria-label="White player"></div>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex (The caption exposes both search ages on keyboard focus.) -->
    <div id="evalSource" class="eval-source" tabindex="0">No evaluation yet</div>
  </section>
  <section id="chartsPanel" class="panel charts-panel" aria-label="Game charts" data-chart="evalChart" hidden>
    <div class="charts-head">
      <h2 class="block-heading">This game</h2>
      <div class="chart-tabs" role="group" aria-label="Chart metric">
        <button type="button" data-chart="evalChart" aria-pressed="true">Eval</button>
        <button type="button" data-chart="timeChart" aria-pressed="false">Time</button>
        <button type="button" data-chart="npsChart" aria-pressed="false">Speed</button>
        <button id="rootChartTab" type="button" data-chart="rootChart" aria-pressed="false" hidden>Root visits</button>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card" data-chart="evalChart"><div class="chart-title">White expected score · 50% = equal</div><div id="evalChart"></div></div>
      <div class="chart-card" data-chart="timeChart"><div class="chart-title">Search time per move · milliseconds</div><div id="timeChart"></div></div>
      <div class="chart-card" data-chart="npsChart"><div class="chart-title">Search speed · nodes or visits per second</div><div id="npsChart"></div></div>
      <div class="chart-card" data-chart="rootChart" hidden><div class="chart-title" id="rootChartTitle">LC0 root visits</div><div id="rootChart"></div>
        <label class="root-review">Preview candidate <select id="rootCandidate" aria-label="Preview a root search candidate"></select></label>
      </div>
    </div>
    <div class="chart-footer">
      <div id="chartLegend" class="chart-legend"></div>
      <div class="chart-review">
        <input id="chartPly" type="range" min="0" max="0" value="0" aria-label="Review game position" title="Review a position (arrow keys move one ply)" />
        <button id="chartLive" type="button">Latest</button>
      </div>
    </div>
  </section>
  </div>
  <section class="panel app-sidebar arena-sidebar" aria-label="Arena">
    <div class="status-block">
      <div id="message" class="arena-status" aria-live="polite">Loading…</div>
      <div id="runtimeWarning" class="runtime-warning" role="status" aria-live="polite" hidden></div>
      <div id="engineWarmupStatus" class="engine-warmup-status" role="status" aria-live="polite" aria-atomic="true" hidden>
        <progress aria-label="Engine preparation in progress"></progress>
        <div id="engineWarmupTitle" class="engine-warmup-title"></div>
        <div id="engineWarmupPhase" class="engine-warmup-phase"></div>
        <div id="engineWarmupDetail" class="engine-warmup-detail"></div>
      </div>
      <div id="downloadProgress" class="model-load-progress" hidden></div>
    </div>
    <section id="checkpointSection" class="checkpoint-block" aria-label="Saved tournament" hidden>
      <p id="checkpointText" role="status"></p>
      <p id="checkpointRoster" class="field-hint"></p>
      <div class="checkpoint-actions">
        <button id="resumeTournament" type="button" class="primary">Resume tournament</button>
        <button id="discardTournament" type="button">Discard</button>
      </div>
    </section>
    <section id="scoreboard" class="scoreboard" aria-label="Score">
      <div class="sb-head">
        <span id="scoreboardTitle" class="sb-title">Match</span>
        <span class="sb-state" data-state="live"><span class="live-dot" aria-hidden="true"></span>Live</span>
        <span class="sb-state" data-state="final">Final</span>
        <span class="sb-state" data-state="stopped">Stopped</span>
        <span class="sb-state" data-state="failed">Not started</span>
        <div class="match-controls">
          <button id="editSetup" class="ghost-button" type="button" aria-expanded="false" aria-controls="setupPane" title="View the match settings (locked until you stop)">Settings</button>
          <button id="stop" class="stop-button" type="button" disabled>Stop</button>
        </div>
      </div>
      <div id="matchScore" aria-live="polite"></div>
      <div class="sb-context">
        <span id="setupSummaryText"></span><span id="pairing" class="sb-progress"></span>
        <details id="openingPairsSection" class="arena-results" hidden>
          <summary>Opening pairs</summary>
          <select id="pairMatchup" aria-label="Opening pairs matchup"></select>
          <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users need to scroll the table.) -->
          <div id="openingPairs" class="arena-table-scroll" tabindex="0" role="region" aria-label="Colour-reversed opening pairs"></div>
        </details>
      </div>
      <div id="gameChips" class="sb-chips" aria-label="Finished games" hidden></div>
    </section>
    <details id="crosstableSection" class="section-block arena-results" open hidden>
      <summary>Crosstable</summary>
      <!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users need to scroll the table.) -->
      <div id="crosstable" class="arena-table-scroll" tabindex="0" role="region" aria-label="Opponent crosstable"></div>
    </details>
    <fieldset id="setupPane" class="setup-pane" aria-label="Tournament settings">
    <p class="locked-note">Settings are locked while games run. Stop to change them.</p>
    <details class="section-block" open>
      <summary>Matchup</summary>
      <select id="seatA" hidden></select><select id="seatB" hidden></select>
      <div class="field mode-field"><label for="tournamentModeSelect">Mode</label>
        <select id="tournamentModeSelect"><option value="match">Match (two engines)</option><option value="round-robin">Round robin</option><option value="gauntlet">Gauntlet (Engine 1 vs field)</option></select></div>
      <div id="arenaSeatList" class="engine-list" aria-label="Arena engine selectors" data-fixed="false"></div>
      <button id="addSeat" class="add-seat" type="button" title="Add another engine seat">+ Add engine</button>
      <p class="matchup-note" id="matchupNote">Equal time per move · colors alternate.</p>
    </details>
    <details class="section-block" open>
      <summary>Time control &amp; openings</summary>
      <div class="time-grid">
        <div class="field"><label for="budgetModeSelect">Budget</label>
          <select id="budgetModeSelect"><option value="fixed">Fixed visits/depth</option><option value="movetime" selected>Equal movetime</option></select></div>
        <div class="field" id="movetimeField"><label for="movetimeInput">Per move</label>
          <div class="unit-input"><input id="movetimeInput" type="number" min="10" max="60000" step="50" value="500" /><span>ms</span></div></div>
        <div class="field"><label for="gamesInput">Games / opening</label>
          <input id="gamesInput" type="number" min="1" max="20" step="1" value="2" /></div>
      </div>
      <div hidden inert>
        <input id="lc0VisitsInput" type="number" value="100" />
        <input id="sfDepthInput" type="number" value="8" />
        <input id="recklessDepthInput" type="number" value="4" />
        <select id="recklessVariantSelect"></select>
        <input id="viridithasDepthInput" type="number" value="6" />
        <select id="viridithasVariantSelect"></select>
        <input id="berserkDepthInput" type="number" value="4" />
        <select id="berserkVariantSelect"></select>
        <input id="plentychessDepthInput" type="number" value="4" />
        <select id="plentychessVariantSelect"></select>
        <input id="stormphraxDepthInput" type="number" value="4" />
        <select id="stormphraxVariantSelect"></select>
      </div>
      <div class="field opening-field"><label for="startingPositionSelect">Openings</label>
        <select id="startingPositionSelect"><option value="start">Start position</option><option value="built-in">Built-in opening suite</option><option value="custom">Custom positions / replays</option></select></div>
      <div class="field wide" id="openingTextField" hidden><label for="openingText">Custom openings (one per line: FEN, UCI moves, PGN/SAN, or Name | ...)</label>
        <textarea id="openingText" spellcheck="false" placeholder="Ruy Lopez | e2e4 e7e5 g1f3 b8c6 f1b5&#10;Italian Game | 1. e4 e5 2. Nf3 Nc6 3. Bc4&#10;Sicilian FEN | rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2"></textarea></div>
      <p id="openingInfo" class="field-hint">Start position only.</p>
      <details class="more-options">
        <summary>Parallel games</summary>
        <div class="field"><label for="arenaConcurrencySelect">Concurrent CPU games</label>
          <select id="arenaConcurrencySelect"><option value="auto">Auto (conservative)</option><option value="1">1</option><option value="2">2</option><option value="3">3 (more memory)</option></select></div>
        <p id="arenaConcurrencyNote" class="field-hint">Neural pools always play one game at a time.</p>
      </details>
    </details>
    {#if devMode}
    <details class="section-block advanced-settings">
      <summary>Advanced settings &amp; diagnostics</summary>
      <div class="row">
        <div class="field wide"><label for="lc0PresetSelect">LC0 engine preset</label>
          <select id="lc0PresetSelect"><option value="stable">Stable LC0 Small (ORT WebGPU/WASM fallback)</option><option value="benchmarked-small">Fast LC0 Small — best benchmarked eval path (research)</option><option value="custom">Custom / advanced</option></select></div>
        <div id="lc0PresetNote" class="preset-note small">Stable default. Use the fast preset to test whether faster eval helps fixed-time arena strength.</div>
        <details id="lc0AdvancedRuntime" class="advanced-runtime">
          <summary>Advanced LC0 runtime knobs</summary>
          <div class="row">
            <div class="field"><label for="lc0RuntimeSelect">LC0 backend</label>
              <select id="lc0RuntimeSelect"><option value="onnx">ORT ONNX</option><option value="hybrid-ort-heads">WGSL encoder + ORT heads</option><option value="hybrid-wgsl-heads">WGSL encoder + WGSL heads (experimental)</option></select></div>
            <div class="field"><label for="lc0BatchSizeInput">LC0 search batch</label>
              <input id="lc0BatchSizeInput" type="number" min="1" max="64" step="1" value="1" /></div>
            <div class="field"><label for="lc0BatchPipelineDepthInput">LC0 pipeline</label>
              <input id="lc0BatchPipelineDepthInput" type="number" min="1" max="16" step="1" value="1" /></div>
            <div class="field"><label for="lc0InputBackendSelect">LC0 input</label>
              <select id="lc0InputBackendSelect"><option value="js">JS</option><option value="wasm">WASM</option><option value="wgsl">WGSL</option></select></div>
            <div class="field"><label for="lc0EncoderKernelSelect">LC0 encoder</label>
              <select id="lc0EncoderKernelSelect"><option value="hand">Hand WGSL</option><option value="mixed-tvm-ffn">Mixed TVM FFN</option><option value="mixed-tvm-ffn-outproj">Mixed TVM FFN+outproj</option><option value="mixed-tvm-ffn-smolgen-project">Mixed TVM FFN+smolgen project</option><option value="tvm-packed-f16">TVM packed f16</option></select></div>
            <div class="field"><label for="lc0LegalPriorsSelect">LC0 legal priors</label>
              <select id="lc0LegalPriorsSelect"><option value="js">JS</option><option value="wasm">WASM</option><option value="gpu">GPU legal (WGSL heads)</option></select></div>
          </div>
        </details>
        <div class="field"><label for="delayInput">Delay ms</label>
          <input id="delayInput" type="number" min="0" max="3000" step="50" value="250" /></div>
        <div class="field"><label for="cacheEntriesInput">NN cache entries</label>
          <input id="cacheEntriesInput" type="number" min="0" max="100000" step="128" value="2048" /></div>
        <div class="field"><label for="stockfishThreadsInput">SF threads (0 = auto)</label>
          <input id="stockfishThreadsInput" type="number" min="0" max="32" step="1" value="1" /></div>
      </div>
      <details class="runtime-details"><summary>Runtime diagnostics</summary>
        <div id="recklessRuntimeInfo" class="diag-block">Reckless: detecting runtime…</div>
        <div id="viridithasRuntimeInfo" class="diag-block">Viridithas: detecting runtime…</div>
        <div id="berserkRuntimeInfo" class="diag-block">Berserk: detecting runtime…</div>
        <div id="plentychessRuntimeInfo" class="diag-block">PlentyChess: detecting runtime…</div>
        <div id="stormphraxRuntimeInfo" class="diag-block">Stormphrax: detecting runtime…</div>
        <div id="cacheInfo" class="diag-block">NN cache: loading…</div>
        <div id="searchTelemetryInfo" class="diag-block">LC0 tree: waiting for searches…</div>
        <div id="runtimeAuditInfo" class="diag-block"><span class="diag-label">LC0 audit</span><span class="diag-value">waiting for runtime selection…</span></div>
      </details>
    </details>
    {:else}
    <div hidden>
      <select id="lc0PresetSelect"><option value="stable">Stable</option></select>
      <div id="lc0PresetNote"></div>
      <details id="lc0AdvancedRuntime"><select id="lc0RuntimeSelect"><option value="onnx">ORT ONNX</option><option value="hybrid-ort-heads">WGSL encoder + ORT heads</option><option value="hybrid-wgsl-heads">WGSL encoder + WGSL heads (experimental)</option><option value="whole-onnx-webgpu">TVM whole-model WebGPU (fast, small net)</option></select>
        <input id="lc0BatchSizeInput" type="number" value="1" />
        <input id="lc0BatchPipelineDepthInput" type="number" value="1" />
        <select id="lc0InputBackendSelect"><option value="js">JS</option><option value="wasm">WASM</option><option value="wgsl">WGSL</option></select>
        <select id="lc0EncoderKernelSelect"><option value="hand">Hand WGSL</option><option value="mixed-tvm-ffn">Mixed TVM FFN</option><option value="mixed-tvm-ffn-outproj">Mixed TVM FFN+outproj</option><option value="mixed-tvm-ffn-smolgen-project">Mixed TVM FFN+smolgen project</option><option value="tvm-packed-f16">TVM packed f16</option></select>
        <select id="lc0LegalPriorsSelect"><option value="js">JS</option><option value="wasm">WASM</option><option value="gpu">GPU legal (WGSL heads)</option></select>
      </details>
      <input id="delayInput" type="number" value="250" />
      <input id="cacheEntriesInput" type="number" value="2048" />
      <input id="stockfishThreadsInput" type="number" value="1" />
      <div id="recklessRuntimeInfo"></div>
      <div id="viridithasRuntimeInfo"></div>
      <div id="berserkRuntimeInfo"></div>
      <div id="plentychessRuntimeInfo"></div>
      <div id="stormphraxRuntimeInfo"></div>
      <div id="cacheInfo"></div>
      <div id="searchTelemetryInfo"></div>
      <div id="runtimeAuditInfo"></div>
    </div>
    {/if}
    </fieldset>
    <div class="arena-actions">
      <div id="startSummary" class="start-summary"></div>
      <button id="start" class="primary" type="button" disabled>Start match</button>
    </div>
    <section id="engineOutputsSection" class="engines-block" aria-label="Engine outputs" hidden>
      <h2 class="block-heading">Engines <span class="block-hint">evals from White's view</span></h2>
      <div id="engineEvalInfo" class="eng-grid"></div>
    </section>
    <section id="movesBlock" class="moves-block" aria-label="Moves" hidden>
      <h2 class="block-heading moves-heading">Moves <span class="block-hint">Select a move to review</span></h2>
      <div class="move-columns" aria-hidden="true"><span>#</span><span>White</span><span>Black</span></div>
      <div id="reviewBar" class="review-bar" hidden><span id="reviewLabel"></span></div>
      <div id="gameMoves" class="move-list"></div>
      <div class="move-nav" role="group" aria-label="Browse moves">
        <button id="revStart" type="button" title="First move" aria-label="First move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14M18 5l-8 7 8 7" /></svg></button>
        <button id="revPrev" type="button" title="Previous move (←)" aria-label="Previous move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-8 7 8 7" /></svg></button>
        <button id="revNext" type="button" title="Next move (→)" aria-label="Next move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 8 7-8 7" /></svg></button>
        <button id="revEnd" type="button" title="Last move" aria-label="Last move"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M6 5l8 7-8 7" /></svg></button>
        <button id="revLive" type="button" class="primary live-button" title="Back to the live game (Esc)" hidden>Back to live</button>
      </div>
    </section>
    <details class="eng-more search-details"><summary>Search details</summary><div id="engineRaw" class="eng-raw"></div></details>
    <details id="resultSection" class="section-block results-block">
      <summary>Game log &amp; PGN</summary>
      <p id="checkpointStorageText" class="field-hint" hidden></p>
      <div class="log" id="log"></div>
      <pre id="benchResult" class="small" hidden></pre>
      <div class="row"><button id="exportPgn" type="button">Export PGN</button></div>
      <textarea id="pgnOut" aria-label="Exported tournament PGNs" spellcheck="false" placeholder="game PGNs appear here after Export PGN" hidden></textarea>
    </details>
    <details class="section-block runtime-block">
      <summary>Runtime status</summary>
      <div id="runtimeBadge" class="runtime-badge">Runtime: detecting browser isolation and GPU…</div>
      <div class="capabilities-slot"><BrowserCapabilities /></div>
    </details>
  </section>
</main>

<style>
  /* Page-local tokens (as on Play): readable secondary text and >=3:1 control boundaries in both themes. */
  .arena-workspace{
    --muted:#5f584c;
    --text-soft:var(--muted);
    --border-input:#8f8470; --border-input-hover:#6f6553;
    --eng:var(--border-input); /* per-engine colour; arenaBrowser.ts sets it inline on each card/bar */
    --coord-dark:#211d17; --coord-light:#fff8eb;
    --strip-h:46px;
    --action-space:132px;
    --eval-dark:#2b2722; --eval-light:#f4f1e8;
    font-variant-numeric:tabular-nums;
    scroll-margin-top:74px;
  }
  :global(:root[data-theme="dark"]) .arena-workspace{--muted:color-mix(in srgb, var(--ink) 72%, var(--panel)); --border-input:#766d5a; --border-input-hover:#958b77}
  @media (prefers-color-scheme: dark){
    :global(:root:not([data-theme])) .arena-workspace{--muted:color-mix(in srgb, var(--ink) 72%, var(--panel)); --border-input:#766d5a; --border-input-hover:#958b77}
  }

  .board-column{display:grid; gap:10px; min-width:0}
  /* Keep one chart in view without shrinking the scoresheet. */
  @media (min-width:901px){
    .arena-workspace{grid-template-columns:minmax(0, clamp(360px, calc(100dvh - 340px), 690px)) minmax(440px,480px); max-width:1250px; column-gap:18px; row-gap:10px}
  }

  /* ── Board column: strip / eval bar + board / strip ── */
  .board-panel{display:flex; flex-direction:column; gap:4px; min-width:0; padding:8px}
  .board-with-eval{display:grid; grid-template-columns:42px minmax(0,1fr); gap:8px; align-items:stretch}
  .board-shell{position:relative}
  .arena-workspace :global(#ground coords){font-size:12px; font-weight:700; opacity:1}
  .arena-eval-bar{
    --pct:0%;
    position:relative; display:grid; grid-template-columns:1fr 1fr; gap:2px;
    border:1px solid var(--border-input); border-radius:4px; background:var(--rule-strong);
  }
  .arena-eval-bar .eval-lane{position:relative; background:var(--eval-dark); border-radius:2px}
  .arena-eval-bar .eval-lane.unavailable{background:var(--panel-inset)}
  .arena-eval-bar .eval-lane:global(.stale){opacity:.4}
  .arena-eval-bar .eval-fill{position:absolute; inset:auto 0 0; height:var(--pct); background:var(--eval-light); border-radius:2px; transition:height .18s ease, width .18s ease}
  .arena-eval-bar .eval-cap{position:absolute; z-index:2; inset:0 0 auto; height:3px; background:var(--eng); transition:opacity .18s ease}
  .arena-eval-bar .eval-lane-label{position:absolute; z-index:2; top:5px; left:0; right:0; background:var(--panel); color:color-mix(in srgb,var(--eng) 65%,var(--ink)); text-align:center; font:700 10px/14px var(--mono); font-variant-numeric:slashed-zero}
  .arena-eval-bar .eval-lane.unavailable :is(.eval-cap,.eval-fill){display:none}
  .arena-eval-bar .eval-lane.unavailable .eval-lane-label{color:var(--muted)}
  .arena-eval-bar :global(.eval-midline){position:absolute; z-index:1; left:0; right:0; top:50%; height:1px; background:#a77a23}
  /* The coloured edge belongs to the moving fill, so they cannot diverge. */
  .arena-eval-bar .eval-fill::after{content:''; position:absolute; z-index:2; left:0; right:0; top:0; height:1px; background:var(--eng)}
  .eval-end{display:none}
  .eval-source{display:flex; align-items:center; flex-wrap:wrap; gap:4px 10px; min-width:0; min-height:18px; color:var(--muted); font:600 13px/18px var(--sans)}
  .eval-source:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
  .eval-source :global(.eval-reading){display:flex; align-items:baseline; gap:5px; min-width:0; color:color-mix(in srgb, var(--eng) 65%, var(--ink))}
  .eval-source :global(.eval-side){font-weight:600; color:var(--ink)}
  .eval-source :global(.eval-engine){font-family:var(--mono); font-variant-numeric:slashed-zero}
  .eval-source :global(.eval-engine){overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  .eval-source :global(strong){font:700 13px/18px var(--mono); white-space:nowrap; color:inherit}
  .eval-source :global(.eval-reading.pending){color:var(--muted)}
  .eval-source :global(.eval-comparison){color:var(--muted); font:500 13px/18px var(--sans)}
  .eval-source :global(.eval-comparison.emphasis){color:var(--warn); font-weight:650}
  .eval-source :global(.eval-comparison.opposite){font-weight:800}

  .player-strip{
    display:flex; align-items:center; gap:10px; min-width:0; height:var(--strip-h);
    padding:0 6px 0 4px; border-radius:8px;
  }
  .player-strip:global([data-turn=true]){background:color-mix(in srgb, var(--accent) 13%, var(--panel))}
  .arena-workspace :global(.arena-avatar){
    display:inline-grid; place-items:center; flex:0 0 32px; width:32px; height:32px; border-radius:7px;
    background:var(--eng); color:#fff; font:700 11px/1 var(--mono); font-variant-numeric:slashed-zero; letter-spacing:0;
    box-shadow:inset 0 0 0 1px #ffffff2e;
  }
  .arena-workspace :global(.arena-avatar.sm){flex-basis:24px; width:24px; height:24px; border-radius:5px; font-size:10px}
  .player-strip :global(.strip-id){display:flex; flex:1 1 auto; flex-direction:column; min-width:0; line-height:1.25}
  .player-strip :global(.strip-id strong){color:var(--ink); font:650 15px var(--sans); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .player-strip :global(.strip-id small){display:flex; align-items:center; gap:6px; color:var(--muted); font-size:12.5px; white-space:nowrap; overflow:hidden}
  .player-strip :global(.side-dot){flex:0 0 10px; width:10px; height:10px; border-radius:50%; border:1px solid #6f6553}
  .player-strip :global(.side-dot.w){background:#f6f3ea}
  .player-strip :global(.side-dot.b){background:#1d1b18; border-color:#8f8573}
  .player-strip :global(.strip-thinking){display:inline-flex; align-items:center; gap:6px; color:var(--accent-soft-text); font-weight:650}
  .player-strip :global(.strip-thinking)::before{
    content:''; width:10px; height:10px; border-radius:50%;
    border:2px solid color-mix(in srgb, var(--accent) 30%, transparent); border-top-color:var(--accent);
    animation:arena-spin .8s linear infinite;
  }
  .player-strip :global(.strip-metric){display:flex; flex:0 0 auto; flex-direction:column; align-items:flex-end; gap:1px; color:var(--ink); font:650 14px/1.2 var(--sans)}
  .player-strip :global(.strip-metric small){color:var(--muted); font:500 11px/1.2 var(--sans)}
  .player-strip :global(.strip-clock){min-width:66px}
  @keyframes arena-spin{to{transform:rotate(360deg)}}

  /* ── Sidebar: setup state vs running state (data-phase on <main>) ── */
  .arena-workspace .arena-sidebar{display:flex; flex-direction:column; min-width:0; max-height:none; overflow:hidden; position:static}
  .arena-sidebar > *{flex:0 0 auto}
  .status-block{padding:12px 14px 11px; border-bottom:1px solid var(--rule); background:var(--panel-inset)}
  .status-block:not(:has(#message:not(:empty), #runtimeWarning:not([hidden]), #engineWarmupStatus:not([hidden]), #downloadProgress:not([hidden]))){display:none}
  .arena-status{color:var(--ink); font:600 14.5px/1.4 var(--sans)}
  .arena-status:empty{display:none}
  .runtime-warning{
    margin-top:7px; padding:7px 9px; border:1px solid color-mix(in srgb, var(--warn) 45%, var(--rule));
    border-radius:6px; background:var(--warn-soft); color:var(--warn); font-size:12.5px; line-height:1.4;
  }
  .engine-warmup-status{
    margin-top:8px; padding:10px 11px; display:grid; gap:5px;
    border:1px solid color-mix(in srgb, var(--accent) 38%, var(--rule)); border-radius:var(--radius-sm); background:var(--accent-soft);
  }
  .engine-warmup-status progress{width:100%; height:7px; accent-color:var(--accent)}
  .engine-warmup-title{font-size:13px; font-weight:700; color:var(--ink)}
  .engine-warmup-phase{font-size:12.5px; font-weight:600; color:var(--text)}
  .engine-warmup-detail{font-size:12.5px; line-height:1.45; color:var(--muted)}
  .model-load-progress{margin-top:8px}
  .model-load-progress :global(progress){width:100%; height:8px; accent-color:var(--accent)}
  .model-load-progress :global(.dl-label){margin-top:2px; font-size:12px; color:var(--muted)}

  .checkpoint-block{padding:12px 14px; border-bottom:1px solid var(--rule); background:var(--panel-inset)}
  .checkpoint-block p{margin:0; color:var(--text); font-size:13px; line-height:1.5; overflow-wrap:anywhere}
  .checkpoint-block .field-hint{margin-top:4px; color:var(--muted)}
  .checkpoint-block .field-hint:empty{display:none}
  .checkpoint-actions{display:flex; flex-wrap:wrap; gap:6px; margin-top:8px}
  .checkpoint-actions:has(button:not([hidden])){margin-top:8px}
  .checkpoint-actions:not(:has(button:not([hidden]))){display:none}
  .checkpoint-actions button{min-height:44px; font-size:13px; padding:8px 12px; white-space:normal; text-align:left}

  /* Scoreboard */
  .scoreboard{display:none; padding:12px 14px; border-bottom:1px solid var(--rule)}
  .arena-workspace:global(:not([data-match=none])) .scoreboard{display:block}
  .sb-head{display:flex; align-items:center; gap:8px}
  .match-controls{display:none; align-items:center; gap:6px; margin-left:4px}
  .arena-workspace:global([data-phase=running]) .match-controls{display:flex}
  .sb-title{flex:1 1 auto; color:var(--ink); font:650 13px var(--sans)}
  .sb-state{display:none; align-items:center; gap:6px; padding:2px 9px; border-radius:999px; font:650 12px var(--sans); border:1px solid var(--rule-strong); color:var(--text)}
  .arena-workspace:global([data-match=live]) .sb-state[data-state=live],
  .arena-workspace:global([data-match=final]) .sb-state[data-state=final],
  .arena-workspace:global([data-match=stopped]) .sb-state[data-state=stopped],
  .arena-workspace:global([data-match=failed]) .sb-state[data-state=failed]{display:inline-flex}
  .sb-state[data-state=live]{color:var(--on-accent); border-color:var(--accent); background:var(--accent)}
  .sb-state[data-state=live] .live-dot{background:currentColor}
  .live-dot{width:7px; height:7px; border-radius:50%; background:var(--accent); animation:arena-pulse 1.6s ease-in-out infinite}
  @keyframes arena-pulse{50%{opacity:.35}}
  .sb-context{display:flex; flex-wrap:wrap; gap:2px 8px; margin-top:9px; color:var(--muted); font:500 12px/1.4 var(--sans)}
  .sb-progress:not(:empty)::before{content:'·'; margin-right:8px}
  .sb-progress:empty{display:none}
  .sb-context .arena-results{padding:0; border:0}
  .sb-context .arena-results>summary{cursor:pointer; color:var(--text); text-decoration:underline; text-underline-offset:3px}
  .sb-context .arena-results[open]{flex-basis:100%}
  .sb-context .arena-results[open]>summary{margin-bottom:8px}
  .scoreboard :global(.sb-match){display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); align-items:center; gap:10px; margin-top:10px}
  .scoreboard :global(.sb-player){display:flex; align-items:center; gap:8px; min-width:0}
  .scoreboard :global(.sb-player.right){flex-direction:row-reverse; text-align:right}
  .scoreboard :global(.sb-name){display:flex; flex-direction:column; min-width:0; line-height:1.25}
  .scoreboard :global(.sb-name strong){color:var(--ink); font:650 14px var(--sans); overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  .scoreboard :global(.sb-name small){color:var(--muted); font-size:12px}
  .scoreboard :global(.sb-points){color:var(--ink); font:750 32px/1 var(--sans); white-space:nowrap}
  .scoreboard :global(.sb-dash){margin:0 7px; color:var(--muted); font-weight:500}
  .scoreboard :global(.sb-record){margin-top:7px; color:var(--muted); font-size:12.5px; line-height:1.4}
  .scoreboard :global(.standings){width:100%; margin-top:8px; border-collapse:collapse; font-size:13px}
  .scoreboard :global(.standings th), .scoreboard :global(.standings td){padding:4px 6px; text-align:right; border-top:1px solid var(--rule); white-space:nowrap}
  .scoreboard :global(.standings th){border-top:0; color:var(--muted); font-weight:600; font-size:12px}
  .scoreboard :global(.standings th:nth-child(2)), .scoreboard :global(.standings td:nth-child(2)){text-align:left; width:99%}
  .scoreboard :global(.standings tr.leader td){background:var(--accent-soft)}
  .scoreboard :global(.sb-cell-name){display:inline-flex; align-items:center; gap:7px; color:var(--ink); font-weight:600}
  .arena-results{min-width:0; border-bottom:1px solid var(--rule)}
  .arena-results select{width:100%; margin-bottom:6px}
  .arena-sidebar :global(.arena-table-scroll){max-width:100%; min-width:0; overflow-x:auto; overscroll-behavior-x:contain; scrollbar-width:thin}
  .arena-sidebar :global(.arena-table-scroll:focus-visible){outline:2px solid var(--accent); outline-offset:-2px}
  .arena-sidebar :global(.arena-detail-table){width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.4}
  .arena-sidebar :global(.arena-detail-table caption){text-align:left; color:var(--muted); padding-bottom:8px; font-size:12px; min-width:240px}
  .arena-sidebar :global(.arena-detail-table th), .arena-sidebar :global(.arena-detail-table td){padding:6px 8px; border-bottom:1px solid var(--rule); text-align:center; white-space:nowrap}
  .arena-sidebar :global(.arena-detail-table th){color:var(--muted); font-weight:600}
  .arena-sidebar :global(.arena-detail-table th:first-child){text-align:left; color:var(--ink)}
  .arena-sidebar :global(.arena-detail-table thead th){white-space:normal; min-width:76px; max-width:160px}
  .arena-sidebar :global(.arena-detail-table .cross-self){background:var(--panel-inset); color:var(--muted)}
  .arena-sidebar :global(.arena-detail-table button){min-height:32px; padding:2px 8px; font-size:12.5px; background:transparent; color:var(--ink); border-color:var(--border-input)}
  .sb-chips{display:flex; flex-wrap:wrap; gap:5px; margin-top:9px}
  .sb-chips :global(.sb-chip){
    min-height:28px; padding:2px 9px; border-radius:999px; border:1px solid var(--border-input);
    background:var(--panel); color:var(--ink); font:600 12.5px var(--sans);
  }
  .sb-chips :global(.sb-chip-n){margin-right:6px; color:var(--muted); font-weight:500}
  .sb-chips :global(.sb-chip.win){border-color:var(--accent); background:var(--accent-soft); color:var(--accent-soft-text)}
  .sb-chips :global(.sb-chip.loss){border-color:color-mix(in srgb, var(--warn) 60%, var(--rule)); color:var(--warn)}

  /* Native fieldset locking also covers asynchronous run preparation. */
  .setup-pane{border:0; margin:0; padding:0; min-width:0}

  /* Locked settings stay available during play. */
  .locked-note{display:none; margin:0; padding:10px 14px 0; color:var(--muted); font-size:12.5px}
  .arena-workspace:global([data-phase=running]) .locked-note{display:block}
  .ghost-button{min-height:32px; padding:0 11px; border-radius:7px; background:transparent; border-color:var(--border-input); color:var(--text); font:600 13px var(--sans)}
  .ghost-button:global([aria-expanded=true]){border-color:var(--accent); color:var(--ink)}
  .stop-button{min-height:32px; padding:0 14px; border-radius:7px; border-color:var(--warn); background:transparent; color:var(--warn); font:650 13.5px var(--sans)}
  .stop-button:hover:not(:disabled){background:var(--warn); border-color:var(--warn); color:var(--bg)}
  .arena-workspace:global([data-phase=running]:not([data-config=open])) .setup-pane{display:none}
  .arena-workspace:global([data-phase=running]) .setup-pane{border-bottom:1px solid var(--rule); background:var(--panel-inset)}
  .arena-workspace:global([data-phase=running]) .arena-actions{display:none}
  .arena-actions{padding:4px 14px 14px}
  .start-summary{display:none}
  .arena-actions #start{width:100%; min-height:46px; border-radius:8px; font:700 15px var(--sans)}
  .arena-actions #start:not(.primary){background:transparent; border-color:var(--border-input); color:var(--ink)}
  .arena-workspace button:disabled{opacity:1; cursor:not-allowed; background:var(--panel-2); border-color:var(--rule-strong); color:color-mix(in srgb, var(--ink) 50%, var(--panel))}

  .arena-workspace .section-block{padding:12px 14px}
  .arena-workspace .section-block>summary{
    display:flex; align-items:center; gap:7px; padding:0 0 10px;
    font:650 13px/1.3 var(--sans); letter-spacing:0; text-transform:none; color:var(--ink);
  }
  .arena-workspace .section-block>summary::before{content:''; width:7px; height:7px; margin:0 2px 0 1px; border:solid var(--muted); border-width:0 2px 2px 0; transform:rotate(-45deg); transition:transform .15s}
  .arena-workspace .section-block[open]>summary::before{transform:rotate(45deg) translate(-2px,-1px); color:inherit}
  .arena-workspace .section-block:not([open])>summary{padding-bottom:0}
  .setup-pane .field{margin-top:0}
  .setup-pane .field label, .more-options .field label{color:var(--muted); font:500 13px var(--sans)}
  .mode-field select{width:100%}
  .field-hint{margin:6px 0 0; color:var(--muted); font-size:12.5px; line-height:1.45}
  .matchup-note{margin:8px 0 0; color:var(--muted); font-size:12.5px; line-height:1.4}
  .add-seat{margin-top:8px; min-height:36px; padding:6px 12px; border-style:dashed; border-color:var(--border-input); background:transparent; color:var(--text); font-weight:600}

  /* Seat rows: one aligned grid; every select gets the same chevron (the shared .engine-row rule's background shorthand dropped it). */
  .setup-pane :global(.engine-list){gap:6px; margin-top:12px}
  .setup-pane :global(.seat-head), .setup-pane :global(.seat-row){
    display:grid; grid-template-columns:96px minmax(0,1fr) minmax(0,1fr) auto auto; gap:6px; align-items:center;
  }
  .setup-pane :global(.seat-head){color:var(--muted); font:500 12px var(--sans)}
  .setup-pane :global(.seat-head span:first-child){visibility:hidden}
  .setup-pane :global(.seat-row .seat-name){display:inline-flex; align-items:center; gap:7px; color:var(--ink); font:600 13px var(--sans); white-space:nowrap}
  .setup-pane :global(.seat-row select){
    min-height:38px; padding:6px 28px 6px 10px; font:500 14px var(--sans); color:var(--ink-body-strong);
    border:1px solid var(--border-input); border-radius:6px; background-color:var(--input-bg);
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%238a7d66' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right 10px center;
  }
  .setup-pane :global(.seat-strength-wrap){display:inline-flex; align-items:center; gap:4px}
  .setup-pane :global(.engine-list[data-fixed=false] .seat-strength-wrap){display:none}
  .setup-pane :global(.seat-row .row-strength){flex:0 0 70px; width:70px; min-height:38px; border-color:var(--border-input)}
  .setup-pane :global(.seat-row .row-unit){color:var(--muted); font-size:12px}
  .setup-pane :global(.seat-inactive){opacity:.5}
  .setup-pane :global(.seat-remove){width:30px; min-height:30px; padding:0; border:1px solid var(--border-input); background:none; border-radius:6px; color:var(--muted); line-height:1}
  .setup-pane :global(.seat-remove:hover){color:var(--warn); border-color:var(--warn)}
  .arena-workspace select, .arena-workspace input{border-color:var(--border-input)}
  .arena-workspace input[type=number]{width:100%; min-height:38px; padding:6px 9px}
  .arena-workspace select{min-height:38px}

  .time-grid{display:grid; grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) minmax(0,1fr); gap:8px; align-items:end}
  .time-grid select{width:100%}
  .unit-input{position:relative; min-width:0}
  .unit-input input{padding-right:30px}
  .unit-input span{position:absolute; right:10px; top:50%; transform:translateY(-50%); color:var(--muted); font-size:12px; pointer-events:none}
  .setup-pane .opening-field{margin-top:12px}
  .opening-field select{width:100%}
  .setup-pane .field.wide{margin-top:10px}
  .field.wide textarea{margin-top:0; min-height:86px; font-size:12px}
  .more-options{margin-top:10px; border-top:1px solid var(--rule); padding-top:8px}
    .more-options select{width:100%}
  .setup-pane .more-options .field{margin-top:4px}

  /* Engine panels: one per engine, same structure for every engine */
  .engines-block{padding:12px 14px; border-top:1px solid var(--rule)}
  .block-heading{display:flex; align-items:baseline; gap:8px; margin:0 0 8px; font:650 13px var(--sans); letter-spacing:0; text-transform:none; color:var(--ink)}
  .block-hint{color:var(--muted); font-weight:500; font-size:12px}
  .eng-grid{display:grid; gap:8px}
  .eng-grid :global(.eng-card){
    display:grid; gap:6px; min-width:0; padding:8px 11px;
    border:1px solid var(--rule-strong); border-radius:8px; background:var(--panel);
  }
  .eng-grid :global(.eng-card.thinking){background:color-mix(in srgb, var(--accent) 7%, var(--panel))}
  .eng-grid :global(.eng-head){display:flex; align-items:center; gap:9px; min-width:0}
  .eng-grid :global(.eng-title){display:flex; flex:1 1 auto; flex-direction:column; min-width:0; line-height:1.25}
  .eng-grid :global(.eng-name){color:var(--ink); font:650 14px var(--sans); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .eng-grid :global(.eng-side){color:var(--muted); font-size:12px}
  .eng-grid :global(.eng-eval){color:var(--ink); font:700 21px/1 var(--sans)}
  .eng-grid :global(.eng-thinking-label){margin-left:7px; color:var(--accent-soft-text); font:650 12px var(--sans)}
  .eng-grid :global(.eng-search-meter){width:100%; height:4px; accent-color:var(--accent)}
  .eng-grid :global(.eng-wdl){display:flex; align-items:center; gap:9px; color:var(--text); font-size:12.5px}
  .eng-grid :global(.wdl-bar){display:flex; flex:0 0 110px; height:11px; box-sizing:border-box; border:1px solid var(--border-input); border-radius:5px; overflow:hidden}
  .eng-grid :global(.wdl-bar .w){background:var(--eval-light)}
  .eng-grid :global(.wdl-bar .d){background:#9a9282}
  .eng-grid :global(.wdl-bar .l){background:var(--eval-dark)}
  .eng-grid :global(.eng-stats){display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:0}
  .eng-grid :global(.eng-stats div){display:grid; align-content:start; gap:1px}
  .eng-grid :global(.eng-stats .move-time){grid-column:4}
  .eng-grid :global(.eng-stats dt){color:var(--muted); font-size:12px}
  .eng-grid :global(.eng-stats dd){margin:0; color:var(--ink); font:600 13px var(--sans)}
  .eng-grid :global(.eng-line){margin:0; overflow:auto; white-space:nowrap; color:var(--text); font:500 13px/1.7 var(--mono); scrollbar-width:thin}
  .eng-grid :global(.eng-line:focus-visible){outline:2px solid var(--accent); outline-offset:2px}
  .eng-grid :global(.eng-line-label){margin-right:4px; color:var(--muted); font-size:12px}
  .eng-grid :global(.pv-move){white-space:nowrap}
  .eng-grid :global(.pv-move.first){color:var(--ink); font-weight:700}
  .eng-grid :global(.pv-num){margin-right:3px; color:var(--muted); font-weight:500}
  .eng-grid :global(.eng-empty){margin:0; color:var(--muted); font-size:13px}
  .eng-more{margin-top:0; padding:7px 14px; border-top:1px solid var(--rule)}
  .eng-more>summary, .more-options>summary{display:flex; align-items:center; gap:7px; min-height:28px; cursor:pointer; list-style:none; color:var(--text); font:600 12.5px var(--sans)}
  .eng-more>summary::-webkit-details-marker, .more-options>summary::-webkit-details-marker{display:none}
  .eng-more>summary::before, .more-options>summary::before{content:''; width:6px; height:6px; margin:0 2px; border:solid var(--muted); border-width:0 2px 2px 0; transform:rotate(-45deg); transition:transform .15s}
  .eng-more[open]>summary::before, .more-options[open]>summary::before{transform:rotate(45deg)}
  .eng-more>summary:hover, .more-options>summary:hover{color:var(--ink)}
  .eng-raw{display:grid; gap:8px; margin-top:4px; color:var(--text); font:11.5px/1.5 var(--mono); overflow-wrap:anywhere}
  .eng-raw :global(strong){display:block; color:var(--ink); font:600 12px var(--sans)}

  /* Move table (number | White | Black), as on Play */
  .arena-sidebar > .moves-block{display:flex; flex-direction:column; border-top:1px solid var(--rule)}
  .moves-heading{padding:10px 14px 0}
  .move-columns{display:grid; grid-template-columns:44px 1fr 1fr; color:var(--muted); font:500 12px/24px var(--sans); border-bottom:1px solid var(--rule)}
  .move-columns span{padding-left:12px}
  .move-columns span:first-child{text-align:center; padding-left:0}
  .arena-sidebar > .moves-block[hidden]{display:none}
  .review-bar{padding:7px 14px; border-bottom:1px solid var(--accent-tint-border); background:var(--accent-soft); color:var(--accent-soft-text); font:600 12.5px/1.4 var(--sans)}
  .move-list{position:relative; height:256px; flex:0 0 256px; overflow:auto; overscroll-behavior:contain; scroll-snap-type:y proximity; font:500 15px/1.4 var(--sans)}
  .move-list :global(.move-row){display:grid; grid-template-columns:44px minmax(0,1fr) minmax(0,1fr); height:32px; scroll-snap-align:start}
  .move-list :global(.move-row:nth-child(even)){background:color-mix(in srgb, var(--ink) 3%, var(--panel))}
  .move-list :global(.num){display:grid; place-items:center; color:var(--muted); font-size:13px; background:var(--panel-inset); border-right:1px solid var(--rule-soft)}
  .move-list :global(.san){display:flex; align-items:center; min-height:32px; padding:0 12px; border:0; border-radius:0; background:transparent; color:var(--ink); font:inherit; text-align:left; cursor:pointer}
  .move-list :global(.san.book){color:var(--muted)}
  .move-list :global(.san.empty){cursor:default; color:var(--muted)}
  .move-list :global(button.san:hover:not(.current)){background:var(--accent-soft)}
  .move-list :global(.san.current){background:color-mix(in srgb, var(--accent) 28%, var(--panel)); color:var(--ink); font-weight:700}
  .move-nav{display:flex; gap:4px; padding:6px 10px; border-top:1px solid var(--rule)}
  .move-nav button{display:grid; place-items:center; width:40px; height:34px; padding:0; border-radius:7px; border:1px solid var(--border-input); background:var(--panel-2); color:var(--ink)}
  .move-nav button:disabled{background:transparent; border-color:var(--rule)}
  .move-nav button:disabled svg{opacity:.3}
  .move-nav svg{width:18px; height:18px; fill:none; stroke:currentColor; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round}
  .move-nav .live-button{width:auto; margin-left:auto; padding:0 12px; color:var(--on-accent); font:650 13px var(--sans)}

  /* Log, PGN, runtime */
  .results-block, .runtime-block{border-top:1px solid var(--rule)}
  .arena-workspace:global([data-phase=running]) .runtime-block{display:none}
  .arena-workspace:global([data-match=none]) .results-block,
  .arena-workspace:global([data-match=none]) .search-details{display:none}
  .log{min-height:0; max-height:200px; overflow:auto; font:11.5px/1.45 var(--mono); color:var(--text)}
  .log :global(div){padding:3px 0; border-top:1px solid var(--rule)}
  .log :global(div.replayable){cursor:pointer}
  .log :global(div.replayable:hover){background:var(--accent-soft)}
  .results-block textarea{width:100%; min-height:72px; margin-top:7px; font:11.5px var(--mono)}
  .results-block .row{margin-top:8px}
  .runtime-badge{
    padding:7px 9px; border:1px solid var(--rule-strong); border-radius:6px; background:var(--card);
    color:var(--text); font:12px/1.45 var(--sans); overflow-wrap:anywhere;
  }
  .runtime-badge:global(.ready){color:var(--accent-soft-text); border-color:var(--accent-tint-border); background:var(--accent-soft)}
  .runtime-badge:global(.warn){color:var(--warn); border-color:color-mix(in srgb, var(--warn) 35%, var(--rule)); background:var(--warn-soft)}
  .runtime-block .capabilities-slot{padding:0; border-top:0}

  /* One compact chart at a time; every metric remains available. */
  .charts-panel{grid-column:1; padding:8px 12px; min-width:0}
  .charts-head{display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:3px}
  .charts-head .block-heading{margin:0; white-space:nowrap}
  .chart-tabs{display:flex; gap:2px; flex-wrap:wrap}
  .chart-tabs button{min-height:28px; padding:2px 8px; background:transparent; border-color:transparent; color:var(--muted); font:600 12px var(--sans)}
  .chart-tabs button[aria-pressed=true]{background:var(--accent-soft); border-color:var(--accent-tint-border); color:var(--accent-soft-text)}
  .chart-grid{min-width:0}
  .chart-card{display:none; min-width:0}
  .charts-panel:global([data-chart=evalChart]) .chart-card[data-chart=evalChart],
  .charts-panel:global([data-chart=timeChart]) .chart-card[data-chart=timeChart],
  .charts-panel:global([data-chart=npsChart]) .chart-card[data-chart=npsChart],
  .charts-panel:global([data-chart=rootChart]) .chart-card[data-chart=rootChart]{display:block}
  .chart-card :global(svg){display:block; width:100%; height:auto; cursor:crosshair}
  .chart-footer{display:flex; align-items:center; flex-wrap:wrap; gap:2px 12px; margin-top:3px}
  .chart-review{display:flex; flex:1; min-width:130px; align-items:center; gap:8px; color:var(--muted); font-size:12px}
  .chart-review input{flex:1; min-width:0; width:auto; min-height:24px; accent-color:var(--accent)}
  .chart-review button{min-height:28px; padding:2px 8px; font-size:12px}
  .root-review{display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px}
  .root-review select{min-width:0; flex:1}
  .chart-title{margin-bottom:2px; color:var(--muted); font-size:12px}
  .chart-legend{display:flex; flex-wrap:wrap; gap:10px; color:var(--text); font-size:12px}
  .chart-legend :global(.swatch){display:inline-block; width:9px; height:9px; margin-right:4px; border-radius:2px; vertical-align:-1px}

  /* Dev-only advanced settings (?dev) */
  .setup-pane :global(.advanced-settings){margin:0 14px 12px}
  .setup-pane .row{margin-top:8px}
  .setup-pane :global(.advanced-runtime){flex:1 1 100%; border:1px solid var(--rule); border-radius:7px; background:var(--card); padding:8px 10px}
  .setup-pane :global(.advanced-runtime summary){cursor:pointer; color:var(--muted); font-size:12px; font-weight:700}
  .setup-pane :global(.preset-note){flex:1 1 100%; color:var(--muted)}
  .setup-pane :global(.diag-block){display:grid; grid-template-columns:84px 1fr; gap:2px 10px; margin:4px 0; font-size:12px; align-items:baseline}
  .setup-pane :global(.diag-block .diag-label){color:var(--muted)}
  .setup-pane :global(.diag-block .diag-value){font-family:var(--mono); overflow-wrap:anywhere}

  @media (max-width:900px){
    .arena-workspace{grid-template-columns:minmax(0,1fr)}
    .arena-workspace .arena-sidebar{max-height:none; overflow:visible}
    .charts-panel{grid-column:auto}
    /* Setup first before the first game; running games keep the board first. */
    .arena-workspace:global([data-match=none]) .arena-sidebar,
    .arena-workspace:global([data-checkpoint=resumable]) .arena-sidebar{grid-row:1}
    /* Reserve the full fixed action bar, including its summary and safe area. */
    .arena-workspace:global(:not([data-phase=running])){padding-bottom:calc(var(--action-space) + env(safe-area-inset-bottom))}
    .arena-workspace :global(:is(input,select,textarea,button,summary)){scroll-margin-bottom:calc(var(--action-space) + env(safe-area-inset-bottom) + 16px)}
    .arena-actions{
      position:fixed; left:0; right:0; bottom:0; z-index:4; background:var(--panel);
      border-top:1px solid var(--border-input); padding:10px max(12px, calc((100vw - 650px) / 2)) calc(10px + env(safe-area-inset-bottom));
      box-shadow:0 -8px 24px #00000020;
    }
    .start-summary{display:block; margin-bottom:7px; color:var(--text); font:500 12px/1.4 var(--sans)}
    .start-summary :global(strong){display:block; color:var(--ink); font-weight:650}
    .chart-tabs button{min-height:36px}
    .chart-review{min-width:100%}
    .move-nav button{height:40px; flex:1 1 0; width:auto}
    .move-nav .live-button{flex:1.6 1 0}
  }
  @media (max-width:600px){
    /* Two 10px lanes; preserve the existing chart/control baseline. */
    .arena-workspace{--strip-h:34px}
    .board-panel{padding:0; border:0; background:transparent; gap:2px}
    .board-with-eval{grid-template-columns:minmax(0,1fr); gap:4px}
    .board-shell{grid-row:1; width:calc(100% - 25px); justify-self:center}
    .arena-eval-bar{grid-row:2; grid-template-columns:1fr; grid-template-rows:1fr 1fr; height:24px; margin:0 12px 0 44px; gap:2px; border-radius:2px}
    .arena-eval-bar .eval-fill{inset:0 auto 0 0; height:auto; width:var(--pct)}
    .arena-eval-bar .eval-cap{inset:0 auto 0 0; width:4px; height:auto}
    .arena-eval-bar .eval-lane-label{top:50%; left:-33px; right:auto; width:28px; transform:translateY(-50%); background:none; font-size:10px; line-height:10px; font-family:var(--mono); font-variant-numeric:slashed-zero}
    .arena-eval-bar :global(.eval-midline){inset:0 auto 0 50%; width:1px; height:auto}
    .arena-eval-bar .eval-fill::after{left:auto; right:0; top:0; bottom:0; width:1px; height:auto}
    .eval-end{display:block; position:absolute; top:50%; transform:translateY(-50%); color:var(--muted); font:650 10px/12px var(--sans)}
    .eval-end.w{left:-44px}
    .eval-end.b{right:-12px}
    .eval-source{gap:8px; min-height:18px; align-items:center; line-height:17px}
    #whiteSideLabel{order:1}
    .eval-source :global(.eval-reading){gap:4px; flex-shrink:0}
    .arena-workspace :global(#ground coords){font-size:11px}
    .player-strip{gap:8px}
    .arena-workspace :global(.player-strip .arena-avatar){flex-basis:30px; width:30px; height:30px; font-size:11.5px}
    .player-strip :global(.strip-id strong){font-size:14px}
    .player-strip :global(.strip-id small){font-size:12px}
    .player-strip :global(.strip-clock){min-width:58px; font-size:13px}
    .status-block{padding:10px 12px}
    .arena-workspace .section-block{padding:11px 12px}
    .setup-pane :global(.seat-head), .setup-pane :global(.seat-row){grid-template-columns:92px minmax(0,1fr) minmax(0,1fr) auto}
    .setup-pane :global(.seat-row .seat-name){font-size:12px}
    .setup-pane :global(.seat-row .seat-name .arena-avatar){width:22px; height:22px; flex-basis:22px}
    .setup-pane :global(.seat-row select){padding-left:8px; font-size:13px}
    .setup-pane :global(.seat-strength-wrap){grid-column:2 / -1}
    .setup-pane :global(.seat-row select), .arena-workspace select, .arena-workspace input[type=number]{min-height:44px}
    .time-grid label{font-size:12px}
    .time-grid select{padding-left:8px; font-size:12px}
    .scoreboard :global(.sb-points){font-size:28px}
    .scoreboard :global(.sb-name strong){font-size:12px}
    .scoreboard :global(.sb-player .arena-avatar){display:none}
    .ghost-button, .stop-button{min-height:44px}
  }
  @media (prefers-reduced-motion: reduce){
    .arena-eval-bar .eval-fill, .arena-eval-bar .eval-cap{transition:none}
    .player-strip :global(.strip-thinking)::before, .live-dot{animation:none}
  }
</style>
