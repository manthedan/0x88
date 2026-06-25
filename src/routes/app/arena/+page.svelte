<script lang="ts">
  import { onMount } from 'svelte';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  const title = "0x88 Chess — engine arena";
  const description = "Run chess engine tournaments in your browser: head-to-head matches, round-robins, and gauntlets with standings, Elo estimates, and per-game charts.";
  const devMode = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('dev');
  onMount(() => {
    let cleanup: () => void = () => undefined;
    let mounted = true;
    void import('../../../lc0/arenaBrowser').then((module) => {
      if (!mounted) return;
      cleanup = module.mountArenaBrowser();
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

<SiteHeader pageTitle="Arena" />
<main id="main">
  <section class="panel" aria-label="Board">
    <div id="blackSideLabel" class="side-label" aria-label="Black side"></div>
    <div class="eval-chip-row"><span id="whiteEngineChip" class="eval-chip" style="display:none"></span><span id="blackEngineChip" class="eval-chip" style="display:none"></span></div>
    <div class="board-with-evals">
      <div id="whiteEngineEvalBar" class="engine-eval-bar empty" aria-label="White engine evaluation bar"></div>
      <div class="board-shell"><div id="ground"></div></div>
      <div id="blackEngineEvalBar" class="engine-eval-bar empty" aria-label="Black engine evaluation bar"></div>
    </div>
    <div id="whiteSideLabel" class="side-label" aria-label="White side"></div>
    <div id="reviewBar" class="review-bar" hidden>
      <span id="reviewLabel"></span>
      <span class="review-nav">
        <button id="revStart" type="button" title="First move">|◀</button>
        <button id="revPrev" type="button" title="Previous move (←)">◀</button>
        <button id="revNext" type="button" title="Next move (→)">▶</button>
        <button id="revEnd" type="button" title="Last move">▶|</button>
        <button id="revLive" type="button" class="primary" title="Back to the live game (Esc)">⏵ Live</button>
      </span>
    </div>
    <div id="gameMoves" class="movestrip" hidden></div>
    <div id="pairing">Select engines and start a tournament.</div>
    <div id="message" aria-live="polite">Loading model…</div>
    <div id="downloadProgress" class="model-load-progress" hidden></div>
    <section id="chartsPanel" aria-label="Game charts" hidden>
      <div class="chart-grid">
        <div class="chart-card"><div class="chart-title">White win % per move</div><div id="evalChart"></div></div>
        <div class="chart-card"><div class="chart-title">Move time (ms)</div><div id="timeChart"></div></div>
        <div class="chart-card"><div class="chart-title">Speed (NPS)</div><div id="npsChart"></div></div>
        <div class="chart-card"><div class="chart-title" id="rootChartTitle">LC0 root visits</div><div id="rootChart"></div></div>
      </div>
      <div id="chartLegend" class="chart-legend"></div>
    </section>
    <details class="runtime-details"><summary>Runtime status</summary>
      <div id="runtimeBadge" class="runtime-badge">Runtime: detecting browser isolation and GPU…</div>
    </details>
  </section>
  <section class="panel app-sidebar" aria-label="Tournament">
    <details class="section-block" open>
      <summary>Matchup</summary>
      <select id="seatA" hidden></select><select id="seatB" hidden></select>
      <div class="row">
        <div class="field"><label for="tournamentModeSelect">Mode</label>
          <select id="tournamentModeSelect"><option value="match">Match (Engine 1 vs 2)</option><option value="round-robin">Round robin</option><option value="gauntlet">Gauntlet (Engine 1 challenges)</option></select></div>
        <button id="addSeat" type="button" title="Add another engine seat">+ Add engine</button>
      </div>
      <div id="arenaSeatList" class="engine-list" aria-label="Arena engine selectors"></div>
      <div class="matchup-note small" id="matchupNote">Pick an engine and strength for each seat; colors alternate each game.</div>
    </details>
    <details class="section-block" open>
      <summary>Time control</summary>
      <div class="row">
        <div class="field"><label for="budgetModeSelect">Budget</label>
          <select id="budgetModeSelect"><option value="fixed">Fixed visits/depth</option><option value="movetime" selected>Equal movetime</option></select></div>
        <div class="field" id="movetimeField"><label for="movetimeInput">Movetime ms</label>
          <input id="movetimeInput" type="number" min="10" max="60000" step="50" value="500" /></div>
        <div class="field"><label for="gamesInput">Games per opening</label>
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
      </div>
      <div class="row">
        <div class="field"><label for="startingPositionSelect">Suite</label>
          <select id="startingPositionSelect"><option value="start">Start position</option><option value="built-in">Built-in opening suite</option><option value="custom">Custom positions / replays</option></select></div>
        <div class="field wide" id="openingTextField" hidden><label for="openingText">Custom openings (one per line: FEN, UCI moves, PGN/SAN, or Name | ...)</label>
          <textarea id="openingText" spellcheck="false" placeholder="Ruy Lopez | e2e4 e7e5 g1f3 b8c6 f1b5&#10;Italian Game | 1. e4 e5 2. Nf3 Nc6 3. Bc4&#10;Sicilian FEN | rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2"></textarea></div>
      </div>
      <div id="openingInfo" class="small">Start position only.</div>
      <div class="row">
        <button id="start" class="primary" type="button" disabled>Start match</button>
        <button id="stop" type="button" disabled>Stop</button>
      </div>
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
      <div id="cacheInfo"></div>
      <div id="searchTelemetryInfo"></div>
      <div id="runtimeAuditInfo"></div>
    </div>
    {/if}
    <details class="section-block" open>
      <summary>Engine outputs</summary>
      <div id="engineEvalInfo" class="eval-grid" aria-label="Engine evaluation outputs"><div class="eval-card"><div class="eval-card-head"><span class="eval-card-name">Waiting…</span></div><div class="eval-card-eval">Engine outputs: waiting for a move…</div></div></div>
    </details>
    <details class="section-block" open>
      <summary>Result</summary>
      <div id="matchScore" class="small" aria-live="polite">No games played yet.</div>
      <div class="log" id="log"></div>
      <pre id="benchResult" class="small" hidden></pre>
      <div class="row"><button id="exportPgn" type="button">Export PGN</button></div>
      <textarea id="pgnOut" spellcheck="false" placeholder="game PGNs appear here after Export PGN"></textarea>
    </details>
  </section>
</main>

<style>
  :root{--acc-w:#2f6e7d;--acc-b:#b15c2b}
  main{
    display:grid;
    grid-template-columns:minmax(0,1fr) 392px;
    gap:24px; align-items:start; justify-content:center; padding:26px 28px 56px;
    max-width:1280px; margin:0 auto;
  }
  .app-sidebar{
    position:sticky; top:72px;
    max-height:calc(100vh - 84px); overflow-y:auto;
  }
  :global(.board-with-evals){
    display:grid; grid-template-columns:18px minmax(0,1fr) 18px;
    gap:10px; align-items:stretch; margin:0 auto;
  }
  :global(.engine-eval-bar){
    position:relative; min-height:100%;
    border:1px solid var(--rule); border-radius:4px;
    background:#3a3a3a; overflow:hidden;
  }
  :global(.engine-eval-bar .eval-fill){
    position:absolute; left:0; right:0; bottom:0;
    height:50%; background:#f4f1e8; transition:height .25s;
  }
  :global(.engine-eval-bar .eval-midline){
    position:absolute; left:0; right:0; top:50%;
    height:1px; background:var(--accent); opacity:.6; border:0;
  }
  :global(.engine-eval-bar .eval-bar-caption, .engine-eval-bar .eval-bar-value){display:none}
  :global(.engine-eval-bar.empty .eval-fill){height:50%}
  :global(#whiteEngineEvalBar){border:2px solid var(--acc-w)}
  :global(#blackEngineEvalBar){border:2px solid var(--acc-b)}
  :global(.eval-chip-row){
    display:flex; align-items:flex-end; margin:0 auto 5px;
    gap:8px; min-height:18px;
  }
  :global(.eval-chip){
    display:inline-flex; align-items:center; gap:3px;
    font-family:var(--mono); font-size:9px; font-weight:700; letter-spacing:.02em;
    color:#fff; border-radius:5px; padding:2px 8px; line-height:1.35;
    white-space:nowrap; overflow:hidden; max-width:44%;
  }
  :global(.eval-chip span){overflow:hidden; text-overflow:ellipsis}
  :global(#whiteEngineChip){background:var(--acc-w)}
  :global(#blackEngineChip){background:var(--acc-b); margin-left:auto}
  :global(.engine-logo){width:14px; height:14px; object-fit:contain; border-radius:3px; vertical-align:middle; flex:0 0 auto}
  :global(.engine-logo-placeholder){display:inline-block; width:14px; height:14px; border-radius:3px; background:var(--rule-soft); flex:0 0 auto; vertical-align:middle}
  :global(.eval-chip .engine-logo), :global(.eval-chip .engine-logo-placeholder){width:11px; height:11px; border-radius:2px}
  :global(.side-label .engine-logo), :global(.side-label .engine-logo-placeholder){width:16px; height:16px}
  :global(.engine-eval-bar.thinking){box-shadow:0 0 0 2px rgba(159,174,111,.6)}
  :global(.side-label){
    display:flex; justify-content:space-between; align-items:center;
    gap:10px; margin:6px 0; min-height:40px;
    padding:10px 14px; border:1px solid #e6decc; border-left:3px solid var(--rule-strong);
    border-radius:11px; background:var(--panel); font-size:13px;
  }
  :global(.side-label .side-main){display:flex; flex:1; min-width:0; flex-wrap:nowrap; align-items:center; gap:8px}
  :global(.side-label .color){
    flex:0 0 auto; font-size:9px; text-transform:uppercase; letter-spacing:.07em;
    color:var(--muted); font-weight:700;
    border:1px solid var(--rule); border-radius:4px; padding:2px 5px; background:var(--bg);
  }
  :global(.side-label .engine){flex:0 0 auto; font-weight:700; font-size:14px}
  :global(.side-label .side-eval){
    margin-left:auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    font-family:var(--mono); font-size:11px; color:var(--muted); font-weight:600;
  }
  :global(.side-label .turn){
    flex:0 0 auto; font-family:var(--mono); font-size:10px; font-weight:700;
    color:var(--accent); white-space:nowrap; text-transform:uppercase; letter-spacing:.04em;
    background:var(--soft); border:1px solid #cbd8a6; border-radius:999px; padding:2px 8px;
  }
  :global(.side-label.active){background:var(--soft); border-color:#cbd8a6; border-left-color:var(--accent)}
  :global(.side-label.active .side-eval){color:var(--accent); font-weight:700}
  :global(#whiteSideLabel .color){color:var(--acc-w); border-color:var(--acc-w); background:#fff}
  :global(#blackSideLabel .color){color:var(--acc-b); border-color:var(--acc-b); background:#fff}
  :global(.engine-row .row-strength){flex:0 0 76px; width:76px}
  :global(.engine-row .seat-name){
    flex:0 0 56px; color:var(--muted); font-size:11px;
    text-transform:uppercase; letter-spacing:.04em; font-weight:700;
  }
  :global(.seat-row .seat-name){flex:0 0 auto; white-space:nowrap; font-size:10px}
  :global(.seat-row .arrow){display:none}
  :global(.seat-row .row-strength){flex:0 0 64px; width:64px}
  :global(.matchup-note){margin-top:8px; color:var(--muted)}
  .section-block{margin-top:0; padding-top:20px}
  .section-block:first-child{padding-top:0}
  .section-block>summary{
    cursor:pointer; font-size:11px; text-transform:uppercase;
    letter-spacing:.12em; color:var(--muted-2); font-weight:500;
    font-family:var(--mono); list-style:none; user-select:none; padding:0 0 10px; transition:color .12s;
  }
  .section-block>summary::-webkit-details-marker{display:none}
  .section-block>summary::before{
    content:"\25B8"; display:inline-block; margin-right:6px;
    font-size:10px; transition:transform .15s; color:var(--muted-2);
  }
  .section-block[open]>summary::before{transform:rotate(90deg)}
  .section-block>summary:hover{color:var(--ink)}
  .section-block[open]>summary{margin-bottom:0}
  .section-block[open]>summary::before{color:var(--accent)}
  :global(#matchScore){
    margin-top:4px;
    font-family:var(--mono); font-size:14px; font-weight:700; color:var(--ink);
    padding:8px 10px; border:1px solid var(--rule); border-radius:7px; background:var(--soft);
  }
  :global(input[type=number]){width:80px; font-family:var(--mono); padding:6px 8px; border:1px solid var(--rule); border-radius:6px}
  :global(select){max-width:100%}
  .row{margin-top:12px}
  .field{margin-top:12px}
  :global(.field.wide){flex:1 1 100%}
  :global(.field.wide textarea){margin-top:0; min-height:86px}
  :global(.engines){display:grid; gap:4px; margin-top:8px}
  :global(.engines label){font-size:13px; display:flex; gap:8px; align-items:center}
  :global(table){width:100%; border-collapse:collapse; font-size:13px; margin-top:8px}
  :global(th, td){text-align:left; padding:5px 8px; border-bottom:1px solid var(--rule)}
  :global(th){color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase}
  :global(td.num, th.num){text-align:right; font-family:var(--mono)}
  :global(tr.leader td){background:var(--soft)}
  :global(.log){font-family:var(--mono); font-size:12px; height:160px; overflow:auto; margin-top:8px}
  :global(.log div){padding:2px 0; border-top:1px solid var(--rule)}
  :global(.log div.replayable){cursor:pointer}
  :global(.log div.replayable:hover){background:var(--soft)}
  :global(.review-bar){
    display:flex; align-items:center; justify-content:space-between;
    gap:10px; margin:8px auto 0; padding:7px 10px;
    border:1px solid #cbd8a6; border-radius:7px; background:var(--soft);
    font-size:12px; font-weight:600;
  }
  :global(.review-bar .review-nav){display:flex; gap:5px}
  :global(.review-bar button){padding:4px 9px; font-size:12px}
  :global(.movestrip){
    margin-top:12px; font-family:var(--mono); font-size:14px;
    line-height:1.9; height:76px; overflow:auto;
    padding:10px 14px; border:1px solid #e6decc; border-radius:var(--radius-sm); background:var(--panel-inset);
  }
  :global(.movestrip .num){color:var(--muted)}
  :global(.movestrip .mv){cursor:pointer; padding:1px 3px; border-radius:4px}
  :global(.movestrip .mv:hover){background:var(--soft)}
  :global(.movestrip .mv.current){background:var(--accent); color:white}
  :global(.chart-card svg){cursor:pointer}
  :global(#pairing){font-size:14px; margin-top:14px; font-weight:600; color:var(--ink); font-family:var(--serif)}
  :global(#message){
    margin-top:6px; padding:8px 12px; border:1px solid var(--rule-strong);
    border-radius:8px; background:var(--panel-inset); color:var(--text-soft);
    font-family:var(--mono); font-size:12px; line-height:1.5;
  }
  :global(.model-load-progress){
    margin-top:6px; padding:8px; border:1px solid var(--rule);
    border-radius:var(--radius-sm); background:var(--panel);
  }
  :global(.model-load-progress progress), :global(.eval-card progress){
    width:100%; height:9px; accent-color:var(--accent);
  }
  :global(.loading-progress-row), :global(.search-progress-row){display:grid; gap:3px; margin:4px 0}
  :global(.dl-label), :global(.search-progress-text){font-family:var(--mono); font-size:11px; color:var(--muted)}
  :global(.eval-grid){display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:8px; margin-top:8px; min-height:78px}
  :global(.eval-card){
    padding:8px 10px; border:1px solid var(--rule); border-radius:var(--radius-sm);
    background:var(--panel);
    display:flex; flex-direction:column; gap:3px;
    height:88px; overflow:hidden;
  }
  :global(.eval-card-head){
    display:flex; justify-content:space-between; align-items:center; gap:8px; flex:0 0 auto;
  }
  :global(.eval-card-name){
    color:var(--ink); font-family:var(--sans); font-size:12px; font-weight:600;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  :global(.eval-card-eval){
    font-family:var(--mono); font-size:12px; font-weight:700; color:var(--ink-soft);
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:0 0 auto;
  }
  :global(.eval-card-stats){
    font-family:var(--mono); font-size:10px; color:var(--muted); flex:0 0 auto;
    display:flex; flex-wrap:wrap; gap:2px 6px;
  }
  :global(.eval-card-scroll){
    font-family:var(--mono); font-size:10px; color:var(--muted);
    flex:1 1 auto; overflow:hidden; line-height:1.4;
    opacity:.6; transition:opacity .15s; min-height:0;
  }
  :global(.eval-card:hover .eval-card-scroll){
    overflow:auto; opacity:1;
  }
  :global(.eval-card-raw){white-space:pre-wrap; overflow-wrap:anywhere}
  :global(.eval-card-pv){color:var(--text-soft)}
  :global(.eval-card .eval-status){font-family:var(--mono); font-size:10px; color:var(--accent); font-weight:600; white-space:nowrap}
  :global(.eval-card.active){border-color:#cbd8a6; background:var(--soft)}
  :global(.runtime-badge){
    margin-top:8px; padding:7px 8px; border:1px solid var(--rule);
    border-radius:6px; background:white; font-family:var(--mono);
    font-size:11px; color:var(--muted); overflow-wrap:anywhere;
  }
  :global(.runtime-badge.ready){color:var(--accent); border-color:#cbd8a6; background:var(--soft)}
  :global(.runtime-badge.warn){color:#a5461b; border-color:#e5b38d; background:#fff4e8}
  :global(.advanced-runtime){
    flex:1 1 100%; border:1px solid var(--rule); border-radius:7px;
    background:#fff; padding:8px 10px;
  }
  :global(.advanced-runtime summary){cursor:pointer; color:var(--muted); font-size:12px; font-weight:700}
  :global(.advanced-runtime[open] summary){margin-bottom:8px}
  :global(.preset-note){flex:1 1 100%; margin-top:-2px; color:var(--muted)}
  :global(textarea){width:100%; min-height:80px; font-family:var(--mono); font-size:11px; padding:8px; margin-top:8px}
  :global(.chart-grid){display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px}
  :global(.chart-card){border:1px solid var(--rule); border-radius:6px; padding:6px 8px; background:#fff}
  :global(.chart-card svg){display:block; width:100%; height:auto}
  :global(.chart-title){font-size:11px; color:var(--muted); margin-bottom:2px}
  :global(.chart-legend){display:flex; gap:14px; margin-top:4px; font-size:11px; color:var(--muted)}
  :global(.chart-legend .swatch){display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:4px; vertical-align:-1px}
  :global(.seat-inactive){opacity:.45}
  :global(.seat-remove){
    margin-left:4px; border:1px solid var(--rule); background:none;
    border-radius:4px; cursor:pointer; color:var(--muted);
    line-height:1; padding:2px 7px;
  }
  :global(.seat-remove:hover){color:#a5461b; border-color:#a5461b}
  :global(.standings){border-collapse:collapse; font-size:12px; margin-top:6px; width:100%}
  :global(.standings th, .standings td){padding:3px 8px; text-align:right; border-top:1px solid var(--rule); font-variant-numeric:tabular-nums}
  :global(.standings th){color:var(--muted); font-weight:600; border-top:none}
  :global(.standings th:nth-child(2), .standings td:nth-child(2)){text-align:left; width:99%}
  :global(.diag-block){display:grid; grid-template-columns:84px 1fr; gap:2px 10px; margin:4px 0; font-size:12px; align-items:baseline}
  :global(.diag-block .diag-label){color:var(--muted)}
  :global(.diag-block .diag-value){font-family:var(--mono); font-variant-numeric:tabular-nums; overflow-wrap:anywhere}
  @media (max-width:900px){
    main{grid-template-columns:1fr; padding:18px}
    .app-sidebar{position:static; max-height:none; overflow-y:visible}
  }
  @media (max-width:900px){.chart-grid{grid-template-columns:1fr}}
  @media (prefers-reduced-motion: reduce){
    :global(.eval-fill){transition:none}
    :global(.engine-eval-bar.thinking){box-shadow:none}
  }
</style>
