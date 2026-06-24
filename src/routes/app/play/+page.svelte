<script lang="ts">
  import { onMount } from 'svelte';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  const title = "0x88 Chess — play chess vs an engine in your browser";
  const description = "Play chess against Leela Chess Zero, Stockfish, and four more engines running entirely in your browser. Five strength levels, takebacks, and PGN export.";
  onMount(() => {
    let cleanup: () => void = () => undefined;
    let mounted = true;
    void import('../../../lc0/playBrowser').then((module) => {
      if (!mounted) return;
      cleanup = module.mountPlayBrowser();
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

<SiteHeader pageTitle="Play" />
<main id="main">
  <section class="panel" aria-label="Board">
    <div class="board-shell"><div id="ground"></div></div>
    <div id="status" aria-live="polite">Loading...</div>
    <div id="restartBanner" class="restart-banner" hidden>
      <span id="restartMessage" class="small"></span>
      <div class="row" style="margin-top:6px">
        <button id="confirmRestart" class="primary" type="button">Start new game</button>
        <button id="dismissRestart" type="button">Keep current game</button>
      </div>
    </div>
    <div id="promoPicker" hidden></div>
    <div id="moveList" class="move-panel"></div>
  </section>
  <section class="panel app-sidebar" aria-label="Game controls">
    <h2>Opponent</h2>
    <div class="field"><label for="engineSelect">Engine</label>
      <select id="engineSelect"></select></div>
    <div class="field"><label for="levelSelect">Strength</label>
      <select id="levelSelect"></select></div>
    <div id="maia3Controls" hidden>
      <div class="field"><label for="maia3Elo">Maia3 rating <span id="maia3EloValue">1500</span></label>
        <input id="maia3Elo" type="range" min="600" max="2600" step="100" value="1500" /></div>
      <div class="field"><label for="maia3Style">Maia3 move style</label>
        <select id="maia3Style">
          <option value="sample">Human sampling</option>
          <option value="argmax">Deterministic top move</option>
        </select></div>
      <div class="field" id="maia3TemperatureField"><label for="maia3Temperature">Temperature</label>
        <input id="maia3Temperature" type="number" min="0.01" max="5" step="0.05" value="1" /></div>
      <div class="field" id="maia3TopPField"><label for="maia3TopP">Top-p</label>
        <input id="maia3TopP" type="number" min="0.01" max="1" step="0.05" value="1" /></div>
    </div>
    <div id="levelCaption" class="small"></div>
    <div id="engineCaution" class="small" hidden></div>
    <div id="engineNote" hidden></div>
    <div id="dlProgress" hidden><progress></progress><div class="dl-label small"></div></div>
    <h2>Game</h2>
    <div class="field"><label for="colorSelect">You play</label>
      <select id="colorSelect">
        <option value="white">White</option>
        <option value="black">Black</option>
        <option value="random">Random</option>
      </select></div>
    <div class="row">
      <button id="newGame" class="primary" type="button">New game</button>
      <button id="takeback" type="button" disabled>Takeback</button>
      <button id="resign" type="button" disabled>Resign</button>
      <button id="flip" type="button" title="Flip board">&#x21c5; Flip</button>
    </div>
    <div class="row">
      <button id="exportPgn" type="button">Show PGN</button>
      <button id="copyPgn" type="button">Copy PGN</button>
    </div>
    <div id="pgnOut"></div>
    <p class="small">Engines run entirely in your browser &#x2014; nothing is sent to a server. The first move against a new engine downloads it (a few MB for CPU engines, up to a few hundred MB for the big Lc0 nets).</p>
  </section>
</main>

<style>
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
  :global(#status){
    margin-top:12px; padding:13px 16px;
    border:1px solid var(--accent-tint-border); border-radius:12px;
    background:var(--accent-soft); font-size:14px; color:var(--accent-soft-text); font-weight:500;
  }
  :global(#status.over){border-color:var(--accent); font-weight:700}
  :global(button.danger){background:var(--warn); border-color:var(--warn); color:#fff; font-weight:700}
  :global(#levelCaption){margin-top:4px}
  :global(#engineCaution){
    margin-top:6px; padding:7px 9px;
    border:1px dashed var(--rule); border-radius:6px; background:#fff;
  }
  :global(#engineNote){
    margin-top:8px; padding:8px 10px;
    border:1px solid var(--rule); border-radius:6px;
    background:#fff; font-family:var(--mono); font-size:12px;
  }
  :global(#engineNote.warn){color:var(--warn); border-color:var(--warn)}
  :global(#dlProgress){margin-top:8px}
  :global(#dlProgress progress){width:100%; height:10px; accent-color:var(--accent)}
  :global(#dlProgress .dl-label){margin-top:2px; font-family:var(--mono)}
  :global(#promoPicker){margin-top:10px; display:flex; gap:6px; flex-wrap:wrap}
  :global(#moveList){
    margin-top:12px; font-family:var(--mono); font-size:14px;
    line-height:1.9; max-height:300px; overflow:auto;
    padding:10px 14px; border:1px solid #e6decc; border-radius:var(--radius-sm); background:var(--panel-inset);
  }
  :global(#moveList .num){color:var(--muted)}
  :global(#moveList .placeholder){color:var(--muted)}
  :global(#pgnOut){
    font-family:var(--mono); font-size:12px;
    white-space:pre-wrap; overflow-wrap:anywhere; margin-top:8px;
  }
  :global(#pgnOut:not(:empty)){
    padding:8px; border:1px solid var(--rule);
    border-radius:6px; background:#fff;
  }
  :global(.restart-banner){
    margin-top:10px; padding:10px 12px;
    border:1px solid var(--accent); border-radius:6px; background:var(--soft);
  }
  :global(#maia3Elo){
    -webkit-appearance:none; appearance:none; width:100%; height:5px;
    border-radius:3px; background:var(--rule-strong); outline:none;
  }
  :global(#maia3Elo::-webkit-slider-thumb){
    -webkit-appearance:none; appearance:none;
    width:18px; height:18px; border-radius:50%; background:#fff;
    border:2px solid var(--accent); cursor:pointer;
    box-shadow:0 2px 6px rgba(80,55,25,.25);
  }
  :global(#maia3Elo::-moz-range-thumb){
    width:18px; height:18px; border-radius:50%; background:#fff;
    border:2px solid var(--accent); cursor:pointer;
    box-shadow:0 2px 6px rgba(80,55,25,.25);
  }
  @media (max-width:900px){
    main{grid-template-columns:1fr; padding:18px}
    .app-sidebar{position:static; max-height:none; overflow-y:visible}
  }
</style>
