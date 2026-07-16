<script lang="ts">
  import { onMount } from 'svelte';
  import { afterNavigate } from '$app/navigation';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  import BrowserCapabilities from '$lib/components/BrowserCapabilities.svelte';
  let mountedUrl: URL | null = null;
  afterNavigate(({ to }) => {
    if (mountedUrl && to?.url.pathname === mountedUrl.pathname && to.url.search !== mountedUrl.search) location.reload();
  });
  const title = "0x88 Chess — play chess vs an engine in your browser";
  const description = "Play chess against Leela Chess Zero, Centipawn, Stockfish, Stormphrax, and other opponents running entirely in your browser. Five strength levels, takebacks, and PGN export.";
  onMount(() => {
    mountedUrl = new URL(location.href);
    let cleanup: () => void = () => undefined;
    let mounted = true;
    void import('../../../lc0/playBrowser').then((module) => {
      if (!mounted) return;
      cleanup = module.mountPlayBrowser();
    }).catch((error) => {
      if (!mounted) return;
      console.error('[play] failed to load page controller', error);
      const node = document.getElementById('status');
      if (node) node.textContent = `Page failed to initialize: ${error instanceof Error ? error.message : String(error)}`;
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
  <section class="panel board-panel" aria-label="Board">
    <div class="board-shell"><div id="ground"></div></div>
    <div class="status-row">
      <div id="status" aria-live="polite">Loading...</div>
      <button id="flip" class="board-action" type="button" title="Flip board">&#x21c5; Flip</button>
    </div>
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
    <details class="section-block" open>
      <summary>Opponent</summary>
      <div class="opponent-grid">
        <div class="field"><label for="engineSelect">Engine</label>
          <select id="engineSelect"></select></div>
        <div class="field"><label for="levelSelect">Strength</label>
          <select id="levelSelect"></select></div>
      </div>
      <div id="maia3Controls" hidden>
        <div class="maia-grid">
          <div class="field rating-field"><label for="maia3Elo">Rating <span id="maia3EloValue">1500</span></label>
            <div class="range-control"><input id="maia3Elo" type="range" min="600" max="2600" step="100" value="1500" /></div></div>
          <div class="field"><label for="maia3Style">Move selection</label>
            <select id="maia3Style">
              <option value="sample">Human sampling</option>
              <option value="argmax">Top move</option>
            </select></div>
        </div>
        <details class="advanced-settings">
          <summary>Sampling settings</summary>
          <div class="sampling-grid">
            <div class="field" id="maia3TemperatureField"><label for="maia3Temperature">Temperature</label>
              <input id="maia3Temperature" type="number" min="0.01" max="5" step="0.05" value="1" /></div>
            <div class="field" id="maia3TopPField"><label for="maia3TopP">Top-p</label>
              <input id="maia3TopP" type="number" min="0.01" max="1" step="0.05" value="1" /></div>
          </div>
        </details>
      </div>
      <div id="levelCaption" class="small setup-note"></div>
      <div id="engineCaution" class="small" hidden></div>
      <div id="engineNote" role="status" aria-live="polite" hidden></div>
      <button id="retryEngine" type="button" hidden>Retry engine</button>
    </details>

    <section class="section-block game-section" aria-labelledby="gameHeading">
      <h2 id="gameHeading" class="section-heading">Game</h2>
      <div class="game-setup">
        <div class="field"><label for="colorSelect">You play</label>
          <select id="colorSelect">
            <option value="white">White</option>
            <option value="black">Black</option>
            <option value="random">Random</option>
          </select></div>
        <button id="newGame" class="primary" type="button">New game</button>
      </div>
      <div class="game-actions" aria-label="Current game actions">
        <button id="takeback" type="button" disabled>Takeback</button>
        <button id="resign" type="button" disabled>Resign</button>
      </div>
      <div id="dlProgress" hidden><progress></progress><div class="dl-label small"></div></div>
      <div id="progressAnnouncement" class="visually-hidden" role="status" aria-live="polite"></div>
    </section>

    <details class="section-block record-section">
      <summary>Game record</summary>
      <div class="record-actions">
        <button id="exportPgn" type="button">Show PGN</button>
        <button id="copyPgn" type="button">Copy PGN</button>
      </div>
      <div id="pgnOut"></div>
    </details>

    <div class="capabilities-slot"><BrowserCapabilities /></div>
  </section>
</main>

<style>
  main{
    display:grid; grid-template-columns:minmax(0,1fr) minmax(370px,420px);
    gap:18px; align-items:start; justify-content:center;
    max-width:1136px; margin:0 auto; padding:16px 24px 48px;
  }
  .board-panel{min-width:0; padding:12px}
  .app-sidebar{
    position:sticky; top:74px; padding:0; overflow:hidden;
    border-color:var(--rule-strong);
  }
  .section-block{padding:0 12px 12px; border:0; border-bottom:1px solid var(--rule)}
  details.section-block:not([open]){padding-bottom:0}
  .section-block>summary,.section-heading{
    margin:0 -12px 10px; padding:12px; list-style:none;
    font-family:var(--mono); font-size:10px; font-weight:650; line-height:1.2;
    letter-spacing:.1em; text-transform:uppercase; color:var(--muted-2);
  }
  .section-block>summary{cursor:pointer; user-select:none}
  .section-heading{margin-bottom:4px; padding:12px 12px 8px 26px}
  details.section-block:not([open])>summary{margin-bottom:0}
  .section-block>summary::-webkit-details-marker{display:none}
  .section-block>summary::before{content:"▸"; display:inline-block; width:14px; color:var(--accent)}
  .section-block[open]>summary::before{content:"▾"}
  .field{min-width:0; margin-top:0}
  .opponent-grid{display:grid; grid-template-columns:minmax(0,1.45fr) minmax(110px,.75fr); gap:8px; align-items:end}
  :global(.opponent-grid:has(> .field[hidden])){grid-template-columns:1fr}
  .maia-grid{display:grid; grid-template-columns:minmax(0,1fr) minmax(142px,.8fr); gap:8px; align-items:end; margin-top:9px}
  .maia-grid>.field{display:grid; grid-template-rows:auto 37px; align-items:stretch}
  .range-control{
    display:flex; align-items:center; min-width:0; height:37px; padding:0 8px;
    border:1px solid var(--border-input); border-radius:var(--radius-input);
    background:var(--input-bg);
  }
  .sampling-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px}
  .advanced-settings{margin-top:8px; padding-top:8px; border-top:1px solid var(--rule)}
  .advanced-settings>summary{
    cursor:pointer; color:var(--muted); font-family:var(--mono); font-size:10px;
    letter-spacing:.04em;
  }
  .advanced-settings[open]>summary{margin-bottom:8px}
  .setup-note{margin-top:8px; line-height:1.45}
  .game-setup{display:grid; grid-template-columns:minmax(112px,.9fr) minmax(0,1.1fr); gap:8px; align-items:end}
  .game-setup>button,.game-actions button{width:100%; min-height:36px; padding:7px 8px}
  .game-actions{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; margin-top:6px}
  .status-row{display:grid; grid-template-columns:minmax(0,1fr) auto; gap:6px; align-items:stretch}
  .board-action{min-width:72px; min-height:34px; margin-top:8px; padding:6px 9px; font-size:12px}
  .record-actions{display:flex; gap:6px}
  .record-actions button{min-height:34px; padding:6px 9px}
  .capabilities-slot{padding:10px 12px 11px}
  :global(.capabilities-slot .capabilities){margin:0}
  :global(#status){
    min-height:34px; margin-top:8px; padding:7px 10px;
    border:1px solid var(--rule-strong); border-left:2px solid var(--accent);
    border-radius:0 6px 6px 0; background:var(--panel-inset);
    color:var(--text-soft); font-family:var(--mono); font-size:11px; line-height:1.45;
  }
  :global(#status.over){border-color:var(--accent); color:var(--ink); font-weight:700}
  :global(button.danger){background:var(--warn); border-color:var(--warn); color:#fff; font-weight:700}
  :global(#engineCaution){
    margin-top:7px; padding:7px 8px; border:1px dashed var(--rule);
    border-radius:6px; background:var(--panel-inset); line-height:1.4;
  }
  :global(#engineNote){
    margin-top:7px; padding:7px 8px; border:1px solid var(--rule); border-radius:6px;
    background:var(--panel-inset); font-family:var(--mono); font-size:10px; line-height:1.4;
  }
  :global(#engineNote.warn){color:var(--warn); border-color:var(--warn)}
  :global(#retryEngine){min-height:34px; margin-top:6px; padding:6px 9px}
  :global(#dlProgress){margin-top:8px; padding:7px 8px; border:1px solid var(--rule); border-radius:6px; background:var(--panel-inset)}
  :global(#dlProgress progress){width:100%; height:8px; accent-color:var(--accent)}
  :global(#dlProgress .dl-label){margin-top:2px; font-family:var(--mono); font-size:10px}
  .visually-hidden{position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0}
  :global(#promoPicker){margin-top:8px; display:flex; gap:6px; flex-wrap:wrap}
  :global(#moveList){
    min-height:40px; max-height:160px; margin-top:8px; padding:7px 10px; overflow:auto;
    border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset);
    font-family:var(--mono); font-size:12px; line-height:1.65;
  }
  :global(#moveList .num),:global(#moveList .placeholder){color:var(--muted)}
  :global(#moveList .san){padding:1px 3px; border-radius:4px}
  :global(#moveList .san.current){background:var(--accent); color:var(--on-accent); font-weight:600}
  :global(#pgnOut){
    max-height:220px; overflow:auto; margin-top:8px;
    font-family:var(--mono); font-size:10px; line-height:1.45;
    white-space:pre-wrap; overflow-wrap:anywhere;
  }
  :global(#pgnOut:not(:empty)){padding:8px; border:1px solid var(--rule); border-radius:6px; background:var(--card)}
  :global(.restart-banner){
    margin-top:8px; padding:9px 10px;
    border:1px solid var(--accent); border-radius:6px; background:var(--soft);
  }
  :global(#maia3Elo){
    -webkit-appearance:none; appearance:none; width:100%; height:5px; margin:0;
    border-radius:3px; background:var(--rule-strong); outline:none;
  }
  :global(#maia3Elo::-webkit-slider-thumb){
    -webkit-appearance:none; appearance:none; width:16px; height:16px;
    border-radius:50%; background:#fff; border:2px solid var(--accent); cursor:pointer;
    box-shadow:0 2px 6px rgba(80,55,25,.22);
  }
  :global(#maia3Elo::-moz-range-thumb){
    width:16px; height:16px; border-radius:50%; background:#fff;
    border:2px solid var(--accent); cursor:pointer; box-shadow:0 2px 6px rgba(80,55,25,.22);
  }
  @media (max-width:900px){
    main{grid-template-columns:1fr; max-width:680px; padding:12px 10px 40px}
    .app-sidebar{position:static; max-height:none; overflow:visible}
  }
  @media (max-width:520px){
    .board-panel{padding:8px}
    .opponent-grid{grid-template-columns:minmax(0,1.2fr) minmax(100px,.8fr)}
    .maia-grid,.sampling-grid{grid-template-columns:1fr}
    .game-setup{grid-template-columns:1fr}
    .game-actions{grid-template-columns:repeat(2,minmax(0,1fr))}
    .status-row{grid-template-columns:minmax(0,1fr) 68px}
    .board-action{min-width:0; padding-inline:6px}
  }
</style>
