<script lang="ts">
  import { onMount } from 'svelte';
  const title = "0x88 Chess — play chess vs an engine in your browser";
  const description = "Play chess against Leela Chess Zero, Stockfish, and four more engines running entirely in your browser. Five strength levels, takebacks, and PGN export.";
  const styles = "main{display:grid;grid-template-columns:minmax(320px,620px) minmax(300px,400px);gap:20px;align-items:start;justify-content:center;padding:18px}\n.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px}\n.board-shell{width:100%;aspect-ratio:1}#ground{width:100%;height:100%;border-radius:6px;overflow:hidden;box-shadow:0 14px 38px -26px #000;background:var(--board-dark)}\n.cg-wrap{width:100%!important;height:100%!important}\n#status{margin-top:12px;padding:10px 12px;border:1px solid var(--rule);border-radius:6px;background:var(--soft);font-size:14px}\n#status.over{border-color:var(--accent);font-weight:700}\nh2{font-size:13px;margin:16px 0 0;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}\n.field{display:grid;gap:3px;margin-top:10px}.field label{font-size:11px;color:var(--muted)}\nselect,input{width:100%}\nbutton.danger{background:var(--warn);border-color:var(--warn);color:#fff;font-weight:700}\n.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.row button{flex:1 1 auto}\n.small{font-size:12px;color:var(--muted);line-height:1.5}\n#levelCaption{margin-top:4px}\n#engineCaution{margin-top:6px;padding:7px 9px;border:1px dashed var(--rule);border-radius:6px;background:#fff}\n#engineNote{margin-top:8px;padding:8px 10px;border:1px solid var(--rule);border-radius:6px;background:#fff;font-family:var(--mono);font-size:12px}\n#engineNote.warn{color:var(--warn);border-color:var(--warn)}\n#dlProgress{margin-top:8px}#dlProgress progress{width:100%;height:10px;accent-color:var(--accent)}#dlProgress .dl-label{margin-top:2px;font-family:var(--mono)}\n#promoPicker{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}#promoPicker button{flex:1 1 auto}\n#moveList{margin-top:8px;font-family:var(--mono);font-size:13px;line-height:1.8;max-height:300px;overflow:auto;padding:8px;border:1px solid var(--rule);border-radius:6px;background:#fff}\n#moveList .num{color:var(--muted)}#moveList .placeholder{color:var(--muted)}\n#pgnOut{font-family:var(--mono);font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:8px}\n#pgnOut:not(:empty){padding:8px;border:1px solid var(--rule);border-radius:6px;background:#fff}\n.restart-banner{margin-top:10px;padding:10px 12px;border:1px solid var(--accent);border-radius:6px;background:var(--soft)}\n@media(max-width:900px){main{grid-template-columns:1fr;padding:12px}}\n[hidden]{display:none !important}\n:where(button,select,input,textarea,a):focus-visible{outline:2px solid var(--accent);outline-offset:2px}\n@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}";
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

<div>{@html `<style>${styles}</style>`}</div>

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
  </section>
  <section class="panel" aria-label="Game controls">
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
    <h2>Moves</h2>
    <div id="moveList"></div>
    <div class="row">
      <button id="exportPgn" type="button">Show PGN</button>
      <button id="copyPgn" type="button">Copy PGN</button>
    </div>
    <div id="pgnOut"></div>
    <p class="small">Engines run entirely in your browser &#x2014; nothing is sent to a server. The first move against a new engine downloads it (a few MB for CPU engines, up to a few hundred MB for the big Lc0 nets).</p>
  </section>
</main>
