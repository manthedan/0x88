<script lang="ts">
import { onMount } from 'svelte';
import { afterNavigate } from '$app/navigation';
import BrowserCapabilities from '$lib/components/BrowserCapabilities.svelte';
import SiteHeader from '$lib/components/SiteHeader.svelte';

let mountedUrl: URL | null = null;
afterNavigate(({ to }) => {
  if (mountedUrl && to?.url.pathname === mountedUrl.pathname && to.url.search !== mountedUrl.search) location.reload();
});
const title = '0x88 Chess — play chess vs an engine in your browser';
const description =
  'Play chess against Leela Chess Zero, Centipawn, Stockfish, Stormphrax, and other opponents running entirely in your browser. Five strength levels, takebacks, and PGN export.';
const ratingPresets = [800, 1100, 1500, 1900, 2200];
onMount(() => {
  mountedUrl = new URL(location.href);
  let cleanup: () => void = () => undefined;
  let mounted = true;
  void import('../../../lc0/playBrowser')
    .then((module) => {
      if (!mounted) return;
      cleanup = module.mountPlayBrowser();
    })
    .catch((error) => {
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
<main id="main" class="app-main play-workspace" data-phase="setup">
  <section class="panel board-panel" aria-label="Board">
    <div id="topStrip" class="player-strip" data-player="engine" data-turn="false">
      <span class="player-avatar" data-avatar aria-hidden="true">M3</span>
      <span class="player-id"><strong data-name>Maia3</strong><small data-meta>1500 rating</small></span>
      <span class="player-material" data-material></span>
      <span class="player-state" data-state></span>
    </div>
    <div class="board-shell">
      <div id="ground"></div>
      <div id="keyboardBoard"></div>
    </div>
    <p id="boardInstructions" class="visually-hidden">Arrow keys move between squares. Enter or Space picks up a piece, then a legal destination. Escape cancels. You can also type a move in the move box.</p>
    <p id="boardAnnouncement" class="visually-hidden" aria-live="polite"></p>
    <div id="bottomStrip" class="player-strip" data-player="you" data-turn="false">
      <span class="player-avatar you" data-avatar aria-hidden="true">You</span>
      <span class="player-id"><strong data-name>You</strong><small data-meta>White</small></span>
      <span class="player-material" data-material></span>
      <span class="player-state" data-state></span>
      <button id="flip" class="strip-button" type="button" title="Flip board" aria-label="Flip board"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 20V4m-4 4 4-4 4 4m4-4v16m-4-4 4 4 4-4" /></svg></button>
    </div>
    <div id="promoPicker" hidden></div>
  </section>

  <section class="panel app-sidebar play-sidebar" aria-label="Game">
    <p id="gameAnnouncement" class="visually-hidden" role="status" aria-live="polite" aria-atomic="true"></p>
    <div class="status-block">
      <div id="status" class="game-status">Loading…</div>
      <div id="engineNote" hidden></div>
      <div id="runtimeWarning" class="runtime-warning small" role="status" aria-live="polite" hidden></div>
      <button id="retryEngine" type="button" hidden>Retry engine</button>
      <div id="dlProgress" hidden><progress></progress><div class="dl-label small"></div></div>
      <div id="progressAnnouncement" class="visually-hidden" role="status" aria-live="polite"></div>
    </div>

    <div class="opponent-summary">
      <span class="player-avatar small" data-summary-avatar aria-hidden="true">M3</span>
      <div class="summary-copy"><span id="opponentSummary" class="summary-text">Maia3 · 1500</span><small>Casual · untimed</small></div>
      <button id="changeOpponent" class="ghost-button" type="button" aria-expanded="false" aria-controls="setupPane opponentDetails">Change</button>
    </div>

    <div id="setupPane" class="setup-pane" role="group" aria-label="Opponent settings">
      <h2 class="pane-heading">Opponent</h2>
      <div class="field"><label for="engineSelect" class="visually-hidden">Opponent</label>
        <select id="engineSelect"></select></div>
      <p id="opponentBlurb" class="opponent-blurb"></p>
      <div id="maia3Controls" hidden>
        <div class="rating-head">
          <span class="field-label">Rating presets</span>
          <span class="rating-value"><strong id="maia3EloValue">1500</strong> <span id="maia3EloBand"></span></span>
        </div>
        <div class="rating-presets" role="group" aria-label="Rating presets">
          {#each ratingPresets as preset}
            <button type="button" class="chip" data-elo={preset} aria-pressed="false">{preset}</button>
          {/each}
        </div>
      </div>
      <div class="field strength-field"><label for="levelSelect">Strength</label>
        <select id="levelSelect"></select></div>

      <div class="color-field">
        <span id="colorLabel" class="field-label">You play</span>
        <div class="color-choice" role="group" aria-labelledby="colorLabel">
          <button type="button" data-color="white" aria-pressed="true"><svg class="check" viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 8.5 3 3 6-7" /></svg><svg class="swatch white" viewBox="4 2.5 16 19.5" aria-hidden="true"><path d="M12 3.4a3.4 3.4 0 0 0-2.2 6H8.6a1 1 0 0 0 0 2h.9c-.1 2.6-1.3 4.5-3 5.8h11c-1.7-1.3-2.9-3.2-3-5.8h.9a1 1 0 0 0 0-2h-1.2a3.4 3.4 0 0 0-2.2-6zM5.6 18.6h12.8v2.6H5.6z" /></svg>White</button>
          <button type="button" data-color="random" aria-pressed="false"><svg class="check" viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 8.5 3 3 6-7" /></svg><svg class="swatch random" viewBox="4 2.5 16 19.5" aria-hidden="true"><defs><linearGradient id="randomPawnFill" x1="0" x2="1" y1="0" y2="0"><stop offset=".5" stop-color="#f6f3ea" /><stop offset=".5" stop-color="#1d1b18" /></linearGradient></defs><path d="M12 3.4a3.4 3.4 0 0 0-2.2 6H8.6a1 1 0 0 0 0 2h.9c-.1 2.6-1.3 4.5-3 5.8h11c-1.7-1.3-2.9-3.2-3-5.8h.9a1 1 0 0 0 0-2h-1.2a3.4 3.4 0 0 0-2.2-6zM5.6 18.6h12.8v2.6H5.6z" fill="url(#randomPawnFill)" /></svg>Random</button>
          <button type="button" data-color="black" aria-pressed="false"><svg class="check" viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 8.5 3 3 6-7" /></svg><svg class="swatch black" viewBox="4 2.5 16 19.5" aria-hidden="true"><path d="M12 3.4a3.4 3.4 0 0 0-2.2 6H8.6a1 1 0 0 0 0 2h.9c-.1 2.6-1.3 4.5-3 5.8h11c-1.7-1.3-2.9-3.2-3-5.8h.9a1 1 0 0 0 0-2h-1.2a3.4 3.4 0 0 0-2.2-6zM5.6 18.6h12.8v2.6H5.6z" /></svg>Black</button>
        </div>
        <select id="colorSelect" class="visually-hidden" tabindex="-1" aria-hidden="true">
          <option value="white">White</option>
          <option value="black">Black</option>
          <option value="random">Random</option>
        </select>
      </div>
    </div>

    <div id="restartBanner" class="restart-banner" hidden>
      <span id="restartMessage" class="small"></span>
      <div class="banner-actions">
        <button id="confirmRestart" class="primary" type="button">Start new game</button>
        <button id="dismissRestart" type="button">Keep current game</button>
      </div>
    </div>

    <div id="gameOverCard" class="game-over" hidden>
      <div class="result-line"><strong id="resultHeadline">Game over</strong><span id="resultScore"></span></div>
      <p id="resultReason"></p>
      <a id="analyzeGameCta" class="cta-link" href="/app/analysis" data-analyze>
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6" /><path d="m20 20-4.5-4.5" /></svg>Analyze this game
      </a>
    </div>

    <div class="moves-block">
      <p id="openingName" class="opening-name" hidden></p>
      <div id="moveList" class="move-list" aria-label="Moves"></div>
    </div>

    <div class="nav-row">
      <div class="move-nav" role="group" aria-label="Browse moves">
        <button id="navFirst" type="button" title="First move (Home / ↑)" aria-label="First move" disabled>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14M18 5l-8 7 8 7" /></svg>
        </button>
        <button id="navPrev" type="button" title="Previous move (←)" aria-label="Previous move" disabled>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-8 7 8 7" /></svg>
        </button>
        <button id="navNext" type="button" title="Next move (→)" aria-label="Next move" disabled>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 8 7-8 7" /></svg>
        </button>
        <button id="navLast" type="button" title="Last move (End / ↓)" aria-label="Last move" disabled>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M6 5l8 7-8 7" /></svg>
        </button>
      </div>
      <form id="moveForm" class="move-form" autocomplete="off">
        <label for="moveInput" class="visually-hidden">Type a move</label>
        <input id="moveInput" type="text" spellcheck="false" autocapitalize="off" placeholder="Type a move" title="Enter SAN (Nf3) or UCI (g1f3), then press Enter. Arrow keys browse moves." aria-describedby="boardInstructions" />
      </form>
    </div>

    <div class="game-actions" aria-label="Game actions">
      <button id="takeback" type="button" disabled>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 14 4 9l5-5" /><path d="M4 9h10a6 6 0 0 1 0 12h-3" /></svg><span class="lbl">Takeback</span>
      </button>
      <button id="resign" type="button" title="Resign (asks to confirm)" disabled>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 21V4m0 0h11l-2 4 2 4H5" /></svg><span class="lbl">Resign</span>
      </button>
      <button id="newGame" class="primary" type="button">
        <svg class="play-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" /></svg><svg class="new-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg><span class="lbl">Play</span>
      </button>
      <p class="play-hint">or just make a move on the board</p>
    </div>

    <details id="opponentDetails" class="settings-disclosure">
      <summary><svg class="chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m6 4 4 4-4 4" /></svg><span id="detailsLabel">Exact rating &amp; opponent details</span></summary>
      <div class="settings-body">
        <div id="maia3Advanced" class="sampling-grid">
          <div class="custom-rating">
            <label for="maia3Elo">Custom rating · 100-point steps</label>
            <div class="range-control">
              <span class="range-end" aria-hidden="true">600</span>
              <input id="maia3Elo" type="range" min="600" max="2600" step="100" value="1500" />
              <span class="range-end" aria-hidden="true">2600</span>
            </div>
          </div>
          <div class="field"><label for="maia3Style">Move selection</label>
            <select id="maia3Style">
              <option value="sample">Human sampling</option>
              <option value="argmax">Top move</option>
            </select></div>
          <div class="field" id="maia3TemperatureField"><label for="maia3Temperature">Temperature</label>
            <input id="maia3Temperature" type="number" min="0.01" max="5" step="0.05" value="1" /></div>
          <div class="field" id="maia3TopPField"><label for="maia3TopP">Top-p</label>
            <input id="maia3TopP" type="number" min="0.01" max="1" step="0.05" value="1" /></div>
        </div>
        <p id="levelCaption" class="small setup-note"></p>
        <div id="engineCaution" class="small" hidden></div>
        <div class="capabilities-slot"><BrowserCapabilities /></div>
      </div>
    </details>

    <div class="record-actions" role="group" aria-label="Game record">
      <button id="copyPgn" type="button" title="Copy the game as PGN">
        <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" /></svg><span class="lbl">Copy PGN</span>
      </button>
      <button id="downloadPgn" type="button" title="Download the game as a .pgn file">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m-4-4 4 4 4-4M5 19h14" /></svg><span class="lbl">Download</span>
      </button>
      <a id="analyzeGame" class="record-link" href="/app/analysis" data-analyze title="Open this game on the Analysis board">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6" /><path d="m20 20-4.5-4.5" /></svg><span class="lbl">Analyze</span>
      </a>
    </div>
  </section>
</main>

<style>
  .play-workspace{
    --muted:color-mix(in srgb, var(--ink) 74%, var(--panel));
    --text-soft:var(--muted);
    --coord-dark:#211d17;
    --coord-light:#fff8eb;
    --strip-h:44px;
    --board-size:min(680px, calc(100dvh - 240px), calc(100vw - 500px));
    grid-template-columns:calc(var(--board-size) + 18px) minmax(330px, 400px);
    gap:16px; max-width:1140px; padding:16px 24px 32px;
    font-variant-numeric:tabular-nums;
    /* Control boundaries reach >=3:1 against the panel (page-local; shared tokens sit near 1.4:1). */
    --border-input:#8f8470; --border-input-hover:#6f6553;
    --selected-fill:color-mix(in srgb, var(--accent) 20%, var(--panel));
  }
  :global(:root[data-theme="dark"]) .play-workspace{--border-input:#766d5a; --border-input-hover:#958b77}
  @media (prefers-color-scheme: dark){
    :global(:root:not([data-theme])) .play-workspace{--border-input:#766d5a; --border-input-hover:#958b77}
  }
  .visually-hidden{position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0}

  /* ── Board column: opponent strip / board / your strip ── */
  .board-panel{display:flex; flex-direction:column; gap:4px; min-width:0; padding:8px}
  .board-shell{position:relative}
  :global(.play-workspace #ground coords){font-size:12px; font-weight:700; opacity:1}
  :global(.play-workspace #keyboardBoard){position:absolute; inset:0; z-index:5; pointer-events:none}
  :global(.play-workspace .keyboard-row){display:contents}
  :global(.play-workspace .keyboard-square){position:absolute; width:12.5%; height:12.5%; padding:0; min-height:0; border:0; border-radius:0; background:transparent; pointer-events:none}
  :global(.play-workspace .keyboard-square:focus-visible){outline:3px solid #fff; outline-offset:-4px; box-shadow:inset 0 0 0 6px #225ca0; background:#fff2}
  .player-strip{
    display:flex; align-items:center; gap:10px; min-width:0; height:var(--strip-h); padding:0 6px 0 4px;
    border-radius:8px; transition:background .15s;
  }
  .player-strip:global([data-turn=true]){background:color-mix(in srgb, var(--accent) 13%, var(--panel))}
  .player-strip:global([data-turn=true]) .player-avatar{box-shadow:0 0 0 2px var(--accent)}
  .player-avatar{
    display:grid; place-items:center; flex:0 0 34px; width:34px; height:34px; border-radius:7px;
    background:var(--accent-soft); color:var(--accent-soft-text); box-shadow:inset 0 0 0 1px var(--accent-tint-border);
    font:700 12.5px/1 var(--sans); letter-spacing:-.01em;
  }
  .player-avatar.you{background:var(--panel-2); color:var(--ink); box-shadow:inset 0 0 0 1px var(--border-input-hover); font-size:11px}
  .player-avatar.small{flex-basis:28px; width:28px; height:28px; font-size:11px; border-radius:6px}
  .player-id{display:flex; flex:0 1 auto; flex-direction:column; min-width:0; line-height:1.25}
  .player-id strong{color:var(--ink); font:650 15px var(--sans); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .player-id small{color:var(--muted); font-size:12.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .player-material{display:inline-flex; align-items:center; gap:0; min-width:24px; margin-right:auto; max-width:120px; color:var(--ink); font-size:20px; line-height:1.2; white-space:nowrap; overflow:hidden}
  .player-material :global(.taken){min-width:0; overflow:hidden; padding-right:3px; letter-spacing:-.18em}
  .player-material :global(.adv){flex:0 0 auto; margin-left:5px; font:600 12.5px var(--sans); color:var(--muted)}
  .player-state{display:inline-flex; align-items:center; gap:6px; color:var(--ink); font:650 14px var(--sans); white-space:nowrap}
  .player-state:empty{display:none}
  .player-state:global([data-kind=thinking])::before{
    content:''; width:12px; height:12px; border-radius:50%;
    border:2px solid color-mix(in srgb, var(--accent) 30%, transparent); border-top-color:var(--accent);
    animation:play-spin .8s linear infinite;
  }
  .player-state:global([data-kind=turn]){color:var(--accent-soft-text); font-weight:700}
  .player-state:global([data-kind=error]){color:var(--warn)}
  .player-state:global([data-kind=ready]){color:var(--muted); font-size:13px}
  .strip-button{display:grid; place-items:center; width:34px; height:34px; padding:0; border-radius:7px; background:transparent; color:var(--muted)}
  .strip-button:hover:not(:disabled){background:var(--panel-2); color:var(--ink)}
  .strip-button svg, .game-actions svg, .record-actions svg, .move-nav svg, .cta-link svg{width:16px; height:16px; flex:0 0 16px; fill:none; stroke:currentColor; stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round}
  @keyframes play-spin{to{transform:rotate(360deg)}}

  /* ── Sidebar: one column, two states (setup vs game) driven by data-phase ── */
  .play-workspace .play-sidebar{
    position:static; display:flex; flex-direction:column; min-width:0;
    max-height:calc(var(--board-size) + 2 * var(--strip-h) + 26px); overflow:hidden;
  }
  .play-sidebar>*{flex:0 0 auto}
  .status-block{padding:12px 14px 10px; border-bottom:1px solid var(--rule); background:var(--panel-inset)}
  :global(#status.game-status){
    display:flex; align-items:center; gap:9px; min-height:24px;
    color:var(--ink); font:600 14.5px/1.4 var(--sans);
  }
  :global(#status.game-status.over){font-size:15px}
  .play-workspace:global([data-phase=setup] #status.game-status){font-size:16px; font-weight:700}
  :global(#status.game-status.browsing){color:var(--text-soft); font-weight:500; font-size:13.5px}
  :global(#engineNote){margin-top:6px; color:var(--muted); font:13px/1.45 var(--sans)}
  :global(#engineNote.warn){padding:7px 9px; border:1px solid color-mix(in srgb, var(--warn) 45%, var(--rule)); border-radius:6px; background:var(--warn-soft); color:var(--warn)}
  :global(#engineNote[data-routine=true]){position:absolute; width:1px; height:1px; margin:-1px; padding:0; overflow:hidden; clip-path:inset(50%)}
  :global(.play-workspace .runtime-warning){
    margin-top:7px; padding:7px 9px; border:1px solid color-mix(in srgb, var(--warn) 45%, var(--rule));
    border-radius:6px; background:var(--warn-soft); color:var(--warn); line-height:1.4;
  }
  :global(#retryEngine){min-height:34px; margin-top:6px; padding:6px 10px}
  :global(#dlProgress){margin-top:8px}
  :global(#dlProgress progress){width:100%; height:8px; accent-color:var(--accent)}
  :global(#dlProgress .dl-label){margin-top:2px; font-size:12px; color:var(--muted)}

  /* Setup pane */
  .setup-pane{display:flex; flex-direction:column; gap:10px; padding:14px; overflow:auto; overscroll-behavior:contain}
  .play-sidebar .pane-heading{margin:0; font:650 13px/1.3 var(--sans); letter-spacing:0; text-transform:none; color:var(--ink)}
  .field{margin-top:0}
  .field-label, .field label{font:500 13px var(--sans); color:var(--muted)}
  #engineSelect{width:100%; min-height:44px; font-size:15px; font-weight:600}
  .opponent-blurb{margin:-2px 0 0; color:var(--text); font:13.5px/1.45 var(--sans)}
  .opponent-blurb:empty{display:none}
  :global(#engineCaution){color:var(--muted); font-size:12.5px; line-height:1.45}
  .rating-head{display:flex; align-items:baseline; justify-content:space-between; gap:8px}
  .rating-value{color:var(--muted); font-size:13px}
  .rating-value strong{color:var(--ink); font:700 15px var(--sans)}
  .range-control{display:flex; align-items:center; gap:10px; min-height:44px}
  .range-end{color:var(--muted); font-size:12px}
  .rating-presets{display:flex; gap:6px; margin-top:6px; flex-wrap:wrap}
  .chip{flex:1 1 0; min-height:44px; padding:4px 6px; border-radius:8px; font:600 13px var(--sans); color:var(--text)}
  .chip:global([aria-pressed=true]), .chip:global([aria-pressed=true]:hover){border-color:var(--accent); box-shadow:inset 0 0 0 1px var(--accent); color:var(--ink); background:var(--selected-fill); font-weight:750}
  .strength-field select{width:100%}
  .color-field{display:grid; gap:6px}
  .color-choice{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0; border:1px solid var(--border-input); border-radius:8px; overflow:hidden}
  .color-choice button{display:inline-flex; align-items:center; justify-content:center; gap:7px; min-height:44px; padding:0 6px; border:0; border-radius:0; background:var(--panel); color:var(--text); font:600 13.5px var(--sans)}
  .color-choice button+button{border-left:1px solid var(--border-input)}
  .color-choice button:global([aria-pressed=true]){position:relative; z-index:1; background:var(--selected-fill); color:var(--ink); font-weight:750; box-shadow:inset 0 0 0 2px var(--accent)}
  .color-choice .check{display:none; width:14px; height:14px; flex:0 0 14px; margin-right:-2px; fill:none; stroke:var(--accent-soft-text); stroke-width:2.4; stroke-linecap:round; stroke-linejoin:round}
  .color-choice button:global([aria-pressed=true]) .check{display:block}
  /* Piece-colour icons (pawns), so they never read as radio or checkbox state. */
  .swatch{width:17px; height:20px; flex:0 0 17px; stroke:#2b261e; stroke-width:1.1; stroke-linejoin:round}
  .swatch.white{fill:#f6f3ea}
  .swatch.black{fill:#1d1b18; stroke:#8f8573}
  .swatch.random{stroke:#6f6553}
  .settings-disclosure{flex:0 1 auto; min-height:45px; overflow:auto; border-top:1px solid var(--rule); padding:0 14px}
  .settings-disclosure[open]{padding-bottom:12px}
  .settings-disclosure>summary{display:flex; align-items:center; gap:6px; min-height:44px; cursor:pointer; list-style:none; color:var(--text); font:600 13px var(--sans)}
  .settings-disclosure>summary::-webkit-details-marker{display:none}
  .settings-disclosure>summary:hover{color:var(--ink)}
  .chevron{width:14px; height:14px; flex:0 0 14px; fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; transition:transform .15s}
  .settings-disclosure[open] .chevron{transform:rotate(90deg)}
  .settings-body{display:grid; gap:10px; padding:6px 0 4px}
  .sampling-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px; align-items:end}
  .sampling-grid>.custom-rating, .sampling-grid>.field:has(#maia3Style){grid-column:1 / -1}
  .custom-rating label{font:500 13px var(--sans); color:var(--muted)}
  .sampling-grid input, .sampling-grid select{width:100%}
  .setup-note{margin:0; line-height:1.5; font-size:12.5px}
  .capabilities-slot :global(.capabilities){margin:0}
  :global(#maia3Elo){
    -webkit-appearance:none; appearance:none; flex:1 1 auto; min-width:0; width:100%; height:44px; margin:0;
    border-radius:3px; background:linear-gradient(var(--border-input-hover),var(--border-input-hover)) center / 100% 5px no-repeat;
  }
  :global(#maia3Elo:focus-visible){outline:2px solid var(--accent); outline-offset:2px
  }
  :global(#maia3Elo::-webkit-slider-thumb){
    -webkit-appearance:none; appearance:none; width:24px; height:24px;
    border-radius:50%; background:#fff; border:2px solid var(--accent); cursor:pointer;
    box-shadow:0 2px 6px rgba(80,55,25,.22);
  }
  :global(#maia3Elo::-moz-range-thumb){
    width:24px; height:24px; border-radius:50%; background:#fff;
    border:2px solid var(--accent); cursor:pointer; box-shadow:0 2px 6px rgba(80,55,25,.22);
  }

  /* Game pane pieces */
  .opponent-summary{display:flex; align-items:center; gap:9px; min-height:48px; padding:6px 10px 6px 14px; border-bottom:1px solid var(--rule)}
  .summary-copy{display:flex; flex:1 1 auto; flex-direction:column; min-width:0; gap:2px}
  .summary-copy small{color:var(--muted); font-size:12px}
  .summary-text{min-width:0; color:var(--ink); font:600 13.5px var(--sans); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .ghost-button{min-height:44px; padding:0 11px; border-radius:7px; background:transparent; color:var(--text); font:600 13px var(--sans)}
  .ghost-button:global([aria-expanded=true]){border-color:var(--accent); color:var(--ink)}
  .restart-banner{margin:10px 12px 0; padding:10px; border:1px solid var(--accent-tint-border); border-radius:8px; background:var(--accent-soft)}
  .restart-banner :global(#restartMessage){color:var(--ink); font-size:13px}
  .banner-actions{display:flex; gap:6px; margin-top:8px}
  .banner-actions button{flex:1 1 0; min-height:34px}
  .game-over{margin:10px 12px 0; padding:12px; border:1px solid var(--rule-strong); border-radius:10px; background:var(--panel-inset)}
  .result-line{display:flex; align-items:baseline; justify-content:space-between; gap:8px}
  .result-line strong{color:var(--ink); font:700 18px var(--sans)}
  .result-line span{color:var(--muted); font:600 14px var(--sans)}
  .game-over p{margin:2px 0 10px; color:var(--text); font-size:13.5px}
  .cta-link{display:flex; align-items:center; justify-content:center; gap:8px; min-height:40px; border-radius:7px; background:var(--accent); color:var(--on-accent); font:650 14px var(--sans); text-decoration:none}
  .cta-link:hover{background:var(--accent-hover); text-decoration:none}
  .play-workspace:global(:not([data-phase=setup])) .play-sidebar{align-self:stretch}
  .play-sidebar .moves-block{display:flex; flex-direction:column; flex:1 1 auto; min-height:96px}
  .opening-name{flex:0 0 auto; margin:0; padding:8px 14px; border-bottom:1px solid var(--rule); color:var(--muted); font:13px/1.4 var(--sans); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .opening-name :global(strong){margin-right:6px; color:var(--ink); font-weight:650}
  /* Empty rows continue the scoresheet below the last move, so an early game reads as a blank sheet, not a gap. */
  :global(#moveList.move-list){
    --row-h:36px;
    flex:1 1 auto; min-height:0; overflow:auto; overscroll-behavior:contain; scrollbar-width:thin; font:500 16px/1.4 var(--sans); font-variant-numeric:tabular-nums;
    background:
      linear-gradient(90deg, var(--panel-inset) 43px, var(--rule-soft) 43px 44px, transparent 44px),
      repeating-linear-gradient(180deg, transparent 0 var(--row-h), color-mix(in srgb, var(--ink) 3%, var(--panel)) var(--row-h) calc(2 * var(--row-h)));
    background-attachment:local;
  }
  :global(#moveList .move-row){display:grid; grid-template-columns:44px minmax(0,1fr) minmax(0,1fr); min-height:var(--row-h)}
  :global(#moveList .move-row:nth-child(even)){background:color-mix(in srgb, var(--ink) 3%, var(--panel))}
  :global(#moveList .num){display:grid; place-items:center; color:var(--muted); font-size:13px; background:var(--panel-inset); border-right:1px solid var(--rule-soft)}
  :global(#moveList .san){display:flex; align-items:center; min-height:var(--row-h); padding:0 12px; border:0; border-radius:0; background:transparent; color:var(--ink); font:inherit; text-align:left; cursor:pointer}
  :global(#moveList .san:hover:not(.current)){background:var(--accent-soft); color:var(--ink)}
  :global(#moveList .san.current){background:color-mix(in srgb, var(--accent) 28%, var(--panel)); color:var(--ink); font-weight:700}
  :global(#moveList .placeholder){display:block; padding:14px; background:var(--panel); color:var(--muted); font-size:13.5px}
  .nav-row{display:flex; align-items:center; gap:8px; padding:8px 10px; border-top:1px solid var(--rule)}
  .move-nav{display:flex; gap:4px; flex:0 0 auto}
  .move-nav button{display:grid; place-items:center; width:38px; height:36px; padding:0; border-radius:7px; border:1px solid var(--border-input); background:var(--panel-2); color:var(--ink)}
  .move-nav button:disabled{opacity:1; cursor:default; background:transparent; border-color:var(--rule)}
  .move-nav button:disabled svg{opacity:.3}
  .move-nav svg{width:18px; height:18px; stroke-width:2.2}
  :global(.move-nav button.live-pending){background:var(--accent); border-color:var(--accent); color:var(--on-accent)}
  :global(.move-nav button.live-pending:hover:not(:disabled)){background:var(--accent-hover); color:var(--on-accent)}
  .move-form{flex:1 1 auto; min-width:0; margin:0}
  .move-form input{width:100%; min-height:36px; padding:6px 10px; font-size:14px}
  .move-form input:global(.invalid){border-color:var(--warn)}
  .move-form input::placeholder{color:var(--muted); opacity:1}
  .game-actions{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; padding:8px 10px; border-top:1px solid var(--rule)}
  .game-actions button, .record-actions :is(button, a){display:inline-flex; align-items:center; justify-content:center; gap:7px; min-height:38px; padding:0 8px; border-radius:7px; font:600 13.5px var(--sans); color:var(--ink); white-space:nowrap}
  .game-actions button{min-width:0; white-space:normal; line-height:1.25}
  .game-actions #newGame.primary, .game-actions #newGame.primary:hover{color:var(--on-accent)}
  .game-actions button:not(.primary){background:transparent; border-color:var(--border-input)}
  .game-actions button:not(.primary):hover:not(:disabled){background:var(--panel-2)}
  .game-actions #resign:not(:disabled){color:var(--warn)}
  .game-actions #resign:hover:not(:disabled):not(:global(.danger)){border-color:var(--warn); background:var(--warn-soft)}
  .game-actions button:disabled{opacity:1; color:color-mix(in srgb, var(--ink) 55%, var(--panel))}
  .game-actions button:disabled svg{opacity:.6}
  .game-actions #resign:global(.danger), .game-actions #resign:global(.danger:hover){background:var(--warn); border-color:var(--warn); color:var(--bg)}
  .game-actions #resign:global(.danger) svg{display:none}
  :global(.game-actions button.armed){border-color:var(--warn); color:var(--warn)}
  .play-icon{display:none}
  .play-hint{display:none; margin:0; color:var(--muted); font:13px/1.4 var(--sans); text-align:center}
  .record-actions{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; padding:0 10px 10px}
  .record-actions :is(button, a){border:1px solid transparent; background:transparent; color:var(--text); font-weight:550; font-size:13px}
  .record-actions :is(button, a):hover:not(:disabled){background:var(--panel-2); color:var(--ink); text-decoration:none}
  .record-actions button:disabled{opacity:.5}
  .record-actions :global([data-copied=true]){color:var(--accent-soft-text)}
  .record-actions :global(a[aria-disabled=true]){opacity:.5; pointer-events:none}

  /* Phase visibility */
  .play-workspace:global([data-phase=setup]) :is(.opponent-summary, .moves-block, .nav-row, .record-actions, #takeback, #resign){display:none}
  .play-workspace:global([data-phase=setup]) .setup-pane{flex:0 1 auto; min-height:0}
  .play-workspace:global([data-phase=setup]) .game-actions{grid-template-columns:1fr; padding:0 14px 14px; border-top:0}
  .play-workspace:global([data-phase=playing][data-quiet=true]) .status-block:not(:has(:global(#dlProgress:not([hidden]), #runtimeWarning:not([hidden]), #retryEngine:not([hidden]), #engineNote[data-routine=false]:not([hidden])))){display:none}
  .play-workspace:global([data-phase=setup]) #newGame{min-height:48px; font-size:16px; font-weight:700}
  .play-workspace:global([data-phase=setup]) .play-icon{display:block}
  .play-workspace:global([data-phase=setup]) .new-icon{display:none}
  .play-workspace:global([data-phase=setup]) .play-hint{display:block}
  .play-workspace:global([data-phase=setup]) .game-actions{gap:8px}
  .play-workspace:global(:not([data-phase=setup]):not([data-config=open])) :is(.setup-pane, .settings-disclosure){display:none}
  .play-workspace:global(:not([data-phase=setup])[data-config=open]) .setup-pane{flex:0 1 auto; max-height:58%; border-bottom:1px solid var(--rule)}
  .play-workspace:global([data-phase=over]) #resign{display:none}
  .play-workspace:global([data-phase=over]) .game-actions{grid-template-columns:1fr 1fr}

  @media (max-width:900px){
    .play-workspace{--board-size:100%; grid-template-columns:1fr; max-width:680px; padding:12px; gap:10px}
    .play-workspace .play-sidebar{max-height:none; overflow:visible}
    .board-panel{scroll-margin-top:74px}
    :global(#moveList.move-list){max-height:180px}
    .play-workspace:global(:not([data-phase=setup])[data-config=open]) .setup-pane{max-height:none}
    /* DOM, visual and keyboard order all follow board → setup/game → actions. */
    .move-nav button, .move-form input, .game-actions button, .record-actions :is(button,a){min-height:44px}
    :global(#moveList.move-list){--row-h:44px}
  }
  @media (max-width:600px){
    .play-workspace{--strip-h:44px; padding:8px 6px 32px; gap:6px}
    .board-panel{padding:0; border:0; background:transparent; gap:2px}
    .player-avatar{flex-basis:30px; width:30px; height:30px; font-size:11.5px}
    .player-id strong{font-size:14px}
    .player-material{max-width:80px}
    .player-strip{gap:7px}
    .strip-button{width:44px; height:44px; flex:0 0 44px; margin-right:-6px}
    .player-id small{font-size:12px}
    :global(.play-workspace #ground coords){font-size:11px}
    .status-block{padding:10px 12px 8px}
    .play-workspace:global([data-phase=setup]) .status-block:not(:has(:global(#dlProgress:not([hidden]), #runtimeWarning:not([hidden]), #retryEngine:not([hidden])))){display:none}
    .setup-pane{padding:12px; gap:9px}
    .play-workspace:global([data-phase=setup]) .pane-heading{display:none}
    .opponent-blurb{display:none}
    .color-field>.field-label{position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%)}
    .settings-disclosure{padding-inline:12px}
    .play-workspace:global([data-phase=setup]) .setup-pane{gap:8px; padding-top:10px}
    .play-workspace:global([data-phase=setup]) .game-actions{padding:0 12px 10px}
    .play-workspace:global([data-phase=setup]) .play-hint{display:none}
    .move-nav button{width:auto; flex:1 1 0; height:40px}
    .move-nav{flex:1 1 60%}
    .move-form{flex:1 1 40%}
    .move-form input{min-height:40px}
    .game-actions button{min-height:44px; gap:4px; font-size:13px}
    :global(#moveList.move-list){max-height:150px}
  }
  @media (prefers-reduced-motion: reduce){
    .player-state:global([data-kind=thinking])::before{animation:none}
  }
</style>
