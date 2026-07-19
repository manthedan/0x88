<script lang="ts">
  import { onMount } from 'svelte';
  import { Chessground } from 'chessground';
  import type { Key } from 'chessground/types';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  import { parseFen } from '../../../chess/board';
  import { moveToUci } from '../../../chess/moveCodec';
  import { parsePgnGame, type PgnGame } from '../../../chess/pgn';
  import { boardCheck } from '../../../lc0/boardUx';
  import { bestMoveShapes } from '../../../lc0/boardArrows';
  import { lineChartSvg } from '../../../lc0/charts';
  import { annotatedPgn, type GameReview, type MoveClass, type ReviewedMove } from '../../../lc0/gameReview';
  import { runGameReview } from '../../../lc0/gameReviewRunner';
  import type { GameNode } from '../../../lc0/gameTree';
  import { listPgnCollectionGames, loadPgnCollection, loadPgnGameReview, pgnGameReviewKey, savePgnGameReview } from '../../../lc0/pgnDatabase';
  import { StockfishEngine } from '../../../lc0/stockfishEngine';

  const title = '0x88 Chess — game review';
  const description = 'Review a saved game move by move with Stockfish running locally in your browser.';
  const REVIEW_ENGINE = 'stockfish-lite';
  const REVIEW_ALGORITHM_VERSION = 1;
  const classLabel: Record<MoveClass, string> = {
    best: 'Best', good: 'Good', inaccuracy: 'Inaccuracy', mistake: 'Mistake', blunder: 'Blunder', forced: 'Forced',
  };

  let groundElement: HTMLElement;
  let ground: ReturnType<typeof Chessground> | null = null;
  let game: PgnGame | null = null;
  let nodes: GameNode[] = [];
  let collectionId = '';
  let collectionName = '';
  let gameId = '';
  let gameIndex = 0;
  let currentPly = 0;
  let orientation: 'white' | 'black' = 'white';
  let depth = 12;
  let status = 'Loading game…';
  let loadError = '';
  let running = false;
  let progress = 0;
  let progressTotal = 0;
  let review: GameReview | null = null;
  let reviewController: AbortController | null = null;
  let activeEngine: StockfishEngine | null = null;
  let restoreGeneration = 0;

  $: currentReviewedMove = currentPly > 0 ? review?.moves[currentPly - 1] ?? null : null;
  $: reviewedByPly = new Map(review?.moves.map((move) => [move.ply, move]) ?? []);
  $: chartSvg = review ? lineChartSvg([{
    label: 'White win %',
    color: '#5f7f35',
    points: review.moves.map((move) => ({ x: move.ply, y: move.winAfter })),
  }], { width: 520, height: 132, yMin: 0, yMax: 1, midline: 0.5, formatY: (value) => `${Math.round(value * 100)}%` }) : '';

  onMount(() => {
    let mounted = true;
    const onKeydown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === 'ArrowLeft') goTo(currentPly - 1);
      else if (event.key === 'ArrowRight') goTo(currentPly + 1);
      else if (event.key === 'ArrowUp') goTo(0);
      else if (event.key === 'ArrowDown') goTo(nodes.length - 1);
    };
    window.addEventListener('keydown', onKeydown);
    void loadRequestedGame().then(() => {
      if (mounted) renderBoard();
    });
    return () => {
      mounted = false;
      restoreGeneration += 1;
      window.removeEventListener('keydown', onKeydown);
      reviewController?.abort();
      activeEngine?.dispose();
      ground?.destroy();
      ground = null;
    };
  });

  async function loadRequestedGame(): Promise<void> {
    const params = new URLSearchParams(location.search);
    collectionId = params.get('collection') ?? '';
    const requestedGameId = params.get('gameId') ?? '';
    gameIndex = params.get('game') === null ? 0 : Number(params.get('game'));
    if (!collectionId) {
      loadError = 'Choose a saved game from the Library before starting a review.';
      status = loadError;
      return;
    }
    try {
      const collection = await loadPgnCollection(collectionId);
      if (!collection) throw new Error('Collection not found in this browser');
      const collectionGames = await listPgnCollectionGames(collectionId);
      const storedGame = requestedGameId
        ? (collectionGames[gameIndex]?.id === requestedGameId ? collectionGames[gameIndex] : collectionGames.find((entry) => entry.id === requestedGameId))
        : Number.isInteger(gameIndex) && gameIndex >= 0 ? collectionGames[gameIndex] : undefined;
      if (!storedGame) throw new Error('Requested game is not present in this collection');
      collectionName = collection.name;
      gameId = storedGame.id;
      gameIndex = storedGame.order;
      game = parsePgnGame(storedGame.pgn);
      nodes = [game.tree.root, ...game.tree.mainlineFrom()];
      currentPly = 0;
      await restoreReviewForDepth();
    } catch (error) {
      loadError = (error as Error).message;
      status = `Could not load game: ${loadError}`;
    }
  }

  async function restoreReviewForDepth(): Promise<void> {
    if (!gameId) return;
    const generation = ++restoreGeneration;
    const requestedGameId = gameId;
    const requestedDepth = depth;
    try {
      const saved = await loadPgnGameReview(requestedGameId, pgnGameReviewKey(REVIEW_ENGINE, requestedDepth, REVIEW_ALGORITHM_VERSION));
      if (generation !== restoreGeneration || requestedGameId !== gameId || requestedDepth !== depth) return;
      if (saved) {
        review = saved.review;
        status = `Loaded saved review from depth ${saved.depth}.`;
      } else {
        review = null;
        status = `${nodes.length - 1} moves ready to review with Stockfish Lite.`;
      }
      renderBoard();
    } catch (error) {
      if (generation !== restoreGeneration || requestedGameId !== gameId || requestedDepth !== depth) return;
      review = null;
      status = `Saved review could not be loaded: ${(error as Error).message}`;
    }
  }

  function renderBoard(): void {
    const node = nodes[currentPly];
    if (!node || !groundElement) return;
    const board = parseFen(node.fen);
    const lastUci = node.move ? moveToUci(node.move) : null;
    const config = {
      orientation,
      fen: node.fen.split(' ')[0],
      turnColor: board.turn === 'w' ? 'white' as const : 'black' as const,
      coordinates: true,
      viewOnly: true,
      check: boardCheck(board),
      highlight: { lastMove: true, check: true },
      animation: { enabled: true, duration: 160 },
      lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
    };
    const typed = config as NonNullable<Parameters<typeof Chessground>[1]>;
    if (!ground) ground = Chessground(groundElement, typed);
    else ground.set(typed);
    const nextReview = review?.moves[currentPly];
    ground.setShapes(bestMoveShapes(nextReview?.bestUci ?? undefined));
  }

  function goTo(ply: number): void {
    if (!nodes.length) return;
    currentPly = Math.max(0, Math.min(nodes.length - 1, ply));
    renderBoard();
  }

  function flipBoard(): void {
    orientation = orientation === 'white' ? 'black' : 'white';
    renderBoard();
  }

  async function startReview(): Promise<void> {
    if (!game || running || nodes.length < 2) return;
    restoreGeneration += 1;
    review = null;
    ground?.setShapes([]);
    running = true;
    progress = 0;
    progressTotal = nodes.length;
    const controller = new AbortController();
    const engine = new StockfishEngine({ depth });
    reviewController = controller;
    activeEngine = engine;
    status = 'Preparing Stockfish Lite…';
    try {
      await engine.prewarm(controller.signal);
      await engine.newGame(controller.signal);
      const result = await runGameReview({
        nodes: nodes.map((node) => ({ fen: node.fen, san: node.san, uci: node.move ? moveToUci(node.move) : null })),
        engine,
        depth,
        signal: controller.signal,
        onProgress: ({ position, total }) => {
          progress = position;
          progressTotal = total;
          status = `Reviewing position ${position} of ${total} at depth ${depth}…`;
        },
      });
      if (reviewController !== controller) return;
      review = result;
      try {
        await savePgnGameReview({
          gameId,
          reviewKey: pgnGameReviewKey(REVIEW_ENGINE, depth, REVIEW_ALGORITHM_VERSION),
          engine: REVIEW_ENGINE,
          depth,
          algorithmVersion: REVIEW_ALGORITHM_VERSION,
          review: result,
          annotatedPgn: annotatedReviewPgn(result),
        });
        status = `Review complete and saved: ${result.moves.length} moves analyzed at depth ${depth}.`;
      } catch (error) {
        status = `Review complete, but could not be saved: ${(error as Error).message}`;
      }
      const firstCritical = result.criticalMoves[0];
      if (firstCritical) goTo(firstCritical.ply);
      else renderBoard();
    } catch (error) {
      if (reviewController === controller) status = (error as Error).name === 'AbortError' ? 'Review stopped.' : `Review failed: ${(error as Error).message}`;
    } finally {
      engine.dispose();
      if (activeEngine === engine) activeEngine = null;
      if (reviewController === controller) reviewController = null;
      running = false;
    }
  }

  function stopReview(): void {
    reviewController?.abort();
    activeEngine?.dispose();
  }

  function annotatedReviewPgn(result: GameReview): string {
    if (!game) return '';
    const start = parseFen(nodes[0].fen);
    return annotatedPgn(result, {
      tags: { ...game.tags, Event: game.tags.Event || '0x88 Game Review' },
      result: game.result,
      startFullmove: start.fullmove,
      startTurn: start.turn,
    });
  }

  function reviewPgn(): string {
    return review ? annotatedReviewPgn(review) : '';
  }

  async function copyAnnotatedPgn(): Promise<void> {
    const pgn = reviewPgn();
    if (!pgn) return;
    try {
      await navigator.clipboard.writeText(pgn);
      status = 'Annotated PGN copied to clipboard.';
    } catch {
      status = 'Clipboard access is unavailable in this browser.';
    }
  }

  function downloadAnnotatedPgn(): void {
    const pgn = reviewPgn();
    if (!pgn || !game) return;
    const players = `${game.tags.White || 'White'}-${game.tags.Black || 'Black'}`.replace(/[^\w.-]+/g, '_').slice(0, 70);
    const url = URL.createObjectURL(new Blob([pgn], { type: 'application/x-chess-pgn' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${players}-review.pgn`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status = 'Annotated PGN downloaded.';
  }

  function moveNumber(move: ReviewedMove): string {
    const before = nodes[move.ply - 1] ? parseFen(nodes[move.ply - 1].fen) : null;
    return before ? `${before.fullmove}${before.turn === 'w' ? '.' : '…'}` : `${Math.ceil(move.ply / 2)}${move.side === 'w' ? '.' : '…'}`;
  }

  function nodeMoveNumber(index: number): string {
    const before = nodes[index] ? parseFen(nodes[index].fen) : null;
    if (!before) return '';
    return before.turn === 'w' ? `${before.fullmove}.` : index === 0 ? `${before.fullmove}…` : '';
  }
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>

<SiteHeader pageTitle="Review" />
<main id="main" class="review-main">
  {#if loadError}
    <section class="panel load-error">
      <h1>Game unavailable</h1><p>{loadError}</p><a href="/app/library/">Return to Library</a>
    </section>
  {:else}
    <section class="panel board-panel" aria-label="Reviewed game board">
      <header class="game-header">
        <div><span>{game?.tags.White || 'White'} {game?.tags.WhiteElo ? `(${game.tags.WhiteElo})` : ''}</span><strong>{game?.result || '*'}</strong><span>{game?.tags.Black || 'Black'} {game?.tags.BlackElo ? `(${game.tags.BlackElo})` : ''}</span></div>
        <small>{game?.tags.Event || collectionName} {game?.tags.Date ? `· ${game.tags.Date}` : ''}</small>
      </header>
      <div class="board-wrap"><div class="board-shell"><div id="ground" bind:this={groundElement}></div></div></div>
      <div class="nav" aria-label="Move navigation">
        <button type="button" on:click={() => goTo(0)} aria-label="First move">|◀</button>
        <button type="button" on:click={() => goTo(currentPly - 1)} aria-label="Previous move">◀</button>
        <span>{currentPly}/{Math.max(0, nodes.length - 1)}</span>
        <button type="button" on:click={() => goTo(currentPly + 1)} aria-label="Next move">▶</button>
        <button type="button" on:click={() => goTo(nodes.length - 1)} aria-label="Last move">▶|</button>
        <button type="button" on:click={flipBoard} aria-label="Flip board">⇅</button>
      </div>
      <div class="move-list">
        {#each nodes.slice(1) as node, index}
          {@const moveReview = reviewedByPly.get(index + 1)}
          <button type="button" class:current={currentPly === index + 1} class:error={moveReview && ['inaccuracy', 'mistake', 'blunder'].includes(moveReview.class)} on:click={() => goTo(index + 1)}>
            {#if nodeMoveNumber(index)}<span>{nodeMoveNumber(index)}</span>{/if}{node.san}{#if moveReview}<i class="class-{moveReview.class}">{classLabel[moveReview.class]}</i>{/if}
          </button>
        {/each}
      </div>
    </section>

    <aside class="panel review-sidebar" aria-label="Game review">
      <section class="review-controls">
        <div class="section-title"><span>Engine review</span><small>Runs locally</small></div>
        <div class="review-setup">
          <label for="reviewDepth">Stockfish depth<select id="reviewDepth" bind:value={depth} disabled={running} on:change={() => void restoreReviewForDepth()}><option value={10}>Quick · 10</option><option value={12}>Standard · 12</option><option value={14}>Deep · 14</option></select></label>
          {#if running}<button type="button" on:click={stopReview}>Stop</button>{:else}<button class="primary" type="button" on:click={() => void startReview()} disabled={!game}>Review game</button>{/if}
        </div>
        {#if running}<progress max={progressTotal} value={progress}></progress>{/if}
        <div class="status" role="status" aria-live="polite">{status}</div>
      </section>

      {#if review}
        <section class="accuracy-section">
          <div class="accuracy-card white"><span>White accuracy</span><strong>{review.accuracy.white.toFixed(1)}</strong><small>{review.counts.white.blunder} blunders · {review.counts.white.mistake} mistakes</small></div>
          <div class="accuracy-card black"><span>Black accuracy</span><strong>{review.accuracy.black.toFixed(1)}</strong><small>{review.counts.black.blunder} blunders · {review.counts.black.mistake} mistakes</small></div>
        </section>
        <section class="chart-section"><h2>Winning chances</h2><div class="chart">{@html chartSvg}</div></section>
        {#if currentReviewedMove}
          <section class="current-review class-{currentReviewedMove.class}">
            <div><span class="classification">{classLabel[currentReviewedMove.class]}</span><strong>{moveNumber(currentReviewedMove)} {currentReviewedMove.san}</strong></div>
            <p>White winning chances: {Math.round(currentReviewedMove.winBefore * 100)}% → {Math.round(currentReviewedMove.winAfter * 100)}%</p>
            {#if currentReviewedMove.bestUci && currentReviewedMove.class !== 'best' && currentReviewedMove.class !== 'forced'}<p>Engine preference: <code>{currentReviewedMove.bestUci}</code></p>{/if}
          </section>
        {/if}
        <section class="critical-section">
          <h2>Critical moments</h2>
          {#if review.criticalMoves.length}
            <ol>{#each review.criticalMoves as move}<li><button type="button" on:click={() => goTo(move.ply)}><span class="classification class-{move.class}">{classLabel[move.class]}</span><strong>{moveNumber(move)} {move.san}</strong><small>{Math.round(move.moverLoss * 100)}% winning-chance loss{move.bestUci ? ` · best ${move.bestUci}` : ''}</small></button></li>{/each}</ol>
          {:else}<p class="quiet">No inaccuracies, mistakes, or blunders were detected.</p>{/if}
        </section>
        <section class="review-actions">
          <button type="button" on:click={() => void copyAnnotatedPgn()}>Copy annotated PGN</button>
          <button type="button" on:click={downloadAnnotatedPgn}>Download PGN</button>
          <a href={`/app/analysis/?collection=${encodeURIComponent(collectionId)}&gameId=${encodeURIComponent(gameId)}&game=${gameIndex}`}>Open in Analysis</a>
        </section>
      {:else}
        <section class="review-empty"><strong>Ready for review</strong><p>Stockfish evaluates every mainline position, then identifies accuracy, mistakes, and the most important moments.</p></section>
      {/if}
    </aside>
  {/if}
</main>

<style>
  .review-main{display:grid; grid-template-columns:minmax(0,1fr) minmax(370px,430px); gap:18px; align-items:start; max-width:1180px; margin:0 auto; padding:16px 24px 48px}
  .board-panel{min-width:0; padding:12px}
  .game-header{display:grid; gap:4px; margin-bottom:9px}
  .game-header>div{display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:10px; font-size:12px}
  .game-header>div span:last-child{text-align:right}
  .game-header strong{font-family:var(--mono); font-size:11px}
  .game-header small{color:var(--muted); font-size:9px; text-align:center}
  .board-shell{min-width:0}
  .nav{display:grid; grid-template-columns:repeat(2,1fr) auto repeat(3,1fr); gap:5px; margin-top:8px}
  .nav button{height:34px; padding:5px}
  .nav span{display:flex; min-width:46px; align-items:center; justify-content:center; color:var(--muted); font-family:var(--mono); font-size:9px}
  .move-list{display:flex; flex-wrap:wrap; gap:3px; max-height:175px; margin-top:8px; padding:8px; overflow:auto; border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset)}
  .move-list button{display:inline-flex; gap:3px; align-items:center; min-height:25px; padding:3px 5px; border-color:transparent; background:transparent; font-family:var(--mono); font-size:10px}
  .move-list button.current{border-color:var(--accent); background:var(--accent); color:var(--on-accent)}
  .move-list button.error:not(.current){border-color:var(--rule-strong)}
  .move-list button span{color:var(--muted)}
  .move-list button i{padding:1px 3px; border-radius:3px; background:var(--rule); color:var(--text-soft); font-size:7px; font-style:normal}
  .review-sidebar{position:sticky; top:74px; overflow:hidden}
  .review-controls,.accuracy-section,.chart-section,.current-review,.critical-section,.review-actions,.review-empty{padding:12px; border-top:1px solid var(--rule)}
  .review-controls{border-top:0}
  .section-title{display:flex; justify-content:space-between; align-items:center; margin-bottom:9px; font-family:var(--mono); font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase}
  .section-title small{color:var(--accent); font-size:8px}
  .review-setup{display:grid; grid-template-columns:1fr auto; gap:7px; align-items:end}
  .review-setup label{display:grid; gap:4px; color:var(--muted); font-family:var(--mono); font-size:9px}
  .review-setup button{min-height:37px}
  progress{width:100%; height:7px; margin-top:9px; accent-color:var(--accent)}
  .status{margin-top:8px; color:var(--text-soft); font-family:var(--mono); font-size:9px; line-height:1.45}
  .accuracy-section{display:grid; grid-template-columns:1fr 1fr; gap:7px}
  .accuracy-card{display:grid; padding:10px; border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset)}
  .accuracy-card span,.accuracy-card small{color:var(--muted); font-size:8px}
  .accuracy-card strong{font-family:var(--mono); font-size:27px; line-height:1.1}
  .accuracy-card.black{background:var(--ink); color:var(--paper)}
  .accuracy-card.black strong{color:inherit}
  .chart-section h2,.critical-section h2{margin:0 0 7px; font-size:11px}
  .chart{overflow:hidden}
  .chart :global(svg){display:block; width:100%; height:auto}
  .current-review{border-left:3px solid var(--accent); background:var(--panel-inset)}
  .current-review>div{display:flex; align-items:center; gap:8px}
  .current-review strong{font-family:var(--mono); font-size:12px}
  .current-review p{margin:6px 0 0; color:var(--text-soft); font-size:10px}
  .classification{display:inline-flex; padding:3px 5px; border-radius:4px; background:var(--accent-soft); color:var(--accent); font-family:var(--mono); font-size:8px; font-weight:700; text-transform:uppercase}
  .class-blunder,.class-mistake{background:color-mix(in srgb,var(--warn) 14%,transparent); color:var(--warn)}
  .class-inaccuracy{background:color-mix(in srgb,#d38b16 15%,transparent); color:#b16c00}
  .critical-section ol{display:grid; gap:5px; margin:0; padding:0; list-style:none}
  .critical-section li button{display:grid; grid-template-columns:auto 1fr; gap:2px 7px; width:100%; padding:7px; text-align:left}
  .critical-section li strong{font-family:var(--mono); font-size:10px}
  .critical-section li small{grid-column:2; color:var(--muted); font-size:8px}
  .quiet,.review-empty p{margin:5px 0 0; color:var(--muted); font-size:10px; line-height:1.5}
  .review-empty{padding-block:28px; text-align:center}
  .review-actions{display:flex; flex-wrap:wrap; gap:6px}
  .review-actions button,.review-actions a{display:inline-flex; min-height:32px; align-items:center; padding:5px 8px; border:1px solid var(--border-input); border-radius:var(--radius-input); color:var(--ink); font-size:9px; text-decoration:none}
  .load-error{grid-column:1/-1; padding:30px; text-align:center}
  .load-error h1{margin-top:0}.load-error p{color:var(--muted)}.load-error a{color:var(--accent)}
  @media (max-width:900px){.review-main{grid-template-columns:1fr; max-width:680px; padding:16px}.review-sidebar{position:static}}
  @media (max-width:600px){.review-main{padding:12px 10px 40px; gap:12px}.accuracy-section{grid-template-columns:1fr 1fr}.game-header>div{font-size:10px}}
</style>
