<script lang="ts">
  import { onMount } from 'svelte';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  import { parsePgnGames } from '../../../chess/pgn';
  import { fetchGameHistoryPgn, type ImportColor, type ImportSite } from '../../../lc0/gameImport';
  import {
    defaultPgnCollectionName,
    deletePgnCollection,
    exportPgnDatabaseBackup,
    formatPgnCollectionSummary,
    importPgnDatabaseBackup,
    listPgnCollectionGames,
    listPgnCollections,
    loadPgnCollection,
    pgnDatabaseAvailable,
    pgnDatabaseBackupFilename,
    rebuildPgnCollectionIndex,
    savePgnCollection,
    type PgnCollectionGame,
    type PgnCollectionSource,
    type PgnCollectionSummary,
  } from '../../../lc0/pgnDatabase';

  const title = '0x88 Chess — game library';
  const description = 'Import and keep your Lichess, Chess.com, and PGN game collections locally in your browser.';

  let collections: PgnCollectionSummary[] = [];
  let selectedId = '';
  let site: ImportSite = 'lichess';
  let username = '';
  let color: ImportColor = '';
  let maxGames = 40;
  let collectionName = '';
  let pgn = '';
  let source: PgnCollectionSource = 'manual';
  let sourceUsername = '';
  let sourceColor = '';
  let previewCount = 0;
  let busy = false;
  let status = 'Your games stay in this browser. Import a PGN or fetch recent public games to begin.';
  let pgnFileInput: HTMLInputElement;
  let backupFileInput: HTMLInputElement;
  let games: PgnCollectionGame[] = [];
  let gameFilter = '';

  $: normalizedGameFilter = gameFilter.trim().toLocaleLowerCase();
  $: filteredGames = normalizedGameFilter
    ? games.filter((game) => [game.white, game.black, game.event, game.opening, game.eco, game.date, game.result, game.whiteElo, game.blackElo]
      .some((value) => String(value ?? '').toLocaleLowerCase().includes(normalizedGameFilter)))
    : games;

  onMount(() => {
    void refreshCollections();
  });

  async function refreshCollections(preferredId = selectedId): Promise<void> {
    if (!pgnDatabaseAvailable()) {
      status = 'IndexedDB is unavailable in this browser; the local library cannot be used.';
      return;
    }
    try {
      collections = await listPgnCollections();
      selectedId = collections.some((entry) => entry.id === preferredId) ? preferredId : '';
    } catch (error) {
      status = `Library failed to open: ${(error as Error).message}`;
    }
  }

  function previewPgn(raw = pgn): number {
    const games = parsePgnGames(raw);
    if (!games.length) throw new Error('No games found in the PGN');
    previewCount = games.length;
    return games.length;
  }

  function setDraft(raw: string, nextSource: PgnCollectionSource, nextUsername = '', nextColor = ''): void {
    pgn = raw;
    source = nextSource;
    sourceUsername = nextUsername;
    sourceColor = nextColor;
    selectedId = '';
    collectionName = defaultPgnCollectionName(nextSource, nextUsername);
    const count = previewPgn(raw);
    status = `Ready to save ${count} ${count === 1 ? 'game' : 'games'} as a new local collection.`;
  }

  async function fetchGames(): Promise<void> {
    if (!username.trim()) {
      status = 'Enter a Lichess or Chess.com username.';
      return;
    }
    busy = true;
    status = `Fetching ${username.trim()}'s recent games from ${site === 'chesscom' ? 'Chess.com' : 'Lichess'}…`;
    try {
      const raw = await fetchGameHistoryPgn(site, username, { max: maxGames, color }, fetch);
      if (!raw.trim()) {
        status = 'No matching games were found.';
        return;
      }
      setDraft(raw, site, username.trim(), color);
    } catch (error) {
      status = `Import failed: ${(error as Error).message || 'network request failed'}`;
    } finally {
      busy = false;
    }
  }

  function importPastedPgn(): void {
    if (!pgn.trim()) {
      status = 'Paste PGN text or choose a PGN file first.';
      return;
    }
    try {
      setDraft(pgn, 'manual');
    } catch (error) {
      previewCount = 0;
      status = `PGN import failed: ${(error as Error).message}`;
    }
  }

  async function importPgnFile(file: File | undefined): Promise<void> {
    if (!file) return;
    try {
      setDraft(await file.text(), 'manual');
      collectionName = file.name.replace(/\.pgn$/i, '').trim() || collectionName;
    } catch (error) {
      status = `Could not read ${file.name}: ${(error as Error).message}`;
    } finally {
      pgnFileInput.value = '';
    }
  }

  async function saveDraft(): Promise<void> {
    if (!pgn.trim()) {
      status = 'Import games before saving a collection.';
      return;
    }
    busy = true;
    status = 'Parsing and indexing games…';
    try {
      const indexed = rebuildPgnCollectionIndex({
        id: selectedId || undefined,
        name: collectionName || defaultPgnCollectionName(source, sourceUsername),
        pgn,
        gameCount: previewPgn(),
        source,
        username: sourceUsername,
        color: sourceColor,
      });
      const record = await savePgnCollection(indexed);
      await refreshCollections(record.id);
      collectionName = record.name;
      previewCount = record.gameCount;
      games = await listPgnCollectionGames(record.id);
      gameFilter = '';
      status = `Saved “${record.name}” with ${record.gameCount} ${record.gameCount === 1 ? 'game' : 'games'} and ${record.indexedPositionCount ?? 0} indexed positions.`;
    } catch (error) {
      status = `Save failed: ${(error as Error).message}`;
    } finally {
      busy = false;
    }
  }

  async function openCollection(id: string): Promise<void> {
    busy = true;
    try {
      const record = await loadPgnCollection(id);
      if (!record) throw new Error('Collection not found');
      selectedId = record.id;
      collectionName = record.name;
      pgn = record.pgn;
      source = record.source;
      sourceUsername = record.username ?? '';
      sourceColor = record.color ?? '';
      previewCount = record.gameCount;
      games = await listPgnCollectionGames(record.id);
      gameFilter = '';
      status = `Loaded “${record.name}”. Editing and saving will update this collection.`;
    } catch (error) {
      status = `Load failed: ${(error as Error).message}`;
    } finally {
      busy = false;
    }
  }

  function newCollection(): void {
    selectedId = '';
    collectionName = '';
    pgn = '';
    source = 'manual';
    sourceUsername = '';
    sourceColor = '';
    previewCount = 0;
    games = [];
    gameFilter = '';
    status = 'New collection. Fetch games, choose a PGN file, or paste PGN text.';
  }

  async function removeCollection(id: string): Promise<void> {
    const entry = collections.find((collection) => collection.id === id);
    if (!entry || !window.confirm(`Delete local collection “${entry.name}”?`)) return;
    try {
      await deletePgnCollection(id);
      if (selectedId === id) newCollection();
      await refreshCollections();
      status = `Deleted “${entry.name}”.`;
    } catch (error) {
      status = `Delete failed: ${(error as Error).message}`;
    }
  }

  function downloadText(filename: string, content: string, type: string): void {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function exportBackup(): Promise<void> {
    try {
      const backup = await exportPgnDatabaseBackup();
      downloadText(pgnDatabaseBackupFilename(new Date(backup.exportedAt)), JSON.stringify(backup, null, 2), 'application/json');
      status = `Exported ${backup.collections.length} local ${backup.collections.length === 1 ? 'collection' : 'collections'}.`;
    } catch (error) {
      status = `Backup export failed: ${(error as Error).message}`;
    }
  }

  async function importBackup(file: File | undefined): Promise<void> {
    if (!file) return;
    busy = true;
    status = `Importing ${file.name} and rebuilding position indexes…`;
    try {
      const count = await importPgnDatabaseBackup(JSON.parse(await file.text()) as unknown);
      await refreshCollections();
      status = `Imported ${count} ${count === 1 ? 'collection' : 'collections'} from ${file.name}.`;
    } catch (error) {
      status = `Backup import failed: ${(error as Error).message}`;
    } finally {
      busy = false;
      backupFileInput.value = '';
    }
  }
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>

<SiteHeader pageTitle="Library" />
<main id="main" class="library-main">
  <header class="library-heading">
    <div>
      <span class="eyebrow">Local-first game workspace</span>
      <h1>Your game library</h1>
      <p>Import public games or PGN files, keep collections locally, and send them to Analysis. Nothing is uploaded by 0x88.</p>
    </div>
    <div class="heading-actions">
      <button type="button" on:click={newCollection}>New collection</button>
      <button type="button" on:click={exportBackup} disabled={busy || !collections.length}>Export backup</button>
      <button type="button" on:click={() => backupFileInput.click()} disabled={busy}>Import backup</button>
      <input bind:this={backupFileInput} class="file-input" type="file" accept="application/json,.json" on:change={(event) => void importBackup((event.currentTarget as HTMLInputElement).files?.[0])} />
    </div>
  </header>

  <div class="library-grid">
    <section class="panel import-panel" aria-labelledby="importHeading">
      <div class="panel-heading">
        <div><span class="step">01</span><h2 id="importHeading">Import games</h2></div>
        {#if previewCount}<span class="count-badge">{previewCount} {previewCount === 1 ? 'game' : 'games'}</span>{/if}
      </div>

      <div class="remote-grid">
        <div class="field"><label for="importSite">Source</label>
          <select id="importSite" bind:value={site}><option value="lichess">Lichess</option><option value="chesscom">Chess.com</option></select></div>
        <div class="field username"><label for="importUsername">Username</label>
          <input id="importUsername" bind:value={username} autocomplete="off" spellcheck="false" placeholder="player name" on:keydown={(event) => { if (event.key === 'Enter') void fetchGames(); }} /></div>
        <div class="field"><label for="importColor">Color</label>
          <select id="importColor" bind:value={color}><option value="">Both</option><option value="white">White</option><option value="black">Black</option></select></div>
        <div class="field max-field"><label for="importMax">Games</label>
          <input id="importMax" type="number" min="1" max="300" bind:value={maxGames} /></div>
        <button class="primary fetch-button" type="button" on:click={() => void fetchGames()} disabled={busy}>Fetch games</button>
      </div>

      <div class="divider"><span>or import PGN</span></div>
      <textarea bind:value={pgn} spellcheck="false" aria-label="PGN text" placeholder="Paste one or more PGN games here"></textarea>
      <div class="pgn-actions">
        <button type="button" on:click={() => pgnFileInput.click()} disabled={busy}>Choose PGN file</button>
        <button type="button" on:click={importPastedPgn} disabled={busy || !pgn.trim()}>Check PGN</button>
        <input bind:this={pgnFileInput} class="file-input" type="file" accept="application/x-chess-pgn,.pgn,text/plain" on:change={(event) => void importPgnFile((event.currentTarget as HTMLInputElement).files?.[0])} />
      </div>

      <div class="save-box">
        <div class="field"><label for="collectionName">Collection name</label>
          <input id="collectionName" bind:value={collectionName} placeholder="My games" /></div>
        <button class="primary" type="button" on:click={() => void saveDraft()} disabled={busy || !pgn.trim()}>{selectedId ? 'Update collection' : 'Save collection'}</button>
      </div>
      <div class="status" role="status" aria-live="polite" class:working={busy}>{status}</div>
    </section>

    <section class="panel collections-panel" aria-labelledby="collectionsHeading">
      <div class="panel-heading">
        <div><span class="step">02</span><h2 id="collectionsHeading">Saved collections</h2></div>
        <span class="collection-total">{collections.length} local</span>
      </div>
      {#if collections.length}
        <div class="collection-list">
          {#each collections as collection}
            <article class:selected={collection.id === selectedId} class="collection-card">
              <button class="collection-main" type="button" on:click={() => void openCollection(collection.id)}>
                <strong>{collection.name}</strong>
                <span>{collection.gameCount} games · {collection.indexedPositionCount ?? 0} positions</span>
                <small>{formatPgnCollectionSummary(collection)}</small>
              </button>
              <div class="collection-actions">
                <a href={`/app/analysis/?collection=${encodeURIComponent(collection.id)}`}>Analyze</a>
                <button type="button" on:click={() => void removeCollection(collection.id)}>Delete</button>
              </div>
            </article>
          {/each}
        </div>
      {:else}
        <div class="empty-state">
          <span aria-hidden="true">♙</span>
          <strong>No saved games yet</strong>
          <p>Your imported collections will appear here and remain available between visits.</p>
        </div>
      {/if}
      <div class="next-step">
        <span>Next</span>
        <div><strong>Opening explorer</strong><p>Position statistics and collection filters will live here rather than crowding the Analysis board.</p></div>
      </div>
    </section>
  </div>

  {#if selectedId && games.length}
    <section class="panel games-panel" aria-labelledby="gamesHeading">
      <div class="games-heading">
        <div>
          <span class="step">03</span>
          <h2 id="gamesHeading">Games in {collectionName}</h2>
          <p>Select a game for unrestricted engine analysis. Guided review will use this same game list next.</p>
        </div>
        <label class="game-search" for="gameFilter">
          <span>Filter games</span>
          <input id="gameFilter" type="search" bind:value={gameFilter} placeholder="player, event, opening, date…" />
        </label>
      </div>
      <div class="game-count">Showing {filteredGames.length} of {games.length} {games.length === 1 ? 'game' : 'games'}</div>
      {#if filteredGames.length}
        <div class="game-table-wrap">
          <table class="game-table">
            <thead><tr><th>White</th><th>Black</th><th>Result</th><th>Date</th><th>Event / opening</th><th><span class="visually-hidden">Actions</span></th></tr></thead>
            <tbody>
              {#each filteredGames as game}
                <tr>
                  <td><strong>{game.white}</strong>{#if game.whiteElo}<small>{game.whiteElo}</small>{/if}</td>
                  <td><strong>{game.black}</strong>{#if game.blackElo}<small>{game.blackElo}</small>{/if}</td>
                  <td><span class="result result-{game.result.replaceAll('/', '-')}">{game.result}</span></td>
                  <td>{game.date || '—'}</td>
                  <td><strong>{game.event || 'Untitled game'}</strong><small>{game.opening || game.eco || `${game.plyCount} ply`}{#if game.latestReview} · reviewed d{game.latestReview.depth}{/if}</small></td>
                  <td><div class="game-actions"><a class="review-game" href={`/app/review/?collection=${encodeURIComponent(selectedId)}&gameId=${encodeURIComponent(game.id)}&game=${game.order}`}>{game.latestReview ? `${game.latestReview.accuracy.white.toFixed(0)} / ${game.latestReview.accuracy.black.toFixed(0)}` : 'Review'}</a><a class="analyze-game" href={`/app/analysis/?collection=${encodeURIComponent(selectedId)}&gameId=${encodeURIComponent(game.id)}&game=${game.order}`}>Analyze</a></div></td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <div class="no-results">No games match “{gameFilter.trim()}”.</div>
      {/if}
    </section>
  {/if}
</main>

<style>
  .library-main{width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:28px 0 56px}
  .library-heading{display:flex; justify-content:space-between; align-items:flex-end; gap:24px; margin-bottom:18px}
  .library-heading h1{margin:5px 0 5px; font-size:clamp(25px,4vw,38px); letter-spacing:-.035em}
  .library-heading p{max-width:700px; margin:0; color:var(--text-soft); font-size:13px; line-height:1.55}
  .eyebrow,.step{color:var(--accent); font-family:var(--mono); font-size:10px; font-weight:700; letter-spacing:.11em; text-transform:uppercase}
  .heading-actions{display:flex; flex-wrap:wrap; justify-content:flex-end; gap:6px}
  .heading-actions button{min-height:34px; padding:6px 10px}
  .library-grid{display:grid; grid-template-columns:minmax(0,1.15fr) minmax(340px,.85fr); gap:16px; align-items:start}
  .panel{overflow:hidden}
  .import-panel,.collections-panel{padding:16px}
  .panel-heading{display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; padding-bottom:11px; border-bottom:1px solid var(--rule)}
  .panel-heading>div{display:flex; align-items:baseline; gap:9px}
  .panel-heading h2{margin:0; font-size:16px; letter-spacing:-.01em}
  .count-badge,.collection-total{padding:4px 7px; border:1px solid var(--rule); border-radius:999px; color:var(--muted); font-family:var(--mono); font-size:9px}
  .remote-grid{display:grid; grid-template-columns:120px minmax(150px,1fr) 100px 74px auto; gap:7px; align-items:end}
  .field{min-width:0; margin:0}
  .field input,.field select{width:100%; box-sizing:border-box}
  .fetch-button{min-height:37px; white-space:nowrap}
  .divider{display:flex; align-items:center; gap:8px; margin:16px 0 9px; color:var(--muted); font-family:var(--mono); font-size:9px; text-transform:uppercase}
  .divider::before,.divider::after{content:''; height:1px; flex:1; background:var(--rule)}
  textarea{width:100%; min-height:180px; box-sizing:border-box; resize:vertical; font-family:var(--mono); font-size:11px; line-height:1.5}
  .pgn-actions{display:flex; gap:6px; margin-top:7px}
  .save-box{display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:end; margin-top:16px; padding:12px; border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset)}
  .save-box button{min-height:37px}
  .status{min-height:18px; margin-top:10px; color:var(--text-soft); font-family:var(--mono); font-size:10px; line-height:1.55}
  .status.working{color:var(--accent)}
  .file-input{display:none}
  .collection-list{display:grid; gap:8px; max-height:570px; overflow:auto}
  .collection-card{border:1px solid var(--rule); border-radius:var(--radius-sm); background:var(--panel-inset); transition:border-color .15s,background .15s}
  .collection-card.selected{border-color:var(--accent); background:var(--accent-soft)}
  .collection-main{display:grid; width:100%; padding:11px; border:0; background:transparent; text-align:left; box-shadow:none}
  .collection-main:hover{background:transparent; border-color:transparent}
  .collection-main strong{color:var(--ink); font-size:13px}
  .collection-main span{margin-top:4px; color:var(--text-soft); font-family:var(--mono); font-size:10px}
  .collection-main small{margin-top:4px; overflow:hidden; color:var(--muted); font-size:9px; text-overflow:ellipsis; white-space:nowrap}
  .collection-actions{display:flex; gap:6px; padding:0 10px 10px}
  .collection-actions a,.collection-actions button{min-height:29px; padding:5px 8px; font-size:10px}
  .collection-actions a{display:inline-flex; align-items:center; border:1px solid var(--border-input); border-radius:var(--radius-input); color:var(--ink); text-decoration:none}
  .empty-state{display:grid; justify-items:center; padding:46px 20px; text-align:center; color:var(--muted)}
  .empty-state span{font-size:34px; color:var(--accent)}
  .empty-state strong{margin-top:8px; color:var(--ink)}
  .empty-state p{max-width:290px; margin:5px 0 0; font-size:11px; line-height:1.5}
  .next-step{display:grid; grid-template-columns:auto 1fr; gap:9px; margin-top:14px; padding:11px; border:1px dashed var(--rule-strong); border-radius:var(--radius-sm)}
  .next-step>span{align-self:start; padding:2px 5px; border-radius:3px; background:var(--accent-soft); color:var(--accent); font-family:var(--mono); font-size:8px; font-weight:700; text-transform:uppercase}
  .next-step strong{font-size:11px}
  .next-step p{margin:2px 0 0; color:var(--muted); font-size:10px; line-height:1.45}
  .games-panel{margin-top:16px; padding:16px}
  .games-heading{display:flex; justify-content:space-between; align-items:end; gap:20px; padding-bottom:12px; border-bottom:1px solid var(--rule)}
  .games-heading>div{display:grid; grid-template-columns:auto 1fr; align-items:baseline; gap:0 9px}
  .games-heading h2{margin:0; font-size:16px}
  .games-heading p{grid-column:2; max-width:620px; margin:3px 0 0; color:var(--muted); font-size:10px; line-height:1.45}
  .game-search{display:grid; gap:4px; width:min(320px,40%); color:var(--muted); font-family:var(--mono); font-size:9px; text-transform:uppercase}
  .game-search input{width:100%; box-sizing:border-box; text-transform:none}
  .game-count{padding:9px 2px 7px; color:var(--muted); font-family:var(--mono); font-size:9px}
  .game-table-wrap{overflow-x:auto; border:1px solid var(--rule); border-radius:var(--radius-sm)}
  .game-table{width:100%; border-collapse:collapse; font-size:11px}
  .game-table th{padding:7px 10px; background:var(--panel-inset); color:var(--muted); font-family:var(--mono); font-size:8px; letter-spacing:.06em; text-align:left; text-transform:uppercase}
  .game-table td{padding:9px 10px; border-top:1px solid var(--rule); color:var(--text-soft); vertical-align:middle}
  .game-table tbody tr:hover{background:var(--panel-inset)}
  .game-table td strong{display:block; color:var(--ink); font-size:11px; white-space:nowrap}
  .game-table td small{display:block; margin-top:2px; color:var(--muted); font-size:9px; white-space:nowrap}
  .result{display:inline-flex; min-width:42px; justify-content:center; padding:3px 5px; border:1px solid var(--rule-strong); border-radius:4px; color:var(--ink); font-family:var(--mono); font-size:9px; font-weight:700}
  .game-actions{display:flex; gap:5px}
  .analyze-game,.review-game{display:inline-flex; min-height:28px; align-items:center; padding:4px 8px; border:1px solid var(--border-input); border-radius:var(--radius-input); color:var(--ink); font-size:10px; text-decoration:none; white-space:nowrap}
  .review-game{border-color:var(--accent); background:var(--accent); color:var(--on-accent)}
  .analyze-game:hover{border-color:var(--accent); color:var(--accent)}
  .no-results{padding:32px; color:var(--muted); text-align:center; font-size:11px}
  .visually-hidden{position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0}
  @media (max-width:900px){
    .library-heading{align-items:flex-start; flex-direction:column}
    .heading-actions{justify-content:flex-start}
    .library-grid{grid-template-columns:1fr}
    .games-heading{align-items:start; flex-direction:column}
    .game-search{width:100%}
  }
  @media (max-width:680px){
    .library-main{width:calc(100% - 20px); padding-top:18px}
    .remote-grid{grid-template-columns:1fr 1fr}
    .remote-grid .username{grid-column:span 2}
    .fetch-button{grid-column:span 2}
    .save-box{grid-template-columns:1fr}
  }
</style>
