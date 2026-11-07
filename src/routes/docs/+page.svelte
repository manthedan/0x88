<script lang="ts">
import { onMount, tick } from 'svelte';
import { type BrowserCapabilityReport, detectBrowserCapabilities } from '$lib/browserCapabilities';
import SiteHeader from '$lib/components/SiteHeader.svelte';

const title = '0x88 Chess — documentation';
const description = 'How to use the 0x88 browser chess workspaces, fix WebGPU and download problems, and find each engine’s license and corresponding source.';

type TocLink = { id: string; label: string };
type TocItem = TocLink & { children?: TocLink[] };
type TocGroup = { label: string; id: string; items: TocItem[] };

// The rail, the mobile contents sheet, scroll-spy and search all read this
// outline. Its order must match the order of the sections on the page.
const toc: TocGroup[] = [
  {
    label: 'Get started',
    id: 'pages',
    items: [
      { id: 'pages-analysis', label: 'Analyse a position' },
      { id: 'pages-arena', label: 'Run an engine match' },
      { id: 'pages-play', label: 'Play a game' },
      { id: 'human-vs-computer', label: 'Human-like opponents' },
    ],
  },
  {
    label: 'Troubleshooting',
    id: 'troubleshooting',
    items: [
      { id: 'trouble-device', label: 'Check this browser' },
      { id: 'trouble-webgpu', label: 'WebGPU unavailable' },
      { id: 'trouble-downloads', label: 'Slow or repeated downloads' },
      { id: 'trouble-cache', label: 'Clear downloaded models' },
      { id: 'trouble-threads', label: 'Stockfish runs single-threaded' },
    ],
  },
  {
    label: 'Engines',
    id: 'engines',
    items: [
      { id: 'engines-lc0', label: 'Leela Chess Zero' },
      { id: 'engines-lqo', label: 'Leela Queen Odds (LQO)' },
      { id: 'engines-maia3', label: 'Maia3' },
      { id: 'engines-centipawn', label: 'Centipawn' },
      { id: 'engines-stockfish', label: 'Stockfish 19' },
      { id: 'engines-berserk', label: 'Berserk' },
      { id: 'engines-viridithas', label: 'Viridithas' },
      { id: 'engines-plentychess', label: 'PlentyChess' },
      { id: 'engines-stormphrax', label: 'Stormphrax' },
      { id: 'engines-reckless', label: 'Reckless' },
    ],
  },
  {
    label: 'Under the hood',
    id: 'under-the-hood',
    items: [
      { id: 'neural-runtimes', label: 'Neural browser runtimes' },
      { id: 'cpu-wasm', label: 'CPU WASM runtimes', children: [{ id: 'cpu-wasm-relaxed-simd', label: 'Relaxed SIMD' }] },
      { id: 'cdn', label: 'Artifact CDN & caching' },
    ],
  },
  {
    label: 'Licenses & source',
    id: 'licenses',
    items: [
      { id: 'licenses-per-engine', label: 'Per-engine source links' },
      { id: 'removal', label: 'Removal requests' },
    ],
  },
];

/** Every linkable heading in page order, with the group it sits in and its top-level rail item. */
type Located = { link: TocLink; group: TocGroup; top: TocLink };
const outline: Located[] = toc.flatMap((group) => [
  ...(group.id ? [{ link: { id: group.id, label: group.label }, group, top: { id: group.id, label: group.label } }] : []),
  ...group.items.flatMap((item) => [{ link: item, group, top: item }, ...(item.children ?? []).map((link) => ({ link, group, top: item }))]),
]);
const byId = new Map(outline.map((entry) => [entry.link.id, entry]));

// ---- scroll-spy -----------------------------------------------------------
let activeId = outline[0].link.id;
$: activeTopId = byId.get(activeId)?.top.id ?? '';
$: activeLabel = byId.get(activeId)?.link.label ?? 'Contents';
$: activeGroupId = byId.get(activeId)?.group.id ?? '';
let expandedGroups: Record<string, boolean> = { pages: true, troubleshooting: true };
let headerHeight = 64;
let rail: HTMLElement | null = null;
let railList: HTMLElement | null = null;

function updateActive(): void {
  // Slightly past the scroll-margin, so a heading reached by a link counts as current.
  const offset = headerHeight + (narrow ? 84 : 24) + 8;
  let current = outline[0].link.id;
  for (const { link } of outline) {
    const el = document.getElementById(link.id);
    if (!el) continue;
    if (el.getBoundingClientRect().top - offset <= 0) current = link.id;
    else break;
  }
  const doc = document.documentElement;
  if (window.innerHeight + window.scrollY >= doc.scrollHeight - 2) current = outline[outline.length - 1].link.id;
  if (current !== activeId) {
    activeId = current;
    const groupId = byId.get(current)?.group.id;
    if (groupId) expandedGroups = { ...expandedGroups, [groupId]: true };
    void tick().then(keepActiveVisible);
  }
}

function keepActiveVisible(): void {
  if (narrow || !railList) return;
  const link = railList.querySelector<HTMLElement>('a[aria-current="location"]');
  if (!link) return;
  const box = railList.getBoundingClientRect();
  const item = link.getBoundingClientRect();
  if (item.top < box.top + 8) railList.scrollTop -= box.top + 8 - item.top;
  else if (item.bottom > box.bottom - 8) railList.scrollTop += item.bottom - box.bottom + 8;
}

// ---- mobile contents sheet --------------------------------------------------
let narrow = false;
let tocOpen = false;
let tocToggle: HTMLButtonElement | null = null;
let mobileSearch: HTMLButtonElement | null = null;
let searchShortcut = 'Ctrl K';
let searchShortcutLabel = 'Control K';

function closeSheet(): void {
  tocOpen = false;
}

// ---- search -----------------------------------------------------------------
type SearchEntry = { id: string; title: string; group: string; parent: string; text: string; titleLc: string; textLc: string };
type SearchHit = { entry: SearchEntry; score: number; pre: string; match: string; post: string };
let index: SearchEntry[] = [];
let query = '';
let searchInput: HTMLInputElement | null = null;
let resultList: HTMLElement | null = null;
$: hits = searchDocs(query, index);
$: searching = query.trim().length > 0;

const CONTAINERS = '.page-block, .engine-entry, .trouble, .opponents, .runtime-block, .runtime-subsection, .source-links, .removal-block';

function cleanText(node: Element): string {
  const clone = node.cloneNode(true) as Element;
  // Headings are indexed as titles; the live device check is not documentation.
  clone.querySelectorAll('.anchor-link, .device-check, h2, h3, script, style').forEach((el) => el.remove());
  // Keep table cells and list items from running together ("PlayAGPL").
  clone.querySelectorAll('td, th, li, dt, dd, p, h4, caption, .sub').forEach((el) => el.append(' '));
  return (clone.textContent ?? '').replace(/\s+/g, ' ').trim();
}

function blockText(el: HTMLElement): string {
  if (el.tagName === 'SECTION') {
    const parts: string[] = [];
    for (const child of Array.from(el.children)) {
      if (child.matches(`h2, nav, ${CONTAINERS}`) || child.id) continue;
      parts.push(cleanText(child));
    }
    return parts.join(' ');
  }
  const container = el.parentElement?.closest(CONTAINERS);
  if (container) return cleanText(container);
  // A bare h3: gather its following siblings up to the next heading.
  const parts: string[] = [];
  let next = el.nextElementSibling;
  while (next && !/^H[23]$/.test(next.tagName)) {
    parts.push(cleanText(next));
    next = next.nextElementSibling;
  }
  return parts.join(' ');
}

function buildIndex(): SearchEntry[] {
  return outline.flatMap(({ link, group, top }) => {
    const el = document.getElementById(link.id);
    if (!el) return [];
    const text = blockText(el);
    return [{ id: link.id, title: link.label, group: group.label, parent: top.id === link.id ? '' : top.label, text, titleLc: link.label.toLowerCase(), textLc: text.toLowerCase() }];
  });
}

function searchDocs(raw: string, entries: SearchEntry[]): SearchHit[] {
  const terms = raw.toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return [];
  const found: SearchHit[] = [];
  for (const entry of entries) {
    let score = 0;
    let ok = true;
    for (const term of terms) {
      if (entry.titleLc.includes(term)) score += entry.titleLc.startsWith(term) ? 12 : 10;
      else if (entry.textLc.includes(term)) score += 1;
      else if (entry.group.toLowerCase().includes(term)) score += 2;
      else {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    found.push({ entry, score, ...snippet(entry, terms) });
  }
  return found.sort((a, b) => b.score - a.score).slice(0, 12);
}

function snippet(entry: SearchEntry, terms: string[]): { pre: string; match: string; post: string } {
  const term = terms.find((t) => entry.textLc.includes(t));
  if (!term) return { pre: entry.text.slice(0, 110) + (entry.text.length > 110 ? '…' : ''), match: '', post: '' };
  const at = entry.textLc.indexOf(term);
  const start = Math.max(0, at - 44);
  const end = Math.min(entry.text.length, at + term.length + 70);
  return {
    pre: (start > 0 ? '…' : '') + entry.text.slice(start, at),
    match: entry.text.slice(at, at + term.length),
    post: entry.text.slice(at + term.length, end) + (end < entry.text.length ? '…' : ''),
  };
}

function goTo(id: string): void {
  query = '';
  closeSheet();
  const el = document.getElementById(id);
  if (!el) return;
  history.replaceState(history.state, '', `#${id}`);
  el.scrollIntoView({ block: 'start' });
  const heading = el.matches('h2, h3, h4') ? el : el.querySelector<HTMLElement>('h2, h3, h4');
  heading?.setAttribute('tabindex', '-1');
  heading?.focus({ preventScroll: true });
}

function onSearchKey(event: KeyboardEvent): void {
  if (event.key === 'Escape') {
    if (query) query = '';
    else if (narrow) {
      closeSheet();
      mobileSearch?.focus();
    }
    return;
  }
  if (event.key === 'Enter' && hits.length) {
    event.preventDefault();
    goTo(hits[0].entry.id);
  } else if (event.key === 'ArrowDown' && hits.length) {
    event.preventDefault();
    resultList?.querySelector<HTMLElement>('a')?.focus();
  }
}

function onResultKey(event: KeyboardEvent): void {
  if (!(event.target instanceof HTMLElement)) return;
  const li = event.target.closest('li');
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    li?.nextElementSibling?.querySelector<HTMLElement>('a')?.focus();
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    const prev = li?.previousElementSibling?.querySelector<HTMLElement>('a');
    if (prev) prev.focus();
    else searchInput?.focus();
  } else if (event.key === 'Escape') {
    searchInput?.focus();
  }
}

async function focusSearch(): Promise<void> {
  if (narrow) tocOpen = true;
  await tick();
  searchInput?.focus();
  searchInput?.select();
}

function onWindowKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement | null;
  const typing = !!target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
  if (event.key === 'Escape' && tocOpen && !typing) {
    closeSheet();
    tocToggle?.focus();
  } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    void focusSearch();
  } else if (event.key === '/' && !typing) {
    event.preventDefault();
    void focusSearch();
  }
}

function onWindowClick(event: MouseEvent): void {
  if (!tocOpen || !rail) return;
  if (event.target instanceof Node && rail.contains(event.target)) return;
  closeSheet();
}

// ---- live browser check -----------------------------------------------------
let caps: BrowserCapabilityReport | null = null;
let capsFailed = false;

onMount(() => {
  if (/Mac|iPhone|iPad|iPod/.test(navigator.platform)) {
    searchShortcut = '⌘K';
    searchShortcutLabel = 'Command K';
  }
  index = buildIndex();
  const header = document.querySelector<HTMLElement>('.site-header');
  const media = window.matchMedia('(max-width: 860px)');
  const measure = () => {
    headerHeight = Math.round(header?.getBoundingClientRect().height ?? 64);
    narrow = media.matches;
    if (!narrow) tocOpen = false;
    updateActive();
  };
  let frame = 0;
  const onScroll = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      updateActive();
    });
  };
  measure();
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', measure);
  detectBrowserCapabilities()
    .then((report) => (caps = report))
    .catch(() => (capsFailed = true));
  return () => {
    window.removeEventListener('scroll', onScroll);
    window.removeEventListener('resize', measure);
    if (frame) cancelAnimationFrame(frame);
  };
});
</script>

<svelte:window on:keydown={onWindowKey} on:click={onWindowClick} />
<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>
<SiteHeader landing />
<main id="main">

<div class="doc-body" style="--hdr:{headerHeight}px">
<div class="wrap doc-grid">

<header class="doc-intro">
  <h1 id="doc-h1">Using 0x88</h1>
  <p class="intro-lede">Workspace guides, troubleshooting, engine details and source links.</p>
</header>

<!-- ===== Contents: left rail on desktop, sticky bar + sheet on phones ===== -->
<aside class="toc" class:open={tocOpen} aria-label="Documentation contents" bind:this={rail}>
  <div class="toc-toolbar">
    <button class="toc-toggle" type="button" aria-expanded={tocOpen} aria-controls="toc-panel" bind:this={tocToggle} on:click={() => (tocOpen = !tocOpen)}>
      <span class="toc-toggle-label">Contents</span>
      <span class="toc-current">{activeLabel}</span>
      <svg class="chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
    </button>
    <button class="mobile-search" type="button" aria-controls="toc-panel" bind:this={mobileSearch} on:click={focusSearch}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6" /><path d="m20 20-4.5-4.5" /></svg>
      Search docs
    </button>
  </div>
  <div id="toc-panel" class="toc-panel">
    <div class="search" role="search">
      <svg class="search-icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6" /><path d="m20 20-4.5-4.5" /></svg>
      <label class="visually-hidden" for="docSearch">Search the docs</label>
      <input
        id="docSearch"
        type="search"
        placeholder="Search docs"
        autocomplete="off"
        spellcheck="false"
        aria-describedby="docSearchHint"
        aria-controls="docSearchResults"
        bind:value={query}
        bind:this={searchInput}
        on:keydown={onSearchKey}
      />
      <span id="docSearchHint" class="visually-hidden">Press / or {searchShortcutLabel} to search</span>
      <span class="search-shortcuts" aria-hidden="true"><kbd>/</kbd><kbd>{searchShortcut}</kbd></span>
    </div>
    {#if searching}
      <div id="docSearchResults" class="results" aria-live="polite">
        {#if hits.length}
          <p class="results-count">{hits.length} {hits.length === 1 ? 'match' : 'matches'}</p>
          <ul bind:this={resultList}>
            {#each hits as hit (hit.entry.id)}
              <li>
                <a href={`#${hit.entry.id}`} on:click|preventDefault={() => goTo(hit.entry.id)} on:keydown={onResultKey}>
                  <span class="hit-path">{hit.entry.group}{hit.entry.parent ? ` › ${hit.entry.parent}` : ''}</span>
                  <span class="hit-title">{hit.entry.title}</span>
                  {#if hit.pre || hit.match}<span class="hit-snippet">{hit.pre}{#if hit.match}<mark>{hit.match}</mark>{/if}{hit.post}</span>{/if}
                </a>
              </li>
            {/each}
          </ul>
        {:else}
          <p class="no-results">No matches for “{query.trim()}”. Try an engine name, “WebGPU”, “cache” or “license”.</p>
        {/if}
      </div>
    {:else}
      <nav class="toc-groups" aria-label="On this page" bind:this={railList}>
        {#each toc as group (group.label)}
          <div class="toc-group">
            <div class="toc-group-heading" class:current-group={group.id === activeGroupId}>
              <a class="toc-group-label" href={`#${group.id}`} class:active={group.id === activeId} aria-current={group.id === activeId ? 'location' : group.id === activeGroupId ? 'true' : undefined} on:click={closeSheet}>{group.label}</a>
              <button type="button" class="group-toggle" aria-label={`${expandedGroups[group.id] ? 'Hide' : 'Show'} ${group.label} topics`} aria-expanded={!!expandedGroups[group.id]} aria-controls={`toc-${group.id}`} on:click={() => (expandedGroups = { ...expandedGroups, [group.id]: !expandedGroups[group.id] })}>
                <span aria-hidden="true">{group.items.length}</span>
                <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
              </button>
            </div>
            <ul id={`toc-${group.id}`} hidden={!expandedGroups[group.id]}>
              {#each group.items as item (item.id)}
                <li>
                  <a href={`#${item.id}`} class:active={item.id === activeId} class:in-active={item.id === activeTopId} aria-current={item.id === activeId ? 'location' : undefined} on:click={closeSheet}>{item.label}</a>
                  {#if item.children}
                    <ul>
                      {#each item.children as child (child.id)}
                        <li><a href={`#${child.id}`} class:active={child.id === activeId} aria-current={child.id === activeId ? 'location' : undefined} on:click={closeSheet}>{child.label}</a></li>
                      {/each}
                    </ul>
                  {/if}
                </li>
              {/each}
            </ul>
          </div>
        {/each}
      </nav>
    {/if}
  </div>
</aside>

{#snippet sectionNav(id: string)}
  {@const position = toc.findIndex((group) => group.id === id)}
  {@const previous = toc[position - 1]}
  {@const next = toc[position + 1]}
  <nav class="section-nav" aria-label={`Continue from ${toc[position].label}`}>
    {#if previous}<a href={`#${previous.id}`}><span class="section-direction">← Previous</span><span>{previous.label}</span></a>{/if}
    {#if next}<a class="next-section" href={`#${next.id}`}><span class="section-direction">Next →</span><span>{next.label}</span></a>{/if}
  </nav>
{/snippet}

<div class="doc-content">

<!-- ===== GET STARTED: THE WORKSPACES ===== -->
<section id="pages">
  <h2>Get started <a class="anchor-link" href="#pages" aria-label="Link to this section">#</a></h2>
  <p class="lead">0x88 has three workspaces: Analysis, Arena and Play. Everything runs in your browser. Each engine downloads the first time you use it and is reused after that.</p>

  <div class="page-block">
    <div class="task-heading">
      <h3 id="pages-analysis" aria-labelledby="analysis-title"><span id="analysis-title">Analyse a position</span><a class="anchor-link" href="#pages-analysis" aria-label="Link to this section">#</a></h3>
      <a class="open-link" href="/app/analysis/">Open Analysis <span aria-hidden="true">→</span></a>
    </div>
    <p>Run several engines on the same position and compare their best moves, evaluations and lines.</p>
    <ol class="steps">
      <li>Open Analysis. It starts with Lc0 (Small net) and Stockfish Lite on the starting position.</li>
      <li>Make moves on the board, or press <strong>Load</strong> to paste a PGN or FEN.</li>
      <li>Press <strong>Add engine</strong> for another engine, or <strong>Engines</strong> to change each engine’s net or build and its search budget (visits or depth).</li>
      <li>With <strong>Auto-analyze</strong> on (the default), analysis reruns whenever the position changes. Otherwise press <strong>Analyze</strong>.</li>
    </ol>
    <div class="task-reference">
    <h4>Exploring the results</h4>
    <ul>
      <li><strong>Engines tab</strong> &mdash; each engine’s evaluation and best lines.</li>
      <li><strong>Compare tab</strong> &mdash; every engine’s best move and evaluation in one table, with the difference between them.</li>
      <li><strong>Human tab (Maia3)</strong> &mdash; what players rated 1100 to 2200 choose in this position, with Maia3’s expected score for a human game. Loading it downloads about 28&nbsp;MB once.</li>
    </ul>
    </div>
    <p class="not-yet"><strong>Not available yet:</strong> game review (move accuracy and critical moments) and a local opening explorer for your own PGN collections are in development and hidden in the current build.</p>
  </div>

  <div class="page-block">
    <div class="task-heading">
      <h3 id="pages-arena" aria-labelledby="arena-title"><span id="arena-title">Run an engine match</span><a class="anchor-link" href="#pages-arena" aria-label="Link to this section">#</a></h3>
      <a class="open-link" href="/app/arena/">Open Arena <span aria-hidden="true">→</span></a>
    </div>
    <p>Play engines against each other and keep score. The project also uses Arena to compare browser engine builds and Lc0 networks.</p>
    <ol class="steps">
      <li>Open Arena. Under <strong>Matchup</strong>, pick a mode: a two-engine match, a round robin, or a gauntlet of Engine 1 against the field. Use <strong>+ Add engine</strong> for more seats.</li>
      <li>Under <strong>Time control &amp; openings</strong>, choose equal time per move or a fixed visit/depth budget, the number of games per opening, and the start position, the built-in opening suite or your own positions.</li>
      <li>Press <strong>Start match</strong>. Games play live on the board, with colours alternating between games.</li>
      <li>Read the standings: points, wins, draws, losses and an Elo estimate against the pool with a ±95% range.</li>
    </ol>
    <div class="task-reference">
    <h4>Following the match</h4>
    <ul>
      <li><strong>Charts</strong> &mdash; evaluation, time and speed for the current game.</li>
      <li><strong>Review</strong> &mdash; select a finished game or a move to replay it.</li>
      <li><strong>Export</strong> &mdash; <em>Export PGN</em> under <em>Game log &amp; PGN</em> lists every game as PGN.</li>
    </ul>
    </div>
  </div>

  <div class="page-block">
    <div class="task-heading">
      <h3 id="pages-play" aria-labelledby="play-title"><span id="play-title">Play a game</span><a class="anchor-link" href="#pages-play" aria-label="Link to this section">#</a></h3>
      <a class="open-link" href="/app/play/">Open Play <span aria-hidden="true">→</span></a>
    </div>
    <p>Play against Maia3, Leela Queen Odds or any of the analysis engines.</p>
    <ol class="steps">
      <li>Open Play. Maia3 is the default opponent; <strong>Change</strong> picks another.</li>
      <li>For Maia3, pick a rating preset, or set a rating from 600 to 2600 in 100-point steps under <em>Exact rating &amp; opponent details</em>. For other opponents, pick one of five strength levels from Fastest to Deep.</li>
      <li>Choose White, Black or Random, then press <strong>Play</strong>.</li>
      <li>Take back or resign during the game. Afterwards, copy or download the PGN, or press <strong>Analyze</strong> to open the game in Analysis.</li>
    </ol>
    <p>Your games are not sent anywhere. The engine, and its network if it has one, downloads on first use and is reused from your browser’s storage after that.</p>
  </div>

<!-- ===== HUMAN VS COMPUTER PLAY ===== -->
<div class="opponents">
  <h3 id="human-vs-computer">Human-like opponents <a class="anchor-link" href="#human-vs-computer" aria-label="Link to this section">#</a></h3>
  <p class="lead">Maia3 predicts a rating-conditioned human move distribution and human-game outcome. Leela Queen Odds is an Lc0-compatible network trained for games where the engine starts without its queen. The optional contempt controls belong to 0x88's search implementation and model an opponent with limited reply search.</p>

  <div class="callout info">
    <h4>Model boundaries</h4>
    <p>Maia3 is used directly as a move policy. Leela Queen Odds runs through 0x88's PUCT search from its specialized starting position. Contempt search does not make a conventional engine human-like or give it a personality.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Question</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Maia3 neural lane</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">LQO / Monty contempt lane</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">What is it modeling?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">The move distribution and expected result of real humans at a chosen rating.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">A queen-odds policy/value model, optionally combined with a limited-reply opponent model in 0x88's search.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">How does it pick moves?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Samples from, or takes argmax of, a rating-conditioned neural human policy.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Runs 0x88's PUCT search over the specialized network, with optional controls such as <code>drawScore</code> and <code>searchContemptLimit</code>.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">What should the UI promise?</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Rating-conditioned human move probabilities and expected game result.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">A specialized queen-odds opponent with clearly labeled search settings.</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Best fit</td>
        <td style="padding:10px">Human sparring, human-move explorer, and rating-conditioned analysis.</td>
        <td style="padding:10px">Queen-odds games and experiments comparing ordinary search with a limited-reply opponent model.</td>
      </tr>
    </tbody>
  </table>

  <p>Maia3's WDL output predicts the result of a human game at the selected rating; it is not an engine centipawn evaluation. The Leela Queen Odds and contempt experiments use 0x88's PUCT search. The full design note lives in <a href="https://github.com/manthedan/0x88/blob/main/docs/human_vs_computer_play.md">docs/human_vs_computer_play.md</a>.</p>
</div>
  {@render sectionNav('pages')}
</section>

<!-- ===== TROUBLESHOOTING ===== -->
<section id="troubleshooting">
  <h2>Troubleshooting <a class="anchor-link" href="#troubleshooting" aria-label="Link to this section">#</a></h2>
  <p class="lead">Most problems come from the browser: no WebGPU, no storage for downloaded models, or no shared memory for threads. Start with the check below, then find the symptom.</p>

  <div class="trouble">
    <h3 id="trouble-device">Check this browser <a class="anchor-link" href="#trouble-device" aria-label="Link to this section">#</a></h3>
    <p>0x88 checks these features when a page loads. The same check appears on <a href="/#device-details">Home</a> and in the <em>Browser capabilities</em> panel on Analysis, Arena and Play.</p>
    <div class="device-check" aria-live="polite" aria-busy={!caps && !capsFailed}>
      {#if capsFailed}
        <p class="dc-pending">Could not check this browser. Each engine still picks a fallback runtime when it loads.</p>
      {:else if !caps}
        <p class="dc-pending">Checking this browser…</p>
      {:else}
        <dl>
          <div>
            <dt>WebGPU</dt>
            <dd><span class="dc-state" data-tone={caps.webgpu === true ? 'ok' : 'warn'}>{caps.webgpu === true ? 'Available' : caps.webgpu === null ? 'No answer' : 'Unavailable'}</span>
              {caps.webgpu === true ? 'Neural engines can use the GPU.' : caps.webgpu === null ? 'The GPU did not answer within 4 seconds. See WebGPU unavailable below.' : 'See WebGPU unavailable below.'}</dd>
          </div>
          <div>
            <dt>WebAssembly</dt>
            <dd><span class="dc-state" data-tone={caps.wasm ? 'ok' : 'warn'}>{caps.wasm ? 'Available' : 'Unavailable'}</span>
              {caps.wasm ? `SIMD ${caps.wasmSimd ? 'yes' : 'no'} · relaxed SIMD ${caps.wasmRelaxedSimd ? 'yes' : 'no'}.` : 'The engines cannot run in this browser.'}</dd>
          </div>
          <div>
            <dt>Threads</dt>
            <dd><span class="dc-state" data-tone={caps.threadedWasm ? 'ok' : 'warn'}>{caps.threadedWasm ? 'Available' : 'Single-thread'}</span>
              {caps.threadedWasm ? `${caps.threads} logical thread${caps.threads === 1 ? '' : 's'} reported.` : 'Shared memory is unavailable, so CPU engines run on one thread.'}</dd>
          </div>
          <div>
            <dt>Model cache</dt>
            <dd><span class="dc-state" data-tone={caps.cacheStorage ? 'ok' : 'warn'}>{caps.cacheStorage ? 'Available' : 'Unavailable'}</span>
              {caps.cacheStorage ? 'Downloaded models are kept for later visits.' : 'Models may download again on every visit.'}</dd>
          </div>
        </dl>
      {/if}
    </div>
  </div>

  <div class="trouble">
    <h3 id="trouble-webgpu">WebGPU unavailable <a class="anchor-link" href="#trouble-webgpu" aria-label="Link to this section">#</a></h3>
    <p>0x88 asks the browser for a WebGPU adapter once per page load. If the browser has no WebGPU or returns no adapter, WebGPU counts as unavailable. If the request takes longer than 4 seconds, the check reports no answer. Without WebGPU:</p>
    <ul>
      <li><strong>CPU engines are unaffected.</strong> Stockfish, Reckless, Berserk, Viridithas, PlentyChess and Stormphrax always run as WebAssembly on the CPU.</li>
      <li><strong>Lc0 (Small net) and Maia3</strong> run through ONNX Runtime and fall back to WebAssembly on the CPU, which is slower.</li>
      <li><strong>Centipawn</strong> tries its TVMJS WebGPU runtime first and falls back to ONNX Runtime automatically.</li>
      <li><strong>Large Lc0 nets (t3, BT4) and Leela Queen Odds</strong> fall back to the CPU in Analysis and Play. Play lowers their visit budgets and warns that each move can take several seconds. In Arena they need WebGPU, and a match that includes them will not start without it.</li>
    </ul>
    <p>If your browser offers WebGPU behind a setting or flag, turn it on and reload the page so the check runs again.</p>
  </div>

  <div class="trouble">
    <h3 id="trouble-downloads">Slow or repeated downloads <a class="anchor-link" href="#trouble-downloads" aria-label="Link to this section">#</a></h3>
    <p>Engines and networks download the first time you use them, and the workspace shows download progress. The largest downloads are the neural networks:</p>
    <table class="doc-table compact">
      <thead><tr><th scope="col">Download</th><th scope="col">Approximate size</th></tr></thead>
      <tbody>
        <tr><td>Lc0 BT4-it332 net</td><td>178&nbsp;MB</td></tr>
        <tr><td>Lc0 t3-512 net</td><td>163&nbsp;MB</td></tr>
        <tr><td>Leela Queen Odds net</td><td>96&nbsp;MB</td></tr>
        <tr><td>Maia3 model</td><td>28&nbsp;MB</td></tr>
        <tr><td>Centipawn TVMJS runtime</td><td>26&nbsp;MB</td></tr>
      </tbody>
    </table>
    <ul>
      <li><strong>Later visits</strong> load neural models from the browser’s Cache Storage instead of the network. Each cached model is checked against its manifest, and a damaged or outdated copy is discarded and downloaded again.</li>
      <li><strong>Downloading every visit?</strong> If the browser reports too little free storage, 0x88 still loads the model but does not cache it, so it downloads again next time. The same happens when the check above shows the model cache as unavailable.</li>
      <li><strong>Low-memory devices.</strong> When a device reports 4&nbsp;GB of RAM or less, Play warns before loading a large Lc0 net. Lc0 Small, Maia3 and the CPU engines are lighter choices.</li>
    </ul>
  </div>

  <div class="trouble">
    <h3 id="trouble-cache">Clear downloaded models <a class="anchor-link" href="#trouble-cache" aria-label="Link to this section">#</a></h3>
    <p>0x88 has no in-app button for this. Downloaded models live in your browser’s storage for this site: Cache Storage entries named <code>lc0-browser-models-v1</code> and <code>maia3-browser-models-v1</code>, plus verification records in an IndexedDB database named <code>lc0-browser-model-cache-metadata</code>. Clearing this site’s data in your browser settings removes them; each model downloads again the next time you use it.</p>
  </div>

  <div class="trouble">
    <h3 id="trouble-threads">Stockfish runs single-threaded <a class="anchor-link" href="#trouble-threads" aria-label="Link to this section">#</a></h3>
    <p>Multi-threaded WebAssembly needs a cross-origin-isolated page with <code>SharedArrayBuffer</code>. 0x88 sends the headers for this, and when they take effect Analysis and Arena run Stockfish’s threaded build. If the check above shows <em>Single-thread</em>, Stockfish uses its single-threaded build instead. The other CPU engines are single-threaded builds either way.</p>
  </div>
  {@render sectionNav('troubleshooting')}
</section>

<!-- ===== ENGINES ===== -->
<section id="engines">
  <h2>The engines <a class="anchor-link" href="#engines" aria-label="Link to this section">#</a></h2>
  <p class="lead">These entries separate upstream engine and model facts from 0x88's browser packaging. The <a href="https://computerchess.org.uk/ccrl/4040/" rel="noopener">CCRL 40/15 list</a> and <a href="https://wiki.chessdom.org/Current_Engine_Status" rel="noopener">TCEC engine status</a> provide context for native builds under their own hardware and tournament conditions; neither measures the browser artifacts served here.</p>

  <table class="doc-table engine-table">
    <caption class="visually-hidden">Engines at a glance</caption>
    <thead>
      <tr><th scope="col">Engine</th><th scope="col">Runs on</th><th scope="col">How it picks moves</th><th scope="col">Where</th><th scope="col">License</th></tr>
    </thead>
    <tbody>
      <tr><th scope="row"><a href="#engines-lc0">Leela Chess Zero</a></th><td>GPU<span class="sub">CPU fallback</span></td><td>Neural network (Small, t3, BT4) with 0x88’s PUCT search</td><td>Analysis, Arena, Play</td><td><span class="lic gpl">GPL-3.0-or-later</span></td></tr>
      <tr><th scope="row"><a href="#engines-lqo">Leela Queen Odds</a></th><td>GPU<span class="sub">CPU fallback</span></td><td>Queen-odds network with 0x88’s PUCT search</td><td>Play</td><td><span class="lic agpl">AGPL-3.0</span></td></tr>
      <tr><th scope="row"><a href="#engines-maia3">Maia3</a></th><td>GPU<span class="sub">CPU fallback</span></td><td>Rating-conditioned human-move model, no search</td><td>Play, Analysis (Human tab)</td><td><span class="lic agpl">AGPL-3.0</span></td></tr>
      <tr><th scope="row"><a href="#engines-centipawn">Centipawn</a></th><td>GPU<span class="sub">ORT fallback</span></td><td>0x88’s experimental neural engine</td><td>Analysis, Arena, Play</td><td><span class="lic own">0x88 project</span></td></tr>
      <tr><th scope="row"><a href="#engines-stockfish">Stockfish 19</a></th><td>CPU</td><td>Alpha-beta search with NNUE, Lite and full builds</td><td>Analysis, Arena, Play</td><td><span class="lic gpl">GPL-3.0</span></td></tr>
      <tr><th scope="row"><a href="#engines-berserk">Berserk</a></th><td>CPU</td><td>Principal-variation search with bucketed NNUE</td><td>Analysis, Arena, Play</td><td><span class="lic gpl">GPL-3.0</span></td></tr>
      <tr><th scope="row"><a href="#engines-viridithas">Viridithas</a></th><td>CPU</td><td>Search with NNUE evaluation</td><td>Analysis, Arena, Play</td><td><span class="lic mit">MIT (pinned)</span></td></tr>
      <tr><th scope="row"><a href="#engines-plentychess">PlentyChess</a></th><td>CPU</td><td>Search with threat-input NNUE evaluation</td><td>Analysis, Arena, Play</td><td><span class="lic gpl">GPL-3.0</span></td></tr>
      <tr><th scope="row"><a href="#engines-stormphrax">Stormphrax</a></th><td>CPU</td><td>Search with self-play-trained NNUE evaluation</td><td>Analysis, Arena, Play</td><td><span class="lic gpl">GPL-3.0-or-later</span></td></tr>
      <tr><th scope="row"><a href="#engines-reckless">Reckless</a></th><td>CPU</td><td>Alpha-beta search with NNUE</td><td>Analysis, Arena, Play</td><td><span class="lic agpl">AGPL-3.0</span></td></tr>
    </tbody>
  </table>
  <p class="table-note">GPU engines use WebGPU when the browser has it; see <a href="#trouble-webgpu">WebGPU unavailable</a> for what each one does without it. Each engine’s license and source offer is listed under <a href="#licenses-per-engine">per-engine source links</a>.</p>

  <div class="engine-entry">
    <h3 id="engines-lc0">Leela Chess Zero <a class="anchor-link" href="#engines-lc0" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Project</dt><dd>Leela Chess Zero &middot; <a href="https://github.com/LeelaChessZero/lc0" rel="noopener">github.com/LeelaChessZero/lc0</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0-or-later</span></dd>
      <dt>Manifest</dt><dd><a href="/models/lc0/manifest.json">/models/lc0/manifest.json</a></dd>
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (browser runtime, ONNX export &amp; quantization scripts)</dd>
    </dl>
    <p>Leela Chess Zero is an open-source UCI engine that combines neural-network policy and value evaluation with Monte Carlo tree search. 0x88 does not compile the upstream C++ engine into WebAssembly. It exports selected Lc0 network weights (T1, T3, and BT4) to browser formats, evaluates them with ONNX Runtime WebGPU or custom WebGPU kernels, and searches them with 0x88's own PUCT implementation.</p>
    <details>
      <summary>Architecture &amp; packaging details</summary>
      <div class="details-body">
        <p>The served networks expose policy and value/WDL outputs used by the browser search. 0x88 exports the weights to ONNX and, for selected artifacts, applies QDQ int8 weight quantization while retaining floating-point computation after dequantization. The T1, T3, and BT4 options form a size ladder for different download, memory, and inference budgets.</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-lqo">Leela Queen Odds (LQO) <a class="anchor-link" href="#engines-lqo" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>notune &middot; <a href="https://github.com/notune/LeelaQueenOdds" rel="noopener">github.com/notune/LeelaQueenOdds</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Artifact</dt><dd><a href="https://assets.0x88.app/models/lc0/lqo_v2.f16.qdq8.onnx">/models/lc0/lqo_v2.f16.qdq8.onnx</a> (~96 MB, QDQ int8)</dd>
      <dt>Derived from</dt><dd><a href="https://assets.0x88.app/models/lc0/lqo_v2.f16.onnx">lqo_v2.f16.onnx</a> (fp16 source export)</dd>
      <dt>Manifest</dt><dd><a href="/models/lc0/manifest.json">/models/lc0/manifest.json</a></dd>
    </dl>
    <p>Leela Queen Odds v2 is an Lc0-compatible network trained specifically to play human opponents while the engine starts without its queen. The Play page therefore removes the bot's queen, not the user's. Upstream warns that the network is unsuitable for ordinary-position analysis because it evaluates its specialized queen-odds starting position as equal.</p>
    <details>
      <summary>Search parameters &amp; packaging details</summary>
      <div class="details-body">
        <p>Upstream recommends 15,000 nodes when the network plays White and 12,000 when it plays Black, with <code>ScLimit</code> 40/32, <code>CPuct 1.5</code>, <code>FpuValue 0.4</code>, color-specific <code>DrawScore</code>, and <code>SwapColors</code> when the network plays Black. 0x88 uses lower visit budgets and maps those settings into its own PUCT and search-contempt implementation.</p>
        <p>The served artifact is a QDQ int8 quantization of the fp16 export, derived in-tree and shipped from the LC0 manifest alongside the other big nets (BT4, t3).</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-maia3">Maia3 <a class="anchor-link" href="#engines-maia3" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Authors</dt><dd>Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, and Ashton Anderson (CSSLab) &middot; <a href="https://github.com/CSSLab/maia3" rel="noopener">github.com/CSSLab/maia3</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Paper</dt><dd><a href="https://arxiv.org/abs/2605.19091" rel="noopener">Chessformer: A Unified Architecture for Chess Modeling</a></dd>
      <dt>Provenance</dt><dd><a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">/docs/model_provenance/maia3.md</a></dd>
      <dt>Frontend</dt><dd><a href="https://github.com/CSSLab/maia-platform-frontend" rel="noopener">github.com/CSSLab/maia-platform-frontend</a> (byte-identical upstream fp16 model)</dd>
    </dl>
    <p>Maia-3 is a family of Chessformer models for predicting human moves across skill levels. The browser model returns rating-conditioned move probabilities and human-game WDL predictions rather than a searched centipawn evaluation. The Play page can sample from that policy or choose its highest-probability move.</p>
    <div class="callout info">
      <h4>AGPL source offer</h4>
      <p>The default browser artifact is a local int8 quantization of the upstream fp16 file. The AGPL source offer includes the <a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">derivation recipe</a> for that artifact.</p>
    </div>
  </div>

  <div class="engine-entry">
    <h3 id="engines-centipawn">Centipawn <a class="anchor-link" href="#engines-centipawn" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (in-house)</dd>
    </dl>
    <p>Centipawn is 0x88's own edge-deployed neural chess engine, and it is experimental. It evaluates the SquareFormer/BT4 network family in the browser with TVMJS WebGPU as the primary runtime and automatic ONNX Runtime WebGPU fallback, and appears on the Play, Analysis, and Arena pages at five strength levels.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-stockfish">Stockfish 19 <a class="anchor-link" href="#engines-stockfish" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Project</dt><dd><a href="https://github.com/official-stockfish/Stockfish" rel="noopener">official-stockfish/Stockfish</a> &middot; browser package via <a href="https://github.com/nmrugg/stockfish.js" rel="noopener">nmrugg/stockfish.js</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/stockfish/stockfish-19.0.1-corresponding-source.tar.gz">stockfish-19.0.1-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Stockfish is a community-developed UCI engine derived from Glaurung 2.1. It combines alpha-beta search with NNUE evaluation, and proposed changes are tested through the distributed Fishtest system. Stockfish 18 led the <a href="https://computerchess.org.uk/ccrl/4040/" rel="noopener">July 2026 CCRL 40/15 list</a>, and Stockfish finished first in the <a href="https://wiki.chessdom.org/Current_Engine_Status" rel="noopener">TCEC Season 29 Superfinal</a>. 0x88 ships the Stockfish.js 19.0.1 package in Lite and full-network forms; SF19 uses new networks for both variants.</p>
    <div class="callout info">
      <h4>Relaxed SIMD candidate</h4>
      <p>The Lite single-thread build has a feature-detected relaxed-SIMD candidate, with the baseline kept as a fallback. SF19 validation compares each WASM variant against its same-network native build using exact node-count bench signatures and fixed-depth best move, score, nodes, and PV. The Lite network has a different feature architecture from full Stockfish. Full single-threaded, Lite pthread, and full pthread relaxed builds remain local candidates, not default release artifacts. Historical SF18 relaxed-SIMD speedups do not describe the new SF19 networks.</p>
    </div>
    <details>
      <summary>Architecture &amp; packaging details</summary>
      <div class="details-body">
        <p>Stockfish uses an efficiently updatable neural network as its static evaluation function inside search. The Stockfish.js package compiles the engine to WebAssembly through Emscripten and embeds the required NNUE data. The manifest serves both the smaller Lite package and the full-network variants.</p>
      </div>
    </details>
  </div>

  <div class="engine-entry">
    <h3 id="engines-berserk">Berserk <a class="anchor-link" href="#engines-berserk" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Jay Honnold &middot; <a href="https://github.com/jhonnold/berserk" rel="noopener">github.com/jhonnold/berserk</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz">berserk-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Berserk 14 is Jay Honnold's UCI engine written in C. Its documented implementation uses principal-variation search with standard alpha-beta pruning techniques and a bucketed NNUE evaluation. 0x88 compiles the pinned release with Emscripten and serves scalar, WebAssembly SIMD, and relaxed-SIMD variants.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-viridithas">Viridithas <a class="anchor-link" href="#engines-viridithas" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Cosmo Bobak &middot; <a href="https://github.com/cosmobobak/viridithas" rel="noopener">github.com/cosmobobak/viridithas</a></dd>
      <dt>License</dt><dd><span class="lic mit">MIT</span> at pinned commit <a href="https://github.com/cosmobobak/viridithas/commit/20d7402065cae084715183e019fdd18089e2dfac" rel="noopener"><code>20d7402</code></a>; upstream relicensed to AGPL-3.0-only as of v21; v20.0.0 and earlier remain MIT</dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/viridithas/viridithas-wasip1-corresponding-source.tar.gz">viridithas-wasip1-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Viridithas is Cosmo Bobak's Rust UCI engine with NNUE evaluation. The project documents that its network training does not use evaluations produced by other chess engines. 0x88 pins commit <code>20d7402</code>, whose license is MIT, and serves scalar, WebAssembly SIMD, and relaxed-SIMD WASI builds.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-plentychess">PlentyChess <a class="anchor-link" href="#engines-plentychess" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Yoshie2000 &middot; <a href="https://github.com/Yoshie2000/PlentyChess" rel="noopener">github.com/Yoshie2000/PlentyChess</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0</span></dd>
      <dt>Manifest</dt><dd><a href="https://assets.0x88.app/channels/stable.json">/channels/stable.json</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz">plentychess-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>PlentyChess 8.0.0 is Patrick Leonhardt's C++17 UCI engine. Its threat-input NNUE is trained on more than 15 billion self-generated standard-chess and Fischer-random positions, with self-distillation used for part of the training. 0x88 compiles the pinned revision with Emscripten and serves SSSE3-shaped WebAssembly SIMD, SSE4.1, and relaxed-SIMD variants sharing one processed NNUE sidecar.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-stormphrax">Stormphrax <a class="anchor-link" href="#engines-stormphrax" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Author</dt><dd>Ciekce &middot; <a href="https://github.com/Ciekce/Stormphrax" rel="noopener">github.com/Ciekce/Stormphrax</a></dd>
      <dt>License</dt><dd><span class="lic gpl">GPL-3.0-or-later</span></dd>
      <dt>Network</dt><dd><code>undertown.nnue</code>, trained from Stormphrax self-play data</dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/stormphrax/stormphrax-emscripten-single-thread-corresponding-source.tar.gz">stormphrax-emscripten-single-thread-corresponding-source.tar.gz</a></dd>
    </dl>
    <p>Stormphrax 8.0.0 is Ciekce's C++ UCI engine with an explicit Chess960 focus. Its NNUE starts from random weights and is trained on data generated by Stormphrax rather than a third-party engine. 0x88 serves a single-thread Emscripten build with WebAssembly SIMD and a feature-detected relaxed-SIMD candidate.</p>
  </div>

  <div class="engine-entry">
    <h3 id="engines-reckless">Reckless <a class="anchor-link" href="#engines-reckless" aria-label="Link to this section">#</a></h3>
    <dl class="engine-meta">
      <dt>Project</dt><dd>CodeDeliveryService &middot; <a href="https://github.com/codedeliveryservice/Reckless" rel="noopener">github.com/codedeliveryservice/Reckless</a></dd>
      <dt>License</dt><dd><span class="lic agpl">AGPL-3.0</span></dd>
      <dt>Notice</dt><dd><a href="/reckless/NOTICE.md">/reckless/NOTICE.md</a></dd>
      <dt>Source</dt><dd><a href="https://assets.0x88.app/reckless/reckless-scalar-corresponding-source.tar.gz">scalar</a>, <a href="https://assets.0x88.app/reckless/reckless-simd128-corresponding-source.tar.gz">SIMD</a>, and <a href="https://assets.0x88.app/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz">relaxed SIMD</a></dd>
      <dt>Packaging</dt><dd><a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a> (build scripts &amp; release policy)</dd>
    </dl>
    <p>Reckless is a competitive Rust UCI engine using alpha-beta search and NNUE evaluation. Native Reckless 0.9.0 placed second on the <a href="https://computerchess.org.uk/ccrl/4040/" rel="noopener">July 2026 CCRL 40/15 list</a>, and a development build finished second in the <a href="https://wiki.chessdom.org/Current_Engine_Status" rel="noopener">TCEC Season 29 Superfinal</a>. 0x88 pins development commit <a href="https://github.com/codedeliveryservice/Reckless/commit/0010617448bdef4c8cd7d4f4825b7e42c8bc262a" rel="noopener"><code>0010617</code></a> and serves scalar, WebAssembly SIMD, and relaxed-SIMD WASI builds.</p>
    <div class="callout info">
      <h4>Public v0 engine</h4>
      <p>Reckless is published through the project CDN for Play, Analysis, and Arena. The browser selects relaxed SIMD or SIMD when supported, and falls back to the scalar WASI build otherwise. Matching source archives for the scalar, SIMD, and relaxed SIMD builds are hosted beside the binaries.</p>
    </div>
  </div>
  {@render sectionNav('engines')}
</section>

<!-- ===== UNDER THE HOOD ===== -->
<section id="under-the-hood">
  <h2>Under the hood <a class="anchor-link" href="#under-the-hood" aria-label="Link to this section">#</a></h2>
<!-- ===== NEURAL BROWSER RUNTIMES ===== -->
<div class="runtime-block">
  <h3 id="neural-runtimes">Neural browser runtimes <a class="anchor-link" href="#neural-runtimes" aria-label="Link to this section">#</a></h3>
  <p class="lead">Lc0 and Maia3 are loaded as browser models rather than compiled UCI executables. 0x88 supplies input encoding, inference, and legal-move mapping. Its own PUCT implementation searches the Lc0 and Leela Queen Odds outputs, while Maia3 is used directly as a rating-conditioned move policy.</p>

  <div class="callout info">
    <h4>Stable baseline</h4>
    <p><strong>ONNX Runtime WebGPU</strong> is the stable neural baseline and rollback path. TVMJS, custom WGSL, WebNN, and QDQ artifacts all need to prove themselves against ORT WebGPU with fixture drift and search-move gates before they can become product defaults. The TVMJS lane has cleared those gates for Centipawn, where it is the promoted primary runtime with automatic ORT fallback; everywhere else the rule still applies.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Lane</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">What it is</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Status</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">ONNX WebGPU</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">ONNX Runtime Web running LC0/Maia models on the browser GPU, with ORT WASM as fallback/control.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Stable default baseline for neural models where WebGPU is available.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">ONNX QDQ</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Weight-only int8 quantize/dequantize ONNX graphs: smaller files, float compute after in-graph dequantization.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Per-model promotion after real chess fixture/search drift gates; mainly a download/cache win.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Custom WGSL</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Hand-written WebGPU kernels for hot LC0 subgraphs: encoder blocks, heads, legal-prior/readback experiments.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Explicit hybrid/runtime lane; parity and readback/batch-fill evidence decide promotion.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">TVMJS WebGPU</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Compiler path from ONNX to TVM Relax to browser-loadable TVMJS wasm plus WebGPU pipelines.</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Promoted primary runtime for Centipawn with automatic ORT fallback; other models remain research/opt-in pending separate hosting, provenance, and cross-device gates.</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">ORT WebNN EP</td>
        <td style="padding:10px">ONNX Runtime through WebNN, potentially reaching CoreML/ANE/NPU hardware instead of WebGPU.</td>
        <td style="padding:10px">Promising but flag-gated; not shippable until WebNN is unflagged and model-specific correctness gates pass.</td>
      </tr>
    </tbody>
  </table>

  <p>WebNN is worth tracking because early probes show real acceleration on supported hardware, but it currently requires Chrome's <code>WebMachineLearningNeuralNetwork</code> feature flag and has model-shape/precision caveats. Likewise, QDQ should be judged on real chess positions: random tensor comparisons can overstate int8 drift compared with actual LC0/Maia activations.</p>
  <p>Full notes live in <a href="https://github.com/manthedan/0x88/blob/main/docs/neural_browser_runtimes.md">docs/neural_browser_runtimes.md</a>, with supporting runbooks for <a href="https://github.com/manthedan/0x88/blob/main/docs/lc0_tvmjs_research_runbook.md">TVMJS</a>, <a href="https://github.com/manthedan/0x88/blob/main/docs/lc0web_custom_inference_checkpoint.md">custom WGSL</a>, and <a href="https://github.com/manthedan/0x88/blob/main/docs/lc0_t3_qdq_webnn_2026-06-10.md">QDQ/WebNN probes</a>.</p>
</div>

<!-- ===== CPU WASM RUNTIMES ===== -->
<div class="runtime-block">
  <h3 id="cpu-wasm">CPU WASM runtimes <a class="anchor-link" href="#cpu-wasm" aria-label="Link to this section">#</a></h3>
  <p class="lead">Stockfish, Reckless, Berserk, Viridithas, PlentyChess, Stormphrax, and the Monty lab port are CPU engines compiled to WebAssembly. Browser workers provide their UCI-style control layer.</p>

  <div class="callout info">
    <h4>Two build lanes</h4>
    <p><strong>Emscripten</strong> is the preferred intake path for C/C++ engines: it gives us Stockfish.js-style JS glue, a worker-loadable <code>.wasm</code>, and optional <code>.data</code> sidecars for NNUE files. <strong>Rust WASI</strong> is the preferred path for Rust engines: compile to <code>wasm32-wasip1</code>, then run through the browser WASI shim in one-shot, batch, or persistent-worker mode.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Runtime path</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Engines</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Build targets</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stockfish.js package</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Stockfish 19</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Lite/full, single-thread/threaded flavors; threaded builds require cross-origin isolation.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Emscripten UCI worker</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Berserk, PlentyChess, Stormphrax</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Single-thread baseline first, then explicit <code>simd128</code>, SSE4.1-shaped wasm SIMD, or relaxed-SIMD variants after smoke/benchmark evidence.</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Rust <code>wasm32-wasip1</code></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Reckless, Viridithas, Monty lab</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)">Scalar fallback, <code>+simd128</code> preferred variants, and relaxed-SIMD as experimental until proven per engine.</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Direct browser API</td>
        <td style="padding:10px">Reckless experiments</td>
        <td style="padding:10px">Bypasses UCI text for structured calls, but only graduates if it beats the simpler UCI/WASI path on lifecycle or latency.</td>
      </tr>
    </tbody>
  </table>

  <div class="runtime-subsection">
  <h4 id="cpu-wasm-relaxed-simd">Relaxed SIMD <a class="anchor-link" href="#cpu-wasm-relaxed-simd" aria-label="Link to this section">#</a></h4>
  <p>Relaxed SIMD is a separate WebAssembly feature from ordinary <code>simd128</code>. A relaxed build can fail to validate on a browser that supports baseline SIMD, so every relaxed artifact needs feature detection, asset fallback, and a non-relaxed path.</p>
  <p>The main win for these chess engines is the relaxed integer dot product used in NNUE layers. When the activation operand is proven to stay in <code>[0, 127]</code>, <code>i32x4_relaxed_dot_i8x16_i7x16_add</code> is value-exact and can replace slower <code>maddubs</code>/<code>dpbusd</code> emulation. Reckless, Viridithas, Berserk, PlentyChess, and Stormphrax all have this proof in their current SIMD audit lanes.</p>
  <p>The rule is still parity-first: inspect the artifact to confirm relaxed opcodes are present, then require fixed-depth equality for best move, score, node count, and PV before promoting a runtime ladder such as <code>relaxed-simd &gt; simd128 &gt; scalar</code>. Without that evidence, relaxed SIMD remains a lab or benchmark variant.</p>

  <div class="callout info">
    <h4>Benchmark context</h4>
    <p>The current relaxed-SIMD snapshot is measured in a Chromium browser on macOS with an Apple M4 chip. Treat the NPS deltas as device/runtime-specific engineering evidence, not universal engine-speed claims; the exact fixed-depth parity result is the stronger promotion signal.</p>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Engine</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Apple M4 / Chromium evidence</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Parity gate</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Reckless</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>+24%</code> NPS vs old kernels</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>60/60</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Viridithas</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>+14%</code> NPS over standard SIMD</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>40/40</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Berserk</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>1.50M</code> NPS vs <code>1.38M</code> SIMD (<code>+8%</code>)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>40/40</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">PlentyChess</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>992k</code> NPS vs <code>603k</code> default (<code>+64%</code>)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>40/40</code> exact fixed-depth parity</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stormphrax</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>540k</code> NPS vs <code>477k</code> SIMD (<code>1.13&times;</code>, Chromium depth 11)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><code>60/60</code> exact fixed-depth parity; <code>20/20</code> browser parity</td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Stockfish</td>
        <td style="padding:10px">SF19 Lite: <code>1.086M</code> median wall NPS vs <code>0.863M</code> baseline (<code>1.27&times;</code> paired median, Chromium depth 12; shared-host spread is wide).</td>
        <td style="padding:10px"><code>600/600</code> exact browser/native comparisons; full and pthread relaxed builds remain unpromoted.</td>
      </tr>
    </tbody>
  </table>

  <p>The promotion rule is deliberately conservative: build scripts must pin upstream source and NNUE assets, Node and browser smokes must pass, benchmarks must justify the variant, and GPL/AGPL artifacts need manifests plus matching source archives before public distribution. Full notes live in <a href="https://github.com/manthedan/0x88/blob/main/docs/cpu_wasm_runtimes.md">docs/cpu_wasm_runtimes.md</a> and the C/C++ porting recipe in <a href="https://github.com/manthedan/0x88/blob/main/docs/browser_c_engine_porting.md">docs/browser_c_engine_porting.md</a>.</p>
  </div>
</div>

<!-- ===== CDN ===== -->
<div class="runtime-block">
  <h3 id="cdn">Artifact CDN &amp; caching <a class="anchor-link" href="#cdn" aria-label="Link to this section">#</a></h3>
  <p class="lead">Engine binaries and neural-network models are served from a Cloudflare R2 bucket fronted by a Worker that resolves friendly URLs to content-addressed blobs. Artifacts range from 130 KB to 60+ MB, so caching and compression are critical for the site to function.</p>

  <div class="callout info">
    <h4>Two-plane model</h4>
    <p><strong>Control plane</strong> &mdash; channel and release manifests (mutable, short TTL). <strong>Data plane</strong> &mdash; content-addressed blobs under <code>/artifacts/sha256/</code> (immutable, 1-year edge cache). The Worker maps friendly URLs like <code>/viridithas/viridithas-relaxed-simd128.wasm</code> to the correct blob by reading the channel manifest.</p>
  </div>

  <p>The Worker also handles range requests (delegated to R2's native reader), sets <code>no-transform</code> on binary types to prevent CDN auto-compression of WASM, and attaches CORS/CORP headers for cross-origin isolation. Pre-compression (brotli + gzip sidecars) is done at publish time, not at the edge. Browser-side, the Cache Storage API validates responses by byte length and SHA-256, and compiled <code>WebAssembly.Module</code> objects are cached per worker session.</p>

  <p>Full architecture, compression pipeline details, artifact size table, and operational playbook (diagnostics, publishing, adding new engines) are in <a href="https://github.com/manthedan/0x88/blob/main/docs/cdn_artifact_caching.md">docs/cdn_artifact_caching.md</a>.</p>
</div>
  {@render sectionNav('under-the-hood')}
</section>

<!-- ===== LICENSES ===== -->
<section id="licenses">
  <h2>Licenses &amp; corresponding source <a class="anchor-link" href="#licenses" aria-label="Link to this section">#</a></h2>
  <p class="lead">Most engines on this site use GPL or AGPL licenses. This section identifies each redistributed artifact, its upstream project, and its corresponding-source archive.</p>

  <div class="callout info">
    <h4>The short version</h4>
    <p>Every GPL- or AGPL-licensed engine we redistribute ships with a matching <strong>corresponding-source archive</strong> from the same deployment as the binary. The archive contains the exact upstream source at the pinned commit, our local patches, the build script that produced the browser artifact, and the toolchain used. This is what the GPL means by "the scripts used to control compilation and installation of the executable."</p>
    <p>Upstream GitHub links are <em>not</em> a sufficient corresponding-source offer on their own &mdash; the FSF position and our own policy agree on this &mdash; so each engine has a real <code>*-corresponding-source.tar.gz</code> sitting next to its <code>.wasm</code>. The Maia3 model is under AGPL-3.0, and the default browser artifact is a local int8 quantization of the upstream fp16 file, so the <a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">derivation recipe</a> is part of the source offer.</p>
    <p><strong>Full policy:</strong> <a href="https://github.com/manthedan/0x88/blob/main/docs/engine_artifact_distribution.md">docs/engine_artifact_distribution.md</a> &middot; <strong>Hosted artifact index:</strong> <a href="https://github.com/manthedan/0x88/blob/main/docs/hosted_artifacts.md">docs/hosted_artifacts.md</a> &middot; <strong>This project's source:</strong> <a href="https://github.com/manthedan/0x88" rel="noopener">github.com/manthedan/0x88</a></p>
  </div>

  <div class="source-links">
  <h3 id="licenses-per-engine" style="font-size:20px">Per-engine source links <a class="anchor-link" href="#licenses-per-engine" aria-label="Link to this section">#</a></h3>
  <p>Each row lists the binary we ship, its license, and the matching source offer. This is the same information as the metadata blocks above, collected in one place for quick reference.</p>
  <table style="width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 24px">
    <thead>
      <tr>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Engine</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">License</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Upstream</th>
        <th style="text-align:left; padding:8px 10px; border-bottom:2px solid var(--rule); font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted)">Corresponding source</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Leela Chess Zero</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0-or-later</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/LeelaChessZero/lc0" rel="noopener">LeelaChessZero/lc0</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/manthedan/0x88" rel="noopener">0x88 repo</a> (browser runtime + export scripts)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Leela Queen Odds (LQO)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/notune/LeelaQueenOdds" rel="noopener">notune/LeelaQueenOdds</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/models/lc0/lqo_v2.f16.onnx">fp16 source export</a> + <a href="https://github.com/manthedan/0x88" rel="noopener">0x88 repo</a> (QDQ int8 derivation scripts)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Maia3</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/CSSLab/maia3" rel="noopener">CSSLab/maia3</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/manthedan/0x88/blob/main/docs/model_provenance/maia3.md">Provenance doc</a> + <a href="https://github.com/CSSLab/maia-platform-frontend" rel="noopener">upstream frontend</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stockfish 19</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/nmrugg/stockfish.js" rel="noopener">nmrugg/stockfish.js</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/stockfish/stockfish-19.0.1-corresponding-source.tar.gz">stockfish-19.0.1-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Berserk</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/jhonnold/berserk" rel="noopener">jhonnold/berserk</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/berserk/berserk-emscripten-single-thread-corresponding-source.tar.gz">berserk-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Viridithas</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">MIT (pinned commit)</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/cosmobobak/viridithas/commit/20d7402065cae084715183e019fdd18089e2dfac" rel="noopener">cosmobobak/viridithas@20d7402</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/viridithas/viridithas-wasip1-corresponding-source.tar.gz">viridithas-wasip1-corresponding-source.tar.gz</a> (license still honored)</td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">PlentyChess</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/Yoshie2000/PlentyChess" rel="noopener">Yoshie2000/PlentyChess</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/plentychess/plentychess-emscripten-single-thread-corresponding-source.tar.gz">plentychess-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-weight:600">Stormphrax</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule); font-family:var(--mono); font-size:12px">GPL-3.0-or-later</td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://github.com/Ciekce/Stormphrax/tree/v8.0.0" rel="noopener">Ciekce/Stormphrax v8.0.0</a></td>
        <td style="padding:10px; border-bottom:1px solid var(--rule)"><a href="https://assets.0x88.app/stormphrax/stormphrax-emscripten-single-thread-corresponding-source.tar.gz">stormphrax-emscripten-single-thread-corresponding-source.tar.gz</a></td>
      </tr>
      <tr>
        <td style="padding:10px; font-weight:600">Reckless</td>
        <td style="padding:10px; font-family:var(--mono); font-size:12px">AGPL-3.0</td>
        <td style="padding:10px"><a href="https://github.com/codedeliveryservice/Reckless" rel="noopener">codedeliveryservice/Reckless</a></td>
        <td style="padding:10px"><a href="/reckless/NOTICE.md">NOTICE.md</a> + <a href="https://assets.0x88.app/reckless/reckless-scalar-corresponding-source.tar.gz">scalar</a>, <a href="https://assets.0x88.app/reckless/reckless-simd128-corresponding-source.tar.gz">SIMD</a>, and <a href="https://assets.0x88.app/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz">relaxed SIMD</a> source archives</td>
      </tr>
    </tbody>
  </table>

  <blockquote>
    <p>A note on engine names and branding: the names "Stockfish", "Leela Chess Zero", "Berserk", "Viridithas", "PlentyChess", "Stormphrax", "Reckless", and "Maia" are project trademarks of their respective owners, separate from the code licenses. We use them here solely to identify the engines (nominative use) and do not claim endorsement. Consult each project's own branding guidance if in doubt.</p>
  </blockquote>
  </div>

<!-- ===== REMOVAL ===== -->
<div class="removal-block">
  <h3 id="removal">Removal requests <a class="anchor-link" href="#removal" aria-label="Link to this section">#</a></h3>
  <p class="lead">The engines and models above belong to their respective projects. 0x88 packages them for browser use.</p>
  <p>Maintainers and rights holders can request removal by <a href="https://github.com/manthedan/0x88/issues/new?title=Engine%20removal%20request&amp;body=Which%20engine%20or%20model%3A%20%0A%0AAre%20you%20a%20maintainer%20or%20rights%20holder%3A%20%0A%0AAnything%20else%3A%20" rel="noopener">opening an issue on GitHub</a>. The requested engine or model will be removed.</p>
  <p>
    <a class="removal-cta" href="https://github.com/manthedan/0x88/issues/new?title=Engine%20removal%20request&amp;body=Which%20engine%20or%20model%3A%20%0A%0AAre%20you%20a%20maintainer%20or%20rights%20holder%3A%20%0A%0AAnything%20else%3A%20" rel="noopener">Request removal on GitHub <span aria-hidden="true">&rarr;</span></a>
  </p>
</div>
  {@render sectionNav('licenses')}
</section>

</div><!-- /doc-content -->

</div><!-- /wrap -->
</div><!-- /doc-body -->

</main>
<footer class="site-footer">
  <div class="wrap">
    <span>Engine binaries and source archives are linked above. Everything runs client-side.</span>
    <span class="footer-links">
      <a href="/">Home</a>
      <a href="https://github.com/manthedan/0x88" rel="noopener">Source</a>
      <code>0x88.app · v0</code>
    </span>
  </div>
</footer>
<style>
  .visually-hidden{position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0}

  /* ---- layout: contents rail | intro + content ---- */
  .doc-body{font-size:16px}
  .doc-body .doc-grid{
    padding:36px 28px 64px;
    display:grid; grid-template-columns:248px minmax(0,70ch);
    grid-template-areas:"toc intro" "toc content";
    column-gap:56px; align-items:start;
  }
  .doc-intro{grid-area:intro; margin-bottom:32px}
  .doc-intro h1{font-size:clamp(30px,4vw,40px); margin:0 0 10px; line-height:1.1; letter-spacing:-.015em}
  .intro-lede{margin:0; color:var(--ink-soft); font-size:16.5px; max-width:66ch; line-height:1.6}
  .doc-content{grid-area:content; min-width:0}

  /* Wide tables scroll inside their own box instead of widening the page. */
  .doc-content :global(table){display:block; overflow-x:auto}
  .doc-content{overflow-wrap:break-word}

  /* ---- contents rail ---- */
  .toc{
    grid-area:toc;
    position:sticky; top:calc(var(--hdr, 64px) + 20px);
    font-size:14px;
  }
  .toc-toolbar{display:none}
  .toc-panel{display:flex; flex-direction:column; max-height:calc(100vh - var(--hdr, 64px) - 40px)}
  .search{position:relative; margin:0 0 14px; flex:0 0 auto}
  .search-icon{position:absolute; left:10px; top:50%; width:15px; height:15px; transform:translateY(-50%); fill:none; stroke:var(--muted); stroke-width:2; stroke-linecap:round; pointer-events:none}
  .search input{
    width:100%; box-sizing:border-box; min-height:40px; padding:8px 86px 8px 32px;
    border:1px solid var(--border-input); border-radius:var(--radius-sm);
    background:var(--input-bg); color:var(--ink); font:inherit; font-size:14px;
  }
  .search input::placeholder{color:var(--muted)}
  .search input:hover{border-color:var(--border-input-hover)}
  .search input:focus{outline:none; border-color:var(--accent); box-shadow:var(--focus-ring)}
  .search input::-webkit-search-cancel-button{display:none}
  .search-shortcuts{position:absolute; right:8px; top:50%; transform:translateY(-50%); display:flex; gap:4px; pointer-events:none}
  .search kbd{
    min-width:18px; padding:1px 4px; border:1px solid var(--rule-strong); border-radius:4px;
    background:var(--panel); color:var(--text-soft); font-family:var(--mono); font-size:10px; text-align:center;
  }
  .search input:focus ~ .search-shortcuts{display:none}
  .toc-groups,.results{overflow-y:auto; min-height:0; padding-right:6px; margin-right:-6px}
  .toc-group + .toc-group{margin-top:12px; padding-top:12px; border-top:1px solid var(--rule)}
  .toc-group-heading{display:flex; align-items:center; gap:4px; margin-bottom:4px}
  .toc-groups .toc-group-label{flex:1; font-size:14px; font-weight:700; color:var(--ink)}
  .toc-group-heading.current-group .toc-group-label{color:var(--accent-soft-text); background:var(--accent-soft)}
  .group-toggle{
    display:flex; align-items:center; justify-content:center; gap:4px; min-width:42px; min-height:30px; padding:4px;
    border:1px solid var(--rule-strong); border-radius:6px; background:var(--panel); color:var(--text); font:12px var(--sans); cursor:pointer;
  }
  .group-toggle:hover{background:var(--panel-2); color:var(--ink)}
  .group-toggle svg{width:14px; height:14px; fill:none; stroke:currentColor; stroke-width:1.8; transition:transform .15s}
  .group-toggle[aria-expanded=true] svg{transform:rotate(180deg)}
  .toc ul[hidden]{display:none}
  .toc ul{list-style:none; padding:0; margin:0; display:grid; gap:1px}
  .toc-groups a{
    display:block; padding:5px 10px; border-radius:6px;
    color:var(--text); text-decoration:none; line-height:1.35;
  }
  .toc-groups a:hover{color:var(--ink); background:var(--rule-soft)}
  .toc-groups a.in-active{color:var(--ink); font-weight:600}
  .toc-groups a.active{color:var(--accent-soft-text); font-weight:700; background:var(--accent-soft)}
  .toc-group > ul{padding-left:10px}
  .toc ul ul{margin:1px 0 4px 12px}
  .toc ul ul a{font-size:13px; padding:4px 10px}

  /* ---- search results ---- */
  .results-count,.no-results{margin:0 0 8px; padding:0 4px; color:var(--text-soft); font-size:12.5px}
  .no-results{color:var(--text); line-height:1.5}
  .results ul{gap:2px}
  .results a{
    display:grid; gap:2px; padding:8px 10px; border-radius:6px;
    text-decoration:none; color:var(--text);
  }
  .results a:hover,.results a:focus-visible{background:var(--rule-soft); outline:none}
  .results a:focus-visible{box-shadow:inset 0 0 0 2px var(--accent)}
  .hit-path{font-size:11.5px; color:var(--text-soft)}
  .hit-title{color:var(--ink); font-weight:650; font-size:14px}
  .hit-snippet{font-size:12.5px; line-height:1.45; color:var(--text)}
  .hit-snippet mark{background:var(--accent-soft); color:var(--accent-soft-text); font-weight:650; border-radius:2px; padding:0 1px}

  /* ---- content ---- */
  .doc-content section{padding:0 0 52px}
  .doc-content section:last-child{padding-bottom:0}
  .doc-content :global([id]){scroll-margin-top:calc(var(--hdr, 64px) + 20px)}
  .doc-content h2{
    font-size:32px; margin:8px 0 14px;
    line-height:1.2; letter-spacing:-.01em;
    display:flex; align-items:center; gap:10px;
  }
  .doc-content h2::before{
    content:""; display:block; width:28px; height:3px; flex:0 0 28px;
    background:var(--accent); border-radius:2px;
  }
  .doc-content h2 .anchor-link,
  .doc-content h3 .anchor-link{
    margin-left:8px; opacity:0;
    color:var(--muted); font-size:.7em;
    text-decoration:none; font-family:var(--mono);
    transition:opacity .12s;
  }
  .doc-content h2:hover .anchor-link,
  .doc-content h3:hover .anchor-link,
  .doc-content .anchor-link:focus-visible{opacity:1}
  .doc-content h2 .anchor-link:hover,
  .doc-content h3 .anchor-link:hover{color:var(--accent); opacity:1}
  .doc-content h3{font-family:var(--sans); font-weight:650; font-size:21px; margin:36px 0 12px; line-height:1.3}
  .doc-content h4{font-family:var(--sans); font-size:17px; margin:28px 0 12px}
  .doc-content h3 .anchor-link,.doc-content h4 .anchor-link{font-size:.6em; vertical-align:middle}
  .doc-content h4 .anchor-link{margin-left:8px; color:var(--muted); text-decoration:none}
  .runtime-block + .runtime-block{padding-top:16px; margin-top:32px; border-top:1px solid var(--rule)}
  .doc-content :global(h2:focus),.doc-content :global(h3:focus){outline:none}
  .doc-content > section > p.lead{
    color:var(--ink-soft); margin:0 0 26px;
    font-size:16px; max-width:66ch; line-height:1.65;
  }
  .doc-content p{margin:0 0 14px; line-height:1.65; color:var(--ink-soft)}
  .doc-content ul,.doc-content ol{
    margin:0 0 16px; padding-left:24px;
    line-height:1.65; color:var(--ink-soft);
  }
  .doc-content ul li::marker{color:var(--accent)}
  .doc-content code{
    font-family:var(--mono); font-size:.88em;
    background:var(--rule-soft); padding:1px 6px;
    border-radius:4px; color:var(--ink);
  }
  .doc-content a{font-weight:500}

  /* ---- workspace cards ---- */
  .page-block{
    background:var(--panel); border:1px solid color-mix(in srgb, var(--ink) 16%, var(--panel));
    border-radius:var(--radius); padding:24px; margin-bottom:24px;
  }
  .task-heading{display:flex; align-items:center; flex-wrap:wrap; gap:12px; justify-content:space-between; margin-bottom:14px}
  .page-block h3{margin:0; font-size:20px; flex:1 1 220px}
  .task-reference{margin-top:24px; padding-top:18px; border-top:1px solid var(--rule-strong)}
  .task-reference h4{font-size:15px; margin:0 0 10px; color:var(--ink)}
  .task-reference ul{font-size:16px}
  .task-reference li + li{margin-top:6px}
  .page-block p{margin:0 0 12px}
  .page-block ul{margin:0 0 14px}
  .doc-content .steps{counter-reset:step; list-style:none; padding-left:0; margin:4px 0 18px; display:grid; gap:8px}
  .steps li{counter-increment:step; position:relative; padding-left:36px; line-height:1.6}
  .steps li::before{
    content:counter(step); position:absolute; left:0; top:1px;
    width:24px; height:24px; border-radius:50%;
    display:inline-flex; align-items:center; justify-content:center;
    background:var(--panel-2); border:1px solid var(--rule-strong);
    color:var(--ink); font:700 12px/1 var(--sans);
  }
  .doc-content .not-yet{
    padding:10px 14px; border:1px dashed var(--rule-strong); border-radius:var(--radius-sm);
    font-size:14px; color:var(--text); max-width:none;
  }
  .doc-content .open-link{
    display:inline-flex; align-items:center; gap:8px; min-height:40px;
    padding:8px 16px; border-radius:var(--radius-sm);
    background:var(--accent); color:var(--on-accent); font-weight:650; font-size:14px;
    text-decoration:none;
  }
  .doc-content .open-link:hover{background:var(--accent-hover); text-decoration:none}
  .open-link span{transition:transform .15s ease}
  .open-link:hover span{transform:translateX(3px)}
  .section-nav{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:32px; padding-top:24px; border-top:1px solid var(--rule-strong)}
  .section-nav a{display:grid; gap:4px; padding:14px 16px; border:1px solid var(--rule-strong); border-radius:var(--radius); color:var(--ink); font-size:15px; background:var(--panel); text-decoration:none}
  .section-nav a:hover{border-color:var(--accent); background:var(--panel-2)}
  .section-nav .next-section{grid-column:2; text-align:right}
  .section-direction{font-size:12px; font-weight:500; color:var(--text-soft)}

  /* ---- troubleshooting ---- */
  .trouble{padding:4px 0 8px}
  .trouble + .trouble{border-top:1px solid var(--rule); margin-top:12px}
  .trouble h3{margin-top:22px}
  .device-check{
    margin:14px 0 8px; padding:14px 18px; max-width:640px;
    border:1px solid var(--rule); border-radius:var(--radius); background:var(--panel);
  }
  .device-check dl{margin:0; display:grid; gap:10px}
  .device-check dl > div{display:grid; grid-template-columns:110px minmax(0,1fr); gap:12px; align-items:baseline}
  .device-check dt{font-weight:650; color:var(--ink); font-size:14px}
  .device-check dd{margin:0; color:var(--text); font-size:14px; line-height:1.5}
  .dc-state{
    display:inline-block; margin-right:8px; padding:1px 8px; border-radius:99px;
    font-family:var(--mono); font-size:11.5px; font-weight:700; border:1px solid var(--rule-strong);
  }
  .dc-state[data-tone=ok]{color:var(--accent-soft-text); background:var(--accent-soft); border-color:var(--accent-tint-border)}
  .dc-state[data-tone=warn]{color:var(--warn); background:var(--warn-soft); border-color:color-mix(in srgb, var(--warn) 35%, var(--rule))}
  .doc-content .dc-pending{margin:0; color:var(--text); font-size:14px}

  /* ---- tables ---- */
  .doc-table{width:100%; border-collapse:collapse; font-size:13.5px; margin:8px 0 18px}
  .doc-table th,.doc-table td{padding:9px 10px; border-bottom:1px solid var(--rule); text-align:left; vertical-align:top; color:var(--ink-soft)}
  .doc-table thead th{
    border-bottom:2px solid var(--rule-strong); padding-top:6px; padding-bottom:6px;
    font-family:var(--mono); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--text-soft);
  }
  .doc-table tbody tr:last-child > *{border-bottom:none}
  .doc-table tbody th{font-weight:650; white-space:nowrap}
  .doc-table.compact{width:auto; min-width:320px}
  .engine-table td:nth-child(2){white-space:nowrap; font-weight:600; color:var(--ink)}
  .engine-table .sub{display:block; font-weight:400; font-size:12px; color:var(--text-soft)}
  .table-note{font-size:14px}

  /* ---- engine entries ---- */
  .engine-entry{padding:8px 0 32px; border-bottom:1px solid var(--rule-soft)}
  .engine-entry:last-of-type{border-bottom:none}
  .engine-meta{
    display:grid; grid-template-columns:auto 1fr;
    gap:4px 16px; margin:10px 0 16px;
    font-size:13.5px; max-width:100%;
  }
  .engine-meta dt{
    font-family:var(--mono); font-size:11px;
    text-transform:uppercase; letter-spacing:.05em;
    color:var(--text-soft); font-weight:700;
    white-space:nowrap; padding:4px 0;
  }
  .engine-meta dd{margin:0; padding:4px 0; color:var(--ink-soft)}
  .engine-meta dd a{font-weight:500}
  .lic{
    font-family:var(--mono); font-size:11px; font-weight:700;
    padding:2px 9px; border-radius:99px;
    white-space:nowrap; display:inline-block;
  }
  .lic.gpl{background:color-mix(in srgb, var(--gold) 16%, transparent); color:color-mix(in srgb, var(--gold) 55%, var(--ink))}
  .lic.agpl{background:var(--warn-soft); color:var(--warn)}
  .lic.mit{background:var(--accent-soft); color:var(--accent-soft-text)}
  .lic.own{background:var(--rule-soft); color:var(--ink-soft)}
  .engine-entry details{
    margin-top:8px; border:1px solid var(--rule);
    border-radius:var(--radius-sm); background:var(--panel); overflow:hidden;
  }
  .engine-entry details summary{
    cursor:pointer; padding:10px 16px;
    font-family:var(--mono); font-size:12px;
    color:var(--text); font-weight:600;
    letter-spacing:.03em; text-transform:uppercase;
    list-style:none; user-select:none;
  }
  .engine-entry details summary::-webkit-details-marker{display:none}
  .engine-entry details summary::before{content:"\25B8  "; color:var(--accent); display:inline-block}
  .engine-entry details[open] summary::before{content:"\25BE  "}
  .engine-entry details summary:hover{color:var(--ink); background:var(--rule-soft)}
  .engine-entry details .details-body{padding:0 16px 14px}
  .engine-entry details .details-body p:last-child{margin-bottom:0}

  /* ---- callouts ---- */
  .callout{
    border-radius:var(--radius); padding:20px 24px;
    margin:20px 0; border:1px solid var(--rule);
  }
  .callout.info{
    background:var(--accent-soft);
    border-color:var(--accent-tint-border);
  }
  .callout h4{
    margin:0 0 10px; font-size:14px;
    font-family:var(--sans); letter-spacing:.02em;
    display:flex; align-items:center; gap:8px;
  }
  .callout.info h4{color:var(--accent-soft-text)}
  .callout h4::before{
    font-family:var(--mono); font-size:14px;
    width:20px; height:20px; border-radius:4px;
    display:inline-flex; align-items:center; justify-content:center;
  }
  .callout.info h4::before{content:"i"; background:var(--accent); color:var(--on-accent); font-style:italic}
  .callout p{margin:0 0 10px; line-height:1.65; font-size:14px; max-width:64ch; color:var(--ink-soft)}
  .callout p:last-child{margin:0}
  .callout code{
    font-family:var(--mono); font-size:12px;
    background:var(--panel); padding:1px 6px;
    border-radius:4px; border:1px solid var(--rule);
  }
  .callout a{font-weight:500}
  .removal-cta{
    display:inline-flex; align-items:center; gap:8px;
    margin-top:4px; padding:11px 18px; border-radius:10px;
    background:var(--accent); color:var(--on-accent); font-weight:600; font-size:14px;
    border:1px solid var(--accent); text-decoration:none;
    transition:background .15s ease;
  }
  .removal-cta:hover{background:var(--accent-hover); text-decoration:none}
  .removal-cta span{transition:transform .15s ease}
  .removal-cta:hover span{transform:translateX(3px)}
  .doc-content blockquote{
    margin:24px 0; padding:16px 22px;
    border:1px solid var(--rule); border-radius:var(--radius);
    color:var(--text); font-size:14px; line-height:1.65; max-width:68ch;
  }
  .doc-content blockquote p{margin:0; color:var(--text)}

  /* ---- phones and narrow tablets: sticky contents bar + sheet ---- */
  @media(max-width:860px){
    .doc-body .doc-grid{
      grid-template-columns:minmax(0,1fr);
      grid-template-areas:"intro" "toc" "content";
      padding:20px 16px 56px;
    }
    .doc-intro{margin-bottom:16px}
    .doc-intro h1{font-size:30px; margin-bottom:8px}
    .intro-lede{font-size:15.5px; line-height:1.55}
    .toc{
      top:var(--hdr, 64px); z-index:30;
      margin:0 -16px 24px; padding:8px 16px;
      background:color-mix(in srgb, var(--bg) 92%, transparent);
      -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
      border-bottom:1px solid var(--rule-strong);
    }
    .toc-toolbar{display:flex; gap:8px}
    .toc-toggle{
      min-width:0; flex:1; min-height:44px; padding:8px 10px;
      display:flex; align-items:center; gap:10px;
      border:1px solid var(--rule-strong); border-radius:var(--radius);
      background:var(--panel); color:var(--ink); font:inherit; text-align:left; cursor:pointer;
    }
    .mobile-search{display:flex; align-items:center; justify-content:center; gap:6px; padding:8px 10px; border:1px solid var(--rule-strong); border-radius:var(--radius); background:var(--panel); color:var(--ink); font:600 13px var(--sans); cursor:pointer}
    .mobile-search svg{width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round}
    .toc-toggle-label{flex:0 0 auto; font-size:12px; font-weight:700; color:var(--text-soft)}
    .toc-current{flex:1 1 auto; min-width:0; font-size:14.5px; font-weight:650; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
    .chevron{width:16px; height:16px; flex:0 0 16px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; transition:transform .15s}
    .toc.open .chevron{transform:rotate(180deg)}
    .toc-panel{
      display:none; position:absolute; left:16px; right:16px; top:calc(100% - 2px);
      max-height:calc(100vh - var(--hdr, 64px) - 80px);
      padding:12px; border:1px solid var(--rule-strong); border-radius:var(--radius);
      background:var(--panel); box-shadow:var(--shadow-lg);
    }
    .toc.open .toc-panel{display:flex}
    .search-shortcuts{display:none}
    .search input{min-height:44px; padding-right:12px; font-size:16px}
    .group-toggle{min-height:44px; min-width:44px}
    .toc-groups a{padding:8px 10px}
    .toc ul ul a{padding:7px 10px}
    .doc-content :global([id]){scroll-margin-top:calc(var(--hdr, 64px) + 76px)}
    .doc-content h2{font-size:25px}
    .page-block{padding:20px 16px}
    .device-check{padding:12px 14px}
    .device-check dl > div{grid-template-columns:1fr; gap:2px}
  }
  @media(max-width:480px){
    .doc-intro{margin-bottom:8px}
    .toc{margin-bottom:20px}
    .page-block{padding:20px 0; border:0; border-top:1px solid var(--rule-strong); border-radius:0; background:none; margin-bottom:16px}
    .task-heading{gap:10px; margin-bottom:12px}
    .page-block h3{flex-basis:100%; font-size:20px}
    .doc-content .open-link{min-height:40px; padding:7px 12px}
    .doc-content .steps{gap:10px}
    .steps li{padding-left:32px}
    .section-nav{gap:8px}
    .section-nav a{padding:12px; font-size:14px}
    .callout{padding:16px}
  }
</style>
