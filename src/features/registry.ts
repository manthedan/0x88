/**
 * Feature map: one typed record per user-facing feature, not per page.
 *
 * Each feature declares the route(s) it lives on, the DOM ids it owns, the user
 * actions a walker can perform, and the invariants to check afterwards. Three
 * consumers read it:
 *   - src/features/*Dom.ts derive per-route id unions that type `el('…')` lookups;
 *   - tests/feature_registry.test.mjs proves every owned id exists in route markup
 *     and every literal controller lookup is owned (a missing id throws at runtime);
 *   - scripts/feature_walk.mjs performs a seeded in-app walk over the actions.
 *
 * Type-only imports keep this file out of the app bundle; controllers import
 * only the per-route Dom modules, which erase to the untyped browserDom helpers.
 */

export type RouteKey = 'home' | 'docs' | 'analysis' | 'play' | 'arena';

export type RouteSpec = {
  path: string;
  /** Svelte page whose markup (plus the $lib components it imports) defines static ids. */
  markup: string;
  /** Controllers whose literal `el('…')` lookups must all be owned by a feature on this route. */
  controllers: readonly string[];
  /** Browser expression that is true once the route's controller has mounted. */
  ready: string;
};

/**
 * A declarative step. `target` is a CSS selector. `pick` steps choose one of
 * the visible matches; `select`/`fill` choose one of `values` (present options
 * only for selects). All choices come from the walk's seeded RNG.
 */
export type Step =
  | { do: 'click'; target: string; pick?: true }
  | { do: 'select'; target: string; values: readonly string[]; pick?: true }
  | { do: 'fill'; target: string; values: readonly string[] }
  | { do: 'check'; target: string }
  | { do: 'press'; target: string; key: string }
  /** Click a random legal non-promotion move on #keyboardBoard for the side named by `side` (an expression). */
  | { do: 'boardMove'; side: string }
  | { do: 'waitFor'; expression: string; timeoutMs?: number };

export type Action = {
  name: string;
  /** Selector that must be visible and enabled for the action to be eligible (inside the open dialog, if any). */
  when: string;
  whenExpr?: string;
  steps: readonly Step[];
  /** Relative selection weight; default 1. */
  weight?: number;
  /** The action may legitimately reload the document (e.g. a query-carrying link). */
  allowsReload?: true;
  /** Route the action must land on; the walk fails the action if the URL never gets there. */
  navigatesTo?: RouteKey;
};

export type InvariantName = 'noPageErrors' | 'noConsoleErrors' | 'noHorizontalOverflow' | 'uniqueIds' | 'ownedIdsPresent' | 'noFullReload' | 'routeReady';

export type Feature = {
  id: string;
  routes: readonly RouteKey[];
  /** DOM ids this feature owns; every one exists in the route markup once mounted. */
  ids: readonly string[];
  actions: readonly Action[];
  invariants: readonly InvariantName[];
};

/** Set by src/routes/+layout.svelte onMount, i.e. after the client hydrated the page. */
export const HYDRATED = `document.documentElement.hasAttribute('data-hydrated')`;

export const ROUTES = {
  // Home and Docs headings are prerendered, so they also wait for the root layout's post-hydration flag.
  home: { path: '/', markup: 'src/routes/+page.svelte', controllers: [], ready: `!!document.querySelector('#hero-h1') && ${HYDRATED}` },
  docs: { path: '/docs/', markup: 'src/routes/docs/+page.svelte', controllers: [], ready: `!!document.querySelector('#doc-h1') && ${HYDRATED}` },
  analysis: {
    path: '/app/analysis/',
    markup: 'src/routes/app/analysis/+page.svelte',
    controllers: ['src/lc0/analysisBrowser.ts', 'src/lc0/criticalMomentsPanel.ts'],
    ready: `!!document.querySelector('#ground cg-board') && !!window.__analysisSchedulingState`,
  },
  play: {
    path: '/app/play/',
    markup: 'src/routes/app/play/+page.svelte',
    controllers: ['src/lc0/playBrowser.ts'],
    ready: `!!document.querySelector('#ground cg-board') && !!window.__playUserMove`,
  },
  arena: {
    path: '/app/arena/',
    markup: 'src/routes/app/arena/+page.svelte',
    controllers: ['src/lc0/arenaBrowser.ts'],
    ready: `!!document.querySelector('#ground cg-board') && !!document.querySelector('#arenaSeatList .seat-fam')`,
  },
} as const satisfies Record<RouteKey, RouteSpec>;

const ALL_ROUTES = ['home', 'docs', 'analysis', 'play', 'arena'] as const satisfies readonly RouteKey[];
const APP_ROUTES = ['analysis', 'play', 'arena'] as const satisfies readonly RouteKey[];
const OWNED = ['ownedIdsPresent'] as const satisfies readonly InvariantName[];

const MIDGAME = 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 6';
const PLAY_USER_SIDE = `document.querySelector('[data-player="you"] [data-meta]')?.textContent?.includes('Black') ? 'Black' : 'White'`;
const ANALYSIS_SIDE = `window.__analysisSchedulingState?.().fen.split(' ')[1] === 'b' ? 'Black' : 'White'`;

/** Header links: desktop primary nav, or the menu toggle on narrow screens. */
function navActions(route: RouteKey): Action[] {
  const href = ROUTES[route].path;
  const away = `location.pathname !== ${JSON.stringify(href)}`;
  return [
    {
      name: `go:${route}`,
      when: `.site-header nav.primary a[href="${href}"]`,
      whenExpr: away,
      steps: [{ do: 'click', target: `.site-header nav.primary a[href="${href}"]` }],
      navigatesTo: route,
    },
    {
      // Toggle only a closed menu: a cancelled navigation can leave it open.
      name: `go:${route}:mobile`,
      when: '.site-header .menu-toggle[aria-expanded="false"]',
      whenExpr: away,
      steps: [
        { do: 'click', target: '.site-header .menu-toggle' },
        { do: 'click', target: `.mobile-nav-menu a[href="${href}"]` },
      ],
      navigatesTo: route,
    },
    {
      name: `go:${route}:menuOpen`,
      when: `.mobile-nav-menu a[href="${href}"]`,
      whenExpr: away,
      steps: [{ do: 'click', target: `.mobile-nav-menu a[href="${href}"]` }],
      navigatesTo: route,
    },
  ];
}

export const FEATURES = [
  {
    id: 'nav',
    routes: ALL_ROUTES,
    ids: [],
    actions: [
      ...ALL_ROUTES.flatMap(navActions),
      { name: 'closeMenu', when: '.site-header .menu-toggle[aria-expanded="true"]', steps: [{ do: 'click', target: '.site-header .menu-toggle' }] },
      { name: 'toggleTheme', when: '.site-header .theme-toggle', steps: [{ do: 'click', target: '.site-header .theme-toggle' }], weight: 0.3 },
    ],
    invariants: ['noPageErrors', 'noConsoleErrors', 'noHorizontalOverflow', 'uniqueIds', 'noFullReload', 'routeReady'],
  },
  {
    // The large-net download dialog is created on demand by largeNetConsent.ts. The walk always
    // declines, so it never starts a multi-hundred-MB download; any open dialog restricts the walk to its actions.
    id: 'consent.largeNet',
    routes: APP_ROUTES,
    ids: [],
    actions: [
      {
        name: 'decline',
        when: 'dialog.large-net-consent[open] button:not(.primary)',
        steps: [{ do: 'click', target: 'dialog.large-net-consent[open] button:not(.primary)' }],
      },
    ],
    invariants: [],
  },
  {
    id: 'home.landing',
    routes: ['home'],
    ids: ['main', 'hero-h1', 'device-details', 'engines-h'],
    // One action per hero link so each declares where it must land (together as likely as one action).
    actions: APP_ROUTES.map(
      (route): Action => ({
        name: `heroCta:${route}`,
        when: `.hero a.btn[href="${ROUTES[route].path}"]`,
        steps: [{ do: 'click', target: `.hero a.btn[href="${ROUTES[route].path}"]` }],
        weight: 1 / APP_ROUTES.length,
        navigatesTo: route,
      }),
    ),
    invariants: OWNED,
  },
  {
    id: 'docs.search',
    routes: ['docs'],
    ids: ['main', 'doc-h1', 'docSearch', 'pages-analysis'],
    actions: [
      // Narrow screens hide the search input until the contents panel opens.
      { name: 'openSearch', when: '.toc .mobile-search', steps: [{ do: 'click', target: '.toc .mobile-search' }] },
      { name: 'search', when: '#docSearch', steps: [{ do: 'fill', target: '#docSearch', values: ['cache', 'arena', 'webgpu', ''] }] },
    ],
    invariants: OWNED,
  },

  {
    id: 'play.board',
    routes: ['play'],
    ids: ['main', 'topStrip', 'ground', 'keyboardBoard', 'boardAnnouncement', 'bottomStrip', 'flip'],
    actions: [
      { name: 'flip', when: '#flip', steps: [{ do: 'click', target: '#flip' }], weight: 0.3 },
      {
        name: 'move',
        when: '#keyboardBoard',
        // Anchored: while browsing history the status starts "Viewing … Live game: Your move." and moves are rejected.
        whenExpr: `/^(Your move|You are in check)/.test(document.querySelector('#status')?.textContent?.trim() ?? '')`,
        steps: [{ do: 'boardMove', side: PLAY_USER_SIDE }],
        weight: 4,
      },
    ],
    invariants: OWNED,
  },
  {
    id: 'play.status',
    routes: ['play'],
    ids: [
      'gameAnnouncement',
      'playRestoreNotice',
      'playRestoreNew',
      'playRestoreDismiss',
      'status',
      'engineNote',
      'runtimeWarning',
      'retryEngine',
      'dlProgress',
      'progressAnnouncement',
    ],
    actions: [
      { name: 'dismissRestore', when: '#playRestoreDismiss', steps: [{ do: 'click', target: '#playRestoreDismiss' }] },
      { name: 'retryEngine', when: '#retryEngine', steps: [{ do: 'click', target: '#retryEngine' }] },
    ],
    invariants: OWNED,
  },
  {
    id: 'play.opponentPicker',
    routes: ['play'],
    ids: [
      'changeOpponent',
      'setupPane',
      'positionNotice',
      'positionNoticeText',
      'useStartingPosition',
      'engineSelect',
      'opponentBlurb',
      'maia3Controls',
      'maia3EloValue',
      'maia3EloBand',
      'levelSelect',
      'colorSelect',
      'opponentDetails',
      'detailsLabel',
      'maia3Advanced',
      'maia3Elo',
      'maia3Style',
      'maia3TemperatureField',
      'maia3Temperature',
      'maia3TopPField',
      'maia3TopP',
      'levelCaption',
      'engineCaution',
    ],
    actions: [
      { name: 'changeOpponent', when: '#changeOpponent', steps: [{ do: 'click', target: '#changeOpponent' }] },
      {
        // No big nets (lc0-bt4/t3): desktop Play loads them without a consent dialog.
        name: 'pickEngine',
        when: '#engineSelect',
        steps: [{ do: 'select', target: '#engineSelect', values: ['maia3', 'lc0-small', 'sf-lite', 'berserk', 'centipawn'] }],
        weight: 2,
      },
      { name: 'pickLevel', when: '#levelSelect', steps: [{ do: 'select', target: '#levelSelect', values: ['0', '1'] }] },
      {
        // Explicit sides only: Random uses the app's unseeded Math.random and breaks seed replay.
        name: 'pickColor',
        when: '.color-choice [data-color="white"], .color-choice [data-color="black"]',
        steps: [{ do: 'click', target: '.color-choice [data-color="white"], .color-choice [data-color="black"]', pick: true }],
      },
      { name: 'ratingPreset', when: '.rating-presets [data-elo]', steps: [{ do: 'click', target: '.rating-presets [data-elo]', pick: true }] },
      { name: 'toggleDetails', when: '#opponentDetails > summary', steps: [{ do: 'click', target: '#opponentDetails > summary' }], weight: 0.5 },
    ],
    invariants: OWNED,
  },
  {
    id: 'play.game',
    routes: ['play'],
    ids: [
      'restartBanner',
      'restartMessage',
      'confirmRestart',
      'dismissRestart',
      'gameOverCard',
      'resultHeadline',
      'resultScore',
      'resultReason',
      'openingName',
      'moveList',
      'navFirst',
      'navPrev',
      'navNext',
      'navLast',
      'moveForm',
      'moveInput',
      'takeback',
      'resign',
      'newGame',
    ],
    actions: [
      { name: 'newGame', when: '#newGame', steps: [{ do: 'click', target: '#newGame' }], weight: 2 },
      { name: 'confirmRestart', when: '#confirmRestart', steps: [{ do: 'click', target: '#confirmRestart' }] },
      { name: 'dismissRestart', when: '#dismissRestart', steps: [{ do: 'click', target: '#dismissRestart' }] },
      {
        name: 'browseMoves',
        when: '#navFirst, #navPrev, #navNext, #navLast',
        steps: [{ do: 'click', target: '#navFirst, #navPrev, #navNext, #navLast', pick: true }],
      },
      { name: 'takeback', when: '#takeback', steps: [{ do: 'click', target: '#takeback' }] },
      {
        name: 'resign',
        when: '#resign',
        steps: [
          { do: 'click', target: '#resign' },
          { do: 'click', target: '#resign' },
        ],
        weight: 0.3,
      },
    ],
    invariants: OWNED,
  },
  {
    id: 'play.record',
    routes: ['play'],
    ids: ['copyPgn', 'downloadPgn', 'analyzeGame', 'analyzeGameCta'],
    actions: [
      { name: 'copyPgn', when: '#copyPgn', steps: [{ do: 'click', target: '#copyPgn' }], weight: 0.5 },
      // Mid-game the link opens a new tab (target=_blank), which the walk cannot follow; only the same-tab variant.
      {
        name: 'analyzeGame',
        when: '#analyzeGame:not([target])',
        steps: [{ do: 'click', target: '#analyzeGame:not([target])' }],
        weight: 0.5,
        allowsReload: true,
        navigatesTo: 'analysis',
      },
    ],
    invariants: OWNED,
  },

  {
    id: 'analysis.board',
    routes: ['analysis'],
    ids: [
      'main',
      'evalSourceButton',
      'evalSource',
      'evalScore',
      'evalSpreadLabel',
      'evalTurn',
      'evalSourceMenu',
      'evalWhite',
      'evalSpread',
      'evalTicks',
      'ground',
      'keyboardBoard',
      'boardAnnouncement',
      'evalCompareDetails',
      'evalDetailsSummary',
      'evalDetailList',
    ],
    actions: [{ name: 'move', when: '#keyboardBoard', steps: [{ do: 'boardMove', side: ANALYSIS_SIDE }], weight: 4 }],
    invariants: OWNED,
  },
  {
    id: 'analysis.toolbar',
    routes: ['analysis'],
    ids: [
      'navStart',
      'navBack',
      'navForward',
      'navEnd',
      'emptyLoadGame',
      'boardToolDisclosure',
      'boardToolsButton',
      'boardTools',
      'flip',
      'copyTools',
      'copyMenuButton',
      'copyMenu',
      'quickCopyFen',
      'quickCopyPgn',
      'copyLink',
      'playFromHere',
      'openPosition',
      'playFromHereReason',
      'positionTools',
      'positionShareFallback',
      'positionShareUrl',
      'fenInput',
      'loadFen',
      'copyFen',
      'reset',
      'pgnInput',
      'loadPgn',
      'copyPgn',
    ],
    actions: [
      {
        name: 'browseMoves',
        when: '#navStart, #navBack, #navForward, #navEnd',
        steps: [{ do: 'click', target: '#navStart, #navBack, #navForward, #navEnd', pick: true }],
      },
      { name: 'boardTools', when: '#boardToolsButton', steps: [{ do: 'click', target: '#boardToolsButton' }], weight: 0.5 },
      { name: 'flip', when: '#flip', steps: [{ do: 'click', target: '#flip' }], weight: 0.3 },
      { name: 'copyMenu', when: '#copyMenuButton', steps: [{ do: 'click', target: '#copyMenuButton' }], weight: 0.5 },
      { name: 'copyFen', when: '#quickCopyFen', steps: [{ do: 'click', target: '#quickCopyFen' }], weight: 0.5 },
      { name: 'openPosition', when: '#openPosition', steps: [{ do: 'click', target: '#openPosition' }], weight: 0.5 },
      {
        name: 'loadFen',
        when: '#fenInput',
        steps: [
          { do: 'fill', target: '#fenInput', values: [MIDGAME, 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'] },
          { do: 'click', target: '#loadFen' },
        ],
      },
      { name: 'reset', when: '#reset', steps: [{ do: 'click', target: '#reset' }], weight: 0.5 },
      {
        name: 'playFromHere',
        when: '#playFromHere',
        steps: [{ do: 'click', target: '#playFromHere' }],
        weight: 0.3,
        allowsReload: true,
        navigatesTo: 'play',
      },
    ],
    invariants: OWNED,
  },
  {
    id: 'analysis.status',
    routes: ['analysis'],
    ids: ['analysisRestoreNotice', 'analysisRestoreNew', 'analysisRestoreDismiss', 'message', 'restorePreviousAnalysis', 'autosaveNotice', 'pgnImportNotice'],
    actions: [{ name: 'dismissRestore', when: '#analysisRestoreDismiss', steps: [{ do: 'click', target: '#analysisRestoreDismiss' }] }],
    invariants: OWNED,
  },
  {
    id: 'analysis.tabs',
    routes: ['analysis'],
    ids: ['analysisContent', 'movesTab', 'enginesTab', 'enginesTabCount', 'reviewTab', 'comparisonTab', 'humanTab'],
    actions: [
      { name: 'switchTab', when: '.workbench-tabs [role="tab"]', steps: [{ do: 'click', target: '.workbench-tabs [role="tab"]', pick: true }], weight: 2 },
    ],
    invariants: OWNED,
  },
  {
    id: 'analysis.engines',
    routes: ['analysis'],
    ids: [
      'analysisStatus',
      'analyze',
      'analyzeLabel',
      'stop',
      'stopProgress',
      'analysisSearchProgress',
      'downloadProgress',
      'runtimeWarning',
      'runtimeAudit',
      'manageEngines',
      'engineCount',
      'addEngine',
      'addEngineHint',
      'comparisonInsight',
      'engineSummaries',
      'engineLegend',
      'engineManager',
      'closeEngineManager',
      'engineSetupSummary',
      'autoAnalyze',
      'multiPvInput',
      'showAllLineArrows',
      'engineList',
      'addEngineInManager',
      'lc0RuntimeSelect',
      'backend',
      'recklessRuntimeInfo',
    ],
    actions: [
      { name: 'analyze', when: '#analyze', steps: [{ do: 'click', target: '#analyze' }] },
      { name: 'stop', when: '#stop', steps: [{ do: 'click', target: '#stop' }] },
      { name: 'manageEngines', when: '#manageEngines', steps: [{ do: 'click', target: '#manageEngines' }] },
      { name: 'closeManager', when: '#closeEngineManager', steps: [{ do: 'click', target: '#closeEngineManager' }], weight: 2 },
      { name: 'autoAnalyze', when: '#autoAnalyze', steps: [{ do: 'check', target: '#autoAnalyze' }] },
      { name: 'multiPv', when: '#multiPvInput', steps: [{ do: 'fill', target: '#multiPvInput', values: ['1', '3'] }] },
      { name: 'lineArrows', when: '#showAllLineArrows', steps: [{ do: 'check', target: '#showAllLineArrows' }], weight: 0.5 },
      {
        name: 'addEngine',
        when: '#addEngineInManager',
        whenExpr: `document.querySelectorAll('#engineList .engine-row').length < 3`,
        steps: [{ do: 'click', target: '#addEngineInManager' }],
        weight: 0.5,
      },
      {
        name: 'pickFamily',
        when: '#engineList .row-fam',
        steps: [{ do: 'select', target: '#engineList .row-fam', values: ['lc0', 'sf', 'berserk', 'centipawn'], pick: true }],
      },
    ],
    invariants: OWNED,
  },
  {
    id: 'analysis.moves',
    routes: ['analysis'],
    ids: ['historyPanel', 'openingName', 'movelist'],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'analysis.explorer',
    routes: ['analysis'],
    ids: [
      'openingExplorer',
      'pgnDbSelect',
      'collectionMenu',
      'loadPgnDb',
      'newPgnCollection',
      'renamePgnDb',
      'duplicatePgnDb',
      'exportPgnDbCollection',
      'exportPgnDbSelected',
      'exportPgnDb',
      'importPgnDb',
      'deletePgnDb',
      'pgnDbName',
      'confirmPgnDbName',
      'pgnDbInfo',
      'openingSummary',
      'opening',
      'openingComparison',
      'explorerRating',
      'explorerComparisonInfo',
      'explorerComparisonRows',
      'collectionImport',
      'importGamesFile',
      'importGamesInput',
      'savePgnDb',
      'importGames',
      'downloadPgn',
      'importProgress',
      'importInfo',
      'importPgnDbFile',
      'searchPgnDbPosition',
      'pgnDbList',
      'pgnDbSearchResults',
    ],
    actions: [{ name: 'toggleExplorer', when: '#openingExplorer > summary', steps: [{ do: 'click', target: '#openingExplorer > summary' }], weight: 0.5 }],
    invariants: OWNED,
  },
  {
    id: 'analysis.review',
    routes: ['analysis'],
    ids: [
      'reviewPanel',
      'retryEntry',
      'retrySide',
      'retryStart',
      'retryEntryStatus',
      'retryActive',
      'retryExit',
      'retryCounts',
      'retryStatus',
      'retrySolution',
      'retrySkip',
      'retryReveal',
      'retryNext',
      'retryHuman',
      'retryRating',
      'retryHumanShow',
      'retryHumanStatus',
      'reviewEngine',
      'reviewDepth',
      'reviewGame',
      'reviewStop',
      'reviewStatus',
      'reviewProgress',
      'reviewResults',
      'reviewSummary',
      'reviewGraphSource',
      'reviewChart',
      'reviewSelection',
      'reviewPrevious',
      'reviewNext',
      'reviewMomentsEmpty',
      'reviewCritical',
      'reviewCopyPgn',
      'reviewDownloadPgn',
    ],
    actions: [
      { name: 'reviewGame', when: '#reviewGame', steps: [{ do: 'click', target: '#reviewGame' }], weight: 0.3 },
      { name: 'reviewStop', when: '#reviewStop', steps: [{ do: 'click', target: '#reviewStop' }] },
    ],
    invariants: OWNED,
  },
  {
    id: 'analysis.criticalMoments',
    routes: ['analysis'],
    ids: [
      'criticalMomentsArea',
      'criticalStatus',
      'criticalCompare',
      'criticalCancel',
      'criticalConsent',
      'criticalMaia',
      'criticalRating',
      'criticalConsentText',
      'criticalStart',
      'criticalEngineProgress',
      'criticalSource',
      'criticalReport',
    ],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'analysis.comparison',
    routes: ['analysis'],
    ids: ['engineConsensus', 'engineCompare', 'deltaBaseline'],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'analysis.human',
    routes: ['analysis'],
    ids: ['maia3Enable', 'maia3PanelStatus', 'maia3Grid', 'maia3Caption'],
    actions: [{ name: 'toggleMaia3', when: '#maia3Enable', steps: [{ do: 'check', target: '#maia3Enable' }], weight: 0.5 }],
    invariants: OWNED,
  },
  {
    id: 'analysis.pvPreview',
    routes: ['analysis'],
    ids: [
      'pvPreview',
      'pvPreviewTitle',
      'closePvPreview',
      'pvPreviewDetail',
      'pvPreviewBoard',
      'pvPreviewPosition',
      'pvStart',
      'pvBack',
      'pvForward',
      'pvEnd',
      'pvPreviewMoves',
      'pvPreviewNote',
      'pvInsertHere',
      'pvInsertLine',
    ],
    actions: [
      // .pv-move inserts the line into the game; [data-pv-preview] opens the preview dialog.
      { name: 'openPv', when: '#engineSummaries [data-pv-preview]', steps: [{ do: 'click', target: '#engineSummaries [data-pv-preview]', pick: true }] },
      { name: 'stepPv', when: '#pvStart, #pvBack, #pvForward, #pvEnd', steps: [{ do: 'click', target: '#pvStart, #pvBack, #pvForward, #pvEnd', pick: true }] },
      { name: 'closePv', when: '#closePvPreview', steps: [{ do: 'click', target: '#closePvPreview' }], weight: 2 },
    ],
    invariants: OWNED,
  },
  {
    id: 'analysis.engineProfiles',
    routes: ['analysis'],
    ids: [
      'engineProfileSelect',
      'engineProfileName',
      'saveEngineProfile',
      'deleteEngineProfile',
      'exportEngineProfiles',
      'importEngineProfiles',
      'importEngineProfilesFile',
      'engineProfileSummary',
      'engineProfileFeedback',
    ],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'analysis.gameImport',
    routes: ['analysis'],
    ids: ['importSite', 'importUser', 'importMax', 'importColor', 'fetchGames'],
    actions: [],
    invariants: OWNED,
  },

  {
    id: 'arena.board',
    routes: ['arena'],
    ids: ['main', 'arenaEvalBar', 'ground', 'evalSource'],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'arena.charts',
    routes: ['arena'],
    ids: [
      'chartsPanel',
      'rootChartTab',
      'evalChart',
      'timeChart',
      'npsChart',
      'rootChartTitle',
      'rootChart',
      'rootCandidate',
      'chartLegend',
      'chartPly',
      'chartLive',
    ],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'arena.status',
    routes: ['arena'],
    ids: ['message', 'runtimeWarning', 'engineWarmupStatus', 'engineWarmupTitle', 'engineWarmupPhase', 'engineWarmupDetail', 'downloadProgress'],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'arena.checkpoint',
    routes: ['arena'],
    ids: ['checkpointSection', 'checkpointText', 'checkpointRoster', 'resumeTournament', 'discardTournament'],
    actions: [{ name: 'discard', when: '#discardTournament', steps: [{ do: 'click', target: '#discardTournament' }] }],
    invariants: OWNED,
  },
  {
    id: 'arena.scoreboard',
    routes: ['arena'],
    ids: [
      'editSetup',
      'stop',
      'matchScore',
      'pairing',
      'openingPairsSection',
      'pairMatchup',
      'openingPairs',
      'gameChips',
      'crosstableSection',
      'crosstable',
      'scoreboardTitle',
      'engineOutputsSection',
      'engineRaw',
      'movesBlock',
    ],
    actions: [
      { name: 'stop', when: '#stop', steps: [{ do: 'click', target: '#stop' }], weight: 0.5 },
      { name: 'editSetup', when: '#editSetup', steps: [{ do: 'click', target: '#editSetup' }] },
      { name: 'openGame', when: '#gameChips button', steps: [{ do: 'click', target: '#gameChips button', pick: true }] },
    ],
    invariants: OWNED,
  },
  {
    // The walk stays on fast CPU engines and 1-game, 100 ms matches; big nets only via the declined consent.
    id: 'arena.tournament',
    routes: ['arena'],
    ids: [
      'setupPane',
      'seatA',
      'seatB',
      'tournamentModeSelect',
      'arenaSeatList',
      'addSeat',
      'matchupNote',
      'arenaMemoryNote',
      'budgetModeSelect',
      'movetimeField',
      'movetimeInput',
      'gamesInput',
      'recklessVariantSelect',
      'viridithasVariantSelect',
      'berserkVariantSelect',
      'plentychessVariantSelect',
      'stormphraxVariantSelect',
      'startingPositionSelect',
      'openingTextField',
      'openingText',
      'openingInfo',
      'arenaConcurrencySelect',
      'arenaConcurrencyNote',
      'startSummary',
      'start',
      'setupSummaryText',
      'cancelSetupLoad',
      'runtimeAuditInfo',
    ],
    actions: [
      { name: 'pickSeat', when: '.seat-fam', steps: [{ do: 'select', target: '.seat-fam', values: ['sf', 'berserk', 'plentychess'], pick: true }] },
      {
        name: 'quickMatch',
        when: '#start',
        steps: [
          { do: 'select', target: '#budgetModeSelect', values: ['movetime'] },
          { do: 'fill', target: '#movetimeInput', values: ['100'] },
          // Games are per opening: the built-in suite would turn 1 into 16 games.
          { do: 'select', target: '#startingPositionSelect', values: ['start'] },
          { do: 'fill', target: '#gamesInput', values: ['1'] },
          { do: 'click', target: '#start' },
        ],
      },
      {
        name: 'startingPosition',
        when: '#startingPositionSelect',
        steps: [{ do: 'select', target: '#startingPositionSelect', values: ['start', 'built-in'] }],
        weight: 0.5,
      },
    ],
    invariants: OWNED,
  },
  {
    id: 'arena.runtime',
    routes: ['arena'],
    ids: [
      'lc0PresetSelect',
      'lc0PresetNote',
      'lc0AdvancedRuntime',
      'lc0RuntimeSelect',
      'lc0BatchSizeInput',
      'lc0BatchPipelineDepthInput',
      'lc0InputBackendSelect',
      'lc0EncoderKernelSelect',
      'lc0LegalPriorsSelect',
      'delayInput',
      'cacheEntriesInput',
      'stockfishThreadsInput',
      'recklessRuntimeInfo',
      'viridithasRuntimeInfo',
      'berserkRuntimeInfo',
      'plentychessRuntimeInfo',
      'stormphraxRuntimeInfo',
      'cacheInfo',
      'searchTelemetryInfo',
      'runtimeBadge',
    ],
    actions: [],
    invariants: OWNED,
  },
  {
    id: 'arena.moves',
    routes: ['arena'],
    ids: ['engineEvalInfo', 'reviewBar', 'reviewLabel', 'gameMoves', 'revStart', 'revPrev', 'revNext', 'revEnd', 'revLive'],
    actions: [
      {
        name: 'review',
        when: '#revStart, #revPrev, #revNext, #revEnd, #revLive',
        steps: [{ do: 'click', target: '#revStart, #revPrev, #revNext, #revEnd, #revLive', pick: true }],
      },
    ],
    invariants: OWNED,
  },
  {
    id: 'arena.results',
    routes: ['arena'],
    ids: ['resultSection', 'checkpointStorageText', 'log', 'benchResult', 'exportPgn', 'pgnOut'],
    actions: [{ name: 'toggleResults', when: '#resultSection > summary', steps: [{ do: 'click', target: '#resultSection > summary' }], weight: 0.5 }],
    invariants: OWNED,
  },
] as const satisfies readonly Feature[];

type AnyFeature = (typeof FEATURES)[number];
type FeatureOn<R extends RouteKey> = AnyFeature extends infer F ? (F extends AnyFeature ? (R extends F['routes'][number] ? F : never) : never) : never;

/** Every DOM id owned by a feature on route R. */
export type RouteId<R extends RouteKey> = FeatureOn<R>['ids'][number];
export type FeatureId = AnyFeature['id'];

export function featuresOn(route: RouteKey): readonly Feature[] {
  return FEATURES.filter((feature) => (feature.routes as readonly RouteKey[]).includes(route));
}

export function routeForPath(pathname: string): RouteKey | null {
  const normalized = pathname.endsWith('/') ? pathname : `${pathname}/`;
  for (const [key, spec] of Object.entries(ROUTES) as [RouteKey, RouteSpec][]) if (normalized === spec.path) return key;
  return null;
}
