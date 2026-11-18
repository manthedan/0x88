// Pure parts of scripts/feature_walk.mjs: seeding, action/value choice, in-page
// expression builders, and invariant evaluation. No browser here, so tests cover it.
import { featuresOn, routeForPath } from '../../src/features/registry.ts';
import { DEFAULT_PLAY_PREFERENCES, PLAY_PREFERENCES_STORAGE_KEY } from '../../src/lc0/playPreferences.ts';
import { actionableErrorEntries } from '../production_browser_smoke.mjs';

/** Numeric seeds are used as-is; any other string hashes (FNV-1a) so `--seed ci-42` works. */
export function parseSeed(value) {
  const text = String(value);
  if (/^\d+$/.test(text)) return Number(text) >>> 0;
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193);
  return hash >>> 0;
}

/** mulberry32: tiny, fast, and identical across Node versions. */
export function createRng(seed) {
  let state = parseSeed(seed);
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function chooseIndex(rng, length) {
  return length > 0 ? Math.min(length - 1, Math.floor(rng() * length)) : -1;
}

/** Every registry action available on a route, flattened with its owning feature. */
export function routeActions(route) {
  return featuresOn(route).flatMap((feature) => feature.actions.map((action) => ({ feature: feature.id, action })));
}

/** Weighted choice among eligible candidates; null when nothing is eligible. */
export function chooseAction(rng, candidates) {
  const total = candidates.reduce((sum, candidate) => sum + (candidate.action.weight ?? 1), 0);
  if (!(total > 0)) return null;
  let target = rng() * total;
  for (const candidate of candidates) {
    target -= candidate.action.weight ?? 1;
    if (target < 0) return candidate;
  }
  return candidates.at(-1);
}

/**
 * Pick one of the registry values that the page offers (`available`, e.g. select
 * options; null = any), preferring a change from `current` so the step does something.
 */
export function chooseValue(rng, values, available = null, current = null) {
  const offered = available ? values.filter((value) => available.includes(value)) : [...values];
  const changing = offered.filter((value) => value !== current);
  const pool = changing.length ? changing : offered;
  return pool.length ? pool[chooseIndex(rng, pool.length)] : null;
}

export function routeInvariants(route) {
  return [...new Set(featuresOn(route).flatMap((feature) => feature.invariants))];
}

export function routeOwnedIds(route) {
  return [...new Set(featuresOn(route).flatMap((feature) => feature.ids))];
}

/**
 * Per node: visible, enabled, and inside the open modal dialog when there is one (a modal
 * makes the rest inert). Self-contained so Playwright's evaluateAll can run it as-is.
 */
export function usableNodes(nodes) {
  const modal = document.querySelector('dialog:modal');
  const shown = (candidate) => candidate?.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) ?? !!candidate?.getClientRects().length;
  return nodes.map((node) => {
    // Styled toggles hide the checkbox itself; its label is what the user sees.
    const visible = shown(node) || (node.type === 'checkbox' && [...(node.labels ?? [])].some(shown));
    // :disabled also covers controls inside a disabled <fieldset> (Arena settings during a match).
    const enabled = !node.disabled && !node.matches?.(':disabled') && node.getAttribute('aria-disabled') !== 'true' && !node.closest('[inert]');
    return visible && enabled && (!modal || modal.contains(node));
  });
}

/** One browser expression returning an eligibility boolean per candidate, in order. */
export function eligibilityExpression(candidates) {
  const checks = candidates.map(
    ({ action }) =>
      `(() => { try { return usableNodes([...document.querySelectorAll(${JSON.stringify(action.when)})]).some(Boolean) && !!(${action.whenExpr ?? 'true'}); } catch { return false; } })()`,
  );
  return `(() => { const usableNodes = ${usableNodes.toString()}; return [${checks.join(',')}]; })()`;
}

/**
 * Index of the step's target among all selector matches, drawn only from usable ones
 * (a visible-but-disabled #navNext must not be clicked); `pick` draws, otherwise the
 * first usable match. -1 when none is usable.
 */
export function chooseTargetIndex(rng, flags, pick) {
  const usable = flags.flatMap((flag, index) => (flag ? [index] : []));
  if (!usable.length) return -1;
  return pick ? usable[chooseIndex(rng, usable.length)] : usable[0];
}

// Click a legal non-promotion move for `side` on the accessible board; `roll` is a host RNG draw.
// Every selection re-renders the grid, so squares are re-queried by key after each click.
function boardMove(side, roll) {
  const host = document.querySelector('#keyboardBoard');
  if (!host) return { error: 'no #keyboardBoard' };
  const square = (key) => host.querySelector(`[data-square="${key}"]`);
  host.querySelector('[aria-selected="true"]')?.click();
  const own = [...host.querySelectorAll('[data-square]')].filter((cell) => cell.getAttribute('aria-label')?.includes(side)).map((cell) => cell.dataset.square);
  const start = Math.floor(roll * own.length);
  for (let i = 0; i < own.length; i++) {
    const from = own[(start + i) % own.length];
    const pawn = square(from).getAttribute('aria-label').includes('pawn');
    square(from).click();
    const destinations = [...host.querySelectorAll('[aria-label*="legal destination"]')]
      .map((cell) => cell.dataset.square)
      .filter((key) => !pawn || !/[18]$/.test(key));
    if (!destinations.length) {
      if (square(from)?.getAttribute('aria-selected') === 'true') square(from).click();
      continue;
    }
    const to = destinations[Math.floor(((roll * 7919) % 1) * destinations.length)];
    square(to).click();
    return { move: from + to };
  }
  // Mate, stalemate, or only promotions: an ordinary position, so the step is a no-op, not a failure.
  return { skipped: `no legal non-promotion move for ${side}` };
}

export function boardMoveExpression(sideExpression, roll) {
  return `(${boardMove.toString()})(${sideExpression}, ${JSON.stringify(roll)})`;
}

/** Route for a page URL, but only on the walk's own origin (another origin is never "in the app"). */
export function locateRoute(href, baseUrl) {
  const url = new URL(href);
  if (url.origin !== new URL(baseUrl).origin) return null;
  return routeForPath(url.pathname);
}

/** Replay command carrying every option that changes which actions are eligible when. */
export function replayCommand(args, seed) {
  const flags = [
    `--base-url ${args.baseUrl}`,
    `--seed ${seed}`,
    `--steps ${args.steps}`,
    `--start ${args.start}`,
    `--viewport ${args.viewport}`,
    `--settle-ms ${args.settleMs}`,
    `--ready-timeout ${args.readyTimeout}`,
    ...(args.keepGoing ? ['--keep-going'] : []),
    ...(args.headed ? ['--headed'] : []),
  ];
  return `node scripts/feature_walk.mjs ${flags.join(' ')}`;
}

// One page read per step; `ids` are the route's owned ids, `ready` its mount expression.
function snapshot(ids, ready) {
  const seen = new Map();
  for (const node of document.querySelectorAll('[id]')) seen.set(node.id, (seen.get(node.id) ?? 0) + 1);
  return {
    pathname: location.pathname,
    documentId: window.__featureWalkDocumentId ?? null,
    ready,
    scrollWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
    presentIds: ids.filter((id) => seen.has(id)),
    duplicateIds: [...seen].filter(([, count]) => count > 1).map(([id]) => id),
  };
}

export function snapshotExpression(ids, readyExpression) {
  return `(${snapshot.toString()})(${JSON.stringify(ids)}, (() => { try { return !!(${readyExpression}); } catch { return false; } })())`;
}

/**
 * Evaluate invariants against one snapshot. `context` carries host-side evidence:
 * route, expectedIds, pageErrors/consoleErrors seen during the step, the document
 * id before the step, and whether the action may reload. Returns failures only.
 */
export function evaluateInvariants(names, snap, context) {
  const failures = [];
  const fail = (invariant, detail) => failures.push({ invariant, detail });
  for (const name of names) {
    if (name === 'noPageErrors' && context.pageErrors?.length) fail(name, context.pageErrors);
    else if (name === 'noConsoleErrors') {
      const errors = actionableErrorEntries(context.consoleErrors ?? []);
      if (errors.length) fail(name, errors);
    } else if (name === 'noHorizontalOverflow' && snap.scrollWidth > snap.viewportWidth + 1)
      fail(name, `document is ${snap.scrollWidth}px wide in a ${snap.viewportWidth}px viewport`);
    else if (name === 'uniqueIds' && snap.duplicateIds.length) fail(name, snap.duplicateIds);
    else if (name === 'ownedIdsPresent') {
      const present = new Set(snap.presentIds);
      const missing = (context.expectedIds ?? []).filter((id) => !present.has(id));
      if (missing.length)
        fail(
          name,
          missing.map((id) => `#${id}`),
        );
    } else if (name === 'noFullReload' && !context.allowsReload && context.documentIdBefore && snap.documentId !== context.documentIdBefore)
      fail(name, `document changed from ${context.documentIdBefore} to ${snap.documentId}`);
    else if (name === 'routeReady' && context.route && !snap.ready) fail(name, `${context.route} did not finish mounting at ${snap.pathname}`);
  }
  return failures;
}

/**
 * localStorage preset for a fresh walk context. Play defaults to Maia3 sampling, which draws
 * from the app's unseeded Math.random; the app's own "Top move" (argmax) style makes replies
 * replayable. Applied only when the key is absent, so in-walk preference changes persist.
 */
export function walkStoragePreset() {
  return { key: PLAY_PREFERENCES_STORAGE_KEY, value: JSON.stringify({ ...DEFAULT_PLAY_PREFERENCES, maiaStyle: 'argmax' }) };
}
