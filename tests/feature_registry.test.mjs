import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { FEATURES, featuresOn, ROUTES, routeForPath } from '../src/features/registry.ts';

// Guard from the "Missing #message" production incident: controllers look elements up by
// string id and a missing id only throws at runtime, so prove both directions statically.

const read = (path) => readFileSync(path, 'utf8');
const LOOKUP = /(?<![\w.$])(?:el|maybeEl|inputEl|selectEl|buttonEl)\(\s*(['"`])([^'"`$]+)\1\s*\)/g;
// `for (const id of ['a', 'b']) el(id)…` — the literal ids reach el() through the loop variable.
const LOOP_LOOKUP = /for \(const (\w+) of \[([^\]]*)\](?: as const)?\)\s*\{?\s*(?:[^;]*?[^\w.$])?(?:el|maybeEl|inputEl|selectEl|buttonEl)\(\1\)/g;
const ID_ATTR = /(?<![\w-])id=(?:"([^"{}]+)"|'([^'{}]+)'|\{`([^`]+)`\})/g;

function controllerLookups(source) {
  const ids = new Set();
  for (const match of source.matchAll(LOOKUP)) ids.add(match[2]);
  for (const match of source.matchAll(LOOP_LOOKUP)) for (const literal of match[2].matchAll(/'([^']+)'/g)) ids.add(literal[1]);
  return ids;
}

/** Static ids of a Svelte file; `${prop}` templates resolve against the given props, others are skipped. */
function markupIds(source, props = {}) {
  const ids = new Set();
  for (const match of source.matchAll(ID_ATTR)) {
    const raw = match[1] ?? match[2] ?? match[3];
    const resolved = raw.replace(/\$\{(\w+)\}/g, (whole, name) => (name in props ? props[name] : whole));
    if (!resolved.includes('${')) ids.add(resolved);
  }
  return ids;
}

/**
 * Page ids that only render inside an `{#if}` branch or `{#each}` body. An id also present in every branch of
 * one top-level `{#if}…{:else}…{/if}` (the dev-mode duplicates) still always renders.
 */
function conditionalIds(source) {
  const occurrences = new Map();
  const stack = [];
  let block = 0;
  for (const match of source.matchAll(/\{#(?:if|each)\b|\{:else\b|\{\/(?:if|each)\}|(?<![\w-])id=(?:"([^"{}]+)"|'([^'{}]+)')/g)) {
    if (match[0].startsWith('{#')) stack.push({ block: block++, branch: 0, hasElse: false });
    else if (match[0] === '{:else') Object.assign(stack.at(-1), { branch: stack.at(-1).branch + 1, hasElse: true });
    else if (match[0].startsWith('{/')) {
      const closed = stack.pop();
      for (const list of occurrences.values())
        for (const at of list) if (at?.block === closed.block) at.branches = closed.hasElse ? closed.branch + 1 : Number.POSITIVE_INFINITY;
    } else {
      const id = match[1] ?? match[2];
      const list = occurrences.get(id) ?? [];
      list.push(stack.length ? { ...stack[0], depth: stack.length } : null);
      occurrences.set(id, list);
    }
  }
  const always = (list) => {
    if (list.includes(null)) return true;
    // Without {:else} a block can render nothing, so only full coverage of an if/else counts.
    const top = list.filter((at) => at.depth === 1);
    return top.some((at) => new Set(top.filter((other) => other.block === at.block).map((other) => other.branch)).size === at.branches);
  };
  return new Set([...occurrences].filter(([, list]) => !always(list)).map(([id]) => id));
}

/** Page ids plus ids of every #lib ($lib before Kit 3) component the page renders, with that usage's string props. */
function routeMarkupIds(pagePath) {
  const page = read(pagePath);
  const ids = markupIds(page);
  for (const [, name, file] of page.matchAll(/import (\w+) from '[$#]lib\/components\/(\w+\.svelte)'/g)) {
    const component = read(`src/lib/components/${file}`);
    for (const usage of page.matchAll(new RegExp(`<${name}\\b([^>]*)>`, 'g'))) {
      const props = Object.fromEntries([...usage[1].matchAll(/(\w+)="([^"]*)"/g)].map((match) => [match[1], match[2]]));
      for (const id of markupIds(component, props)) ids.add(id);
    }
  }
  return ids;
}

/** Ids a controller creates itself: `id="x"` in template HTML or `.id = 'x'`. */
function generatedIds(source) {
  const ids = markupIds(source);
  for (const match of source.matchAll(/\.id = '([^']+)'/g)) ids.add(match[1]);
  return ids;
}

const routeKeys = Object.keys(ROUTES);

test('markupIds resolves component props and skips unresolvable templates', () => {
  // biome-ignore lint/suspicious/noTemplateCurlyInString: Svelte template source under test.
  const ids = markupIds('<div id={`${area}Notice`}></div><b id="plain"></b><i data-id="no"></i><ul id={`toc-${group.id}`}></ul>', { area: 'play' });
  assert.deepEqual([...ids].sort(), ['plain', 'playNotice']);
});

test('controllerLookups sees literal and for-of loop lookups, not computed ones', () => {
  const ids = controllerLookups(`el('a'); maybeEl("b"); x.el('no'); el(dynamic); for (const id of ['c', 'd']) el(id).hidden = true;`);
  assert.deepEqual([...ids].sort(), ['a', 'b', 'c', 'd']);
});

test('conditionalIds flags ids rendered only in some {#if} branches', () => {
  const source = '<a id="always"></a>{#if dev}<b id="both"></b><c id="onlyIf"></c>{:else}<b id="both"></b>{/if}{#if q}<d id="search"></d>{/if}';
  assert.deepEqual([...conditionalIds(source)].sort(), ['onlyIf', 'search']);
});

test('owned ids are not inside conditional markup (ownedIdsPresent checks them on every step)', () => {
  for (const [route, spec] of Object.entries(ROUTES)) {
    const conditional = conditionalIds(read(spec.markup));
    const owned = featuresOn(route).flatMap((feature) => feature.ids.filter((id) => conditional.has(id)).map((id) => `${feature.id}: #${id}`));
    assert.deepEqual(owned, [], `${route} owns ids that are not always rendered`);
  }
});

test('feature ids are unique, and no two features on a route own the same DOM id', () => {
  assert.equal(new Set(FEATURES.map((feature) => feature.id)).size, FEATURES.length);
  for (const route of routeKeys) {
    const owner = new Map();
    for (const feature of featuresOn(route)) {
      for (const id of feature.ids) {
        assert.ok(!owner.has(id), `#${id} on ${route} is owned by both ${owner.get(id)} and ${feature.id}`);
        owner.set(id, feature.id);
      }
    }
  }
});

test('every owned id exists in its route markup, $lib components, or controller-generated HTML', () => {
  for (const [route, spec] of Object.entries(ROUTES)) {
    const available = routeMarkupIds(spec.markup);
    for (const controller of spec.controllers) for (const id of generatedIds(read(controller))) available.add(id);
    const missing = featuresOn(route).flatMap((feature) => feature.ids.filter((id) => !available.has(id)).map((id) => `${feature.id}: #${id}`));
    assert.deepEqual(missing, [], `${route} registry ids absent from ${spec.markup}`);
  }
});

test('every literal el() lookup in a route controller is owned by a feature on that route', () => {
  for (const [route, spec] of Object.entries(ROUTES)) {
    const owned = new Set(featuresOn(route).flatMap((feature) => feature.ids));
    for (const controller of spec.controllers) {
      assert.ok(existsSync(controller), `${controller} is missing`);
      const unowned = [...controllerLookups(read(controller))].filter((id) => !owned.has(id));
      assert.deepEqual(unowned, [], `${controller} looks up ids no ${route} feature owns; add them to src/features/registry.ts`);
    }
  }
});

test('action selectors only reference ids owned on the feature routes', () => {
  for (const feature of FEATURES) {
    const owned = new Set(feature.routes.flatMap((route) => featuresOn(route).flatMap((owner) => owner.ids)));
    for (const action of feature.actions) {
      const selectors = [action.when, ...action.steps.map((step) => step.target).filter(Boolean)];
      for (const selector of selectors)
        for (const [, id] of selector.matchAll(/#([\w-]+)/g))
          assert.ok(owned.has(id), `${feature.id}/${action.name}: #${id} is not owned on ${feature.routes.join(', ')}`);
      for (const step of action.steps) if ('values' in step) assert.ok(step.values.length > 0, `${feature.id}/${action.name} has an empty value list`);
    }
  }
});

test('every route is reachable by nav and has its own walkable actions', () => {
  const nav = FEATURES.find((feature) => feature.id === 'nav');
  for (const route of routeKeys) {
    assert.ok(
      nav.actions.some((action) => action.name === `go:${route}`),
      `no nav action reaches ${route}`,
    );
    assert.ok(
      featuresOn(route).some((feature) => feature.id !== 'nav' && feature.actions.length > 0),
      `${route} has no feature actions to walk`,
    );
    assert.equal(routeForPath(ROUTES[route].path), route);
    assert.equal(routeForPath(ROUTES[route].path.replace(/\/$/, '')), route);
  }
  assert.equal(routeForPath('/lab/mobile-soak/'), null);
});
