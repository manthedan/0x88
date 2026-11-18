import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import test from 'node:test';
import { LAB_ROUTE_ID, PRODUCT_ROUTES, prerenderConfig } from '../svelte.config.js';

// Route ids (`/app/play`) of every +page.svelte under src/routes.
function pageRouteIds() {
  return readdirSync(new URL('../src/routes', import.meta.url), { recursive: true })
    .map((file) => file.split('\\').join('/'))
    .filter((file) => file === '+page.svelte' || file.endsWith('/+page.svelte'))
    .map((file) => `/${file.slice(0, -'+page.svelte'.length)}`.replace(/(.)\/$/, '$1'))
    .sort();
}

const asRouteId = (entry) => entry.replace(/(.)\/$/, '$1');

test('product builds prerender only product entries and never crawl into lab routes', () => {
  const config = prerenderConfig('product');
  assert.equal(config.crawl, false);
  assert.deepEqual(config.entries, PRODUCT_ROUTES);
  for (const entry of config.entries) {
    assert.notEqual(entry, '*');
    assert.doesNotMatch(entry, LAB_ROUTE_ID, `${entry} is a lab route in the product entries`);
  }
});

test('every page route is either a product entry or a lab route', () => {
  const routes = pageRouteIds();
  const product = PRODUCT_ROUTES.map(asRouteId);
  assert.deepEqual(
    routes.filter((id) => !LAB_ROUTE_ID.test(id)),
    [...product].sort(),
    'classify the new route in svelte.config.js',
  );
  for (const id of ['/lab/mobile-soak', '/single-engine']) assert.ok(routes.includes(id), `${id} disappeared; update this test`);
});

test('a product build fails when a non-lab route goes unprerendered', () => {
  const { handleUnseenRoutes } = prerenderConfig('product');
  assert.doesNotThrow(() => handleUnseenRoutes({ routes: ['/lab/mobile-soak', '/single-engine'], message: 'unseen' }));
  assert.throws(() => handleUnseenRoutes({ routes: ['/lab/mobile-soak', '/app/new'], message: 'unseen' }), /unseen/);
});

test('dev and non-product builds keep the Kit default of prerendering every route', () => {
  assert.equal(prerenderConfig(undefined), undefined);
  assert.equal(prerenderConfig(''), undefined);
});
