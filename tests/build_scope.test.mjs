import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import test from 'node:test';
import { LAB_ROUTE_ID, PRODUCT_ROUTES, prerenderConfig } from '../kit.config.js';
import { LAB_PATH, UNROUTED_PATH } from '../src/lib/buildScope.ts';

const routeFiles = () => readdirSync(new URL('../src/routes', import.meta.url), { recursive: true }).map((file) => file.split('\\').join('/'));

// Route ids (`/app/play`) of every +page.svelte under src/routes.
function pageRouteIds() {
  return routeFiles()
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
    'classify the new route in kit.config.js',
  );
  for (const id of ['/lab/mobile-soak', '/single-engine']) assert.ok(routes.includes(id), `${id} disappeared; update this test`);
});

test('a product build fails when a non-lab route goes unprerendered', () => {
  const { handleUnseenRoutes } = prerenderConfig('product');
  assert.doesNotThrow(() => handleUnseenRoutes({ routes: ['/lab/mobile-soak', '/single-engine'], message: 'unseen' }));
  assert.throws(() => handleUnseenRoutes({ routes: ['/lab/mobile-soak', '/app/new'], message: 'unseen' }), /unseen/);
});

test('dev and non-product builds keep the Kit default of prerendering every route', (t) => {
  const saved = process.env.BUILD_SCOPE;
  t.after(() => {
    if (saved === undefined) delete process.env.BUILD_SCOPE;
    else process.env.BUILD_SCOPE = saved;
  });
  delete process.env.BUILD_SCOPE;
  assert.equal(prerenderConfig(), undefined);
  assert.equal(prerenderConfig(''), undefined);
});

// Product builds still bundle lab routes into Kit's client router; the reroute hook must refuse them there.
async function rerouteFor(scope) {
  globalThis.__BUILD_SCOPE__ = scope;
  try {
    return (await import(`../src/hooks.ts?scope=${scope}`)).reroute;
  } finally {
    delete globalThis.__BUILD_SCOPE__;
  }
}

test('product builds refuse client navigation to lab routes; other builds install no hook', async () => {
  assert.equal(LAB_PATH.source, LAB_ROUTE_ID.source);
  const reroute = await rerouteFor('product');
  for (const pathname of ['/lab/mobile-soak/', '/lab/copy-prototypes/x/', '/single-engine/', '/single-engine', '/l%61b/mobile-soak/', '/%6Cab/']) {
    assert.equal(reroute({ url: new URL(pathname, 'https://0x88.app') }), UNROUTED_PATH, pathname);
  }
  for (const pathname of [...PRODUCT_ROUTES, '/labs/', '/single-engines/']) {
    assert.equal(reroute({ url: new URL(pathname, 'https://0x88.app') }), undefined);
  }
  for (const scope of ['', 'dev']) assert.equal(await rerouteFor(scope), undefined);
  // The target must match no route, or the hook would mount that page instead of 404ing.
  assert.ok(!pageRouteIds().some((id) => id === UNROUTED_PATH.replace(/\/$/, '')), UNROUTED_PATH);
});
