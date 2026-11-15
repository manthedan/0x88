import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { externalProxyRewrites } from '../scripts/dist_browser_smoke.mjs';

test('dist smoke proxies the paths netlify.toml rewrites to the artifact host', () => {
  const rewrites = externalProxyRewrites(readFileSync('netlify.toml', 'utf8'));
  assert.ok(
    rewrites.some(({ from, to }) => from === '/stockfish/stockfish-19-lite.wasm' && to.startsWith('https://')),
    'the Stockfish wasm is served through a Netlify proxy, so the static smoke server must proxy it too',
  );
});

test('only status-200 rewrites to another host count as proxies', () => {
  const toml = `
[[redirects]]
  from = "/a.wasm"
  to = "https://cdn.example/a.wasm"
  status = 200
  force = true

[[redirects]]
  from = "/old"
  to = "/new/"
  status = 301

[[redirects]]
  from = "/moved"
  to = "https://elsewhere.example/"
  status = 301

[[headers]]
  for = "/*"
`;
  assert.deepEqual(externalProxyRewrites(toml), [{ from: '/a.wasm', to: 'https://cdn.example/a.wasm' }]);
});
