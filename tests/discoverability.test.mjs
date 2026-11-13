import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { primaryNav } from '../src/lib/nav.ts';

// public/sitemap.xml, robots.txt and llms.txt are static files for search engines
// and AI agents. Keep them in step with the site's primary navigation.

const root = fileURLToPath(new URL('..', import.meta.url));
const read = (rel) => readFileSync(join(root, rel), 'utf8');
const ORIGIN = 'https://0x88.app';
const navUrls = primaryNav.map((item) => ORIGIN + item.href).sort();

test('sitemap.xml lists exactly the primary navigation pages', () => {
  const locs = [...read('public/sitemap.xml').matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]).sort();
  assert.deepEqual(locs, navUrls);
});

test('robots.txt points crawlers at the sitemap and blocks nothing itself', () => {
  const robots = read('public/robots.txt');
  assert.match(robots, /^Sitemap: https:\/\/0x88\.app\/sitemap\.xml$/m);
  assert.doesNotMatch(robots, /^Disallow:/im);
});

test('robots.txt re-allows Google-Extended so Gemini can read the site', () => {
  // Cloudflare's managed robots.txt (prepended at the edge) disallows Google-Extended.
  const robots = read('public/robots.txt');
  assert.match(robots, /^User-agent: Google-Extended\nAllow: \/$/m);
});

test('llms.txt links every workspace and the docs', () => {
  const llms = read('public/llms.txt');
  assert.match(llms, /^# 0x88$/m);
  const links = new Set([...llms.matchAll(/\]\((https:\/\/0x88\.app\/[^)]*)\)/g)].map((m) => m[1]));
  for (const url of navUrls.filter((u) => u !== ORIGIN + '/')) assert.ok(links.has(url), `llms.txt missing ${url}`);
});
