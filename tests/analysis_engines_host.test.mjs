import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// The engine host stays page-free so the Svelte route can own the DOM: it
// reaches the page only through its AnalysisEnginePort.
const HOST = new URL('../src/lc0/analysis/analysisEngines.ts', import.meta.url);
const IMPORT = /^\s*(?:import|export)\s[^;]*?from\s+'([^']+)'/gm;

function importGraph(entry) {
  const seen = new Map();
  const visit = (url) => {
    if (seen.has(url.href)) return;
    const text = readFileSync(url, 'utf8');
    const specifiers = [...text.matchAll(IMPORT)].map((match) => match[1]);
    seen.set(url.href, { text, specifiers });
    for (const specifier of specifiers) if (specifier.startsWith('.')) visit(new URL(specifier, url));
  };
  visit(entry);
  return seen;
}

test('the Analysis engine host and its imports load no DOM, chessground or Svelte code', () => {
  const graph = importGraph(HOST);
  assert.ok(graph.size > 10, 'the walk followed the host imports');
  for (const [file, { specifiers }] of graph) {
    for (const specifier of specifiers) {
      assert.doesNotMatch(specifier, /(?:^|\/)browserDom(?:\.ts)?$|^chessground|^svelte|\.svelte$/, `${file} imports ${specifier}`);
    }
  }
  const host = graph.get(HOST.href).text;
  assert.doesNotMatch(host, /\bdocument\b|\bHTMLElement\b/);
});
