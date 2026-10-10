#!/usr/bin/env node
// Timing driver: invoke this whole command under benchlock. Serve the browser
// module + protocol next to an HTML page at --url with COOP/COEP headers.
// --config cells.json --url http://localhost:6190 --json report.json
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium, firefox, webkit } from 'playwright';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
const browsers = { chromium, firefox, webkit };
const browserType = browsers[args.get('browser') || 'chromium'];
if (!browserType) throw Error('Unknown browser');
const browser = await browserType.launch({ headless: true });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
try {
  await page.goto(args.get('url'), { waitUntil: 'load' });
  const config = JSON.parse(readFileSync(args.get('config'), 'utf8'));
  const options = { depth: Number(args.get('depth') || 12), rounds: Number(args.get('rounds') || 5),
    positions: Number(args.get('positions') || 20), threads: Number(args.get('threads') || 1) };
  const report = await page.evaluate(async ({ config, options }) => {
    const { runStockfishBrowserMatrix } = await import('./stockfish_upgrade_browser_matrix.mjs');
    return runStockfishBrowserMatrix(config, options);
  }, { config, options });
  report.errors = errors;
  writeFileSync(args.get('json'), `${JSON.stringify(report, null, 2)}\n`);
  for (const cell of report.cells) console.log(`${cell.round} ${cell.label}: ${Math.round(cell.summary.wallNps)} NPS, startup=${Math.round(cell.startupMs)}ms`);
  if (errors.length || report.parity.some((p) => p.exact !== p.pairs)) process.exitCode = 1;
} finally { await browser.close(); }
