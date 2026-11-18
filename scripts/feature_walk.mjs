#!/usr/bin/env node
// Seeded random in-app walk over the feature registry (src/features/registry.ts).
// Each step picks an eligible action on the current route, performs its declarative
// steps with real Playwright input, then checks the route's invariants.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { FEATURES, ROUTES, routeForPath } from '../src/features/registry.ts';
import { parseScriptArgs } from './lib/cli.mjs';
import {
  boardMoveExpression,
  chooseAction,
  chooseIndex,
  chooseValue,
  createRng,
  eligibilityExpression,
  evaluateInvariants,
  parseSeed,
  routeActions,
  routeInvariants,
  routeOwnedIds,
  snapshotExpression,
} from './lib/feature_walk_core.mjs';

const VIEWPORTS = { desktop: { width: 1366, height: 900 }, phone: { width: 390, height: 844 } };
const STEP_TIMEOUT_MS = 10_000;

const USAGE = `Usage: node scripts/feature_walk.mjs --base-url URL [options]

Seeded random in-app navigation walk over src/features/registry.ts actions.
After every action it checks the current route's invariants (page errors, console
errors, horizontal overflow, duplicate ids, owned ids present, no full reload, route mounted).
Requires a Playwright Chromium build: npx playwright install chromium

Options:
  --base-url URL       App origin, e.g. a built preview (required unless --dry-run)
  --seed SEED          Number or string; printed in the report for replay (default: time-based)
  --steps N            Actions to perform (default 40)
  --start ROUTE        ${Object.keys(ROUTES).join(' | ')} (default home)
  --viewport NAME      desktop | phone (default desktop)
  --settle-ms MS       Wait after each action before checking (default 600)
  --ready-timeout MS   Wait for a route to mount after navigation (default 60000)
  --keep-going         Record failures and continue instead of stopping at the first
  --headed             Visible browser window
  --out PATH           JSON report path
  --dry-run            Print the registry coverage without opening a browser
  -h, --help           Show this help
`;

export function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'base-url': { type: 'string' },
      seed: { type: 'string', default: String(Date.now() % 1_000_000_007) },
      steps: { type: 'string', default: '40' },
      start: { type: 'string', default: 'home' },
      viewport: { type: 'string', default: 'desktop' },
      'settle-ms': { type: 'string', default: '600' },
      'ready-timeout': { type: 'string', default: '60000' },
      'keep-going': { type: 'boolean', default: false },
      headed: { type: 'boolean', default: false },
      out: { type: 'string' },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  for (const key of ['steps', 'settleMs', 'readyTimeout']) {
    args[key] = Number(args[key]);
    if (!Number.isInteger(args[key]) || args[key] < 0) throw new Error(`Invalid --${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}: ${args[key]}`);
  }
  if (!(args.start in ROUTES)) throw new Error(`Unknown --start route: ${args.start}`);
  if (!(args.viewport in VIEWPORTS)) throw new Error(`Unknown --viewport: ${args.viewport}`);
  if (!args.dryRun) {
    if (!args.baseUrl) throw new Error('--base-url is required');
    const base = new URL(args.baseUrl);
    if (!['http:', 'https:'].includes(base.protocol)) throw new Error(`Unsupported --base-url protocol: ${base.protocol}`);
    args.baseUrl = base.origin;
  }
  return args;
}

export function coverage() {
  return Object.fromEntries(
    Object.keys(ROUTES).map((route) => [
      route,
      {
        features: FEATURES.filter((feature) => feature.routes.includes(route)).map((feature) => feature.id),
        actions: routeActions(route).map(({ feature, action }) => `${feature}/${action.name}`),
        invariants: routeInvariants(route),
      },
    ]),
  );
}

async function waitForExpression(page, expression, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  do {
    try {
      if (await page.evaluate(expression)) return true;
    } catch {
      // Navigation can destroy the execution context mid-poll; retry until the deadline.
    }
    await delay(150);
  } while (Date.now() < deadline);
  return false;
}

/** Visible matches of `target`, choosing one with the walk RNG when the step says `pick`. */
async function targetLocator(page, rng, step) {
  const visible = page.locator(step.target).filter({ visible: true });
  const count = await visible.count();
  if (!count) throw new Error(`No visible ${step.target}`);
  return step.pick ? visible.nth(chooseIndex(rng, count)) : visible.first();
}

async function performStep(page, rng, step) {
  switch (step.do) {
    case 'click': {
      await (await targetLocator(page, rng, step)).click({ timeout: STEP_TIMEOUT_MS });
      return { target: step.target };
    }
    case 'select': {
      const locator = await targetLocator(page, rng, step);
      const { options, current } = await locator.evaluate((node) => ({ options: [...node.options].map((option) => option.value), current: node.value }));
      const value = chooseValue(rng, step.values, options, current);
      if (value === null) return { target: step.target, skipped: 'no registry value offered' };
      await locator.selectOption(value, { timeout: STEP_TIMEOUT_MS });
      return { target: step.target, value };
    }
    case 'fill': {
      const locator = await targetLocator(page, rng, step);
      const value = chooseValue(rng, step.values, null, await locator.inputValue());
      await locator.fill(value, { timeout: STEP_TIMEOUT_MS });
      // fill() fires input only; controllers commit numeric/text settings on change.
      await locator.dispatchEvent('change');
      return { target: step.target, value };
    }
    case 'check': {
      // A DOM click toggles styled checkboxes whose input is visually hidden behind the label.
      const checked = await page
        .locator(step.target)
        .first()
        .evaluate((node) => {
          node.click();
          return node.checked;
        });
      return { target: step.target, checked };
    }
    case 'press': {
      await (await targetLocator(page, rng, step)).press(step.key, { timeout: STEP_TIMEOUT_MS });
      return { target: step.target, key: step.key };
    }
    case 'boardMove': {
      const result = await page.evaluate(boardMoveExpression(step.side, rng()));
      if (result?.error) throw new Error(result.error);
      return result;
    }
    case 'waitFor': {
      if (!(await waitForExpression(page, step.expression, step.timeoutMs ?? STEP_TIMEOUT_MS))) throw new Error(`Timed out: ${step.expression}`);
      return {};
    }
    default:
      throw new Error(`Unknown step ${JSON.stringify(step)}`);
  }
}

async function walk(args) {
  const { chromium } = await import('playwright');
  const rng = createRng(args.seed);
  const report = { status: 'FEATURE_WALK_RUNNING', seed: parseSeed(args.seed), baseUrl: args.baseUrl, viewport: args.viewport, steps: [], failures: [] };
  const browser = await chromium.launch({ headless: !args.headed });
  try {
    const context = await browser.newContext({ viewport: VIEWPORTS[args.viewport] });
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: args.baseUrl });
    // A fresh id per document makes full reloads visible to the noFullReload invariant.
    await context.addInitScript(() => {
      window.__featureWalkDocumentId = crypto.randomUUID();
    });
    const page = await context.newPage();
    const pageErrors = [];
    const consoleErrors = [];
    page.on('pageerror', (error) => pageErrors.push(`Uncaught ${error?.stack ?? error}`));
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push({ level: 'error', text: message.text() });
    });

    await page.goto(new URL(ROUTES[args.start].path, args.baseUrl).href, { waitUntil: 'domcontentloaded', timeout: 45_000 });
    await waitForExpression(page, ROUTES[args.start].ready, args.readyTimeout);

    for (let index = 0; index < args.steps; index++) {
      const route = routeForPath(new URL(page.url()).pathname);
      if (!route) {
        report.failures.push({ step: index, invariant: 'stayedInApp', detail: `left the registered routes: ${page.url()}` });
        break;
      }
      const candidates = routeActions(route);
      const eligible = await page.evaluate(eligibilityExpression(candidates)).catch(() => []);
      const choice = chooseAction(
        rng,
        candidates.filter((_, i) => eligible[i]),
      );
      if (!choice) {
        report.failures.push({ step: index, route, invariant: 'hasEligibleAction', detail: 'no registry action is usable here' });
        break;
      }
      const documentIdBefore = await page.evaluate(() => window.__featureWalkDocumentId).catch(() => null);
      const errorMarks = [pageErrors.length, consoleErrors.length];
      const record = { step: index, route, feature: choice.feature, action: choice.action.name, results: [] };
      report.steps.push(record);
      try {
        for (const step of choice.action.steps) {
          record.results.push(await performStep(page, rng, step));
          await delay(100);
        }
      } catch (error) {
        report.failures.push({ step: index, route, action: `${choice.feature}/${choice.action.name}`, invariant: 'actionCompleted', detail: error.message });
      }
      await delay(args.settleMs);
      const after = routeForPath(new URL(page.url()).pathname);
      if (after) await waitForExpression(page, ROUTES[after].ready, after === route ? args.settleMs : args.readyTimeout);
      record.routeAfter = after;
      const snap = after
        ? await page.evaluate(snapshotExpression(routeOwnedIds(after), ROUTES[after].ready)).catch((error) => ({ error: error.message }))
        : null;
      if (snap && !snap.error) {
        const failures = evaluateInvariants(routeInvariants(after), snap, {
          route: after,
          expectedIds: routeOwnedIds(after),
          pageErrors: pageErrors.slice(errorMarks[0]),
          consoleErrors: consoleErrors.slice(errorMarks[1]),
          documentIdBefore,
          allowsReload: choice.action.allowsReload,
        });
        for (const failure of failures) report.failures.push({ step: index, route: after, action: `${choice.feature}/${choice.action.name}`, ...failure });
      } else if (snap?.error) report.failures.push({ step: index, route: after, invariant: 'snapshot', detail: snap.error });
      if (report.failures.length && !args.keepGoing) break;
    }
  } finally {
    await browser.close().catch(() => undefined);
  }
  report.status = report.failures.length ? 'FEATURE_WALK_FAILED' : 'FEATURE_WALK_DONE';
  report.replay = `node scripts/feature_walk.mjs --base-url ${args.baseUrl} --seed ${report.seed} --steps ${args.steps} --start ${args.start} --viewport ${args.viewport}`;
  return report;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.dryRun) {
    console.log(JSON.stringify({ status: 'FEATURE_WALK_DRY_RUN', seed: parseSeed(args.seed), coverage: coverage() }, null, 2));
    return;
  }
  const report = await walk(args);
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, `${JSON.stringify(report, null, 2)}\n`);
  }
  console.log(JSON.stringify(report, null, 2));
  if (report.failures.length) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack ?? error);
    process.exitCode = 1;
  });
}
