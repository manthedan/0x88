#!/usr/bin/env node
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer, waitForHttp } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5196;
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_AGENT_BROWSER = process.env.AGENT_BROWSER_BIN ?? 'agent-browser';
const REQUIRED_FAMILIES = ['lc0', 'centipawn', 'sf', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax'];
// Lc0 BT4 diagnostics are intentionally hidden in the pruned v0 profile, so
// only require the runtime-family tokens shared by development and production.
const REQUIRED_RUNTIME_TOKENS = ['Centipawn:', 'Reckless:', 'Viridithas:', 'Berserk:', 'PlentyChess:', 'Stormphrax:'];

const USAGE = `Usage: node scripts/lc0_analysis_browser_smoke.mjs [options]\n\nRuns a fast browser smoke for /app/analysis UI wiring. It verifies the multi-engine selector, profile controls, local PGN database controls, runtime status text, and actionable browser console errors.\n\nOptions:\n  --base-url URL        Use an existing server instead of starting Vite\n  --host HOST           Vite host (default ${DEFAULT_HOST})\n  --port N              Vite port when auto-starting (default ${DEFAULT_PORT})\n  --agent-browser BIN   Browser automation binary (default AGENT_BROWSER_BIN or agent-browser)\n  --timeout MS          Browser wait timeout (default ${DEFAULT_TIMEOUT_MS})\n  --out PATH            Optional JSON artifact path\n  --no-server           Do not auto-start Vite\n  --dry-run             Print planned smoke URL without running\n  -h, --help            Show this help\n`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      'base-url': { type: 'string' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      'agent-browser': { type: 'string', default: DEFAULT_AGENT_BROWSER },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      out: { type: 'string' },
      'no-server': { type: 'boolean', default: false },
      'dry-run': { type: 'boolean', default: false },
    },
    usage: USAGE,
  });
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  args.explicitBaseUrl = args.baseUrl !== undefined;
  if (!args.baseUrl) args.baseUrl = `http://${args.host}:${args.port}`;
  if (args.explicitBaseUrl) args.noServer = true;
  for (const [name, value] of [
    ['port', args.port],
    ['timeout', args.timeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --${name}: ${value}`);
  }
  return args;
}

async function closeSession(args, session) {
  try {
    await runAgent(args, ['close'], 5_000, session);
  } catch (error) {
    process.stderr.write(`[analysis-smoke] warning: failed to close ${session}: ${error.message ?? error}\n`);
  }
}

async function waitForEval(args, session, expression, validate, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastValue;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const data = await runAgent(args, ['eval', expression], Math.min(10_000, timeoutMs), session);
      lastValue = data?.result ?? data;
      if (validate(lastValue)) return lastValue;
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Timed out waiting for browser condition. Last value=${JSON.stringify(lastValue)} Last error=${lastError?.message ?? 'none'}`);
}

function normalizeConsoleEntries(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.entries)) return payload.entries;
  if (Array.isArray(payload?.messages)) return payload.messages;
  if (Array.isArray(payload?.logs)) return payload.logs;
  if (Array.isArray(payload?.result)) return payload.result;
  if (typeof payload?.text === 'string')
    return payload.text
      .split('\n')
      .filter(Boolean)
      .map((text) => ({ text }));
  return [];
}

function isActionableConsoleEntry(entry) {
  const text = typeof entry === 'string' ? entry : JSON.stringify(entry);
  if (!text) return false;
  if (/\[vite\] connecting|\[vite\] connected/i.test(text)) return false;
  if (/favicon\.ico/i.test(text)) return false;
  return /error|pageerror|uncaught|failed|exception/i.test(text);
}

async function consoleErrors(args, session) {
  try {
    const payload = await runAgent(args, ['errors'], 10_000, session);
    return normalizeConsoleEntries(payload).filter(isActionableConsoleEntry);
  } catch (error) {
    return [{ error: `console inspection failed: ${error.message ?? error}` }];
  }
}

const SEARCH_STATE_EXPRESSION = `(() => {
  const progressText = document.querySelector('#analysisSearchProgress .search-progress-text')?.textContent ?? '';
  const match = /Lc0:\\s*(\\d+)\\/(\\d+) visits(?:\\s+\\d+%)?(?:\\s+·\\s+([\\d.]+)s)?/.exec(progressText);
  return {
    progressText,
    visits: match ? Number(match[1]) : 0,
    requested: match ? Number(match[2]) : 0,
    elapsedSeconds: match?.[3] ? Number(match[3]) : 0,
    rowCount: document.querySelectorAll('#engineList .engine-row').length,
    strength: Number(document.querySelector('#engineList .row-strength')?.value ?? 0),
    analyzing: document.querySelector('#analyze')?.hasAttribute('disabled') ?? false,
  };
})()`;

const RUNTIME_RELOAD_STATE_EXPRESSION = `(() => {
  const search = ${SEARCH_STATE_EXPRESSION};
  return {
    ...search,
    backendText: document.querySelector('#backend')?.textContent ?? '',
    backendStates: globalThis.__analysisSmokeBackendStates ?? [],
    runtimeAudit: document.querySelector('#runtimeAudit')?.textContent ?? '',
    messageText: document.querySelector('#message')?.textContent ?? '',
    runtimeValue: document.querySelector('#lc0RuntimeSelect')?.value ?? '',
    scheduling: globalThis.__analysisSchedulingState?.() ?? null,
  };
})()`;

async function evalValue(args, session, expression, timeoutMs = 10_000) {
  const payload = await runAgent(args, ['eval', expression], timeoutMs, session);
  return payload?.result ?? payload;
}

function assertSearchContinued(before, after, action) {
  if (after.visits < before.visits || after.elapsedSeconds < before.elapsedSeconds) {
    throw new Error(`${action} restarted active Lc0 analysis: before=${JSON.stringify(before)} after=${JSON.stringify(after)}`);
  }
}

async function runSmoke(args) {
  const session = `lc0-analysis-smoke-${process.pid}`;
  const url = new URL('/app/analysis', args.baseUrl);
  process.stderr.write(`[analysis-smoke] ${url}\n`);
  try {
    await runAgent(args, ['open', String(url)], 30_000, session);
    const snapshot = await waitForEval(
      args,
      session,
      `(() => {
      const optionValues = (selector) => [...document.querySelectorAll(selector)].map((option) => option.value);
      const text = (selector) => document.querySelector(selector)?.textContent ?? '';
      return {
        title: document.title,
        families: optionValues('.row-fam option'),
        hasProfileSelect: !!document.querySelector('#engineProfileSelect'),
        hasProfileName: !!document.querySelector('#engineProfileName'),
        hasSaveProfile: !!document.querySelector('#saveEngineProfile'),
        hasDeleteProfile: !!document.querySelector('#deleteEngineProfile'),
        hasExportProfiles: !!document.querySelector('#exportEngineProfiles'),
        hasImportProfiles: !!document.querySelector('#importEngineProfiles'),
        hasProfileSummary: !!document.querySelector('#engineProfileSummary'),
        profileOptionValues: [...document.querySelectorAll('#engineProfileSelect option')].map((option) => option.value),
        runtimeOptionValues: optionValues('#lc0RuntimeSelect option'),
        hasPgnDbSelect: !!document.querySelector('#pgnDbSelect'),
        hasPgnDbName: !!document.querySelector('#pgnDbName'),
        hasSavePgnDb: !!document.querySelector('#savePgnDb'),
        hasLoadPgnDb: !!document.querySelector('#loadPgnDb'),
        hasSearchPgnDbPosition: !!document.querySelector('#searchPgnDbPosition'),
        hasPgnDbList: !!document.querySelector('#pgnDbList'),
        hasPgnDbSearchResults: !!document.querySelector('#pgnDbSearchResults'),
        hasEngineCompare: !!document.querySelector('#engineCompare tbody'),
        runtimeText: text('#recklessRuntimeInfo'),
        pgnDbInfo: text('#pgnDbInfo'),
      };
    })()`,
      (value) => {
        if (!value || typeof value !== 'object') return false;
        if (!REQUIRED_FAMILIES.every((family) => value.families?.includes(family))) return false;
        if (
          !value.hasProfileSelect ||
          !value.hasProfileName ||
          !value.hasSaveProfile ||
          !value.hasDeleteProfile ||
          !value.hasExportProfiles ||
          !value.hasImportProfiles ||
          !value.hasProfileSummary
        )
          return false;
        if (!['builtin:lc0-stockfish', 'builtin:browser-native-survey'].every((option) => value.profileOptionValues?.includes(option))) return false;
        const hasExperimentalRuntimes = value.runtimeOptionValues?.includes('hybrid-wgsl-heads');
        if (hasExperimentalRuntimes && !value.profileOptionValues?.includes('builtin:lc0-wgsl-heads')) return false;
        if (!hasExperimentalRuntimes && value.profileOptionValues?.some((option) => option === 'builtin:lc0-wgsl-heads' || option === 'builtin:lc0-ladder'))
          return false;
        if (
          !value.hasPgnDbSelect ||
          !value.hasPgnDbName ||
          !value.hasSavePgnDb ||
          !value.hasLoadPgnDb ||
          !value.hasSearchPgnDbPosition ||
          !value.hasPgnDbList ||
          !value.hasPgnDbSearchResults
        )
          return false;
        if (!value.hasEngineCompare) return false;
        if (value.runtimeText === 'Reckless: detecting runtime…') return false;
        return REQUIRED_RUNTIME_TOKENS.every((token) => value.runtimeText?.includes(token));
      },
      args.timeoutMs,
    );
    const missingFamilies = REQUIRED_FAMILIES.filter((family) => !snapshot.families.includes(family));
    if (missingFamilies.length) throw new Error(`Missing engine families: ${missingFamilies.join(', ')}`);
    const missingRuntimeTokens = REQUIRED_RUNTIME_TOKENS.filter((token) => !snapshot.runtimeText.includes(token));
    if (missingRuntimeTokens.length) throw new Error(`Missing runtime status tokens: ${missingRuntimeTokens.join(', ')}`);
    await waitForEval(
      args,
      session,
      `document.querySelector('#message')?.textContent ?? ''`,
      (value) => typeof value === 'string' && value.startsWith('Analysis complete · Lc0'),
      args.timeoutMs,
    );
    await evalValue(
      args,
      session,
      `(() => {
        const input = document.querySelector('#engineList .row-strength');
        if (!(input instanceof HTMLInputElement)) return false;
        input.value = '5000';
        input.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('#analyze')?.click();
        return true;
      })()`,
    );
    const beforeMutation = await waitForEval(
      args,
      session,
      SEARCH_STATE_EXPRESSION,
      (value) => value?.requested === 5000 && value.elapsedSeconds >= 1 && value.visits > 0,
      args.timeoutMs,
    );

    await evalValue(args, session, `document.querySelector('#addEngine')?.click()`);
    await delay(350);
    const afterAdd = await evalValue(args, session, SEARCH_STATE_EXPRESSION);
    if (afterAdd.rowCount !== 2) throw new Error(`Add engine did not render a second row: ${JSON.stringify(afterAdd)}`);
    assertSearchContinued(beforeMutation, afterAdd, 'Adding an engine');

    await evalValue(args, session, `document.querySelector('.row-rm[data-i="1"]')?.click()`);
    await delay(250);
    const afterRemove = await evalValue(args, session, SEARCH_STATE_EXPRESSION);
    if (afterRemove.rowCount !== 1) throw new Error(`Remove engine did not restore one row: ${JSON.stringify(afterRemove)}`);
    assertSearchContinued(afterAdd, afterRemove, 'Removing an engine');

    await evalValue(
      args,
      session,
      `(() => {
        const input = document.querySelector('#engineList .row-strength');
        if (!(input instanceof HTMLInputElement)) return false;
        input.value = '4500';
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`,
    );
    await delay(250);
    const afterSettingChange = await evalValue(args, session, SEARCH_STATE_EXPRESSION);
    if (afterSettingChange.strength !== 4500 || afterSettingChange.requested !== 5000) {
      throw new Error(`Engine setting change did not stay deferred: ${JSON.stringify(afterSettingChange)}`);
    }
    assertSearchContinued(afterRemove, afterSettingChange, 'Changing engine settings');

    await evalValue(args, session, `document.querySelector('#stop')?.click()`);
    await waitForEval(args, session, SEARCH_STATE_EXPRESSION, (value) => !value?.analyzing && value?.progressText === '', args.timeoutMs);
    await delay(500);
    const stoppedState = await evalValue(args, session, SEARCH_STATE_EXPRESSION);
    if (stoppedState.analyzing || stoppedState.progressText !== '') {
      throw new Error(`Stop did not cancel the deferred follow-up analysis: ${JSON.stringify(stoppedState)}`);
    }

    await evalValue(
      args,
      session,
      `(() => {
        const input = document.querySelector('#engineList .row-strength');
        if (!(input instanceof HTMLInputElement)) return false;
        input.value = '300';
        input.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('#analyze')?.click();
        input.value = '200';
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`,
    );
    const deferredRun = await waitForEval(
      args,
      session,
      SEARCH_STATE_EXPRESSION,
      (value) => value?.analyzing && value?.requested === 200 && value?.visits > 0,
      args.timeoutMs,
    );
    await evalValue(args, session, `document.querySelector('#stop')?.click()`);
    await waitForEval(args, session, SEARCH_STATE_EXPRESSION, (value) => !value?.analyzing && value?.progressText === '', args.timeoutMs);

    await evalValue(
      args,
      session,
      `(() => {
        const input = document.querySelector('#engineList .row-strength');
        if (!(input instanceof HTMLInputElement)) return false;
        input.value = '5000';
        input.dispatchEvent(new Event('change', { bubbles: true }));
        document.querySelector('#analyze')?.click();
        return true;
      })()`,
    );
    const runtimeBefore = await waitForEval(
      args,
      session,
      SEARCH_STATE_EXPRESSION,
      (value) => value?.analyzing && value?.requested === 5000 && value?.elapsedSeconds >= 0.5,
      args.timeoutMs,
    );
    const alternateRuntimeListed = await evalValue(
      args,
      session,
      `(() => {
        const backend = document.querySelector('#backend');
        const states = [backend?.textContent ?? ''];
        globalThis.__analysisSmokeBackendStates = states;
        new MutationObserver(() => states.push(backend?.textContent ?? '')).observe(backend, { childList: true, characterData: true, subtree: true });
        const runtime = document.querySelector('#lc0RuntimeSelect');
        if (!(runtime instanceof HTMLSelectElement)) return false;
        const nextRuntime = [...runtime.options].map((option) => option.value).find((value) => value === 'hybrid-ort-heads' && value !== runtime.value);
        if (!nextRuntime) return false;
        runtime.value = nextRuntime;
        runtime.dispatchEvent(new Event('change', { bubbles: true }));
        return true;
      })()`,
    );
    let runtimeDeferred = { skipped: 'onnx-only deployment' };
    let runtimeReload = { skipped: 'onnx-only deployment' };
    let stoppedReload = { skipped: 'onnx-only deployment' };
    let alternateRuntimeSupported = false;
    if (alternateRuntimeListed) {
      await delay(350);
      runtimeDeferred = await evalValue(args, session, RUNTIME_RELOAD_STATE_EXPRESSION);
      if (runtimeDeferred.runtimeValue !== 'hybrid-ort-heads' || runtimeDeferred.backendStates.some((text) => /loading/i.test(text))) {
        throw new Error(`LC0 runtime change did not stay deferred: ${JSON.stringify(runtimeDeferred)}`);
      }
      assertSearchContinued(runtimeBefore, runtimeDeferred, 'Changing LC0 runtime');
      await evalValue(args, session, `(() => { globalThis.__analysisUserMove?.('e2', 'e4'); globalThis.__analysisUserMove?.('e7', 'e5'); })()`);
      const runtimeOutcome = await waitForEval(
        args,
        session,
        RUNTIME_RELOAD_STATE_EXPRESSION,
        (value) =>
          /load failed/i.test(value?.messageText ?? '') ||
          (value?.backendStates?.some((text) => /loading/i.test(text)) &&
            !/loading/i.test(value?.backendText ?? '') &&
            value?.runtimeValue === 'hybrid-ort-heads' &&
            value?.runtimeAudit?.includes('requested hybrid-ort-heads') &&
            value?.runtimeAudit?.includes('resolved hybrid-ort-heads-lazy') &&
            value?.analyzing &&
            value?.requested === 5000),
        args.timeoutMs,
      );
      if (/load failed/i.test(runtimeOutcome.messageText ?? '')) {
        runtimeReload = { skipped: 'hybrid runtime unavailable', outcome: runtimeOutcome };
        stoppedReload = { skipped: 'hybrid runtime unavailable' };
      } else {
        alternateRuntimeSupported = true;
        runtimeReload = runtimeOutcome;
        await waitForEval(args, session, RUNTIME_RELOAD_STATE_EXPRESSION, (value) => value?.scheduling?.searchActive === true, args.timeoutMs);
        await evalValue(
          args,
          session,
          `(() => {
            const runtime = document.querySelector('#lc0RuntimeSelect');
            if (!(runtime instanceof HTMLSelectElement)) return false;
            runtime.value = 'onnx';
            runtime.dispatchEvent(new Event('change', { bubbles: true }));
            globalThis.__analysisUserMove?.('g1', 'f3');
            queueMicrotask(() => document.querySelector('#stop')?.click());
            return true;
          })()`,
        );
        const stoppedDuringReload = await waitForEval(
          args,
          session,
          RUNTIME_RELOAD_STATE_EXPRESSION,
          (value) =>
            value?.backendStates?.some((text) => /loading ORT ONNX/i.test(text)) &&
            !/loading/i.test(value?.backendText ?? '') &&
            !value?.analyzing &&
            value?.progressText === '',
          args.timeoutMs,
        );
        await delay(750);
        const stableStoppedReload = await evalValue(args, session, RUNTIME_RELOAD_STATE_EXPRESSION);
        if (stableStoppedReload.analyzing || stableStoppedReload.progressText !== '' || /loading/i.test(stableStoppedReload.backendText)) {
          throw new Error(`Stop did not cancel the deferred runtime follow-up: ${JSON.stringify(stableStoppedReload)}`);
        }
        stoppedReload = { stoppedDuringReload, stableStoppedReload };
      }
    } else {
      await evalValue(args, session, `document.querySelector('#stop')?.click()`);
      await waitForEval(args, session, SEARCH_STATE_EXPRESSION, (value) => !value?.analyzing && value?.progressText === '', args.timeoutMs);
    }
    const continuity = {
      beforeMutation,
      afterAdd,
      afterRemove,
      afterSettingChange,
      stoppedState,
      deferredRun,
      runtimeBefore,
      alternateRuntimeListed,
      alternateRuntimeSupported,
      runtimeDeferred,
      runtimeReload,
      stoppedReload,
    };
    const errors = await consoleErrors(args, session);
    if (errors.length) throw new Error(`Actionable console errors found: ${JSON.stringify(errors, null, 2)}`);
    return { status: 'LC0_ANALYSIS_BROWSER_SMOKE_DONE', url: String(url), snapshot, continuity, consoleErrors: errors };
  } finally {
    await closeSession(args, session);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const smokeUrl = new URL('/app/analysis', args.baseUrl);
  if (args.dryRun) {
    console.log(JSON.stringify({ status: 'LC0_ANALYSIS_BROWSER_SMOKE_DRY_RUN', url: String(smokeUrl) }, null, 2));
    return;
  }
  const server = startViteServer(args);
  const startedAt = new Date().toISOString();
  let result;
  let runError;
  try {
    if (server) await server.ready;
    await waitForHttp(args.baseUrl);
    result = await runSmoke(args);
  } catch (error) {
    runError = error;
  } finally {
    server?.kill('SIGTERM');
    if (server) await delay(1000);
  }
  if (runError) throw runError;
  const artifact = { ...result, startedAt, finishedAt: new Date().toISOString() };
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, JSON.stringify(artifact, null, 2));
  }
  console.log(JSON.stringify(artifact, null, 2));
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
