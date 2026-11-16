<script lang="ts">
import { onMount } from 'svelte';
import { writeClipboard } from '../../../lc0/clipboard.ts';
import { buildSoakSteps, soakRuntimeSummary, type SoakReport } from '../../../lc0/mobileSoak.ts';
import type { MobileSoakApi } from '../../../lc0/mobileSoakBrowser.ts';

let report: SoakReport | null = null;
let api: MobileSoakApi | undefined;
let notice = '';
const steps = buildSoakSteps();
$: running = report?.state === 'running';
$: json = report ? JSON.stringify(report, null, 2) : '';
$: selected = (report?.stepIds ?? []).flatMap((id) => steps.filter((step) => step.id === id));
const ms = (value: number | null | undefined) => value == null ? '—' : `${Math.round(value)} ms`;

onMount(() => {
  let disposed = false;
  let destroy: (() => void) | undefined;
  void import('../../../lc0/mobileSoakBrowser.ts').then((module) => {
    if (disposed) return;
    const controller = module.mountMobileSoak((value) => { report = value; });
    api = controller.api;
    destroy = controller.destroy;
    return controller.ready;
  }).catch((error) => { notice = String(error); });
  return () => { disposed = true; destroy?.(); };
});
async function start() {
  try { await api?.start(); } catch (error) { notice = String(error); }
}
async function resume() {
  try { await api?.continueAfterCrash(); } catch (error) { notice = String(error); }
}
async function copy() {
  notice = await writeClipboard(json) ? 'Results copied.' : 'Clipboard unavailable. Select the JSON below to copy it.';
}
</script>

<svelte:head><title>Mobile soak · 0x88 lab</title><meta name="robots" content="noindex" /></svelte:head>
<main class="soak">
  <h1>Mobile soak</h1>
  <p>Real production engines, one step at a time. Large downloads and sustained compute may heat this phone or kill this tab. Keep it foregrounded. No results leave this device.</p>
  <section aria-labelledby="env-heading">
    <h2 id="env-heading">Environment</h2>
    <pre id="mobile-soak-env" data-testid="environment">{report?.env ? JSON.stringify(report.env, null, 2) : 'Collecting environment…'}</pre>
    <p class="note">Memory is normally unavailable on Safari (null). Chrome numbers exclude some GPU memory. WASM heaps are private in these production adapters, so heap byteLength is null.</p>
  </section>
  <div class="controls">
    <button id="mobile-soak-start" data-testid="start" class="primary" disabled={!api || running || report?.state === 'crashed-awaiting-choice'} on:click={start}>Start</button>
    <button id="mobile-soak-continue" data-testid="continue" disabled={!api || report?.state !== 'crashed-awaiting-choice'} on:click={resume}>Continue after that step</button>
    <button id="mobile-soak-reset" data-testid="start-over" disabled={!api || running} on:click={() => api?.reset()}>Start over</button>
    <button id="mobile-soak-copy" data-testid="copy-results" disabled={!report} on:click={copy}>Copy results</button>
  </div>
  <p id="mobile-soak-status" data-testid="status" role="status" aria-live="polite">
    {report?.state ?? 'idle'}{report?.inProgress ? ` · ${report.inProgress.id} (loading, searching or disposing)` : ''} · Reloads: {report?.reloadCount ?? 0}
  </p>
  {#if report?.state === 'crashed-awaiting-choice'}
    <p role="alert">The tab was killed during {report.crashStep}. Reload/navigation also leaves this breadcrumb; it is not proof of an OS kill. Continue skips that step and keeps earlier results.</p>
  {/if}
  {#if notice}<p role="status">{notice}</p>{/if}
  <p class="note">Heat: {report?.heatSeconds ?? 120}s. Configure with <code>?steps=sf-lite,lc0-small,t3,heat&amp;heatSeconds=120&amp;auto=1</code>. Defaults include all engines. Start over resets to the URL selection.</p>
  <table>
    <caption>Selected steps · load includes download/cache read and initialization · searches: two fixed positions</caption>
    <thead><tr><th scope="col">Engine</th><th scope="col">Status</th><th scope="col">Measurements / error</th></tr></thead>
    <tbody>
      {#each selected as step}
        {@const result = report?.steps.find((entry) => entry.id === step.id)}
        <tr id={`mobile-soak-step-${step.id}`} data-testid={`step-${step.id}`}>
          <th scope="row">{step.label}</th>
          <td>{report?.inProgress?.id === step.id ? 'running' : result?.status ?? 'pending'}</td>
          <td>
            {#if result}
              <div>Load {ms(result.loadMs)}</div><div>First {ms(result.firstSearchMs)}</div>
              {#each result.searches as sample}<div>{sample.engine}: {sample.perSecond?.toFixed(1) ?? '—'} {sample.unit}/s</div>{/each}
              {#each Object.entries(result.runtime) as [id, runtime]}<div class="runtime">{id}: {soakRuntimeSummary(runtime)}</div>{/each}
              {#if result.slices.length}<div>{result.slices.length} heat slices (see JSON)</div>{/if}
              {#if result.error}<div class="error">{result.error}</div>{/if}
            {:else}—{/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
  <h2>Results JSON</h2>
  <pre id="mobile-soak-results" data-testid="results">{json}</pre>
</main>

<style>
.soak { max-width: 960px; min-width: 0; margin: 0 auto; padding: 16px; overflow-wrap: anywhere; }
h1 { font-size: 28px; } h2 { font-size: 20px; }
p { line-height: 1.5; }
section { background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius); padding: 12px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px; }
button { min-height: 44px; padding: 8px 12px; cursor: pointer; }
button:disabled { opacity: 0.5; cursor: default; }
.primary { background: var(--accent); color: var(--on-accent); }
pre { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font-size: 12px; max-height: 420px; overflow-y: auto; max-width: 100%; }
table { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 12px; }
caption { text-align: left; color: var(--muted); padding: 12px 0; }
th, td { text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); padding: 10px 4px; overflow-wrap: anywhere; }
th:first-child { width: 28%; } th:nth-child(2) { width: 19%; }
.runtime, .note { color: var(--muted); } .note { font-size: 13px; }
.error, [role='alert'] { color: var(--warn); }
</style>
