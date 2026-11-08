<script lang="ts">
import { onMount } from 'svelte';
import type { runSimulTrial } from '../../../lc0/arenaSimulRunner.ts';
import { simulHelps4 } from '../../../lc0/arenaSimulGate.ts';

type Row = Awaited<ReturnType<typeof runSimulTrial>> & {
  index: number; control: boolean; physical: number; occupancy: number;
  memory?: { bytes: number | null; note: string };
};
type Result = { n: number; visits: number; maxPlies: number; pairs: number; rows: Row[]; [key: string]: unknown };
let n = 2;
let visits = 8;
let maxPlies = 160;
let busy = false;
let status = 'Idle. No model is loaded until you start a gate.';
let generation = '';
let rows: Row[] = [];
let results: Result[] = [];
let worker: Worker | undefined;
let currentGame = 0;
let error = '';

function helps4(effectiveVisits = visits, effectivePlies = maxPlies): boolean {
  return simulHelps4(results, effectiveVisits, effectivePlies);
}

function start(count = n, options: { visits?: number; maxPlies?: number; pairs?: number } = {}) {
  if (busy) throw new Error('Gate already active');
  const effectiveVisits = options.visits ?? visits;
  const effectivePlies = options.maxPlies ?? maxPlies;
  if (count === 8 && !helps4(effectiveVisits, effectivePlies)) throw new Error('N=8 stays parked unless N=4 improves median throughput by 10%');
  worker?.terminate();
  worker = new Worker(new URL('../../../lc0/arenaSimulWorker.ts', import.meta.url), { type: 'module', name: 'arena-simul-lab' });
  rows = [];
  generation = '';
  error = '';
  busy = true;
  status = 'Loading the local small-net export.';
  worker.onmessage = (event) => {
    const message = event.data;
    if (message.generation) generation = message.generation;
    if (message.type === 'phase') status = `Trial ${message.index + 1}: ${message.concurrency} game(s), physical batch ${message.physical}${message.control ? ', numerical control' : ''}. ${message.phase === 'memory' ? 'Sampling memory outside the timing interval.' : 'Loading and warming up.'}`;
    if (message.type === 'memory') {
      const target = worker;
      const measure = (performance as Performance & { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> }).measureUserAgentSpecificMemory;
      const sample = measure ? measure.call(performance).then(({ bytes }) => ({ bytes, note: 'Page + dedicated worker + WASM after trial, session still loaded; excludes GPU residency' }),
        (failure) => ({ bytes: null, note: String(failure) })) : Promise.resolve({ bytes: null, note: 'UASM unavailable' });
      void sample.then((value) => target?.postMessage({ type: 'memory', generation: message.generation, sample: value }));
    }
    if (message.type === 'progress') {
      currentGame = message.game;
      status = `Trial ${message.index + 1}: game ${message.game + 1}, ply ${message.ply}. ${message.completed}/8 games completed.`;
    }
    if (message.type === 'trial') rows = [...rows, message.row];
    if (message.type === 'done') {
      results = [...results, { ...message.result, userAgent: navigator.userAgent, measuredAt: new Date().toISOString() }];
      busy = false;
      status = 'Gate complete. Export the raw evidence before leaving this page.';
    }
    if (message.type === 'error') { busy = false; error = message.error; status = 'Stopped. Incomplete games are not results.'; }
  };
  worker.onerror = (event) => { busy = false; error = event.message; status = 'Worker failed.'; worker?.terminate(); };
  worker.postMessage({ type: 'start', n: count, visits: effectiveVisits, maxPlies: effectivePlies, pairs: options.pairs ?? 3, runId: crypto.randomUUID(), allow8: helps4(effectiveVisits, effectivePlies) });
}
function stop() { worker?.postMessage({ type: 'stop' }); status = 'Stopping after the in-flight GPU run settles.'; }
function cancelGame(game = currentGame, token = generation) { worker?.postMessage({ type: 'cancelGame', game, generation: token }); }
function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'arena-simul-gate.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

onMount(() => {
  // Explicit lab automation only. No autostart query, product import or persisted setting.
  const api = { start, stop, cancelGame, snapshot: () => ({ busy, status, error, generation, currentGame, rows, results, helps4: helps4() }) };
  Object.assign(window, { __arenaSimul: api });
  return () => { worker?.terminate(); delete (window as unknown as Record<string, unknown>).__arenaSimul; };
});
</script>

<svelte:head><title>Arena simul feasibility lab</title></svelte:head>

<main>
  <a href="/app/arena/">Return to Arena</a>
  <h1>Arena simul feasibility lab</h1>
  <p>This research gate shares one Lc0 Small session across independent games. Arena, BT4 and product search defaults are unchanged.</p>
  <section aria-label="Gate setup">
    <div class="controls">
      <label>Simultaneous games<select bind:value={n} disabled={busy}><option value={2}>2</option><option value={4}>4</option><option value={8} disabled={!helps4()}>8 (requires a helpful N=4 gate)</option></select></label>
      <label>Visits per move<input type="number" min="1" max="128" bind:value={visits} disabled={busy} /></label>
      <label>Searched ply cap<input type="number" min="1" max="240" bind:value={maxPlies} disabled={busy} /></label>
    </div>
    <p>Four distinct openings, each with reversed engine colours. Three serial/simul pairs run in ABAB order, followed by a same-export serial control. Each game keeps search batch 1, full history and a fresh tree per move. Capped games count as draws.</p>
    <div class="actions">
      <button class="primary" disabled={busy} onclick={() => start()}>Run gate</button>
      <button disabled={!busy} onclick={stop}>Stop</button>
      <button disabled={!busy} onclick={() => cancelGame()}>Cancel current game</button>
      <button disabled={!results.length || busy} onclick={download}>Export evidence</button>
    </div>
  </section>
  <p role="status">{status}</p>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if rows.length}
    <div class="table-scroll" role="region" aria-label="Trial measurements">
      <table>
        <thead><tr><th>Trial</th><th>Games/hour</th><th>NN evals/s</th><th>Occupancy</th><th>p95 move</th><th>Memory*</th></tr></thead>
        <tbody>{#each rows as row}<tr>
          <td>{row.control ? 'Serial control' : row.concurrency === 1 ? 'Serial' : `Simul ${row.concurrency}`}</td>
          <td>{row.gamesPerHour.toFixed(1)}</td><td>{row.evalsPerSecond.toFixed(1)}</td>
          <td>{(row.occupancy * 100).toFixed(1)}%</td><td>{row.p95MoveMs.toFixed(1)} ms</td>
          <td>{row.memory?.bytes ? `${(row.memory.bytes / 1048576).toFixed(1)} MiB` : 'Unavailable'}</td>
        </tr>{/each}</tbody>
      </table>
    </div>
  {/if}
  <p class="note">*Page, worker and WASM memory sampled after each trial with its session still loaded. This is not a peak and excludes GPU residency. GPU allocation volume is recorded separately, not called live memory. Timing on this shared machine is noisy; an idle-machine re-measure is required before any promotion decision. The lab has no strength-testing claim.</p>
</main>

<style>
  main { max-width: 1040px; margin: 0 auto; padding: 32px 24px; }
  h1 { margin: 24px 0 12px; }
  p { line-height: 1.6; max-width: 85ch; }
  section { background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius); padding: 20px; margin: 24px 0; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px; }
  label { display: grid; gap: 8px; flex: 1 1 170px; }
  input, select { width: 100%; min-width: 0; }
  .actions { display: flex; gap: 10px; flex-wrap: wrap; }
  .primary { background: var(--accent); color: var(--on-accent); }
  .table-scroll { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; white-space: nowrap; }
  th, td { text-align: left; padding: 12px; border-bottom: 1px solid var(--rule); }
  .note { color: var(--muted); font-size: 14px; margin-top: 24px; }
  [role='alert'] { color: var(--warn); }
  @media (max-width: 500px) { main { padding: 20px 16px; } h1 { font-size: 28px; } section { padding: 16px; } }
</style>
