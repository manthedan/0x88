<script lang="ts">
  import { onMount } from 'svelte';
  const title = "LC0 browser engine";
  const description = "LC0 single-engine lab console: policy-only moves, configurable PUCT search, and parity debugging.";
  const styles = "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}header{padding:16px 24px;border-bottom:1px solid var(--rule);background:var(--panel)}h1{margin:0;font-size:22px;display:inline-block}header p{margin:4px 0 0;color:var(--muted)}nav.modes{display:inline-block;margin-left:18px}nav.modes a{color:var(--muted);text-decoration:none;font-size:13px;margin-right:12px}nav.modes a.active{color:var(--accent);font-weight:700}main{display:grid;grid-template-columns:minmax(320px,560px) minmax(320px,430px);gap:24px;align-items:start;justify-content:center;padding:24px}.board-card,.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px}.board-shell{width:100%;aspect-ratio:1}#ground{width:100%;height:100%;border-radius:6px;overflow:hidden;box-shadow:0 16px 42px -28px #000}.cg-wrap{width:100%!important;height:100%!important}.controls{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:12px}button{border:1px solid var(--rule);border-radius:6px;background:white;color:var(--ink);padding:10px 8px;font:inherit;font-size:13px;cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:white;font-weight:700}button:disabled{opacity:.55;cursor:not-allowed}.status-grid{display:grid;grid-template-columns:120px 1fr;gap:8px 12px;font-size:13px}.status-grid dt{color:var(--muted)}.status-grid dd{margin:0;overflow-wrap:anywhere}.mono,code{font-family:var(--mono)}#message{margin-top:12px;padding:10px;border:1px solid var(--rule);border-radius:6px;background:var(--soft);font-family:var(--mono);font-size:12px}.policy-list{list-style:none;margin:10px 0 0;padding:0}.policy-list li{display:grid;grid-template-columns:24px 72px 1fr 64px;gap:8px;align-items:center;padding:6px 0;border-top:1px solid var(--rule);font-family:var(--mono);font-size:12px}.policy-list li.best b{color:var(--accent)}meter{width:100%}.fen{font-family:var(--mono);font-size:12px;line-height:1.45;overflow-wrap:anywhere}.small{font-size:12px;color:var(--muted);line-height:1.5}.settings{margin-top:12px;display:grid;gap:8px}.settings-row{display:flex;flex-wrap:wrap;gap:8px;align-items:end}.settings .field{display:grid;gap:3px;flex:1 1 90px;min-width:80px}.settings .fen-field{flex:1 1 100%}.settings label{font-size:11px;color:var(--muted)}.settings input,.settings select{font:inherit;font-size:13px;padding:7px 8px;border:1px solid var(--rule);border-radius:6px;background:white;color:var(--ink);width:100%}.settings input[type=number]{font-family:var(--mono)}.settings .settings-row button{flex:1 1 auto}#gpuStatus.warn{color:#a5461b;font-weight:600}.battle-list{list-style:none;margin:8px 0 0;padding:0;font-family:var(--mono);font-size:12px}.battle-list li{padding:4px 0;border-top:1px solid var(--rule)}@media(max-width:900px){main{grid-template-columns:1fr;padding:14px}.controls{grid-template-columns:repeat(2,1fr)}}";
  ;

  onMount(() => {
    let cleanup: () => void = () => undefined;
    let mounted = true;
    void import('../../lc0/policyOnlyBrowser').then((module) => {
      if (!mounted) return;
      cleanup = module.mountPolicyOnlyBrowser();
    });
    return () => {
      mounted = false;
      cleanup();
    };
  });
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={description} />
</svelte:head>
<div>{@html `<style>${styles}</style>`}</div>
<header>
  <h1>LC0 single engine</h1>
  <nav class="modes"><a href="/">Home</a><a href="/app/play/">Play</a><a href="/app/analysis/">Analysis</a><a href="/app/arena/">Arena</a><a class="active" href="/single-engine/">Single engine</a></nav>
  <p>Play / test / debug one engine · policy-only moves · configurable PUCT search · watch vs Stockfish</p>
</header>
<main>
  <section class="board-card" aria-label="Playable chess board">
    <div class="board-shell"><div id="ground"></div></div>
    <div class="controls">
      <button id="engineMove" class="primary" type="button" disabled>Policy move</button>
      <button id="searchMove" type="button" disabled>Search 32</button>
      <button id="stopSearch" type="button" disabled>Stop</button>
      <button id="runParity" type="button" disabled>Run parity</button>
      <button id="reset" type="button">Reset</button>
      <button id="flip" type="button">Flip</button>
    </div>
    <div class="settings">
      <div class="field fen-field">
        <label for="fenInput">FEN</label>
        <input id="fenInput" type="text" spellcheck="false" autocomplete="off" placeholder="paste a FEN to analyze" />
      </div>
      <div class="settings-row">
        <button id="loadFen" type="button">Load FEN</button>
        <button id="analyze" type="button" disabled>Analyze position</button>
        <button id="clearCache" type="button">Clear model cache</button>
      </div>
      <div class="settings-row">
        <div class="field"><label for="sideSelect">Your side</label>
          <select id="sideSelect"><option value="white">White</option><option value="black">Black</option></select></div>
        <div class="field"><label for="modeSelect">Engine reply</label>
          <select id="modeSelect"><option value="policy">Policy-only</option><option value="search">PUCT search</option></select></div>
        <div class="field"><label for="visitsInput">Visits</label>
          <input id="visitsInput" type="number" min="1" max="100000" step="1" value="32" /></div>
        <div class="field"><label for="batchInput">Batch / lanes</label>
          <input id="batchInput" type="number" min="1" max="512" step="1" value="1" /></div>
        <div class="field"><label for="collisionSelect">Leaf collisions</label>
          <select id="collisionSelect"><option value="retry">Retry / virtual lanes</option><option value="backup">Shared backup</option></select></div>
        <div class="field"><label for="multiPvInput">MultiPV</label>
          <input id="multiPvInput" type="number" min="1" max="20" step="1" value="1" /></div>
        <div class="field"><label for="earlyStopSelect">Early stop</label>
          <select id="earlyStopSelect"><option value="none">None</option><option value="root-dominance">Root dominance</option><option value="best-stable">Best stable</option><option value="kld-stable">KLD stable</option></select></div>
        <div class="field"><label for="movetimeInput">Move ms</label>
          <input id="movetimeInput" type="number" min="0" max="600000" step="10" value="0" /></div>
      </div>
      <div class="settings-row">
        <div class="field"><label for="cpuctInput">CPuct</label>
          <input id="cpuctInput" type="number" min="0" max="100" step="0.01" value="1.5" /></div>
        <div class="field"><label for="cpuctScheduleSelect">CPuct schedule</label>
          <select id="cpuctScheduleSelect"><option value="lc0-log">LC0 log</option><option value="constant">Constant</option></select></div>
        <div class="field"><label for="fpuStrategySelect">FPU</label>
          <select id="fpuStrategySelect"><option value="lc0-reduction">LC0 reduction</option><option value="constant">Constant</option></select></div>
        <div class="field"><label for="fpuReductionInput">FPU reduction</label>
          <input id="fpuReductionInput" type="number" min="0" max="5" step="0.001" value="0.330" /></div>
        <div class="field"><label for="temperatureInput">Temperature</label>
          <input id="temperatureInput" type="number" min="0" max="10" step="0.01" value="0" /></div>
      </div>
      <div class="settings-row">
        <div class="field"><label for="opponentSelect">LC0 opponent</label>
          <select id="opponentSelect"><option value="policy">LC0 policy</option><option value="stockfish">Stockfish lite</option></select></div>
        <div class="field"><label for="sfDepthInput">SF depth</label>
          <input id="sfDepthInput" type="number" min="1" max="20" step="1" value="4" /></div>
        <div class="field"><label for="battleGamesInput">Games</label>
          <input id="battleGamesInput" type="number" min="1" max="100" step="1" value="1" /></div>
        <button id="battleStart" type="button" disabled>Watch LC0 game</button>
      </div>
    </div>
    <div id="message">Loading…</div>
  </section>
  <aside class="panel" aria-label="LC0 debug panel">
    <dl class="status-grid">
      <dt>Status</dt><dd id="status">loading</dd>
      <dt>Backend</dt><dd id="backend" class="mono">—</dd>
      <dt>WebGPU</dt><dd id="gpuStatus" class="mono">—</dd>
      <dt>Model</dt><dd id="modelPath" class="mono">—</dd>
      <dt>Model cache</dt><dd id="modelCache" class="mono">disabled</dd>
      <dt>Side to move</dt><dd id="sideToMove">—</dd>
      <dt>Best move</dt><dd id="bestMove" class="mono">—</dd>
      <dt>WDL</dt><dd id="wdl">—</dd>
      <dt>Q / MLH</dt><dd id="qMlh" class="mono">—</dd>
      <dt>Parity</dt><dd id="parity" class="mono">not run</dd>
      <dt>Benchmark</dt><dd id="benchResult" class="mono">not run</dd>
      <dt>Search</dt><dd id="searchSummary" class="mono">not run</dd>
      <dt>Search mode</dt><dd id="searchMode" class="mono">main thread</dd>
      <dt>Search batch</dt><dd id="searchBatch" class="mono">1</dd>
      <dt>Search timing</dt><dd id="searchLatency" class="mono">—</dd>
      <dt>PV</dt><dd id="searchPv" class="mono">—</dd>
      <dt>Moves</dt><dd id="moveList" class="mono">—</dd>
    </dl>
    <h2>Top legal priors</h2>
    <ol id="priors" class="policy-list"></ol>
    <h2>Top search children</h2>
    <ol id="searchChildren" class="policy-list"></ol>
    <h2>EngineBattle</h2>
    <p id="battleSummary" class="mono small">not run</p>
    <ol id="battleResults" class="battle-list"></ol>
    <h2>FEN</h2>
    <p id="fen" class="fen">—</p>
    <p class="small">Use the controls above to paste a FEN and analyze it (analysis is drawn on the board as arrows), choose your side, pick the engine reply mode (policy-only or configurable PUCT search), set visits/batch/time and search knobs, and clear the model cache. <b>Watch LC0 game</b> plays the LC0 search engine out on the board against LC0 policy or <b>Stockfish lite</b> (adjustable depth) so you can watch the moves; press Stop to end it. The <b>WebGPU</b> row flags when an inference backend silently fell back to WASM. Query params still work for smokes: open with <code>?ep=wasm&amp;parity=1</code> to force WASM and automatically compare browser outputs against committed native BLAS FEN-only plus explicit-history fixture priors with elapsed time and eval/s. Use <code>?search=1</code> to auto-run a PUCT root search, <code>?worker=1&amp;search=1</code> to run search in a worker, or <code>?workerOnly=1</code> / <code>?bigModel=1</code> to load the LC0 model only inside the dedicated worker and broker all eval/search requests through it. Add <code>?bench=1&amp;ep=webgpu&amp;benchWarmup=5&amp;benchIters=25</code> to run a dedicated-worker eval timing harness. Add <code>?batch=8</code> with a matching fixed-batch model such as <code>?model=/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.onnx</code> to select/evaluate multiple in-flight search leaves in real ONNX batches; <code>?collision=retry</code> keeps virtual lane visits during collection to avoid duplicate leaves, while <code>?collision=backup</code> preserves shared-backup behavior. Use <code>?earlyStop=root-dominance</code>, <code>?earlyStop=best-stable</code>, or <code>?earlyStop=kld-stable</code> to opt into guarded LC0-style early stop telemetry. LC0-style search knobs are also exposed through query params: <code>?movetime=250</code>, <code>?cpuct=1.5</code>, <code>?cpuctSchedule=lc0-log</code>, <code>?fpuStrategy=lc0-reduction</code>, <code>?fpuReduction=0.33</code>, and <code>?temperature=0</code>. Add <code>?packProbe=1</code> to load and verify the batch-8 <code>lc0web</code> packed-model shards inside the dedicated worker without creating an ONNX session, <code>?kernelProbe=1</code> to run one custom WGSL fixed-shape MatMul+Add kernel against tensors from that pack, or <code>?kernelBench=1&amp;kernelBenchIters=1000</code> to queue many WGSL dispatches and read back once for correctness; add <code>?kernelVariant=tiled16</code> for the workgroup-reduction kernel or <code>?kernelVariant=scalar-transposed</code> to benchmark transposed weight layout. Use <code>?ortOpBench=1&amp;ep=webgpu</code> to compare the same extracted MatMul+Add as a tiny ORT ONNX op, <code>?qkvProbe=1</code> to run the encoder0 Q/K/V projection sub-block with per-iteration readback, or <code>?qkvBench=1&amp;qkvBenchIters=1000</code> to queue many Q/K/V projection dispatches and read back once. Use <code>?attentionScoreBench=1</code> for the encoder0 <code>Q @ Kᵀ * scale</code> WGSL score op, <code>?attentionScoreOrtBench=1&amp;ep=webgpu</code> for the equivalent tiny ORT score op, <code>?softmaxBench=1</code> for the fixed-shape attention softmax probe, <code>?attentionValueBench=1</code> for the fixed-shape <code>softmax(QK) @ V</code> value op, <code>?attentionBlockBench=1</code> for the combined <code>input → Q/K/V → QK → softmax → attn@V</code> block, or <code>?attentionOutputBench=1</code> to include output projection, alpha residual, and ln1. Add <code>?cache=1</code> to persist the selected ONNX model in Cache Storage. Use <code>?model=/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.onnx</code> to smoke the f16 deployment artifact.</p>
  </aside>
</main>
