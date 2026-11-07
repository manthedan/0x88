<script lang="ts">
type PieceRole = 'rook' | 'knight' | 'bishop' | 'queen' | 'king' | 'pawn';
type Piece = { color: 'white' | 'black'; role: PieceRole; square: string };

// Sicilian Najdorf, English Attack, after 12...b4. These results were copied
// from the built Analysis page with Lc0 Small (400 visits) and Stockfish Lite
// (depth 12). Re-run before editing the scores, search metadata or variations.
const FEN = 'r2q1rk1/3nbppp/p2pbn2/4p1P1/1p2P3/1NN1BP2/PPPQ3P/2KR1B1R w - - 0 13';
const ROLE: Record<string, PieceRole> = { r: 'rook', n: 'knight', b: 'bishop', q: 'queen', k: 'king', p: 'pawn' };
const FILES = 'abcdefgh';

function fenPieces(fen: string): Piece[] {
  const out: Piece[] = [];
  fen.split(' ')[0].split('/').forEach((row, r) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) f += Number(ch);
      else {
        out.push({ color: ch === ch.toUpperCase() ? 'white' : 'black', role: ROLE[ch.toLowerCase()], square: `${FILES[f]}${8 - r}` });
        f += 1;
      }
    }
  });
  return out;
}

/** Square → [file, row-from-top], the order chessground translates by. */
function xy(square: string): [number, number] {
  return [FILES.indexOf(square[0]), 8 - Number(square[1])];
}

// Match the owner's uniform chessground brush width of 15 (64-unit squares).
// Only one best-move arrow per engine; arrows sit UNDER pieces, not over them.
const STROKE = (15 / 64) * 100;
const MARGIN = (10 / 64) * 100;
function arrow(from: string, to: string) {
  const [fx, fy] = xy(from).map((v) => v * 100 + 50);
  const [tx, ty] = xy(to).map((v) => v * 100 + 50);
  const angle = Math.atan2(ty - fy, tx - fx);
  return { x1: fx, y1: fy, x2: tx - Math.cos(angle) * MARGIN, y2: ty - Math.sin(angle) * MARGIN };
}

const pieces = fenPieces(FEN);
const lastMove = ['b5', 'b4'];
const engines = [
  {
    id: 'lc0', name: 'Lc0 Small', color: '#7a4fd6', meta: '365 visits',
    evalText: '+0.25', best: 'Ne2', line: 'Nh5 14. Ng3 Nf4 15. h4 a5',
    arrow: ['c3', 'e2'],
  },
  {
    id: 'sf', name: 'Stockfish Lite', color: '#1f5fcf', meta: 'depth 12',
    evalText: '+0.55', best: 'gxf6', line: 'bxc3 14. Qxc3 Nxf6 15. Na5 Rc8',
    arrow: ['g5', 'f6'],
  },
];
// Keep each numbered move pair together instead of orphaning a final reply.
function variation(best: string, line: string): string[] {
  return `13. ${best} ${line}`.split(/ (?=\d+\.)/);
}
const analysisHref = `/app/analysis/?fen=${encodeURIComponent(FEN)}`;
</script>

<div class="preview-shell">
<figure class="analysis-preview" aria-labelledby="preview-caption">
  <figcaption class="pv-caption" id="preview-caption">
    <span class="pv-example">Example engine comparison</span>
    <strong>Engines disagree: Ne2 / gxf6</strong>
    <span class="pv-perspective">White to move · evaluations from White’s perspective</span>
  </figcaption>

  <div class="pv-board">
    <div class="ranks" aria-hidden="true">{#each [8, 7, 6, 5, 4, 3, 2, 1] as rank}<span>{rank}</span>{/each}</div>
    <div
      class="preview-board cg-wrap"
      role="img"
      aria-label="Board after 12…b4, White to move. Lc0's violet arrow shows Ne2 and Stockfish's blue arrow shows gxf6."
    >
      <cg-board>
        {#each lastMove as sq (sq)}
          <square class="last-move" style={`transform:translate(${xy(sq)[0] * 100}%,${xy(sq)[1] * 100}%)`}></square>
        {/each}
        {#each pieces as piece (piece.square)}
          <piece class={`${piece.color} ${piece.role}`} style={`transform:translate(${xy(piece.square)[0] * 100}%,${xy(piece.square)[1] * 100}%)`}></piece>
        {/each}
      </cg-board>
      <svg class="arrows" viewBox="0 0 800 800" aria-hidden="true">
        <defs>
          {#each engines as engine (engine.id)}
            <marker id={`preview-head-${engine.id}`} markerWidth="4" markerHeight="4" refX="2.05" refY="2" orient="auto">
              <path d="M0,0 V4 L3,2 Z" fill={engine.color}></path>
            </marker>
          {/each}
        </defs>
        {#each engines as engine (engine.id)}
          {@const a = arrow(engine.arrow[0], engine.arrow[1])}
          <line {...a} stroke={engine.color} stroke-width={STROKE} stroke-linecap="round" marker-end={`url(#preview-head-${engine.id})`}></line>
        {/each}
      </svg>
    </div>
    <div class="files" aria-hidden="true">{#each FILES as file}<span>{file}</span>{/each}</div>
  </div>

  <div class="pv-results">
    <p class="pv-legend">Best-move arrows <span>Eval spread <b>0.30 pawns</b></span></p>
    <ol class="pv-engines" aria-label="Example engine results">
      {#each engines as engine (engine.id)}
        <li style={`--engine-color:${engine.color}`}>
          <div class="pv-result">
            <span class="pv-swatch" aria-hidden="true">↗</span>
            <b class="pv-name">{engine.name}</b>
            <strong class="pv-eval">{engine.evalText}</strong>
            <span class="pv-best"><span class="sr-only">best move </span>{engine.best}</span>
          </div>
          <span class="pv-meta">{engine.meta}</span>
          <p class="pv-line pv-desktop-line">{#each variation(engine.best, engine.line) as move}<span>{move}</span>{' '}{/each}</p>
          <details class="pv-expand">
            <summary>{engine.meta} <span>· View line</span></summary>
            <p class="pv-line">{#each variation(engine.best, engine.line) as move}<span>{move}</span>{' '}{/each}</p>
          </details>
        </li>
      {/each}
    </ol>
  </div>

  <div class="pv-foot">
    <p class="pv-context"><b>B90</b> Sicilian Najdorf, English Attack · after 12…b4</p>
    <p>Lc0 moves the attacked knight to safety. Stockfish captures on f6 first. Positive scores favour White.</p>
    <a href={analysisHref}>Analyse this position <span aria-hidden="true">→</span></a>
  </div>
</figure>
</div>

<style>
  .preview-shell{container-type:inline-size; min-width:0}
  .analysis-preview{
    min-width:0; margin:0; padding:14px;
    display:grid; grid-template-columns:minmax(0,1fr) 258px;
    grid-template-areas:"caption caption" "board results" "board foot";
    grid-template-rows:auto auto 1fr; gap:16px;
    border:1px solid var(--rule-strong); border-radius:12px; background:var(--panel);
    box-shadow:0 30px 70px -45px rgba(45,35,20,.6);
  }
  .pv-caption{grid-area:caption; display:grid; grid-template-columns:1fr auto; gap:2px 14px; color:var(--ink)}
  .pv-example{font:600 12px/1.5 var(--mono); color:var(--muted)}
  .pv-caption strong{grid-column:1; font-size:17px; font-weight:650}
  .pv-perspective{grid-column:2; grid-row:1 / 3; align-self:end; max-width:28ch; color:var(--muted); font-size:13px; line-height:1.5}
  .pv-board{grid-area:board; position:relative; align-self:start; padding:0 0 18px 18px}
  .ranks,.files{position:absolute; display:grid; color:var(--ink-soft); font:500 12px/1 var(--mono); text-align:center}
  .ranks{left:0; top:0; bottom:18px; width:14px; grid-template-rows:repeat(8,1fr); align-items:center}
  .files{left:18px; bottom:0; right:0; height:14px; grid-template-columns:repeat(8,1fr); align-items:end}
  .preview-board.cg-wrap{
    position:relative; width:100%!important; height:auto!important; aspect-ratio:1; box-sizing:border-box;
    overflow:hidden; border-radius:3px; background:var(--board-light); isolation:isolate;
  }
  .preview-board cg-board{position:absolute; inset:0; display:block; overflow:hidden}
  .arrows{position:absolute; inset:0; width:100%; height:100%; z-index:1; pointer-events:none; opacity:.9; filter:drop-shadow(0 0 1px #fff)}
  .pv-results{grid-area:results; min-width:0}
  .pv-legend{display:grid; gap:2px; margin:0 0 10px; color:var(--ink-soft); font-size:13px; line-height:1.5}
  .pv-legend>span{color:var(--muted)}
  .pv-legend b{font-weight:600; font-variant-numeric:tabular-nums}
  .pv-engines{display:grid; gap:10px; margin:0; padding:0; list-style:none; min-width:0}
  .pv-engines li{padding:10px; min-width:0; border:1px solid var(--rule-strong); border-radius:7px; background:var(--panel-inset)}
  .pv-result{display:grid; grid-template-columns:16px minmax(0,1fr) auto auto; gap:6px; align-items:center}
  .pv-swatch{color:var(--engine-color); font-size:22px; font-weight:750; line-height:1}
  .pv-name{color:var(--ink); font-size:14px; line-height:1.3; font-weight:650}
  .pv-eval{color:var(--ink); font:700 17px/1.4 var(--mono); letter-spacing:-.03em}
  .pv-best{padding:3px 6px; border-radius:4px; color:var(--ink); font:650 13px/1.5 var(--mono); background:color-mix(in srgb, var(--engine-color) 18%, var(--panel))}
  .pv-meta{display:block; margin:3px 0 7px 22px; color:var(--muted); font-size:13px}
  .pv-line{display:flex; flex-wrap:wrap; gap:2px 8px; margin:0; color:var(--ink-soft); font:400 13px/1.65 var(--mono); font-variant-numeric:tabular-nums}
  .pv-line span{white-space:nowrap}
  .pv-line span:first-child{color:var(--ink); font-weight:600}
  .pv-expand{display:none}
  .pv-expand summary{cursor:pointer; color:var(--muted); font-size:13px; min-height:24px; padding-top:3px}
  .pv-expand summary span{color:var(--ink-soft)}
  .pv-expand[open] summary{margin-bottom:6px}
  .pv-foot{grid-area:foot; align-self:end; display:grid; gap:10px; min-width:0; padding-bottom:18px}
  .pv-foot p{margin:0; color:var(--muted); font-size:13px; line-height:1.6}
  .pv-context b{color:var(--ink); font-family:var(--mono); font-weight:600}
  .pv-foot a{width:max-content; color:var(--ink); font-size:14px; font-weight:650}
  .pv-foot a span{margin-left:4px; color:var(--accent)}
  .sr-only{position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap}

  @container(max-width:680px){
    .analysis-preview{grid-template-columns:minmax(0,1fr); grid-template-areas:"caption" "board" "results" "foot"; grid-template-rows:none; padding:10px; gap:10px}
    .pv-caption{grid-template-columns:1fr; gap:1px}
    .pv-caption strong{font-size:15px}
    .pv-perspective{grid-column:1; grid-row:auto; max-width:none; font-size:12px}
    .pv-example{font-size:12px}
    .pv-legend{display:flex; justify-content:space-between; gap:8px; margin-bottom:7px; font-size:12px}
    .pv-engines{gap:6px}
    .pv-engines li{padding:7px 10px}
    .pv-name{font-size:14px}
    .pv-meta,.pv-desktop-line{display:none}
    .pv-expand{display:block}
    .pv-foot{align-self:start; padding:4px 0 2px; gap:8px}
  }
</style>
