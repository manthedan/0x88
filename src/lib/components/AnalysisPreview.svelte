<script lang="ts">
type PieceRole = 'rook' | 'knight' | 'bishop' | 'queen' | 'king' | 'pawn';
type Piece = { color: 'white' | 'black'; role: PieceRole; square: string };

// Sicilian Najdorf, English Attack, after 12...b4 (the same game the
// Analysis page is reviewed on). Engine numbers below were copied from a run
// of the built Analysis page with its default engines, Lc0 Small (400 visits)
// and Stockfish Lite (depth 12), on this position. Re-run before editing them.
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

// Chessground draws engine brushes at lineWidth 15 on a 64-unit square and
// stops the shaft 10 units short of the destination centre; mirror that here
// on a 100-unit square so the preview arrows match the Analysis board.
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

type EngineResult = {
  id: string;
  name: string;
  monogram: string;
  color: string;
  meta: string;
  evalText: string;
  best: string;
  /** Principal variation after the best move, starting with Black's reply. */
  line: string;
  arrow: [string, string];
};

const engines: EngineResult[] = [
  {
    id: 'lc0',
    name: 'Lc0 Small',
    monogram: 'L0',
    color: '#7a4fd6',
    meta: '365 visits',
    evalText: '+0.25',
    best: 'Ne2',
    line: 'Nh5 14. Ng3 Nf4 15. h4 a5',
    arrow: ['c3', 'e2'],
  },
  {
    id: 'sf',
    name: 'Stockfish Lite',
    monogram: 'SF',
    color: '#1f5fcf',
    meta: 'depth 12',
    evalText: '+0.55',
    best: 'gxf6',
    line: 'bxc3 14. Qxc3 Nxf6 15. Na5 Rc8',
    arrow: ['g5', 'f6'],
  },
];

// Eval bar follows the Analysis page's logistic squash of Lc0's +0.25.
const whitePct = 50 + 50 * (2 / (1 + Math.exp(-0.004 * 25)) - 1);
const analysisHref = `/app/analysis/?fen=${encodeURIComponent(FEN)}`;
</script>

<div class="preview-shell">
<figure class="analysis-preview" aria-labelledby="preview-caption">
  <div class="pv-board">
    <div class="evalbar" aria-hidden="true"><span style={`height:${whitePct.toFixed(1)}%`}></span></div>
    <div
      class="preview-board cg-wrap"
      role="img"
      aria-label="Board after 12…b4, White to move. Lc0's arrow shows Ne2 and Stockfish's arrow shows gxf6."
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
  </div>

  <div class="pv-head" id="preview-caption">
    <span class="pv-context"><b>B90</b> Sicilian Najdorf, English Attack · <span class="nowrap">after 12…b4</span></span>
    <span class="pv-verdict"><strong>Engines disagree: Ne2 / gxf6</strong><span>Eval spread 0.30 <span class="pv-unit">pawns</span></span></span>
  </div>

  <ol class="pv-engines" aria-label="Engine results">
    {#each engines as engine (engine.id)}
      <li style={`--engine-color:${engine.color}`}>
        <span class="pv-avatar" aria-hidden="true">{engine.monogram}</span>
        <span class="pv-name"><b>{engine.name}</b><small>{engine.meta}</small></span>
        <strong class="pv-eval">{engine.evalText}</strong>
        <span class="pv-best"><span class="sr-only">best move </span>{engine.best}</span>
        <span class="pv-line"><b>13. {engine.best}</b> {engine.line}</span>
      </li>
    {/each}
  </ol>

  <div class="pv-foot">
    <p>Each engine draws its own best move. Lc0 moves the attacked knight to safety; Stockfish takes on f6 first.</p>
    <a href={analysisHref}>Analyse this position <span aria-hidden="true">→</span></a>
  </div>
</figure>
</div>

<style>
  .preview-shell{container-type:inline-size; min-width:0}
  .analysis-preview{
    min-width:0; margin:0; padding:16px;
    display:grid; grid-template-columns:minmax(0,1.3fr) minmax(250px,1fr);
    grid-template-areas:"board head" "board engines" "board foot";
    grid-template-rows:auto auto 1fr;
    column-gap:18px; row-gap:12px;
    border:1px solid color-mix(in srgb, var(--ink) 16%, transparent); border-radius:12px;
    background:var(--panel);
    box-shadow:0 30px 70px -45px rgba(45,35,20,.6);
  }
  .pv-board{grid-area:board; align-self:start; display:grid; grid-template-columns:10px minmax(0,1fr); gap:6px}
  .evalbar{position:relative; overflow:hidden; border:1px solid #8a8780; border-radius:3px; background:#555550}
  .evalbar span{position:absolute; inset:auto 0 0; background:var(--eval-white)}
  .preview-board.cg-wrap{
    position:relative; width:100%!important; height:auto!important; aspect-ratio:1; box-sizing:border-box;
    overflow:hidden; border-radius:3px; background:var(--board-light);
  }
  .preview-board cg-board{position:absolute; inset:0; display:block; overflow:hidden}
  .arrows{position:absolute; inset:0; width:100%; height:100%; z-index:4; pointer-events:none; opacity:.6}

  .pv-head{grid-area:head; display:grid; gap:8px; min-width:0}
  .pv-context{color:var(--muted); font-size:13px; line-height:1.45}
  .pv-context b{margin-right:4px; color:var(--ink); font-family:var(--mono); font-size:12px; font-weight:650}
  .nowrap{white-space:nowrap}
  .pv-verdict{display:grid; gap:2px; padding-top:9px; border-top:1px solid var(--rule)}
  .pv-verdict strong{color:var(--ink); font-size:15px; line-height:1.35; font-weight:650}
  .pv-verdict span{color:var(--muted); font-size:13px}

  .pv-engines{grid-area:engines; display:grid; gap:8px; margin:0; padding:0; list-style:none; min-width:0}
  .pv-engines li{
    display:grid; grid-template-columns:24px minmax(0,1fr) auto auto; align-items:center;
    column-gap:9px; row-gap:6px; padding:10px 11px 10px 10px; min-width:0;
    border:1px solid var(--rule); border-left:3px solid var(--engine-color); border-radius:7px;
    background:var(--panel-inset);
  }
  .pv-avatar{
    width:24px; height:24px; display:grid; place-items:center; border-radius:5px;
    background:var(--engine-color); color:#fff; font:700 11px/1 var(--mono); box-shadow:inset 0 0 0 1px #ffffff26;
  }
  .pv-name{min-width:0; display:grid; line-height:1.25}
  .pv-name b{overflow:hidden; white-space:nowrap; text-overflow:ellipsis; color:var(--ink); font-size:14px; font-weight:650}
  .pv-name small{color:var(--muted); font-size:12px}
  .pv-eval{color:var(--ink); font-family:var(--mono); font-size:18px; font-weight:700; letter-spacing:-.02em}
  .pv-best{
    padding:3px 8px; border-radius:5px; color:var(--ink); font-size:14px; font-weight:700;
    background:color-mix(in srgb, var(--engine-color) 22%, var(--panel));
  }
  .pv-line{grid-column:2 / -1; min-width:0; color:var(--ink-soft); font-size:13px; line-height:1.45}
  .pv-line b{color:var(--ink); font-weight:700}

  .pv-foot{grid-area:foot; align-self:end; display:grid; gap:10px; min-width:0}
  .pv-foot p{margin:0; color:var(--muted); font-size:13px; line-height:1.5}
  .pv-foot a{width:max-content; color:var(--ink); font-size:14px; font-weight:650}
  .pv-foot a span{margin-left:4px; color:var(--accent)}
  .sr-only{position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap}

  /* Narrow (phone) frame: the engine verdict and both results lead, the
     board follows, so a named engine result shows before the board. */
  @container(max-width:520px){
    .analysis-preview{
      grid-template-columns:minmax(0,1fr);
      grid-template-areas:"head" "engines" "board" "foot";
      grid-template-rows:none;
      padding:12px; row-gap:10px;
    }
    .pv-context{display:none}
    .pv-verdict{padding-top:0; border-top:0; grid-template-columns:1fr auto; align-items:baseline; gap:8px}
    .pv-verdict strong{font-size:14px; white-space:nowrap}
    .pv-unit{display:none}
    .pv-verdict span{font-size:12px}
    .pv-engines{gap:6px}
    .pv-engines li{padding:8px 10px 8px 9px; row-gap:4px}
    .pv-foot{align-self:start}
  }
</style>
